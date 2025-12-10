"""
Player active/bench labelling based on depth from pitch edges and hysteresis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _choose_time_col(df: pd.DataFrame):
    """
    Choose a usable time column:

    1) Prefer 'timestamp' if it exists and at least some values parse.
    2) Otherwise fall back to 'time' if it exists and parses.
    3) Raise if neither works.
    """
    # 1) Try 'timestamp' if present
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        if t.notna().any():
            return "timestamp", t

    # 2) Fall back to 'time'
    if "time" in df.columns:
        # allow 'HH:MM:SS[.fff]' and 'HH:MM'
        s = df["time"].astype(str).str.strip()
        t = pd.to_datetime(s, errors="coerce", utc=False)

        # second pass: try explicit formats if still all NaT
        if t.isna().all():
            t = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
        if t.isna().all():
            t = pd.to_datetime(s, format="%H:%M", errors="coerce")

        if t.notna().any():
            return "time", t

    # 3) Nothing usable
    raise ValueError("Could not parse datetimes from 'timestamp' or 'time'.")


def _compute_depth_from_edges(
    df_xy: pd.DataFrame, pitch_xy, x_col: str = "x_m", y_col: str = "y_m"
) -> pd.Series:
    """
    For each sample, compute depth (>=0) inside the true pitch:
    min distance to any of the 4 pitch lines, 0 if outside the rectangle.
    """
    import numpy as np

    P = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    x = df_xy[x_col].to_numpy(dtype=float)
    y = df_xy[y_col].to_numpy(dtype=float)

    dx_left = x - xmin
    dx_right = xmax - x
    dy_bottom = y - ymin
    dy_top = ymax - y

    depth = np.minimum.reduce([dx_left, dx_right, dy_bottom, dy_top])
    depth = np.where(depth < 0, 0.0, depth)  # 0 outside pitch

    return pd.Series(depth, index=df_xy.index, name="depth_from_edge")


def label_active_players(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col: str = "player_name",
    x_col: str = "x_m",
    y_col: str = "y_m",
    active_depth_m: float = 3.0,  # metres inside from nearest line to count as "active zone"
    activate_s: float = 60.0,  # seconds continuous in active zone to go bench -> active
    bench_off_s: float = 120.0,  # seconds continuous OFF pitch to go active -> bench
    on_pitch_eps_m: float = 0.1,  # >= this depth counts as "on pitch"
    label_col: str = "player_status",
) -> pd.DataFrame:
    """
    Add a per-row label 'active' / 'bench' to df_xy using hysteresis:

      - Bench -> Active:
          player must have been in the active zone (depth_from_edge >= active_depth_m)
          continuously for at least `activate_s` seconds.

      - Active -> Bench:
          player must have been OFF the pitch (depth_from_edge < on_pitch_eps_m)
          continuously for at least `bench_off_s` seconds.

    Between these thresholds, the current status is kept (hysteresis).
    """
    if player_col not in df_xy.columns:
        raise ValueError(f"df_xy must contain a '{player_col}' column.")

    df = df_xy.copy()

    # --- time handling ---
    time_col, t = _choose_time_col(df)
    df[time_col] = t
    df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("No valid time values after parsing 'time'/'timestamp'.")

    # --- depth & flags ---
    df["depth_from_edge"] = _compute_depth_from_edges(
        df, pitch_xy, x_col=x_col, y_col=y_col
    )

    # on pitch if inside the true pitch by at least on_pitch_eps_m
    df["on_pitch"] = df["depth_from_edge"] >= on_pitch_eps_m
    # in active zone if deep enough inside
    df["in_active_zone"] = df["depth_from_edge"] >= active_depth_m

    # --- sort & per-player time deltas ---
    df = df.sort_values([player_col, time_col])

    dt_s = (
        df.groupby(player_col)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
    )
    df["_dt_s"] = dt_s

    # --- hysteresis status per player ---
    df[label_col] = "bench"  # initial state

    for pid, idx in df.groupby(player_col, sort=False).groups.items():
        g = df.loc[idx]

        in_active = g["in_active_zone"].to_numpy()
        on_pitch = g["on_pitch"].to_numpy()
        dts = g["_dt_s"].to_numpy()

        status_arr = np.empty(len(g), dtype=object)
        current_status = "bench"
        time_in_active = 0.0
        time_off_pitch = 0.0

        for k in range(len(g)):
            dt = dts[k]

            if current_status == "bench":
                # grow/reset active-zone timer
                if in_active[k]:
                    time_in_active += dt
                else:
                    time_in_active = 0.0

                # bench -> active transition
                if time_in_active >= activate_s:
                    current_status = "active"
                    time_off_pitch = (
                        0.0  # reset off-pitch timer when they become active
                    )

            else:  # current_status == "active"
                # grow/reset off-pitch timer
                if not on_pitch[k]:
                    time_off_pitch += dt
                else:
                    time_off_pitch = 0.0

                # active -> bench transition
                if time_off_pitch >= bench_off_s:
                    current_status = "bench"
                    time_in_active = 0.0  # must build up active time again

            status_arr[k] = current_status

        df.loc[idx, label_col] = status_arr

    df = df.drop(columns=["_dt_s"])
    return df
