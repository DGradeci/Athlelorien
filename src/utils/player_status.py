"""
Player activity, bench status, and automatic substitution detection.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


# ---------------------------------------------------------
# TIME COLUMN SELECTION
# ---------------------------------------------------------


def _choose_time_col(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Always use 'timestamp' if it exists.
    Fallback to 'time' only if timestamp missing.
    """
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        return "timestamp", t

    if "time" in df.columns:
        s = df["time"].astype(str).str.strip()
        t = pd.to_datetime(s, errors="coerce", utc=False)
        return "time", t

    raise ValueError("No usable time column ('timestamp' or 'time').")


# ---------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------


def _compute_depth_from_edges(
    df_xy: pd.DataFrame, pitch_xy, x_col="x_m", y_col="y_m"
) -> pd.Series:
    """
    Depth inside pitch boundary (0 outside).
    """
    P = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    x = df_xy[x_col].to_numpy(float)
    y = df_xy[y_col].to_numpy(float)

    depth = np.minimum.reduce(
        [
            x - xmin,
            xmax - x,
            y - ymin,
            ymax - y,
        ]
    )

    depth = np.where(depth < 0, 0.0, depth)
    return pd.Series(depth, index=df_xy.index, name="depth_from_edge")


# ---------------------------------------------------------
# ACTIVE/BENCH LABELLING
# ---------------------------------------------------------


def label_active_players(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col="player_name",
    x_col="x_m",
    y_col="y_m",
    active_depth_m: float = 3.0,
    activate_s: float = 60.0,
    bench_off_s: float = 120.0,
    on_pitch_eps_m: float = 0.1,
    label_col="player_status",
) -> pd.DataFrame:
    """
    Hysteresis-based active/bench status.
    """
    df = df_xy.copy()

    time_col, t = _choose_time_col(df)
    df[time_col] = t
    df = df.dropna(subset=[time_col])
    df = df.sort_values([player_col, time_col]).reset_index(drop=True)

    df["depth_from_edge"] = _compute_depth_from_edges(df, pitch_xy)
    df["on_pitch"] = df["depth_from_edge"] >= on_pitch_eps_m
    df["in_active_zone"] = df["depth_from_edge"] >= active_depth_m

    # time deltas per-player
    df["_dt_s"] = (
        df.groupby(player_col)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        .clip(lower=0)
    )

    df[label_col] = "bench"

    # hysteresis
    for pid, idx in df.groupby(player_col).groups.items():
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
                if in_active[k]:
                    time_in_active += dt
                else:
                    time_in_active = 0.0

                if time_in_active >= activate_s:
                    current_status = "active"
                    time_off_pitch = 0.0

            else:  # currently active
                if not on_pitch[k]:
                    time_off_pitch += dt
                else:
                    time_off_pitch = 0.0

                if time_off_pitch >= bench_off_s:
                    current_status = "bench"
                    time_in_active = 0.0

            status_arr[k] = current_status

        df.loc[idx, label_col] = status_arr

    df.drop(columns=["_dt_s"], inplace=True)
    return df


# ---------------------------------------------------------
# SUBSTITUTION DETECTION
# ---------------------------------------------------------


def detect_substitutions(
    df_status: pd.DataFrame,
    player_col="player_name",
    status_col="player_status",
    time_col="timestamp",
) -> pd.DataFrame:
    """
    Detect automatic substitutions:
      - Entry time = first switch bench -> active
      - Exit time  = first switch active -> bench
    Returns table with player, entry_time, exit_time, minutes_played.
    """

    rows = []

    for pid, g in df_status.groupby(player_col):
        g = g.sort_values(time_col)

        statuses = g[status_col].to_numpy()
        times = g[time_col].to_numpy()

        entry_time = None
        exit_time = None

        prev = statuses[0]
        for k in range(1, len(g)):
            curr = statuses[k]

            # bench -> active = sub ON
            if prev == "bench" and curr == "active" and entry_time is None:
                entry_time = times[k]

            # active -> bench = sub OFF
            if prev == "active" and curr == "bench" and exit_time is None:
                exit_time = times[k]

            prev = curr

        # if player never left pitch
        if entry_time is not None and exit_time is None:
            exit_time = times[-1]

        # Compute minutes played
        if entry_time is None or exit_time is None:
            mp = 0.0
        else:
            delta = exit_time - entry_time  # numpy.timedelta64
            mp = pd.Timedelta(delta).total_seconds() / 60.0

        rows.append(
            {
                "player_name": pid,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "minutes_played": round(mp, 2),
            }
        )

    return pd.DataFrame(rows)
