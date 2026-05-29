# src/utils/player_status.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _choose_time_col(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Choose 'timestamp' if present else 'time', and parse to datetime.
    Returns (time_col, parsed_series).
    """
    if "timestamp" in df.columns:
        tcol = "timestamp"
    elif "time" in df.columns:
        tcol = "time"
    else:
        raise ValueError("df must contain either 'timestamp' or 'time' column.")

    t = pd.to_datetime(df[tcol], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse any datetimes from column '{tcol}'.")
    return tcol, t


def _compute_depth_from_edges(
    df_xy: pd.DataFrame,
    pitch_xy,
    x_col: str = "x_m",
    y_col: str = "y_m",
) -> pd.Series:
    """
    For each sample, compute depth (>=0) inside the pitch rectangle:
      depth = min distance to any of the 4 pitch lines
      depth = 0 if outside the rectangle.
    pitch_xy is expected to be 4 corners or a rectangle-like set of points.
    """
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
    active_depth_m: float = 3.0,
    activate_s: float = 60.0,
    bench_off_s: float = 120.0,
    on_pitch_eps_m: float = 0.1,
    label_col: str = "player_status",
    keep_debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Add a per-row label 'active' / 'bench' using hysteresis + geometry.

    Geometry:
      depth_from_edge = min distance to any pitch boundary line (0 if outside)
      on_pitch       = depth_from_edge >= on_pitch_eps_m
      in_active_zone = depth_from_edge >= active_depth_m

    State machine (per player, in time order):
      - Bench -> Active:
          must be continuously in_active_zone for >= activate_s seconds.
      - Active -> Bench:
          must be continuously OFF pitch (not on_pitch) for >= bench_off_s seconds.

    Notes:
      - Uses per-player actual dt from timestamps (robust to irregular sampling).
      - Does NOT filter rows: it only labels. Filter afterwards if desired.
    """
    if player_col not in df_xy.columns:
        raise ValueError(f"df_xy must contain '{player_col}' column.")
    for c in (x_col, y_col):
        if c not in df_xy.columns:
            raise ValueError(f"df_xy must contain '{c}' column.")

    df = df_xy.copy()

    # --- time handling ---
    time_col, t = _choose_time_col(df)
    df[time_col] = t
    df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("No valid time values after parsing 'time'/'timestamp'.")

    # --- depth & flags ---
    df["depth_from_edge"] = _compute_depth_from_edges(df, pitch_xy, x_col=x_col, y_col=y_col)
    df["on_pitch"] = df["depth_from_edge"] >= float(on_pitch_eps_m)
    df["in_active_zone"] = df["depth_from_edge"] >= float(active_depth_m)

    # --- sort & per-player dt ---
    df = df.sort_values([player_col, time_col])

    dt_s = (
        df.groupby(player_col, sort=False)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
    )
    df["_dt_s"] = dt_s

    # --- hysteresis per player ---
    df[label_col] = "bench"

    # Iterate per player group efficiently
    for pid, idx in df.groupby(player_col, sort=False).groups.items():
        g = df.loc[idx]

        in_active = g["in_active_zone"].to_numpy(dtype=bool)
        on_pitch = g["on_pitch"].to_numpy(dtype=bool)
        dts = g["_dt_s"].to_numpy(dtype=float)

        status_arr = np.empty(len(g), dtype=object)
        current = "bench"
        time_in_active = 0.0
        time_off_pitch = 0.0

        for k in range(len(g)):
            dt = dts[k]

            if current == "bench":
                if in_active[k]:
                    time_in_active += dt
                else:
                    time_in_active = 0.0

                if time_in_active >= activate_s:
                    current = "active"
                    time_off_pitch = 0.0

            else:  # active
                if not on_pitch[k]:
                    time_off_pitch += dt
                else:
                    time_off_pitch = 0.0

                if time_off_pitch >= bench_off_s:
                    current = "bench"
                    time_in_active = 0.0

            status_arr[k] = current

        df.loc[idx, label_col] = status_arr

    df = df.drop(columns=["_dt_s"])

    # Store params so viz can auto-match overlay to classifier
    df.attrs["active_depth_m"] = float(active_depth_m)
    df.attrs["activate_s"] = float(activate_s)
    df.attrs["bench_off_s"] = float(bench_off_s)
    df.attrs["on_pitch_eps_m"] = float(on_pitch_eps_m)
    df.attrs["label_col"] = label_col

    if not keep_debug_cols:
        df = df.drop(columns=["depth_from_edge", "on_pitch", "in_active_zone"])

    return df


def detect_substitutions(
    df_labeled: pd.DataFrame,
    player_col: str = "player_name",
    time_col: str = "timestamp",
    status_col: str = "player_status",
    active_value: str = "active",
) -> pd.DataFrame:
    """
    Simple utility: detect (bench->active) and (active->bench) transition times.
    Returns a table with player_name, event_time, event_type.
    """
    if time_col not in df_labeled.columns:
        # fallback
        time_col, _ = _choose_time_col(df_labeled)

    df = df_labeled.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([player_col, time_col])

    out = []
    for pid, g in df.groupby(player_col, sort=False):
        s = g[status_col].astype(str).to_numpy()
        t = g[time_col].to_numpy()
        prev = None
        for i in range(len(s)):
            cur = s[i]
            if prev is None:
                prev = cur
                continue
            if prev != cur:
                if prev != active_value and cur == active_value:
                    out.append((pid, t[i], "on"))
                elif prev == active_value and cur != active_value:
                    out.append((pid, t[i], "off"))
            prev = cur

    return pd.DataFrame(out, columns=[player_col, "event_time", "event_type"])
