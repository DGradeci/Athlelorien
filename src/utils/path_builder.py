"""
Build continuous active paths (e.g. 5+ minute segments) using time-of-day only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _parse_time_of_day(value: Any) -> pd.Timedelta:
    """
    Robustly parse a time-of-day from various formats into a Timedelta since midnight.

    Handles examples like:
      - '18:00'
      - '18:00:00'
      - '17:22:09.7'
      - Timestamp('2024-05-10 18:00:00')
      - '2024-05-10 18:00:00'
    """
    # First try: interpret as full datetime and strip date
    try:
        ts = pd.to_datetime(value)
        return pd.to_timedelta(ts.strftime("%H:%M:%S.%f"))
    except Exception:
        pass

    s = str(value).strip()

    # If it looks like "YYYY-MM-DD HH:MM:SS", keep only the last piece
    if " " in s:
        s = s.split()[-1]

    # If it's just "HH:MM", add seconds
    if s.count(":") == 1:
        s = s + ":00"

    return pd.to_timedelta(s)


def build_active_match_paths(
    df_labeled: pd.DataFrame,
    matches: pd.DataFrame,
    game_number: int,
    player_col: str = "player_name",
    status_col: str = "player_status",
    min_active_s: float = 300.0,
    max_gap_s: float = 2.0,
) -> pd.DataFrame:

    if player_col not in df_labeled.columns:
        raise ValueError(f"df_labeled must contain '{player_col}'.")
    if status_col not in df_labeled.columns:
        raise ValueError(f"df_labeled must contain '{status_col}'.")
    if "time" not in df_labeled.columns:
        raise ValueError("df_labeled must contain a 'time' column.")

    df = df_labeled.copy()

    # -----------------------------
    # FIX: robust & safe parsing
    # -----------------------------
    t = pd.to_datetime(df["time"].astype(str), errors="coerce")
    df["time_td"] = pd.to_timedelta(t.dt.strftime("%H:%M:%S.%f"), errors="coerce")

    time_col = "time_td"
    t = df[time_col]

    # kickoff
    row = matches.iloc[game_number - 1]
    kickoff_td = _parse_time_of_day(row["time"])

    # half windows
    first_start = kickoff_td
    first_end = kickoff_td + pd.Timedelta(minutes=45)
    second_start = kickoff_td + pd.Timedelta(minutes=60)
    second_end = kickoff_td + pd.Timedelta(minutes=105)

    mask_first = (t >= first_start) & (t < first_end)
    mask_second = (t >= second_start) & (t < second_end)
    mask_halves = mask_first | mask_second

    df = df[mask_halves].copy()
    if df.empty:
        raise ValueError("No data within match halves.")

    df["half"] = np.where(
        mask_first.loc[df.index],
        1,
        np.where(mask_second.loc[df.index], 2, np.nan),
    )

    df = df[df[status_col] == "active"].copy()
    if df.empty:
        raise ValueError("No active rows within match halves.")

    df = df.sort_values([player_col, time_col])
    df["path_id"] = -1
    global_path_counter = 0

    for pid, g in df.groupby(player_col, sort=False):
        times = g[time_col].values.astype("timedelta64[ns]")
        n = len(g)
        path_ids = np.full(n, -1, dtype=int)

        prev_idx = None
        for k in range(n):
            if prev_idx is None:
                global_path_counter += 1
                path_ids[k] = global_path_counter
            else:
                dt = (times[k] - times[prev_idx]) / np.timedelta64(1, "s")
                if dt <= max_gap_s:
                    path_ids[k] = global_path_counter
                else:
                    global_path_counter += 1
                    path_ids[k] = global_path_counter
            prev_idx = k

        df.loc[g.index, "path_id"] = path_ids

    durations = (
        df.groupby("path_id")[time_col]
        .agg(["min", "max"])
        .rename(columns={"min": "t_start", "max": "t_end"})
    )
    durations["duration_s"] = (
        durations["t_end"] - durations["t_start"]
    ) / np.timedelta64(1, "s")

    valid_paths = durations[durations["duration_s"] >= float(min_active_s)].index
    df_paths = df[df["path_id"].isin(valid_paths)].copy()

    return df_paths
