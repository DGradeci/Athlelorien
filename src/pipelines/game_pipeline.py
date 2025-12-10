"""
High-level pipeline to load and process one game's GPS data.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import pandas as pd
import s3fs

from utils.day_processing import ensure_timestamp_fast, downsample_to_1hz_fast
from utils.pitch_calibration import calibrate_pitch_from_df, attach_xy_from_pitch
from utils.player_status import label_active_players
from utils.path_builder import build_active_match_paths
from utils.day_loader import DayDataLoader
from utils.player_utils import PlayerNameMapper
from utils.match_loader import MatchesLoader


def pick_game_row(matches: pd.DataFrame, game_number: int = 1) -> pd.Series:
    """
    Pick Nth game (1-based) from matches DataFrame, sorted by date/time.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "time" in df.columns:
        tt = pd.to_datetime(df["time"], errors="coerce")
        df["_time_sort"] = tt.dt.strftime("%H:%M").fillna("00:00")
    else:
        df["_time_sort"] = "00:00"

    df = (
        df[df["date"].notna()]
        .sort_values(["date", "_time_sort"])
        .reset_index(drop=True)
    )
    if not (1 <= game_number <= len(df)):
        raise IndexError(
            f"Requested game_number={game_number}, "
            f"but only {len(df)} dated games found."
        )
    return df.iloc[game_number - 1]


def build_s3_prefix(bucket: str, date_str: str) -> str:
    """
    Build the S3 key prefix for a given match date (YYYY-MM-DD).
    """
    year = date_str[:4]
    month = date_str[:7]
    return f"{bucket}/data/objective_TEAM_A_{year}/{month}/{date_str}"


def load_processed_game(
    *,
    matches: pd.DataFrame,
    game_number: int,
    fs: s3fs.S3FileSystem,
    pitches: Dict[str, Dict[str, Any]],
    bucket: str = "ucl-ai-soccormon-dataset",
    mapper: PlayerNameMapper | None = None,
    min_active_s: float = 300.0,
    max_gap_s: float = 2.0,
    active_depth_m: float = 6.0,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]
]:
    """
    Full pipeline for one game:

      matches -> pick game -> S3 prefix -> load day (concatenate parquets)
        -> ensure timestamps -> 1 Hz -> calibrate pitch -> x/y metres
        -> label active/bench -> build continuous active paths

    Returns
    -------
    df_raw, df_1hz, df_xy, df_labeled, df_paths, meta
    """
    # 1) pick game
    row = pick_game_row(matches, game_number=game_number)
    date_str = str(row["date"]).split(" ")[0]
    prefix = build_s3_prefix(bucket, date_str)
    print("Loading prefix:", prefix)

    # 2) loader
    if mapper is None:
        mapper = PlayerNameMapper("src/config/player_map.json")
    day_loader = DayDataLoader(fs, mapper)

    df_raw = day_loader.load_day(prefix)  # expects prefix WITHOUT s3://
    if df_raw.empty:
        raise ValueError(f"No data loaded for prefix {prefix}")

    # 3) timestamp + 1 Hz
    df_ts = ensure_timestamp_fast(df_raw, time_col="timestamp")
    df_1hz = downsample_to_1hz_fast(
        df_ts,
        player_col="player_name",
        time_col="timestamp",
        keep="first",
        assume_sorted=True,
    )

    # 4) pitch calibration + XY
    stadium, center_latlon, R, pitch_xy = calibrate_pitch_from_df(
        df_1hz, pitches, lat_col="lat", lon_col="lon"
    )
    df_xy = attach_xy_from_pitch(
        df_1hz, center_latlon, R, lat_col="lat", lon_col="lon", stadium_name=stadium
    )

    # 5) active/bench labels
    df_labeled = label_active_players(
        df_xy,
        pitch_xy,
        player_col="player_name",
        x_col="x_m",
        y_col="y_m",
        active_depth_m=active_depth_m,
        activate_s=60.0,
        bench_off_s=110.0,
        on_pitch_eps_m=0.1,
        label_col="player_status",
    )

    # 6) continuous active paths
    df_paths = build_active_match_paths(
        df_labeled,
        matches,
        game_number=game_number,
        player_col="player_name",
        status_col="player_status",
        min_active_s=min_active_s,
        max_gap_s=max_gap_s,
    )

    meta = {
        "date": date_str,
        "time": row.get("time", ""),
        "home": row.get("home", ""),
        "away": row.get("away", ""),
        "score": row.get("score", ""),
        "stadium": stadium,
        "center_latlon": center_latlon,
        "R": R,
        "pitch_xy": pitch_xy,
    }

    return df_raw, df_1hz, df_xy, df_labeled, df_paths, meta
