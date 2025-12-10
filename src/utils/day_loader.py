"""
Minimal, correct day loader with Norway-local timestamps + aggregation.
"""

from __future__ import annotations
import pandas as pd
import s3fs
from datetime import datetime, date, timedelta
from typing import List
from utils.player_utils import PlayerNameMapper


# ---------------------------------------------------------
# Norway DST using EU rules (correct)
# ---------------------------------------------------------


def norway_offset(d: date) -> int:
    """Return UTC offset for Norway on a given date."""
    # Last Sunday in March
    last_march = date(d.year, 3, 31)
    last_sun_march = last_march - timedelta(days=(last_march.weekday() + 1) % 7)

    # Last Sunday in October
    last_oct = date(d.year, 10, 31)
    last_sun_oct = last_oct - timedelta(days=(last_oct.weekday() + 1) % 7)

    if last_sun_march < d < last_sun_oct:
        return 2  # DST
    return 1  # winter time


# ---------------------------------------------------------
# Main Loader
# ---------------------------------------------------------


class DayDataLoader:
    def __init__(self, fs: s3fs.S3FileSystem, mapper: PlayerNameMapper):
        self.fs = fs
        self.mapper = mapper

    def _extract_date(self, path: str) -> date:
        """Assumes final folder is YYYY-MM-DD."""
        d = path.rstrip("/").split("/")[-1]
        return datetime.strptime(d, "%Y-%m-%d").date()

    def load_day(self, path: str) -> pd.DataFrame:
        """
        Load raw player parquet files and construct REAL Norway-local timestamps.
        """
        files = self.fs.ls(path)
        parquet_files = [f for f in files if f.endswith(".parquet")]

        if not parquet_files:
            raise ValueError(f"No parquet found in {path}")

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f, filesystem=self.fs)

            # Keep raw device time as string
            df["time"] = df["time"].astype(str)

            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)

        # -----------------------------------------------------
        # Build real timestamp = match_date + time-of-day + DST
        # -----------------------------------------------------
        match_date = self._extract_date(path)
        offset = norway_offset(match_date)

        # Extract HH:MM:SS(.ms)
        hhmmss = df_all["time"].str.extract(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")[0]
        td = pd.to_timedelta(hhmmss, errors="coerce")

        base = datetime(match_date.year, match_date.month, match_date.day)
        df_all["timestamp"] = pd.to_datetime(base) + td + pd.Timedelta(hours=offset)

        # Player mapping
        df_all = self.mapper.apply_mapping(df_all, col="player_name")

        return df_all

    # ---------------------------------------------------------
    # SIMPLE CLEAN AGGREGATION
    # ---------------------------------------------------------

    def downsample_1hz(self, df: pd.DataFrame, method="first") -> pd.DataFrame:
        """
        Downsample to 1 Hz grouped by player.
        method: 'first' or 'mean'
        """
        if "timestamp" not in df.columns:
            raise ValueError("'timestamp' missing — call load_day first")

        df = df.set_index("timestamp")
        out = []

        for player, sub in df.groupby("player_name"):
            if method == "first":
                agg = sub.resample("1s").first()
            elif method == "mean":
                agg = sub.resample("1s").mean()
            else:
                raise ValueError("method must be 'first' or 'mean'")

            agg["player_name"] = player
            out.append(agg.reset_index())

        return pd.concat(out, ignore_index=True)
