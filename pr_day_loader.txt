"""
Day loader with Norway-local timestamps + optional streaming 1 Hz downsample.

Key changes vs older version:
- Avoids heavy regex .str.extract on huge day-level data (prevents MemoryError).
- Adds load_day_1hz() that downsamples *per player parquet* while reading (best for head-to-head).
- Uses resample("1s") (lowercase) to avoid pandas FutureWarning.
- Supports max_files and columns to help iterative dev / memory control.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import List, Optional

import pandas as pd
import s3fs

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
# Internal helpers
# ---------------------------------------------------------
def _extract_hhmmss_as_timedelta(time_series: pd.Series) -> pd.Series:
    """
    Robustly convert a 'time' column into a timedelta since midnight.
    Avoids regex by default (much lighter for large arrays).
    Falls back to a regex extract only if parsing fails badly.
    """
    if pd.api.types.is_timedelta64_dtype(time_series):
        return time_series

    s = time_series.astype(str)

    # Common case: already 'HH:MM:SS' or 'HH:MM:SS.ms'
    td = pd.to_timedelta(s, errors="coerce")

    # If many NaT, try stripping date prefixes without regex
    if td.isna().mean() > 0.10:
        # keep last token after whitespace and/or 'T'
        s2 = s.str.split().str[-1]
        s2 = s2.str.split("T").str[-1]
        td2 = pd.to_timedelta(s2, errors="coerce")
        if td2.isna().mean() < td.isna().mean():
            td = td2

    # Rare: still failing -> regex fallback (smaller chance; can be expensive on huge arrays)
    if td.isna().mean() > 0.50:
        hhmmss = s.str.extract(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", expand=False)
        td = pd.to_timedelta(hhmmss, errors="coerce")

    return td


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

    def _list_parquets(self, path: str, max_files: Optional[int] = None) -> List[str]:
        files = self.fs.ls(path)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            raise ValueError(f"No parquet found in {path}")
        parquet_files.sort()
        if max_files is not None:
            parquet_files = parquet_files[: int(max_files)]
        return parquet_files

    def load_day(
        self,
        path: str,
        max_files: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load raw player parquet files for a DAY folder and construct Norway-local timestamps.

        Parameters
        ----------
        max_files : optionally load only first N player parquets (dev/memory control)
        columns   : optionally only read a subset of columns from parquet
        """
        parquet_files = self._list_parquets(path, max_files=max_files)

        match_date = self._extract_date(path)
        offset = norway_offset(match_date)
        base = pd.Timestamp(datetime(match_date.year, match_date.month, match_date.day))

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f, filesystem=self.fs, columns=columns)

            # Build timestamp cheaply (avoid huge regex allocations)
            td = _extract_hhmmss_as_timedelta(df["time"])
            df["timestamp"] = base + td + pd.Timedelta(hours=offset)

            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)

        # Player mapping (usually lightweight)
        df_all = self.mapper.apply_mapping(df_all, col="player_name")

        return df_all

    def load_day_1hz(
        self,
        path: str,
        method: str = "first",
        max_files: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a day but downsample to 1 Hz *per player parquet while reading*.
        This is the most memory-safe way to load head-to-head days.

        method: 'first' or 'mean'
        """
        parquet_files = self._list_parquets(path, max_files=max_files)

        match_date = self._extract_date(path)
        offset = norway_offset(match_date)
        base = pd.Timestamp(datetime(match_date.year, match_date.month, match_date.day))

        out = []
        for f in parquet_files:
            df = pd.read_parquet(f, filesystem=self.fs, columns=columns)

            td = _extract_hhmmss_as_timedelta(df["time"])
            df["timestamp"] = base + td + pd.Timedelta(hours=offset)

            df = df.sort_values("timestamp").set_index("timestamp")

            if method == "first":
                df_1hz = df.resample("1s").first()
            elif method == "mean":
                # mean only on numeric cols; keep non-numeric via first() join
                numeric = df.select_dtypes(include="number").resample("1s").mean()
                nonnum = df.select_dtypes(exclude="number").resample("1s").first()
                df_1hz = pd.concat([numeric, nonnum], axis=1)
            else:
                raise ValueError("method must be 'first' or 'mean'")

            df_1hz = df_1hz.reset_index()
            out.append(df_1hz)

        df_all = pd.concat(out, ignore_index=True)
        df_all = self.mapper.apply_mapping(df_all, col="player_name")
        return df_all

    # ---------------------------------------------------------
    # SIMPLE CLEAN AGGREGATION (for already-loaded df)
    # ---------------------------------------------------------
    def downsample_1hz(self, df: pd.DataFrame, method: str = "first") -> pd.DataFrame:
        """
        Downsample an existing dataframe to 1 Hz grouped by player.
        method: 'first' or 'mean'
        """
        if "timestamp" not in df.columns:
            raise ValueError("'timestamp' missing — call load_day first")

        df = df.sort_values(["player_name", "timestamp"]).set_index("timestamp")
        out = []

        for player, sub in df.groupby("player_name", sort=False):
            if method == "first":
                agg = sub.resample("1s").first()
            elif method == "mean":
                agg = sub.resample("1s").mean(numeric_only=True)
            else:
                raise ValueError("method must be 'first' or 'mean'")

            agg["player_name"] = player
            out.append(agg.reset_index())

        return pd.concat(out, ignore_index=True)
