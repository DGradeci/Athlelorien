"""
Utilities to load and preprocess one day's worth of player data from S3.
"""

import pandas as pd
import s3fs
from typing import List
from src.utils.player_utils import PlayerNameMapper


class DayDataLoader:
    def __init__(self, fs: s3fs.S3FileSystem, mapper: PlayerNameMapper):
        """
        Args:
            fs (s3fs.S3FileSystem): Connected S3 filesystem
            mapper (PlayerNameMapper): Player name mapper
        """
        self.fs = fs
        self.mapper = mapper

    def load_day(self, path: str) -> pd.DataFrame:
        """
        Load and concatenate all parquet files from a day.

        Args:
            path (str): Path to a daily folder in the S3 bucket.

        Returns:
            pd.DataFrame: Combined data for all players, with mapped names.
        """
        files: List[str] = self.fs.ls(path)
        parquet_files = [f for f in files if f.endswith(".parquet")]

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f, filesystem=self.fs)
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No parquet files found in {path}")

        df_all = pd.concat(dfs, ignore_index=True)

        # Ensure time is parsed
        if "time" in df_all.columns:
            df_all["time"] = pd.to_datetime(df_all["time"], errors="coerce")

        # Apply player mapping
        df_all = self.mapper.apply_mapping(df_all, col="player_name")

        return df_all

    def aggregate_time(
        self,
        df: pd.DataFrame,
        freq: str = "1s",
        method: str = "mean",
        keep_cols: List[str] | None = None,
        nth: int = 0,
    ) -> pd.DataFrame:
        """
        Aggregate data over time intervals.

        Args:
            df (pd.DataFrame): Input DataFrame with 'player_name' and 'time'
            freq (str): Resampling frequency (e.g. '1s', '5s', '100ms')
            method (str): 'mean', 'first', or 'nth'
            keep_cols (list): Columns to aggregate. Defaults to numeric ones.
            nth (int): Which row to pick if method='nth'

        Returns:
            pd.DataFrame: Aggregated dataframe
        """
        if "time" not in df.columns or "player_name" not in df.columns:
            raise ValueError("DataFrame must contain 'time' and 'player_name'")

        df = df.set_index("time")

        if keep_cols is None:
            keep_cols = df.select_dtypes(include="number").columns.tolist()

        out = []
        for name, sub in df.groupby("player_name"):
            if method == "mean":
                agg = sub[keep_cols].resample(freq).mean()
            elif method == "first":
                agg = sub[keep_cols].resample(freq).first()
            elif method == "nth":
                agg = (
                    sub[keep_cols]
                    .resample(freq)
                    .apply(lambda x: x.iloc[nth] if len(x) > nth else None)
                )
            else:
                raise ValueError("method must be 'mean', 'first', or 'nth'")

            agg["player_name"] = name
            out.append(agg.reset_index())

        return pd.concat(out, ignore_index=True)
