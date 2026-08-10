"""AWS S3 data-access helpers."""

from __future__ import annotations

import os

import pandas as pd
import s3fs
from dotenv import load_dotenv


load_dotenv()


class S3DataAccess:
    """Connect to S3 and load tracking tables."""

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        session_token: str | None = None,
        profile: str | None = None,
    ):
        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region or os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
        session_token = session_token or os.getenv("AWS_SESSION_TOKEN")
        profile = profile or os.getenv("AWS_PROFILE")

        fs_kwargs: dict[str, object] = {"client_kwargs": {"region_name": region}}
        if access_key or secret_key:
            if not access_key or not secret_key:
                raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together.")
            fs_kwargs.update(key=access_key, secret=secret_key)
            if session_token:
                fs_kwargs["token"] = session_token
        elif profile:
            fs_kwargs["profile"] = profile

        # With no explicit keys/profile, s3fs uses the standard AWS credential chain.
        self.fs = s3fs.S3FileSystem(**fs_kwargs)

    def list_files(self, bucket: str, prefix: str = "") -> list[str]:
        """List files below an S3 bucket prefix."""
        return self.fs.ls(f"{bucket}/{prefix}")

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Load a parquet object from S3."""
        return pd.read_parquet(f"s3://{path}", filesystem=self.fs)
