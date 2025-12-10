"""
Utility functions for AWS S3 data access.
"""

import os
import pandas as pd
import s3fs
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()


class S3DataAccess:
    """
    Helper class to connect to AWS S3 and load data.
    """

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ):
        """
        Initialise connection to AWS S3.
        """
        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region or os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

        if not access_key or not secret_key:
            raise ValueError(
                "AWS credentials not found. " "Set them in .env or pass explicitly."
            )

        # ✅ Correct s3fs initialisation (no boto3.Session)
        self.fs = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            client_kwargs={"region_name": region},
        )

    def list_files(self, bucket: str, prefix: str = "") -> list[str]:
        """
        List files in an S3 bucket.
        """
        return self.fs.ls(f"{bucket}/{prefix}")

    def read_parquet(self, path: str) -> pd.DataFrame:
        """
        Load a Parquet file from S3 directly into a DataFrame.
        """
        return pd.read_parquet(f"s3://{path}", filesystem=self.fs)
