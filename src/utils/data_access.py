"""
Utility functions for AWS S3 data access.
"""

import os

import boto3
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

        Args:
            access_key (str, optional): AWS Access Key ID (falls back to env)
            secret_key (str, optional): AWS Secret Access Key (falls back to env)
            region (str, optional): AWS region (default: 'eu-west-2')
        """
        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region or os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

        if not access_key or not secret_key:
            raise ValueError(
                "AWS credentials not found. "
                "Set them in a .env file, environment variables, or pass explicitly."
            )

        self.session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self.fs = s3fs.S3FileSystem(session=self.session)

    def list_files(self, bucket: str, prefix: str = "") -> list[str]:
        """
        List files in an S3 bucket.

        Args:
            bucket (str): S3 bucket name
            prefix (str): Optional folder prefix

        Returns:
            list[str]: List of file paths
        """
        return self.fs.ls(f"{bucket}/{prefix}")

    def read_parquet(self, path: str) -> pd.DataFrame:
        """
        Load a Parquet file from S3 directly into a DataFrame.

        Args:
            path (str): Full S3 path to parquet file

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        return pd.read_parquet(f"s3://{path}", filesystem=self.fs)
