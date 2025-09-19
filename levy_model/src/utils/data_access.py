"""
Utility functions for AWS S3 data access.
"""

import boto3
import s3fs
import pandas as pd


class S3DataAccess:
    """
    Helper class to connect to AWS S3 and load data.
    """

    def __init__(self, access_key: str, secret_key: str, region: str = "eu-west-2"):
        """
        Initialise connection to AWS S3.

        Args:
            access_key (str): AWS Access Key ID
            secret_key (str): AWS Secret Access Key
            region (str): AWS region (default: 'eu-west-2')
        """
        self.session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self.fs = s3fs.S3FileSystem(session=self.session)

    def list_files(self, bucket: str, prefix: str = "") -> list[str]:
        """
                List files in an S3 bucket.
        pi
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
