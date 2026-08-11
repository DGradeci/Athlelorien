from unittest.mock import patch

import pytest

from src.utils.data_access import S3DataAccess


@patch("src.utils.data_access.s3fs.S3FileSystem")
def test_s3_access_uses_profile(mock_filesystem, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "athlelorien")

    S3DataAccess(region="eu-west-2")

    mock_filesystem.assert_called_once_with(
        profile="athlelorien",
        client_kwargs={"region_name": "eu-west-2"},
    )


@patch("src.utils.data_access.s3fs.S3FileSystem")
def test_s3_access_uses_session_token(mock_filesystem, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "example-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "example-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "example-token")
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    S3DataAccess(region="eu-west-2")

    mock_filesystem.assert_called_once_with(
        key="example-key",
        secret="example-secret",
        token="example-token",
        client_kwargs={"region_name": "eu-west-2"},
    )


@patch("src.utils.data_access.s3fs.S3FileSystem")
def test_s3_access_rejects_partial_key_pair(mock_filesystem, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "example-key")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    with pytest.raises(ValueError, match="must be set together"):
        S3DataAccess()

    mock_filesystem.assert_not_called()
