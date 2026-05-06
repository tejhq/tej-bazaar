from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.publish import (
    DEFAULT_REPO_ID,
    PublishError,
    PublishResult,
    publish_to_hf,
)


def _seed_parquets(root: Path, n: int = 2) -> int:
    total = 0
    for i in range(n):
        p = root / "nse" / f"year=2025" / f"month=04" / f"date=2025-04-{i+1:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        body = b"x" * (1024 * (i + 1))
        p.write_bytes(body)
        total += len(body)
    return total


def test_publish_missing_dir_raises(tmp_path: Path):
    with pytest.raises(PublishError, match="does not exist"):
        publish_to_hf(tmp_path / "nope", token="t")


def test_publish_no_parquet_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PublishError, match="no parquet"):
        publish_to_hf(tmp_path / "empty", token="t")


def test_publish_dry_run_no_upload(tmp_path: Path):
    total = _seed_parquets(tmp_path, n=3)
    api = MagicMock()
    result = publish_to_hf(tmp_path, dry_run=True, api=api)
    assert isinstance(result, PublishResult)
    assert result.file_count == 3
    assert result.total_bytes == total
    assert result.commit_url is None
    api.upload_folder.assert_not_called()


def test_publish_missing_token_raises(tmp_path: Path, monkeypatch):
    _seed_parquets(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(PublishError, match="HF_TOKEN not set"):
        publish_to_hf(tmp_path)


def test_publish_uploads_and_creates_repo(tmp_path: Path):
    _seed_parquets(tmp_path, n=2)
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url="https://hf.co/x/commit/abc")

    with patch("pipeline.publish.create_repo") as cr:
        result = publish_to_hf(tmp_path, token="t", api=api)

    cr.assert_called_once_with(
        DEFAULT_REPO_ID, repo_type="dataset", token="t", exist_ok=True
    )
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == DEFAULT_REPO_ID
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["allow_patterns"] == ["*.parquet"]
    assert "tej-bazaar" in kwargs["commit_message"]
    assert result.file_count == 2
    assert result.commit_url == "https://hf.co/x/commit/abc"


def test_publish_token_from_env(tmp_path: Path, monkeypatch):
    _seed_parquets(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "env-token")
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url=None)
    with patch("pipeline.publish.create_repo") as cr:
        publish_to_hf(tmp_path, api=api)
    cr.assert_called_once()
    assert cr.call_args.kwargs["token"] == "env-token"


def test_publish_custom_repo_and_message(tmp_path: Path):
    _seed_parquets(tmp_path)
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url=None)
    with patch("pipeline.publish.create_repo"):
        publish_to_hf(
            tmp_path,
            repo_id="me/foo",
            token="t",
            commit_message="custom msg",
            api=api,
        )
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "me/foo"
    assert kwargs["commit_message"] == "custom msg"
