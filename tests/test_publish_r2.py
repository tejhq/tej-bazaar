import hashlib
from pathlib import Path
from datetime import date
from unittest.mock import MagicMock

import pytest

from pipeline.publish_r2 import (
    DEFAULT_BUCKET,
    PublishError,
    PublishResult,
    publish_to_r2,
)


def _seed_parquets(root: Path, n: int = 2) -> tuple[int, list[Path]]:
    paths: list[Path] = []
    total = 0
    for i in range(n):
        p = root / "nse" / "year=2025" / "month=04" / f"date=2025-04-{i+1:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        body = b"x" * (1024 * (i + 1))
        p.write_bytes(body)
        paths.append(p)
        total += len(body)
    return total, paths


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def _mock_head_404(_client, **kwargs):
    raise Exception("Not Found: 404")


def test_publish_missing_dir_raises(tmp_path: Path):
    with pytest.raises(PublishError, match="does not exist"):
        publish_to_r2(tmp_path / "nope", client=MagicMock())


def test_publish_no_parquet_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PublishError, match="no parquet"):
        publish_to_r2(tmp_path / "empty", client=MagicMock())


def test_publish_dry_run_no_upload(tmp_path: Path):
    total, _ = _seed_parquets(tmp_path, n=3)
    client = MagicMock()
    result = publish_to_r2(tmp_path, dry_run=True, client=client)
    assert isinstance(result, PublishResult)
    assert result.file_count == 3
    assert result.total_bytes == total
    assert result.uploaded_count == 0
    assert result.skipped_count == 0
    client.head_object.assert_not_called()
    client.put_object.assert_not_called()


def test_publish_missing_creds_raises(tmp_path: Path, monkeypatch):
    _seed_parquets(tmp_path)
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(PublishError, match="credentials missing"):
        publish_to_r2(tmp_path)


def test_publish_uploads_fresh_files(tmp_path: Path):
    _, paths = _seed_parquets(tmp_path, n=2)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    result = publish_to_r2(tmp_path, client=client)

    assert client.head_object.call_count == 2
    assert client.put_object.call_count == 2
    assert result.uploaded_count == 2
    assert result.skipped_count == 0
    assert result.uploaded_bytes == sum(p.stat().st_size for p in paths)


def test_publish_skips_when_etag_matches(tmp_path: Path):
    _, paths = _seed_parquets(tmp_path, n=2)
    client = MagicMock()
    # Simulate remote ETag matching local md5 for both files.
    def head(Bucket, Key):  # noqa: N803
        for p in paths:
            rel = p.relative_to(tmp_path).as_posix()
            if Key == rel:
                return {"ETag": f'"{_md5(p)}"'}
        raise Exception("Not Found: 404")
    client.head_object.side_effect = head

    result = publish_to_r2(tmp_path, client=client)

    assert client.put_object.call_count == 0
    assert result.uploaded_count == 0
    assert result.skipped_count == 2


def test_publish_uploads_only_changed(tmp_path: Path):
    _, paths = _seed_parquets(tmp_path, n=2)
    client = MagicMock()
    # First file's ETag matches, second is stale.
    def head(Bucket, Key):  # noqa: N803
        rel0 = paths[0].relative_to(tmp_path).as_posix()
        if Key == rel0:
            return {"ETag": f'"{_md5(paths[0])}"'}
        return {"ETag": '"deadbeef"'}
    client.head_object.side_effect = head

    result = publish_to_r2(tmp_path, client=client)

    assert client.put_object.call_count == 1
    assert result.uploaded_count == 1
    assert result.skipped_count == 1


def test_publish_object_key_mirrors_local_path(tmp_path: Path):
    _seed_parquets(tmp_path, n=1)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, client=client)

    put_keys = [c.kwargs["Key"] for c in client.put_object.call_args_list]
    assert "nse/year=2025/month=04/date=2025-04-01.parquet" in put_keys


def test_publish_cache_control_immutable_for_past(tmp_path: Path):
    _seed_parquets(tmp_path, n=1)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, client=client, today=date(2026, 1, 1))

    cc = client.put_object.call_args_list[0].kwargs["CacheControl"]
    assert "immutable" in cc


def test_publish_cache_control_short_for_today(tmp_path: Path):
    p = tmp_path / "nse" / "year=2026" / "month=06" / "date=2026-06-02.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 100)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, client=client, today=date(2026, 6, 2))

    cc = client.put_object.call_args_list[0].kwargs["CacheControl"]
    assert "max-age=300" in cc
    assert "immutable" not in cc


def test_publish_prefix_prepended_to_keys(tmp_path: Path):
    _seed_parquets(tmp_path, n=1)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, prefix="snapshot", client=client)

    put_keys = [c.kwargs["Key"] for c in client.put_object.call_args_list]
    assert all(k.startswith("snapshot/") for k in put_keys)


def test_publish_uses_default_bucket(tmp_path: Path):
    _seed_parquets(tmp_path, n=1)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    result = publish_to_r2(tmp_path, client=client)

    assert result.bucket == DEFAULT_BUCKET
    assert client.put_object.call_args_list[0].kwargs["Bucket"] == DEFAULT_BUCKET


def test_publish_derived_files_get_immutable(tmp_path: Path):
    p = tmp_path / "actions" / "nse_2024.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 50)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, client=client, today=date(2026, 6, 2))

    cc = client.put_object.call_args_list[0].kwargs["CacheControl"]
    assert "immutable" in cc
