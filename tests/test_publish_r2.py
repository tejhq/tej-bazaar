import hashlib
from pathlib import Path
from datetime import date
from unittest.mock import MagicMock

import pytest

from pipeline.publish_r2 import (
    DEFAULT_BUCKET,
    PublishError,
    PublishResult,
    PullResult,
    publish_to_r2,
    pull_from_r2,
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


# --- pull_from_r2 -----------------------------------------------------------


def _mock_pull_client(keys: dict[str, bytes]) -> MagicMock:
    """Client whose paginator lists `keys` and whose download writes bodies."""
    client = MagicMock()

    def paginate(Bucket, Prefix):  # noqa: N803
        matched = [
            {"Key": k, "Size": len(v)}
            for k, v in keys.items()
            if k.startswith(Prefix)
        ]
        return [{"Contents": matched}] if matched else [{}]

    client.get_paginator.return_value.paginate.side_effect = paginate

    def download_file(bucket, key, local):
        Path(local).write_bytes(keys[key])

    client.download_file.side_effect = download_file
    return client


def test_pull_downloads_missing_files(tmp_path: Path):
    keys = {
        "nse/year=2025/nse_2025.parquet": b"a" * 100,
        "nse/year=2026/month=06/date=2026-06-05.parquet": b"b" * 50,
    }
    client = _mock_pull_client(keys)

    result = pull_from_r2(tmp_path, prefixes=["nse/"], client=client)

    assert isinstance(result, PullResult)
    assert result.listed_count == 2
    assert result.downloaded_count == 2
    assert result.skipped_count == 0
    assert result.downloaded_bytes == 150
    assert (tmp_path / "nse/year=2025/nse_2025.parquet").read_bytes() == b"a" * 100


def test_pull_local_wins_skips_existing(tmp_path: Path):
    keys = {"actions/nse_2026.parquet": b"remote-stale"}
    local = tmp_path / "actions" / "nse_2026.parquet"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local-fresh")
    client = _mock_pull_client(keys)

    result = pull_from_r2(tmp_path, prefixes=["actions/"], client=client)

    assert result.downloaded_count == 0
    assert result.skipped_count == 1
    assert local.read_bytes() == b"local-fresh"
    client.download_file.assert_not_called()


def test_pull_multiple_prefixes(tmp_path: Path):
    keys = {
        "nse/year=2025/nse_2025.parquet": b"n",
        "bse/year=2025/bse_2025.parquet": b"b",
        "metrics/nse_2025.parquet": b"m",  # not requested, must be ignored
    }
    client = _mock_pull_client(keys)

    result = pull_from_r2(tmp_path, prefixes=["nse/", "bse/"], client=client)

    assert result.listed_count == 2
    assert (tmp_path / "nse/year=2025/nse_2025.parquet").exists()
    assert (tmp_path / "bse/year=2025/bse_2025.parquet").exists()
    assert not (tmp_path / "metrics/nse_2025.parquet").exists()


def test_pull_unscoped_prefix_raises(tmp_path: Path):
    with pytest.raises(PublishError, match="refusing to pull"):
        pull_from_r2(tmp_path, prefixes=["nse"], client=MagicMock())


def test_pull_no_prefixes_raises(tmp_path: Path):
    with pytest.raises(PublishError, match="at least one prefix"):
        pull_from_r2(tmp_path, prefixes=[], client=MagicMock())


def test_pull_skips_non_parquet_keys(tmp_path: Path):
    keys = {
        "nse/year=2025/nse_2025.parquet": b"a",
        "nse/year=2025/README.md": b"doc",
    }
    client = _mock_pull_client(keys)

    result = pull_from_r2(tmp_path, prefixes=["nse/"], client=client)

    assert result.listed_count == 1
    assert not (tmp_path / "nse/year=2025/README.md").exists()


def test_publish_derived_files_get_immutable(tmp_path: Path):
    p = tmp_path / "actions" / "nse_2024.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 50)
    client = MagicMock()
    client.head_object.side_effect = Exception("Not Found: 404")

    publish_to_r2(tmp_path, client=client, today=date(2026, 6, 2))

    cc = client.put_object.call_args_list[0].kwargs["CacheControl"]
    assert "immutable" in cc


def test_large_plan_uses_prefix_listing(tmp_path: Path, monkeypatch):
    from pipeline import publish_r2 as mod
    monkeypatch.setattr(mod, "LIST_THRESHOLD", 1)
    root = tmp_path / "api"
    (root / "v1/ohlcv/nse").mkdir(parents=True)
    a = root / "v1/ohlcv/nse/A.json"; a.write_text('{"data":[]}')
    b = root / "v1/ohlcv/nse/B.json"; b.write_text('{"data":[1]}')
    a_md5 = hashlib.md5(a.read_bytes()).hexdigest()
    client = MagicMock()
    page = {"Contents": [{"Key": "api/v1/ohlcv/nse/A.json", "ETag": f'"{a_md5}"'}, {"Key": "api/v1/ohlcv/nse/B.json", "ETag": '"stale"'}]}
    client.get_paginator.return_value.paginate.return_value = [page]
    res = publish_to_r2(root, bucket="b", prefix="api", client=client)
    assert res.uploaded_count == 1 and res.skipped_count == 1
    client.head_object.assert_not_called()
    client.get_paginator.return_value.paginate.assert_called_once_with(Bucket="b", Prefix="api/v1/ohlcv/nse/")
    put_keys = [c.kwargs["Key"] for c in client.put_object.call_args_list]
    assert put_keys == ["api/v1/ohlcv/nse/B.json"]
