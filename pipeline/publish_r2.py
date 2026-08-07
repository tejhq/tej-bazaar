"""Publish local partitioned parquet to Cloudflare R2 via the S3-compatible API.

We mirror the on-disk layout (`<dataset>/year=YYYY/month=MM/date=YYYY-MM-DD.parquet`)
into the R2 bucket. Each object is content-hashed locally (md5) and compared
against the R2 object's ETag (which equals the md5 for single-part PUTs).
Files whose ETag already matches are skipped, making the operation idempotent.

The bucket layout is what tej-api reads via DuckDB httpfs, so keep the keys
mirror-stable.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any

DEFAULT_BUCKET = "tej-bazaar"
DEFAULT_PREFIX = ""  # write at bucket root, no nesting
HEAD_WORKERS = 8
PUT_WORKERS = 4
TODAY_CACHE = "public, max-age=300"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


class PublishError(RuntimeError):
    """Raised when the publish step fails for a recoverable reason."""


@dataclass(frozen=True)
class PublishResult:
    bucket: str
    file_count: int
    total_bytes: int
    uploaded_count: int
    skipped_count: int
    uploaded_bytes: int


def publish_to_r2(
    data_dir: Path,
    *,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    dry_run: bool = False,
    client: Any | None = None,
    today: date_cls | None = None,
) -> PublishResult:
    """Push parquet files under `data_dir` to R2 bucket `bucket`.

    - Credentials resolve from arg, then env (`R2_ENDPOINT`, `R2_ACCESS_KEY_ID`,
      `R2_SECRET_ACCESS_KEY`). Raises if missing (unless dry_run).
    - Each object key is the file path relative to `data_dir`, joined with `prefix`.
    - Skips objects whose remote ETag matches local md5 (idempotent re-runs).
    - Past-date files get `Cache-Control: immutable`; today's file is short-cached.
    """
    if not data_dir.exists():
        raise PublishError(f"data dir {data_dir} does not exist")

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise PublishError(f"no parquet files under {data_dir}")
    total_bytes = sum(f.stat().st_size for f in files)

    if dry_run:
        return PublishResult(
            bucket=bucket,
            file_count=len(files),
            total_bytes=total_bytes,
            uploaded_count=0,
            skipped_count=0,
            uploaded_bytes=0,
        )

    s3 = client or _build_client(endpoint_url, access_key_id, secret_access_key)
    today = today or date_cls.today()

    plan: list[tuple[Path, str, str, str]] = []
    for path in files:
        key = _object_key(path, data_dir, prefix)
        local_md5 = _md5_of(path)
        cache = _cache_control(path.stem, today)
        plan.append((path, key, local_md5, cache))

    needs_upload = _filter_uploads(s3, bucket, plan)

    uploaded_count = 0
    uploaded_bytes = 0
    if needs_upload:
        with ThreadPoolExecutor(max_workers=PUT_WORKERS) as pool:
            futs = {
                pool.submit(_put_one, s3, bucket, key, path, cache): (path, key)
                for path, key, _md5, cache in needs_upload
            }
            for fut in as_completed(futs):
                path, key = futs[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001 — surface as PublishError below
                    raise PublishError(f"upload failed for {key}: {e}") from e
                uploaded_count += 1
                uploaded_bytes += path.stat().st_size

    return PublishResult(
        bucket=bucket,
        file_count=len(files),
        total_bytes=total_bytes,
        uploaded_count=uploaded_count,
        skipped_count=len(plan) - uploaded_count,
        uploaded_bytes=uploaded_bytes,
    )


def _build_client(
    endpoint_url: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
) -> Any:
    endpoint = endpoint_url or os.environ.get("R2_ENDPOINT")
    akid = access_key_id or os.environ.get("R2_ACCESS_KEY_ID")
    secret = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (endpoint and akid and secret):
        raise PublishError(
            "R2 credentials missing, set R2_ENDPOINT, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY or pass them explicitly"
        )

    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise PublishError(
            "boto3 is required for R2 publish; install with `pip install boto3`"
        ) from e

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=akid,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def _object_key(path: Path, data_dir: Path, prefix: str) -> str:
    rel = path.relative_to(data_dir).as_posix()
    return f"{prefix.rstrip('/')}/{rel}" if prefix else rel


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 — R2 ETag is md5, not a security primitive here
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _cache_control(stem: str, today: date_cls) -> str:
    # File stem is `date=YYYY-MM-DD` for partitioned bhavcopy, or arbitrary
    # for derived datasets (e.g. `nse_2024`). Only date= files get the
    # immutable header keyed on the actual date.
    if not stem.startswith("date="):
        return IMMUTABLE_CACHE
    try:
        d = date_cls.fromisoformat(stem.split("=", 1)[1])
    except ValueError:
        return IMMUTABLE_CACHE
    return TODAY_CACHE if d >= today else IMMUTABLE_CACHE


def _filter_uploads(
    s3: Any,
    bucket: str,
    plan: list[tuple[Path, str, str, str]],
) -> list[tuple[Path, str, str, str]]:
    """Return only the plan entries whose ETag does not already match local md5."""
    needs: list[tuple[Path, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
        futs = {pool.submit(_remote_etag, s3, bucket, key): (path, key, md5, cache)
                for path, key, md5, cache in plan}
        for fut in as_completed(futs):
            path, key, md5, cache = futs[fut]
            remote = fut.result()
            if remote is None or remote != md5:
                needs.append((path, key, md5, cache))
    return needs


def _remote_etag(s3: Any, bucket: str, key: str) -> str | None:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:  # noqa: BLE001 — boto raises ClientError; treat any miss as "not present"
        msg = str(e)
        if "Not Found" in msg or "NoSuchKey" in msg or "404" in msg:
            return None
        raise PublishError(f"head_object failed for {key}: {e}") from e
    etag = resp.get("ETag", "").strip('"')
    return etag or None


def _put_one(s3: Any, bucket: str, key: str, path: Path, cache: str) -> None:
    with path.open("rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType="application/octet-stream",
            CacheControl=cache,
        )


@dataclass(frozen=True)
class PullResult:
    bucket: str
    listed_count: int
    downloaded_count: int
    skipped_count: int
    downloaded_bytes: int


def pull_from_r2(
    data_dir: Path,
    *,
    prefixes: list[str],
    bucket: str = DEFAULT_BUCKET,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    client: Any | None = None,
) -> PullResult:
    """Mirror R2 parquet keys under `prefixes` down into `data_dir`.

    Local-wins: a key whose local file already exists is skipped, whatever
    its content. In the daily cron the runner starts empty, so anything
    already on disk was fetched fresh THIS run (today's bhavcopy, the YTD
    actions file) and is newer than its R2 copy. For local dev the same
    rule means a pull never clobbers in-progress work.

    Used by the daily cron to seed price history before the all-years
    derived steps (symbol-history, adjust, metrics). Without this seed the
    runner sees one day of data and the derived artifacts collapse to a
    single day.
    """
    if not prefixes:
        raise PublishError("pull_from_r2 requires at least one prefix")
    for p in prefixes:
        if not p or "/" not in p:
            raise PublishError(
                f"refusing to pull prefix {p!r}: must contain a `/` to scope "
                "(e.g. `nse/`)"
            )

    s3 = client or _build_client(endpoint_url, access_key_id, secret_access_key)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[tuple[str, int]] = []
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append((obj["Key"], obj.get("Size", 0)))

    to_get: list[tuple[str, int]] = []
    skipped = 0
    for key, size in keys:
        if (data_dir / key).exists():
            skipped += 1
        else:
            to_get.append((key, size))

    def _get_one(item: tuple[str, int]) -> int:
        key, size = item
        local = data_dir / key
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local))
        return size

    downloaded_bytes = 0
    if to_get:
        with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
            futs = {pool.submit(_get_one, item): item for item in to_get}
            for fut in as_completed(futs):
                key, _size = futs[fut]
                try:
                    downloaded_bytes += fut.result()
                except Exception as e:  # noqa: BLE001 — surface as PublishError below
                    raise PublishError(f"download failed for {key}: {e}") from e

    return PullResult(
        bucket=bucket,
        listed_count=len(keys),
        downloaded_count=len(to_get),
        skipped_count=skipped,
        downloaded_bytes=downloaded_bytes,
    )


@dataclass(frozen=True)
class PruneResult:
    bucket: str
    prefix: str
    deleted_count: int


def prune_r2_prefix(
    prefix: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> PruneResult:
    """Delete every R2 object whose key begins with `prefix`.

    Used by the year-end compaction flow to remove the now-redundant daily
    bhavcopy keys after their rollup parquet has been uploaded. The prefix
    MUST be specific (e.g. `nse/year=2010/month=`) so a typo cannot wipe
    the bucket.
    """
    if not prefix or "/" not in prefix:
        raise PublishError(
            f"refusing to prune prefix {prefix!r}: must contain a `/` to scope"
        )

    s3 = client or _build_client(endpoint_url, access_key_id, secret_access_key)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    if dry_run or not keys:
        return PruneResult(bucket=bucket, prefix=prefix, deleted_count=0 if dry_run else 0)

    # Batch delete: S3/R2 caps DeleteObjects at 1000 keys per call.
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        errors = resp.get("Errors", [])
        if errors:
            raise PublishError(f"delete failed for {len(errors)} keys, first: {errors[0]}")
        deleted += len(batch)

    return PruneResult(bucket=bucket, prefix=prefix, deleted_count=deleted)
