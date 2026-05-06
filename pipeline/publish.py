"""Publish local partitioned parquet to HuggingFace Hub as a dataset repo.

We mirror the on-disk layout (`<exchange>/year=YYYY/month=MM/date=YYYY-MM-DD.parquet`)
into the HF repo. `upload_folder` is content-hashed server-side, so unchanged
files are not re-uploaded — making the operation safely idempotent across runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_REPO_ID = "tejhq/indian-markets"


class PublishError(RuntimeError):
    """Raised when the publish step fails for a recoverable reason."""


@dataclass(frozen=True)
class PublishResult:
    repo_id: str
    file_count: int
    total_bytes: int
    commit_url: str | None  # None when dry_run=True


def publish_to_hf(
    data_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> PublishResult:
    """Push parquet files under `data_dir` to HF dataset repo `repo_id`.

    - Token resolves from arg → HF_TOKEN env. Raises if neither set (unless dry_run).
    - Creates the repo with `exist_ok=True` if missing.
    - Uploads only `*.parquet` files; HF dedupes by content hash.
    """
    if not data_dir.exists():
        raise PublishError(f"data dir {data_dir} does not exist")

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise PublishError(f"no parquet files under {data_dir}")
    total_bytes = sum(f.stat().st_size for f in files)

    if dry_run:
        return PublishResult(
            repo_id=repo_id,
            file_count=len(files),
            total_bytes=total_bytes,
            commit_url=None,
        )

    tok = token or os.environ.get("HF_TOKEN")
    if not tok:
        raise PublishError(
            "HF_TOKEN not set — pass --token or export HF_TOKEN before publishing"
        )

    hf = api or HfApi(token=tok)
    try:
        create_repo(repo_id, repo_type="dataset", token=tok, exist_ok=True)
        commit = hf.upload_folder(
            folder_path=str(data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            commit_message=commit_message
            or f"tej-bazaar: sync {len(files)} parquet files",
        )
    except HfHubHTTPError as e:
        raise PublishError(f"HF upload failed: {e}") from e

    return PublishResult(
        repo_id=repo_id,
        file_count=len(files),
        total_bytes=total_bytes,
        commit_url=getattr(commit, "commit_url", None),
    )
