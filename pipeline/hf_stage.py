"""Stage the derived tree for HuggingFace with metrics and universe cut at
the start of the month. The API and R2 keep the daily files."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl

LAGGED = {"metrics": "date", "universe": "rebalance_date"}


def month_start(d: date) -> date:
    return d.replace(day=1)


def stage_for_hf(src: Path, dst: Path, cutoff: date) -> list[Path]:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    written: list[Path] = []
    for tree, col in LAGGED.items():
        root = dst / tree
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.parquet")):
            if p.stem.endswith("_latest"):
                p.unlink()
                continue
            df = pl.read_parquet(p)
            kept = df.filter(pl.col(col) < cutoff)
            if kept.is_empty():
                p.unlink()
                continue
            if kept.height != df.height:
                kept.write_parquet(p)
            written.append(p)
    return written
