"""Write transformed DataFrame to partitioned parquet on local disk.

Layout (Hive-style, plays well with DuckDB / pyarrow / HF datasets):

    <base_dir>/
        <exchange>/
            year=YYYY/
                month=MM/
                    date=YYYY-MM-DD.parquet

One file per (exchange, date). Multi-date input is split into one file each.
Re-running for the same date overwrites the existing file (idempotent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

Exchange = Literal["NSE", "BSE"]
_VALID_EXCHANGES: set[str] = {"NSE", "BSE"}

COMPRESSION = "zstd"


class WriteError(ValueError):
    """Raised when input cannot be written (e.g. missing date column)."""


def partition_path(base_dir: Path, exchange: Exchange, d) -> Path:
    """Return the parquet path for a given (exchange, date)."""
    return (
        base_dir
        / exchange.lower()
        / f"year={d.year}"
        / f"month={d.month:02d}"
        / f"date={d.isoformat()}.parquet"
    )


def write_partitioned(
    df: pl.DataFrame,
    base_dir: Path,
    exchange: Exchange,
) -> list[Path]:
    """Write `df` to partitioned parquet under `base_dir`. Returns paths written.

    Splits by `date` column — one parquet file per distinct date. Empty input
    is a no-op (returns []).
    """
    if exchange not in _VALID_EXCHANGES:
        raise WriteError(f"unknown exchange {exchange!r}; expected one of {_VALID_EXCHANGES}")
    if "date" not in df.columns:
        raise WriteError("input DataFrame missing 'date' column")
    if df.height == 0:
        return []

    written: list[Path] = []
    for (d,), group in df.group_by(["date"], maintain_order=True):
        path = partition_path(base_dir, exchange, d)
        path.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(path, compression=COMPRESSION)
        written.append(path)

    return written
