"""Pre-render tej-api free-tier responses as static JSON for edge serving.

The Cloudflare Worker in front of api.tejhq.dev serves these objects
straight from R2 and slices them in memory, so the common queries never
touch DuckDB or Cloud Run. Shapes mirror tej-api's Go wire models exactly
(``models.OHLCV``, ``models.SnapshotRow``, ``models.Action``); the Worker
adds the ``meta`` block.

Layout under ``out_dir``::

    api/v1/ohlcv/<ex>/<SYMBOL>.json      {"data": [rows by date asc]}
    api/v1/snapshot/<ex>/<DATE>.json     {"data": [rows by symbol asc]}
    api/v1/actions/<SYMBOL>.json         {"data": [rows by ex_date desc]}

Raw bhavcopy rows for a past date never change, so snapshots are only
re-exported for ``snapshot_years`` (default: current year). Per-symbol OHLCV
and actions files are rewritten every run; ``publish-r2`` skips the ones
whose content did not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Iterable

import polars as pl

OHLCV_COLS = [
    "date", "open", "high", "low", "close", "last", "prev_close",
    "volume", "turnover", "trades",
]
SNAPSHOT_COLS = [
    "symbol", "series", "isin", "name", "open", "high", "low", "close",
    "last", "prev_close", "volume", "turnover", "trades",
]
ACTION_COLS = [
    "exchange", "symbol", "isin", "company", "ex_date", "record_date", "type",
    "ratio_num", "ratio_den", "cash_amount", "face_value_from", "face_value_to",
    "raw_subject",
]
# Go marshals these with omitempty; drop when null so the JSON is identical.
ACTION_OPTIONAL = {
    "record_date", "ratio_num", "ratio_den", "cash_amount",
    "face_value_from", "face_value_to",
}


class ExportError(RuntimeError):
    """Raised when an export cannot proceed."""


@dataclass(frozen=True)
class ExportResult:
    ohlcv_files: int
    snapshot_files: int
    action_files: int


def read_bhavcopy(paths: Iterable[Path | str]) -> pl.DataFrame:
    """Read bhavcopy parquet that may mix daily files and year rollups.

    Rollups carry ``year``/``month`` as real columns, dailies only in the
    hive path. Normalise per file, then concat relaxed.
    """
    frames = []
    for p in paths:
        f = pl.read_parquet(p, hive_partitioning=False)
        f = f.drop([c for c in ("year", "month") if c in f.columns])
        frames.append(f)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def export_ohlcv(prices: pl.DataFrame, exchange: str, out_dir: Path) -> int:
    """One file per symbol, all series, rows ordered by (date, series)."""
    if prices.is_empty():
        return 0
    df = (
        prices
        .with_columns(pl.col("trades").fill_null(0))
        .sort(["symbol", "date", "series"])
    )
    root = out_dir / "api" / "v1" / "ohlcv" / exchange.lower()
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for (symbol,), part in df.partition_by("symbol", as_dict=True, maintain_order=True).items():
        _write(root / f"{symbol}.json", part.select(OHLCV_COLS))
        n += 1
    return n


def export_snapshots(
    prices: pl.DataFrame, exchange: str, out_dir: Path, years: Iterable[int] | None
) -> int:
    """One file per trading date, restricted to ``years`` when given."""
    if prices.is_empty():
        return 0
    df = prices.with_columns(
        pl.col("isin").fill_null(""),
        pl.col("name").fill_null(""),
        pl.col("trades").fill_null(0),
    )
    if years is not None:
        yrs = list(years)
        df = df.filter(pl.col("date").dt.year().is_in(yrs))
    df = df.sort(["date", "symbol"])
    root = out_dir / "api" / "v1" / "snapshot" / exchange.lower()
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for (d,), part in df.partition_by("date", as_dict=True, maintain_order=True).items():
        _write(root / f"{d.isoformat()}.json", part.select(SNAPSHOT_COLS))
        n += 1
    return n


def export_actions(actions: pl.DataFrame, out_dir: Path) -> int:
    """One file per symbol across exchanges, rows by ex_date desc."""
    if actions.is_empty():
        return 0
    df = actions.select(ACTION_COLS).sort(["symbol", "ex_date"], descending=[False, True])
    root = out_dir / "api" / "v1" / "actions"
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for (symbol,), part in df.partition_by("symbol", as_dict=True, maintain_order=True).items():
        rows = [
            {k: v for k, v in row.items() if not (k in ACTION_OPTIONAL and v is None)}
            for row in part.to_dicts()
        ]
        (root / f"{symbol}.json").write_text(
            json.dumps({"data": rows}, default=_json_default, separators=(",", ":")),
            encoding="utf-8",
        )
        n += 1
    return n


def export_api_json(
    *,
    prices_dir: Path,
    actions_dir: Path | None,
    out_dir: Path,
    exchanges: Iterable[str],
    snapshot_years: Iterable[int] | None,
    only: Iterable[str] = ("ohlcv", "snapshot", "actions"),
) -> ExportResult:
    """Export the selected artifacts. ``snapshot_years=None`` means all years."""
    only = set(only)
    unknown = only - {"ohlcv", "snapshot", "actions"}
    if unknown:
        raise ExportError(f"unknown export kinds: {sorted(unknown)}")
    ohlcv_n = snap_n = act_n = 0
    if only & {"ohlcv", "snapshot"}:
        for ex in exchanges:
            paths = sorted((prices_dir / ex.lower()).rglob("*.parquet"))
            if not paths:
                raise ExportError(f"no bhavcopy parquet under {prices_dir / ex.lower()}")
            prices = read_bhavcopy(paths)
            if "ohlcv" in only:
                ohlcv_n += export_ohlcv(prices, ex, out_dir)
            if "snapshot" in only:
                snap_n += export_snapshots(prices, ex, out_dir, snapshot_years)
    if "actions" in only and actions_dir is not None:
        apaths = sorted(Path(actions_dir).glob("*.parquet"))
        if apaths:
            act_n = export_actions(pl.concat([pl.read_parquet(p) for p in apaths], how="vertical_relaxed"), out_dir)
    return ExportResult(ohlcv_files=ohlcv_n, snapshot_files=snap_n, action_files=act_n)


def _write(path: Path, df: pl.DataFrame) -> None:
    # polars row-oriented JSON: dates as YYYY-MM-DD, compact, no trailing newline.
    path.write_text('{"data":' + df.write_json() + "}", encoding="utf-8")


def _json_default(v):  # noqa: ANN001
    if isinstance(v, date_cls):
        return v.isoformat()
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")
