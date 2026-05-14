"""Build per-ISIN symbol-history intervals from a bhavcopy price series.

Tickers change. A company can rename (e.g. WIPRO -> WIPROBPO -> ITES) or be
delisted then re-list under a new symbol. ISIN is the stable identifier.
This module scans a price series and emits a table of `(isin, symbol,
valid_from, valid_to)` intervals: for every ISIN, the contiguous date
ranges during which it traded under each symbol.

The output makes it possible to reverse-resolve `(symbol, date)` -> ISIN
even after a rename, and gives downstream consumers a single source of
truth for current-symbol-per-ISIN.
"""

from __future__ import annotations

import polars as pl

SYMBOL_HISTORY_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "isin": pl.Utf8,
    "symbol": pl.Utf8,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
    "trading_days": pl.Int64,
}


def build_symbol_history(prices: pl.DataFrame, exchange: str) -> pl.DataFrame:
    """Collapse a price series into per-ISIN symbol intervals.

    `prices` must have columns (date, symbol, isin). Rows with null ISIN
    are dropped (cannot anchor an interval). Within each ISIN, consecutive
    trading days under the same symbol form a single interval; whenever
    the symbol changes a new interval starts.

    Returns a DataFrame matching SYMBOL_HISTORY_SCHEMA, sorted by
    (isin, valid_from).
    """
    _require_columns(prices, ["date", "symbol", "isin"])

    df = prices.filter(
        pl.col("isin").is_not_null() & pl.col("symbol").is_not_null()
    ).select(["isin", "symbol", "date"])

    if df.height == 0:
        return pl.DataFrame(schema=SYMBOL_HISTORY_SCHEMA)

    df = df.sort(["isin", "date"]).with_columns(
        # Within each ISIN, mark where the symbol differs from the previous row.
        # First row of an ISIN group has shift==null -> treat as new segment.
        _new_segment=(
            (pl.col("symbol") != pl.col("symbol").shift(1).over("isin"))
            | pl.col("symbol").shift(1).over("isin").is_null()
        )
    ).with_columns(
        _segment_id=pl.col("_new_segment").cum_sum().over("isin"),
    )

    intervals = (
        df.group_by(["isin", "symbol", "_segment_id"])
        .agg(
            valid_from=pl.col("date").min(),
            valid_to=pl.col("date").max(),
            trading_days=pl.len(),
        )
        .drop("_segment_id")
        .with_columns(exchange=pl.lit(exchange, dtype=pl.Utf8))
        .select(["exchange", "isin", "symbol", "valid_from", "valid_to", "trading_days"])
        .sort(["isin", "valid_from"])
    )
    return intervals.cast(SYMBOL_HISTORY_SCHEMA)  # type: ignore[arg-type]


def lookup_isin(history: pl.DataFrame, symbol: str, on_date) -> str | None:
    """Reverse lookup: which ISIN traded under `symbol` on `on_date`?

    Returns the matching ISIN, or None when no interval covers it. If
    multiple ISINs claim the same symbol on the same date (extremely rare
    but possible for delisted-and-reused symbols), the first match by ISIN
    sort order is returned.
    """
    hits = history.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("valid_from") <= on_date)
        & (pl.col("valid_to") >= on_date)
    ).sort("isin")
    if hits.height == 0:
        return None
    return hits["isin"][0]


def lookup_current_symbol(history: pl.DataFrame, isin: str) -> str | None:
    """Return the most recent symbol an ISIN traded under, or None."""
    hits = history.filter(pl.col("isin") == isin).sort("valid_to", descending=True)
    if hits.height == 0:
        return None
    return hits["symbol"][0]


def _require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
