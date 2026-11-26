"""Backfill ISINs across the legacy bhavcopy cutover.

NSE bhavcopies before 2012-01-02 carry no ISIN. Every derived step keys on
ISIN, so without this the adjust step applies no corporate actions to 2010
and 2011 and the metrics step lumps every ISIN-less row into one series.

Rule: a symbol inherits its first observed ISIN backwards only if it traded
on both the last ISIN-less day and the first ISIN day. That continuity is
strong evidence it is the same instrument; a ticker that vanished before the
cutover and was reused later is not touched. Rows that stay null are keyed
by symbol downstream (``series_key``) so they never share a window.

The raw bhavcopy parquet is never rewritten; this applies to derived
outputs only, and the README documents the null ISINs in the raw data.
"""
from __future__ import annotations

import polars as pl

# A symbol's last ISIN-less session and its first session with an ISIN this close together are one
# instrument. Some 2011 files already carry ISINs for part of the market, before the cutover.
MAX_GAP_DAYS = 7


def isin_anchors(prices: pl.DataFrame) -> pl.DataFrame:
    """Return ``(symbol, isin)`` for symbols continuous across the cutover.

    ``prices`` needs ``date``, ``symbol``, ``isin`` and should span both sides
    of the cutover (at least the two years around it). Empty result when
    there is no cutover in the data.
    """
    if prices.is_empty() or "isin" not in prices.columns:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "isin": pl.Utf8})
    has = prices.filter(pl.col("isin").is_not_null() & (pl.col("isin") != ""))
    missing = prices.filter(pl.col("isin").is_null() | (pl.col("isin") == ""))
    if has.is_empty() or missing.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "isin": pl.Utf8})
    first_isin_day = has["date"].min()
    last_null_day = missing.filter(pl.col("date") < first_isin_day)["date"].max()
    if last_null_day is None:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "isin": pl.Utf8})
    after = (
        has.filter(pl.col("date") == first_isin_day)
        .sort(["symbol", "isin"])
        .unique(subset=["symbol"], keep="first")
        .select("symbol", "isin")
    )
    before = missing.filter(pl.col("date") == last_null_day).select("symbol").unique()
    cutover = after.join(before, on="symbol", how="inner")
    return pl.concat([cutover, _early_anchors(has, missing).join(cutover, on="symbol", how="anti")]).sort("symbol")


def _early_anchors(has: pl.DataFrame, missing: pl.DataFrame) -> pl.DataFrame:
    """Symbols whose ISIN-less run ends within ``MAX_GAP_DAYS`` of their first ISIN session."""
    last_null = missing.group_by("symbol").agg(pl.col("date").max().alias("_last_null"))
    first = (
        has.sort(["symbol", "date", "isin"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("date").first().alias("_first"), pl.col("isin").first())
    )
    gap = (pl.col("_first") - pl.col("_last_null")).dt.total_days()
    return (
        first.join(last_null, on="symbol", how="inner")
        .filter((gap > 0) & (gap <= MAX_GAP_DAYS))
        .select("symbol", "isin")
    )


def apply_isin_anchors(prices: pl.DataFrame, anchors: pl.DataFrame) -> pl.DataFrame:
    """Fill null or empty ISINs from ``anchors`` by symbol. Other rows unchanged."""
    if anchors.is_empty() or "isin" not in prices.columns:
        return prices
    return (
        prices.join(anchors.rename({"isin": "_isin_fill"}), on="symbol", how="left")
        .with_columns(
            pl.when(pl.col("isin").is_null() | (pl.col("isin") == ""))
            .then(pl.col("_isin_fill"))
            .otherwise(pl.col("isin"))
            .alias("isin")
        )
        .drop("_isin_fill")
    )


def series_key() -> pl.Expr:
    """Partition key for per-instrument windows: ISIN, else the symbol.

    Null ISINs would otherwise collapse into a single polars group.
    """
    return (
        pl.when(pl.col("isin").is_null() | (pl.col("isin") == ""))
        .then(pl.concat_str([pl.lit("sym:"), pl.col("symbol")]))
        .otherwise(pl.col("isin"))
    )
