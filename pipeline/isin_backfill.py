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
    return after.join(before, on="symbol", how="inner").sort("symbol")


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
