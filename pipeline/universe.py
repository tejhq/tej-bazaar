"""Point-in-time liquidity universes: survivorship-bias-free by construction.

A backtest that uses today's NIFTY500 list for 2012 only sees companies
that survived to today. This module rebuilds membership from what was
actually trading on each rebalance date, using data available up to that
date only:

- Rank every instrument by trailing ``WINDOW``-trading-day mean turnover
  (rupees traded; invariant to splits, so raw bhavcopy is the right input).
- Rebalance on the first trading day of each calendar month; hold until
  the next rebalance. Monthly, not daily, so membership does not churn on
  one noisy session.
- Keep the top ``TOP_N`` per rebalance. ``liquid100`` / ``liquid250`` /
  ``liquid500`` are ``rank <= N`` filters over the same rows.

Delisted names stay in the months they qualified for. Symbols are the
symbols in force on the rebalance date.

Output schema (one file per exchange, ``universe/<ex>_liquid.parquet``)::

    exchange           Utf8    NSE | BSE
    rebalance_date     Date    First trading day of the month
    valid_to           Date    Day before the next rebalance (last month: last date seen)
    rank               Int64   1 = highest trailing turnover
    symbol             Utf8    Symbol on rebalance_date
    isin               Utf8    ISIN on rebalance_date ("" when source lacks it)
    name               Utf8    Instrument name on rebalance_date
    avg_turnover_63d   Float64 Trailing mean turnover, rupees
"""
from __future__ import annotations

import polars as pl

WINDOW = 63
TOP_N = 500
MIN_HISTORY = WINDOW  # require a full window, same rule as metrics

UNIVERSE_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "rebalance_date": pl.Date,
    "valid_to": pl.Date,
    "rank": pl.Int64,
    "symbol": pl.Utf8,
    "isin": pl.Utf8,
    "name": pl.Utf8,
    "avg_turnover_63d": pl.Float64,
}


def build_universe(prices: pl.DataFrame, exchange: str, *, top_n: int = TOP_N, window: int = WINDOW) -> pl.DataFrame:
    """Compute monthly point-in-time top-``top_n`` by trailing turnover.

    ``prices`` is raw bhavcopy for one exchange (any span of years). Rows
    are keyed by ISIN where present, else by symbol (pre-2012 NSE).
    """
    if prices.is_empty():
        return pl.DataFrame(schema=UNIVERSE_SCHEMA)

    df = (
        prices
        .select("date", "symbol", "isin", "name", "turnover")
        .with_columns(
            pl.col("isin").fill_null(""),
            pl.col("name").fill_null(""),
            pl.col("turnover").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("isin") == "").then(pl.col("symbol")).otherwise(pl.col("isin")).alias("key")
        )
        # One row per (key, date): a symbol listed in two series the same
        # day counts its turnover once, on the busier series.
        .sort(["key", "date", "turnover"], descending=[False, False, True])
        .unique(subset=["key", "date"], keep="first", maintain_order=True)
        .sort(["key", "date"])
        .with_columns(
            pl.col("turnover").rolling_mean(window_size=window, min_samples=window).over("key").alias("avg_turnover"),
        )
        .drop_nulls("avg_turnover")
    )
    if df.is_empty():
        return pl.DataFrame(schema=UNIVERSE_SCHEMA)

    # First trading day of each month, from the exchange calendar as seen in
    # the raw data (not `df`, which has already dropped bootstrap rows).
    trading_days = prices.select("date").unique().sort("date")
    rebalance_days = (
        trading_days
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month").agg(pl.col("date").min().alias("rebalance_date"))
        .sort("rebalance_date")
        .with_columns(
            (pl.col("rebalance_date").shift(-1) - pl.duration(days=1)).alias("valid_to")
        )
        .with_columns(
            pl.col("valid_to").fill_null(trading_days["date"].max())
        )
        .select("rebalance_date", "valid_to")
    )

    ranked = (
        df.join(rebalance_days, left_on="date", right_on="rebalance_date", how="inner")
        .rename({"date": "rebalance_date"})
        .sort(["rebalance_date", "avg_turnover", "key"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("rebalance_date").alias("rank"))
        .filter(pl.col("rank") <= top_n)
        .with_columns(pl.lit(exchange.upper()).alias("exchange"))
        .select(
            "exchange", "rebalance_date", "valid_to", "rank", "symbol", "isin", "name",
            pl.col("avg_turnover").alias("avg_turnover_63d"),
        )
        .sort(["rebalance_date", "rank"])
    )
    return ranked.cast(UNIVERSE_SCHEMA)


def members_as_of(universe: pl.DataFrame, as_of, top_n: int) -> pl.DataFrame:
    """Rows in force on ``as_of`` (the latest rebalance on or before it), rank <= top_n."""
    eligible = universe.filter(pl.col("rebalance_date") <= as_of)
    if eligible.is_empty():
        return universe.clear()
    latest = eligible["rebalance_date"].max()
    return eligible.filter((pl.col("rebalance_date") == latest) & (pl.col("rank") <= top_n)).sort("rank")
