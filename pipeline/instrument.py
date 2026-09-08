"""Instrument chains: one key per listed company across ISIN changes.

In India a face value split, a scheme of arrangement, or a re-listing issues
a new ISIN for the same instrument. The bhavcopy then carries the old ISIN
up to the ex date and the new one from the ex date on. Every derived step
that partitions by ISIN alone (back-adjustment, rolling windows, universe
eligibility) silently treats the two halves as unrelated instruments: the
split factor never reaches the old rows, a 52 week window restarts, and the
name drops out of the liquidity universe for a full lookback.

This module links ISIN intervals into chains. Two intervals belong to the
same chain when the same symbol moved from one ISIN to another with at most
``MAX_GAP_DAYS`` calendar days between the last trading day of the old
interval and the first of the new. A symbol that vanished for longer and
came back on a fresh ISIN is treated as a different instrument, which also
covers ticker reuse by an unrelated company.

The chain id is the ISIN with the latest trading day in the chain, so it is
the current ISIN for live names. Raw ``isin`` columns in every published
tree stay exactly as the exchange reported them; the chain is an internal
partition key only.
"""
from __future__ import annotations

import polars as pl

from pipeline.symbol_history import build_symbol_history

MAX_GAP_DAYS = 30

CHAIN_SCHEMA: dict[str, pl.DataType] = {"isin": pl.Utf8, "chain_isin": pl.Utf8}


def chain_map(prices: pl.DataFrame, *, max_gap_days: int = MAX_GAP_DAYS) -> pl.DataFrame:
    """Return ``(isin, chain_isin)`` for every ISIN seen in ``prices``.

    ``prices`` needs ``date``, ``symbol``, ``isin``; span the full history
    or chains that cross a year boundary will not link. ISINs that never
    changed map to themselves.
    """
    history = build_symbol_history(prices, "X")
    if history.is_empty():
        return pl.DataFrame(schema=CHAIN_SCHEMA)

    # Links: consecutive intervals of the same symbol on different ISINs,
    # close enough in time to be the same instrument.
    ordered = history.sort(["symbol", "valid_from"]).with_columns(
        pl.col("isin").shift(1).over("symbol").alias("_prev_isin"),
        pl.col("valid_to").shift(1).over("symbol").alias("_prev_to"),
    )
    links = ordered.filter(
        pl.col("_prev_isin").is_not_null()
        & (pl.col("_prev_isin") != pl.col("isin"))
        & ((pl.col("valid_from") - pl.col("_prev_to")).dt.total_days() <= max_gap_days)
    ).select("_prev_isin", "isin")

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links.iter_rows():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    last_day = history.group_by("isin").agg(pl.col("valid_to").max()).to_dict(as_series=False)
    latest: dict[str, tuple] = {}
    for isin, day in zip(last_day["isin"], last_day["valid_to"]):
        root = find(isin)
        if root not in latest or (day, isin) > latest[root]:
            latest[root] = (day, isin)

    isins = sorted(set(last_day["isin"]))
    return pl.DataFrame(
        {"isin": isins, "chain_isin": [latest[find(i)][1] for i in isins]},
        schema=CHAIN_SCHEMA,
    )


def key_expr() -> pl.Expr:
    """Partition key once ``chain_isin`` is joined on: chain, else ISIN, else symbol."""
    return pl.coalesce(
        pl.col("chain_isin"),
        pl.when(pl.col("isin").is_null() | (pl.col("isin") == "")).then(None).otherwise(pl.col("isin")),
        pl.concat_str([pl.lit("sym:"), pl.col("symbol")]),
    )


def attach_key(df: pl.DataFrame, chain: pl.DataFrame) -> pl.DataFrame:
    """Add ``_key`` to ``df`` (needs ``isin``, ``symbol``). ``chain_isin`` is not kept."""
    if chain.is_empty():
        joined = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("chain_isin"))
    else:
        joined = df.join(chain, on="isin", how="left")
    return joined.with_columns(key_expr().alias("_key")).drop("chain_isin")


def remap_isin(df: pl.DataFrame, chain: pl.DataFrame) -> pl.DataFrame:
    """Replace ``isin`` by its chain id for computation, keeping the original in ``_isin_raw``.

    Rows whose ISIN is not in ``chain`` (null, or unseen) keep their value.
    Undo with :func:`restore_isin`.
    """
    out = df.with_columns(pl.col("isin").alias("_isin_raw"))
    if chain.is_empty():
        return out
    return (
        out.join(chain, on="isin", how="left")
        .with_columns(pl.coalesce(pl.col("chain_isin"), pl.col("isin")).alias("isin"))
        .drop("chain_isin")
    )


def restore_isin(df: pl.DataFrame) -> pl.DataFrame:
    """Put the exchange-reported ISIN back after :func:`remap_isin`."""
    if "_isin_raw" not in df.columns:
        return df
    return df.with_columns(pl.col("_isin_raw").alias("isin")).drop("_isin_raw")
