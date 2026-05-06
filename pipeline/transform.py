"""Transform parsed bhavcopy DataFrame: filter, validate, dedupe, sort.

Input is the normalized DF from parse.py. Output is publication-ready.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import polars as pl

Exchange = Literal["NSE", "BSE"]

# Equity-segment series codes per exchange.
# NSE: EQ = main board, BE/BZ = trade-to-trade (still settled by delivery).
# BSE: A = large-cap, B = small/mid, T = trade-to-trade.
EQUITY_SERIES: dict[Exchange, tuple[str, ...]] = {
    "NSE": ("EQ", "BE", "BZ"),
    "BSE": ("A", "B", "T"),
}

# Back-compat: NSE default.
DEFAULT_EQUITY_SERIES: tuple[str, ...] = EQUITY_SERIES["NSE"]


class TransformError(ValueError):
    """Raised when input violates an invariant we cannot recover from."""


def transform(
    df: pl.DataFrame,
    *,
    exchange: Exchange = "NSE",
    series: Iterable[str] | None = None,
    drop_zero_volume: bool = True,
) -> pl.DataFrame:
    """Filter to equity series, drop bad/null rows, dedupe, sort.

    - Filters to `series` (defaults from EQUITY_SERIES[exchange]).
    - Drops rows with null OHLC or null volume.
    - Drops zero-volume rows when `drop_zero_volume` (no real trading happened).
    - Drops rows that violate price invariants (low > high, low > open/close, etc.).
    - Dedupes on (date, symbol), keeping the first occurrence.
    - Sorts by (date asc, symbol asc).
    """
    if df.height == 0:
        return df

    required = {"date", "symbol", "series", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise TransformError(f"input missing required columns: {sorted(missing)}")

    series_set = list(series) if series is not None else list(EQUITY_SERIES[exchange])
    out = df.filter(pl.col("series").is_in(series_set))

    out = out.drop_nulls(subset=["open", "high", "low", "close", "volume"])

    if drop_zero_volume:
        out = out.filter(pl.col("volume") > 0)

    valid_prices = (
        (pl.col("low") <= pl.col("high"))
        & (pl.col("low") <= pl.col("open"))
        & (pl.col("low") <= pl.col("close"))
        & (pl.col("high") >= pl.col("open"))
        & (pl.col("high") >= pl.col("close"))
        & (pl.col("open") > 0)
        & (pl.col("close") > 0)
    )
    out = out.filter(valid_prices)

    out = out.unique(subset=["date", "symbol"], keep="first", maintain_order=True)
    out = out.sort(["date", "symbol"])

    return out
