"""Rolling-window metrics: 52-week highs/lows and average liquidity.

The price-based metrics (52-week high / low, distance from each) use
back-adjusted close so splits and bonuses don't create artificial new
"lows" the day after the event. Volume and turnover come straight from
the raw bhavcopy, unadjusted:

  - Turnover (rupees changed hands) is invariant to splits, so the raw
    series is the right signal for "how much money is trading."
  - Volume (share count) does jump on splits, but a 20-day mean as a
    liquidity filter only cares about the recent window. Splits inside
    a 20-day window are rare enough that we accept the distortion in
    exchange for keeping volume as the exchange reported it.

Rolling windows are positional over each ISIN's sorted trading days, so
"20-day average volume" means the mean of the last 20 trading days
regardless of calendar gaps from weekends or holidays.

Bootstrap rows (fewer than the window's worth of prior history for that
ISIN) get null at that horizon. We require a full window rather than
silently degrading: a "52-week high" computed off 40 days is not a
52-week high.

Output schema:

    date                   Date    Trading date
    symbol                 Utf8    Symbol on `date`
    isin                   Utf8    Stable instrument ID (partition key)
    adj_close              Float64 Back-adjusted close
    high_52w               Float64 Max adj_close over last 252 trading days
    low_52w                Float64 Min adj_close over last 252 trading days
    pct_off_52w_high       Float64 adj_close / high_52w - 1 (<= 0)
    pct_off_52w_low        Float64 adj_close / low_52w - 1 (>= 0)
    avg_vol_20d            Float64 Mean raw volume over last 20 trading days
    avg_vol_60d            Float64 Mean raw volume over last 60 trading days
    avg_turnover_20d       Float64 Mean raw turnover over last 20 trading days
"""

from __future__ import annotations

import polars as pl

WINDOW_52W = 252
WINDOW_VOL_SHORT = 20
WINDOW_VOL_LONG = 60
WINDOW_TURNOVER = 20

ROLLING_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "isin": pl.Utf8,
    "adj_close": pl.Float64,
    "high_52w": pl.Float64,
    "low_52w": pl.Float64,
    "pct_off_52w_high": pl.Float64,
    "pct_off_52w_low": pl.Float64,
    "avg_vol_20d": pl.Float64,
    "avg_vol_60d": pl.Float64,
    "avg_turnover_20d": pl.Float64,
}


def compute_rolling(prices_with_adj: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling window metrics from adjusted prices + raw activity.

    Input must have `(isin, date, symbol, adj_close, volume, turnover)`.
    Other columns are dropped. Windows are positional per ISIN: the
    function sorts internally, so input order is irrelevant.

    The 252 / 60 / 20 trading-day windows demand a full window's worth
    of prior data; rows with fewer prior bars produce null at that
    horizon. This avoids reporting a "52-week high" that's really a
    40-day high.
    """
    _require_columns(
        prices_with_adj,
        ["isin", "date", "symbol", "adj_close", "volume", "turnover"],
    )

    if prices_with_adj.height == 0:
        return pl.DataFrame(schema=ROLLING_SCHEMA)

    df = prices_with_adj.select(
        ["isin", "date", "symbol", "adj_close", "volume", "turnover"]
    ).sort(["isin", "date"])

    # `over("isin")` keeps each ISIN's rolling window self-contained, so
    # the first 252 rows of ISIN B never reach back into ISIN A's tail.
    return df.with_columns(
        high_52w=pl.col("adj_close")
        .rolling_max(window_size=WINDOW_52W, min_samples=WINDOW_52W)
        .over("isin"),
        low_52w=pl.col("adj_close")
        .rolling_min(window_size=WINDOW_52W, min_samples=WINDOW_52W)
        .over("isin"),
        avg_vol_20d=pl.col("volume")
        .cast(pl.Float64)
        .rolling_mean(window_size=WINDOW_VOL_SHORT, min_samples=WINDOW_VOL_SHORT)
        .over("isin"),
        avg_vol_60d=pl.col("volume")
        .cast(pl.Float64)
        .rolling_mean(window_size=WINDOW_VOL_LONG, min_samples=WINDOW_VOL_LONG)
        .over("isin"),
        avg_turnover_20d=pl.col("turnover")
        .cast(pl.Float64)
        .rolling_mean(window_size=WINDOW_TURNOVER, min_samples=WINDOW_TURNOVER)
        .over("isin"),
    ).with_columns(
        pct_off_52w_high=(pl.col("adj_close") / pl.col("high_52w") - 1.0),
        pct_off_52w_low=(pl.col("adj_close") / pl.col("low_52w") - 1.0),
    ).select(list(ROLLING_SCHEMA.keys()))


def _require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
