"""Unusual activity: days on which a stock did something far outside its own
recent normal.

Every measure is compared with the same stock's previous LOOKBACK sessions,
never with the market, and never with the day itself. Prices and volume are
split and bonus adjusted (volume divided by the cumulative factor) so a split
does not read as a crash or a volume spike. Stocks below MIN_TURNOVER of mean
daily turnover over the previous 20 sessions, and ETF and fund units (ISIN
prefix INF), are skipped.

One output row per (date, symbol, signal):

  volume     value = volume / mean of previous 20 sessions, z on log volume
  move       value = adjusted 1-day return, z = return / std of previous returns
  gap        value = adjusted open / previous adjusted close - 1, z as move
  delivery   value = delivery %, z against previous delivery % (high side only)
  fut_oi     value = % change in total futures open interest, z against previous;
             skipped within a session of a split or bonus, when OI in units jumps,
             and until the stock has 20 straight sessions of futures history
  high_52w   value = adjusted close above the previous 252-session high, on
  low_52w      at least twice normal volume; z is the volume z
"""

from __future__ import annotations

import polars as pl

LOOKBACK = 60
MIN_OBS = 40
MIN_TURNOVER = 1e7
Z_MIN = 3.0

UNUSUAL_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "isin": pl.Utf8,
    "signal": pl.Utf8,
    "value": pl.Float64,
    "normal": pl.Float64,
    "z": pl.Float64,
    "close": pl.Float64,
    "ret_1d": pl.Float64,
    "volume_ratio": pl.Float64,
    "turnover": pl.Float64,
}

SERIES_RANK = {"EQ": 0, "BE": 1, "BZ": 2}


def _prior_mean(col: str, n: int = LOOKBACK) -> pl.Expr:
    return pl.col(col).shift(1).rolling_mean(n, min_samples=MIN_OBS if n == LOOKBACK else n).over("symbol")


def _prior_std(col: str, n: int = LOOKBACK) -> pl.Expr:
    return pl.col(col).shift(1).rolling_std(n, min_samples=MIN_OBS).over("symbol")


def _base(prices: pl.DataFrame) -> pl.DataFrame:
    p = (
        prices.with_columns(pl.col("series").replace_strict(SERIES_RANK, default=9).alias("_r"))
        .sort(["symbol", "date", "_r"])
        .unique(["symbol", "date"], keep="first", maintain_order=True)
        .with_columns(
            (pl.col("volume") / pl.col("adj_factor_cumulative")).alias("adj_volume"),
            (pl.col("open") * pl.col("adj_factor_cumulative")).alias("adj_open"),
        )
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("ret_1d"),
            (pl.col("adj_open") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("gap"),
            pl.col("adj_volume").clip(lower_bound=1).log().alias("log_vol"),
            (pl.col("adj_factor_cumulative").shift(1).over("symbol") / pl.col("adj_factor_cumulative") < 0.95).alias("_step"),
        )
        .with_columns(
            (pl.col("_step") | pl.col("_step").shift(-1).over("symbol") | pl.col("_step").shift(1).over("symbol"))
            .fill_null(False)
            .alias("near_split")
        )
    )
    return p.with_columns(
        _prior_mean("adj_volume", 20).alias("vol_mean20"),
        _prior_mean("turnover", 20).alias("turnover_mean20"),
        _prior_mean("log_vol").alias("log_vol_mean"),
        _prior_std("log_vol").alias("log_vol_std"),
        _prior_std("ret_1d").alias("ret_std"),
        pl.col("adj_close").shift(1).rolling_max(252, min_samples=200).over("symbol").alias("prior_high"),
        pl.col("adj_close").shift(1).rolling_min(252, min_samples=200).over("symbol").alias("prior_low"),
    ).with_columns(
        (pl.col("adj_volume") / pl.col("vol_mean20")).alias("volume_ratio"),
        ((pl.col("log_vol") - pl.col("log_vol_mean")) / pl.col("log_vol_std")).alias("vol_z"),
    )


def _delivery(delivery: pl.DataFrame) -> pl.DataFrame:
    d = (
        delivery.with_columns(pl.col("series").replace_strict(SERIES_RANK, default=9).alias("_r"))
        .sort(["symbol", "date", "_r"])
        .unique(["symbol", "date"], keep="first", maintain_order=True)
        .select("date", "symbol", "delivery_pct")
    )
    return d.with_columns(
        _prior_mean("delivery_pct").alias("dlv_mean"),
        _prior_std("delivery_pct").alias("dlv_std"),
    )


def _fut_oi(fo_stats: pl.DataFrame) -> pl.DataFrame:
    f = (
        fo_stats.filter(pl.col("fut_oi").is_not_null())
        .group_by(["date", "symbol"])
        .agg(pl.col("fut_oi").sum(), pl.col("fut_oi_change").sum())
        .sort(["symbol", "date"])
        .with_columns(
            pl.when(pl.col("fut_oi") - pl.col("fut_oi_change") > 0)
            .then(pl.col("fut_oi_change") / (pl.col("fut_oi") - pl.col("fut_oi_change")) * 100)
            .otherwise(None)
            .alias("oi_pct")
        )
    )
    settled = (pl.col("date") - pl.col("date").shift(20).over("symbol")).dt.total_days() <= 35
    return f.with_columns(
        _prior_mean("oi_pct").alias("oi_mean"),
        _prior_std("oi_pct").alias("oi_std"),
        settled.fill_null(False).alias("oi_settled"),
    ).select("date", "symbol", "oi_pct", "oi_mean", "oi_std", "oi_settled")


def _signal(df: pl.DataFrame, name: str, cond: pl.Expr, value: pl.Expr, normal: pl.Expr, z: pl.Expr) -> pl.DataFrame:
    return df.filter(cond).select(
        "date",
        "symbol",
        "isin",
        pl.lit(name).alias("signal"),
        value.cast(pl.Float64).alias("value"),
        normal.cast(pl.Float64).alias("normal"),
        z.cast(pl.Float64).alias("z"),
        pl.col("close"),
        "ret_1d",
        "volume_ratio",
        "turnover",
    )


def compute_unusual(
    prices: pl.DataFrame,
    delivery: pl.DataFrame | None = None,
    fo_stats: pl.DataFrame | None = None,
    since=None,
) -> pl.DataFrame:
    """Unusual-activity rows for every day in `prices` on or after `since`.

    `prices` is the adjusted price tree for one exchange and should start at
    least 252 sessions before `since` so the windows are full.
    """
    if prices.is_empty():
        return pl.DataFrame(schema=UNUSUAL_SCHEMA)
    df = _base(prices)
    if delivery is not None and not delivery.is_empty():
        df = df.join(_delivery(delivery), on=["date", "symbol"], how="left")
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias(c) for c in ("delivery_pct", "dlv_mean", "dlv_std"))
    if fo_stats is not None and not fo_stats.is_empty():
        df = df.join(_fut_oi(fo_stats), on=["date", "symbol"], how="left")
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias(c) for c in ("oi_pct", "oi_mean", "oi_std")).with_columns(pl.lit(False).alias("oi_settled"))
    if since is not None:
        df = df.filter(pl.col("date") >= since)
    df = df.filter((pl.col("turnover_mean20") >= MIN_TURNOVER) & ~pl.col("isin").fill_null("").str.starts_with("INF"))

    move_z = pl.col("ret_1d") / pl.col("ret_std")
    gap_z = pl.col("gap") / pl.col("ret_std")
    dlv_z = (pl.col("delivery_pct") - pl.col("dlv_mean")) / pl.col("dlv_std")
    oi_z = (pl.col("oi_pct") - pl.col("oi_mean")) / pl.col("oi_std")
    vol_ok = pl.col("volume_ratio") >= 2

    parts = [
        _signal(df, "volume", (pl.col("vol_z") >= Z_MIN) & (pl.col("volume_ratio") >= 3),
                pl.col("volume_ratio"), pl.lit(1.0), pl.col("vol_z")),
        _signal(df, "move", (move_z.abs() >= Z_MIN) & (pl.col("ret_1d").abs() >= 0.03),
                pl.col("ret_1d"), pl.col("ret_std"), move_z),
        _signal(df, "gap", (gap_z.abs() >= Z_MIN) & (pl.col("gap").abs() >= 0.02) & (pl.col("volume_ratio") >= 0.5),
                pl.col("gap"), pl.col("ret_std"), gap_z),
        _signal(df, "delivery", (dlv_z >= Z_MIN) & (pl.col("volume_ratio") >= 0.5),
                pl.col("delivery_pct"), pl.col("dlv_mean"), dlv_z),
        _signal(df, "fut_oi", (oi_z.abs() >= Z_MIN) & (pl.col("oi_pct").abs() >= 10) & ~pl.col("near_split") & pl.col("oi_settled").fill_null(False),
                pl.col("oi_pct"), pl.col("oi_mean"), oi_z),
        _signal(df, "high_52w", (pl.col("adj_close") > pl.col("prior_high")) & vol_ok,
                pl.col("adj_close"), pl.col("prior_high"), pl.col("vol_z")),
        _signal(df, "low_52w", (pl.col("adj_close") < pl.col("prior_low")) & vol_ok,
                pl.col("adj_close"), pl.col("prior_low"), pl.col("vol_z")),
    ]
    out = pl.concat(parts, how="vertical_relaxed")
    return out.select([pl.col(c).cast(t) for c, t in UNUSUAL_SCHEMA.items()]).sort(["date", "symbol", "signal"])
