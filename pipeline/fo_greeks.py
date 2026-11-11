"""Implied volatility and Greeks for traded NSE options, end of day.

Black-76 on the forward of the option's own expiry. The forward is the
same-expiry future's close when that future traded, else its settle price;
an expiry with no future (weeklies, some stock expiries) takes the forward
from put-call parity at the strike where call and put prices are closest.
Only contracts that traded that day (contracts > 0, close > 0) are priced:
the close of an untraded contract is not a price from that day. CA/PA
(American stock options before 2011) are treated as European.

Time to expiry counts calendar days from the close to the 15:30 expiry, over
365. Expiry-day rows (T = 0) are skipped. The rate is the RBI policy repo
rate in force that day. delta and gamma are with respect to the forward,
theta is per calendar day and vega per one volatility point.

fo_stats gains atm_iv (call and put IV at the strike nearest the forward,
averaged when both traded) and iv_skew (IV of the put nearest delta -0.25
minus IV of the call nearest delta 0.25, when both are within 0.1 of it).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

GREEKS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "instrument": pl.Utf8,
    "expiry": pl.Date,
    "strike": pl.Float64,
    "option_type": pl.Utf8,
    "close": pl.Float64,
    "forward": pl.Float64,
    "forward_source": pl.Utf8,
    "days": pl.Int32,
    "iv": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "theta": pl.Float64,
    "vega": pl.Float64,
}

IV_LOW, IV_HIGH = 0.005, 5.0

# RBI policy repo rate, effective date -> rate.
REPO_RATES: list[tuple[str, float]] = [
    ("2009-04-21", 4.75),
    ("2010-03-19", 5.00),
    ("2010-04-20", 5.25),
    ("2010-07-02", 5.50),
    ("2010-07-27", 5.75),
    ("2010-09-16", 6.00),
    ("2010-11-02", 6.25),
    ("2011-01-25", 6.50),
    ("2011-03-17", 6.75),
    ("2011-05-03", 7.25),
    ("2011-06-16", 7.50),
    ("2011-07-26", 8.00),
    ("2011-09-16", 8.25),
    ("2011-10-25", 8.50),
    ("2012-04-17", 8.00),
    ("2013-01-29", 7.75),
    ("2013-03-19", 7.50),
    ("2013-05-03", 7.25),
    ("2013-09-20", 7.50),
    ("2013-10-29", 7.75),
    ("2014-01-28", 8.00),
    ("2015-01-15", 7.75),
    ("2015-03-04", 7.50),
    ("2015-06-02", 7.25),
    ("2015-09-29", 6.75),
    ("2016-04-05", 6.50),
    ("2016-10-04", 6.25),
    ("2017-08-02", 6.00),
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
    ("2025-06-06", 5.50),
    ("2025-12-05", 5.25),
]


def repo_rate_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "from": [date.fromisoformat(d) for d, _ in REPO_RATES],
            "rate": [r / 100 for _, r in REPO_RATES],
        }
    ).sort("from")


def _ncdf(x: np.ndarray) -> np.ndarray:
    # Abramowitz and Stegun 26.2.17, |error| < 7.5e-8
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * a)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = _npdf(a) * poly
    return np.where(x >= 0, 1.0 - tail, tail)


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def black76(f, k, t, r, sigma, is_call):
    """Black-76 price; all array arguments broadcast."""
    f, k, t, r, sigma = (np.asarray(v, dtype=float) for v in (f, k, t, r, sigma))
    sq = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    df = np.exp(-r * t)
    call = df * (f * _ncdf(d1) - k * _ncdf(d2))
    put = df * (k * _ncdf(-d2) - f * _ncdf(-d1))
    return np.where(is_call, call, put)


def implied_vol(price, f, k, t, r, is_call, iters: int = 80) -> np.ndarray:
    """Bisection on Black-76. NaN where the price is outside the no-arbitrage
    bounds or needs a volatility outside [IV_LOW, IV_HIGH]."""
    price, f, k, t, r = (np.asarray(v, dtype=float) for v in (price, f, k, t, r))
    is_call = np.asarray(is_call, dtype=bool)
    lo = np.full(price.shape, IV_LOW)
    hi = np.full(price.shape, IV_HIGH)
    p_lo = black76(f, k, t, r, lo, is_call)
    p_hi = black76(f, k, t, r, hi, is_call)
    ok = (price > p_lo) & (price < p_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = black76(f, k, t, r, mid, is_call) > price
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def greeks(f, k, t, r, sigma, is_call) -> dict[str, np.ndarray]:
    f, k, t, r, sigma = (np.asarray(v, dtype=float) for v in (f, k, t, r, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    sqt = np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * t) / (sigma * sqt)
    df = np.exp(-r * t)
    pdf = _npdf(d1)
    price = black76(f, k, t, r, sigma, is_call)
    delta = np.where(is_call, df * _ncdf(d1), -df * _ncdf(-d1))
    gamma = df * pdf / (f * sigma * sqt)
    vega = f * df * pdf * sqt / 100.0
    theta = (r * price - f * df * pdf * sigma / (2.0 * sqt)) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _traded_options(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("instrument").is_in(["OPTIDX", "OPTSTK"])
        & (pl.col("contracts") > 0)
        & (pl.col("close") > 0)
        & pl.col("strike").is_not_null()
        & (pl.col("expiry") > pl.col("date"))
    ).with_columns(pl.col("option_type").replace({"CA": "CE", "PA": "PE"}).alias("ot"))


def _futures_forward(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("instrument").is_in(["FUTIDX", "FUTSTK"]))
        .with_columns(
            pl.when((pl.col("contracts") > 0) & (pl.col("close") > 0))
            .then(pl.col("close"))
            .otherwise(pl.col("settle_price"))
            .alias("fut_fwd")
        )
        .filter(pl.col("fut_fwd") > 0)
        .group_by(["date", "symbol", "expiry"])
        .agg(pl.col("fut_fwd").first())
    )


def _parity_forward(opt: pl.DataFrame) -> pl.DataFrame:
    keys = ["date", "symbol", "expiry", "strike"]
    ce = opt.filter(pl.col("ot") == "CE").group_by(keys).agg(pl.col("close").first().alias("c"))
    pe = opt.filter(pl.col("ot") == "PE").group_by(keys).agg(pl.col("close").first().alias("p"))
    both = ce.join(pe, on=keys, how="inner")
    return (
        both.with_columns((pl.col("c") - pl.col("p")).abs().alias("_gap"))
        .sort([*keys[:3], "_gap"])
        .group_by(["date", "symbol", "expiry"], maintain_order=True)
        .first()
        .with_columns(
            ((pl.col("expiry") - pl.col("date")).dt.total_days() / 365.0).alias("_t"),
        )
        .select("date", "symbol", "expiry", "strike", "c", "p", "_t")
    )


def compute_fo_greeks(df: pl.DataFrame) -> pl.DataFrame:
    """IV and Greeks for every traded option row in `df` (the F&O tree)."""
    opt = _traded_options(df)
    if opt.is_empty():
        return pl.DataFrame(schema=GREEKS_SCHEMA)
    rates = repo_rate_frame()

    par = (
        _parity_forward(opt)
        .sort("date")
        .join_asof(rates, left_on="date", right_on="from", strategy="backward")
    )
    par = par.with_columns(
        (
            pl.col("strike") + (pl.col("c") - pl.col("p")) * (pl.col("rate") * pl.col("_t")).exp()
        ).alias("par_fwd")
    ).select("date", "symbol", "expiry", "par_fwd")

    x = (
        opt.join(_futures_forward(df), on=["date", "symbol", "expiry"], how="left")
        .join(par, on=["date", "symbol", "expiry"], how="left")
        .with_columns(
            pl.coalesce("fut_fwd", "par_fwd").alias("forward"),
            pl.when(pl.col("fut_fwd").is_not_null())
            .then(pl.lit("future"))
            .when(pl.col("par_fwd").is_not_null())
            .then(pl.lit("parity"))
            .alias("forward_source"),
            (pl.col("expiry") - pl.col("date")).dt.total_days().cast(pl.Int32).alias("days"),
        )
        .filter(pl.col("forward") > 0)
        .sort("date")
        .join_asof(rates, left_on="date", right_on="from", strategy="backward")
    )
    if x.is_empty():
        return pl.DataFrame(schema=GREEKS_SCHEMA)

    f = x["forward"].to_numpy()
    k = x["strike"].to_numpy()
    t = x["days"].to_numpy() / 365.0
    r = x["rate"].to_numpy()
    is_call = (x["ot"] == "CE").to_numpy()
    iv = implied_vol(x["close"].to_numpy(), f, k, t, r, is_call)
    g = greeks(f, k, t, r, np.where(np.isnan(iv), 0.2, iv), is_call)
    nan = np.isnan(iv)
    cols = {name: pl.Series(name, np.where(nan, np.nan, v)).fill_nan(None) for name, v in g.items()}
    out = x.with_columns(pl.Series("iv", iv).fill_nan(None), *cols.values())
    return out.select([pl.col(c).cast(t) for c, t in GREEKS_SCHEMA.items()]).sort(
        ["date", "symbol", "expiry", "strike", "option_type"]
    )


def expiry_iv(greeks_df: pl.DataFrame) -> pl.DataFrame:
    """atm_iv and iv_skew per (date, symbol, expiry)."""
    keys = ["date", "symbol", "expiry"]
    g = greeks_df.filter(pl.col("iv").is_not_null()).with_columns(
        pl.col("option_type").replace({"CA": "CE", "PA": "PE"}).alias("ot")
    )
    if g.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "expiry": pl.Date,
                "atm_iv": pl.Float64,
                "iv_skew": pl.Float64,
            }
        )
    dist = (pl.col("strike") - pl.col("forward")).abs()
    atm_strike = g.group_by(keys).agg(pl.col("strike").sort_by(dist).first().alias("_atm"))
    atm = (
        g.join(atm_strike, on=keys)
        .filter(pl.col("strike") == pl.col("_atm"))
        .group_by(keys)
        .agg(pl.col("iv").mean().alias("atm_iv"))
    )

    def wing(ot: str, target: float, name: str) -> pl.DataFrame:
        gap = (pl.col("delta") - target).abs()
        return (
            g.filter(pl.col("ot") == ot)
            .group_by(keys)
            .agg(pl.col("iv").sort_by(gap).first().alias(name), gap.min().alias(f"_{name}_gap"))
        )

    put = wing("PE", -0.25, "_p25")
    call = wing("CE", 0.25, "_c25")
    skew = (
        put.join(call, on=keys)
        .with_columns(
            pl.when((pl.col("__p25_gap") <= 0.1) & (pl.col("__c25_gap") <= 0.1))
            .then(pl.col("_p25") - pl.col("_c25"))
            .alias("iv_skew")
        )
        .select(*keys, "iv_skew")
    )
    return atm.join(skew, on=keys, how="left")
