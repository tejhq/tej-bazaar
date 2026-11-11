"""Daily F&O statistics per underlying and expiry, derived from the F&O tree.

One row per (date, symbol, expiry). Option figures cover CE/CA and PE/PA
contracts of that expiry; futures figures come from the future of that
expiry. Open interest is in units, as NSE publishes it.

max_pain is the strike at which option holders of that expiry would be paid
the least if the underlying settled there, using that day's open interest.
buildup classifies the future's day: settle price change against the
previous session's settle of the same contract, and the change in open
interest. rollover_pct is futures open interest outside the nearest expiry as
a share of all futures open interest of the symbol that day. is_nearest
marks the nearest expiry with any contract, often a weekly option expiry.
atm_iv and iv_skew come from `pipeline.fo_greeks.expiry_iv` when passed in.
"""

from __future__ import annotations

import polars as pl

GROUP = ["date", "symbol", "expiry"]

STATS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "kind": pl.Utf8,
    "expiry": pl.Date,
    "is_nearest": pl.Boolean,
    "ce_oi": pl.Int64,
    "pe_oi": pl.Int64,
    "ce_oi_change": pl.Int64,
    "pe_oi_change": pl.Int64,
    "ce_contracts": pl.Int64,
    "pe_contracts": pl.Int64,
    "pcr_oi": pl.Float64,
    "pcr_volume": pl.Float64,
    "max_pain": pl.Float64,
    "strikes": pl.Int64,
    "fut_settle": pl.Float64,
    "fut_settle_change": pl.Float64,
    "fut_oi": pl.Int64,
    "fut_oi_change": pl.Int64,
    "fut_contracts": pl.Int64,
    "buildup": pl.Utf8,
    "underlying": pl.Float64,
    "basis": pl.Float64,
    "rollover_pct": pl.Float64,
    "atm_iv": pl.Float64,
    "iv_skew": pl.Float64,
}


def _ratio(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


def _options(df: pl.DataFrame) -> pl.DataFrame:
    opt = df.filter(pl.col("instrument").is_in(["OPTIDX", "OPTSTK"])).with_columns(
        pl.col("option_type").replace({"CA": "CE", "PA": "PE"}).alias("ot")
    )
    ce, pe = pl.col("ot") == "CE", pl.col("ot") == "PE"
    agg = opt.group_by(GROUP).agg(
        pl.col("open_interest").filter(ce).sum().alias("ce_oi"),
        pl.col("open_interest").filter(pe).sum().alias("pe_oi"),
        pl.col("oi_change").filter(ce).sum().alias("ce_oi_change"),
        pl.col("oi_change").filter(pe).sum().alias("pe_oi_change"),
        pl.col("contracts").filter(ce).sum().alias("ce_contracts"),
        pl.col("contracts").filter(pe).sum().alias("pe_contracts"),
        pl.col("strike").filter(pl.col("open_interest") > 0).n_unique().alias("strikes"),
    )
    return agg.join(_max_pain(opt), on=GROUP, how="left")


def _max_pain(opt: pl.DataFrame) -> pl.DataFrame:
    per = (
        opt.group_by(GROUP + ["strike"])
        .agg(
            pl.col("open_interest").filter(pl.col("ot") == "CE").sum().cast(pl.Float64).alias("c"),
            pl.col("open_interest").filter(pl.col("ot") == "PE").sum().cast(pl.Float64).alias("p"),
        )
        .sort(GROUP + ["strike"])
    )
    k = pl.col("strike")
    over = GROUP
    pain = (
        per.with_columns(
            pl.col("c").cum_sum().over(over).alias("cc"),
            (pl.col("c") * k).cum_sum().over(over).alias("cck"),
            pl.col("p").cum_sum(reverse=True).over(over).alias("pp"),
            (pl.col("p") * k).cum_sum(reverse=True).over(over).alias("ppk"),
            (pl.col("c") + pl.col("p")).sum().over(over).alias("tot"),
        )
        .filter(pl.col("tot") > 0)
        .with_columns(
            (k * pl.col("cc") - pl.col("cck") + pl.col("ppk") - k * pl.col("pp")).alias("pain")
        )
    )
    return (
        pain.sort(GROUP + ["pain", "strike"])
        .group_by(GROUP, maintain_order=True)
        .first()
        .select(GROUP + [pl.col("strike").alias("max_pain")])
    )


def _futures(df: pl.DataFrame) -> pl.DataFrame:
    fut = df.filter(pl.col("instrument").str.starts_with("FUT")).sort(GROUP)
    fut = fut.with_columns(
        (pl.col("settle_price") - pl.col("settle_price").shift(1).over(["symbol", "expiry"]))
        .alias("fut_settle_change")
    )
    up, down = pl.col("fut_settle_change") > 0, pl.col("fut_settle_change") < 0
    oi_up, oi_down = pl.col("oi_change") > 0, pl.col("oi_change") < 0
    buildup = (
        pl.when(up & oi_up).then(pl.lit("long_buildup"))
        .when(down & oi_up).then(pl.lit("short_buildup"))
        .when(up & oi_down).then(pl.lit("short_covering"))
        .when(down & oi_down).then(pl.lit("long_unwinding"))
        .otherwise(None)
    )
    return fut.select(
        *GROUP,
        pl.col("settle_price").alias("fut_settle"),
        "fut_settle_change",
        pl.col("open_interest").alias("fut_oi"),
        pl.col("oi_change").alias("fut_oi_change"),
        pl.col("contracts").alias("fut_contracts"),
        buildup.alias("buildup"),
        "underlying",
    )


def compute_fo_stats(df: pl.DataFrame, since=None, iv: pl.DataFrame | None = None) -> pl.DataFrame:
    """Stats for every (date, symbol, expiry) in df.

    df may start before `since` so the first kept day still gets a settle
    change; rows before `since` are dropped from the result.
    """
    if df.is_empty():
        return pl.DataFrame(schema=STATS_SCHEMA)
    kind = (
        df.group_by(["date", "symbol"])
        .agg(pl.col("instrument").first())
        .select(
            "date",
            "symbol",
            pl.when(pl.col("instrument").str.ends_with("STK"))
            .then(pl.lit("stock"))
            .otherwise(pl.lit("index"))
            .alias("kind"),
        )
    )
    keys = df.select(GROUP).unique()
    out = (
        keys.join(_options(df), on=GROUP, how="left")
        .join(_futures(df), on=GROUP, how="left")
        .join(kind, on=["date", "symbol"], how="left")
        .with_columns(
            (pl.col("expiry") == pl.col("expiry").min().over(["date", "symbol"])).alias("is_nearest"),
            _ratio("pe_oi", "ce_oi").alias("pcr_oi"),
            _ratio("pe_contracts", "ce_contracts").alias("pcr_volume"),
            (pl.col("fut_settle") - pl.col("underlying")).alias("basis"),
        )
    )
    fut_total = pl.col("fut_oi").sum().over(["date", "symbol"])
    near_fut = pl.col("expiry").filter(pl.col("fut_oi").is_not_null()).min().over(["date", "symbol"])
    near_oi = pl.col("fut_oi").filter(pl.col("expiry") == near_fut).sum().over(["date", "symbol"])
    out = out.with_columns(
        pl.when(fut_total > 0).then((fut_total - near_oi) / fut_total * 100).otherwise(None).alias("rollover_pct")
    )
    if iv is not None and not iv.is_empty():
        out = out.join(iv.select(*GROUP, "atm_iv", "iv_skew"), on=GROUP, how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("atm_iv"), pl.lit(None, pl.Float64).alias("iv_skew"))
    if since is not None:
        out = out.filter(pl.col("date") >= since)
    return out.select([pl.col(c).cast(t) for c, t in STATS_SCHEMA.items()]).sort(["symbol", "date", "expiry"])
