"""Cumulative back-adjustment of prices using corporate-action factors.

Given a price series and a list of corporate actions (already classified +
normalized), produce a back-adjusted price series where the historical
prices are scaled so a continuous chart can be drawn across splits, bonus
issues, and dividends.

Convention: factors are applied to prices BEFORE the ex_date. The price on
ex_date itself is the post-action price and stays unchanged. So the
cumulative factor at any date D is the product of factors of all actions
whose ex_date is strictly greater than D.

The price on ex_date itself ("post-action") and the unadjusted historical
price both remain in the output. Adjusted close is added as `adj_close`,
the cumulative factor as `adj_factor_cumulative`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.actions.factors import compute_factor
from pipeline.actions.schema import CorporateAction


def compute_action_factors(
    actions: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-action adjustment factor.

    For dividend actions, looks up the close on the trading day immediately
    preceding `ex_date` from `prices`. Returns a DataFrame with columns
    (isin, ex_date, factor).

    Actions without an ISIN are dropped: back-adjustment is keyed on ISIN
    so symbol-only rows can't be joined safely against the price series.

    `prices` must have columns (isin, date, close).
    """
    _require_columns(prices, ["isin", "date", "close"])
    if actions.height == 0:
        return pl.DataFrame(
            schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64}
        )

    # Drop ISIN-less rows: no join key.
    a = actions.filter(pl.col("isin").is_not_null())
    if a.height == 0:
        return pl.DataFrame(
            schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64}
        )

    # prev_close lookup: for each (isin, ex_date), find max(close) on dates < ex_date.
    # join_asof "backward" finds the closest right row with key <= left.key. We want
    # strictly less than, so subtract 1 day from the action's ex_date as the join key.
    a_with_lookup = a.with_columns(
        _join_key=pl.col("ex_date") - pl.duration(days=1)
    ).sort(["isin", "_join_key"])
    p_sorted = prices.select(["isin", "date", "close"]).sort(["isin", "date"])

    joined = a_with_lookup.join_asof(
        p_sorted.rename({"date": "_p_date", "close": "_prev_close"}),
        left_on="_join_key",
        right_on="_p_date",
        by="isin",
        strategy="backward",
    )

    factors: list[float] = []
    isins: list[str] = []
    ex_dates: list = []
    for row in joined.iter_rows(named=True):
        ca = _row_to_action(row)
        prev_close = row.get("_prev_close")
        factors.append(compute_factor(ca, prev_close=prev_close))
        isins.append(row["isin"])
        ex_dates.append(row["ex_date"])

    return pl.DataFrame(
        {"isin": isins, "ex_date": ex_dates, "factor": factors},
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )


def back_adjust(
    prices: pl.DataFrame,
    action_factors: pl.DataFrame,
) -> pl.DataFrame:
    """Add `adj_factor_cumulative` and `adj_close` columns.

    `prices` keeps every original column, with two added:
      * adj_factor_cumulative: product of factors of all actions with
        ex_date > date for that ISIN. 1.0 when no later actions.
      * adj_close: close * adj_factor_cumulative.

    `action_factors` must have columns (isin, ex_date, factor). Use
    `compute_action_factors` to build it from raw actions.
    """
    _require_columns(prices, ["isin", "date", "close"])
    _require_columns(action_factors, ["isin", "ex_date", "factor"])

    if action_factors.height == 0:
        return prices.with_columns(
            adj_factor_cumulative=pl.lit(1.0, dtype=pl.Float64),
            adj_close=pl.col("close").cast(pl.Float64),
        )

    # Per-ISIN numpy: searchsorted into action ex_dates, look up reverse cum_prod.
    out_chunks: list[pl.DataFrame] = []
    isins_with_actions = set(action_factors["isin"].unique().to_list())

    for key, p_df in prices.partition_by("isin", as_dict=True).items():
        isin = key[0] if isinstance(key, tuple) else key
        p_df = p_df.sort("date")

        if isin not in isins_with_actions:
            out_chunks.append(p_df.with_columns(
                adj_factor_cumulative=pl.lit(1.0, dtype=pl.Float64),
                adj_close=pl.col("close").cast(pl.Float64),
            ))
            continue

        a_df = action_factors.filter(pl.col("isin") == isin).sort("ex_date")
        a_dates = a_df["ex_date"].to_numpy()
        a_factors = a_df["factor"].to_numpy()

        rev_cum = np.cumprod(a_factors[::-1])[::-1]

        p_dates = p_df["date"].to_numpy()
        # searchsorted side="right" returns idx s.t. a_dates[idx-1] <= v < a_dates[idx].
        # We want first idx where a_dates[idx] > p_date, equivalent to side="right"
        # because for ties (p_date == ex_date), action does NOT apply to that price.
        idx = np.searchsorted(a_dates, p_dates, side="right")
        cum = np.where(idx < len(a_factors), rev_cum[np.minimum(idx, len(a_factors) - 1)], 1.0)

        out_chunks.append(p_df.with_columns(
            adj_factor_cumulative=pl.Series(cum, dtype=pl.Float64),
            adj_close=pl.col("close").cast(pl.Float64) * pl.Series(cum),
        ))

    return pl.concat(out_chunks)


def _row_to_action(row: dict) -> CorporateAction:
    return CorporateAction(
        exchange=row["exchange"],
        symbol=row["symbol"],
        isin=row["isin"],
        company=row.get("company", ""),
        ex_date=row["ex_date"],
        record_date=row.get("record_date"),
        type=row["type"],
        ratio_num=row.get("ratio_num"),
        ratio_den=row.get("ratio_den"),
        cash_amount=row.get("cash_amount"),
        face_value_from=row.get("face_value_from"),
        face_value_to=row.get("face_value_to"),
        raw_subject=row.get("raw_subject", ""),
    )


def _require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
