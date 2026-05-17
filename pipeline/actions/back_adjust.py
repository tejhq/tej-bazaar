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
from pipeline.symbol_history import build_symbol_history


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


def resolve_isin_via_symbol_history(
    actions: pl.DataFrame, prices: pl.DataFrame
) -> pl.DataFrame:
    """Override `actions.isin` with the ISIN that `actions.symbol` traded
    under on `actions.ex_date`, using `prices` as the authoritative ledger.

    Why: NSE's corporate-actions API sometimes tags actions to a stale
    ISIN that no longer reflects the trading instrument. The clearest
    case is HDFC Bank: post-merger (July 2023) the ticker `HDFCBANK`
    trades under `INE040A01034`, yet NSE still reports its 2025 1:1 bonus
    against the legacy HDFC Ltd ISIN `INE040A01018`. Joining actions to
    prices by that stale ISIN drops the bonus on the floor and leaves
    pre-event prices ~2x too high.

    For each action, we look up the symbol in symbol_history at the action's
    ex_date and rewrite the ISIN to whatever the symbol actually traded
    under that day. If lookup fails (delisted at ex_date, or symbol absent
    from price history), the original ISIN is kept.
    """
    _require_columns(actions, ["symbol", "ex_date", "isin"])
    _require_columns(prices, ["symbol", "isin", "date"])

    if actions.height == 0:
        return actions

    exchange = (
        actions["exchange"][0] if "exchange" in actions.columns and actions.height > 0
        else "?"
    )
    history = build_symbol_history(prices, exchange)

    # For each action, attach the factor to the ISIN that the symbol's
    # active interval started on STRICTLY BEFORE ex_date. Two cases:
    #  - Dividend / bonus on a stable ISIN: the active interval starts long
    #    before ex_date; we pick that interval's ISIN (== current ISIN).
    #  - Split with face-value change: the ISIN flips on ex_date itself, so
    #    the new interval has valid_from == ex_date. The `<` filter skips
    #    that brand-new interval and reaches back to the prior one. The
    #    factor then attaches to the OLD ISIN, which carries the long
    #    pre-event price history.
    new_isins: list[str | None] = []
    for row in actions.iter_rows(named=True):
        sym = row["symbol"]
        ex = row["ex_date"]
        sym_intervals = history.filter(
            (pl.col("symbol") == sym) & (pl.col("valid_from") < ex)
        ).sort("valid_from", descending=True)
        if sym_intervals.height > 0:
            new_isins.append(sym_intervals["isin"][0])
        else:
            # No prior interval (action precedes our price coverage). Keep
            # the original ISIN; if it doesn't match anything, the join in
            # compute_action_factors will simply drop the row.
            new_isins.append(row.get("isin"))

    return actions.with_columns(isin=pl.Series(new_isins, dtype=pl.Utf8))


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
