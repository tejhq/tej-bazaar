"""Compare tej-bazaar adjusted prices against an external reference series.

Given our `(date, adj_close)` series for one symbol and the reference
`(date, ref_adjclose)` series, we inner-join on date and compute the
relative difference per row. The dataset's claim is that, for the bulk of
rows, the gap is below a small tolerance (default 0.5%).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SymbolReconcileStats:
    symbol: str
    rows_compared: int
    pct_within_tol: float
    max_abs_diff_pct: float
    mean_abs_diff_pct: float


@dataclass(frozen=True)
class ReconcileResult:
    per_symbol: list[SymbolReconcileStats]
    overall_pct_within_tol: float
    overall_rows: int


def reconcile_symbol(
    symbol: str,
    ours: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    tolerance_pct: float = 0.5,
    ours_col: str = "adj_close",
    ref_col: str = "yahoo_adjclose",
) -> SymbolReconcileStats:
    """Inner-join ours+reference on date, compute relative diff stats.

    `tolerance_pct` is in percent units (0.5 == 0.5%). Returns
    SymbolReconcileStats. If no overlapping dates exist, returns zero rows.
    """
    if "date" not in ours.columns or ours_col not in ours.columns:
        raise ValueError(f"ours missing required cols: date, {ours_col}")
    if "date" not in reference.columns or ref_col not in reference.columns:
        raise ValueError(f"reference missing required cols: date, {ref_col}")

    joined = (
        ours.select(["date", ours_col])
        .join(reference.select(["date", ref_col]), on="date", how="inner")
        .filter(pl.col(ours_col).is_not_null() & pl.col(ref_col).is_not_null())
        .filter(pl.col(ref_col) > 0)
    )

    if joined.height == 0:
        return SymbolReconcileStats(
            symbol=symbol,
            rows_compared=0,
            pct_within_tol=0.0,
            max_abs_diff_pct=0.0,
            mean_abs_diff_pct=0.0,
        )

    diffed = joined.with_columns(
        diff_pct=(pl.col(ours_col) - pl.col(ref_col)).abs() / pl.col(ref_col) * 100.0
    )
    n = diffed.height
    within = diffed.filter(pl.col("diff_pct") <= tolerance_pct).height
    max_diff = float(diffed["diff_pct"].max() or 0.0)
    mean_diff = float(diffed["diff_pct"].mean() or 0.0)

    return SymbolReconcileStats(
        symbol=symbol,
        rows_compared=n,
        pct_within_tol=100.0 * within / n,
        max_abs_diff_pct=max_diff,
        mean_abs_diff_pct=mean_diff,
    )


def summarize(per_symbol: list[SymbolReconcileStats]) -> ReconcileResult:
    """Roll per-symbol stats into an overall pass-rate weighted by row count."""
    total = sum(s.rows_compared for s in per_symbol)
    if total == 0:
        return ReconcileResult(per_symbol=per_symbol, overall_pct_within_tol=0.0, overall_rows=0)
    weighted_within = sum(s.rows_compared * s.pct_within_tol / 100.0 for s in per_symbol)
    return ReconcileResult(
        per_symbol=per_symbol,
        overall_pct_within_tol=100.0 * weighted_within / total,
        overall_rows=total,
    )
