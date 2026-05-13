"""Per-event adjustment factor computation.

Given one CorporateAction, compute the multiplier that, applied to all
prices BEFORE the action's ex_date, makes the pre-action price series
comparable to the post-action series.

Factor conventions:
    * `factor < 1`: pre-action prices are scaled down (forward split, bonus,
      dividend, etc.). This is the common case.
    * `factor > 1`: pre-action prices are scaled up (reverse split /
      consolidation).
    * `factor == 1.0`: no price impact (AGM, buyback, merger, rights v1).

Cumulative back-adjustment is the running product of these factors, applied
in reverse chronological order. See `pipeline.actions.back_adjust`.
"""

from __future__ import annotations

import math

from pipeline.actions.schema import CorporateAction


def compute_factor(action: CorporateAction, prev_close: float | None = None) -> float:
    """Return the adjustment factor for a single corporate action.

    `prev_close` is the close on the trading day immediately preceding the
    ex_date. Required for dividend factor; ignored for ratio-based actions.
    When required data is missing, returns 1.0 (action passes through
    without adjustment, raw_subject preserved for manual review).
    """
    t = action.type

    if t == "split":
        fv_from = action.face_value_from
        fv_to = action.face_value_to
        if fv_from is None or fv_to is None or fv_from <= 0:
            return 1.0
        return fv_to / fv_from

    if t == "bonus":
        n = action.ratio_num
        d = action.ratio_den
        if n is None or d is None or n + d <= 0:
            return 1.0
        return d / (n + d)

    if t == "dividend":
        cash = action.cash_amount
        if cash is None or prev_close is None or prev_close <= 0:
            return 1.0
        adj = (prev_close - cash) / prev_close
        # Defensive: dividends shouldn't exceed close price; clamp to (0, 1]
        if adj <= 0 or not math.isfinite(adj):
            return 1.0
        return adj

    # rights, buyback, agm, demerger, merger, other -> no adjustment (v1)
    return 1.0


def needs_prev_close(action: CorporateAction) -> bool:
    """Does this action's factor require a prev_close lookup?"""
    return action.type == "dividend" and action.cash_amount is not None
