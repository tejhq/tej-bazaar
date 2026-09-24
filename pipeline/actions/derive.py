"""Derive split actions for cliffs the exchange feed never announced.

Some instruments split their units and no corporate action row ever appears. ETF unit splits are
the common case: NIFTYBEES, GOLDBEES, BANKBEES and about fifty others show a clean 10:1 or 100:1
step in an otherwise continuous series with nothing to explain it. There is nothing to parse, so
the only evidence is the price itself.

This module turns that evidence into an action, conservatively. A cliff becomes a derived split
only when every one of these holds:

  * no corporate action of any kind sits within `NEARBY_DAYS` of the step, so a demerger, rights
    issue or buyback that the pipeline deliberately does not scale is never overridden;
  * the implied ratio is within `RATIO_TOLERANCE` of a ratio issuers actually use;
  * the price on both sides is sane and the two bars are consecutive sessions;
  * the instrument is not a rights entitlement, which halves and expires as a matter of course;
  * both prices are above a few rupees, since at 5 paise one tick is a 100% move;
  * the symbol does not produce a stream of these, which means an oscillating series rather than a
    step.

The residual risk is a genuine fall that lands exactly on a standard ratio with no announcement
anywhere near it. Circuit bands make that close to impossible for a listed equity, and every
derived row is printed, marked in `raw_subject` and visible to `actions audit`, so a wrong call is
findable and reversible.

Anything else is left alone and stays visible in `actions audit`. Derived rows carry their
provenance in `raw_subject` so they can always be told apart from exchange announcements.
"""

from __future__ import annotations


import polars as pl

from pipeline.actions.audit import Jump
from pipeline.actions.schema import ACTION_SCHEMA, CorporateAction

# A corporate action this close to the step means the step is already accounted for, or is an
# event the pipeline scales by design. Either way, hands off.
NEARBY_DAYS = 5

# Ratios issuers actually use. A crash is not a round number; a split almost always is.
KNOWN_RATIOS = (2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)

# How close the implied ratio must sit to one of those. Small ratios need a tight band, because a
# genuine 50% fall is possible and must not be read as a 2:1 split. At 10:1 and beyond nothing else
# lands near the number, and the step is often accompanied by a few percent of real price move, so
# a wider band is both safe and necessary.
RATIO_TOLERANCE = 0.02
WIDE_RATIO_FROM = 10.0
WIDE_RATIO_TOLERANCE = 0.06

DERIVED_NOTE = "Derived from an unexplained price step; no exchange action was published"

# Rights entitlements trade under a -RE suffix, carry no circuit band and expire worthless, so
# halvings are ordinary for them and must never be read as splits.
ENTITLEMENT_SUFFIXES = ("-RE", "-RE1", "-RE2", "-RE3", "-RE4", "-RE5", "-E1", "-E2")

# Below this price the tick size itself is a large fraction of the price: a stock at 5 paise moves
# 100% on one tick and oscillates between ratios that look exactly like splits.
MIN_PRICE = 5.0

# An instrument splits a handful of times in a decade. More than this from one symbol is a series
# that oscillates rather than steps, so none of them are trusted.
MAX_PER_SYMBOL = 3


def _clean_ratio(ratio: float) -> float | None:
    """Return the standard split ratio this price step implies, or None when it is not one."""
    r = ratio if ratio >= 1 else 1 / ratio
    for known in KNOWN_RATIOS:
        tol = WIDE_RATIO_TOLERANCE if known >= WIDE_RATIO_FROM else RATIO_TOLERANCE
        if abs(r - known) / known <= tol:
            return known
    return None


def derive_splits(
    jumps: list[Jump],
    actions: pl.DataFrame | None = None,
    nearby_days: int = NEARBY_DAYS,
    max_per_symbol: int = MAX_PER_SYMBOL,
) -> list[CorporateAction]:
    """Turn unexplained cliffs into split actions, skipping anything doubtful.

    `jumps` comes from `find_jumps`; `actions` is the full action set, used to check the
    neighbourhood of each step rather than only its exact date.
    """
    nearby: dict[str, list] = {}
    if actions is not None and not actions.is_empty():
        for sym, ex in actions.select("symbol", "ex_date").iter_rows():
            nearby.setdefault(sym, []).append(ex)
    out: list[CorporateAction] = []
    for j in jumps:
        if j.explained or j.prev_close <= 0 or j.close <= 0:
            continue
        if j.symbol.upper().endswith(ENTITLEMENT_SUFFIXES):
            continue
        if min(j.prev_close, j.close) < MIN_PRICE:
            continue
        if any(abs((ex - j.date).days) <= nearby_days for ex in nearby.get(j.symbol, ())):
            continue
        ratio = _clean_ratio(j.ratio)
        if ratio is None:
            continue
        # A forward split multiplies the unit count, so the face value falls by the same ratio.
        fv_from, fv_to = (ratio, 1.0) if j.ratio < 1 else (1.0, ratio)
        out.append(
            CorporateAction(
                exchange="NSE",
                symbol=j.symbol,
                isin=None,
                company=j.symbol,
                ex_date=j.date,
                record_date=None,
                type="split",
                ratio_num=None,
                ratio_den=None,
                cash_amount=None,
                face_value_from=fv_from,
                face_value_to=fv_to,
                raw_subject=f"{DERIVED_NOTE}: {j.prev_close:,.2f} on {j.prev_date} to {j.close:,.2f} on {j.date}",
            )
        )
    # A symbol that produces many steps is oscillating, not splitting; drop all of its rows.
    counts: dict[str, int] = {}
    for a in out:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1
    return [a for a in out if counts[a.symbol] <= max_per_symbol]


def merge_derived(existing: pl.DataFrame, derived: list[CorporateAction]) -> pl.DataFrame:
    """Add derived rows to an action set, replacing any earlier derived row for the same step."""
    if not derived:
        return existing
    rows = pl.DataFrame(
        [
            {
                "exchange": a.exchange,
                "symbol": a.symbol,
                "isin": a.isin,
                "company": a.company,
                "ex_date": a.ex_date,
                "record_date": a.record_date,
                "type": a.type,
                "ratio_num": a.ratio_num,
                "ratio_den": a.ratio_den,
                "cash_amount": a.cash_amount,
                "face_value_from": a.face_value_from,
                "face_value_to": a.face_value_to,
                "raw_subject": a.raw_subject,
            }
            for a in derived
        ],
        schema=ACTION_SCHEMA,
    )
    keep = existing.filter(
        ~(
            pl.col("raw_subject").str.starts_with(DERIVED_NOTE)
            & pl.struct("symbol", "ex_date").is_in(rows.select("symbol", "ex_date").to_struct())
        )
    )
    return pl.concat([keep, rows]).sort(["symbol", "ex_date"])


def is_derived(raw_subject: str | None) -> bool:
    return bool(raw_subject) and raw_subject.startswith(DERIVED_NOTE)


__all__ = ["derive_splits", "merge_derived", "is_derived", "DERIVED_NOTE"]
