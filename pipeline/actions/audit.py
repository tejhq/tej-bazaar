"""Find adjusted price series that still jump, and say what each jump lands on.

A correctly adjusted series has no cliffs. NSE circuit bands cap a genuine one day move well
below 40% for any liquid name, so a larger move in `adj_close` is a corporate action the pipeline
did not apply: a split whose ratio never parsed, a split typed as something else, or an action that
is absent from the exchange feed entirely, which is the usual case for ETF unit splits.

Each jump gets a cause from every action on its date:

  * no action: nothing was announced; `actions derive` turns the clean splits among these into
    actions;
  * factor 1.0: a split, bonus or dividend gets a factor of 1.0, usually because its ratio or
    amount was not read;
  * residual: every split, bonus and dividend on the date has a factor other than 1.0 and the step
    is still there, because a factor is wrong, never reached the series, or shares the date with
    another event;
  * not adjusted: only actions the pipeline leaves unadjusted by design, such as demergers, schemes
    of arrangement and rights issues (the NSE convention).

Only the last counts as explained. This is the check that catches what the parser misses. Run it
after `actions adjust`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import polars as pl

from pipeline.actions.back_adjust import _row_to_action
from pipeline.actions.factors import compute_factor
from pipeline.actions.schema import ACTION_TYPES, CorporateAction

# Beyond this, a one day move in an adjusted series is a data error rather than a price.
DEFAULT_THRESHOLD = 0.40

# On a split or bonus ex date the step left after adjusting is held to this instead. A missed 1:2
# bonus is a 33% step and a 1:3 bonus 25%, both inside the 40% above. Stocks often close at the 20%
# upper circuit on a bonus ex date; a rise must pass 1 / (1 - 0.20) = 1.25, so those are no jump.
SPLIT_THRESHOLD = 0.20

# Consecutive rows further apart than this are not a one day move. Without it a gap in the data,
# a long suspension or a partial set of year files reads as a cliff.
MAX_GAP_DAYS = 7

NO_ACTION = "no action"
FACTOR_ONE = "factor 1.0"
RESIDUAL = "residual"
NOT_ADJUSTED = "not adjusted"
# In the order the audit reports them, the ones to fix first.
CAUSES = (FACTOR_ONE, RESIDUAL, NO_ACTION, NOT_ADJUSTED)


def _scales(kind: str) -> bool:
    """Whether `compute_factor` moves prices for this type once its details are read."""
    full = CorporateAction(
        exchange="NSE",
        symbol="",
        isin=None,
        company="",
        ex_date=date.min,
        record_date=None,
        type=kind,
        ratio_num=1,
        ratio_den=1,
        cash_amount=1.0,
        face_value_from=2.0,
        face_value_to=1.0,
    )
    return compute_factor(full, prev_close=100.0) != 1.0


# Split, bonus and dividend: the types that can move prices.
SCALED_TYPES = frozenset(t for t in ACTION_TYPES if _scales(t))


class DateAction(NamedTuple):
    """One action on a jump's date, with its factor computed from the preceding bar."""

    type: str
    factor: float
    raw_subject: str | None


@dataclass(frozen=True)
class Jump:
    symbol: str
    prev_date: date
    prev_close: float
    date: date
    close: float
    ratio: float
    cause: str
    actions: tuple[DateAction, ...] = ()
    raw_ratio: float | None = None

    @property
    def explained(self) -> bool:
        """The step is an event the pipeline leaves unadjusted by design: nothing to fix."""
        return self.cause == NOT_ADJUSTED

    @property
    def implied(self) -> str:
        """The split ratio the price move implies, as a readable string."""
        r = self.ratio if self.ratio >= 1 else 1 / self.ratio
        return f"{round(r)}:1" if abs(r - round(r)) < 0.08 else f"{r:.2f}:1"


def _outside(ratio, threshold: float):
    """A step beyond `threshold` either way; works on a float or a polars expression."""
    return (ratio < 1 - threshold) | (ratio > 1 / (1 - threshold))


def _cause(on_date: tuple[DateAction, ...]) -> str:
    if not on_date:
        return NO_ACTION
    scaled = [a for a in on_date if a.type in SCALED_TYPES]
    if any(a.factor == 1.0 for a in scaled):
        return FACTOR_ONE
    return RESIDUAL if scaled else NOT_ADJUSTED


def find_jumps(
    adjusted: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_gap_days: int = MAX_GAP_DAYS,
    split_threshold: float = SPLIT_THRESHOLD,
) -> list[Jump]:
    """Return every one day move in `adj_close` beyond `threshold`, by symbol and date.

    Rows more than `max_gap_days` apart are skipped, so a hole in the data is not read as a cliff.
    On the ex date of a split or bonus the bar is `split_threshold` when that is lower.

    `adjusted` needs columns date, symbol, adj_close. With isin, an action is matched on the ISIN
    of either bar as well as on the symbol, so one filed under a later symbol still counts; with
    close, a dividend is priced off the previous close as in `actions adjust`. `actions`, when
    given, has the columns of `ACTION_SCHEMA`. Every action on the date sets the jump's cause.
    """
    if adjusted.is_empty():
        return []
    isin = pl.col("isin") if "isin" in adjusted.columns else pl.lit(None, dtype=pl.Utf8)
    raw = pl.col("close") if "close" in adjusted.columns else pl.col("adj_close")
    df = (
        adjusted.select("date", "symbol", "adj_close", isin=isin, raw=raw)
        .filter(pl.col("adj_close") > 0)
        .sort(["symbol", "date"])
        .with_columns(
            prev_close=pl.col("adj_close").shift(1).over("symbol"),
            prev_date=pl.col("date").shift(1).over("symbol"),
            prev_raw=pl.col("raw").shift(1).over("symbol"),
            prev_isin=pl.col("isin").shift(1).over("symbol"),
        )
        .drop_nulls("prev_close")
        .with_columns(
            ratio=pl.col("adj_close") / pl.col("prev_close"),
            gap=(pl.col("date") - pl.col("prev_date")).dt.total_days(),
        )
        .filter(pl.col("gap") <= max_gap_days)
        .filter(_outside(pl.col("ratio"), min(threshold, split_threshold)))
    )
    if df.is_empty():
        return []
    rows: list[dict] = []
    on: dict[tuple, list[int]] = {}
    if actions is not None and not actions.is_empty():
        for i, row in enumerate(actions.iter_rows(named=True)):
            rows.append(row)
            on.setdefault(("symbol", row["symbol"], row["ex_date"]), []).append(i)
            if row["isin"]:
                on.setdefault(("isin", row["isin"], row["ex_date"]), []).append(i)
    out: list[Jump] = []
    for sym, d, close, prev_close, prev_date, raw_now, prev_raw, isin_now, isin_before, ratio in df.select(
        "symbol",
        "date",
        "adj_close",
        "prev_close",
        "prev_date",
        "raw",
        "prev_raw",
        "isin",
        "prev_isin",
        "ratio",
    ).iter_rows():
        keys = [("symbol", sym, d)] + [("isin", x, d) for x in (isin_now, isin_before) if x]
        hits = sorted({i for key in keys for i in on.get(key, ())})
        on_date = tuple(
            DateAction(
                rows[i]["type"],
                compute_factor(_row_to_action(rows[i]), prev_raw),
                rows[i]["raw_subject"],
            )
            for i in hits
        )
        split_date = any(a.type in ("split", "bonus") for a in on_date)
        if not _outside(ratio, min(threshold, split_threshold) if split_date else threshold):
            continue
        out.append(
            Jump(
                symbol=sym,
                prev_date=prev_date,
                prev_close=prev_close,
                date=d,
                close=close,
                ratio=ratio,
                cause=_cause(on_date),
                actions=on_date,
                raw_ratio=raw_now / prev_raw if raw_now and prev_raw else None,
            )
        )
    return out
