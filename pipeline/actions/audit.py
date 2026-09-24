"""Find adjusted price series that still jump, and say whether an action explains the jump.

A correctly adjusted series has no cliffs. NSE circuit bands cap a genuine one day move well
below 40% for any liquid name, so a larger move in `adj_close` is a corporate action the pipeline
did not apply: a split whose ratio never parsed, a split typed as something else, or an action that
is absent from the exchange feed entirely, which is the usual case for ETF unit splits.

This is the check that catches what the parser misses. Run it after `actions adjust`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

# Beyond this, a one day move in an adjusted series is a data error rather than a price.
DEFAULT_THRESHOLD = 0.40

# Consecutive rows further apart than this are not a one day move. Without it a gap in the data,
# a long suspension or a partial set of year files reads as a cliff.
MAX_GAP_DAYS = 7


@dataclass(frozen=True)
class Jump:
    symbol: str
    prev_date: date
    prev_close: float
    date: date
    close: float
    ratio: float
    explained: bool
    action_type: str | None
    raw_subject: str | None

    @property
    def implied(self) -> str:
        """The split ratio the price move implies, as a readable string."""
        r = self.ratio if self.ratio >= 1 else 1 / self.ratio
        return f"{round(r)}:1" if abs(r - round(r)) < 0.08 else f"{r:.2f}:1"


def find_jumps(
    adjusted: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_gap_days: int = MAX_GAP_DAYS,
) -> list[Jump]:
    """Return every one day move in `adj_close` beyond `threshold`, newest symbol order.

    Rows more than `max_gap_days` apart are skipped, so a hole in the data is not read as a cliff.

    `adjusted` needs columns date, symbol, adj_close. `actions`, when given, needs symbol,
    ex_date, type and raw_subject; a jump whose ex date carries an action is reported as
    explained so a real event is not mistaken for a bug.
    """
    if adjusted.is_empty():
        return []
    df = (
        adjusted.select("date", "symbol", "adj_close")
        .filter(pl.col("adj_close") > 0)
        .sort(["symbol", "date"])
        .with_columns(
            prev_close=pl.col("adj_close").shift(1).over("symbol"),
            prev_date=pl.col("date").shift(1).over("symbol"),
        )
        .drop_nulls("prev_close")
        .with_columns(
            ratio=pl.col("adj_close") / pl.col("prev_close"),
            gap=(pl.col("date") - pl.col("prev_date")).dt.total_days(),
        )
        .filter(pl.col("gap") <= max_gap_days)
        .filter((pl.col("ratio") < 1 - threshold) | (pl.col("ratio") > 1 / (1 - threshold)))
    )
    if df.is_empty():
        return []
    lookup: dict[tuple[str, date], tuple[str, str | None]] = {}
    if actions is not None and not actions.is_empty():
        for row in actions.select("symbol", "ex_date", "type", "raw_subject").iter_rows():
            lookup[(row[0], row[1])] = (row[2], row[3])
    out: list[Jump] = []
    for sym, d, close, prev_close, prev_date, ratio in df.select(
        "symbol", "date", "adj_close", "prev_close", "prev_date", "ratio"
    ).iter_rows():
        hit = lookup.get((sym, d))
        out.append(
            Jump(
                symbol=sym,
                prev_date=prev_date,
                prev_close=prev_close,
                date=d,
                close=close,
                ratio=ratio,
                explained=hit is not None,
                action_type=hit[0] if hit else None,
                raw_subject=hit[1] if hit else None,
            )
        )
    return out
