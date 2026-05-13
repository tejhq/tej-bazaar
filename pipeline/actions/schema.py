"""Schema for corporate actions.

One row = one corporate action event. Free-text fields from the source
(`subject` on NSE, `Purpose` on BSE) are kept verbatim in `raw_subject` so
mis-classifications can be audited and re-parsed without re-fetching.

Adjustment math lives downstream. This module only normalizes structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl

Exchange = Literal["NSE", "BSE"]

ActionType = Literal[
    "dividend",  # interim, final, special, distribution: same math
    "split",     # face-value sub-division (price / ratio)
    "bonus",     # free shares (ratio_num : ratio_den)
    "rights",    # discounted issue to existing holders
    "buyback",   # company repurchase, no price adjustment
    "demerger",  # incl. spin-off, needs manual override for ratio
    "merger",    # incl. amalgamation
    "agm",       # AGM/EGM, no price impact, kept for audit
    "other",
]

ACTION_TYPES: tuple[ActionType, ...] = (
    "dividend", "split", "bonus", "rights", "buyback",
    "demerger", "merger", "agm", "other",
)


@dataclass(frozen=True)
class CorporateAction:
    exchange: Exchange
    symbol: str
    isin: str | None
    company: str
    ex_date: date
    record_date: date | None
    type: ActionType
    ratio_num: int | None = None
    ratio_den: int | None = None
    cash_amount: float | None = None
    face_value_from: float | None = None
    face_value_to: float | None = None
    raw_subject: str = ""


ACTION_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "isin": pl.Utf8,
    "company": pl.Utf8,
    "ex_date": pl.Date,
    "record_date": pl.Date,
    "type": pl.Utf8,
    "ratio_num": pl.Int64,
    "ratio_den": pl.Int64,
    "cash_amount": pl.Float64,
    "face_value_from": pl.Float64,
    "face_value_to": pl.Float64,
    "raw_subject": pl.Utf8,
}


def to_polars(actions: list[CorporateAction]) -> pl.DataFrame:
    rows = [
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
        for a in actions
    ]
    return pl.DataFrame(rows, schema=ACTION_SCHEMA)
