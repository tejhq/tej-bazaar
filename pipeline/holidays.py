"""Trading-day calendar for Indian exchanges, from the dated list in `nse_calendar`.

NSE and BSE share trading days. A session is a weekday not listed as a holiday,
or a listed special session (budget Saturdays, Muhurat trading). A year with no
list counts every weekday as a session, so a run keeps fetching and a holiday
just 404s. Add each new year from NSE's circular.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pipeline.nse_calendar import HOLIDAYS, SPECIAL_SESSIONS

Exchange = Literal["NSE", "BSE"]

LAST_LISTED_YEAR = max(d.year for d in HOLIDAYS)


def is_listed(d: date) -> bool:
    """Whether `d`'s year has a holiday list, so a missing session means something."""
    return d.year <= LAST_LISTED_YEAR


def is_trading_day(d: date, exchange: Exchange = "NSE") -> bool:
    return d in SPECIAL_SESSIONS or (d.weekday() < 5 and d not in HOLIDAYS)


def previous_trading_day(d: date, exchange: Exchange = "NSE") -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d, exchange):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date, exchange: Exchange = "NSE") -> date:
    d += timedelta(days=1)
    while not is_trading_day(d, exchange):
        d += timedelta(days=1)
    return d


def trading_days_between(start: date, end: date, exchange: Exchange = "NSE") -> list[date]:
    """Inclusive range of trading sessions in [start, end]."""
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [d for d in days if is_trading_day(d, exchange)]
