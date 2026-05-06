"""Trading-day calendar for Indian exchanges.

Wraps `exchange_calendars` so the rest of the pipeline can ask simple questions:
is `date` a trading day on `exchange`, and what is the previous trading day.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal

import exchange_calendars as ecals
import pandas as pd

Exchange = Literal["NSE", "BSE"]

_CALENDAR_CODE: dict[Exchange, str] = {
    "NSE": "XBOM",  # exchange_calendars groups NSE+BSE under XBOM (same trading days)
    "BSE": "XBOM",
}


@lru_cache(maxsize=4)
def _calendar(exchange: Exchange):
    code = _CALENDAR_CODE[exchange]
    return ecals.get_calendar(code)


def is_trading_day(d: date, exchange: Exchange = "NSE") -> bool:
    cal = _calendar(exchange)
    return cal.is_session(pd.Timestamp(d))


def previous_trading_day(d: date, exchange: Exchange = "NSE") -> date:
    cal = _calendar(exchange)
    ts = pd.Timestamp(d)
    prev = cal.previous_session(ts) if cal.is_session(ts) else cal.date_to_session(ts, direction="previous")
    return prev.date()


def next_trading_day(d: date, exchange: Exchange = "NSE") -> date:
    cal = _calendar(exchange)
    ts = pd.Timestamp(d)
    nxt = cal.next_session(ts) if cal.is_session(ts) else cal.date_to_session(ts, direction="next")
    return nxt.date()


def trading_days_between(start: date, end: date, exchange: Exchange = "NSE") -> list[date]:
    """Inclusive range of trading sessions in [start, end]."""
    cal = _calendar(exchange)
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [s.date() for s in sessions]
