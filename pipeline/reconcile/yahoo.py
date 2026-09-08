"""Fetch Yahoo Finance daily adjusted close through yfinance.

Yahoo's chart endpoint needs a session cookie and a crumb, and answers a
bare request with HTTP 429 whatever the pace. yfinance manages that
handshake, so it is the transport here; the pipeline itself never imports
it (it drags in pandas), only this module does, lazily. `Adj Close` is
Yahoo's series back-adjusted for splits and cash dividends and is the
reference our `back_adjust` output should match.

Indian symbols carry an exchange suffix: `.NS` for NSE, `.BO` for BSE.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import polars as pl

EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}

# history(yahoo_symbol, start, end_exclusive) -> pandas DataFrame with a
# datetime index and `Close` / `Adj Close` columns, as yfinance returns.
History = Callable[[str, date, date], Any]


class YahooFetchError(RuntimeError):
    pass


def _yfinance_history(yahoo_symbol: str, start: date, end_exclusive: date) -> Any:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise YahooFetchError("yfinance is not installed: pip install 'tej-bazaar[reconcile]'") from e
    try:
        return yf.Ticker(yahoo_symbol).history(
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )
    except Exception as e:  # yfinance raises a mix of its own and requests errors
        raise YahooFetchError(f"fetch {yahoo_symbol} failed: {e}") from e


def fetch_yahoo_adjusted(
    symbol: str,
    exchange: str,
    start: date,
    end: date,
    *,
    history: History | None = None,
) -> pl.DataFrame:
    """Pull daily close and adjusted close for `symbol` between `start` and `end` inclusive.

    Returns a DataFrame with columns (date, yahoo_close, yahoo_adjclose).
    Rows where Yahoo reports a null close (suspended days) are dropped.
    """
    suffix = EXCHANGE_SUFFIX.get(exchange.upper())
    if suffix is None:
        raise YahooFetchError(f"unsupported exchange: {exchange}")
    yahoo_symbol = f"{symbol}{suffix}"
    frame = (history or _yfinance_history)(yahoo_symbol, start, end + timedelta(days=1))
    return _from_pandas(frame, yahoo_symbol)


def _from_pandas(frame: Any, yahoo_symbol: str) -> pl.DataFrame:
    empty = pl.DataFrame(schema={"date": pl.Date, "yahoo_close": pl.Float64, "yahoo_adjclose": pl.Float64})
    if frame is None or len(frame) == 0:
        return empty
    cols = set(frame.columns)
    if "Close" not in cols or "Adj Close" not in cols:
        raise YahooFetchError(f"{yahoo_symbol} history lacks Close/Adj Close: {sorted(cols)}")
    idx = frame.index
    # yfinance indexes by exchange-local midnight (tz-aware); a naive index is
    # taken as already local. Either way the calendar date is the trading date.
    dates = [d.date() for d in idx.to_pydatetime()] if hasattr(idx, "to_pydatetime") else [d.date() for d in idx]
    df = pl.DataFrame(
        {
            "date": dates,
            "yahoo_close": [float(x) if x == x and x is not None else None for x in frame["Close"].tolist()],
            "yahoo_adjclose": [float(x) if x == x and x is not None else None for x in frame["Adj Close"].tolist()],
        },
        schema={"date": pl.Date, "yahoo_close": pl.Float64, "yahoo_adjclose": pl.Float64},
    )
    return df.filter(pl.col("yahoo_close").is_not_null()).sort("date")
