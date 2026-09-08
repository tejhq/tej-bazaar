from datetime import date, datetime

import pandas as pd
import pytest

from pipeline.reconcile.yahoo import YahooFetchError, fetch_yahoo_adjusted


def _frame(days, closes, adjcloses, tz="Asia/Kolkata"):
    idx = pd.DatetimeIndex([datetime(*d) for d in days]).tz_localize(tz)
    return pd.DataFrame({"Open": closes, "Close": closes, "Adj Close": adjcloses}, index=idx)


def _history_returning(frame, calls):
    def history(sym, start, end_exclusive):
        calls.append((sym, start, end_exclusive))
        return frame
    return history


def test_fetch_yahoo_adjusted_basic_nse():
    calls = []
    df = fetch_yahoo_adjusted(
        "NESTLEIND", "NSE", date(2024, 1, 2), date(2024, 1, 3),
        history=_history_returning(_frame([(2024, 1, 2), (2024, 1, 3)], [27000.0, 27100.0], [2700.0, 2710.0]), calls),
    )
    assert calls == [("NESTLEIND.NS", date(2024, 1, 2), date(2024, 1, 4))]  # end is exclusive
    assert df["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert df["yahoo_close"].to_list() == [27000.0, 27100.0]
    assert df["yahoo_adjclose"].to_list() == [2700.0, 2710.0]


def test_fetch_yahoo_bse_suffix():
    calls = []
    fetch_yahoo_adjusted("RELIANCE", "bse", date(2024, 1, 2), date(2024, 1, 2),
                         history=_history_returning(_frame([(2024, 1, 2)], [1.0], [1.0]), calls))
    assert calls[0][0] == "RELIANCE.BO"


def test_fetch_yahoo_drops_null_close_rows():
    df = fetch_yahoo_adjusted(
        "RELIANCE", "NSE", date(2024, 1, 2), date(2024, 1, 3),
        history=_history_returning(_frame([(2024, 1, 2), (2024, 1, 3)], [float("nan"), 2900.0], [float("nan"), 2890.0]), []),
    )
    assert df.height == 1
    assert df["date"][0] == date(2024, 1, 3)


def test_fetch_yahoo_empty_returns_empty_with_schema():
    df = fetch_yahoo_adjusted("X", "NSE", date(2024, 1, 2), date(2024, 1, 3),
                              history=lambda *_: pd.DataFrame())
    assert df.height == 0
    assert df.columns == ["date", "yahoo_close", "yahoo_adjclose"]


def test_fetch_yahoo_missing_columns_raises():
    bad = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([datetime(2024, 1, 2)]))
    with pytest.raises(YahooFetchError, match="Adj Close"):
        fetch_yahoo_adjusted("X", "NSE", date(2024, 1, 2), date(2024, 1, 2), history=lambda *_: bad)


def test_fetch_yahoo_unsupported_exchange_raises():
    with pytest.raises(YahooFetchError, match="unsupported exchange"):
        fetch_yahoo_adjusted("X", "MCX", date(2024, 1, 2), date(2024, 1, 2), history=lambda *_: pd.DataFrame())


def test_fetch_yahoo_transport_error_wrapped():
    def boom(*_):
        raise YahooFetchError("fetch X.NS failed: HTTP 429")
    with pytest.raises(YahooFetchError, match="429"):
        fetch_yahoo_adjusted("X", "NSE", date(2024, 1, 2), date(2024, 1, 2), history=boom)
