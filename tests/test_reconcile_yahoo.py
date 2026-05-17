from datetime import date

import pytest
import responses

from pipeline.reconcile.yahoo import YahooFetchError, fetch_yahoo_adjusted


def _chart_payload(timestamps, closes, adjcloses, *, error=None):
    if error is not None:
        return {"chart": {"result": None, "error": error}}
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{"close": closes}],
                        "adjclose": [{"adjclose": adjcloses}],
                    },
                }
            ],
            "error": None,
        }
    }


@responses.activate
def test_fetch_yahoo_adjusted_basic_nse():
    # 2024-01-02 IST 09:15 (market open) ~= 2024-01-02 03:45 UTC = ts 1704166500.
    # Yahoo returns midnight US/Eastern timestamps; we just need any ts that
    # converts to the right IST date. Pick UTC noon for clarity.
    ts = [
        int(__import__("datetime").datetime(2024, 1, 2, 12, tzinfo=__import__("datetime").timezone.utc).timestamp()),
        int(__import__("datetime").datetime(2024, 1, 3, 12, tzinfo=__import__("datetime").timezone.utc).timestamp()),
    ]
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/NESTLEIND.NS",
        json=_chart_payload(ts, [27000.0, 27100.0], [2700.0, 2710.0]),
        status=200,
    )
    df = fetch_yahoo_adjusted("NESTLEIND", "NSE", date(2024, 1, 2), date(2024, 1, 3))
    assert df.height == 2
    assert df["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert df["yahoo_close"].to_list() == [27000.0, 27100.0]
    assert df["yahoo_adjclose"].to_list() == [2700.0, 2710.0]


@responses.activate
def test_fetch_yahoo_drops_null_close_rows():
    # Suspended day: close=None should be filtered out.
    ts = [
        int(__import__("datetime").datetime(2024, 1, 2, 12, tzinfo=__import__("datetime").timezone.utc).timestamp()),
        int(__import__("datetime").datetime(2024, 1, 3, 12, tzinfo=__import__("datetime").timezone.utc).timestamp()),
    ]
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/RELIANCE.NS",
        json=_chart_payload(ts, [None, 2900.0], [None, 2890.0]),
        status=200,
    )
    df = fetch_yahoo_adjusted("RELIANCE", "NSE", date(2024, 1, 2), date(2024, 1, 3))
    assert df.height == 1
    assert df["date"].to_list() == [date(2024, 1, 3)]


@responses.activate
def test_fetch_yahoo_empty_timestamps_returns_empty_with_schema():
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/UNKNOWN.NS",
        json=_chart_payload([], [], []),
        status=200,
    )
    df = fetch_yahoo_adjusted("UNKNOWN", "NSE", date(2024, 1, 2), date(2024, 1, 3))
    assert df.height == 0
    assert df.columns == ["date", "yahoo_close", "yahoo_adjclose"]


@responses.activate
def test_fetch_yahoo_error_payload_raises():
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/BADSYM.NS",
        json=_chart_payload(
            None, None, None,
            error={"code": "Not Found", "description": "No data found, symbol may be delisted"},
        ),
        status=200,
    )
    with pytest.raises(YahooFetchError, match="returned error"):
        fetch_yahoo_adjusted("BADSYM", "NSE", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_fetch_yahoo_429_then_recovers(monkeypatch):
    # Throttle backoff sleep (10s, 20s, ...) is too slow for tests.
    monkeypatch.setattr("pipeline.reconcile.yahoo.time.sleep", lambda _: None)
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/THR.NS",
        json={"err": "throttled"},
        status=429,
    )
    ts = [int(__import__("datetime").datetime(2024, 1, 2, 12, tzinfo=__import__("datetime").timezone.utc).timestamp())]
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/THR.NS",
        json=_chart_payload(ts, [100.0], [100.0]),
        status=200,
    )
    df = fetch_yahoo_adjusted("THR", "NSE", date(2024, 1, 2), date(2024, 1, 3),
                              max_retries=3, backoff=1.0)
    assert df.height == 1


@responses.activate
def test_fetch_yahoo_429_exhausted_raises_throttled(monkeypatch):
    monkeypatch.setattr("pipeline.reconcile.yahoo.time.sleep", lambda _: None)
    for _ in range(3):
        responses.add(
            responses.GET,
            "https://query2.finance.yahoo.com/v8/finance/chart/THR.NS",
            json={"err": "throttled"},
            status=429,
        )
    with pytest.raises(YahooFetchError, match="throttled"):
        fetch_yahoo_adjusted("THR", "NSE", date(2024, 1, 2), date(2024, 1, 3),
                             max_retries=3, backoff=1.0)


@responses.activate
def test_fetch_yahoo_500_retries_then_raises():
    for _ in range(3):
        responses.add(
            responses.GET,
            "https://query2.finance.yahoo.com/v8/finance/chart/FOO.NS",
            json={"err": "server"},
            status=500,
        )
    with pytest.raises(YahooFetchError, match="HTTP 500"):
        fetch_yahoo_adjusted("FOO", "NSE", date(2024, 1, 2), date(2024, 1, 3),
                             max_retries=3, backoff=1.0)


@responses.activate
def test_fetch_yahoo_500_then_recovers():
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/FOO.NS",
        json={"err": "server"},
        status=500,
    )
    ts = [int(__import__("datetime").datetime(2024, 1, 2, 12, tzinfo=__import__("datetime").timezone.utc).timestamp())]
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/FOO.NS",
        json=_chart_payload(ts, [100.0], [100.0]),
        status=200,
    )
    df = fetch_yahoo_adjusted("FOO", "NSE", date(2024, 1, 2), date(2024, 1, 3),
                              max_retries=3, backoff=1.0)
    assert df.height == 1


def test_fetch_yahoo_unsupported_exchange_raises():
    with pytest.raises(YahooFetchError, match="unsupported exchange"):
        fetch_yahoo_adjusted("FOO", "XYZ", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_fetch_yahoo_array_length_mismatch_raises():
    ts = [1, 2, 3]
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/FOO.NS",
        json=_chart_payload(ts, [100.0, 101.0], [100.0, 101.0]),  # only 2 closes
        status=200,
    )
    with pytest.raises(YahooFetchError, match="length mismatch"):
        fetch_yahoo_adjusted("FOO", "NSE", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_fetch_yahoo_bse_uses_bo_suffix():
    responses.add(
        responses.GET,
        "https://query2.finance.yahoo.com/v8/finance/chart/RELIANCE.BO",
        json=_chart_payload([], [], []),
        status=200,
    )
    df = fetch_yahoo_adjusted("RELIANCE", "BSE", date(2024, 1, 2), date(2024, 1, 3))
    assert df.height == 0
