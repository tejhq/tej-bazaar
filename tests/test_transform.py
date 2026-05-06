from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pipeline.parse import parse_nse
from pipeline.transform import TransformError, transform

FIXTURE = Path(__file__).parent / "fixtures" / "nse_bhavcopy_20250430_sample.csv"


def _row(
    *,
    d=date(2025, 4, 30),
    symbol="ACME",
    series="EQ",
    isin="INE000A00001",
    name="ACME LTD",
    open_=100.0,
    high=110.0,
    low=95.0,
    close=105.0,
    last=105.0,
    prev_close=99.0,
    volume=1000,
    turnover=105000.0,
    trades=10,
) -> dict:
    return {
        "date": d, "symbol": symbol, "series": series, "isin": isin, "name": name,
        "open": open_, "high": high, "low": low, "close": close, "last": last,
        "prev_close": prev_close, "volume": volume, "turnover": turnover, "trades": trades,
    }


def _df(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "date": pl.Date, "symbol": pl.Utf8, "series": pl.Utf8, "isin": pl.Utf8,
        "name": pl.Utf8, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "last": pl.Float64, "prev_close": pl.Float64,
        "volume": pl.Int64, "turnover": pl.Float64, "trades": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def test_transform_filters_to_equity_series():
    df = parse_nse(FIXTURE)
    out = transform(df)
    # Fixture has 5 EQ + 1 GB; GB must be dropped
    assert "GB" not in out["series"].to_list()
    assert set(out["series"].to_list()) == {"EQ"}
    assert out.height == 5


def test_transform_dedupes_on_date_symbol():
    df = _df([
        _row(symbol="ACME", close=100.0),
        _row(symbol="ACME", close=104.0),  # duplicate; should be dropped
        _row(symbol="BETA", open_=200.0, high=210.0, low=195.0, close=205.0),
    ])
    out = transform(df)
    assert out.height == 2
    acme = out.filter(pl.col("symbol") == "ACME").row(0, named=True)
    assert acme["close"] == 100.0  # first wins


def test_transform_sorts_by_date_then_symbol():
    df = _df([
        _row(symbol="ZZZ"),
        _row(symbol="AAA"),
        _row(symbol="MMM"),
    ])
    out = transform(df)
    assert out["symbol"].to_list() == ["AAA", "MMM", "ZZZ"]


def test_transform_drops_zero_volume():
    df = _df([
        _row(symbol="LIVE", volume=100),
        _row(symbol="DEAD", volume=0),
    ])
    out = transform(df)
    assert out["symbol"].to_list() == ["LIVE"]


def test_transform_keeps_zero_volume_when_disabled():
    df = _df([
        _row(symbol="LIVE", volume=100),
        _row(symbol="DEAD", volume=0),
    ])
    out = transform(df, drop_zero_volume=False)
    assert set(out["symbol"].to_list()) == {"LIVE", "DEAD"}


def test_transform_drops_null_ohlc():
    df = _df([
        _row(symbol="OK"),
        _row(symbol="BAD", close=None),
    ])
    out = transform(df)
    assert out["symbol"].to_list() == ["OK"]


def test_transform_drops_invalid_prices():
    df = _df([
        _row(symbol="OK", low=95, high=110, open_=100, close=105),
        _row(symbol="LOWGTHIGH", low=200, high=100, open_=150, close=150),
        _row(symbol="OPENBELOWLOW", low=100, high=110, open_=50, close=105),
        _row(symbol="ZEROOPEN", low=0, high=110, open_=0, close=105),
    ])
    out = transform(df)
    assert out["symbol"].to_list() == ["OK"]


def test_transform_custom_series_filter():
    df = _df([
        _row(symbol="A", series="EQ"),
        _row(symbol="B", series="GB"),
        _row(symbol="C", series="SM"),
    ])
    out = transform(df, series=["EQ", "SM"])
    assert set(out["symbol"].to_list()) == {"A", "C"}


def test_transform_empty_input():
    df = _df([])
    out = transform(df)
    assert out.height == 0


def test_transform_missing_columns_raises():
    df = pl.DataFrame({"date": [date(2025, 4, 30)], "symbol": ["X"]})
    with pytest.raises(TransformError, match="missing required columns"):
        transform(df)


def test_transform_bse_default_series():
    # BSE defaults: A, B, T. Other series (X, Z) must drop.
    df = _df([
        _row(symbol="LARGECAP", series="A"),
        _row(symbol="MIDCAP", series="B"),
        _row(symbol="T2T", series="T"),
        _row(symbol="SME", series="X"),
        _row(symbol="PENALTY", series="Z"),
    ])
    out = transform(df, exchange="BSE")
    assert set(out["symbol"].to_list()) == {"LARGECAP", "MIDCAP", "T2T"}


def test_transform_nse_default_series_via_exchange():
    df = _df([
        _row(symbol="EQ1", series="EQ"),
        _row(symbol="A1", series="A"),  # BSE series — should drop on NSE
    ])
    out = transform(df, exchange="NSE")
    assert out["symbol"].to_list() == ["EQ1"]


def test_transform_bse_fixture():
    bse_fixture = Path(__file__).parent / "fixtures" / "bse_bhavcopy_20250430_sample.csv"
    from pipeline.parse import parse_bhavcopy
    df = parse_bhavcopy(bse_fixture)
    out = transform(df, exchange="BSE")
    # 4 A-series + 1 T-series; X and Z drop
    assert out.height == 5
    assert set(out["series"].to_list()) == {"A", "T"}
    assert "DEFAULTER" not in out["symbol"].to_list()
    assert "SAMPLESME" not in out["symbol"].to_list()
