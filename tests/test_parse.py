from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pipeline.parse import parse_bhavcopy, parse_nse

FIXTURE = Path(__file__).parent / "fixtures" / "nse_bhavcopy_20250430_sample.csv"
BSE_FIXTURE = Path(__file__).parent / "fixtures" / "bse_bhavcopy_20250430_sample.csv"


def test_parse_nse_schema():
    df = parse_nse(FIXTURE)
    expected_cols = [
        "date", "symbol", "series", "isin", "name",
        "open", "high", "low", "close", "last", "prev_close",
        "volume", "turnover", "trades",
    ]
    assert df.columns == expected_cols


def test_parse_nse_dtypes():
    df = parse_nse(FIXTURE)
    assert df.schema["date"] == pl.Date
    assert df.schema["symbol"] == pl.Utf8
    assert df.schema["open"] == pl.Float64
    assert df.schema["close"] == pl.Float64
    assert df.schema["volume"] == pl.Int64
    assert df.schema["trades"] == pl.Int64


def test_parse_nse_row_count():
    df = parse_nse(FIXTURE)
    assert df.height == 6  # 5 EQ + 1 GB in fixture


def test_parse_nse_known_value_reliance():
    df = parse_nse(FIXTURE)
    row = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert row["date"] == date(2025, 4, 30)
    assert row["series"] == "EQ"
    assert row["isin"] == "INE002A01018"
    assert row["open"] == 1402.00
    assert row["high"] == 1412.40
    assert row["low"] == 1369.00
    assert row["close"] == 1405.00
    assert row["volume"] == 25480745
    assert row["turnover"] == pytest.approx(35797220282.00)
    assert row["trades"] == 598769


def test_parse_nse_includes_non_equity_series():
    # Parser does not filter; SGBJUN28 (gold bond, series=GB) must be present
    df = parse_nse(FIXTURE)
    assert "GB" in df["series"].to_list()
    assert "SGBJUN28" in df["symbol"].to_list()


def test_parse_nse_missing_columns_raises(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="schema not recognised"):
        parse_nse(bad)


def test_parse_bse_schema_matches_nse():
    df = parse_bhavcopy(BSE_FIXTURE)
    expected_cols = [
        "date", "symbol", "series", "isin", "name",
        "open", "high", "low", "close", "last", "prev_close",
        "volume", "turnover", "trades",
    ]
    assert df.columns == expected_cols
    assert df.schema["date"] == pl.Date
    assert df.schema["volume"] == pl.Int64


def test_parse_bse_known_value_reliance():
    df = parse_bhavcopy(BSE_FIXTURE)
    row = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert row["date"] == date(2025, 4, 30)
    assert row["series"] == "A"
    assert row["isin"] == "INE002A01018"
    assert row["open"] == 1404.90
    assert row["close"] == 1408.35
    assert row["volume"] == 3375628


def test_parse_bse_includes_all_series():
    df = parse_bhavcopy(BSE_FIXTURE)
    series = set(df["series"].to_list())
    assert {"A", "T", "X", "Z"}.issubset(series)


def test_parse_legacy_nse_pre2012_no_isin_no_trades(tmp_path: Path):
    # 2010-era NSE bhavcopy: 11 cols, no ISIN, no TOTALTRADES, trailing comma.
    csv = tmp_path / "cm04JAN2010bhav.csv"
    csv.write_text(
        "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
        "20MICRONS,EQ,46,47.9,45.7,47.55,47.5,46.05,36282,1719201.35,4-JAN-2010,\n"
        "RELIANCE,EQ,1100,1120,1090,1110,1108,1095,5000000,5550000000,4-JAN-2010,\n"
    )
    df = parse_bhavcopy(csv)
    assert df.columns == [
        "date", "symbol", "series", "isin", "name",
        "open", "high", "low", "close", "last", "prev_close",
        "volume", "turnover", "trades",
    ]
    assert df.schema["date"] == pl.Date
    assert df["date"].to_list() == [date(2010, 1, 4), date(2010, 1, 4)]
    assert df["isin"].null_count() == 2
    assert df["trades"].null_count() == 2
    assert (df["name"] == "").all()
    rel = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert rel["close"] == 1110.0
    assert rel["volume"] == 5_000_000


def test_parse_legacy_nse_post2012_has_isin_and_trades(tmp_path: Path):
    csv = tmp_path / "cm01JUN2015bhav.csv"
    csv.write_text(
        "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
        "RELIANCE,EQ,900,905,898,902,901.5,899,1234567,1112000000,1-JUN-2015,15000,INE002A01018,\n"
    )
    df = parse_bhavcopy(csv)
    assert df["isin"].to_list() == ["INE002A01018"]
    assert df["trades"].to_list() == [15000]
    assert df["date"].to_list() == [date(2015, 6, 1)]


def test_parse_legacy_nse_handles_2digit_year_and_mixed_case(tmp_path: Path):
    # Regression: NSE 2020-07-13 raw bhavcopy shipped dates as `13-Jul-20`
    # instead of the usual `13-JUL-2020`. Parser must accept both.
    csv = tmp_path / "cm13JUL2020bhav.csv"
    csv.write_text(
        "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
        "20MICRONS,EQ,32.85,33.85,31.85,33.45,33.85,32.3,187303,6187285.7,13-Jul-20,1382,INE144J01027,\n"
    )
    df = parse_bhavcopy(csv)
    assert df["date"].to_list() == [date(2020, 7, 13)]


def test_parse_legacy_nse_raises_on_unparseable_date(tmp_path: Path):
    csv = tmp_path / "cm01JUN2015bhav.csv"
    csv.write_text(
        "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
        "RELIANCE,EQ,900,905,898,902,901.5,899,1234567,1112000000,01/06/2015,15000,INE002A01018,\n"
    )
    with pytest.raises(ValueError, match="unparseable date"):
        parse_bhavcopy(csv)
