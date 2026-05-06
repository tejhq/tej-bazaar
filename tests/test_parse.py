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
    with pytest.raises(ValueError, match="missing columns"):
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
