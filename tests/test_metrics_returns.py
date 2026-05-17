"""Tests for pipeline.metrics.returns."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from pipeline.metrics.returns import RETURNS_SCHEMA, compute_returns


def _adjusted(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.Utf8,
            "date": pl.Date,
            "symbol": pl.Utf8,
            "adj_close": pl.Float64,
        },
    )


def test_returns_schema_matches_constant():
    df = compute_returns(_adjusted([]))
    assert df.schema == RETURNS_SCHEMA


def test_empty_input_returns_empty_frame():
    df = compute_returns(_adjusted([]))
    assert df.height == 0
    assert list(df.columns) == list(RETURNS_SCHEMA.keys())


def test_missing_required_column_raises():
    bad = pl.DataFrame({"isin": ["X"], "date": [date(2025, 1, 1)]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_returns(bad)


def test_one_day_return_basic():
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 1), "symbol": "ACME", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "ACME", "adj_close": 110.0},
        {"isin": "INE001", "date": date(2025, 1, 3), "symbol": "ACME", "adj_close": 99.0},
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    assert out["ret_1d"].to_list()[0] is None
    assert out["ret_1d"].to_list()[1] == pytest.approx(0.10)
    assert out["ret_1d"].to_list()[2] == pytest.approx(-0.10)


def test_unsorted_input_is_sorted_before_compute():
    # Out of order: should still produce correct 1d return.
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 3), "symbol": "ACME", "adj_close": 99.0},
        {"isin": "INE001", "date": date(2025, 1, 1), "symbol": "ACME", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "ACME", "adj_close": 110.0},
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    assert out["ret_1d"].to_list() == pytest.approx([None, 0.10, -0.10], nan_ok=True)


def test_bootstrap_rows_null_at_long_horizons():
    # 10 trading days; 5d return populated from index 5 onward, 21d never.
    rows = [
        {
            "isin": "INE001",
            "date": date(2025, 1, 1 + i),
            "symbol": "ACME",
            "adj_close": 100.0 + i,
        }
        for i in range(10)
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    ret_5d = out["ret_5d"].to_list()
    ret_21d = out["ret_21d"].to_list()
    assert all(v is None for v in ret_5d[:5])
    assert ret_5d[5] == pytest.approx(105 / 100 - 1)
    assert all(v is None for v in ret_21d)


def test_five_day_return_spans_calendar_gaps():
    # Trading days only; intentionally skip weekends. 5-day return is
    # purely positional (5 rows apart), not 5 calendar days.
    rows = [
        {"isin": "INE001", "date": d, "symbol": "ACME", "adj_close": p}
        for d, p in [
            (date(2025, 1, 6), 100.0),  # Mon
            (date(2025, 1, 7), 101.0),
            (date(2025, 1, 8), 102.0),
            (date(2025, 1, 9), 103.0),
            (date(2025, 1, 10), 104.0),  # Fri
            (date(2025, 1, 13), 110.0),  # next Mon
        ]
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    # 5d return on the 6th row compares 110 vs 100.
    assert out["ret_5d"].to_list()[-1] == pytest.approx(110 / 100 - 1)


def test_per_isin_partition_no_cross_leak():
    # Two ISINs with overlapping dates. Returns must be per-ISIN; the
    # shift must not pull a value from the other ISIN's prior row.
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 1), "symbol": "A", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "A", "adj_close": 110.0},
        {"isin": "INE002", "date": date(2025, 1, 1), "symbol": "B", "adj_close": 50.0},
        {"isin": "INE002", "date": date(2025, 1, 2), "symbol": "B", "adj_close": 55.0},
    ]
    out = compute_returns(_adjusted(rows)).sort(["isin", "date"])
    # Both ISINs: first row null, second row 10%.
    isin1 = out.filter(pl.col("isin") == "INE001")["ret_1d"].to_list()
    isin2 = out.filter(pl.col("isin") == "INE002")["ret_1d"].to_list()
    assert isin1[0] is None and isin1[1] == pytest.approx(0.10)
    assert isin2[0] is None and isin2[1] == pytest.approx(0.10)


def test_ytd_anchored_to_first_day_of_year():
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "A", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 6, 1), "symbol": "A", "adj_close": 120.0},
        {"isin": "INE001", "date": date(2025, 12, 31), "symbol": "A", "adj_close": 150.0},
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    ytd = out["ret_ytd"].to_list()
    assert ytd[0] == pytest.approx(0.0)
    assert ytd[1] == pytest.approx(0.20)
    assert ytd[2] == pytest.approx(0.50)


def test_ytd_resets_at_year_boundary():
    # First trading day of 2026 is the new YTD anchor for that year.
    rows = [
        {"isin": "INE001", "date": date(2025, 12, 30), "symbol": "A", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 12, 31), "symbol": "A", "adj_close": 200.0},
        {"isin": "INE001", "date": date(2026, 1, 2), "symbol": "A", "adj_close": 210.0},
        {"isin": "INE001", "date": date(2026, 1, 3), "symbol": "A", "adj_close": 252.0},
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    ytd = out["ret_ytd"].to_list()
    # 2025 anchor = 100. 2026 anchor = 210.
    assert ytd[0] == pytest.approx(0.0)
    assert ytd[1] == pytest.approx(1.0)
    assert ytd[2] == pytest.approx(0.0)
    assert ytd[3] == pytest.approx(252 / 210 - 1)


def test_ytd_per_isin_does_not_leak_across_instruments():
    # Two ISINs in the same year: each gets its own anchor.
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "A", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 6, 1), "symbol": "A", "adj_close": 150.0},
        {"isin": "INE002", "date": date(2025, 3, 1), "symbol": "B", "adj_close": 80.0},
        {"isin": "INE002", "date": date(2025, 9, 1), "symbol": "B", "adj_close": 96.0},
    ]
    out = compute_returns(_adjusted(rows)).sort(["isin", "date"])
    a = out.filter(pl.col("isin") == "INE001")["ret_ytd"].to_list()
    b = out.filter(pl.col("isin") == "INE002")["ret_ytd"].to_list()
    assert a[0] == pytest.approx(0.0) and a[1] == pytest.approx(0.50)
    # INE002 anchor is March (first row in 2025 for that ISIN), not Jan.
    assert b[0] == pytest.approx(0.0) and b[1] == pytest.approx(0.20)


def test_output_carries_symbol_and_adj_close():
    rows = [
        {"isin": "INE001", "date": date(2025, 1, 1), "symbol": "ACME", "adj_close": 100.0},
        {"isin": "INE001", "date": date(2025, 1, 2), "symbol": "ACME", "adj_close": 110.0},
    ]
    out = compute_returns(_adjusted(rows)).sort("date")
    assert out["symbol"].to_list() == ["ACME", "ACME"]
    assert out["adj_close"].to_list() == [100.0, 110.0]


def test_extra_input_columns_dropped():
    # Passing in adjusted parquet with extra cols (open/high/low/etc):
    # the result schema should still match RETURNS_SCHEMA exactly.
    df = pl.DataFrame(
        {
            "isin": ["INE001", "INE001"],
            "date": [date(2025, 1, 1), date(2025, 1, 2)],
            "symbol": ["A", "A"],
            "adj_close": [100.0, 110.0],
            "open": [99.0, 108.0],
            "volume": [1000, 1500],
        },
        schema_overrides={"date": pl.Date},
    )
    out = compute_returns(df)
    assert out.schema == RETURNS_SCHEMA
