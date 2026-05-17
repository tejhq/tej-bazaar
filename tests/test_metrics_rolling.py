"""Tests for pipeline.metrics.rolling."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pipeline.metrics.rolling import (
    ROLLING_SCHEMA,
    WINDOW_52W,
    WINDOW_VOL_LONG,
    WINDOW_VOL_SHORT,
    compute_rolling,
)


def _rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.Utf8,
            "date": pl.Date,
            "symbol": pl.Utf8,
            "adj_close": pl.Float64,
            "volume": pl.Int64,
            "turnover": pl.Float64,
        },
    )


def _series(n: int, isin: str = "INE001", symbol: str = "ACME") -> pl.DataFrame:
    # Synthetic n trading days starting 2025-01-01, monotonically rising price.
    return _rows([
        {
            "isin": isin,
            "date": date(2025, 1, 1) + timedelta(days=i),
            "symbol": symbol,
            "adj_close": 100.0 + i,
            "volume": 1_000 + i * 10,
            "turnover": (100.0 + i) * (1_000 + i * 10),
        }
        for i in range(n)
    ])


def test_schema_matches_constant():
    assert compute_rolling(_rows([])).schema == ROLLING_SCHEMA


def test_empty_input_returns_empty_frame():
    out = compute_rolling(_rows([]))
    assert out.height == 0
    assert list(out.columns) == list(ROLLING_SCHEMA.keys())


def test_missing_required_column_raises():
    bad = pl.DataFrame({"isin": ["X"], "date": [date(2025, 1, 1)]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_rolling(bad)


def test_bootstrap_nulls_until_full_window():
    # 252-day window: rows 0..250 null for high_52w, row 251 populated.
    df = _series(WINDOW_52W + 5)
    out = compute_rolling(df).sort("date")
    h = out["high_52w"].to_list()
    assert all(v is None for v in h[: WINDOW_52W - 1])
    assert h[WINDOW_52W - 1] is not None
    # Last row sees a full window ending today. Rising series, so high = latest.
    assert h[-1] == pytest.approx(100.0 + WINDOW_52W + 4)


def test_52w_high_and_low_track_full_window():
    n = WINDOW_52W + 10
    df = _series(n)
    out = compute_rolling(df).sort("date")
    # On the 252nd row (index 251), window is rows [0..251] = 252 entries.
    # Rising series so high = price[251] = 100+251, low = price[0] = 100.
    idx = WINDOW_52W - 1
    assert out["high_52w"][idx] == pytest.approx(100.0 + idx)
    assert out["low_52w"][idx] == pytest.approx(100.0)
    # On row 252 (window [1..252]), low advances to 101.
    assert out["low_52w"][WINDOW_52W] == pytest.approx(101.0)


def test_pct_off_52w_high_zero_at_new_high():
    df = _series(WINDOW_52W + 1)
    out = compute_rolling(df).sort("date")
    # Series is monotonically rising, so once populated, current adj_close
    # IS the 52w high. pct_off_52w_high = 0.
    last = out.tail(1)
    assert last["pct_off_52w_high"][0] == pytest.approx(0.0)
    # Bottom of window is rolling 0 -> N-WINDOW_52W, so pct_off_low > 0.
    assert last["pct_off_52w_low"][0] > 0


def test_avg_vol_20d_uses_raw_volume_mean():
    df = _series(WINDOW_VOL_SHORT + 5)
    out = compute_rolling(df).sort("date")
    # On row WINDOW_VOL_SHORT-1, volumes are 1000, 1010, ..., 1000 + 19*10.
    # Mean = 1000 + 9.5*10 = 1095.
    idx = WINDOW_VOL_SHORT - 1
    assert out["avg_vol_20d"][idx] == pytest.approx(1095.0)


def test_avg_vol_60d_separate_window():
    df = _series(WINDOW_VOL_LONG + 1)
    out = compute_rolling(df).sort("date")
    idx = WINDOW_VOL_LONG - 1
    expected = sum(1_000 + i * 10 for i in range(WINDOW_VOL_LONG)) / WINDOW_VOL_LONG
    assert out["avg_vol_60d"][idx] == pytest.approx(expected)
    # Pre-window: null.
    assert out["avg_vol_60d"][idx - 1] is None


def test_per_isin_no_cross_window_leak():
    # ISIN A has 30 days, ISIN B has 5 days. ISIN B must never see A's data.
    a = _series(30, isin="INE001", symbol="A")
    b = _series(5, isin="INE002", symbol="B")
    out = compute_rolling(pl.concat([a, b])).sort(["isin", "date"])

    b_rows = out.filter(pl.col("isin") == "INE002")
    # Only 5 days for B, far less than any window. All rolling cols null.
    assert all(v is None for v in b_rows["high_52w"].to_list())
    assert all(v is None for v in b_rows["low_52w"].to_list())
    assert all(v is None for v in b_rows["avg_vol_20d"].to_list())

    a_rows = out.filter(pl.col("isin") == "INE001").sort("date")
    # A has 30 days >= 20 day vol window, so last row populated.
    assert a_rows["avg_vol_20d"][-1] is not None


def test_avg_turnover_20d_independent_of_volume_window():
    df = _series(WINDOW_VOL_SHORT + 1)
    out = compute_rolling(df).sort("date")
    idx = WINDOW_VOL_SHORT - 1
    expected = sum(
        (100.0 + i) * (1_000 + i * 10) for i in range(WINDOW_VOL_SHORT)
    ) / WINDOW_VOL_SHORT
    assert out["avg_turnover_20d"][idx] == pytest.approx(expected)


def test_unsorted_input_handled():
    # Reverse the rows; rolling should still be correct.
    df = _series(WINDOW_VOL_SHORT + 1)
    rev = df.reverse()
    out = compute_rolling(rev).sort("date")
    idx = WINDOW_VOL_SHORT - 1
    assert out["avg_vol_20d"][idx] == pytest.approx(1095.0)


def test_extra_input_columns_dropped():
    df = pl.DataFrame(
        {
            "isin": ["INE001"] * 25,
            "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(25)],
            "symbol": ["A"] * 25,
            "adj_close": [100.0 + i for i in range(25)],
            "volume": [1000] * 25,
            "turnover": [100_000.0] * 25,
            "open": [99.0] * 25,
            "high": [101.0] * 25,
        },
        schema_overrides={"date": pl.Date},
    )
    out = compute_rolling(df)
    assert out.schema == ROLLING_SCHEMA


def test_output_carries_symbol_and_adj_close():
    df = _series(5)
    out = compute_rolling(df).sort("date")
    assert out["symbol"].to_list() == ["ACME"] * 5
    assert out["adj_close"].to_list() == [100.0, 101.0, 102.0, 103.0, 104.0]
