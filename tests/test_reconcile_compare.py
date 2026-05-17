from datetime import date

import polars as pl
import pytest

from pipeline.reconcile.compare import (
    reconcile_symbol,
    summarize,
)


def _ours(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"date": pl.Date, "adj_close": pl.Float64}, orient="row")


def _ref(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"date": pl.Date, "yahoo_adjclose": pl.Float64}, orient="row"
    )


def test_perfect_match_pct_within_tol_is_100():
    ours = _ours([(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 101.0)])
    ref = _ref([(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 101.0)])
    s = reconcile_symbol("FOO", ours, ref)
    assert s.rows_compared == 2
    assert s.pct_within_tol == 100.0
    assert s.max_abs_diff_pct == 0.0
    assert s.mean_abs_diff_pct == 0.0


def test_diff_at_tolerance_boundary_counted_within():
    # exact 0.5% diff => within tolerance (<=)
    ours = _ours([(date(2024, 1, 2), 100.5)])
    ref = _ref([(date(2024, 1, 2), 100.0)])
    s = reconcile_symbol("FOO", ours, ref, tolerance_pct=0.5)
    assert s.pct_within_tol == 100.0
    assert s.max_abs_diff_pct == pytest.approx(0.5)


def test_diff_above_tolerance_counted_outside():
    ours = _ours([(date(2024, 1, 2), 102.0), (date(2024, 1, 3), 100.5)])
    ref = _ref([(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 100.0)])
    s = reconcile_symbol("FOO", ours, ref, tolerance_pct=0.5)
    assert s.rows_compared == 2
    assert s.pct_within_tol == 50.0  # only second row within
    assert s.max_abs_diff_pct == pytest.approx(2.0)


def test_inner_join_drops_unmatched_dates():
    ours = _ours([
        (date(2024, 1, 2), 100.0),
        (date(2024, 1, 3), 101.0),
        (date(2024, 1, 4), 102.0),
    ])
    ref = _ref([(date(2024, 1, 3), 101.0)])
    s = reconcile_symbol("FOO", ours, ref)
    assert s.rows_compared == 1


def test_no_overlap_returns_zero_rows():
    ours = _ours([(date(2024, 1, 2), 100.0)])
    ref = _ref([(date(2024, 1, 3), 100.0)])
    s = reconcile_symbol("FOO", ours, ref)
    assert s.rows_compared == 0
    assert s.pct_within_tol == 0.0


def test_null_values_dropped_before_compare():
    ours = pl.DataFrame(
        {"date": [date(2024, 1, 2), date(2024, 1, 3)], "adj_close": [None, 101.0]},
        schema={"date": pl.Date, "adj_close": pl.Float64},
    )
    ref = _ref([(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 101.0)])
    s = reconcile_symbol("FOO", ours, ref)
    assert s.rows_compared == 1


def test_zero_or_negative_reference_dropped():
    ours = _ours([(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 100.0)])
    ref = _ref([(date(2024, 1, 2), 0.0), (date(2024, 1, 3), 100.0)])
    s = reconcile_symbol("FOO", ours, ref)
    assert s.rows_compared == 1


def test_missing_required_columns_raises():
    bad = pl.DataFrame({"date": [date(2024, 1, 2)]})
    ref = _ref([(date(2024, 1, 2), 100.0)])
    with pytest.raises(ValueError, match="ours missing"):
        reconcile_symbol("FOO", bad, ref)
    ours = _ours([(date(2024, 1, 2), 100.0)])
    bad_ref = pl.DataFrame({"date": [date(2024, 1, 2)]})
    with pytest.raises(ValueError, match="reference missing"):
        reconcile_symbol("FOO", ours, bad_ref)


def test_summarize_weights_by_rows():
    from pipeline.reconcile.compare import SymbolReconcileStats

    stats = [
        SymbolReconcileStats(symbol="A", rows_compared=100, pct_within_tol=100.0,
                             max_abs_diff_pct=0.1, mean_abs_diff_pct=0.05),
        SymbolReconcileStats(symbol="B", rows_compared=900, pct_within_tol=99.0,
                             max_abs_diff_pct=2.0, mean_abs_diff_pct=0.3),
    ]
    r = summarize(stats)
    assert r.overall_rows == 1000
    # 100 within from A, 891 within from B = 991/1000 = 99.1
    assert r.overall_pct_within_tol == pytest.approx(99.1)


def test_summarize_empty_returns_zero():
    r = summarize([])
    assert r.overall_rows == 0
    assert r.overall_pct_within_tol == 0.0
