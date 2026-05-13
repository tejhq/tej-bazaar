from datetime import date

import polars as pl
import pytest

from pipeline.actions import (
    CorporateAction,
    back_adjust,
    compute_action_factors,
    to_polars,
)


def _prices(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"isin": pl.Utf8, "date": pl.Date, "close": pl.Float64},
        orient="row",
    )


# --- compute_action_factors ---------------------------------------------


def test_compute_action_factors_dividend_uses_prev_close():
    actions = to_polars([
        CorporateAction(
            exchange="NSE", symbol="X", isin="INE001",
            company="X", ex_date=date(2024, 6, 1),
            record_date=None, type="dividend", cash_amount=10.0,
        ),
    ])
    prices = _prices([
        ("INE001", date(2024, 5, 30), 200.0),
        ("INE001", date(2024, 5, 31), 200.0),  # most recent before ex_date
        ("INE001", date(2024, 6, 1), 190.0),
    ])
    f = compute_action_factors(actions, prices)
    assert f.height == 1
    # prev_close = 200, div = 10 -> factor = 190/200 = 0.95
    assert f["factor"][0] == pytest.approx(0.95)


def test_compute_action_factors_split_ignores_prev_close():
    actions = to_polars([
        CorporateAction(
            exchange="NSE", symbol="Y", isin="INE002",
            company="Y", ex_date=date(2024, 6, 1),
            record_date=None, type="split",
            face_value_from=10.0, face_value_to=1.0,
        ),
    ])
    prices = _prices([("INE002", date(2024, 5, 31), 1000.0)])
    f = compute_action_factors(actions, prices)
    assert f["factor"][0] == pytest.approx(0.1)


def test_compute_action_factors_drops_isin_null():
    # BSE row without scrip-map join: isin is null, must be dropped.
    actions = to_polars([
        CorporateAction(
            exchange="BSE", symbol="X", isin=None,
            company="X", ex_date=date(2024, 6, 1),
            record_date=None, type="bonus", ratio_num=1, ratio_den=1,
        ),
    ])
    prices = _prices([("INE001", date(2024, 5, 31), 100.0)])
    f = compute_action_factors(actions, prices)
    assert f.height == 0


def test_compute_action_factors_dividend_no_prev_close_falls_to_one():
    # Dividend with ex_date BEFORE any price -> no prev_close -> factor 1.0
    actions = to_polars([
        CorporateAction(
            exchange="NSE", symbol="X", isin="INE001",
            company="X", ex_date=date(2024, 1, 1),
            record_date=None, type="dividend", cash_amount=5.0,
        ),
    ])
    prices = _prices([("INE001", date(2024, 6, 1), 100.0)])
    f = compute_action_factors(actions, prices)
    assert f["factor"][0] == 1.0


def test_compute_action_factors_empty_actions_returns_empty():
    f = compute_action_factors(
        to_polars([]),
        _prices([("INE001", date(2024, 1, 1), 100.0)]),
    )
    assert f.height == 0
    assert set(f.columns) == {"isin", "ex_date", "factor"}


# --- back_adjust --------------------------------------------------------


def test_back_adjust_single_split_factor_applied_to_history():
    prices = _prices([
        ("INE001", date(2024, 1, 1), 1000.0),  # pre-split
        ("INE001", date(2024, 1, 2), 1000.0),  # pre-split
        ("INE001", date(2024, 1, 3), 100.0),   # ex_date (post-split)
        ("INE001", date(2024, 1, 4), 100.0),   # post-split
    ])
    factors = pl.DataFrame(
        {"isin": ["INE001"], "ex_date": [date(2024, 1, 3)], "factor": [0.1]},
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, factors).sort("date")
    # Pre-ex_date rows get factor=0.1; ex_date and after get factor=1.0
    assert out["adj_factor_cumulative"].to_list() == [0.1, 0.1, 1.0, 1.0]
    assert out["adj_close"].to_list() == [100.0, 100.0, 100.0, 100.0]


def test_back_adjust_two_actions_compound():
    # Bonus 1:1 on Mar 1 (factor 0.5), then 10:1 split on Jun 1 (factor 0.1)
    # Price before bonus: should be * (0.5 * 0.1) = 0.05
    # Price between bonus and split: should be * 0.1
    # Price on/after split: should be * 1.0
    prices = _prices([
        ("INE001", date(2024, 2, 1), 2000.0),   # pre-bonus
        ("INE001", date(2024, 4, 1), 1000.0),   # post-bonus, pre-split
        ("INE001", date(2024, 7, 1), 100.0),    # post-split
    ])
    factors = pl.DataFrame(
        {
            "isin": ["INE001", "INE001"],
            "ex_date": [date(2024, 3, 1), date(2024, 6, 1)],
            "factor": [0.5, 0.1],
        },
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, factors).sort("date")
    assert out["adj_factor_cumulative"].to_list() == pytest.approx([0.05, 0.1, 1.0])
    assert out["adj_close"].to_list() == pytest.approx([100.0, 100.0, 100.0])


def test_back_adjust_isin_without_actions_passthrough():
    prices = _prices([
        ("INE001", date(2024, 1, 1), 100.0),
        ("INE002", date(2024, 1, 1), 200.0),
    ])
    factors = pl.DataFrame(
        {"isin": ["INE001"], "ex_date": [date(2024, 6, 1)], "factor": [0.5]},
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, factors).sort(["isin", "date"])
    by_isin = {r["isin"]: r for r in out.iter_rows(named=True)}
    # INE002 has no actions: passthrough
    assert by_isin["INE002"]["adj_factor_cumulative"] == 1.0
    assert by_isin["INE002"]["adj_close"] == 200.0
    # INE001 has 1 future action: 0.5
    assert by_isin["INE001"]["adj_factor_cumulative"] == 0.5
    assert by_isin["INE001"]["adj_close"] == 50.0


def test_back_adjust_empty_factors_passthrough():
    prices = _prices([("INE001", date(2024, 1, 1), 100.0)])
    empty = pl.DataFrame(
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, empty)
    assert out["adj_factor_cumulative"][0] == 1.0
    assert out["adj_close"][0] == 100.0


def test_back_adjust_ex_date_price_not_adjusted():
    # Price ON ex_date is the post-action price; factor at that date is 1.0
    prices = _prices([
        ("INE001", date(2024, 6, 1), 100.0),  # ex_date itself
    ])
    factors = pl.DataFrame(
        {"isin": ["INE001"], "ex_date": [date(2024, 6, 1)], "factor": [0.1]},
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, factors)
    assert out["adj_factor_cumulative"][0] == 1.0
    assert out["adj_close"][0] == 100.0


def test_back_adjust_preserves_other_columns():
    prices = pl.DataFrame(
        [("INE001", date(2024, 1, 1), 100.0, "X", 1_000_000)],
        schema={
            "isin": pl.Utf8, "date": pl.Date, "close": pl.Float64,
            "symbol": pl.Utf8, "volume": pl.Int64,
        },
        orient="row",
    )
    factors = pl.DataFrame(
        {"isin": ["INE001"], "ex_date": [date(2024, 6, 1)], "factor": [0.5]},
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    out = back_adjust(prices, factors)
    # Extra columns survive
    assert "symbol" in out.columns
    assert "volume" in out.columns
    assert out["volume"][0] == 1_000_000
    assert out["adj_factor_cumulative"][0] == 0.5


def test_back_adjust_missing_close_column_raises():
    prices = pl.DataFrame(
        {"isin": ["INE001"], "date": [date(2024, 1, 1)]},
        schema={"isin": pl.Utf8, "date": pl.Date},
    )
    factors = pl.DataFrame(
        schema={"isin": pl.Utf8, "ex_date": pl.Date, "factor": pl.Float64},
    )
    with pytest.raises(ValueError, match="missing required columns"):
        back_adjust(prices, factors)


# --- end-to-end ---------------------------------------------------------


def test_e2e_compute_and_back_adjust():
    """Wire compute_action_factors -> back_adjust with a real scenario."""
    actions = to_polars([
        CorporateAction(
            exchange="NSE", symbol="X", isin="INE001",
            company="X", ex_date=date(2024, 3, 1),
            record_date=None, type="dividend", cash_amount=10.0,
        ),
        CorporateAction(
            exchange="NSE", symbol="X", isin="INE001",
            company="X", ex_date=date(2024, 6, 1),
            record_date=None, type="split",
            face_value_from=10.0, face_value_to=1.0,
        ),
    ])
    prices = _prices([
        ("INE001", date(2024, 1, 1), 1000.0),  # pre-div, pre-split
        ("INE001", date(2024, 2, 29), 200.0),  # prev_close for dividend
        ("INE001", date(2024, 3, 1), 190.0),   # ex_date of dividend
        ("INE001", date(2024, 4, 1), 1200.0),  # post-div, pre-split
        ("INE001", date(2024, 6, 1), 120.0),   # ex_date of split
        ("INE001", date(2024, 7, 1), 130.0),   # post-split
    ])
    f = compute_action_factors(actions, prices)
    # dividend factor: (200 - 10) / 200 = 0.95
    # split factor: 1/10 = 0.1
    assert f.sort("ex_date")["factor"].to_list() == pytest.approx([0.95, 0.1])

    out = back_adjust(prices, f).sort("date")
    # Cumulative factors per date:
    #   2024-01-01: both later actions apply -> 0.95 * 0.1 = 0.095
    #   2024-02-29: same -> 0.095
    #   2024-03-01: only split applies (div ex_date == this date, not counted) -> 0.1
    #   2024-04-01: only split applies -> 0.1
    #   2024-06-01: no later actions -> 1.0
    #   2024-07-01: 1.0
    assert out["adj_factor_cumulative"].to_list() == pytest.approx(
        [0.095, 0.095, 0.1, 0.1, 1.0, 1.0]
    )
