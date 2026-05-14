from datetime import date

import polars as pl
import pytest

from pipeline.symbol_history import (
    SYMBOL_HISTORY_SCHEMA,
    build_symbol_history,
    lookup_current_symbol,
    lookup_isin,
)


def _prices(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"isin": pl.Utf8, "symbol": pl.Utf8, "date": pl.Date},
        orient="row",
    )


# --- build_symbol_history ----------------------------------------------


def test_single_isin_single_symbol():
    p = _prices([
        ("INE001", "X", date(2024, 1, 1)),
        ("INE001", "X", date(2024, 1, 2)),
        ("INE001", "X", date(2024, 1, 3)),
    ])
    h = build_symbol_history(p, "NSE")
    assert h.height == 1
    row = h.row(0, named=True)
    assert row["isin"] == "INE001"
    assert row["symbol"] == "X"
    assert row["valid_from"] == date(2024, 1, 1)
    assert row["valid_to"] == date(2024, 1, 3)
    assert row["trading_days"] == 3
    assert row["exchange"] == "NSE"


def test_symbol_rename_creates_two_intervals():
    p = _prices([
        ("INE001", "OLDSYM", date(2024, 1, 1)),
        ("INE001", "OLDSYM", date(2024, 1, 2)),
        ("INE001", "NEWSYM", date(2024, 1, 3)),
        ("INE001", "NEWSYM", date(2024, 1, 4)),
    ])
    h = build_symbol_history(p, "NSE").sort("valid_from")
    assert h.height == 2
    assert h["symbol"].to_list() == ["OLDSYM", "NEWSYM"]
    assert h["valid_from"].to_list() == [date(2024, 1, 1), date(2024, 1, 3)]
    assert h["valid_to"].to_list() == [date(2024, 1, 2), date(2024, 1, 4)]


def test_symbol_change_back_creates_three_intervals():
    # X -> Y -> X (rare but possible after temporary rebrand)
    p = _prices([
        ("INE001", "X", date(2024, 1, 1)),
        ("INE001", "Y", date(2024, 1, 2)),
        ("INE001", "X", date(2024, 1, 3)),
    ])
    h = build_symbol_history(p, "NSE").sort("valid_from")
    assert h.height == 3
    assert h["symbol"].to_list() == ["X", "Y", "X"]


def test_multiple_isins_independent():
    p = _prices([
        ("INE001", "A", date(2024, 1, 1)),
        ("INE002", "B", date(2024, 1, 1)),
        ("INE001", "A", date(2024, 1, 2)),
        ("INE002", "B", date(2024, 1, 2)),
    ])
    h = build_symbol_history(p, "NSE").sort("isin")
    assert h.height == 2
    assert h["isin"].to_list() == ["INE001", "INE002"]
    assert h["symbol"].to_list() == ["A", "B"]


def test_gap_in_trading_with_same_symbol_one_interval():
    # Suspension: missing dates but same symbol on resume -> single interval
    # (valid_from..valid_to spans the gap; trading_days reflects actual trading)
    p = _prices([
        ("INE001", "X", date(2024, 1, 1)),
        ("INE001", "X", date(2024, 1, 2)),
        # gap: Jan 3, 4, 5
        ("INE001", "X", date(2024, 1, 8)),
    ])
    h = build_symbol_history(p, "NSE")
    assert h.height == 1
    row = h.row(0, named=True)
    assert row["valid_from"] == date(2024, 1, 1)
    assert row["valid_to"] == date(2024, 1, 8)
    assert row["trading_days"] == 3  # actual trading days, not span


def test_drops_null_isin_rows():
    p = _prices([
        ("INE001", "X", date(2024, 1, 1)),
        (None, "Y", date(2024, 1, 2)),  # dropped
        ("INE001", "X", date(2024, 1, 3)),
    ])
    h = build_symbol_history(p, "NSE")
    assert h.height == 1
    assert h["trading_days"][0] == 2


def test_empty_input_returns_empty_with_schema():
    p = _prices([])
    h = build_symbol_history(p, "NSE")
    assert h.height == 0
    assert dict(h.schema) == SYMBOL_HISTORY_SCHEMA


def test_missing_required_column_raises():
    p = pl.DataFrame({"isin": ["INE001"], "date": [date(2024, 1, 1)]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_symbol_history(p, "NSE")


def test_unsorted_input_handled():
    # Algorithm sorts internally; out-of-order input must produce same result
    p = _prices([
        ("INE001", "Y", date(2024, 1, 3)),
        ("INE001", "X", date(2024, 1, 1)),
        ("INE001", "X", date(2024, 1, 2)),
    ])
    h = build_symbol_history(p, "NSE").sort("valid_from")
    assert h["symbol"].to_list() == ["X", "Y"]
    assert h["valid_from"].to_list() == [date(2024, 1, 1), date(2024, 1, 3)]


def test_exchange_label_attached():
    p = _prices([("INE001", "X", date(2024, 1, 1))])
    h = build_symbol_history(p, "BSE")
    assert h["exchange"][0] == "BSE"


# --- lookup_isin ------------------------------------------------------


def test_lookup_isin_finds_match():
    p = _prices([
        ("INE001", "OLDSYM", date(2024, 1, 1)),
        ("INE001", "OLDSYM", date(2024, 1, 2)),
        ("INE001", "NEWSYM", date(2024, 1, 3)),
    ])
    h = build_symbol_history(p, "NSE")
    assert lookup_isin(h, "OLDSYM", date(2024, 1, 1)) == "INE001"
    assert lookup_isin(h, "NEWSYM", date(2024, 1, 3)) == "INE001"


def test_lookup_isin_returns_none_outside_range():
    p = _prices([("INE001", "X", date(2024, 1, 1))])
    h = build_symbol_history(p, "NSE")
    assert lookup_isin(h, "X", date(2023, 12, 31)) is None
    assert lookup_isin(h, "Y", date(2024, 1, 1)) is None


def test_lookup_isin_respects_interval_boundary():
    # Symbol changes on Jan 3: query on Jan 2 must return OLD interval
    p = _prices([
        ("INE001", "OLD", date(2024, 1, 1)),
        ("INE001", "OLD", date(2024, 1, 2)),
        ("INE002", "OLD", date(2024, 1, 3)),  # different ISIN reuses symbol
    ])
    h = build_symbol_history(p, "NSE")
    assert lookup_isin(h, "OLD", date(2024, 1, 2)) == "INE001"
    assert lookup_isin(h, "OLD", date(2024, 1, 3)) == "INE002"


# --- lookup_current_symbol --------------------------------------------


def test_lookup_current_symbol_returns_latest():
    p = _prices([
        ("INE001", "OLD", date(2024, 1, 1)),
        ("INE001", "NEW", date(2024, 6, 1)),
    ])
    h = build_symbol_history(p, "NSE")
    assert lookup_current_symbol(h, "INE001") == "NEW"


def test_lookup_current_symbol_unknown_isin():
    h = pl.DataFrame(schema=SYMBOL_HISTORY_SCHEMA)
    assert lookup_current_symbol(h, "INE999") is None
