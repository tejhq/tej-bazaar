from datetime import date

import polars as pl

from pipeline.actions.audit import find_jumps


def _series(symbol: str, closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, d + 1) for d in range(len(closes))],
            "symbol": [symbol] * len(closes),
            "adj_close": [float(c) for c in closes],
        }
    )


def test_a_clean_series_reports_nothing():
    assert find_jumps(_series("CLEAN", [100, 101, 99, 103, 98])) == []


def test_an_unadjusted_split_is_reported_as_unexplained():
    jumps = find_jumps(_series("SPLITTER", [1000, 1010, 101, 102]))
    assert len(jumps) == 1
    j = jumps[0]
    assert j.symbol == "SPLITTER" and j.date == date(2020, 1, 3)
    assert not j.explained and j.implied == "10:1"


def test_a_jump_with_an_action_on_the_ex_date_is_explained():
    actions = pl.DataFrame(
        {
            "symbol": ["SPLITTER"],
            "ex_date": [date(2020, 1, 3)],
            "type": ["split"],
            "raw_subject": ["Face Value Split Rs 10 To Re 1"],
        }
    )
    jumps = find_jumps(_series("SPLITTER", [1000, 1010, 101, 102]), actions)
    assert len(jumps) == 1 and jumps[0].explained
    assert jumps[0].action_type == "split"


def test_a_reverse_split_is_caught_too():
    jumps = find_jumps(_series("CONSOL", [10.0, 10.1, 101.0, 102.0]))
    assert len(jumps) == 1 and jumps[0].ratio > 1 and jumps[0].implied == "10:1"


def test_symbols_do_not_bleed_into_each_other():
    two = pl.concat([_series("A", [100.0, 101.0]), _series("B", [5.0, 5.1])])
    assert find_jumps(two) == []


def test_a_hole_in_the_data_is_not_a_jump():
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2024, 6, 1)],
            "symbol": ["GAPPY", "GAPPY"],
            "adj_close": [10.0, 500.0],
        }
    )
    assert find_jumps(frame) == []
