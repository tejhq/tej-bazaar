from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest

from pipeline.actions.audit import (
    FACTOR_ONE,
    NO_ACTION,
    NOT_ADJUSTED,
    RESIDUAL,
    SCALED_TYPES,
    find_jumps,
)
from pipeline.actions.parse import parse_nse_record
from pipeline.actions.schema import to_polars


def _series(symbol: str, closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, d + 1) for d in range(len(closes))],
            "symbol": [symbol] * len(closes),
            "adj_close": [float(c) for c in closes],
        }
    )


def _bars(
    symbol: str, ex_date: str, before: float, after: float, isin: str | None = None
) -> pl.DataFrame:
    """Two sessions at `before`, then the ex date and the session after at `after`."""
    ex = date.fromisoformat(ex_date)
    return pl.DataFrame(
        {
            "date": [ex - timedelta(days=2), ex - timedelta(days=1), ex, ex + timedelta(days=1)],
            "symbol": [symbol] * 4,
            "isin": [isin] * 4,
            "adj_close": [before, before, after, after],
        },
        schema_overrides={"isin": pl.Utf8},
    )


def _feed(symbol: str, isin: str, ex_date: str, subject: str):
    """The rows this parser writes for one NSE announcement."""
    d = date.fromisoformat(ex_date).strftime("%d-%b-%Y")
    rec = {
        "symbol": symbol,
        "series": "EQ",
        "isin": isin,
        "subject": subject,
        "exDate": d,
        "recDate": d,
    }
    return parse_nse_record(rec)


def _unread(action):
    """The row with nothing read but its type, as the parser at 9d0845d stored the B8 lines."""
    return replace(
        action,
        ratio_num=None,
        ratio_den=None,
        cash_amount=None,
        face_value_from=None,
        face_value_to=None,
    )


def _found(j):
    return [(a.type, a.factor) for a in j.actions]


def test_a_clean_series_reports_nothing():
    assert find_jumps(_series("CLEAN", [100, 101, 99, 103, 98])) == []


def test_an_unadjusted_split_is_reported_as_unexplained():
    jumps = find_jumps(_series("SPLITTER", [1000, 1010, 101, 102]))
    assert len(jumps) == 1
    j = jumps[0]
    assert j.symbol == "SPLITTER" and j.date == date(2020, 1, 3)
    assert j.cause == NO_ACTION and j.actions == () and not j.explained and j.implied == "10:1"


def test_the_types_that_can_be_left_at_factor_1_are_the_ones_compute_factor_scales():
    assert {"split", "bonus", "dividend"} == SCALED_TYPES


# B8: an action on the date made a jump "explained", whatever its factor.
@pytest.mark.parametrize(
    ("symbol", "isin", "ex_date", "subject", "kind", "before", "after"),
    [
        ("INVENTURE", "INE878H01016", "2012-05-22", "Bonus - 3:1", "bonus", 202.05, 47.55),
        (
            "CORPBANK",
            "INE112A01015",
            "2015-01-22",
            "Face Valus Split (Sub-Division) - From Rs 10/- Per To Rs 2/- Per Share",
            "split",
            388.35,
            74.30,
        ),
    ],
)
def test_a_split_or_bonus_left_at_factor_1_is_not_explained(
    symbol, isin, ex_date, subject, kind, before, after
):
    [stored] = _feed(symbol, isin, ex_date, subject)
    [j] = find_jumps(_bars(symbol, ex_date, before, after), to_polars([_unread(stored)]))
    assert j.cause == FACTOR_ONE and not j.explained
    assert _found(j) == [(kind, 1.0)] and j.actions[0].raw_subject == subject


def test_a_missed_1_to_2_bonus_is_caught_under_40_percent():
    # AJANTPHARM 2022-06-22: a 34% step, inside DEFAULT_THRESHOLD and beyond SPLIT_THRESHOLD.
    [stored] = _feed("AJANTPHARM", "INE031B01015", "2022-06-22", "Bonus- 1:2")
    bars = _bars("AJANTPHARM", "2022-06-22", 1840.80, 1214.80)
    [j] = find_jumps(bars, to_polars([_unread(stored)]))
    assert j.cause == FACTOR_ONE and _found(j) == [("bonus", 1.0)]
    assert find_jumps(bars, to_polars([_unread(stored)]), split_threshold=0.40) == []
    assert find_jumps(bars) == []


def test_a_bonus_closing_at_the_upper_circuit_is_not_a_jump():
    # ANMOL 2023-07-18, Bonus 4:1: 220.20 adjusts to 44.04, and the stock closed 20% up at 52.85.
    actions = to_polars(_feed("ANMOL", "INE02AR01019", "2023-07-18", "Bonus 4:1"))
    assert find_jumps(_bars("ANMOL", "2023-07-18", 220.20 * 0.2, 52.85), actions) == []


def test_a_split_applied_without_its_bonus_leaves_a_residual():
    # HINDZINC 2011-03-07: the parser at 9d0845d read the split from Rs 10 to Rs 2 and not the bonus.
    [split] = _feed(
        "HINDZINC",
        "INE267A01017",
        "2011-03-07",
        "Bonus - 1:1 And Face Value Split From Rs. 10 To Rs. 2",
    )
    actions = to_polars([replace(split, ratio_num=None, ratio_den=None)])
    [j] = find_jumps(_bars("HINDZINC", "2011-03-07", 1343.25 * 0.2, 130.80), actions)
    assert j.cause == RESIDUAL and not j.explained
    assert _found(j) == [("split", 0.2)]


def test_every_action_on_the_date_counts_and_is_found_by_isin():
    # MOTHERSON's 2015 line, filed under its later symbol while the shares traded as MOTHERSUMI. The
    # parser now writes a bonus and a dividend row; the one at 9d0845d read neither.
    subject = "Annual General Meeting / Dividend - Rs 3/- Per Share / Bonus - 1:2"
    isin = "INE775A01035"
    bonus, dividend = _feed("MOTHERSON", isin, "2015-07-23", subject)
    bars = _bars("MOTHERSUMI", "2015-07-23", 523.20, 349.55, isin=isin).with_columns(
        close=pl.col("adj_close"), adj_close=pl.col("adj_close") * 0.1
    )
    dividend_factor = (523.20 - 3) / 523.20
    [j] = find_jumps(bars, to_polars([_unread(bonus), dividend]))
    assert j.cause == FACTOR_ONE
    assert _found(j) == [("bonus", 1.0), ("dividend", pytest.approx(dividend_factor))]
    # Both read, on a series adjusted before they were: the step is left over.
    actions = to_polars([bonus, dividend])
    [j] = find_jumps(bars, actions)
    assert j.cause == RESIDUAL
    assert _found(j) == [
        ("bonus", pytest.approx(2 / 3)),
        ("dividend", pytest.approx(dividend_factor)),
    ]
    for isins in ([isin, isin, None, None], [None, None, isin, isin]):
        [j] = find_jumps(bars.with_columns(pl.Series("isin", isins)), actions)
        assert j.cause == RESIDUAL
    # Matched on the symbol alone, nothing is on the date, and 33% is no jump there.
    assert find_jumps(bars.drop("isin"), to_polars([_unread(bonus), dividend])) == []


@pytest.mark.parametrize(
    ("symbol", "isin", "ex_date", "subject", "before", "after"),
    [
        ("SIEMENS", "INE003A01024", "2025-04-07", "Demerger", 4928.15, 2812.45),
        ("ADANIENT", "INE423A01024", "2015-06-03", "Scheme Of Arrangement", 637.00, 109.75),
        ("PRAXIS", "INE546Y01022", "2021-09-07", "Rights 11:8 @ Premium Rs 6.35/-", 55.25, 31.30),
    ],
)
def test_events_left_unadjusted_by_design_are_explained(
    symbol, isin, ex_date, subject, before, after
):
    [j] = find_jumps(
        _bars(symbol, ex_date, before, after), to_polars(_feed(symbol, isin, ex_date, subject))
    )
    assert j.cause == NOT_ADJUSTED and j.explained
    assert [a.raw_subject for a in j.actions] == [subject]


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
