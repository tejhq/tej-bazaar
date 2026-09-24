from datetime import date

import polars as pl

from pipeline.actions.audit import find_jumps
from pipeline.actions.derive import DERIVED_NOTE, derive_splits, is_derived, merge_derived
from pipeline.actions.factors import compute_factor
from pipeline.actions.schema import ACTION_SCHEMA


def _prices(symbol: str, closes: list[float], start: date = date(2020, 1, 1)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [start.replace(day=start.day + i) for i in range(len(closes))],
            "symbol": [symbol] * len(closes),
            "adj_close": [float(c) for c in closes],
        }
    )


def _actions(rows: list[tuple[str, date, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "exchange": "NSE", "symbol": s, "isin": None, "company": s, "ex_date": d,
                "record_date": None, "type": t, "ratio_num": None, "ratio_den": None,
                "cash_amount": None, "face_value_from": None, "face_value_to": None,
                "raw_subject": t,
            }
            for s, d, t in rows
        ],
        schema=ACTION_SCHEMA,
    )


def test_an_unannounced_ten_to_one_split_is_derived():
    jumps = find_jumps(_prices("NIFTYBEES", [1292.54, 1290.0, 130.20, 131.0]))
    derived = derive_splits(jumps)
    assert len(derived) == 1
    a = derived[0]
    assert a.type == "split" and (a.face_value_from, a.face_value_to) == (10.0, 1.0)
    assert abs(compute_factor(a) - 0.1) < 1e-9
    assert is_derived(a.raw_subject)


def test_a_reverse_split_derives_the_other_way():
    jumps = find_jumps(_prices("CONSOL", [10.0, 10.1, 101.0, 102.0]))
    a = derive_splits(jumps)[0]
    assert (a.face_value_from, a.face_value_to) == (1.0, 10.0)
    assert abs(compute_factor(a) - 10.0) < 1e-9


def test_a_nearby_demerger_is_never_overridden():
    # A clean 10:1 step that would otherwise be derived, with a demerger announced two days later.
    # The pipeline does not scale demergers by design, so the step must be left alone.
    jumps = find_jumps(_prices("SPLITCO", [1000.0, 990.0, 99.0, 98.0]))
    assert len(jumps) == 1 and not jumps[0].explained
    assert len(derive_splits(jumps)) == 1
    acts = _actions([("SPLITCO", date(2020, 1, 5), "demerger")])
    assert derive_splits(jumps, acts) == []


def test_a_move_just_inside_the_threshold_is_not_touched():
    # Tata Motors fell 40% on its demerger; a fraction under the bar must stay out of scope.
    assert find_jumps(_prices("BORDERLINE", [660.75, 660.75, 396.45, 396.0])) == []


def test_a_crash_that_is_not_a_clean_ratio_is_left_alone():
    # Down 55%, which no standard split produces.
    assert derive_splits(find_jumps(_prices("CRASH", [100.0, 101.0, 45.0, 44.0]))) == []


def test_merging_is_idempotent():
    jumps = find_jumps(_prices("NIFTYBEES", [1292.54, 1290.0, 130.20, 131.0]))
    derived = derive_splits(jumps)
    empty = pl.DataFrame([], schema=ACTION_SCHEMA)
    once = merge_derived(empty, derived)
    twice = merge_derived(once, derived)
    assert once.height == 1 and twice.height == 1
    assert twice["raw_subject"][0].startswith(DERIVED_NOTE)


def test_exchange_announcements_survive_a_merge():
    real = _actions([("NIFTYBEES", date(2019, 5, 1), "dividend")])
    derived = derive_splits(find_jumps(_prices("NIFTYBEES", [1292.54, 1290.0, 130.20, 131.0])))
    merged = merge_derived(real, derived)
    assert merged.height == 2
    assert sorted(merged["type"].to_list()) == ["dividend", "split"]


def test_a_ten_to_one_split_with_a_few_percent_of_real_move_is_still_derived():
    # SILVER1 fell 89.7%, not exactly 90%: the unit split and the price also moved that day.
    a = derive_splits(find_jumps(_prices("SILVER1", [249.55, 249.55, 25.75, 26.0])))
    assert len(a) == 1 and (a[0].face_value_from, a[0].face_value_to) == (10.0, 1.0)


def test_a_rights_entitlement_halving_is_never_called_a_split():
    # A -RE instrument halves and expires as a matter of course; it carries no circuit band.
    assert derive_splits(find_jumps(_prices("SEPC-RE3", [2.71, 2.71, 1.35, 1.30]))) == []
    assert derive_splits(find_jumps(_prices("PRAXIS-RE2", [0.40, 0.40, 0.04, 0.03]))) == []


def test_an_ordinary_equity_halving_with_no_announcement_is_derived():
    # Circuit bands make a genuine 50% single day fall close to impossible, so with no action
    # within five days this is a split the exchange never published.
    a = derive_splits(find_jumps(_prices("HALVED", [100.0, 100.0, 50.0, 49.0])))
    assert len(a) == 1 and (a[0].face_value_from, a[0].face_value_to) == (2.0, 1.0)


def test_a_penny_stock_oscillating_on_the_tick_is_never_derived():
    # VISESHINFO bounced between 5 and 10 paise for years; one tick is a 100% move there.
    closes = [0.10, 0.05, 0.10, 0.05, 0.10, 0.05, 0.10, 0.05]
    assert derive_splits(find_jumps(_prices("VISESHINFO", closes))) == []


def test_a_symbol_with_a_stream_of_steps_is_dropped_entirely():
    # Four clean 10:1 steps from one name is an oscillating series, not four splits.
    closes = [1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0]
    assert derive_splits(find_jumps(_prices("BOUNCY", closes))) == []
