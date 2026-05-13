from datetime import date

import pytest

from pipeline.actions import CorporateAction
from pipeline.actions.factors import compute_factor, needs_prev_close


def _make(type_, **kw) -> CorporateAction:
    base = dict(
        exchange="NSE",
        symbol="X",
        isin="INE000A00001",
        company="X Ltd",
        ex_date=date(2024, 1, 1),
        record_date=None,
        type=type_,
    )
    base.update(kw)
    return CorporateAction(**base)


# --- split (forward) -----------------------------------------------------


def test_split_forward_10_to_1():
    a = _make("split", face_value_from=10.0, face_value_to=1.0)
    # Pre-split price gets multiplied by 0.1 to align with post-split
    assert compute_factor(a) == pytest.approx(0.1)


def test_split_forward_10_to_2():
    a = _make("split", face_value_from=10.0, face_value_to=2.0)
    assert compute_factor(a) == pytest.approx(0.2)


# --- split (reverse / consolidation) -------------------------------------


def test_split_reverse_consolidation():
    # Consolidation Re 1 -> Rs 10: pre-consolidation prices need to scale UP
    a = _make("split", face_value_from=1.0, face_value_to=10.0)
    assert compute_factor(a) == pytest.approx(10.0)


def test_split_missing_face_values_returns_one():
    # BSE consolidation often has no face values in text -> pass through
    a = _make("split", face_value_from=None, face_value_to=None)
    assert compute_factor(a) == 1.0


def test_split_zero_fv_from_returns_one():
    a = _make("split", face_value_from=0.0, face_value_to=10.0)
    assert compute_factor(a) == 1.0


# --- bonus --------------------------------------------------------------


def test_bonus_1_to_1():
    # 1 free share per 1 held: total shares doubles, price halves
    a = _make("bonus", ratio_num=1, ratio_den=1)
    assert compute_factor(a) == pytest.approx(0.5)


def test_bonus_3_to_1():
    # 3 free per 1: 4x shares, factor = 1/4
    a = _make("bonus", ratio_num=3, ratio_den=1)
    assert compute_factor(a) == pytest.approx(0.25)


def test_bonus_1_to_2():
    # 1 free per 2 held: 3 shares for every 2, factor = 2/3
    a = _make("bonus", ratio_num=1, ratio_den=2)
    assert compute_factor(a) == pytest.approx(2 / 3)


def test_bonus_high_ratio_1_to_50():
    # BSE-observed pattern: 1 bonus per 50 (tiny dilution)
    a = _make("bonus", ratio_num=1, ratio_den=50)
    assert compute_factor(a) == pytest.approx(50 / 51)


def test_bonus_missing_ratio_returns_one():
    a = _make("bonus", ratio_num=None, ratio_den=None)
    assert compute_factor(a) == 1.0


# --- dividend -----------------------------------------------------------


def test_dividend_simple():
    a = _make("dividend", cash_amount=10.0)
    # Close 200, div 10 -> factor (200-10)/200 = 0.95
    assert compute_factor(a, prev_close=200.0) == pytest.approx(0.95)


def test_dividend_re_paisa_amount():
    # Re 0.01 dividend on Rs 100 close -> factor 99.99/100
    a = _make("dividend", cash_amount=0.01)
    assert compute_factor(a, prev_close=100.0) == pytest.approx(0.9999)


def test_dividend_missing_prev_close_returns_one():
    a = _make("dividend", cash_amount=10.0)
    assert compute_factor(a, prev_close=None) == 1.0


def test_dividend_missing_amount_returns_one():
    a = _make("dividend", cash_amount=None)
    assert compute_factor(a, prev_close=100.0) == 1.0


def test_dividend_exceeds_close_clamps_to_one():
    # Defensive: if div > close (data error), don't produce negative factor
    a = _make("dividend", cash_amount=200.0)
    assert compute_factor(a, prev_close=100.0) == 1.0


def test_dividend_equals_close_clamps_to_one():
    # factor would be exactly 0 -> useless, pass through
    a = _make("dividend", cash_amount=100.0)
    assert compute_factor(a, prev_close=100.0) == 1.0


def test_dividend_zero_prev_close_returns_one():
    a = _make("dividend", cash_amount=5.0)
    assert compute_factor(a, prev_close=0.0) == 1.0


# --- no-impact action types --------------------------------------------


def test_buyback_no_adjustment():
    a = _make("buyback", cash_amount=None)
    assert compute_factor(a) == 1.0


def test_agm_no_adjustment():
    a = _make("agm")
    assert compute_factor(a) == 1.0


def test_rights_no_adjustment_v1():
    # Rights math deferred; v1 leaves prices untouched
    a = _make("rights", ratio_num=6, ratio_den=179)
    assert compute_factor(a, prev_close=1000.0) == 1.0


def test_merger_demerger_other_no_adjustment():
    for t in ("merger", "demerger", "other"):
        assert compute_factor(_make(t)) == 1.0


# --- needs_prev_close --------------------------------------------------


def test_needs_prev_close_for_dividend_with_amount():
    assert needs_prev_close(_make("dividend", cash_amount=5.0)) is True


def test_needs_prev_close_false_for_dividend_without_amount():
    assert needs_prev_close(_make("dividend", cash_amount=None)) is False


def test_needs_prev_close_false_for_ratio_actions():
    assert needs_prev_close(_make("split", face_value_from=10.0, face_value_to=1.0)) is False
    assert needs_prev_close(_make("bonus", ratio_num=1, ratio_den=1)) is False
    assert needs_prev_close(_make("buyback")) is False
