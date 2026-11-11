from datetime import date

import polars as pl

from pipeline.fo import FO_SCHEMA
from pipeline.fo_stats import compute_fo_stats

D1, D2, E1, E2 = date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 26), date(2025, 7, 31)


def _rows(rows: list[dict]) -> pl.DataFrame:
    base = {c: None for c in FO_SCHEMA}
    return pl.DataFrame([{**base, **r} for r in rows], schema=FO_SCHEMA)


def _opt(d, sym, exp, k, ot, oi, contracts=1, inst="OPTSTK"):
    return dict(date=d, instrument=inst, symbol=sym, expiry=exp, strike=k, option_type=ot,
                open_interest=oi, oi_change=0, contracts=contracts)


def _fut(d, sym, exp, settle, oi, oi_change, inst="FUTSTK", underlying=None):
    return dict(date=d, instrument=inst, symbol=sym, expiry=exp, settle_price=settle,
                open_interest=oi, oi_change=oi_change, contracts=5, underlying=underlying)


def _brute_max_pain(chain: list[tuple[float, int, int]]) -> float:
    def pain(s):
        return sum(c * max(0.0, s - k) + p * max(0.0, k - s) for k, c, p in chain)
    return min(sorted(k for k, _, _ in chain), key=pain)


def test_pcr_and_max_pain_match_brute_force():
    chain = [(90.0, 10, 500), (100.0, 300, 400), (110.0, 600, 100), (120.0, 200, 0)]
    rows = []
    for k, c, p in chain:
        rows += [_opt(D1, "ABC", E1, k, "CE", c, contracts=2), _opt(D1, "ABC", E1, k, "PE", p, contracts=1)]
    s = compute_fo_stats(_rows(rows)).row(0, named=True)
    assert s["ce_oi"] == 1110 and s["pe_oi"] == 1000
    assert abs(s["pcr_oi"] - 1000 / 1110) < 1e-12
    assert s["pcr_volume"] == 0.5
    assert s["max_pain"] == _brute_max_pain(chain)
    assert s["strikes"] == 4
    assert s["kind"] == "stock"


def test_legacy_ca_pa_count_as_ce_pe():
    s = compute_fo_stats(_rows([_opt(D1, "ABC", E1, 100.0, "CA", 10), _opt(D1, "ABC", E1, 100.0, "PA", 30)])).row(0, named=True)
    assert s["ce_oi"] == 10 and s["pe_oi"] == 30 and s["pcr_oi"] == 3.0


def test_buildup_rollover_basis_and_since():
    rows = [
        _fut(D1, "ABC", E1, 100.0, 1000, 0), _fut(D1, "ABC", E2, 101.0, 100, 0),
        _fut(D2, "ABC", E1, 102.0, 900, -100, underlying=101.0),
        _fut(D2, "ABC", E2, 100.5, 300, 200),
    ]
    out = compute_fo_stats(_rows(rows), since=D2)
    assert out["date"].unique().to_list() == [D2]
    near = out.filter(pl.col("expiry") == E1).row(0, named=True)
    far = out.filter(pl.col("expiry") == E2).row(0, named=True)
    assert near["buildup"] == "short_covering" and near["fut_settle_change"] == 2.0
    assert far["buildup"] == "short_buildup"
    assert near["basis"] == 1.0
    assert near["rollover_pct"] == 25.0 and far["rollover_pct"] == 25.0
    assert near["is_nearest"] and not far["is_nearest"]


def test_rollover_ignores_weekly_option_only_expiry():
    weekly = date(2025, 6, 5)
    rows = [_opt(D1, "NIFTY", weekly, 100.0, "CE", 5, inst="OPTIDX"),
            _fut(D1, "NIFTY", E1, 100.0, 800, 0, inst="FUTIDX"), _fut(D1, "NIFTY", E2, 100.0, 200, 0, inst="FUTIDX")]
    out = compute_fo_stats(_rows(rows))
    assert out.filter(pl.col("expiry") == weekly)["is_nearest"].item()
    assert out["rollover_pct"].unique().to_list() == [20.0]
    assert out["kind"].unique().to_list() == ["index"]


def test_empty_input():
    assert compute_fo_stats(_rows([])).is_empty()
