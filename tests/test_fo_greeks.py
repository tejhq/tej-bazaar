from datetime import date

import numpy as np
import polars as pl
import pytest

from pipeline.fo_greeks import black76, compute_fo_greeks, expiry_iv, greeks, implied_vol
from pipeline.fo_stats import compute_fo_stats

D, E = date(2026, 10, 30), date(2026, 11, 23)
T = (E - D).days / 365
R = 0.0525


def test_iv_round_trip():
    f, k = np.array([23000.0, 23000.0, 100.0]), np.array([22000.0, 24000.0, 100.0])
    calls = np.array([True, False, True])
    p = black76(f, k, 0.1, 0.05, 0.18, calls)
    assert np.allclose(implied_vol(p, f, k, 0.1, 0.05, calls), 0.18, atol=1e-9)


def test_iv_nan_outside_bounds():
    iv = implied_vol([0.0, 1e9], 100, 100, 0.1, 0.05, [True, False])
    assert np.isnan(iv).all()


def test_greeks_match_finite_differences():
    f, k, s = 23100.0, 23000.0, 0.14
    g = greeks(f, k, T, R, s, True)
    h = 0.01
    assert g["delta"] == pytest.approx(
        (black76(f + h, k, T, R, s, True) - black76(f - h, k, T, R, s, True)) / (2 * h), rel=1e-4
    )
    assert g["vega"] == pytest.approx(
        black76(f, k, T, R, s + 0.005, True) - black76(f, k, T, R, s - 0.005, True), rel=1e-3
    )
    put = greeks(f, k, T, R, s, False)
    assert put["delta"] == pytest.approx(g["delta"] - np.exp(-R * T), rel=1e-9)


def _row(
    instrument,
    strike=None,
    ot=None,
    close=1.0,
    contracts=10,
    expiry=E,
    symbol="NIFTY",
    d=D,
    settle=None,
):
    return {
        "date": d,
        "instrument": instrument,
        "symbol": symbol,
        "expiry": expiry,
        "strike": strike,
        "option_type": ot,
        "close": close,
        "settle_price": settle if settle is not None else close,
        "contracts": contracts,
        "open_interest": 100,
        "oi_change": 0,
        "turnover": 1.0,
        "underlying": None,
    }


def _chain(fwd=23000.0, vol=0.15, expiry=E, with_future=True, symbol="NIFTY"):
    t = (expiry - D).days / 365
    rows = [_row("FUTIDX", close=fwd, expiry=expiry, symbol=symbol)] if with_future else []
    for k in (22000.0, 22500.0, 23000.0, 23500.0, 24000.0):
        for ot, call in (("CE", True), ("PE", False)):
            rows.append(
                _row(
                    "OPTIDX",
                    k,
                    ot,
                    float(black76(fwd, k, t, R, vol, call)),
                    expiry=expiry,
                    symbol=symbol,
                )
            )
    return rows


def test_compute_recovers_vol_from_future_forward():
    g = compute_fo_greeks(pl.DataFrame(_chain()))
    assert g.height == 10
    assert set(g["forward_source"]) == {"future"}
    assert g["iv"].to_numpy() == pytest.approx(np.full(10, 0.15), abs=1e-6)


def test_parity_forward_when_no_future():
    g = compute_fo_greeks(pl.DataFrame(_chain(with_future=False)))
    assert set(g["forward_source"]) == {"parity"}
    assert g["forward"][0] == pytest.approx(23000.0, rel=1e-9)
    assert g["iv"].to_numpy() == pytest.approx(np.full(10, 0.15), abs=1e-6)


def test_untraded_and_expiry_day_rows_skipped():
    rows = _chain()
    rows.append(_row("OPTIDX", 25000.0, "CE", 5.0, contracts=0))
    rows.append(_row("OPTIDX", 23000.0, "CE", 50.0, expiry=D))
    g = compute_fo_greeks(pl.DataFrame(rows))
    assert g.height == 10
    assert 25000.0 not in g["strike"].to_list()


def test_expiry_iv_and_stats_columns():
    df = pl.DataFrame(_chain(vol=0.2))
    e = expiry_iv(compute_fo_greeks(df))
    assert e["atm_iv"][0] == pytest.approx(0.2, abs=1e-6)
    assert e["iv_skew"][0] == pytest.approx(0.0, abs=1e-6)
    s = compute_fo_stats(df, iv=e)
    assert s["atm_iv"][0] == pytest.approx(0.2, abs=1e-6)
    assert compute_fo_stats(df)["atm_iv"].null_count() == 1
