"""Instrument chains across ISIN changes."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pipeline.actions.back_adjust import back_adjust, compute_action_factors
from pipeline.instrument import attach_key, chain_map, remap_isin, restore_isin
from pipeline.metrics.returns import compute_returns
from pipeline.universe import build_universe


def _days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _prices(rows):
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8, "isin": pl.Utf8, "close": pl.Float64}, orient="row")


def test_chain_links_same_symbol_new_isin_within_gap():
    d = _days(date(2024, 1, 1), 6)
    prices = _prices(
        [(d[i], "NESTLEIND", "OLD", 100.0) for i in range(3)]
        + [(d[i], "NESTLEIND", "NEW", 10.0) for i in range(3, 6)]
        + [(d[i], "OTHER", "ZZZ", 1.0) for i in range(6)]
    )
    cm = chain_map(prices)
    got = dict(zip(cm["isin"], cm["chain_isin"]))
    assert got == {"OLD": "NEW", "NEW": "NEW", "ZZZ": "ZZZ"}


def test_chain_does_not_link_reused_symbol_after_long_gap():
    prices = _prices(
        [(date(2012, 1, 2), "ABC", "ISIN1", 1.0), (date(2012, 1, 3), "ABC", "ISIN1", 1.0)]
        + [(date(2015, 6, 1), "ABC", "ISIN2", 1.0), (date(2015, 6, 2), "ABC", "ISIN2", 1.0)]
    )
    cm = chain_map(prices)
    got = dict(zip(cm["isin"], cm["chain_isin"]))
    assert got == {"ISIN1": "ISIN1", "ISIN2": "ISIN2"}


def test_chain_of_three_resolves_to_latest():
    d = _days(date(2010, 1, 4), 9)
    prices = _prices(
        [(d[i], "HDFCBANK", "A", 1.0) for i in range(3)]
        + [(d[i], "HDFCBANK", "B", 1.0) for i in range(3, 6)]
        + [(d[i], "HDFCBANK", "C", 1.0) for i in range(6, 9)]
    )
    cm = chain_map(prices)
    assert set(cm["chain_isin"]) == {"C"}


def test_attach_key_falls_back_to_isin_then_symbol():
    df = pl.DataFrame({"isin": ["OLD", "X", None, ""], "symbol": ["N", "X", "S1", "S2"]})
    chain = pl.DataFrame({"isin": ["OLD"], "chain_isin": ["NEW"]})
    assert attach_key(df, chain)["_key"].to_list() == ["NEW", "X", "sym:S1", "sym:S2"]
    assert "chain_isin" not in attach_key(df, chain).columns


def test_back_adjust_carries_split_and_later_dividend_across_isin_change():
    # Nestle shape: 1:10 split issues a new ISIN on the ex date, a dividend
    # follows on the new ISIN. Pre-split rows must carry both factors.
    d = _days(date(2024, 1, 1), 8)
    prices = pl.DataFrame(
        [(d[i], "NESTLEIND", "OLD", 27000.0) for i in range(4)]
        + [(d[i], "NESTLEIND", "NEW", 2700.0) for i in range(4, 8)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "isin": pl.Utf8, "close": pl.Float64},
        orient="row",
    )
    actions = pl.DataFrame(
        {
            "exchange": ["NSE", "NSE"],
            "symbol": ["NESTLEIND", "NESTLEIND"],
            "isin": ["OLD", "NEW"],
            "company": ["Nestle", "Nestle"],
            "ex_date": [d[4], d[6]],
            "record_date": [None, None],
            "type": ["split", "dividend"],
            "ratio_num": [None, None],
            "ratio_den": [None, None],
            "cash_amount": [None, 27.0],
            "face_value_from": [10.0, None],
            "face_value_to": [1.0, None],
            "raw_subject": ["split", "div"],
        },
        schema_overrides={"ratio_num": pl.Int64, "ratio_den": pl.Int64, "record_date": pl.Date},
    )
    chain = chain_map(prices)

    # Without chains: the old ISIN never sees the dividend, and the factor
    # is discontinuous at the split.
    naive = back_adjust(prices, compute_action_factors(actions, prices)).sort("date")
    assert naive["adj_factor_cumulative"][3] == pytest.approx(0.1)
    assert naive["adj_factor_cumulative"][4] == pytest.approx(0.99)  # (2700-27)/2700

    # With chains: pre-split factor = split x dividend, monotone through the change.
    factors = compute_action_factors(remap_isin(actions, chain).drop("_isin_raw"), remap_isin(prices, chain))
    adj = restore_isin(back_adjust(remap_isin(prices, chain), factors)).sort("date")
    f = adj["adj_factor_cumulative"].to_list()
    assert f[3] == pytest.approx(0.1 * 0.99)
    assert f[4] == pytest.approx(0.99)
    assert f[6] == pytest.approx(1.0)
    assert all(a <= b + 1e-12 for a, b in zip(f, f[1:]))
    # adj_close is continuous across the split: 27000 * 0.099 == 2700 * 0.99
    assert adj["adj_close"][3] == pytest.approx(adj["adj_close"][4])
    # Raw ISINs come back untouched.
    assert adj["isin"].to_list() == ["OLD"] * 4 + ["NEW"] * 4
    assert "_isin_raw" not in adj.columns


def test_returns_continuous_across_chain():
    d = _days(date(2024, 1, 1), 6)
    adjusted = pl.DataFrame(
        {
            "date": d,
            "symbol": ["N"] * 6,
            "isin": ["OLD"] * 3 + ["NEW"] * 3,
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    chain = chain_map(adjusted)
    r = compute_returns(attach_key(adjusted, chain)).sort("date")
    assert r["ret_1d"][3] == pytest.approx(103.0 / 102.0 - 1)
    # Same rows without the chain restart the series at the ISIN change.
    r0 = compute_returns(adjusted).sort("date")
    assert r0["ret_1d"][3] is None


def test_universe_eligibility_survives_isin_change():
    d = _days(date(2024, 1, 1), 40)
    rows = []
    for i, day in enumerate(d):
        isin = "OLD" if i < 20 else "NEW"
        rows.append((day, "N", isin, "Nestle", 1e9))
        rows.append((day, "M", "MMM", "Other", 1e8))
    prices = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8, "isin": pl.Utf8, "name": pl.Utf8, "turnover": pl.Float64}, orient="row")
    with_chain = build_universe(prices, "NSE", top_n=5, window=10, chain=chain_map(prices))
    without = build_universe(prices, "NSE", top_n=5, window=10)
    feb = date(2024, 2, 1)
    assert "N" in with_chain.filter(pl.col("rebalance_date") == feb)["symbol"].to_list()
    assert "N" not in without.filter(pl.col("rebalance_date") == feb)["symbol"].to_list()
