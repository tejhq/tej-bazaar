from datetime import date, timedelta

import polars as pl

from pipeline.universe import build_universe, members_as_of


def _prices(n_days=140, symbols=("AAA", "BBB", "CCC", "DDD")):
    # Weekdays only, starting 2025-01-01. Turnover: AAA >> BBB > CCC > DDD,
    # except DDD delists after day 100 and CCC spikes late.
    rows = []
    d = date(2025, 1, 1)
    i = 0
    while i < n_days:
        if d.weekday() < 5:
            for s in symbols:
                base = {"AAA": 1e9, "BBB": 5e8, "CCC": 1e8, "DDD": 2e8}[s]
                if s == "DDD" and i >= 100:
                    d += timedelta(days=1); continue
                if s == "CCC" and i >= 80:
                    base = 8e8
                rows.append({"date": d, "symbol": s, "series": "EQ", "isin": f"IN{s}" if s != "AAA" else None,
                             "name": f"{s} Ltd", "turnover": base, "volume": 1, "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 1.0, "last": 1.0, "prev_close": 1.0, "trades": 1})
            i += 1
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def test_monthly_rebalance_ranks_and_validity():
    u = build_universe(_prices(), "nse", top_n=3)
    assert set(u.columns) == {"exchange", "rebalance_date", "valid_to", "rank", "symbol", "isin", "name", "avg_turnover_63d"}
    # 63-day window: first eligible rebalance is the first trading day of the month after ~3 months.
    first = u["rebalance_date"].min()
    assert first >= date(2025, 4, 1)
    # Each rebalance has ranks 1..3 and valid_to = day before next rebalance.
    per = u.group_by("rebalance_date").agg(pl.col("rank").sort(), pl.col("valid_to").first()).sort("rebalance_date")
    assert all(r == [1, 2, 3] for r in per["rank"].to_list())
    rbs = per["rebalance_date"].to_list(); vts = per["valid_to"].to_list()
    for a, b, v in zip(rbs, rbs[1:], vts):
        assert v == b - timedelta(days=1)
    # AAA (no ISIN, keyed by symbol) is always rank 1.
    assert (u.filter(pl.col("rank") == 1)["symbol"] == "AAA").all()
    assert (u.filter(pl.col("symbol") == "AAA")["isin"] == "").all()


def test_survivorship_delisted_stays_in_past_months_and_spike_enters_later():
    u = build_universe(_prices(), "nse", top_n=3)
    early = members_as_of(u, date(2025, 4, 15), 3)
    late = members_as_of(u, u["rebalance_date"].max(), 3)
    assert "DDD" in early["symbol"].to_list()
    assert "DDD" not in late["symbol"].to_list()
    assert "CCC" in late["symbol"].to_list()


def test_members_as_of_picks_latest_rebalance_and_rank_filter():
    u = build_universe(_prices(), "nse", top_n=3)
    rb = sorted(u["rebalance_date"].unique().to_list())
    mid = rb[1] + timedelta(days=10)
    m = members_as_of(u, mid, 2)
    assert m["rebalance_date"].unique().to_list() == [rb[1]]
    assert m["rank"].to_list() == [1, 2]
    assert members_as_of(u, date(2020, 1, 1), 3).is_empty()


def test_empty_and_short_history():
    assert build_universe(pl.DataFrame(), "nse").is_empty()
    assert build_universe(_prices(n_days=30), "nse").is_empty()
