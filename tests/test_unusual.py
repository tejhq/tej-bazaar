from datetime import date, timedelta

import polars as pl

from pipeline.unusual import compute_unusual

N = 300
DAYS = [date(2025, 1, 1) + timedelta(days=i) for i in range(N)]
LAST = DAYS[-1]


def _prices(sym="ABC", isin="INE000000001", factor_step_at=None, spike_at=None, close=100.0, turnover=5e7):
    rows = []
    for i, d in enumerate(DAYS):
        wiggle = 1 + 0.004 * (1 if i % 2 else -1)
        vol = 100_000 * (1 + 0.05 * (i % 3))
        f = 1.0
        if factor_step_at is not None and i < factor_step_at:
            f = 0.1
        c = close * wiggle / (f if factor_step_at is not None and i < factor_step_at else 1)
        if factor_step_at is not None and i >= factor_step_at:
            vol *= 10
        if spike_at is not None and i == spike_at:
            vol *= 12
            c *= 1.08
        rows.append(dict(date=d, symbol=sym, series="EQ", isin=isin, open=c, close=c, volume=int(vol),
                         turnover=turnover, adj_factor_cumulative=f, adj_close=c * f))
    return pl.DataFrame(rows)


def _signals(df, day=LAST, **kw):
    out = compute_unusual(df, since=DAYS[260], **kw)
    return set(out.filter(pl.col("date") == day)["signal"].to_list())


def test_volume_and_move_spike_flagged():
    sig = _signals(_prices(spike_at=N - 1))
    assert {"volume", "move"} <= sig


def test_quiet_day_has_no_flags():
    assert _signals(_prices()) == set()


def test_split_day_is_not_a_spike():
    df = _prices(factor_step_at=N - 1)
    assert not ({"volume", "move", "gap"} & _signals(df))


def test_low_turnover_and_fund_units_skipped():
    assert _signals(_prices(spike_at=N - 1, turnover=1e5)) == set()
    assert _signals(_prices(spike_at=N - 1, isin="INF000000001")) == set()


def _fo(sym="ABC", jump_at=None, start=0):
    rows = []
    oi = 1_000_000
    for i, d in enumerate(DAYS[start:], start=start):
        ch = 10_000 if i % 2 else -10_000
        if jump_at is not None and i == jump_at:
            ch = 5_000_000
        oi += ch
        rows.append(dict(date=d, symbol=sym, expiry=date(2026, 12, 31), fut_oi=oi, fut_oi_change=ch))
    return pl.DataFrame(rows)


def test_fut_oi_jump_flagged_but_not_near_split_or_new_listing():
    assert "fut_oi" in _signals(_prices(), fo_stats=_fo(jump_at=N - 1))
    assert "fut_oi" not in _signals(_prices(factor_step_at=N - 1), fo_stats=_fo(jump_at=N - 1))
    assert "fut_oi" not in _signals(_prices(), fo_stats=_fo(jump_at=N - 1, start=N - 5))


def test_delivery_high_flagged():
    d = pl.DataFrame([dict(date=x, symbol="ABC", series="EQ", delivery_pct=40.0 + (i % 5)) for i, x in enumerate(DAYS)])
    d = d.with_columns(pl.when(pl.col("date") == LAST).then(95.0).otherwise(pl.col("delivery_pct")).alias("delivery_pct"))
    assert "delivery" in _signals(_prices(), delivery=d)


def test_empty():
    assert compute_unusual(_prices().clear()).is_empty()
