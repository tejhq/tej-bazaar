from datetime import date

import polars as pl

from pipeline.isin_backfill import apply_isin_anchors, isin_anchors, series_key


def _frame():
    rows = [
        # last legacy day: AAA, BBB, GONE trade without ISIN
        (date(2011, 12, 30), "AAA", None), (date(2011, 12, 30), "BBB", None), (date(2011, 12, 30), "GONE", None),
        # first ISIN day: AAA, BBB continue; NEWCO appears; GONE is absent
        (date(2012, 1, 2), "AAA", "INAAA"), (date(2012, 1, 2), "BBB", "INBBB"), (date(2012, 1, 2), "NEWCO", "INNEW"),
        # earlier legacy rows
        (date(2010, 5, 3), "AAA", None), (date(2010, 5, 3), "GONE", None),
        # a later reuse of GONE's ticker by a different company must not leak backwards
        (date(2015, 3, 2), "GONE", "INREUSED"),
    ]
    return pl.DataFrame(rows, schema=["date", "symbol", "isin"], orient="row")


def test_anchors_require_continuity_across_cutover():
    a = isin_anchors(_frame())
    assert a.to_dicts() == [{"symbol": "AAA", "isin": "INAAA"}, {"symbol": "BBB", "isin": "INBBB"}]


def test_apply_fills_only_null_rows_of_anchored_symbols():
    f = _frame()
    out = apply_isin_anchors(f, isin_anchors(f)).sort(["date", "symbol"])
    got = {(r["date"], r["symbol"]): r["isin"] for r in out.to_dicts()}
    assert got[(date(2010, 5, 3), "AAA")] == "INAAA"
    assert got[(date(2011, 12, 30), "BBB")] == "INBBB"
    assert got[(date(2010, 5, 3), "GONE")] is None
    assert got[(date(2015, 3, 2), "GONE")] == "INREUSED"
    assert got[(date(2012, 1, 2), "NEWCO")] == "INNEW"


def test_no_cutover_means_no_anchors():
    all_have = pl.DataFrame({"date": [date(2020, 1, 1)], "symbol": ["A"], "isin": ["X"]})
    assert isin_anchors(all_have).is_empty()
    assert apply_isin_anchors(all_have, isin_anchors(all_have)).equals(all_have)


def test_series_key_uses_symbol_when_isin_missing():
    f = pl.DataFrame({"symbol": ["A", "B"], "isin": [None, "X"]})
    assert f.with_columns(series_key().alias("k"))["k"].to_list() == ["sym:A", "X"]
