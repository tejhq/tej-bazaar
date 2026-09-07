import json
from datetime import date
from pathlib import Path

import polars as pl

from pipeline.export_json import export_api_json, export_ohlcv, export_snapshots, export_actions, read_bhavcopy


def _prices() -> pl.DataFrame:
    return pl.DataFrame({
        "date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3), date(2026, 1, 2)],
        "symbol": ["AAA", "BBB", "AAA", "AAA"],
        "series": ["EQ", "EQ", "EQ", "EQ"],
        "isin": ["INE1", None, "INE1", "INE1"],
        "name": ["Aaa Ltd", None, "Aaa Ltd", "Aaa Ltd"],
        "open": [10.0, 20.0, 11.0, 12.0],
        "high": [10.5, 20.5, 11.5, 12.5],
        "low": [9.5, 19.5, 10.5, 11.5],
        "close": [10.2, 20.2, 11.2, 12.2],
        "last": [10.2, 20.2, 11.2, 12.2],
        "prev_close": [9.9, 19.9, 10.2, 11.2],
        "volume": [100, 200, 110, 120],
        "turnover": [1000.0, 4000.0, 1200.0, 1400.0],
        "trades": [5, None, 6, 7],
    })


def _actions() -> pl.DataFrame:
    return pl.DataFrame({
        "exchange": ["NSE", "NSE"],
        "symbol": ["AAA", "AAA"],
        "isin": ["INE1", "INE1"],
        "company": ["Aaa Ltd", "Aaa Ltd"],
        "ex_date": [date(2024, 5, 1), date(2025, 5, 1)],
        "record_date": [None, date(2025, 5, 2)],
        "type": ["dividend", "split"],
        "ratio_num": [None, None],
        "ratio_den": [None, None],
        "cash_amount": [2.5, None],
        "face_value_from": [None, 10.0],
        "face_value_to": [None, 1.0],
        "raw_subject": ["Dividend Rs 2.5", "Split 10 to 1"],
    })


def test_ohlcv_one_file_per_symbol_sorted_and_null_trades_zeroed(tmp_path: Path):
    n = export_ohlcv(_prices(), "NSE", tmp_path)
    assert n == 2
    aaa = json.loads((tmp_path / "api/v1/ohlcv/nse/AAA.json").read_text())["data"]
    assert [r["date"] for r in aaa] == ["2025-01-02", "2025-01-03", "2026-01-02"]
    assert list(aaa[0].keys()) == ["date", "open", "high", "low", "close", "last", "prev_close", "volume", "turnover", "trades"]
    bbb = json.loads((tmp_path / "api/v1/ohlcv/nse/BBB.json").read_text())["data"]
    assert bbb[0]["trades"] == 0


def test_snapshot_filters_years_and_coalesces_isin_name(tmp_path: Path):
    n = export_snapshots(_prices(), "NSE", tmp_path, years=[2025])
    assert n == 2
    assert not (tmp_path / "api/v1/snapshot/nse/2026-01-02.json").exists()
    rows = json.loads((tmp_path / "api/v1/snapshot/nse/2025-01-02.json").read_text())["data"]
    assert [r["symbol"] for r in rows] == ["AAA", "BBB"]
    assert rows[1]["isin"] == "" and rows[1]["name"] == "" and rows[1]["trades"] == 0
    assert export_snapshots(_prices(), "NSE", tmp_path, years=None) == 3


def test_actions_desc_and_optional_nulls_omitted(tmp_path: Path):
    assert export_actions(_actions(), tmp_path) == 1
    rows = json.loads((tmp_path / "api/v1/actions/AAA.json").read_text())["data"]
    assert [r["ex_date"] for r in rows] == ["2025-05-01", "2024-05-01"]
    assert "record_date" not in rows[1] and "cash_amount" not in rows[0]
    assert rows[1]["cash_amount"] == 2.5 and rows[0]["record_date"] == "2025-05-02"
    assert rows[0]["face_value_from"] == 10.0


def test_read_bhavcopy_mixes_rollup_and_daily(tmp_path: Path):
    p = _prices()
    p.with_columns(pl.lit(2025, dtype=pl.Int32).alias("year"), pl.lit(1, dtype=pl.Int8).alias("month")).write_parquet(tmp_path / "rollup.parquet")
    p.head(1).write_parquet(tmp_path / "daily.parquet")
    df = read_bhavcopy([tmp_path / "rollup.parquet", tmp_path / "daily.parquet"])
    assert df.height == 5 and "year" not in df.columns


def test_export_api_json_end_to_end(tmp_path: Path):
    (tmp_path / "out/nse/year=2025").mkdir(parents=True)
    _prices().write_parquet(tmp_path / "out/nse/year=2025/nse_2025.parquet")
    (tmp_path / "out/actions").mkdir()
    _actions().write_parquet(tmp_path / "out/actions/nse_2025.parquet")
    res = export_api_json(
        prices_dir=tmp_path / "out", actions_dir=tmp_path / "out/actions",
        out_dir=tmp_path / "api", exchanges=["NSE"], snapshot_years=[2026],
    )
    assert (res.ohlcv_files, res.snapshot_files, res.action_files) == (2, 1, 1)


def test_export_only_actions_skips_prices(tmp_path: Path):
    (tmp_path / "out/actions").mkdir(parents=True)
    _actions().write_parquet(tmp_path / "out/actions/nse_2025.parquet")
    res = export_api_json(
        prices_dir=tmp_path / "out", actions_dir=tmp_path / "out/actions",
        out_dir=tmp_path / "api", exchanges=["NSE"], snapshot_years=None, only=["actions"],
    )
    assert (res.ohlcv_files, res.snapshot_files, res.action_files) == (0, 0, 1)
