import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from typer.testing import CliRunner

from pipeline import __version__
from pipeline.actions import ACTION_SCHEMA, CorporateAction
from pipeline.cli import app
from pipeline.fetch import BhavcopyNotFoundError
from pipeline.push import partition_path

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("fetch", "backfill", "info", "version"):
        assert cmd in result.stdout


def test_fetch_bad_date_format():
    result = runner.invoke(app, ["fetch", "30-04-2025"])
    assert result.exit_code != 0


def test_info_empty_dir(tmp_path: Path):
    result = runner.invoke(app, ["info", "--data-dir", str(tmp_path / "missing")])
    assert result.exit_code == 0
    assert "no data directory" in result.stdout


def test_backfill_inverted_range_rejected():
    result = runner.invoke(
        app, ["backfill", "--from", "2025-04-30", "--to", "2025-04-01"]
    )
    assert result.exit_code != 0


def test_fetch_help_lists_exchange_flag():
    result = runner.invoke(app, ["fetch", "--help"])
    assert result.exit_code == 0
    assert "--exchange" in result.stdout


def test_fetch_invalid_exchange_rejected():
    result = runner.invoke(app, ["fetch", "2025-04-30", "--exchange", "MCX"])
    assert result.exit_code != 0


def test_publish_dry_run(tmp_path: Path):
    p = tmp_path / "nse" / "year=2025" / "month=04" / "date=2025-04-30.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x" * 100)
    result = runner.invoke(app, ["publish", "--data-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout


def test_publish_missing_token_exits_nonzero(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = tmp_path / "nse" / "year=2025" / "month=04" / "date=2025-04-30.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x" * 100)
    result = runner.invoke(app, ["publish", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0


# --- actions subcommand ------------------------------------------------


def test_actions_help_lists_fetch():
    result = runner.invoke(app, ["actions", "--help"])
    assert result.exit_code == 0
    assert "fetch" in result.stdout


def test_actions_fetch_help_lists_flags():
    result = runner.invoke(app, ["actions", "fetch", "--help"])
    assert result.exit_code == 0
    assert "--from" in result.stdout
    assert "--to" in result.stdout
    assert "--exchange" in result.stdout


def test_actions_fetch_inverted_range_rejected():
    result = runner.invoke(
        app,
        ["actions", "fetch", "--from", "2024-12-31", "--to", "2024-01-01"],
    )
    assert result.exit_code != 0


def test_actions_fetch_writes_parquet(tmp_path: Path):
    nse_raw = [
        {"symbol": "HCLTECH", "series": "EQ", "isin": "INE860A01027",
         "comp": "HCL Technologies", "subject": "Interim Dividend - Rs 12 Per Share",
         "exDate": "19-Jan-2024", "recDate": "20-Jan-2024"},
    ]
    bse_raw = [
        {"scrip_code": 532281, "short_name": "HCLTECH",
         "Ex_date": "19 Jan 2024", "exdate": "20240119",
         "Purpose": "Interim Dividend - Rs. - 12.0000",
         "RD_Date": "20 Jan 2024", "long_name": "HCL Technologies Ltd"},
    ]
    cache = tmp_path / "cache"
    out = tmp_path / "out"

    with patch("pipeline.cli.fetch_nse_actions", return_value=nse_raw) as mn, \
         patch("pipeline.cli.fetch_bse_actions", return_value=bse_raw) as mb, \
         patch("pipeline.cli.load_bse_scrip_to_isin",
               return_value={"532281": "INE860A01027"}) as mm:
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--from", "2024-01-01", "--to", "2024-01-31",
             "--exchange", "both",
             "--cache-dir", str(cache),
             "--out-dir", str(out)],
        )

    assert result.exit_code == 0, result.stdout
    mn.assert_called_once()
    mb.assert_called_once()
    mm.assert_called_once()

    nse_path = out / "nse_20240101_20240131.parquet"
    bse_path = out / "bse_20240101_20240131.parquet"
    assert nse_path.exists()
    assert bse_path.exists()

    df = pl.read_parquet(nse_path)
    assert df.height == 1
    assert set(df.columns) == set(ACTION_SCHEMA.keys())
    assert df["type"][0] == "dividend"
    assert df["cash_amount"][0] == 12.0
    assert df["exchange"][0] == "NSE"

    bse_df = pl.read_parquet(bse_path)
    assert bse_df["isin"][0] == "INE860A01027"


def test_actions_fetch_single_exchange(tmp_path: Path):
    raw = [
        {"symbol": "X", "series": "EQ", "subject": "Bonus 1:1",
         "exDate": "01-Feb-2024", "recDate": "01-Feb-2024",
         "comp": "X Ltd", "isin": "INE000A00001"},
    ]
    out = tmp_path / "out"
    with patch("pipeline.cli.fetch_nse_actions", return_value=raw), \
         patch("pipeline.cli.fetch_bse_actions") as mb:
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--from", "2024-02-01", "--to", "2024-02-29",
             "--exchange", "NSE",
             "--cache-dir", str(tmp_path / "cache"),
             "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    mb.assert_not_called()
    assert (out / "nse_20240201_20240229.parquet").exists()
    assert not (out / "bse_20240201_20240229.parquet").exists()


def test_actions_fetch_propagates_fetch_error(tmp_path: Path):
    from pipeline.actions import ActionsFetchError

    with patch("pipeline.cli.fetch_nse_actions",
               side_effect=ActionsFetchError("boom")):
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--from", "2024-01-01", "--to", "2024-01-31",
             "--exchange", "NSE",
             "--cache-dir", str(tmp_path / "cache"),
             "--out-dir", str(tmp_path / "out")],
        )
    assert result.exit_code != 0


def test_actions_fetch_keeps_nse_when_bse_fails(tmp_path: Path):
    from pipeline.actions import ActionsFetchError

    raw = [
        {"symbol": "X", "series": "EQ", "subject": "Bonus 1:1",
         "exDate": "01-Feb-2024", "recDate": "01-Feb-2024",
         "comp": "X Ltd", "isin": "INE000A00001"},
    ]
    out = tmp_path / "out"
    with patch("pipeline.cli.fetch_nse_actions", return_value=raw), \
         patch("pipeline.cli.fetch_bse_actions", side_effect=ActionsFetchError("403")), \
         patch("pipeline.cli.load_bse_scrip_to_isin", return_value={}):
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--from", "2024-02-01", "--to", "2024-02-29",
             "--exchange", "both",
             "--cache-dir", str(tmp_path / "cache"),
             "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    assert "::warning::BSE" in result.stdout
    assert (out / "nse_20240201_20240229.parquet").exists()
    assert not (out / "bse_20240201_20240229.parquet").exists()


def test_help_lists_actions_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "actions" in result.stdout


def test_actions_fetch_year_flag(tmp_path: Path):
    raw = [
        {"symbol": "X", "series": "EQ", "subject": "Bonus 1:1",
         "exDate": "01-Feb-2024", "recDate": "01-Feb-2024",
         "comp": "X Ltd", "isin": "INE000A00001"},
    ]
    out = tmp_path / "out"
    with patch("pipeline.cli.fetch_nse_actions", return_value=raw) as mn:
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--year", "2024",
             "--exchange", "NSE",
             "--cache-dir", str(tmp_path / "cache"),
             "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    # Stable annual file name
    assert (out / "nse_2024.parquet").exists()
    # Range passed to fetcher covers full year
    args, _ = mn.call_args
    assert args[0] == date(2024, 1, 1)
    assert args[1] == date(2024, 12, 31)


def test_actions_fetch_year_conflicts_with_range(tmp_path: Path):
    result = runner.invoke(
        app,
        ["actions", "fetch",
         "--year", "2024",
         "--from", "2024-01-01",
         "--to", "2024-06-30",
         "--exchange", "NSE",
         "--cache-dir", str(tmp_path / "cache"),
         "--out-dir", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.stdout or "mutually exclusive" in (result.stderr or "")


def test_actions_fetch_missing_range_rejected(tmp_path: Path):
    result = runner.invoke(
        app,
        ["actions", "fetch",
         "--exchange", "NSE",
         "--cache-dir", str(tmp_path / "cache"),
         "--out-dir", str(tmp_path / "out")],
    )
    assert result.exit_code != 0


def test_actions_fetch_scrip_map_failure_warns_but_continues(tmp_path: Path):
    from pipeline.actions import ActionsFetchError

    bse_raw = [
        {"scrip_code": 532281, "short_name": "HCLTECH",
         "Ex_date": "19 Jan 2024", "exdate": "20240119",
         "Purpose": "Bonus 1:1"},
    ]
    out = tmp_path / "out"
    with patch("pipeline.cli.fetch_bse_actions", return_value=bse_raw), \
         patch("pipeline.cli.load_bse_scrip_to_isin",
               side_effect=ActionsFetchError("scrip master down")):
        result = runner.invoke(
            app,
            ["actions", "fetch",
             "--from", "2024-01-01", "--to", "2024-01-31",
             "--exchange", "BSE",
             "--cache-dir", str(tmp_path / "cache"),
             "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    bse_df = pl.read_parquet(out / "bse_20240101_20240131.parquet")
    # ISIN stays null when map lookup fails
    assert bse_df["isin"][0] is None


# --- actions adjust ----------------------------------------------------


def _write_prices_parquet(root: Path, exchange: str, year: int, rows: list[tuple]) -> None:
    """Write rows under partitioned layout: <exchange>/year=YYYY/month=MM/date=...parquet."""
    schema = {"isin": pl.Utf8, "date": pl.Date, "symbol": pl.Utf8, "close": pl.Float64}
    df = pl.DataFrame(rows, schema=schema, orient="row")
    for d, group in df.group_by("date"):
        d = d[0]
        path = (root / exchange.lower()
                / f"year={d.year}" / f"month={d.month:02d}"
                / f"date={d.isoformat()}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(path)


def _write_actions_parquet(actions_dir: Path, exchange: str, year: int,
                           ex_dates: list[date], factors_meta: list[dict]) -> None:
    """Write a minimal actions parquet matching ACTION_SCHEMA."""
    from pipeline.actions import to_polars, CorporateAction
    objs = [
        CorporateAction(
            exchange=exchange, symbol="X", isin=m.get("isin", "INE001"),
            company="X", ex_date=ex, record_date=None, type=m["type"],
            ratio_num=m.get("ratio_num"), ratio_den=m.get("ratio_den"),
            cash_amount=m.get("cash_amount"),
            face_value_from=m.get("fv_from"), face_value_to=m.get("fv_to"),
        )
        for ex, m in zip(ex_dates, factors_meta)
    ]
    actions_dir.mkdir(parents=True, exist_ok=True)
    to_polars(objs).write_parquet(actions_dir / f"{exchange.lower()}_{year}.parquet")


def test_actions_adjust_writes_adjusted_parquet(tmp_path: Path):
    prices_dir = tmp_path / "out"
    actions_dir = tmp_path / "actions"
    out_dir = tmp_path / "adjusted"

    _write_prices_parquet(prices_dir, "NSE", 2024, [
        ("INE001", date(2024, 1, 1), "X", 1000.0),
        ("INE001", date(2024, 6, 1), "X", 100.0),  # ex_date
        ("INE001", date(2024, 7, 1), "X", 110.0),
    ])
    _write_actions_parquet(
        actions_dir, "NSE", 2024,
        [date(2024, 6, 1)],
        [{"type": "split", "fv_from": 10.0, "fv_to": 1.0}],
    )

    result = runner.invoke(
        app,
        ["actions", "adjust",
         "--year", "2024",
         "--exchange", "NSE",
         "--prices-dir", str(prices_dir),
         "--actions-dir", str(actions_dir),
         "--out-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout
    out_path = out_dir / "nse_2024.parquet"
    assert out_path.exists()

    adjusted = pl.read_parquet(out_path).sort("date")
    assert adjusted["adj_factor_cumulative"].to_list() == pytest.approx([0.1, 1.0, 1.0])
    assert adjusted["adj_close"].to_list() == pytest.approx([100.0, 100.0, 110.0])


def test_actions_adjust_skips_missing_year(tmp_path: Path):
    # No prices written for 2024; should skip cleanly
    result = runner.invoke(
        app,
        ["actions", "adjust",
         "--year", "2024",
         "--exchange", "NSE",
         "--prices-dir", str(tmp_path / "out"),
         "--actions-dir", str(tmp_path / "actions"),
         "--out-dir", str(tmp_path / "adjusted")],
    )
    assert result.exit_code == 0
    assert "no bhavcopy" in result.stdout


def test_actions_adjust_skips_missing_actions(tmp_path: Path):
    prices_dir = tmp_path / "out"
    _write_prices_parquet(prices_dir, "NSE", 2024, [
        ("INE001", date(2024, 1, 1), "X", 100.0),
    ])
    # No actions parquet
    result = runner.invoke(
        app,
        ["actions", "adjust",
         "--year", "2024",
         "--exchange", "NSE",
         "--prices-dir", str(prices_dir),
         "--actions-dir", str(tmp_path / "actions"),
         "--out-dir", str(tmp_path / "adjusted")],
    )
    assert result.exit_code == 0
    assert "no actions parquet for years" in result.stdout


# --- metrics build -----------------------------------------------------


def _write_adjusted_parquet(adjusted_dir: Path, exchange: str, year: int,
                            rows: list[tuple]) -> None:
    """Write a minimal adjusted parquet with the cols metrics requires."""
    schema = {
        "isin": pl.Utf8,
        "date": pl.Date,
        "symbol": pl.Utf8,
        "adj_close": pl.Float64,
        "volume": pl.Int64,
        "turnover": pl.Float64,
    }
    df = pl.DataFrame(rows, schema=schema, orient="row")
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(adjusted_dir / f"{exchange.lower()}_{year}.parquet")


def test_metrics_build_writes_per_year_parquet(tmp_path: Path):
    adjusted_dir = tmp_path / "adjusted"
    out_dir = tmp_path / "metrics"
    rows = [
        ("INE001", date(2025, 1, d), "X", 100.0 + d, 1000, (100.0 + d) * 1000)
        for d in range(1, 6)
    ]
    _write_adjusted_parquet(adjusted_dir, "NSE", 2025, rows)

    result = runner.invoke(
        app,
        ["metrics", "build",
         "--year", "2025",
         "--exchange", "NSE",
         "--adjusted-dir", str(adjusted_dir),
         "--out-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout

    out_path = out_dir / "nse_2025.parquet"
    assert out_path.exists()
    df = pl.read_parquet(out_path).sort("date")
    # Returns + rolling cols are all present
    for col in ("ret_1d", "ret_ytd", "high_52w", "avg_vol_20d", "avg_turnover_20d"):
        assert col in df.columns
    # 5 rows in, 5 rows out
    assert df.height == 5
    # 1d return: row 1 null, row 2 = (102-101)/101
    assert df["ret_1d"][0] is None
    assert df["ret_1d"][1] == pytest.approx((102 - 101) / 101)

    # Newest day also lands in <ex>_latest.parquet, one row per instrument.
    latest = pl.read_parquet(out_dir / "nse_latest.parquet")
    assert latest.height == 1
    assert latest["date"][0] == date(2025, 1, 5)
    assert latest["ret_1d"][0] == pytest.approx((105 - 104) / 104)


def test_metrics_build_monthly_slices(tmp_path: Path):
    adjusted_dir = tmp_path / "adjusted"
    out_dir = tmp_path / "metrics"
    slices_dir = tmp_path / "slices"
    rows = [
        ("INE001", date(2025, m, d), "X", 100.0 + m + d, 1000, 1.0)
        for m in (1, 2) for d in (2, 3)
    ] + [("INE002", date(2025, 2, 3), "Y", 50.0, 10, 1.0)]
    _write_adjusted_parquet(adjusted_dir, "NSE", 2025, rows)

    result = runner.invoke(
        app,
        ["metrics", "build", "--year", "2025", "--exchange", "NSE",
         "--adjusted-dir", str(adjusted_dir), "--out-dir", str(out_dir),
         "--slices-dir", str(slices_dir)],
    )
    assert result.exit_code == 0, result.stdout
    assert sorted(p.name for p in slices_dir.glob("*.parquet")) == [
        "nse_2025-01.parquet", "nse_2025-02.parquet",
    ]
    feb = pl.read_parquet(slices_dir / "nse_2025-02.parquet")
    assert feb.height == 3
    assert feb["date"].to_list() == [date(2025, 2, 2), date(2025, 2, 3), date(2025, 2, 3)]
    assert feb["symbol"].to_list() == ["X", "X", "Y"]
    # No month files leak into the per-year output dir.
    assert sorted(p.name for p in out_dir.glob("*.parquet")) == [
        "nse_2025.parquet", "nse_latest.parquet",
    ]


def test_metrics_build_all_years_writes_each(tmp_path: Path):
    adjusted_dir = tmp_path / "adjusted"
    out_dir = tmp_path / "metrics"
    for y in (2024, 2025):
        rows = [
            ("INE001", date(y, 1, d), "X", 100.0 + d, 1000, (100.0 + d) * 1000)
            for d in range(1, 4)
        ]
        _write_adjusted_parquet(adjusted_dir, "NSE", y, rows)

    result = runner.invoke(
        app,
        ["metrics", "build",
         "--all-years",
         "--exchange", "NSE",
         "--adjusted-dir", str(adjusted_dir),
         "--out-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "nse_2024.parquet").exists()
    assert (out_dir / "nse_2025.parquet").exists()


def test_metrics_build_requires_year_or_all(tmp_path: Path):
    result = runner.invoke(
        app,
        ["metrics", "build",
         "--exchange", "NSE",
         "--adjusted-dir", str(tmp_path / "adjusted"),
         "--out-dir", str(tmp_path / "metrics")],
    )
    # typer raises BadParameter -> exit_code 2 (CliRunner sends usage to
    # stderr, not stdout, so we just assert on the exit code).
    assert result.exit_code != 0


def test_metrics_build_year_and_all_mutually_exclusive(tmp_path: Path):
    result = runner.invoke(
        app,
        ["metrics", "build",
         "--year", "2025",
         "--all-years",
         "--exchange", "NSE",
         "--adjusted-dir", str(tmp_path / "adjusted"),
         "--out-dir", str(tmp_path / "metrics")],
    )
    assert result.exit_code != 0


def test_metrics_build_skips_year_without_adjusted_file(tmp_path: Path):
    adjusted_dir = tmp_path / "adjusted"
    _write_adjusted_parquet(adjusted_dir, "NSE", 2024, [
        ("INE001", date(2024, 1, 1), "X", 100.0, 1000, 100_000.0),
    ])
    # Ask for 2025 which doesn't exist on disk; skips cleanly.
    result = runner.invoke(
        app,
        ["metrics", "build",
         "--year", "2025",
         "--exchange", "NSE",
         "--adjusted-dir", str(adjusted_dir),
         "--out-dir", str(tmp_path / "metrics")],
    )
    assert result.exit_code == 0
    assert "no adjusted parquet for that year" in result.stdout


def test_metrics_build_year_uses_prior_years_for_window(tmp_path: Path):
    # 2024 has 25 days of history; 2025 has 5 rows. With --year 2025, the
    # 20-day vol window for 2025-01-01 should already be populated using
    # the 20 most recent 2024 rows as seed.
    adjusted_dir = tmp_path / "adjusted"
    rows_2024 = [
        ("INE001", date(2024, 12, 1) + (date(2024, 12, 2) - date(2024, 12, 1)) * i,
         "X", 100.0 + i, 1000 + i * 10, (100.0 + i) * (1000 + i * 10))
        for i in range(25)
    ]
    rows_2025 = [
        ("INE001", date(2025, 1, d), "X", 200.0 + d, 2000, (200.0 + d) * 2000)
        for d in range(1, 6)
    ]
    _write_adjusted_parquet(adjusted_dir, "NSE", 2024, rows_2024)
    _write_adjusted_parquet(adjusted_dir, "NSE", 2025, rows_2025)

    result = runner.invoke(
        app,
        ["metrics", "build",
         "--year", "2025",
         "--exchange", "NSE",
         "--adjusted-dir", str(adjusted_dir),
         "--out-dir", str(tmp_path / "metrics")],
    )
    assert result.exit_code == 0, result.stdout

    df = pl.read_parquet(tmp_path / "metrics" / "nse_2025.parquet").sort("date")
    # First row of 2025 has 20 prior days from 2024 -> avg_vol_20d populated.
    assert df["avg_vol_20d"][0] is not None
    # Output is filtered to 2025 only (5 rows).
    assert df.height == 5
    assert all(d.year == 2025 for d in df["date"].to_list())


# --- mixed rollup/daily reading (regression: ShapeError width 16 vs 14) -----


def _write_price_frame(path: Path, dates: list[date], with_year_month: bool):
    """Minimal bhavcopy-shaped frame; rollups add year/month real columns."""
    df = pl.DataFrame({
        "date": dates,
        "symbol": ["TCS"] * len(dates),
        "series": ["EQ"] * len(dates),
        "isin": ["INE467B01029"] * len(dates),
        "name": ["Tata Consultancy"] * len(dates),
        "close": [100.0 + i for i in range(len(dates))],
    })
    if with_year_month:
        df = df.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_read_bhavcopy_mixed_rollup_and_daily(tmp_path: Path):
    from pipeline.cli import _read_bhavcopy_mixed

    rollup = tmp_path / "nse" / "year=2025" / "nse_2025.parquet"
    daily = tmp_path / "nse" / "year=2026" / "month=08" / "date=2026-08-06.parquet"
    _write_price_frame(rollup, [date(2025, 1, 1), date(2025, 1, 2)], with_year_month=True)
    _write_price_frame(daily, [date(2026, 8, 6)], with_year_month=False)

    df = _read_bhavcopy_mixed([rollup, daily])

    assert df.height == 3
    assert "year" not in df.columns
    assert "month" not in df.columns


def test_read_bhavcopy_mixed_uniform_files_still_work(tmp_path: Path):
    from pipeline.cli import _read_bhavcopy_mixed

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write_price_frame(a, [date(2026, 8, 5)], with_year_month=False)
    _write_price_frame(b, [date(2026, 8, 6)], with_year_month=False)

    df = _read_bhavcopy_mixed([a, b])
    assert df.height == 2


def test_price_years_on_disk_sees_rollup_only_years(tmp_path: Path):
    from pipeline.cli import _price_years_on_disk

    # 2024 exists ONLY as a compacted rollup (post-prune steady state),
    # 2026 has a daily file. Both years must be discovered.
    _write_price_frame(
        tmp_path / "year=2024" / "nse_2024.parquet",
        [date(2024, 1, 1)], with_year_month=True,
    )
    _write_price_frame(
        tmp_path / "year=2026" / "month=08" / "date=2026-08-06.parquet",
        [date(2026, 8, 6)], with_year_month=False,
    )

    assert _price_years_on_disk(tmp_path) == [2024, 2026]


def test_symbol_history_build_handles_mixed_tree(tmp_path: Path):
    prices = tmp_path / "prices"
    out = tmp_path / "sh"
    _write_price_frame(
        prices / "nse" / "year=2025" / "nse_2025.parquet",
        [date(2025, 1, 1)], with_year_month=True,
    )
    _write_price_frame(
        prices / "nse" / "year=2026" / "month=08" / "date=2026-08-06.parquet",
        [date(2026, 8, 6)], with_year_month=False,
    )

    result = runner.invoke(app, [
        "symbol-history", "build", "--exchange", "NSE",
        "--prices-dir", str(prices), "--out-dir", str(out),
    ])
    assert result.exit_code == 0, result.output


def test_status_write_prices_then_derived(tmp_path: Path):
    import json

    out = tmp_path / "api" / "v1" / "status.json"
    r = runner.invoke(app, ["status", "write", "--stage", "prices", "--trading-date", "2026-09-08",
                            "--run-url", "https://github.com/tejhq/tej-bazaar/actions/runs/1", "--out", str(out)])
    assert r.exit_code == 0, r.stdout
    body = json.loads(out.read_text())
    assert body["trading_date"] == "2026-09-08"
    assert body["stage"] == "prices"
    assert body["prices_published_at"].endswith("Z")
    assert body["derived_published_at"] is None
    assert body["run_url"].endswith("/runs/1")

    r = runner.invoke(app, ["status", "write", "--stage", "derived", "--trading-date", "2026-09-08",
                            "--prices-published-at", body["prices_published_at"], "--out", str(out)])
    assert r.exit_code == 0, r.stdout
    body2 = json.loads(out.read_text())
    assert body2["prices_published_at"] == body["prices_published_at"]
    assert body2["derived_published_at"].endswith("Z")

    r = runner.invoke(app, ["status", "write", "--stage", "nope", "--trading-date", "2026-09-08", "--out", str(out)])
    assert r.exit_code != 0
    r = runner.invoke(app, ["status", "write", "--stage", "prices", "--trading-date", "8 Sep", "--out", str(out)])
    assert r.exit_code != 0


def test_actions_reparse_recomputes_from_subject(tmp_path: Path):
    adir = tmp_path / "actions"
    adir.mkdir()
    df = pl.DataFrame(
        {
            "exchange": ["NSE", "NSE"],
            "symbol": ["TCS", "X"],
            "isin": ["I1", "I2"],
            "company": ["TCS", "X"],
            "ex_date": [date(2024, 1, 19), date(2024, 2, 1)],
            "record_date": [None, None],
            "type": ["dividend", "bonus"],
            "ratio_num": [None, None],
            "ratio_den": [None, None],
            "cash_amount": [9.0, None],
            "face_value_from": [None, None],
            "face_value_to": [None, None],
            "raw_subject": ["Interim Dividend - Rs 9 Per Share Special Dividend - Rs 18 Per Share", "Bonus 1:2"],
        },
        schema_overrides={"ratio_num": pl.Int64, "ratio_den": pl.Int64, "record_date": pl.Date,
                          "cash_amount": pl.Float64, "face_value_from": pl.Float64, "face_value_to": pl.Float64},
    )
    df.write_parquet(adir / "nse_2024.parquet")
    r = runner.invoke(app, ["actions", "reparse", "--actions-dir", str(adir)])
    assert r.exit_code == 0, r.stdout
    out = pl.read_parquet(adir / "nse_2024.parquet")
    assert out["cash_amount"].to_list() == [27.0, None]
    assert out["ratio_num"].to_list() == [None, 1]
    assert out["ratio_den"].to_list() == [None, 2]
    # Second run changes nothing.
    r = runner.invoke(app, ["actions", "reparse", "--actions-dir", str(adir)])
    assert r.exit_code == 0 and "0" in r.stdout


def test_actions_reparse_writes_a_file_whose_only_change_fills_a_null(tmp_path: Path):
    # A ratio stored as None that now parses is a change. Comparing with != gives null for it, so a
    # file whose only changes were of this kind was never rewritten.
    adir = tmp_path / "actions"
    adir.mkdir()
    pl.DataFrame(
        {
            "exchange": ["NSE"],
            "symbol": ["X"],
            "isin": ["I1"],
            "company": ["X"],
            "ex_date": [date(2022, 6, 22)],
            "record_date": [None],
            "type": ["bonus"],
            "ratio_num": [None],
            "ratio_den": [None],
            "cash_amount": [None],
            "face_value_from": [None],
            "face_value_to": [None],
            "raw_subject": ["Bonus 1:2"],
        },
        schema_overrides={
            "ratio_num": pl.Int64,
            "ratio_den": pl.Int64,
            "record_date": pl.Date,
            "cash_amount": pl.Float64,
            "face_value_from": pl.Float64,
            "face_value_to": pl.Float64,
        },
    ).write_parquet(adir / "nse_2022.parquet")
    r = runner.invoke(app, ["actions", "reparse", "--actions-dir", str(adir)])
    assert r.exit_code == 0, r.stdout
    out = pl.read_parquet(adir / "nse_2022.parquet")
    assert (out["ratio_num"].to_list(), out["ratio_den"].to_list()) == ([1], [2])


def _adjusted(tmp: Path, ex: str, sym: str, cliff: bool) -> None:
    """Write one exchange's adjusted prices, optionally with an unexplained 10:1 step."""
    rows = []
    for i in range(6):
        d = date(2024, 1, 1 + i)
        px = 100.0 if not (cliff and i >= 3) else 10.0
        rows.append({"date": d, "symbol": sym, "adj_close": px})
    pl.DataFrame(rows).write_parquet(tmp / f"{ex}_2024.parquet")


# `--exchange both` used to glob "both_*.parquet", which matches nothing, so audit and derive
# silently reported no data and BSE was never covered by either command.
@pytest.mark.parametrize("cmd", ["audit", "derive"])
def test_actions_both_covers_each_exchange(tmp_path: Path, cmd: str):
    adj, acts = tmp_path / "adj", tmp_path / "acts"
    adj.mkdir()
    acts.mkdir()
    _adjusted(adj, "nse", "AAA", cliff=True)
    _adjusted(adj, "bse", "BBB", cliff=True)
    result = runner.invoke(
        app,
        ["actions", cmd, "--exchange", "both", "--adjusted-dir", str(adj), "--actions-dir", str(acts)],
    )
    assert "no adjusted prices" not in result.stdout
    assert "NSE" in result.stdout and "BSE" in result.stdout
    assert "AAA" in result.stdout and "BBB" in result.stdout


def _session(prices_dir: Path, ex: str, d: str) -> None:
    day = date.fromisoformat(d)
    p = prices_dir / ex / f"year={day.year}" / f"month={day.month:02d}" / f"date={d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": ["RELIANCE"], "date": [day], "close": [1200.0]}).write_parquet(p)


def test_check_sessions_passes_when_every_due_day_is_there(tmp_path: Path):
    for d in ("2026-09-23", "2026-09-24", "2026-09-25"):
        _session(tmp_path, "nse", d)
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-23", "--to", "2026-09-27",
         "--prices-dir", str(tmp_path), "--now", "2026-09-27T09:00"],
    )
    assert result.exit_code == 0, result.stdout
    assert "every due session is present" in result.stdout


def test_check_sessions_fails_on_the_gap_that_went_unnoticed(tmp_path: Path):
    # The 25th traded and was never fetched; the run that missed it went green.
    _session(tmp_path, "nse", "2026-09-24")
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-23", "--to", "2026-09-26",
         "--prices-dir", str(tmp_path), "--now", "2026-09-26T00:15"],
    )
    assert result.exit_code == 1
    assert "2026-09-25" in result.stdout
    assert "missing past due" in result.stdout


def test_check_sessions_waits_for_todays_bhavcopy(tmp_path: Path):
    _session(tmp_path, "nse", "2026-09-24")
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-24", "--to", "2026-09-25",
         "--prices-dir", str(tmp_path), "--now", "2026-09-25T17:00"],
    )
    assert result.exit_code == 0, result.stdout
    assert "not published yet" in result.stdout


def test_check_sessions_ignores_weekends_and_holidays(tmp_path: Path):
    # 2 October is Gandhi Jayanti and the 3rd and 4th are a weekend, so a range
    # holding only those three days has nothing to be missing.
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-10-02", "--to", "2026-10-04",
         "--prices-dir", str(tmp_path), "--now", "2026-10-05T09:00"],
    )
    assert result.exit_code == 0, result.stdout
    assert "no trading day in range" in result.stdout


def test_check_sessions_empty_range_is_not_a_failure(tmp_path: Path):
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-26", "--to", "2026-09-27",
         "--prices-dir", str(tmp_path), "--now", "2026-09-28T09:00"],
    )
    assert result.exit_code == 0, result.stdout
    assert "no trading day in range" in result.stdout


def test_check_sessions_counts_already_published_days(tmp_path: Path):
    # A day on R2 that this sweep failed to re-fetch is not an outage.
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-21", "--to", "2026-09-25",
         "--prices-dir", str(tmp_path), "--published", "2026-09-25",
         "--now", "2026-09-26T00:15"],
    )
    assert result.exit_code == 0, result.stdout
    assert "published" in result.stdout


def test_check_sessions_still_fails_on_the_day_past_what_is_published(tmp_path: Path):
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-09-21", "--to", "2026-09-26",
         "--prices-dir", str(tmp_path), "--published", "2026-09-24",
         "--now", "2026-09-26T00:15"],
    )
    assert result.exit_code == 1
    assert "missing past due: 2026-09-25" in result.stdout


def test_check_sessions_grace_warns_then_fails(tmp_path: Path):
    # Delivery for Monday 26 October never arrived. With three trading days of
    # grace it only warns until Thursday's due hour, then fails the run.
    args = ["check-sessions", "--from", "2026-10-26", "--to", "2026-10-26",
            "--prices-dir", str(tmp_path), "--grace", "3"]
    warned = runner.invoke(app, [*args, "--now", "2026-10-29T19:59"])
    assert warned.exit_code == 0, warned.stdout
    assert "::warning::2026-10-26 missing" in warned.stdout
    assert "20:00 IST on 2026-10-29" in warned.stdout

    failed = runner.invoke(app, [*args, "--now", "2026-10-29T20:00"])
    assert failed.exit_code == 1
    assert "missing past due: 2026-10-26" in failed.stdout


def test_check_sessions_grace_counts_trading_days(tmp_path: Path):
    # 1 October 2026 is a Thursday; the 2nd is a holiday and the 3rd and 4th a
    # weekend, so its third trading day after is Wednesday the 7th.
    args = ["check-sessions", "--from", "2026-10-01", "--to", "2026-10-01",
            "--prices-dir", str(tmp_path), "--grace", "3"]
    assert runner.invoke(app, [*args, "--now", "2026-10-07T19:00"]).exit_code == 0
    assert runner.invoke(app, [*args, "--now", "2026-10-07T20:00"]).exit_code == 1


def test_check_sessions_grace_runs_past_the_holiday_list(tmp_path: Path):
    # The list ends with 2026, so the third trading day after the 29th is
    # counted on weekdays: Friday 1 January 2027.
    args = ["check-sessions", "--from", "2026-12-29", "--to", "2026-12-29",
            "--prices-dir", str(tmp_path), "--grace", "3"]
    warned = runner.invoke(app, [*args, "--now", "2026-12-31T21:30"])
    assert warned.exit_code == 0, warned.stdout
    assert "20:00 IST on 2027-01-01" in warned.stdout
    assert runner.invoke(app, [*args, "--now", "2027-01-01T20:00"]).exit_code == 1


def test_check_sessions_warns_only_for_a_year_without_a_list(tmp_path: Path):
    # A missing day in 2027 may be a holiday nobody has listed yet.
    args = ["check-sessions", "--from", "2027-01-04", "--to", "2027-01-04", "--prices-dir", str(tmp_path)]
    result = runner.invoke(app, [*args, "--now", "2027-01-20T20:00"])
    assert result.exit_code == 0, result.stdout
    assert "2027-01-04 missing" in result.stdout
    assert "no holiday list" in result.stdout


def test_check_sessions_reminds_in_december_to_add_the_next_year(tmp_path: Path):
    args = ["check-sessions", "--from", "2026-11-28", "--to", "2026-11-29", "--prices-dir", str(tmp_path)]
    assert "Add the next year" not in runner.invoke(app, [*args, "--now", "2026-11-30T20:00"]).stdout
    assert "Add the next year" in runner.invoke(app, [*args, "--now", "2026-12-01T20:00"]).stdout


def test_check_sessions_counts_days_inside_a_year_rollup(tmp_path: Path):
    # The delivery job checks against the published year rollup, which holds
    # many days and has no date= file per day.
    p = tmp_path / "nse" / "year=2026" / "nse_2026.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame(
        {"symbol": ["RELIANCE", "RELIANCE"], "date": [date(2026, 10, 26), date(2026, 10, 27)]}
    ).write_parquet(p)
    result = runner.invoke(
        app,
        ["check-sessions", "--from", "2026-10-26", "--to", "2026-10-27",
         "--prices-dir", str(tmp_path), "--now", "2026-10-30T09:00"],
    )
    assert result.exit_code == 0, result.stdout
    assert "every due session is present" in result.stdout


# --- delivery subcommand -----------------------------------------------

MTO_FIXTURE = Path(__file__).parent / "fixtures" / "nse_mto_20261028_sample.DAT"


def test_delivery_backfill_writes_skips_and_keeps_going(tmp_path: Path):
    # The 27th is not published (404), the 28th is good, and the 29th serves
    # the 28th's file, which the date check rejects. None of it stops the loop.
    def fake_fetch(d, dest_dir):
        if d == date(2026, 10, 27):
            raise BhavcopyNotFoundError("bhavcopy not found (404)")
        return MTO_FIXTURE

    out = tmp_path / "delivery"
    with patch("pipeline.cli.fetch_nse_delivery", side_effect=fake_fetch), \
         patch("pipeline.cli.time.sleep") as sleep:
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--from", "2026-10-27", "--to", "2026-10-29",
             "--raw-dir", str(tmp_path / "raw"), "--out-dir", str(out), "--sleep", "1.5"],
        )

    assert result.exit_code == 0, result.stdout
    written = partition_path(out, "NSE", date(2026, 10, 28))
    assert pl.read_parquet(written).height == 5
    assert not partition_path(out, "NSE", date(2026, 10, 27)).exists()
    assert not partition_path(out, "NSE", date(2026, 10, 29)).exists()
    assert "missing NSE 2026-10-27" in result.stdout
    assert "DeliveryParseError" in result.stdout
    # One pause between each pair of downloads, none before the first.
    assert [c.args for c in sleep.call_args_list] == [(1.5,), (1.5,)]


def test_delivery_backfill_skips_written_days_without_a_request(tmp_path: Path):
    out = tmp_path / "delivery"
    existing = partition_path(out, "NSE", date(2026, 10, 28))
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")
    with patch("pipeline.cli.fetch_nse_delivery") as fetch, \
         patch("pipeline.cli.time.sleep") as sleep:
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--from", "2026-10-28", "--to", "2026-10-28",
             "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    fetch.assert_not_called()
    sleep.assert_not_called()


def _fetched_days(tmp_path: Path, *args: str) -> list[date]:
    with patch("pipeline.cli.fetch_nse_delivery",
               side_effect=BhavcopyNotFoundError("not found")) as fetch, \
         patch("pipeline.cli.time.sleep"):
        result = runner.invoke(
            app,
            ["delivery", "backfill", *args, "--raw-dir", str(tmp_path / "raw"),
             "--out-dir", str(tmp_path / "delivery")],
        )
    assert result.exit_code == 0, result.stdout
    return [c.args[0] for c in fetch.call_args_list]


def test_delivery_backfill_adds_the_sessions_nse_has(tmp_path: Path):
    # Saturday 2024-01-20 traded and is in nse/, but the calendar misses it.
    p = tmp_path / "prices" / "nse" / "year=2024" / "nse_2024.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["RELIANCE"], "date": [date(2024, 1, 20)]}).write_parquet(p)
    days = _fetched_days(tmp_path, "--from", "2024-01-19", "--to", "2024-01-23",
                         "--prices-dir", str(tmp_path / "prices"))
    # The 22nd was a holiday, the reason for the Saturday session.
    assert days == [date(2024, 1, 19), date(2024, 1, 20), date(2024, 1, 23)]


def test_delivery_backfill_runs_past_the_holiday_list(tmp_path: Path):
    # The list ends with 2026; after it every weekday is tried.
    days = _fetched_days(tmp_path, "--from", "2026-12-31", "--to", "2027-01-05")
    assert days == [date(2026, 12, 31), date(2027, 1, 1), date(2027, 1, 4), date(2027, 1, 5)]


def test_delivery_backfill_inverted_range_rejected():
    result = runner.invoke(
        app, ["delivery", "backfill", "--from", "2026-10-28", "--to", "2026-10-01"]
    )
    assert result.exit_code != 0
    assert "--to must be on or after --from" in result.output


@pytest.mark.parametrize("cached_days", [[27, 28, 29], [28]])
def test_delivery_backfill_sleeps_only_between_downloads(tmp_path: Path, cached_days: list[int]):
    raw = tmp_path / "raw" / "nse"
    raw.mkdir(parents=True)
    for day in cached_days:
        (raw / f"MTO_{day}102026.DAT").write_bytes(
            MTO_FIXTURE.read_bytes().replace(b"28102026", f"{day}102026".encode())
        )

    def download(url, *args):
        return MTO_FIXTURE.read_bytes().replace(b"28102026", url[-12:-4].encode())

    out = tmp_path / "delivery"
    with patch("pipeline.delivery._http_get_with_retry", side_effect=download) as fetch, \
         patch("pipeline.cli.time.sleep") as sleep:
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--from", "2026-10-27", "--to", "2026-10-29",
             "--raw-dir", str(raw.parent), "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    assert fetch.call_count == 3 - len(cached_days)
    assert [c.args for c in sleep.call_args_list] == [(1.0,)] * max(0, fetch.call_count - 1)
    assert len(list(out.rglob("date=*.parquet"))) == 3


BSE_FIXTURE = Path(__file__).parent / "fixtures" / "bse_scbseall_20261028_sample.TXT"
BSE_BHAVCOPY = Path(__file__).parent / "fixtures" / "bse_bhavcopy_20261028_sample.csv"


def _bse_day(day: int) -> tuple[bytes, bytes]:
    """The SCBSEALL zip and the bhavcopy fixtures, moved to `day` October 2026."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            f"SCBSEALL{day}10.TXT",
            BSE_FIXTURE.read_bytes().replace(b"28102026", f"{day}102026".encode()),
        )
    bhavcopy = BSE_BHAVCOPY.read_bytes().replace(b"2026-10-28", f"2026-10-{day}".encode())
    return buf.getvalue(), bhavcopy


@pytest.mark.parametrize("cached_bhavcopies", [[], [27, 28]])
def test_delivery_backfill_bse_sleeps_between_all_downloads(
    tmp_path: Path, cached_bhavcopies: list[int]
):
    # A BSE day downloads SCBSEALL, then the day's bhavcopy for the scrip codes,
    # unless the prices backfill already left it in --raw-dir.
    raw = tmp_path / "raw"
    (raw / "bse").mkdir(parents=True)
    for day in cached_bhavcopies:
        cached = raw / "bse" / f"BhavCopy_BSE_CM_0_0_0_202610{day}_F_0000.csv"
        cached.write_bytes(_bse_day(day)[1])

    events = []

    def scbseall(url, *args):  # .../gross/2026/SCBSEALL2710.zip
        events.append("download")
        return _bse_day(int(url[-8:-6]))[0]

    def bhavcopy(url, *args):  # .../BhavCopy_BSE_CM_0_0_0_20261027_F_0000.CSV
        events.append("download")
        return _bse_day(int(url[-13:-11]))[1]

    out = tmp_path / "delivery"
    with patch("pipeline.delivery._http_get_with_retry", side_effect=scbseall) as get_scbseall, \
         patch("pipeline.fetch._http_get_with_retry", side_effect=bhavcopy) as get_bhavcopy, \
         patch("pipeline.cli.time.sleep", side_effect=lambda _: events.append("sleep")) as sleep:
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--exchange", "BSE", "--from", "2026-10-27",
             "--to", "2026-10-28", "--raw-dir", str(raw), "--out-dir", str(out),
             "--prices-dir", str(tmp_path / "prices")],
        )
    assert result.exit_code == 0, result.stdout
    assert get_scbseall.call_count == 2
    assert get_bhavcopy.call_count == 2 - len(cached_bhavcopies)
    downloads = get_scbseall.call_count + get_bhavcopy.call_count
    assert [c.args for c in sleep.call_args_list] == [(1.0,)] * (downloads - 1)
    assert events == ["download", "sleep"] * (downloads - 1) + ["download"]
    written = pl.read_parquet(partition_path(out, "BSE", date(2026, 10, 28)))
    assert written["symbol"].to_list() == [
        "20MICRONS", "AMBALALSA", "BOMDYEING", "RELIANCE", "SBIN", "TCS"
    ]
    assert len(list(out.rglob("date=*.parquet"))) == 2


def test_delivery_backfill_bse_fetches_no_bhavcopy_without_scbseall(tmp_path: Path):
    # Before about 18:15 IST the day's SCBSEALL is a 404, so its bhavcopy is not asked for.
    with patch("pipeline.cli.fetch_bse_delivery",
               side_effect=BhavcopyNotFoundError("bhavcopy not found (404)")), \
         patch("pipeline.cli.fetch_bse") as fetch_bhavcopy, \
         patch("pipeline.cli.time.sleep"):
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--exchange", "BSE", "--from", "2026-10-28",
             "--to", "2026-10-28", "--raw-dir", str(tmp_path / "raw"),
             "--out-dir", str(tmp_path / "delivery"),
             "--prices-dir", str(tmp_path / "prices")],
        )
    assert result.exit_code == 0, result.stdout
    fetch_bhavcopy.assert_not_called()
    assert "missing BSE 2026-10-28" in result.stdout


def test_delivery_backfill_adds_the_sessions_each_exchange_has(tmp_path: Path):
    # The calendar has no Saturday 2026-01-31, so only a date in bse/ brings
    # it in, and only for BSE.
    p = tmp_path / "prices" / "bse" / "year=2026" / "bse_2026.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["RELIANCE"], "date": [date(2026, 1, 31)]}).write_parquet(p)
    missing = BhavcopyNotFoundError("not found")
    with patch("pipeline.cli.fetch_nse_delivery", side_effect=missing) as nse, \
         patch("pipeline.cli.fetch_bse_delivery", side_effect=missing) as bse, \
         patch("pipeline.cli.time.sleep"):
        result = runner.invoke(
            app,
            ["delivery", "backfill", "--exchange", "both", "--from", "2026-01-30",
             "--to", "2026-02-02", "--raw-dir", str(tmp_path / "raw"),
             "--out-dir", str(tmp_path / "delivery"),
             "--prices-dir", str(tmp_path / "prices")],
        )
    assert result.exit_code == 0, result.stdout
    assert [c.args[0] for c in nse.call_args_list] == [
        date(2026, 1, 30), date(2026, 2, 1), date(2026, 2, 2)
    ]
    assert [c.args[0] for c in bse.call_args_list] == [
        date(2026, 1, 30), date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2)
    ]


@pytest.mark.parametrize(
    ("extra", "listed"),
    [([], "nse/year=2026/"), (["--prefix", "delivery"], "delivery/nse/year=2026/")],
)
def test_compact_from_r2_lists_under_the_prefix(tmp_path: Path, extra: list[str], listed: str):
    s3 = MagicMock()
    s3.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
    with patch("pipeline.publish_r2._build_client", return_value=s3):
        result = runner.invoke(
            app,
            ["compact", "--year", "2026", "--exchange", "NSE", "--data-dir", str(tmp_path),
             "--from-r2", *extra],
        )
    assert result.exit_code == 0, result.stdout
    s3.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="tej-bazaar", Prefix=listed
    )
