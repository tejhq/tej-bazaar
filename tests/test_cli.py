from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from typer.testing import CliRunner

from pipeline import __version__
from pipeline.actions import ACTION_SCHEMA, CorporateAction
from pipeline.cli import app

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
