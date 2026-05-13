from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
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
