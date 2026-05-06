from pathlib import Path

from typer.testing import CliRunner

from pipeline import __version__
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
