from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import responses
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.fetch import BhavcopyNotFoundError
from pipeline.fo import (
    FO_KEY,
    FO_SCHEMA,
    FoParseError,
    fetch_nse_fo,
    fo_csv_name,
    parse_fo,
)
from pipeline.push import partition_path, write_partitioned

# Trimmed from the real NSE F&O bhavcopies of 2026-10-30 (UDiFF) and
# 2010-01-04 (legacy): NIFTY futures and one strike's CE/PE, RELIANCE stock
# future and option. The legacy file keeps an American-style (CA) option.
UDIFF = Path(__file__).parent / "fixtures" / "nse_fo_20261030_sample.csv"
LEGACY = Path(__file__).parent / "fixtures" / "nse_fo_20100104_sample.csv"

runner = CliRunner()


@pytest.mark.parametrize(("path", "d"), [(UDIFF, date(2026, 10, 30)), (LEGACY, date(2010, 1, 4))])
def test_parse_fo_schema_and_key(path: Path, d: date):
    df = parse_fo(path, d)
    assert df.schema == pl.Schema(FO_SCHEMA)
    assert df.height == 6
    assert set(df["date"]) == {d}
    assert df.height == df.unique(FO_KEY).height


def test_futures_have_no_strike_or_option_type():
    for path, d in [(UDIFF, date(2026, 10, 30)), (LEGACY, date(2010, 1, 4))]:
        fut = parse_fo(path, d).filter(pl.col("instrument").str.starts_with("FUT"))
        assert fut.height == 3
        assert fut["strike"].null_count() == 3
        assert fut["option_type"].null_count() == 3


def test_udiff_known_values():
    df = parse_fo(UDIFF, date(2026, 10, 30))
    ce = df.filter(
        (pl.col("symbol") == "NIFTY") & (pl.col("option_type") == "CE")
    ).row(0, named=True)
    assert ce["instrument"] == "OPTIDX"
    assert ce["expiry"] == date(2026, 11, 3)
    assert ce["strike"] == 23000.0
    assert ce["open"] == 104.9
    assert ce["lot_size"] == 65
    assert ce["underlying"] == pytest.approx(23053.1)
    near = df.filter(pl.col("instrument") == "FUTIDX").sort("expiry").row(0, named=True)
    assert near["expiry"] == date(2026, 11, 23)
    assert near["open"] == 23010.1


def test_legacy_known_values_and_lakh_turnover():
    df = parse_fo(LEGACY, date(2010, 1, 4))
    fut = df.filter(
        (pl.col("instrument") == "FUTIDX") & (pl.col("expiry") == date(2010, 1, 28))
    ).row(0, named=True)
    assert fut["close"] == 5239.8
    assert fut["contracts"] == 237231
    assert fut["turnover"] == pytest.approx(620467.68 * 1e5)
    assert fut["open_interest"] == 23014150
    assert fut["oi_change"] == 1012250
    assert fut["lot_size"] is None and fut["underlying"] is None
    assert "CA" in df["option_type"].to_list()


def test_parse_fo_rejects_wrong_date():
    with pytest.raises(FoParseError, match="trade dates"):
        parse_fo(UDIFF, date(2026, 10, 29))


def test_parse_fo_rejects_unknown_instrument(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text(LEGACY.read_text().replace("FUTSTK,RELIANCE", "FUTCUR,USDINR"))
    with pytest.raises(FoParseError, match="unknown instruments"):
        parse_fo(p, date(2010, 1, 4))


def test_parse_fo_rejects_repeated_contract(tmp_path: Path):
    lines = LEGACY.read_text().splitlines(keepends=True)
    p = tmp_path / "x.csv"
    p.write_text("".join(lines + [lines[1]]))
    with pytest.raises(FoParseError, match="repeated"):
        parse_fo(p, date(2010, 1, 4))


def test_fo_csv_name_switches_at_cutover():
    assert fo_csv_name(date(2024, 7, 5)) == "fo05JUL2024bhav.csv"
    assert fo_csv_name(date(2024, 7, 8)) == "BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv"


@responses.activate
def test_fetch_nse_fo_404_and_cache(tmp_path: Path):
    responses.get(
        "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20261102_F_0000.csv.zip",
        status=404,
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_nse_fo(date(2026, 11, 2), tmp_path, retries=1)
    cached = tmp_path / fo_csv_name(date(2010, 1, 4))
    cached.write_text("cached")
    assert fetch_nse_fo(date(2010, 1, 4), tmp_path) == cached
    assert len(responses.calls) == 1


def test_fo_backfill_writes_and_keeps_going(tmp_path: Path):
    def fake_fetch(d, dest_dir):
        if d == date(2026, 10, 29):
            raise BhavcopyNotFoundError("bhavcopy not found (404)")
        return UDIFF

    out = tmp_path / "fo"
    with patch("pipeline.cli.fetch_nse_fo", side_effect=fake_fetch), \
         patch("pipeline.cli.time.sleep"):
        result = runner.invoke(
            app,
            ["fo", "backfill", "--from", "2026-10-28", "--to", "2026-10-30",
             "--raw-dir", str(tmp_path / "raw"), "--out-dir", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    assert pl.read_parquet(partition_path(out, "NSE", date(2026, 10, 30))).height == 6
    assert not partition_path(out, "NSE", date(2026, 10, 29)).exists()
    # The 28th gets the 30th's file, which the date check rejects.
    assert not partition_path(out, "NSE", date(2026, 10, 28)).exists()
    assert "missing NSE 2026-10-29" in result.stdout
    assert "FoParseError" in result.stdout


def test_compact_keeps_every_contract(tmp_path: Path):
    day = parse_fo(UDIFF, date(2026, 10, 30))
    write_partitioned(day, tmp_path, "NSE")
    write_partitioned(day.with_columns(pl.lit(date(2026, 10, 29)).alias("date")), tmp_path, "NSE")
    result = runner.invoke(
        app, ["compact", "--year", "2026", "--exchange", "NSE", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    rollup = pl.read_parquet(tmp_path / "nse" / "year=2026" / "nse_2026.parquet")
    assert rollup.height == 12
    assert rollup.select(FO_KEY).is_duplicated().sum() == 0
    keys = rollup.select("symbol", "date")
    assert keys.equals(keys.sort("symbol", "date"))
