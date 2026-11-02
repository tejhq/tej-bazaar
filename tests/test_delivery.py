import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import responses

from pipeline import delivery
from pipeline.delivery import (
    DeliveryParseError,
    fetch_bse_delivery,
    fetch_nse_delivery,
    parse_mto,
    parse_scbseall,
)
from pipeline.fetch import BhavcopyFetchError, BhavcopyNotFoundError

# Trimmed from the real MTO_28102026.DAT (2019+ layout) and MTO_13062012.DAT
# (2010-2019 layout, CRLF); the 10,MTO line is rewritten to the kept records.
FIXTURE = Path(__file__).parent / "fixtures" / "nse_mto_20261028_sample.DAT"
FIXTURE_2012 = Path(__file__).parent / "fixtures" / "nse_mto_20120613_sample.DAT"
# Trimmed from the real SCBSEALL2810.zip member and the same day's BSE
# bhavcopy to the same ten scrips, six of them in A/B/T.
BSE_FIXTURE = Path(__file__).parent / "fixtures" / "bse_scbseall_20261028_sample.TXT"
BSE_BHAVCOPY = Path(__file__).parent / "fixtures" / "bse_bhavcopy_20261028_sample.csv"


def test_parse_mto_schema():
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    assert df.columns == ["date", "symbol", "series", "volume", "delivery_qty", "delivery_pct"]
    assert df.schema["date"] == pl.Date
    assert df.schema["volume"] == pl.Int64
    assert df.schema["delivery_qty"] == pl.Int64
    assert df.schema["delivery_pct"] == pl.Float64


def test_parse_mto_keeps_only_the_series_nse_carries():
    # The fixture also holds a YW debt row and an SM (SME) row; nse/ has neither.
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    assert df["symbol"].to_list() == ["20MICRONS", "INFY", "RELIANCE", "SBIN", "TCS"]
    assert set(df["series"]) == {"EQ"}
    assert set(df["date"]) == {date(2026, 10, 28)}


def test_parse_mto_known_value_reliance():
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    row = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert row["volume"] == 21866598
    assert row["delivery_qty"] == 9900037
    assert row["delivery_pct"] == pytest.approx(45.27)


def test_parse_mto_older_layout_with_crlf():
    df = parse_mto(FIXTURE_2012, date(2012, 6, 13))
    assert df.height == 8
    row = df.filter(pl.col("symbol") == "3IINFOTECH").row(0, named=True)
    assert (row["volume"], row["delivery_qty"], row["delivery_pct"]) == (5514499, 3788329, 68.70)


def test_parse_mto_rejects_a_file_for_another_day():
    with pytest.raises(DeliveryParseError, match="header date 28102026"):
        parse_mto(FIXTURE, date(2026, 10, 29))


def test_parse_mto_rejects_a_short_file(tmp_path: Path):
    # A file cut off mid-download has fewer records than its header counts.
    short = tmp_path / "MTO_28102026.DAT"
    short.write_text("".join(FIXTURE.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(DeliveryParseError, match="header says 7 records"):
        parse_mto(short, date(2026, 10, 28))


def test_parse_mto_rejects_a_truncated_final_percentage(tmp_path: Path):
    short = tmp_path / "MTO_28102026.DAT"
    short.write_text(FIXTURE.read_text()[:-2])  # 38.06 becomes 38.0, without a newline.
    with pytest.raises(DeliveryParseError, match="incomplete final line"):
        parse_mto(short, date(2026, 10, 28))


def test_parse_mto_rejects_a_wrong_total_with_the_right_count(tmp_path: Path):
    wrong = tmp_path / "MTO_28102026.DAT"
    wrong.write_text(FIXTURE.read_text().replace("024680455", "024680456"))
    with pytest.raises(DeliveryParseError, match="header says 7 records and 24680456 delivered"):
        parse_mto(wrong, date(2026, 10, 28))


def test_parse_mto_rejects_a_file_without_the_header_line(tmp_path: Path):
    html = tmp_path / "MTO_28102026.DAT"
    html.write_text("<!DOCTYPE html>\n<html></html>\n")
    with pytest.raises(DeliveryParseError, match="no 10,MTO header"):
        parse_mto(html, date(2026, 10, 28))


# --- fetch -------------------------------------------------------------


@responses.activate
def test_fetch_nse_delivery_happy_path(tmp_path: Path):
    body = FIXTURE.read_bytes()
    responses.add(
        responses.GET,
        delivery.NSE_MTO_URL.format(ddmmyyyy="28102026"),
        body=body,
        status=200,
        content_type="application/octet-stream",
    )
    out = fetch_nse_delivery(date(2026, 10, 28), tmp_path)
    assert out == tmp_path / "MTO_28102026.DAT"
    assert out.read_bytes() == body


@responses.activate
def test_fetch_nse_delivery_idempotent(tmp_path: Path):
    existing = tmp_path / "MTO_28102026.DAT"
    existing.write_bytes(b"cached")
    assert fetch_nse_delivery(date(2026, 10, 28), tmp_path) == existing
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_nse_delivery_404_raises_notfound(tmp_path: Path):
    # What NSE answers before about 16:30 IST on the trade date, and on holidays.
    responses.add(
        responses.GET, delivery.NSE_MTO_URL.format(ddmmyyyy="28102026"), status=404
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_nse_delivery(date(2026, 10, 28), tmp_path, retries=1)
    assert not list(tmp_path.glob("*.DAT"))


@responses.activate
def test_fetch_nse_delivery_asks_nse_about_a_special_weekend_session(tmp_path: Path):
    # Saturday 2024-01-20 traded and is in nse/, but the calendar misses it, so
    # the fetch leaves the choice of days to its caller.
    responses.add(
        responses.GET, delivery.NSE_MTO_URL.format(ddmmyyyy="20012024"), body=b"x", status=200
    )
    assert fetch_nse_delivery(date(2024, 1, 20), tmp_path).read_bytes() == b"x"


@responses.activate
def test_fetch_nse_delivery_gives_up_after_retries(tmp_path: Path):
    url = delivery.NSE_MTO_URL.format(ddmmyyyy="28102026")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, status=503)
    with pytest.raises(BhavcopyFetchError):
        fetch_nse_delivery(date(2026, 10, 28), tmp_path, retries=2, backoff_seconds=0.0)
    assert not list(tmp_path.glob("*.DAT"))


# --- BSE ---------------------------------------------------------------


def test_parse_scbseall_names_rows_from_the_bhavcopy():
    df = parse_scbseall(BSE_FIXTURE, date(2026, 10, 28), BSE_BHAVCOPY)
    assert df.schema == delivery.DELIVERY_SCHEMA
    # ANDHRAPET (X), ANSALAPI (Z), BHAGGAS (XT) and LIQUIDETF (F) are not in bse/.
    assert df["symbol"].to_list() == [
        "20MICRONS", "AMBALALSA", "BOMDYEING", "RELIANCE", "SBIN", "TCS"
    ]
    assert df["series"].to_list() == ["B", "T", "B", "A", "A", "A"]
    assert set(df["date"]) == {date(2026, 10, 28)}


def test_parse_scbseall_known_value_reliance():
    # Scrip code 500325 is RELIANCE's FinInstrmId; the volume is its TtlTradgVol.
    df = parse_scbseall(BSE_FIXTURE, date(2026, 10, 28), BSE_BHAVCOPY)
    row = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert row["volume"] == 2297453
    assert row["delivery_qty"] == 1271387
    assert row["delivery_pct"] == pytest.approx(55.34)


def test_parse_scbseall_reads_columns_by_position(tmp_path: Path):
    # The regenerated 2024-04-25 file spells the header DAY`S, and files before
    # it pad quantities to 9 digits instead of 16.
    old = tmp_path / "SCBSEALL28102026.TXT"
    old.write_text(
        BSE_FIXTURE.read_text()
        .replace("DAY'S", "DAY`S")
        .replace("|0000000002297453|", "|002297453|")
        .replace("|0000000001271387|", "|001271387|")
    )
    assert parse_scbseall(old, date(2026, 10, 28), BSE_BHAVCOPY).equals(
        parse_scbseall(BSE_FIXTURE, date(2026, 10, 28), BSE_BHAVCOPY)
    )


def test_parse_scbseall_rejects_a_file_for_another_day():
    with pytest.raises(DeliveryParseError, match=r"records dated \['28102026'\]"):
        parse_scbseall(BSE_FIXTURE, date(2026, 10, 29), BSE_BHAVCOPY)


def test_parse_scbseall_rejects_a_bhavcopy_for_another_day(tmp_path: Path):
    bhav = tmp_path / "bhav.csv"
    bhav.write_text(
        BSE_BHAVCOPY.read_text().replace("2026-10-28,2026-10-28,", "2026-10-27,2026-10-27,")
    )
    with pytest.raises(DeliveryParseError, match=r"TradDt \['2026-10-27'\]"):
        parse_scbseall(BSE_FIXTURE, date(2026, 10, 28), bhav)


def test_parse_scbseall_rejects_a_short_record(tmp_path: Path):
    short = tmp_path / "SCBSEALL28102026.TXT"
    short.write_text(BSE_FIXTURE.read_text().rsplit("|", 1)[0] + "\r\n")
    with pytest.raises(DeliveryParseError, match="does not have 7 fields"):
        parse_scbseall(short, date(2026, 10, 28), BSE_BHAVCOPY)


def test_parse_scbseall_rejects_a_file_without_the_header_line(tmp_path: Path):
    html = tmp_path / "SCBSEALL28102026.TXT"
    html.write_text("<!DOCTYPE html>\n<html></html>\n")
    with pytest.raises(DeliveryParseError, match="no SCBSEALL header"):
        parse_scbseall(html, date(2026, 10, 28), BSE_BHAVCOPY)


def _zip_of(member: str, body: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, body)
    return buf.getvalue()


@responses.activate
@pytest.mark.parametrize("member", ["SCBSEALL2810.TXT", "SCBSEALL2810.txt"])
def test_fetch_bse_delivery_extracts_the_text_file(tmp_path: Path, member: str):
    # The member has no year in its name, and is .txt in some years.
    body = BSE_FIXTURE.read_bytes()
    responses.add(
        responses.GET,
        "https://www.bseindia.com/BSEDATA/gross/2026/SCBSEALL2810.zip",
        body=_zip_of(member, body),
        status=200,
        content_type="application/x-zip-compressed",
    )
    out = fetch_bse_delivery(date(2026, 10, 28), tmp_path)
    assert out == tmp_path / "SCBSEALL28102026.TXT"
    assert out.read_bytes() == body


@responses.activate
def test_fetch_bse_delivery_idempotent(tmp_path: Path):
    existing = tmp_path / "SCBSEALL28102026.TXT"
    existing.write_bytes(b"cached")
    assert fetch_bse_delivery(date(2026, 10, 28), tmp_path) == existing
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_bse_delivery_404_raises_notfound(tmp_path: Path):
    # What BSE answers before about 18:15 IST on the trade date, and on holidays.
    responses.add(
        responses.GET, delivery.BSE_SCBSEALL_URL.format(yyyy=2026, ddmm="2810"), status=404
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_bse_delivery(date(2026, 10, 28), tmp_path, retries=1)
    assert not list(tmp_path.iterdir())


@responses.activate
def test_fetch_bse_delivery_rejects_a_payload_that_is_not_a_zip(tmp_path: Path):
    responses.add(
        responses.GET,
        delivery.BSE_SCBSEALL_URL.format(yyyy=2026, ddmm="2810"),
        body=b"<!doctype html><html></html>",
        status=200,
    )
    with pytest.raises(BhavcopyFetchError, match="invalid zip payload"):
        fetch_bse_delivery(date(2026, 10, 28), tmp_path)
    assert not list(tmp_path.iterdir())
