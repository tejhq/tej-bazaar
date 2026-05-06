import io
import zipfile
from datetime import date
from pathlib import Path

import pytest
import responses

from pipeline import fetch
from pipeline.fetch import (
    BhavcopyFetchError,
    BhavcopyNotFoundError,
    fetch_bse,
    fetch_nse,
)


def _make_zip(csv_name: str, csv_body: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(csv_name, csv_body)
    return buf.getvalue()


@responses.activate
def test_fetch_nse_happy_path(tmp_path: Path):
    d = date(2025, 4, 30)  # Wednesday, trading day
    csv_name = "BhavCopy_NSE_CM_0_0_0_20250430_F_0000.csv"
    payload = _make_zip(csv_name, b"col1,col2\n1,2\n")
    responses.add(
        responses.GET,
        fetch.NSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        body=payload,
        status=200,
        content_type="application/zip",
    )

    out = fetch_nse(d, tmp_path)
    assert out == tmp_path / csv_name
    assert out.read_bytes() == b"col1,col2\n1,2\n"


@responses.activate
def test_fetch_nse_idempotent(tmp_path: Path):
    d = date(2025, 4, 30)
    csv_name = "BhavCopy_NSE_CM_0_0_0_20250430_F_0000.csv"
    existing = tmp_path / csv_name
    existing.write_bytes(b"already here")

    out = fetch_nse(d, tmp_path)
    assert out == existing
    assert out.read_bytes() == b"already here"
    assert len(responses.calls) == 0  # no HTTP call when cached


@responses.activate
def test_fetch_nse_404_raises_notfound(tmp_path: Path):
    d = date(2025, 4, 30)
    responses.add(
        responses.GET,
        fetch.NSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        status=404,
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_nse(d, tmp_path, retries=1)


def test_fetch_nse_non_trading_day_raises_notfound(tmp_path: Path):
    # Aug 15 2025 = Independence Day, no HTTP should be made
    with pytest.raises(BhavcopyNotFoundError, match="not an NSE trading day"):
        fetch_nse(date(2025, 8, 15), tmp_path)


@responses.activate
def test_fetch_nse_retries_on_5xx(tmp_path: Path):
    d = date(2025, 4, 30)
    csv_name = "BhavCopy_NSE_CM_0_0_0_20250430_F_0000.csv"
    payload = _make_zip(csv_name, b"col1\n1\n")
    url = fetch.NSE_BHAVCOPY_URL.format(yyyymmdd="20250430")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, body=payload, status=200, content_type="application/zip")

    out = fetch_nse(d, tmp_path, retries=3, backoff_seconds=0.0)
    assert out.exists()
    assert len(responses.calls) == 2


@responses.activate
def test_fetch_nse_gives_up_after_retries(tmp_path: Path):
    d = date(2025, 4, 30)
    url = fetch.NSE_BHAVCOPY_URL.format(yyyymmdd="20250430")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, status=503)

    with pytest.raises(BhavcopyFetchError):
        fetch_nse(d, tmp_path, retries=2, backoff_seconds=0.0)


@responses.activate
def test_fetch_nse_extracts_first_csv_when_name_mismatches(tmp_path: Path):
    d = date(2025, 4, 30)
    actual_name = "renamed_bhavcopy.csv"
    payload = _make_zip(actual_name, b"col\n1\n")
    responses.add(
        responses.GET,
        fetch.NSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        body=payload,
        status=200,
        content_type="application/zip",
    )
    out = fetch_nse(d, tmp_path)
    # Falls back to extracting whatever CSV was in the zip and returns its path
    assert out == tmp_path / actual_name
    assert out.exists()


# --- BSE ---------------------------------------------------------------


@responses.activate
def test_fetch_bse_happy_path(tmp_path: Path):
    d = date(2025, 4, 30)
    csv_name = "BhavCopy_BSE_CM_0_0_0_20250430_F_0000.csv"
    body = b"TradDt,TckrSymb\n2025-04-30,RELIANCE\n"
    responses.add(
        responses.GET,
        fetch.BSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        body=body,
        status=200,
        content_type="application/octet-stream",
    )

    out = fetch_bse(d, tmp_path)
    assert out == tmp_path / csv_name
    assert out.read_bytes() == body


@responses.activate
def test_fetch_bse_idempotent(tmp_path: Path):
    d = date(2025, 4, 30)
    csv_name = "BhavCopy_BSE_CM_0_0_0_20250430_F_0000.csv"
    existing = tmp_path / csv_name
    existing.write_bytes(b"cached")

    out = fetch_bse(d, tmp_path)
    assert out == existing
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_bse_404_raises_notfound(tmp_path: Path):
    d = date(2025, 4, 30)
    responses.add(
        responses.GET,
        fetch.BSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        status=404,
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_bse(d, tmp_path, retries=1)


def test_fetch_bse_non_trading_day_raises(tmp_path: Path):
    with pytest.raises(BhavcopyNotFoundError, match="not a BSE trading day"):
        fetch_bse(date(2025, 8, 15), tmp_path)


@responses.activate
def test_fetch_bse_html_response_treated_as_notfound(tmp_path: Path):
    # When a bhavcopy isn't published yet, BSE serves its homepage with status 200.
    # Must be detected as "not yet published", not parsed as CSV.
    d = date(2025, 4, 30)
    html = b"<!doctype html>\n<html><head><meta charset='utf-8'><title>BSE</title>"
    responses.add(
        responses.GET,
        fetch.BSE_BHAVCOPY_URL.format(yyyymmdd="20250430"),
        body=html,
        status=200,
        content_type="text/html",
    )
    with pytest.raises(BhavcopyNotFoundError, match="not yet published"):
        fetch_bse(d, tmp_path)
    # And nothing should have been written
    assert not list(tmp_path.glob("*.csv"))


@responses.activate
def test_fetch_bse_retries_on_5xx(tmp_path: Path):
    d = date(2025, 4, 30)
    url = fetch.BSE_BHAVCOPY_URL.format(yyyymmdd="20250430")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, body=b"col\n1\n", status=200)

    out = fetch_bse(d, tmp_path, retries=3, backoff_seconds=0.0)
    assert out.exists()
    assert len(responses.calls) == 2
