import json
from pathlib import Path

import pytest
import responses

from pipeline.actions import (
    ActionsFetchError,
    build_scrip_to_isin,
    fetch_bse_scrip_master,
    load_bse_scrip_to_isin,
    parse_bse_record,
)
from pipeline.actions.scrip_map import BSE_SCRIP_MASTER_URL, CACHE_FILENAME

SAMPLE = [
    {"SCRIP_CD": "532281", "Scrip_Name": "HCL Technologies Ltd",
     "ISIN_NUMBER": "INE860A01027", "scrip_id": "HCLTECH",
     "Status": "Active", "GROUP": "A"},
    {"SCRIP_CD": "500325", "Scrip_Name": "Reliance Industries Ltd",
     "ISIN_NUMBER": "INE002A01018", "scrip_id": "RELIANCE",
     "Status": "Active", "GROUP": "A"},
    # Missing ISIN should be skipped
    {"SCRIP_CD": "999999", "Scrip_Name": "Bogus Ltd",
     "ISIN_NUMBER": "", "Status": "Active"},
]


# --- build_scrip_to_isin ----------------------------------------------


def test_build_scrip_to_isin_drops_empty_isin():
    m = build_scrip_to_isin(SAMPLE)
    assert m == {"532281": "INE860A01027", "500325": "INE002A01018"}


def test_build_scrip_to_isin_strips_keys():
    m = build_scrip_to_isin([
        {"SCRIP_CD": " 500325 ", "ISIN_NUMBER": " INE002A01018 "},
    ])
    assert m == {"500325": "INE002A01018"}


def test_build_scrip_to_isin_handles_int_keys():
    # BSE returns SCRIP_CD as string, but be defensive against int input
    m = build_scrip_to_isin([{"SCRIP_CD": 532281, "ISIN_NUMBER": "INE860A01027"}])
    assert m == {"532281": "INE860A01027"}


# --- fetch_bse_scrip_master -------------------------------------------


@responses.activate
def test_fetch_bse_scrip_master_happy_path(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, json=SAMPLE, status=200)

    out = fetch_bse_scrip_master(tmp_path)
    assert out == SAMPLE
    cached = tmp_path / CACHE_FILENAME
    assert cached.exists()
    assert json.loads(cached.read_text()) == SAMPLE


@responses.activate
def test_fetch_bse_scrip_master_uses_cache(tmp_path: Path):
    cached = tmp_path / CACHE_FILENAME
    cached.write_text(json.dumps([{"cached": True}]))

    out = fetch_bse_scrip_master(tmp_path)
    assert out == [{"cached": True}]
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_bse_scrip_master_refresh_overrides_cache(tmp_path: Path):
    cached = tmp_path / CACHE_FILENAME
    cached.write_text(json.dumps([{"old": True}]))
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, json=SAMPLE, status=200)

    out = fetch_bse_scrip_master(tmp_path, refresh=True)
    assert out == SAMPLE
    assert json.loads(cached.read_text()) == SAMPLE


@responses.activate
def test_fetch_bse_scrip_master_retries_on_5xx(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, status=503)
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, json=SAMPLE, status=200)

    out = fetch_bse_scrip_master(tmp_path, retries=3, backoff_seconds=0.0)
    assert out == SAMPLE


@responses.activate
def test_fetch_bse_scrip_master_gives_up(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, status=503)
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, status=503)

    with pytest.raises(ActionsFetchError):
        fetch_bse_scrip_master(tmp_path, retries=2, backoff_seconds=0.0)


@responses.activate
def test_fetch_bse_scrip_master_unexpected_shape(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL,
                  json={"data": SAMPLE}, status=200)
    with pytest.raises(ActionsFetchError, match="unexpected JSON shape"):
        fetch_bse_scrip_master(tmp_path, retries=1)


@responses.activate
def test_fetch_bse_scrip_master_non_json_raises(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL,
                  body=b"<html>blocked</html>", status=200)
    with pytest.raises(ActionsFetchError, match="non-JSON"):
        fetch_bse_scrip_master(tmp_path, retries=1)


# --- load_bse_scrip_to_isin (one-shot helper) -------------------------


@responses.activate
def test_load_bse_scrip_to_isin_returns_dict(tmp_path: Path):
    responses.add(responses.GET, BSE_SCRIP_MASTER_URL, json=SAMPLE, status=200)
    m = load_bse_scrip_to_isin(tmp_path)
    assert m["532281"] == "INE860A01027"
    assert m["500325"] == "INE002A01018"
    assert "999999" not in m


# --- integration: parse_bse_record uses scrip_to_isin -----------------


def test_parse_bse_record_joins_isin_when_map_provided():
    rec = {
        "scrip_code": 532281, "short_name": "HCLTECH",
        "Ex_date": "19 Jan 2024", "exdate": "20240119",
        "Purpose": "Interim Dividend - Rs. - 12.0000",
        "RD_Date": "20 Jan 2024", "long_name": "HCL Technologies Ltd",
    }
    a = parse_bse_record(rec, scrip_to_isin={"532281": "INE860A01027"})
    assert a.isin == "INE860A01027"


def test_parse_bse_record_isin_none_without_map():
    rec = {
        "scrip_code": 532281, "short_name": "HCLTECH",
        "Ex_date": "19 Jan 2024", "exdate": "20240119",
        "Purpose": "Interim Dividend - Rs. - 12.0000",
    }
    a = parse_bse_record(rec)
    assert a.isin is None


def test_parse_bse_record_unknown_scrip_isin_none():
    rec = {
        "scrip_code": 111111, "short_name": "UNKNOWN",
        "Ex_date": "19 Jan 2024", "exdate": "20240119",
        "Purpose": "Bonus 1:1",
    }
    a = parse_bse_record(rec, scrip_to_isin={"532281": "INE860A01027"})
    assert a.isin is None


def test_parse_bse_record_str_scrip_code_works():
    # Some feeds send scrip_code as string already
    rec = {
        "scrip_code": "532281", "short_name": "HCLTECH",
        "Ex_date": "19 Jan 2024", "exdate": "20240119",
        "Purpose": "Bonus 1:1",
    }
    a = parse_bse_record(rec, scrip_to_isin={"532281": "INE860A01027"})
    assert a.isin == "INE860A01027"
