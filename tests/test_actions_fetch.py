import json
from datetime import date
from pathlib import Path

import pytest
import responses

from pipeline.actions import fetch as af
from pipeline.actions.fetch import (
    ActionsFetchError,
    _date_chunks,
    fetch_bse_actions,
    fetch_nse_actions,
)


# --- chunking ----------------------------------------------------------


def test_date_chunks_single_chunk_when_short_range():
    chunks = _date_chunks(date(2024, 1, 1), date(2024, 2, 1), 85)
    assert chunks == [(date(2024, 1, 1), date(2024, 2, 1))]


def test_date_chunks_splits_long_range():
    # 200-day span chunked at 85: should be 3 chunks, no gaps, no overlap
    chunks = _date_chunks(date(2024, 1, 1), date(2024, 7, 19), 85)
    assert len(chunks) == 3
    # No gap: each chunk's end + 1 day == next chunk's start
    for (_, end), (start, _) in zip(chunks, chunks[1:]):
        assert (start - end).days == 1
    # First start + last end cover full range
    assert chunks[0][0] == date(2024, 1, 1)
    assert chunks[-1][1] == date(2024, 7, 19)


def test_date_chunks_inverted_range_via_fetch_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        fetch_nse_actions(date(2024, 2, 1), date(2024, 1, 1), tmp_path)


# --- NSE ---------------------------------------------------------------


def _add_nse_warmup():
    responses.add(responses.GET, af.NSE_HOME_URL, body=b"<html></html>", status=200)


@responses.activate
def test_fetch_nse_actions_happy_path(tmp_path: Path):
    _add_nse_warmup()
    payload = [
        {"symbol": "HCLTECH", "subject": "Interim Dividend - Rs 12 Per Share",
         "exDate": "19-Jan-2024", "recDate": "20-Jan-2024",
         "series": "EQ", "isin": "INE860A01027", "comp": "HCL Technologies"},
    ]
    responses.add(
        responses.GET, af.NSE_ACTIONS_URL, json=payload, status=200,
    )

    out = fetch_nse_actions(date(2024, 1, 1), date(2024, 2, 1), tmp_path)
    assert out == payload
    # warm-up + 1 chunk = 2 calls
    assert len(responses.calls) == 2
    # Chunked file cached on disk
    cached = tmp_path / "nse_20240101_20240201.json"
    assert cached.exists()
    assert json.loads(cached.read_text()) == payload


@responses.activate
def test_fetch_nse_actions_uses_cache_skips_http(tmp_path: Path):
    cached = tmp_path / "nse_20240101_20240201.json"
    cached.write_text(json.dumps([{"symbol": "X", "subject": "cached"}]))

    out = fetch_nse_actions(date(2024, 1, 1), date(2024, 2, 1), tmp_path)
    assert out == [{"symbol": "X", "subject": "cached"}]
    # All chunks cached: skip warm-up, zero HTTP
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_nse_actions_chunks_long_range_and_concats(tmp_path: Path):
    _add_nse_warmup()
    # Each chunk returns one distinct record so we can verify concat order
    responses.add(responses.GET, af.NSE_ACTIONS_URL,
                  json=[{"symbol": "A", "subject": "Bonus 1:1"}], status=200)
    responses.add(responses.GET, af.NSE_ACTIONS_URL,
                  json=[{"symbol": "B", "subject": "Bonus 2:1"}], status=200)
    responses.add(responses.GET, af.NSE_ACTIONS_URL,
                  json=[{"symbol": "C", "subject": "Bonus 3:1"}], status=200)

    out = fetch_nse_actions(date(2024, 1, 1), date(2024, 7, 19), tmp_path)
    assert [r["symbol"] for r in out] == ["A", "B", "C"]
    # 1 warmup + 3 chunks
    assert len(responses.calls) == 4
    # All three chunk files cached
    assert len(list(tmp_path.glob("nse_*.json"))) == 3


@responses.activate
def test_fetch_nse_actions_retries_on_5xx(tmp_path: Path):
    _add_nse_warmup()
    responses.add(responses.GET, af.NSE_ACTIONS_URL, status=503)
    responses.add(responses.GET, af.NSE_ACTIONS_URL, json=[{"x": 1}], status=200)

    out = fetch_nse_actions(
        date(2024, 1, 1), date(2024, 2, 1), tmp_path,
        retries=3, backoff_seconds=0.0,
    )
    assert out == [{"x": 1}]


@responses.activate
def test_fetch_nse_actions_gives_up_after_retries(tmp_path: Path):
    _add_nse_warmup()
    responses.add(responses.GET, af.NSE_ACTIONS_URL, status=503)
    responses.add(responses.GET, af.NSE_ACTIONS_URL, status=503)

    with pytest.raises(ActionsFetchError):
        fetch_nse_actions(
            date(2024, 1, 1), date(2024, 2, 1), tmp_path,
            retries=2, backoff_seconds=0.0,
        )


@responses.activate
def test_fetch_nse_actions_non_json_raises(tmp_path: Path):
    _add_nse_warmup()
    responses.add(responses.GET, af.NSE_ACTIONS_URL,
                  body=b"<html>blocked</html>", status=200)
    with pytest.raises(ActionsFetchError, match="non-JSON"):
        fetch_nse_actions(date(2024, 1, 1), date(2024, 2, 1), tmp_path,
                          retries=1)


@responses.activate
def test_fetch_nse_actions_unwraps_data_key(tmp_path: Path):
    # Some NSE responses wrap rows under a top-level key; should still work
    _add_nse_warmup()
    responses.add(
        responses.GET, af.NSE_ACTIONS_URL,
        json={"data": [{"symbol": "Z", "subject": "Bonus 1:1"}]},
        status=200,
    )
    out = fetch_nse_actions(date(2024, 1, 1), date(2024, 2, 1), tmp_path)
    assert out == [{"symbol": "Z", "subject": "Bonus 1:1"}]


# --- BSE ---------------------------------------------------------------


@responses.activate
def test_fetch_bse_actions_happy_path(tmp_path: Path):
    payload = [
        {"scrip_code": 532281, "short_name": "HCLTECH",
         "Ex_date": "19 Jan 2024", "exdate": "20240119",
         "Purpose": "Interim Dividend - Rs. - 12.0000",
         "RD_Date": "20 Jan 2024", "long_name": "HCL Technologies Ltd"},
    ]
    url = af.BSE_ACTIONS_URL.format(fdate="20240101", tdate="20241231")
    responses.add(responses.GET, url, json=payload, status=200)

    out = fetch_bse_actions(date(2024, 1, 1), date(2024, 12, 31), tmp_path)
    assert out == payload
    assert (tmp_path / "bse_20240101_20241231.json").exists()


@responses.activate
def test_fetch_bse_actions_idempotent(tmp_path: Path):
    cached = tmp_path / "bse_20240101_20241231.json"
    cached.write_text(json.dumps([{"cached": True}]))

    out = fetch_bse_actions(date(2024, 1, 1), date(2024, 12, 31), tmp_path)
    assert out == [{"cached": True}]
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_bse_actions_404_raises(tmp_path: Path):
    url = af.BSE_ACTIONS_URL.format(fdate="20240101", tdate="20241231")
    responses.add(responses.GET, url, status=404)
    with pytest.raises(ActionsFetchError):
        fetch_bse_actions(date(2024, 1, 1), date(2024, 12, 31), tmp_path,
                          retries=1)


@responses.activate
def test_fetch_bse_actions_retries_on_5xx(tmp_path: Path):
    url = af.BSE_ACTIONS_URL.format(fdate="20240101", tdate="20241231")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, json=[{"ok": 1}], status=200)

    out = fetch_bse_actions(
        date(2024, 1, 1), date(2024, 12, 31), tmp_path,
        retries=3, backoff_seconds=0.0,
    )
    assert out == [{"ok": 1}]


def test_fetch_bse_actions_inverted_range_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        fetch_bse_actions(date(2024, 12, 31), date(2024, 1, 1), tmp_path)
