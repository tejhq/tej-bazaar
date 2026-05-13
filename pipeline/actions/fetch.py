"""Fetch raw corporate-action JSON from NSE and BSE.

NSE requires a cookie warm-up against the homepage before any /api/* call
will succeed (CDN bot gate). Its corp-actions endpoint also caps each query
at roughly three months, so longer ranges are split into ~90-day chunks
and concatenated.

BSE has no cookie requirement and accepts a full year in one call.

Each chunk's response is cached to disk as JSON so repeated runs are cheap
and offline-replayable.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import requests

Exchange = Literal["NSE", "BSE"]

NSE_HOME_URL = "https://www.nseindia.com/"
NSE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"

BSE_ACTIONS_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
    "?Fdate={fdate}&TDate={tdate}&Purposecode=&strSearch=S"
    "&ddlcategorys=E&ddlindustrys=&segment=0&strType=CA"
)

NSE_CHUNK_DAYS = 85  # stay under the ~90-day server cap

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}

_BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
}


class ActionsFetchError(RuntimeError):
    """Network or HTTP failure after retries."""


def fetch_nse_actions(
    from_date: date,
    to_date: date,
    cache_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch NSE corporate actions between [from_date, to_date] inclusive.

    Splits the range into ~85-day chunks. Each chunk's raw JSON is cached
    under `cache_dir/nse_<from>_<to>.json` and reused on subsequent calls.
    Returns the concatenated list of records.
    """
    if from_date > to_date:
        raise ValueError(f"from_date {from_date} after to_date {to_date}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    chunks = _date_chunks(from_date, to_date, NSE_CHUNK_DAYS)
    needs_warmup = any(
        not (cache_dir / f"nse_{cf:%Y%m%d}_{ct:%Y%m%d}.json").exists()
        for cf, ct in chunks
    )
    if needs_warmup:
        _warm_nse_cookies(sess, timeout)

    out: list[dict[str, Any]] = []
    for chunk_from, chunk_to in chunks:
        cached = cache_dir / f"nse_{chunk_from:%Y%m%d}_{chunk_to:%Y%m%d}.json"
        if cached.exists():
            out.extend(json.loads(cached.read_text()))
            continue

        params = {
            "index": "equities",
            "from_date": chunk_from.strftime("%d-%m-%Y"),
            "to_date": chunk_to.strftime("%d-%m-%Y"),
        }
        body = _http_get_with_retry(
            NSE_ACTIONS_URL, _NSE_HEADERS, retries, backoff_seconds, timeout, sess,
            params=params,
        )
        records = _parse_json_list(body, NSE_ACTIONS_URL)
        cached.write_text(json.dumps(records))
        out.extend(records)

    return out


def fetch_bse_actions(
    from_date: date,
    to_date: date,
    cache_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch BSE corporate actions between [from_date, to_date] inclusive.

    BSE has no cookie requirement and accepts long ranges, so a single call
    is issued. Result is cached under `cache_dir/bse_<from>_<to>.json`.
    """
    if from_date > to_date:
        raise ValueError(f"from_date {from_date} after to_date {to_date}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"bse_{from_date:%Y%m%d}_{to_date:%Y%m%d}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = BSE_ACTIONS_URL.format(
        fdate=from_date.strftime("%Y%m%d"),
        tdate=to_date.strftime("%Y%m%d"),
    )
    body = _http_get_with_retry(
        url, _BSE_HEADERS, retries, backoff_seconds, timeout, session,
    )
    records = _parse_json_list(body, url)
    cached.write_text(json.dumps(records))
    return records


def fetch_actions(
    from_date: date,
    to_date: date,
    cache_dir: Path,
    exchange: Exchange,
    **kwargs,
) -> list[dict[str, Any]]:
    """Dispatch to fetch_nse_actions / fetch_bse_actions by exchange."""
    if exchange == "NSE":
        return fetch_nse_actions(from_date, to_date, cache_dir, **kwargs)
    if exchange == "BSE":
        return fetch_bse_actions(from_date, to_date, cache_dir, **kwargs)
    raise ValueError(f"unknown exchange {exchange!r}")


def _warm_nse_cookies(sess: requests.Session, timeout: float) -> None:
    """Hit the NSE homepage so the CDN sets the cookies that /api/* requires."""
    try:
        sess.get(NSE_HOME_URL, headers=_NSE_HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        raise ActionsFetchError(f"NSE cookie warm-up failed: {exc}") from exc


def _date_chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = start
    step = timedelta(days=days - 1)
    while cur <= end:
        chunk_end = min(cur + step, end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out


def _http_get_with_retry(
    url: str,
    headers: dict[str, str],
    retries: int,
    backoff_seconds: float,
    timeout: float,
    session: requests.Session | None,
    *,
    params: dict[str, str] | None = None,
) -> bytes:
    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise ActionsFetchError(f"failed to fetch {url} after {retries} attempts: {last_exc}")


def _parse_json_list(body: bytes, url: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ActionsFetchError(f"non-JSON response from {url}: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "Table", "rows"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    raise ActionsFetchError(f"unexpected JSON shape from {url}: {type(data).__name__}")
