"""Bhavcopy fetchers for Indian exchanges.

NSE serves the official EOD bhavcopy as a zipped CSV; BSE serves the same
SEBI-CMTS schema as a plain CSV. Both endpoints require browser-like headers
to avoid bot blocks. Caller owns subsequent parsing.
"""

from __future__ import annotations

import time
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Literal

import requests

from pipeline import holidays

Exchange = Literal["NSE", "BSE"]

NSE_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

BSE_BHAVCOPY_URL = (
    "https://www.bseindia.com/download/BhavCopy/Equity/"
    "BhavCopy_BSE_CM_0_0_0_{yyyymmdd}_F_0000.CSV"
)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

_BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
}


class BhavcopyNotFoundError(RuntimeError):
    """Bhavcopy returned 404 — usually non-trading day or not yet published."""


class BhavcopyFetchError(RuntimeError):
    """Network or HTTP failure after retries."""


def fetch_nse(
    d: date,
    dest_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Path:
    """Download NSE bhavcopy for `d` into `dest_dir`. Returns path to extracted CSV.

    Idempotent: if the CSV already exists, returns its path without re-downloading.
    Raises BhavcopyNotFoundError on 404, BhavcopyFetchError on other failures.
    """
    if not holidays.is_trading_day(d, "NSE"):
        raise BhavcopyNotFoundError(f"{d} is not an NSE trading day")

    dest_dir.mkdir(parents=True, exist_ok=True)
    yyyymmdd = d.strftime("%Y%m%d")
    csv_name = f"BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv"
    csv_path = dest_dir / csv_name
    if csv_path.exists():
        return csv_path

    url = NSE_BHAVCOPY_URL.format(yyyymmdd=yyyymmdd)
    body = _http_get_with_retry(url, _NSE_HEADERS, retries, backoff_seconds, timeout, session)
    return _extract_zip_to(body, dest_dir, csv_name)


def fetch_bse(
    d: date,
    dest_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Path:
    """Download BSE bhavcopy for `d` into `dest_dir`. Returns path to CSV.

    BSE serves a plain CSV (no zip wrapping). Idempotent: returns cached path
    if already present. Raises BhavcopyNotFoundError on 404.
    """
    if not holidays.is_trading_day(d, "BSE"):
        raise BhavcopyNotFoundError(f"{d} is not a BSE trading day")

    dest_dir.mkdir(parents=True, exist_ok=True)
    yyyymmdd = d.strftime("%Y%m%d")
    csv_name = f"BhavCopy_BSE_CM_0_0_0_{yyyymmdd}_F_0000.csv"
    csv_path = dest_dir / csv_name
    if csv_path.exists():
        return csv_path

    url = BSE_BHAVCOPY_URL.format(yyyymmdd=yyyymmdd)
    body = _http_get_with_retry(url, _BSE_HEADERS, retries, backoff_seconds, timeout, session)

    # BSE serves its HTML homepage with status 200 when the bhavcopy isn't published yet
    # (e.g. fetching same-day before EOD). Sniff the first non-whitespace bytes.
    head = body.lstrip()[:64].lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html") or head.startswith(b"<meta"):
        raise BhavcopyNotFoundError(
            f"BSE bhavcopy not yet published for {d} (server returned HTML): {url}"
        )

    csv_path.write_bytes(body)
    return csv_path


def _http_get_with_retry(
    url: str,
    headers: dict[str, str],
    retries: int,
    backoff_seconds: float,
    timeout: float,
    session: requests.Session | None,
) -> bytes:
    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                raise BhavcopyNotFoundError(f"bhavcopy not found (404): {url}")
            resp.raise_for_status()
            return resp.content
        except BhavcopyNotFoundError:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise BhavcopyFetchError(f"failed to fetch {url} after {retries} attempts: {last_exc}")


def _extract_zip_to(zip_bytes: bytes, dest_dir: Path, expected_csv: str) -> Path:
    try:
        zf = zipfile.ZipFile(BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise BhavcopyFetchError(f"invalid zip payload: {e}") from e
    with zf:
        names = zf.namelist()
        target = expected_csv if expected_csv in names else None
        if target is None:
            csvs = [n for n in names if n.lower().endswith(".csv")]
            if not csvs:
                raise BhavcopyFetchError(f"no CSV found in zip; got {names}")
            target = csvs[0]
        zf.extract(target, dest_dir)
        return dest_dir / target


def fetch(
    d: date,
    dest_dir: Path,
    exchange: Exchange,
    **kwargs,
) -> Path:
    """Dispatch to fetch_nse / fetch_bse by exchange."""
    if exchange == "NSE":
        return fetch_nse(d, dest_dir, **kwargs)
    if exchange == "BSE":
        return fetch_bse(d, dest_dir, **kwargs)
    raise ValueError(f"unknown exchange {exchange!r}")
