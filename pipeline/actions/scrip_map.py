"""BSE scrip_code to ISIN mapping.

The BSE corporate-actions feed identifies companies by numeric scrip_code only;
the NSE feed gives ISIN directly. To keep BSE rows joinable across exchanges,
we fetch the BSE active-equities master (a single JSON call covering ~5k
scrips) and build a `scrip_code -> ISIN` lookup. The raw JSON is cached on
disk; callers pass `refresh=True` to re-pull (e.g. monthly).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from pipeline.actions.fetch import ActionsFetchError, _BSE_HEADERS

BSE_SCRIP_MASTER_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
)

CACHE_FILENAME = "bse_scrip_master.json"


def fetch_bse_scrip_master(
    cache_dir: Path,
    *,
    refresh: bool = False,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Return BSE active-equities master records. Cached as JSON in `cache_dir`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / CACHE_FILENAME
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())

    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(BSE_SCRIP_MASTER_URL, headers=_BSE_HEADERS, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ActionsFetchError(
                    f"unexpected JSON shape from {BSE_SCRIP_MASTER_URL}: "
                    f"{type(data).__name__}"
                )
            cached.write_text(json.dumps(data))
            return data
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as exc:
            raise ActionsFetchError(
                f"non-JSON response from {BSE_SCRIP_MASTER_URL}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise ActionsFetchError(
        f"failed to fetch BSE scrip master after {retries} attempts: {last_exc}"
    )


def build_scrip_to_isin(records: list[dict[str, Any]]) -> dict[str, str]:
    """Build {scrip_code (str) -> ISIN} from master records.

    Drops rows with missing or empty ISIN. Keys are stringified scrip codes
    so callers can look up using either int or str without coercion.
    """
    out: dict[str, str] = {}
    for r in records:
        code = r.get("SCRIP_CD")
        isin = (r.get("ISIN_NUMBER") or "").strip()
        if code is None or not isin:
            continue
        out[str(code).strip()] = isin
    return out


def load_bse_scrip_to_isin(
    cache_dir: Path,
    *,
    refresh: bool = False,
    **kwargs: Any,
) -> dict[str, str]:
    """One-shot helper: fetch (or read cache) and build scrip_code to ISIN map."""
    records = fetch_bse_scrip_master(cache_dir, refresh=refresh, **kwargs)
    return build_scrip_to_isin(records)
