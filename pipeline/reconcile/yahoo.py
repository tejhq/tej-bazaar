"""Fetch Yahoo Finance daily adjusted close via the v8 chart API.

Endpoint: `query2.finance.yahoo.com/v8/finance/chart/<symbol>?...`. Returns
a JSON envelope with parallel arrays of timestamps, OHLCV, and `adjclose`
already back-adjusted for splits and (cash) dividends. We treat `adjclose`
as the reference series our `back_adjust` output should match.

Indian symbols carry an exchange suffix: `.NS` for NSE, `.BO` for BSE.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

import polars as pl
import requests

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


class YahooFetchError(RuntimeError):
    pass


def fetch_yahoo_adjusted(
    symbol: str,
    exchange: str,
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> pl.DataFrame:
    """Pull daily adjclose from Yahoo for `symbol` between `start` and `end`.

    Returns a DataFrame with columns (date, yahoo_close, yahoo_adjclose).
    Rows where Yahoo reports null close (suspended days) are dropped.
    """
    suffix = EXCHANGE_SUFFIX.get(exchange.upper())
    if suffix is None:
        raise YahooFetchError(f"unsupported exchange: {exchange}")
    yahoo_symbol = f"{symbol}{suffix}"

    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    # +1 day so Yahoo includes `end` itself; their period2 is exclusive of the boundary tick.
    period2 = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    sess = session or requests.Session()
    url = YAHOO_CHART_URL.format(symbol=yahoo_symbol)
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history,div,splits",
        "includeAdjustedClose": "true",
    }

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = sess.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if resp.status_code == 429:
                # Yahoo throttles aggressively. Use much longer cooldown
                # than transient 5xx so we actually clear the bucket.
                raise _Throttled(f"HTTP 429: {resp.text[:200]}")
            if resp.status_code >= 500:
                raise YahooFetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            payload = resp.json()
            return _parse_chart_payload(payload, yahoo_symbol)
        except _Throttled as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(10.0 * (attempt + 1))  # 10s, 20s, 30s
                continue
            raise YahooFetchError(f"fetch {yahoo_symbol} throttled: {e}") from e
        except (requests.RequestException, YahooFetchError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise YahooFetchError(f"fetch {yahoo_symbol} failed: {e}") from e
    raise YahooFetchError(f"fetch {yahoo_symbol} exhausted retries: {last_exc}")


class _Throttled(Exception):
    """Internal: Yahoo returned 429. Triggers long backoff."""


def _parse_chart_payload(payload: dict, yahoo_symbol: str) -> pl.DataFrame:
    chart = payload.get("chart", {})
    err = chart.get("error")
    if err:
        raise YahooFetchError(f"{yahoo_symbol} returned error: {err}")
    results = chart.get("result") or []
    if not results:
        raise YahooFetchError(f"{yahoo_symbol} returned empty result set")
    result = results[0]

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adj_block = (indicators.get("adjclose") or [{}])[0]
    closes = quote.get("close") or []
    adjcloses = adj_block.get("adjclose") or []

    if not timestamps:
        return pl.DataFrame(
            schema={"date": pl.Date, "yahoo_close": pl.Float64, "yahoo_adjclose": pl.Float64}
        )

    if len(closes) != len(timestamps) or len(adjcloses) != len(timestamps):
        raise YahooFetchError(
            f"{yahoo_symbol} array length mismatch: ts={len(timestamps)} "
            f"close={len(closes)} adj={len(adjcloses)}"
        )

    df = pl.DataFrame(
        {
            "ts": timestamps,
            "yahoo_close": closes,
            "yahoo_adjclose": adjcloses,
        }
    ).with_columns(
        date=pl.from_epoch("ts", time_unit="s").dt.convert_time_zone("Asia/Kolkata").dt.date()
    ).drop("ts").select(["date", "yahoo_close", "yahoo_adjclose"])

    return df.filter(pl.col("yahoo_close").is_not_null())
