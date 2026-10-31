"""NSE equity derivatives (F&O) end-of-day bhavcopy.

One row per contract per trading day: index and stock futures and options,
plus India VIX futures in the years NSE listed them. Two source formats:

- Before 2024-07-08: `fo{DD}{MMM}{YYYY}bhav.csv` (INSTRUMENT, SYMBOL,
  EXPIRY_DT, STRIKE_PR, OPTION_TYP, OHLC, SETTLE_PR, CONTRACTS, VAL_INLAKH,
  OPEN_INT, CHG_IN_OI, TIMESTAMP). Futures carry strike 0 and type XX.
- From 2024-07-08: the UDiFF `BhavCopy_NSE_FO_..._F_0000.csv`, which adds
  last price, previous close, underlying price, trade count and lot size.

Both map onto FO_SCHEMA. On 2024-07-05, when NSE published both, every row
matches: UDiFF `TtlTradgVol` is the contract count, open interest is in the
same units, and `TtlTrfVal` (rupees) is `VAL_INLAKH` x 1e5.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import requests

from pipeline.fetch import (
    _NSE_HEADERS,
    NSE_LEGACY_CUTOVER,
    _extract_zip_to,
    _http_get_with_retry,
)

NSE_FO_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
NSE_FO_URL_LEGACY = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{yyyy}/{mmm}/fo{dd}{mmm}{yyyy}bhav.csv.zip"
)

FO_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "instrument": pl.Utf8,
    "symbol": pl.Utf8,
    "expiry": pl.Date,
    "strike": pl.Float64,
    "option_type": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "settle_price": pl.Float64,
    "contracts": pl.Int64,
    "turnover": pl.Float64,
    "open_interest": pl.Int64,
    "oi_change": pl.Int64,
    "last": pl.Float64,
    "prev_close": pl.Float64,
    "underlying": pl.Float64,
    "trades": pl.Int64,
    "lot_size": pl.Int64,
}

FO_KEY = ["date", "instrument", "symbol", "expiry", "strike", "option_type"]

INSTRUMENTS = {"FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK", "FUTIVX"}
_UDIFF_INSTRUMENT = {"IDF": "FUTIDX", "IDO": "OPTIDX", "STF": "FUTSTK", "STO": "OPTSTK"}


class FoParseError(ValueError):
    """F&O file is for another day, has an unknown instrument, or repeats a contract."""


def fo_csv_name(d: date) -> str:
    if d < NSE_LEGACY_CUTOVER:
        mmm = d.strftime("%b").upper()
        return f"fo{d.day:02d}{mmm}{d.year}bhav.csv"
    return f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv"


def fetch_nse_fo(
    d: date,
    dest_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> Path:
    """Download the NSE F&O bhavcopy for `d` into `dest_dir`. Returns the CSV path.

    Idempotent: a cached CSV is returned without a request. Raises
    BhavcopyNotFoundError on 404 (holiday or not published yet) and
    BhavcopyFetchError on other failures. No calendar check: the caller
    picks the days.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = fo_csv_name(d)
    path = dest_dir / name
    if path.exists():
        return path
    if d < NSE_LEGACY_CUTOVER:
        mmm = d.strftime("%b").upper()
        url = NSE_FO_URL_LEGACY.format(yyyy=d.year, mmm=mmm, dd=f"{d.day:02d}")
    else:
        url = NSE_FO_URL.format(yyyymmdd=d.strftime("%Y%m%d"))
    body = _http_get_with_retry(url, _NSE_HEADERS, retries, backoff_seconds, timeout, session)
    return _extract_zip_to(body, dest_dir, name)


def parse_fo(path: Path, d: date) -> pl.DataFrame:
    """Read one F&O bhavcopy, either format, into FO_SCHEMA sorted by FO_KEY order.

    Raises FoParseError when the file's trade date is not `d`, an instrument
    type is unknown, or a contract appears twice.
    """
    raw = pl.read_csv(path, infer_schema=False, truncate_ragged_lines=True)
    if "FinInstrmTp" in raw.columns:
        df = _parse_udiff(raw)
    elif "INSTRUMENT" in raw.columns:
        df = _parse_legacy(raw)
    else:
        raise FoParseError(f"{path.name}: unrecognised header {raw.columns[:5]}")

    dates = df["date"].unique().to_list()
    if dates != [d]:
        raise FoParseError(f"{path.name}: trade dates {dates} are not [{d}]")
    unknown = set(df["instrument"].unique().to_list()) - INSTRUMENTS
    if unknown:
        raise FoParseError(f"{path.name}: unknown instruments {sorted(unknown)}")
    dupes = df.height - df.unique(FO_KEY).height
    if dupes:
        raise FoParseError(f"{path.name}: {dupes} repeated contracts")
    return df.sort(["symbol", "instrument", "expiry", "strike", "option_type"], nulls_last=True)


def _num(col: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col(col).str.strip_chars().cast(pl.Float64, strict=True).cast(dtype)


def _parse_udiff(raw: pl.DataFrame) -> pl.DataFrame:
    tp = pl.col("FinInstrmTp").str.strip_chars()
    is_opt = tp.is_in(["IDO", "STO"])
    return raw.select(
        pl.col("TradDt").str.strip_chars().str.to_date("%Y-%m-%d").alias("date"),
        tp.replace(_UDIFF_INSTRUMENT).alias("instrument"),
        pl.col("TckrSymb").str.strip_chars().alias("symbol"),
        pl.col("XpryDt").str.strip_chars().str.to_date("%Y-%m-%d").alias("expiry"),
        pl.when(is_opt).then(_num("StrkPric", pl.Float64)).alias("strike"),
        pl.when(is_opt).then(pl.col("OptnTp").str.strip_chars()).alias("option_type"),
        _num("OpnPric", pl.Float64).alias("open"),
        _num("HghPric", pl.Float64).alias("high"),
        _num("LwPric", pl.Float64).alias("low"),
        _num("ClsPric", pl.Float64).alias("close"),
        _num("SttlmPric", pl.Float64).alias("settle_price"),
        _num("TtlTradgVol", pl.Int64).alias("contracts"),
        _num("TtlTrfVal", pl.Float64).alias("turnover"),
        _num("OpnIntrst", pl.Int64).alias("open_interest"),
        _num("ChngInOpnIntrst", pl.Int64).alias("oi_change"),
        _num("LastPric", pl.Float64).alias("last"),
        _num("PrvsClsgPric", pl.Float64).alias("prev_close"),
        _num("UndrlygPric", pl.Float64).alias("underlying"),
        _num("TtlNbOfTxsExctd", pl.Int64).alias("trades"),
        _num("NewBrdLotQty", pl.Int64).alias("lot_size"),
    ).cast(FO_SCHEMA)


def _parse_legacy(raw: pl.DataFrame) -> pl.DataFrame:
    inst = pl.col("INSTRUMENT").str.strip_chars()
    is_opt = inst.str.starts_with("OPT")
    return raw.select(
        pl.col("TIMESTAMP").str.strip_chars().str.to_titlecase().str.to_date("%d-%b-%Y").alias("date"),
        inst.alias("instrument"),
        pl.col("SYMBOL").str.strip_chars().alias("symbol"),
        pl.col("EXPIRY_DT").str.strip_chars().str.to_titlecase().str.to_date("%d-%b-%Y").alias("expiry"),
        pl.when(is_opt).then(_num("STRIKE_PR", pl.Float64)).alias("strike"),
        pl.when(is_opt).then(pl.col("OPTION_TYP").str.strip_chars()).alias("option_type"),
        _num("OPEN", pl.Float64).alias("open"),
        _num("HIGH", pl.Float64).alias("high"),
        _num("LOW", pl.Float64).alias("low"),
        _num("CLOSE", pl.Float64).alias("close"),
        _num("SETTLE_PR", pl.Float64).alias("settle_price"),
        _num("CONTRACTS", pl.Int64).alias("contracts"),
        (_num("VAL_INLAKH", pl.Float64) * 1e5).round(2).alias("turnover"),
        _num("OPEN_INT", pl.Int64).alias("open_interest"),
        _num("CHG_IN_OI", pl.Int64).alias("oi_change"),
        pl.lit(None, pl.Float64).alias("last"),
        pl.lit(None, pl.Float64).alias("prev_close"),
        pl.lit(None, pl.Float64).alias("underlying"),
        pl.lit(None, pl.Int64).alias("trades"),
        pl.lit(None, pl.Int64).alias("lot_size"),
    ).cast(FO_SCHEMA)
