"""NSE and BSE delivery positions, from NSE's MTO file and BSE's SCBSEALL file.

NSE publishes a security-wise delivery position file for each trading day at
`archives.nseindia.com/archives/equities/mto/MTO_DDMMYYYY.DAT`, around 16:30
to 17:00 IST, and answers 404 before then and on holidays. Since 2010 every
record line has the same shape:

    20,SrNo,SYMBOL,SERIES,QtyTraded,DelivQty,DelivPct

The header lines above the records changed in 2019 and are ignored, except
`10,MTO,DDMMYYYY,<sum of DelivQty>,<record count>`, which is checked against
the records so a misdated or short file is never written.

The file has no ISIN. Rows join to `nse/` on (date, symbol, series), and
`volume` here equals `volume` there.

BSE publishes `bseindia.com/BSEDATA/gross/YYYY/SCBSEALLDDMM.zip`, around 18:15
to 20:30 IST, and answers 404 before then and on holidays. It holds one
pipe-delimited text file:

    DATE|SCRIP CODE|DELIVERY QTY|DELIVERY VAL|DAY'S VOLUME|DAY'S TURNOVER|DELV. PER.

The file names no symbol. SCRIP CODE is the `FinInstrmId` of the same day's
BSE bhavcopy, which gives the symbol and series that `bse/` carries, so BSE
rows join to `bse/` on (date, symbol, series) the same way.
"""

from __future__ import annotations

import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import polars as pl
import requests

from pipeline.fetch import _BSE_HEADERS, _NSE_HEADERS, BhavcopyFetchError, _http_get_with_retry
from pipeline.transform import EQUITY_SERIES

NSE_MTO_URL = "https://archives.nseindia.com/archives/equities/mto/MTO_{ddmmyyyy}.DAT"
BSE_SCBSEALL_URL = "https://www.bseindia.com/BSEDATA/gross/{yyyy}/SCBSEALL{ddmm}.zip"

DELIVERY_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "series": pl.Utf8,
    "volume": pl.Int64,
    "delivery_qty": pl.Int64,
    "delivery_pct": pl.Float64,
}


class DeliveryParseError(ValueError):
    """A delivery file is for another day, or its records disagree with its header."""


def fetch_nse_delivery(
    d: date,
    dest_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Path:
    """Download NSE's MTO file for `d` into `dest_dir`. Returns its path.

    Idempotent: if the file already exists, returns its path without a request.
    Raises BhavcopyNotFoundError on 404, BhavcopyFetchError on other failures.
    Unlike fetch_nse there is no calendar check: the caller picks the days
    (the calendar misses special weekend sessions), and a holiday is a 404.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ddmmyyyy = d.strftime("%d%m%Y")
    path = dest_dir / f"MTO_{ddmmyyyy}.DAT"
    if path.exists():
        return path

    url = NSE_MTO_URL.format(ddmmyyyy=ddmmyyyy)
    path.write_bytes(_http_get_with_retry(url, _NSE_HEADERS, retries, backoff_seconds, timeout, session))
    return path


def parse_mto(path: Path, d: date) -> pl.DataFrame:
    """Read the MTO file for trade date `d` into DELIVERY_SCHEMA, equity series only.

    Keeps the series `nse/` carries (transform.EQUITY_SERIES); MTO has no
    BE/BZ rows, so in practice that is EQ. Raises DeliveryParseError when
    the `10,MTO` line names another date or its record count or delivery
    total disagrees with the records.
    """
    text = path.read_text(encoding="latin-1")
    if not text.endswith("\n"):
        raise DeliveryParseError(f"{path.name}: incomplete final line")
    lines = text.splitlines()
    header = next((ln.split(",") for ln in lines if ln.startswith("10,MTO,")), None)
    if header is None or len(header) != 5:
        raise DeliveryParseError(f"{path.name}: no 10,MTO header line")
    if datetime.strptime(header[2], "%d%m%Y").date() != d:
        raise DeliveryParseError(f"{path.name}: header date {header[2]} is not {d}")

    records = [ln.split(",") for ln in lines if ln.startswith("20,")]
    if any(len(r) != 7 for r in records):
        raise DeliveryParseError(f"{path.name}: a 20 record does not have 7 fields")
    total = sum(int(r[5]) for r in records)
    if int(header[4]) != len(records) or int(header[3]) != total:
        raise DeliveryParseError(
            f"{path.name}: header says {int(header[4])} records and {int(header[3])} delivered, "
            f"file has {len(records)} and {total}"
        )

    rows = [r for r in records if r[3] in EQUITY_SERIES["NSE"]]
    return pl.DataFrame(
        {
            "date": [d] * len(rows),
            "symbol": [r[2] for r in rows],
            "series": [r[3] for r in rows],
            "volume": [int(r[4]) for r in rows],
            "delivery_qty": [int(r[5]) for r in rows],
            "delivery_pct": [float(r[6]) for r in rows],
        },
        schema=DELIVERY_SCHEMA,
    ).sort(["date", "symbol"])


def fetch_bse_delivery(
    d: date,
    dest_dir: Path,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Path:
    """Download BSE's SCBSEALL zip for `d` and extract its text file into `dest_dir`.

    The member is `SCBSEALLDDMM.TXT` (`.txt` in some years) with no year in
    it, so it is saved as `SCBSEALL<DDMMYYYY>.TXT`. Idempotent and without a
    calendar check, like fetch_nse_delivery. Raises BhavcopyNotFoundError on
    404, BhavcopyFetchError on other failures or a payload that is not a zip
    holding one text file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"SCBSEALL{d:%d%m%Y}.TXT"
    if path.exists():
        return path

    url = BSE_SCBSEALL_URL.format(yyyy=d.year, ddmm=d.strftime("%d%m"))
    body = _http_get_with_retry(url, _BSE_HEADERS, retries, backoff_seconds, timeout, session)
    try:
        with zipfile.ZipFile(BytesIO(body)) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if len(members) != 1:
                raise BhavcopyFetchError(f"expected one text file in {url}, got {zf.namelist()}")
            text = zf.read(members[0])
    except zipfile.BadZipFile as e:
        raise BhavcopyFetchError(f"invalid zip payload from {url}: {e}") from e
    path.write_bytes(text)
    return path


def parse_scbseall(path: Path, d: date, bhavcopy: Path) -> pl.DataFrame:
    """Read the SCBSEALL file for trade date `d` into DELIVERY_SCHEMA, equity series only.

    `bhavcopy` is the same day's BSE bhavcopy CSV. A row's SCRIP CODE is its
    `FinInstrmId`, whose `TckrSymb` and `SctySrs` are the symbol and series in
    `bse/`; rows outside the series `bse/` carries (transform.EQUITY_SERIES)
    are dropped. Columns are read by position, as the regenerated 2024-04-25
    file spells the header `DAY`S VOLUME`. Raises DeliveryParseError when
    either file names another date.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    if not lines or not lines[0].startswith("DATE|SCRIP CODE|"):
        raise DeliveryParseError(f"{path.name}: no SCBSEALL header line")
    records = [ln.split("|") for ln in lines[1:] if ln.strip()]
    if any(len(r) != 7 for r in records):
        raise DeliveryParseError(f"{path.name}: a record does not have 7 fields")
    others = {r[0] for r in records} - {d.strftime("%d%m%Y")}
    if others:
        raise DeliveryParseError(f"{path.name}: records dated {sorted(others)}, not {d}")

    bhav = pl.read_csv(
        bhavcopy, columns=["TradDt", "FinInstrmId", "TckrSymb", "SctySrs"], infer_schema_length=0
    )
    others = set(bhav["TradDt"]) - {d.isoformat()}
    if others:
        raise DeliveryParseError(f"{bhavcopy.name}: TradDt {sorted(others)}, not {d}")
    names = bhav.select(
        pl.col("FinInstrmId").cast(pl.Int64, strict=False).alias("scrip"),
        pl.col("TckrSymb").str.strip_chars().alias("symbol"),
        pl.col("SctySrs").str.strip_chars().alias("series"),
    ).filter(pl.col("series").is_in(EQUITY_SERIES["BSE"]))

    rows = pl.DataFrame(
        {
            "scrip": [int(r[1]) for r in records],
            "volume": [int(r[4]) for r in records],
            "delivery_qty": [int(r[2]) for r in records],
            "delivery_pct": [float(r[6]) for r in records],
        },
        schema={
            "scrip": pl.Int64,
            "volume": pl.Int64,
            "delivery_qty": pl.Int64,
            "delivery_pct": pl.Float64,
        },
    )
    return (
        rows.join(names, on="scrip")
        .with_columns(pl.lit(d).alias("date"))
        .select(list(DELIVERY_SCHEMA))
        .sort(["date", "symbol"])
    )
