"""Parse exchange bhavcopy CSV → Polars DataFrame with normalized schema.

NSE and BSE both publish under SEBI's CMTS spec, so a single parser handles both.
Raw columns use compact codes (OpnPric, HghPric, ...). This module maps them to a
clean schema. Filtering and validation live in transform.py.

Normalized schema:
    date          Date
    symbol        Utf8
    series        Utf8     (NSE: EQ/BE/BZ/GB/...  BSE: A/B/T/X/...)
    isin          Utf8
    name          Utf8
    open          Float64
    high          Float64
    low           Float64
    close         Float64
    last          Float64
    prev_close    Float64
    volume        Int64
    turnover      Float64  (rupees)
    trades        Int64
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# CMTS column → normalized name. Same mapping for NSE and BSE bhavcopies.
_COLUMN_MAP: dict[str, str] = {
    "TradDt": "date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "ISIN": "isin",
    "FinInstrmNm": "name",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}

_NUMERIC_FLOAT = ["open", "high", "low", "close", "last", "prev_close", "turnover"]
_NUMERIC_INT = ["volume", "trades"]


def parse_bhavcopy(csv_path: Path) -> pl.DataFrame:
    """Read NSE/BSE bhavcopy CSV and return a normalized Polars DataFrame.

    All rows are returned (series filtering belongs in transform.py).
    """
    df = pl.read_csv(
        csv_path,
        try_parse_dates=False,
        infer_schema_length=0,  # read everything as string, cast deliberately
        null_values=["", " "],
    )

    missing = [src for src in _COLUMN_MAP if src not in df.columns]
    if missing:
        raise ValueError(f"bhavcopy missing columns: {missing}")

    df = df.select(list(_COLUMN_MAP.keys())).rename(_COLUMN_MAP)

    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        pl.col("symbol").str.strip_chars(),
        pl.col("series").str.strip_chars(),
        pl.col("isin").str.strip_chars(),
        pl.col("name").str.strip_chars(),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in _NUMERIC_FLOAT],
        *[pl.col(c).cast(pl.Int64, strict=False) for c in _NUMERIC_INT],
    )

    return df


# Back-compat alias.
parse_nse = parse_bhavcopy
