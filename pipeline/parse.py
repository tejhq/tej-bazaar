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

# CMTS column → normalized name. Same mapping for NSE and BSE bhavcopies
# from 2024-07-08 onwards. Legacy NSE schema is handled separately below.
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

# Pre-2024-07-08 NSE legacy bhavcopy columns. ISIN + TOTALTRADES are absent
# for very early years (~2010), so they get treated as optional.
_LEGACY_NSE_COLUMN_MAP: dict[str, str] = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TIMESTAMP": "date",
    "TOTALTRADES": "trades",
    "ISIN": "isin",
}
_LEGACY_NSE_OPTIONAL = {"TOTALTRADES", "ISIN"}

_NUMERIC_FLOAT = ["open", "high", "low", "close", "last", "prev_close", "turnover"]
_NUMERIC_INT = ["volume", "trades"]


def parse_bhavcopy(csv_path: Path) -> pl.DataFrame:
    """Read NSE/BSE bhavcopy CSV and return a normalized Polars DataFrame.

    Auto-detects the modern SEBI CMTS schema (NSE post 2024-07-08, all BSE)
    vs. the legacy NSE schema (SYMBOL, SERIES, ..., TIMESTAMP). All rows
    returned (series filtering belongs in transform.py).
    """
    df = pl.read_csv(
        csv_path,
        try_parse_dates=False,
        infer_schema_length=0,  # read everything as string, cast deliberately
        null_values=["", " "],
    )

    if "TradDt" in df.columns:
        return _parse_modern(df)
    if "SYMBOL" in df.columns and "TIMESTAMP" in df.columns:
        return _parse_legacy_nse(df)
    raise ValueError(f"bhavcopy schema not recognised, got columns {df.columns}")


def _parse_modern(df: pl.DataFrame) -> pl.DataFrame:
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


def _parse_legacy_nse(df: pl.DataFrame) -> pl.DataFrame:
    # Legacy NSE bhavcopies have a trailing blank column (header ends with `,`).
    # Drop unnamed cols so the column map is the only source of truth.
    df = df.drop([c for c in df.columns if not c or c.startswith("_duplicated_")])

    required = [c for c in _LEGACY_NSE_COLUMN_MAP if c not in _LEGACY_NSE_OPTIONAL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"legacy NSE bhavcopy missing columns: {missing}")

    keep = [c for c in _LEGACY_NSE_COLUMN_MAP if c in df.columns]
    df = df.select(keep).rename({c: _LEGACY_NSE_COLUMN_MAP[c] for c in keep})

    # Fill optional columns with nulls so downstream schema is stable.
    for col in ("isin", "trades"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    # Legacy bhavcopy carries no instrument name; emit empty string.
    df = df.with_columns(pl.lit("").alias("name"))

    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%d-%b-%Y", strict=True),
        pl.col("symbol").str.strip_chars(),
        pl.col("series").str.strip_chars(),
        pl.col("isin").cast(pl.Utf8, strict=False).str.strip_chars(),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in _NUMERIC_FLOAT],
        *[pl.col(c).cast(pl.Int64, strict=False) for c in _NUMERIC_INT],
    )

    # Reorder to match modern output for a stable schema.
    return df.select(
        "date", "symbol", "series", "isin", "name",
        "open", "high", "low", "close", "last", "prev_close",
        "volume", "turnover", "trades",
    )


# Back-compat alias.
parse_nse = parse_bhavcopy
