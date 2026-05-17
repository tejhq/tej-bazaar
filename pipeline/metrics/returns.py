"""Per-symbol return series at standard horizons.

Computes simple price returns over fixed trading-day windows plus a
calendar-year-to-date return. All horizons use back-adjusted close
(`adj_close`), so splits and bonuses do not show up as -50% / -90%
shocks. The 1d return between two consecutive trading dates already
"strips" dividends through the adjusted-close layer above.

The window math is purely positional over each ISIN's sorted trading
days: a 5-day return is `adj_close[t] / adj_close[t-5] - 1` regardless
of whether those five days span a weekend or a holiday. That matches
how research desks quote weekly/monthly returns from daily closes.

Output schema:

    date            Date    Trading date
    symbol          Utf8    Symbol on `date`
    isin            Utf8    Stable instrument ID (partition key)
    adj_close       Float64 Back-adjusted close (carried for joins)
    ret_1d          Float64 1 trading-day return
    ret_5d          Float64 5 trading-day return (~1 week)
    ret_21d         Float64 21 trading-day return (~1 month)
    ret_63d         Float64 63 trading-day return (~3 months)
    ret_126d        Float64 126 trading-day return (~6 months)
    ret_252d        Float64 252 trading-day return (~1 year)
    ret_ytd         Float64 Return since first trading day of `date`'s year

Bootstrap rows (not enough prior history) get null at that horizon.
"""

from __future__ import annotations

import polars as pl

_HORIZONS: list[tuple[str, int]] = [
    ("ret_1d", 1),
    ("ret_5d", 5),
    ("ret_21d", 21),
    ("ret_63d", 63),
    ("ret_126d", 126),
    ("ret_252d", 252),
]

RETURNS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "isin": pl.Utf8,
    "adj_close": pl.Float64,
    **{name: pl.Float64 for name, _ in _HORIZONS},
    "ret_ytd": pl.Float64,
}


def compute_returns(adjusted: pl.DataFrame) -> pl.DataFrame:
    """Compute fixed-horizon and YTD returns from back-adjusted prices.

    `adjusted` must have at minimum `(isin, date, symbol, adj_close)`.
    Other columns are dropped. Returns are computed per ISIN over the
    sorted trading-day axis; ISIN is the partition because splits with a
    face-value change can flip the symbol while the underlying continues
    via the new ISIN (handled by symbol-history elsewhere).

    Sorting is performed inside this function; do not assume any input
    ordering.
    """
    _require_columns(adjusted, ["isin", "date", "symbol", "adj_close"])

    if adjusted.height == 0:
        return pl.DataFrame(schema=RETURNS_SCHEMA)

    df = adjusted.select(["isin", "date", "symbol", "adj_close"]).sort(
        ["isin", "date"]
    )

    horizon_exprs = [
        (pl.col("adj_close") / pl.col("adj_close").shift(n).over("isin") - 1.0)
        .alias(name)
        for name, n in _HORIZONS
    ]

    # YTD return is anchored to the first trading day of the same
    # calendar year for the same ISIN. `first().over([isin, year])`
    # would pick the chronologically-first row only if the frame is
    # sorted, which we did above.
    ytd_expr = (
        pl.col("adj_close")
        / pl.col("adj_close").first().over(["isin", pl.col("date").dt.year()])
        - 1.0
    ).alias("ret_ytd")

    return df.with_columns(horizon_exprs + [ytd_expr]).select(
        list(RETURNS_SCHEMA.keys())
    )


def _require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
