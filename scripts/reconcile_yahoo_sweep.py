"""One-off reconciliation sweep against Yahoo Finance via yfinance.

Why this lives in `scripts/` and not `pipeline/`:
- yfinance pulls pandas as a transitive dep. The pipeline package is
  intentionally polars-only, so we keep yfinance out of project deps.
- The dataset's quality claim (matches Yahoo within 0.5%) only needs to be
  verified periodically, not as part of the daily cron. A standalone script
  is the right blast-radius.

Usage:
    python scripts/reconcile_yahoo_sweep.py \\
        --top 50 --from 2024-01-01 --to 2026-05-06 --exchange NSE
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.reconcile.compare import reconcile_symbol, summarize  # noqa: E402

EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_ours(adjusted_dir: Path, exchange: str, start: date, end: date) -> pl.DataFrame:
    paths = sorted(adjusted_dir.glob(f"{exchange.lower()}_*.parquet"))
    if not paths:
        raise SystemExit(f"no adjusted parquet under {adjusted_dir}")
    df = pl.concat([pl.read_parquet(p) for p in paths])
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def fetch_yahoo_via_yfinance(symbol: str, suffix: str, start: date, end: date) -> pl.DataFrame:
    """Pull adjclose via yfinance. Returns DataFrame(date, yahoo_adjclose)."""
    ticker = yf.Ticker(f"{symbol}{suffix}")
    # yfinance `end` is exclusive, so add a day so it includes the requested end.
    end_excl = end.toordinal() + 1
    end_dt = date.fromordinal(end_excl)
    hist = ticker.history(
        start=start.isoformat(),
        end=end_dt.isoformat(),
        interval="1d",
        auto_adjust=False,  # keep raw Close + Adj Close separate
        actions=False,
    )
    if hist.empty:
        return pl.DataFrame(schema={"date": pl.Date, "yahoo_adjclose": pl.Float64})
    # hist index is a DatetimeIndex with tz; reset to date col.
    rows = [
        (idx.date(), float(row["Adj Close"]))
        for idx, row in hist.iterrows()
        if row["Adj Close"] == row["Adj Close"]  # NaN check
    ]
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "yahoo_adjclose": pl.Float64},
        orient="row",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"])
    ap.add_argument("--adjusted-dir", default="data/out/prices_adjusted")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--out", default="reconcile_report.md")
    args = ap.parse_args()

    start = parse_date(args.from_date)
    end = parse_date(args.to_date)
    suffix = EXCHANGE_SUFFIX[args.exchange]

    ours_full = load_ours(Path(args.adjusted_dir), args.exchange, start, end)

    ranked = (
        ours_full.group_by("symbol")
        .agg(mean_turnover=pl.col("turnover").mean())
        .sort("mean_turnover", descending=True)
        .head(args.top)
    )
    sym_list = ranked["symbol"].to_list()
    print(f"selected top {len(sym_list)} symbols by mean turnover", flush=True)

    stats_list = []
    failures = []
    for i, sym in enumerate(sym_list, start=1):
        ours = ours_full.filter(pl.col("symbol") == sym).select(["date", "adj_close"])
        if ours.height == 0:
            failures.append((sym, "no rows in ours"))
            continue
        try:
            ref = fetch_yahoo_via_yfinance(sym, suffix, start, end)
        except Exception as e:
            failures.append((sym, f"yahoo error: {e}"))
            continue
        if ref.height == 0:
            failures.append((sym, "yahoo returned empty"))
            continue
        s = reconcile_symbol(sym, ours, ref, tolerance_pct=args.tolerance)
        stats_list.append(s)
        marker = "OK" if s.pct_within_tol >= 99.0 else ("WARN" if s.pct_within_tol >= 95.0 else "FAIL")
        print(
            f"  [{i:3}/{len(sym_list):3}] {sym:14} "
            f"rows={s.rows_compared:4} within={s.pct_within_tol:6.2f}% "
            f"max={s.max_abs_diff_pct:6.3f}% mean={s.mean_abs_diff_pct:6.3f}%  {marker}",
            flush=True,
        )
        if i < len(sym_list):
            time.sleep(args.sleep)

    overall = summarize(stats_list)

    lines = [
        f"# tej-bazaar vs Yahoo Finance — reconciliation report",
        "",
        f"- Range: **{start} → {end}**",
        f"- Exchange: **{args.exchange}**",
        f"- Symbols requested: {len(sym_list)} (top by mean daily turnover)",
        f"- Symbols matched: {len(stats_list)}",
        f"- Total row-comparisons: {overall.overall_rows}",
        f"- Tolerance: ±{args.tolerance}%",
        f"- **Overall within tolerance: {overall.overall_pct_within_tol:.2f}%**",
        "",
        "## Per-symbol",
        "",
        "| Symbol | Rows | Within tol | Max diff % | Mean diff % |",
        "|--------|-----:|-----------:|-----------:|------------:|",
    ]
    for s in sorted(stats_list, key=lambda x: -x.rows_compared):
        lines.append(
            f"| {s.symbol} | {s.rows_compared} | {s.pct_within_tol:.2f}% | "
            f"{s.max_abs_diff_pct:.3f} | {s.mean_abs_diff_pct:.3f} |"
        )
    if failures:
        lines += ["", "## Failures", ""]
        for sym, why in failures:
            lines.append(f"- `{sym}`: {why}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote report to {args.out}")
    print(f"OVERALL: {overall.overall_pct_within_tol:.2f}% within ±{args.tolerance}% "
          f"({overall.overall_rows} rows, {len(stats_list)} symbols)")


if __name__ == "__main__":
    main()
