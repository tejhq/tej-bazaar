# tej-bazaar

> Free, open EOD data for Indian stock markets, NSE and BSE. Built from official exchange Bhavcopy. Published as partitioned Parquet on HuggingFace.

Part of the [TejHQ](https://github.com/tejhq) ecosystem.

> **Status:** NSE 2010 to today and BSE 2024-07-08 to today, live. A two-job GitHub Actions cron runs at 20:00 IST on trading days: `prices` folds the day into the year rollup and publishes to R2 and [`tejhq/indian-markets`](https://huggingface.co/datasets/tejhq/indian-markets); `derived` rebuilds corporate actions, symbol history, adjusted prices, metrics and the liquidity universe from full history. Both jobs also pre-render the JSON that `api.tejhq.dev` serves at the edge. See [ROADMAP.md](./ROADMAP.md).

---

## What is this?

**tej-bazaar** ingests end-of-day OHLCV data for NSE and BSE listed instruments straight from the exchanges' official Bhavcopy, normalizes it, and publishes it as partitioned Parquet on HuggingFace.

- **Source:** NSE/BSE Bhavcopy (official, free, no auth, redistributable)
- **Format:** Polars-friendly Parquet, Hive-partitioned by date
- **Latency:** End-of-day, published by 20:30 IST on trading days
- **License:** Code MIT. Data is exchange-published Bhavcopy, free to redistribute.

This repo is the **ingest pipeline**. [`tej-api`](../tej-api) serves the same parquet over REST at `api.tejhq.dev`.

---

## Coverage

| Exchange | Series | Instruments | Coverage |
|----------|--------|-------------|----------|
| NSE Equity | `EQ`, `BE`, `BZ` | ~2,300 / day | 2010-01-04 to today, ~4,070 trading days |
| BSE Equity | `A`, `B`, `T` | ~2,200 / day | 2024-07-08 to today |

### Coverage notes

NSE and BSE moved to the **SEBI CMTS bhavcopy format** in 2024. The pipeline detects the format per file and parses both the modern CMTS layout and the legacy NSE `SYMBOL/SERIES/TIMESTAMP` layout, which is how NSE coverage reaches back to 2010. Pre-2012 NSE rows carry no ISIN and no per-row trade count because the source never had them; those fields are null in the raw parquet rather than invented. Derived datasets (adjusted prices, symbol history, metrics) backfill the ISIN for symbols that traded on both the last legacy day and the first CMTS day, so corporate actions and rolling windows apply correctly back to 2010. Symbols that did not survive the cutover stay ISIN-less and are windowed by symbol.

Legacy BSE files identify instruments by numeric `SC_CODE` with no clean bridge to modern tickers, so BSE starts at the CMTS cutover.

### Output trees

Six parquet trees are produced and published, each under its own top-level
prefix on R2 and HuggingFace, plus a JSON tree for the API edge.

#### 1. Bhavcopy (`nse/`, `bse/`)

One rollup per exchange per year, `<ex>/year=YYYY/<ex>_YYYY.parquet`, sorted by `(symbol, date)` so single-symbol queries prune to a few row groups. The daily cron fetches one day, folds it into the current year's rollup, and deletes the daily file.

| Field | Type | Description |
|-------|------|-------------|
| `date` | Date | Trading date |
| `symbol` | Utf8 | Ticker (e.g. `RELIANCE`) |
| `series` | Utf8 | Exchange series code (NSE: `EQ`/`BE`/`BZ`, BSE: `A`/`B`/`T`) |
| `isin` | Utf8 | International Securities ID |
| `name` | Utf8 | Full instrument name |
| `open` / `high` / `low` / `close` | Float64 | OHLC |
| `last` | Float64 | Last traded price |
| `prev_close` | Float64 | Previous close |
| `volume` | Int64 | Total traded volume (shares) |
| `turnover` | Float64 | Total traded value (rupees) |
| `trades` | Int64 | Number of trades executed |

#### 2. Corporate actions (`actions/`)

One file per exchange per calendar year: `actions/<ex>_<YYYY>.parquet`.

| Field | Type | Description |
|-------|------|-------------|
| `exchange` | Utf8 | `NSE` or `BSE` |
| `symbol` | Utf8 | Ticker on `ex_date` |
| `isin` | Utf8 | ISIN as reported by the source (may be a stale post-merger ISIN; see resolver in `pipeline/actions/back_adjust.py`) |
| `company` | Utf8 | Issuer name |
| `ex_date` | Date | First trading day on which the price is ex the action |
| `record_date` | Date | Record date if reported, else null |
| `type` | Utf8 | One of `dividend`, `split`, `bonus`, `rights`, `buyback`, `demerger`, `merger`, `agm`, `other` |
| `ratio_num` / `ratio_den` | Int64 | Bonus / rights ratio numerator and denominator |
| `cash_amount` | Float64 | Per-share cash for dividends |
| `face_value_from` / `face_value_to` | Float64 | Face values for splits |
| `raw_subject` | Utf8 | Verbatim source description, kept for audit |

#### 3. Back-adjusted prices (`prices_adjusted/`)

One file per exchange per calendar year: `prices_adjusted/<ex>_<YYYY>.parquet`.
Same columns as the bhavcopy, plus:

| Field | Type | Description |
|-------|------|-------------|
| `adj_factor_cumulative` | Float64 | Product of factors of all corporate actions with ex_date > `date` for this ISIN. `1.0` when no later actions. |
| `adj_close` | Float64 | `close * adj_factor_cumulative`; continuous through splits, bonuses, dividends. |

#### 4. Symbol history (`symbol_history/`)

One file per exchange: `symbol_history/<ex>.parquet`. Each row is one
contiguous interval during which an ISIN traded under a single symbol.

| Field | Type | Description |
|-------|------|-------------|
| `exchange` | Utf8 | `NSE` or `BSE` |
| `isin` | Utf8 | Stable instrument ID |
| `symbol` | Utf8 | Symbol active during the interval |
| `valid_from` | Date | First trading day of the interval |
| `valid_to` | Date | Last trading day of the interval |
| `trading_days` | Int64 | Number of trading days in the interval |

#### 5. Derived metrics (`metrics/`)

One file per exchange per calendar year: `metrics/<ex>_<YYYY>.parquet`, plus
`metrics/<ex>_latest.parquet` holding only the newest trading day so the API
screener can answer "today" with one small read. Rewritten every run.

| Field | Type | Description |
|-------|------|-------------|
| `date`, `symbol`, `isin`, `adj_close` | (as above) | Carried for joining |
| `ret_1d` / `ret_5d` / `ret_21d` / `ret_63d` / `ret_126d` / `ret_252d` | Float64 | Simple price returns over N trading days |
| `ret_ytd` | Float64 | Return since first trading day of `date`'s calendar year |
| `high_52w` / `low_52w` | Float64 | Max / min `adj_close` over last 252 trading days |
| `pct_off_52w_high` / `pct_off_52w_low` | Float64 | `adj_close / high_52w - 1` and `adj_close / low_52w - 1` |
| `avg_vol_20d` / `avg_vol_60d` | Float64 | Rolling mean of raw `volume` |
| `avg_turnover_20d` | Float64 | Rolling mean of raw `turnover` |

Rolling windows require a full window of prior history; bootstrap rows are
null at that horizon rather than computed off a partial window.

#### 6. Liquidity universe (`universe/`)

One file per exchange: `universe/<ex>_liquid.parquet`. Every month, on the
first trading day, instruments are ranked by trailing 63-trading-day mean
turnover and the top 500 are kept. Membership uses only data available on
the rebalance date, so a name that later delisted still appears in the
months it qualified for. This is a survivorship-bias-free backtest universe.

| Field | Type | Description |
|-------|------|-------------|
| `exchange` | Utf8 | `NSE` or `BSE` |
| `rebalance_date` | Date | First trading day of the month |
| `valid_to` | Date | Day before the next rebalance |
| `rank` | Int64 | 1 = highest trailing turnover |
| `symbol`, `isin`, `name` | Utf8 | As in force on `rebalance_date` |
| `avg_turnover_63d` | Float64 | Trailing mean turnover, rupees |

`liquid100`, `liquid250` and `liquid500` are `rank <= N` filters over the same rows.

#### 7. API edge JSON (`api/v1/`, R2 only)

`tej-bazaar export-json` pre-renders the tej-api free tier as static JSON:
`api/v1/ohlcv/<ex>/<SYMBOL>.json` (full history per symbol),
`api/v1/snapshot/<ex>/<DATE>.json` and `api/v1/actions/<SYMBOL>.json`.
The Cloudflare Worker in front of `api.tejhq.dev` serves and slices these
directly, so free-tier requests never reach DuckDB. Not mirrored to HuggingFace.

---

## Use the data

### From HuggingFace (recommended)

The dataset card at [huggingface.co/datasets/tejhq/indian-markets](https://huggingface.co/datasets/tejhq/indian-markets) is `hf/README.md` in this repo, uploaded by the daily cron with `publish --card`. Its `configs` block gives the viewer one tab per tree.

```python
import polars as pl
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    "tejhq/indian-markets",
    "nse/year=2025/nse_2025.parquet",
    repo_type="dataset",
)
df = pl.read_parquet(p)
```

Or the whole tree with DuckDB:

```sql
SELECT *
FROM read_parquet('hf://datasets/tejhq/indian-markets/nse/**/*.parquet', union_by_name=true)
WHERE symbol = 'RELIANCE' AND date >= '2025-01-01';
```

### From R2 (bulk, edge cached)

```bash
curl -O https://data.tejhq.dev/nse/year=2025/nse_2025.parquet
```

### From local parquet (after running the pipeline yourself)

```python
import polars as pl
df = pl.read_parquet("data/out/nse/year=2025/nse_2025.parquet")
```

### Via REST API

```bash
curl "https://api.tejhq.dev/v1/ohlcv/nse/RELIANCE?from=2025-01-01&to=2025-04-30"
```

Docs at [tejhq.dev/docs](https://tejhq.dev/docs). Python SDK: `pip install tejhq`.

---

## Run it yourself

Requires Python 3.11+.

```bash
git clone https://github.com/tejhq/tej-bazaar
cd tej-bazaar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### CLI commands

| Command | What it does |
|---------|--------------|
| `tej-bazaar fetch DATE` | Run full pipeline for one trading date |
| `tej-bazaar fetch DATE --exchange BSE` | BSE instead of default NSE |
| `tej-bazaar fetch DATE --exchange both` | Both exchanges |
| `tej-bazaar backfill --from D --to D` | Range; skips weekends + NSE/BSE holidays automatically |
| `tej-bazaar backfill --from D --to D --exchange both` | Range, both exchanges |
| `tej-bazaar actions fetch --year 2024 --exchange both` | Pull NSE+BSE corporate actions for a calendar year (annual rolling file) |
| `tej-bazaar actions adjust --year 2024 --exchange NSE` | Compute back-adjusted prices from bhavcopy + actions (single year) |
| `tej-bazaar actions adjust --all-years --exchange both` | Re-adjust every year on disk (cron default; needed when future actions land) |
| `tej-bazaar symbol-history build --exchange both` | Per-ISIN symbol-history intervals across the full price series |
| `tej-bazaar metrics build --all-years --exchange both` | Returns (1d/5d/21d/63d/126d/252d/YTD) + rolling 52w hi/lo + 20d/60d avg vol + 20d avg turnover |
| `tej-bazaar universe build --exchange both` | Monthly top-500 by trailing 63-day turnover, point-in-time, one file per exchange |
| `tej-bazaar export-json --only ohlcv,snapshot` | Pre-render tej-api free-tier JSON under `data/api/api/v1/`; `--snapshot-years all` for a one-time seed |
| `tej-bazaar reconcile --from D --to D --top 50` | Compare local adjusted closes against Yahoo Finance |
| `tej-bazaar compact --year YYYY --exchange both --from-r2 --refresh` | Fold daily files into the year rollup on R2, dedupe by `(symbol, date)` |
| `tej-bazaar publish --dry-run` | List local parquet files; no upload |
| `tej-bazaar publish --repo tejhq/indian-markets` | Push to HuggingFace (needs `HF_TOKEN`) |
| `tej-bazaar publish --card hf/README.md` | Push parquet and upload the dataset card as the repo README |
| `tej-bazaar publish-r2 --data-dir data/out` | Push parquet and JSON to R2, ETag-deduped (needs `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) |
| `tej-bazaar pull-r2 --prefix nse/ --prefix actions/` | Seed a local `data/out` from R2, local files win |
| `tej-bazaar r2-prune --prefix nse/year=2026/month=` | Delete a key prefix on R2, refuses bucket-root prefixes |
| `tej-bazaar info` | Inventory of local parquet on disk |
| `tej-bazaar version` | Print version |

Common flags: `--out-dir PATH` (parquet output, default `data/out/`), `--raw-dir PATH` (downloaded CSV cache, default `data/raw/`), `--skip-existing/--overwrite` (backfill resume behaviour, default skip).

### Quick smoke test

```bash
tej-bazaar fetch 2025-04-30 --exchange both
tej-bazaar info
pytest
```

### Pipeline

```
NSE/BSE Bhavcopy (official EOD, SEBI CMTS format)
  → fetch (HTTP, browser headers, idempotent)
  → parse (CSV → Polars, normalized 14-column schema)
  → transform (filter equity series, dedupe, validate prices)
  → write (parquet, zstd)
  → compact (fold into one rollup per exchange per year, sorted by symbol, date)
  → publish (R2 with ETag dedup, HuggingFace with content-hash dedup)
  → export-json (per-symbol OHLCV and per-date snapshots for the API edge)

NSE/BSE corporate actions (REST API, BSE scrip-master ISIN lookup)
  → fetch (per-year annual file, idempotent)
  → parse (classify split / bonus / dividend / rights / merger)
  → resolve ISIN via symbol-history (handles post-merger ISIN drift)
  → factors (split: fv_to/fv_from, bonus: d/(n+d), dividend: 1 - cash/prev_close)
  → back-adjust (per-ISIN reverse cumprod, polars partition + numpy searchsorted)
  → metrics (returns, 52w hi/lo, avg volume and turnover)
  → universe (monthly top-500 by trailing turnover, point-in-time)
```

### Daily cron

`.github/workflows/daily.yml`, 20:00 IST on weekdays, two jobs:

- `prices`: fetch today's bhavcopy, publish, refresh the year rollup, prune dailies, export and publish edge JSON, publish to HuggingFace. Skips cleanly on holidays.
- `derived`: pull full history from R2, refresh corporate actions, rebuild symbol history, adjusted prices, metrics and universe into `data/derived/`, publish that directory only. A failure here publishes nothing stale and fails the run, so GitHub emails on it. Only the NSE corporate-actions website fetch is allowed to fail, falling back to the history already on R2.

Manual runs: `gh workflow run daily-bhavcopy -f date=YYYY-MM-DD`, or `-f from=D -f to=D` to backfill a range. Set the `ALERT_WEBHOOK_URL` secret for Discord or Slack failure pings.

### Verification vs Yahoo Finance

Adjusted closes for the top 50 NSE names (by mean daily turnover) over
2024-01-01 → 2026-05-06 reconcile against Yahoo's `Adj Close` as follows:

- **89% of row-comparisons within ±1%** (~25,000 daily closes across 48 symbols)
- The residual gap is driven by methodology differences in dividend
  adjustment: NSE official uses `(prev_close - cash) / prev_close`;
  Yahoo's CRSP factor uses `1 - cash / close_on_ex_date`. For
  dividend-heavy names like INFY, TCS, HINDUNILVR, this compounds to a
  systematic ~1% offset that is not a bug in either source.
- Splits and bonus issues match Yahoo within ~1% on the day after the
  event (the difference is the dividend layer above, not the split math).
- Run `python scripts/reconcile_yahoo_sweep.py --top 50 --from D --to D --tolerance 1.0`
  to reproduce. The script lives outside `pipeline/` because `yfinance`
  pulls pandas as a transitive dep, which the pipeline package
  intentionally avoids; install it via the optional `reconcile` extra:
  `pip install -e ".[reconcile]"`.

The pipeline skips market holidays automatically using `exchange_calendars` (NSE/BSE share trading days).

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full plan.

- [x] **Phase 1**: NSE pipeline (fetch, parse, transform, parquet write, CLI)
- [x] **Phase 2**: BSE pipeline, HuggingFace publish, GitHub Actions cron
- [x] **Phase 3**: R2 mirror, split cron with failure alerts, edge JSON export
- [x] **Phase 3.5**: Legacy NSE format, coverage back to 2010
- [x] **Phase 3.6**: Per-year rollups sorted by `(symbol, date)`
- [x] **Phase 4**: Corporate actions, adjusted close, symbol history, Yahoo reconciliation
- [x] **Phase 5**: Derived metrics (returns, 52w hi/lo, avg volume and turnover)
- [x] **Phase 5.5**: Point-in-time liquidity universe
- [ ] **Phase 6**: TypeScript SDK (Python SDK shipped as `tejhq`)

---

## Contributing

PRs welcome. If you find data quality issues, missing stocks, or holiday gaps, open an issue.

---

## License

Code: **MIT** (see [LICENSE](./LICENSE)).

Data: NSE/BSE Bhavcopy is published openly by the exchanges; redistribution as cleaned Parquet is permitted. Always verify exchange terms before commercial use.

---

## Part of TejHQ

TejHQ is building developer-first financial data infrastructure for India.

- 🌐 [tejhq.dev](https://tejhq.dev)
- 🤗 [HuggingFace dataset](https://huggingface.co/datasets/tejhq/indian-markets)
- 💬 Discussions tab for questions

> *Tej: sharp, fast, bright. Just like the data should be.*
