# tej-bazaar

> Free, open EOD data for Indian stock markets — NSE & BSE. Built from official exchange Bhavcopy. Published as partitioned Parquet on HuggingFace.

Part of the [TejHQ](https://github.com/tejhq) ecosystem.

> **Status:** NSE + BSE pipelines live. Daily cron publishes to [`tejhq/indian-markets`](https://huggingface.co/datasets/tejhq/indian-markets) at 19:00 IST on trading days. Corporate actions, back-adjusted prices, symbol-history, Yahoo reconciliation, and derived metrics (returns + rolling 52w / 20d / 60d windows) all shipped. See [ROADMAP.md](./ROADMAP.md).

---

## What is this?

**tej-bazaar** ingests end-of-day OHLCV data for NSE and BSE listed instruments straight from the exchanges' official Bhavcopy, normalizes it, and publishes it as partitioned Parquet on HuggingFace.

- **Source:** NSE/BSE Bhavcopy (official, free, no auth, redistributable)
- **Format:** Polars-friendly Parquet, Hive-partitioned by date
- **Latency:** End-of-day, ~6:30 PM IST after market close
- **License:** Code MIT. Data is exchange-published Bhavcopy — free to redistribute.

This repo is the **ingest pipeline**. A separate `tej-api` repo will serve the data over REST.

---

## Coverage

| Exchange | Series | Instruments | Coverage start |
|----------|--------|-------------|----------------|
| NSE Equity | `EQ`, `BE`, `BZ` | ~2,300 / day | 2024-01-01 |
| BSE Equity | `A`, `B`, `T` | ~2,200 / day | 2024-01-01 |

### Why 2024-01-01?

NSE and BSE both moved to the new **SEBI CMTS bhavcopy format** around late 2023 / early 2024. NSE's CMTS file (`BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip`) starts **2024-01-01** — December 2023 returns 404.

Pre-cutover bhavcopies use legacy formats with different filenames and column names. Parsing them needs a separate code path; tracked under [Phase 3.5 in ROADMAP](./ROADMAP.md#phase-35--legacy-historical-data). For now we publish a clean, uniform CMTS-era dataset.

### Output trees

Five parquet trees are produced and published, each under its own top-level
prefix on HuggingFace and `data/out/` locally.

#### 1. Bhavcopy (`nse/`, `bse/`)

Hive-partitioned by trading date: `<ex>/year=YYYY/month=MM/date=YYYY-MM-DD.parquet`.

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

One file per exchange per calendar year: `metrics/<ex>_<YYYY>.parquet`.

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

---

## Use the data

### From HuggingFace (recommended)

```python
import polars as pl
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    "tejhq/indian-markets",
    "nse/year=2025/month=04/date=2025-04-30.parquet",
    repo_type="dataset",
)
df = pl.read_parquet(p)
```

Or the whole partition tree with DuckDB:

```sql
SELECT *
FROM read_parquet('hf://datasets/tejhq/indian-markets/nse/**/*.parquet', hive_partitioning=1)
WHERE symbol = 'RELIANCE' AND date >= '2025-01-01';
```

### From local parquet (after running the pipeline yourself)

```python
import polars as pl
df = pl.read_parquet("data/out/nse/year=2025/month=04/date=2025-04-30.parquet")
```

### Via REST API (Phase 6, separate `tej-api` repo, not live)

```bash
curl https://api.tejhq.dev/v1/ohlcv?symbol=RELIANCE&from=2025-01-01&to=2025-04-30
```

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
| `tej-bazaar reconcile --from D --to D --top 50` | Compare local adjusted closes against Yahoo Finance |
| `tej-bazaar publish --dry-run` | List local parquet files; no upload |
| `tej-bazaar publish --repo tejhq/indian-markets` | Push to HuggingFace (needs `HF_TOKEN`) |
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
  → write (partitioned parquet, zstd, Hive layout)
  → publish (HuggingFace upload_folder, content-hash dedup)

NSE/BSE corporate actions (REST API, BSE scrip-master ISIN lookup)
  → fetch (per-year annual file, idempotent)
  → parse (classify split / bonus / dividend / rights / merger)
  → resolve ISIN via symbol-history (handles post-merger ISIN drift)
  → factors (split: fv_to/fv_from, bonus: d/(n+d), dividend: 1 - cash/prev_close)
  → back-adjust (per-ISIN reverse cumprod, polars partition + numpy searchsorted)
```

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

- [x] **Phase 1** — NSE pipeline (fetch, parse, transform, parquet write, CLI)
- [x] **Phase 2a** — BSE pipeline (same SEBI CMTS schema; series A/B/T)
- [x] **Phase 2b** — HuggingFace publish (`tej-bazaar publish`)
- [x] **Phase 2c** — GitHub Actions cron (19:00 IST weekdays, holiday-aware)
- [ ] **Phase 3** — S3/R2 mirror, failure alerts (Slack/Discord webhook)
- [ ] **Phase 3.5** — Legacy historical data (pre-2024 NSE/BSE formats)
- [x] **Phase 4** - Corporate actions, adjusted close, symbol-change history, Yahoo reconciliation
- [x] **Phase 5** - Derived metrics (returns, 52w hi/lo, avg vol / turnover)
- [ ] **Phase 6** - REST API handoff to `tej-api`, Python + JS SDKs

---

## Contributing

PRs welcome. If you find data quality issues, missing stocks, or holiday gaps — open an issue.

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

> *Tej — sharp, fast, bright. Just like the data should be.*
