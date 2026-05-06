# tej-bazaar

> Free, open EOD data for Indian stock markets — NSE & BSE. Built from official exchange Bhavcopy. Published as partitioned Parquet on HuggingFace.

Part of the [TejHQ](https://github.com/tejhq) ecosystem.

> **Status:** NSE + BSE pipelines live, publishing to [`tejhq/indian-markets`](https://huggingface.co/datasets/tejhq/indian-markets). GitHub Actions cron next. See [ROADMAP.md](./ROADMAP.md).

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

### Output schema (per row)

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

### Via REST API (Phase 6 — separate `tej-api` repo, not live)

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
```

The pipeline skips market holidays automatically using `exchange_calendars` (NSE/BSE share trading days).

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full plan.

- [x] **Phase 1** — NSE pipeline (fetch, parse, transform, parquet write, CLI)
- [x] **Phase 2a** — BSE pipeline (same SEBI CMTS schema; series A/B/T)
- [x] **Phase 2b** — HuggingFace publish (`tej-bazaar publish`)
- [ ] **Phase 2c** — GitHub Actions cron (6:30 PM IST weekdays, holiday-aware)
- [ ] **Phase 3** — S3/R2 mirror, retry/alerting, source-diff checks
- [ ] **Phase 3.5** — Legacy historical data (pre-2024 NSE/BSE formats)
- [ ] **Phase 4** — Corporate actions, adjusted close, symbol-change history
- [ ] **Phase 5** — Derived metrics (returns, 52w hi/lo, avg vol)
- [ ] **Phase 6** — REST API handoff to `tej-api`, Python + JS SDKs

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
