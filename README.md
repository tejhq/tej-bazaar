# tej-bazaar

> Free, open EOD data for Indian stock markets — NSE & BSE. Built from official exchange Bhavcopy. Available as Parquet (and soon via HuggingFace + REST API).

Part of the [TejHQ](https://github.com/tejhq) ecosystem.

> **Status:** NSE + BSE pipelines live end-to-end (fetch → parse → transform → partitioned parquet). HuggingFace publish and scheduled cron come next. See [ROADMAP.md](./ROADMAP.md).

---

## What is this?

**tej-bazaar** ingests end-of-day OHLCV data for NSE and BSE listed instruments straight from the exchanges' official Bhavcopy, normalizes it, and publishes it as partitioned Parquet.

- **Source:** NSE/BSE Bhavcopy (official, free, no auth, redistributable)
- **Format:** Polars-friendly Parquet, Hive-partitioned by date
- **Latency:** End-of-day, ~6:30 PM IST after market close
- **License:** Code MIT. Data is exchange-published Bhavcopy — free to redistribute.

This repo is the **ingest pipeline**. A separate `tej-api` repo will serve the data over REST.

---

## Data Coverage

| Exchange | Status | Instruments | Updated |
|----------|--------|-------------|---------|
| NSE Equity (`EQ`, `BE`, `BZ`) | ✅ live (local) | ~2,300 / day | Manual today; cron next |
| BSE Equity (`A`, `B`, `T`) | ✅ live (local) | ~2,200 / day | Manual today; cron next |

### Output schema (per row)

| Field | Type | Description |
|-------|------|-------------|
| `date` | Date | Trading date |
| `symbol` | Utf8 | Ticker (e.g. `RELIANCE`) |
| `series` | Utf8 | NSE series code (`EQ`, `BE`, `BZ`, ...) |
| `isin` | Utf8 | International Securities ID |
| `name` | Utf8 | Full instrument name |
| `open` / `high` / `low` / `close` | Float64 | OHLC |
| `last` | Float64 | Last traded price |
| `prev_close` | Float64 | Previous close |
| `volume` | Int64 | Total traded volume (shares) |
| `turnover` | Float64 | Total traded value (rupees) |
| `trades` | Int64 | Number of trades executed |

---

## Access the Data

### Local Parquet (today)

After running the pipeline (see below), files land at:

```
data/out/nse/year=2025/month=04/date=2025-04-30.parquet
```

Read with Polars or DuckDB:

```python
import polars as pl
df = pl.read_parquet("data/out/nse/year=2025/month=04/date=2025-04-30.parquet")
```

```sql
-- DuckDB query across the whole partition tree:
SELECT * FROM read_parquet('data/out/nse/**/*.parquet', hive_partitioning=1)
WHERE symbol = 'RELIANCE';
```

### HuggingFace (Phase 2 — not live yet)

```python
# Once published:
from datasets import load_dataset
df = load_dataset("tejhq/indian-markets", split="nse")
```

### TejHQ REST API (Phase 6 — separate repo)

```bash
curl https://api.tejhq.dev/v1/ohlcv?symbol=RELIANCE&from=2025-01-01&to=2025-04-30
```

Join the waitlist at [tejhq.dev](https://tejhq.dev).

---

## Pipeline

```
NSE/BSE Bhavcopy (official EOD)
  → fetch (HTTP)
  → parse (CSV → Polars)
  → transform (filter, dedupe, validate)
  → write (partitioned Parquet)
  → [Phase 2] HuggingFace push
```

Bhavcopy is the exchanges' official end-of-day dump — free, no auth, redistributable. The pipeline skips market holidays automatically using `exchange_calendars` (NSE/BSE share trading days).

---

## Repo Structure

```
tej-bazaar/
├── pipeline/
│   ├── __init__.py
│   ├── fetch.py        # NSE bhavcopy download (zip → CSV)
│   ├── parse.py        # CSV → normalized Polars DataFrame
│   ├── transform.py    # filter EQ series, dedupe, validate
│   ├── push.py         # write partitioned parquet
│   ├── holidays.py     # NSE/BSE trading calendar
│   └── cli.py          # typer CLI: fetch / backfill / info / version
├── tests/
│   ├── fixtures/       # tiny golden bhavcopy sample
│   ├── test_holidays.py
│   ├── test_fetch.py
│   ├── test_parse.py
│   ├── test_transform.py
│   ├── test_push.py
│   └── test_cli.py
├── pyproject.toml
├── ROADMAP.md
├── LICENSE
└── README.md
```

---

## Running Locally

Requires Python 3.11+.

```bash
git clone https://github.com/tejhq/tej-bazaar
cd tej-bazaar

# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run pipeline for one date (NSE by default)
tej-bazaar fetch 2025-04-30
tej-bazaar fetch 2025-04-30 --exchange BSE
tej-bazaar fetch 2025-04-30 --exchange both

# Backfill a range (skips holidays + weekends)
tej-bazaar backfill --from 2025-04-01 --to 2025-04-30 --exchange both

# What's on disk?
tej-bazaar info

# Run the test suite
pytest
```

Output lands under `data/out/` by default. Override with `--out-dir`.

`HF_TOKEN` env var is only needed once Phase 2 (HuggingFace push) lands.

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full plan.

- [x] **Phase 1** — NSE pipeline (fetch, parse, transform, parquet write, CLI)
- [x] **Phase 2a** — BSE pipeline (same SEBI CMTS schema; series A/B/T)
- [ ] **Phase 2b** — HuggingFace publish, GitHub Actions cron, backfill scripts
- [ ] **Phase 3** — S3/R2 mirror, retry/alerting, source-diff checks
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
- 🐦 [@tejhq](https://x.com/tejhq)
- 💬 Discussions tab for questions

> *Tej — sharp, fast, bright. Just like the data should be.*
