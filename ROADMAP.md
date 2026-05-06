# tej-bazaar Roadmap

> **Mission**: Free, open, redistributable EOD market data for India. Audit-grade source. Zero auth to consume.

## Guiding principles

1. **Official source only** — Bhavcopy from NSE/BSE. No broker redistribution. Keeps us legally clean to republish.
2. **Idempotent pipeline** — re-running a date produces identical output. No hidden state.
3. **Partitioned parquet** — `exchange/year=YYYY/month=MM/date=YYYY-MM-DD.parquet`. Scales to decades, prunes well in DuckDB / HF.
4. **Pipeline ≠ API** — this repo ingests + publishes. Serving lives in a separate `tej-api` repo that reads our parquet.

---

## Phase 1 — Foundation (DONE)

- [x] README + license + scaffold
- [x] NSE holiday calendar (`exchange_calendars` wrapper, XBOM)
- [x] NSE bhavcopy fetcher (zip → CSV, retries, idempotent)
- [x] CSV → Polars DataFrame parser, normalized schema (SEBI CMTS)
- [x] Transform: filter series, validate prices, dedupe, sort
- [x] Local parquet writer with Hive partition layout
- [x] Typer + Rich CLI: `tej-bazaar fetch | backfill | info | version`
- [x] Golden-fixture tests (parser + transform + push + cli)

## Phase 2 — Publish

- [x] BSE bhavcopy fetcher + parser (same SEBI CMTS schema, plain CSV)
- [x] Exchange-aware transform (NSE: EQ/BE/BZ; BSE: A/B/T)
- [x] CLI `--exchange NSE|BSE|both` for fetch + backfill
- [x] HuggingFace push (`tej-bazaar publish`, content-hash dedup, dry-run)
- [ ] GitHub Actions cron (6:30 PM IST, weekdays, holiday-aware)
- [ ] Backfill script (one-shot historical load, both exchanges)
- [ ] Sample data committed under `data/sample/`

## Phase 3 — Mirror & resilience

- [ ] S3/R2 mirror (in addition to HF) for direct parquet download
- [ ] Health check / failure alerts (cron failures → GitHub issue or webhook)
- [ ] Retry + backoff on transient bhavcopy 5xx
- [ ] Source diff check — flag rows that change after publish (corporate actions)

## Phase 3.5 — Legacy historical data

The current pipeline targets the **SEBI CMTS bhavcopy format** (NSE: from 2024-01-01,
BSE: from ~mid-2023). Pre-cutover bhavcopies use legacy formats with different column
names and layouts:

- NSE legacy: `cm{DDMMYY}bhav.csv.zip` (e.g. `cm30APR23bhav.csv.zip`)
- BSE legacy: `EQ_ISINCODE_{DDMMYY}.zip` containing `EQ{DDMMYY}.CSV`

To extend coverage backward (2010s → 2023):

- [ ] Legacy NSE parser (different columns: SYMBOL, OPEN, HIGH, ...)
- [ ] Legacy BSE parser
- [ ] Format detection in `parse.py` — sniff header, dispatch to right parser
- [ ] Backfill validation against known-good external sources for sanity

Decision deferred — start with clean CMTS-era data, expand backward once steady-state
publishing works.

## Phase 4 — Corporate actions & adjustments

- [ ] Splits, bonuses, dividends feed (NSE corporate actions endpoint)
- [ ] Adjusted-close column, separate from raw close
- [ ] Symbol change history (renames, mergers, delistings)

## Phase 5 — Derived metrics

- [ ] Returns (daily, 5d, 20d, YTD)
- [ ] 52-week high/low
- [ ] Avg volume (5d, 20d)
- [ ] Distance from VWAP / EMA
- [ ] Published as separate `derived/` parquet alongside raw

## Phase 6 — SDKs & API handoff

- [ ] Hand off serving to `tej-api` (REST + auth tiers)
- [ ] `tej-sdk-py` — thin Python client over API + parquet
- [ ] `tej-sdk-js` — TypeScript client

---

## Open questions

- Bhavcopy URL stability — NSE has changed paths historically. Pin specific fetcher + integration test against live URL.
- BSE bhavcopy format drift — verify schema across years before backfill.
- Volume/turnover units — Bhavcopy reports paise vs rupees inconsistently. Lock units in transform layer.
- Delisted symbols — Bhavcopy includes them on their last trading day. Decide retention policy.
- Holiday calendar source of truth — `exchange_calendars` lib vs scraping NSE holiday master JSON. Lib is easier; scrape is canonical.
