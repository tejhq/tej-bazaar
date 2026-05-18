# tej-bazaar Roadmap

> **Mission**: Free, open, redistributable EOD market data for India. Audit-grade source. Zero auth to consume.

## Guiding principles

1. **Official source only**. Bhavcopy from NSE/BSE. No broker redistribution. Keeps us legally clean to republish.
2. **Idempotent pipeline**. Re-running a date produces identical output. No hidden state.
3. **Partitioned parquet**. Bhavcopy is Hive-partitioned by date; derived datasets are one file per exchange per year. Scales to decades, prunes well in DuckDB / HF.
4. **Pipeline is not the API**. This repo ingests and publishes. Serving lives in a separate `tej-api` repo that reads our parquet.

---

## Phase 1 - Foundation (DONE)

- [x] README + license + scaffold
- [x] NSE holiday calendar (`exchange_calendars` wrapper, XBOM)
- [x] NSE bhavcopy fetcher (zip to CSV, retries, idempotent)
- [x] CSV to Polars DataFrame parser, normalized schema (SEBI CMTS)
- [x] Transform: filter series, validate prices, dedupe, sort
- [x] Local parquet writer with Hive partition layout
- [x] Typer + Rich CLI: `tej-bazaar fetch | backfill | info | version`
- [x] Golden-fixture tests (parser + transform + push + cli)

## Phase 2 - Publish (DONE)

- [x] BSE bhavcopy fetcher + parser (same SEBI CMTS schema, plain CSV)
- [x] Exchange-aware transform (NSE: EQ/BE/BZ; BSE: A/B/T)
- [x] CLI `--exchange NSE|BSE|both` for fetch + backfill
- [x] HuggingFace push (`tej-bazaar publish`, content-hash dedup, dry-run)
- [x] GitHub Actions cron in `.github/workflows/daily.yml`: 13:30 UTC (19:00 IST), Mon-Fri, holiday-safe (skip publish if no parquet written)
- [x] Backfill script (`tej-bazaar backfill --from D --to D --exchange both`)
- [ ] Sample data committed under `data/sample/`

## Phase 3 - Mirror & resilience (NOT STARTED)

- [ ] S3/R2 mirror (in addition to HF) for direct parquet download
- [ ] Health check / failure alerts (cron failures to GitHub issue or Slack/Discord webhook)
- [ ] Retry + backoff on transient bhavcopy 5xx (partially in place for fetch; extend to actions + publish)
- [ ] Source diff check: flag rows that change after publish (late corp action attribution, restatements)

## Phase 3.5 - Legacy historical data (NOT STARTED)

The current pipeline targets the **SEBI CMTS bhavcopy format** (NSE: from 2024-01-01,
BSE: from ~mid-2023). Pre-cutover bhavcopies use legacy formats with different
column names and layouts:

- NSE legacy: `cm{DDMMYY}bhav.csv.zip` (e.g. `cm30APR23bhav.csv.zip`)
- BSE legacy: `EQ_ISINCODE_{DDMMYY}.zip` containing `EQ{DDMMYY}.CSV`

To extend coverage backward (2010s to 2023):

- [ ] Legacy NSE parser (different columns: SYMBOL, OPEN, HIGH, ...)
- [ ] Legacy BSE parser
- [ ] Format detection in `parse.py`: sniff header, dispatch to right parser
- [ ] Backfill validation against known-good external sources for sanity

## Phase 4 - Corporate actions & adjustments (DONE)

Bhavcopy publishes **unadjusted** prices. A 1:1 split looks like a -50% crash;
a Rs 5 dividend on a Rs 100 stock looks like a -5% drop. This phase added an
adjusted-close layer alongside raw.

### 4a - Corporate actions ingestion (DONE)

- [x] Fetch NSE corporate actions feed (`/api/corporates-corporateActions`, browser-header dance, ISIN-keyed)
- [x] Fetch BSE corporate actions feed (direct REST + scrip-master ISIN map)
- [x] Normalize into single `actions` table: exchange, symbol, isin, ex_date, type, ratio_num/den, cash_amount, face_value_from/to, raw_subject. Types: `dividend, split, bonus, rights, buyback, demerger, merger, agm, other`.
- [x] Idempotent fetcher: annual per-exchange parquet `actions/<ex>_<YYYY>.parquet`
- [x] Fixture-driven parser tests for each action type

### 4b - Adjustment factor computation (DONE)

- [x] Per (isin, ex_date) compute multiplicative factor:
  - Split / face-value change: factor = `face_value_to / face_value_from`
  - Bonus N:M: factor = `M / (N + M)`
  - Dividend D on prior close C: factor = `(C - D) / C`
- [x] Apply factors **backward** from latest date (reverse cumulative product, per-ISIN numpy / `searchsorted`)
- [x] Emit `prices_adjusted/<ex>_<YYYY>.parquet` with `adj_factor_cumulative` and `adj_close` columns alongside raw OHLC

### 4c - Symbol continuity (DONE)

- [x] `symbol_history/<ex>.parquet`: per-ISIN intervals of `(symbol, valid_from, valid_to, trading_days)`
- [x] Helper APIs `lookup_isin(symbol, on_date)` and `lookup_current_symbol(isin)`
- [x] In-memory build also used by the adjust step to resolve stale post-merger / pre-split ISINs that NSE still tags to legacy identifiers (HDFCBANK / KOTAKBANK / BAJFINANCE / SHRIRAMFIN cases)

### 4d - Reconciliation (DONE)

- [x] `tej-bazaar reconcile` CLI: compares local adjusted close to Yahoo `Adj Close` over a date range and symbol set
- [x] Headline: top 50 NSE by mean turnover, 2024-01-01 to 2026-05-06, **89% of ~25K daily comparisons within +-1%**
- [x] Residual gap is a documented methodology delta (NSE `(prev_close - cash) / prev_close` vs Yahoo CRSP `1 - cash / close_on_ex_date`), not a bug
- [x] `scripts/reconcile_yahoo_sweep.py` for bulk regression runs via `yfinance` (kept out of pipeline deps; install via optional `[reconcile]` extra)

## Phase 5 - Derived metrics (DONE)

- [x] Returns at 1d / 5d / 21d / 63d / 126d / 252d horizons plus YTD anchored to first trading day of the calendar year (`pipeline/metrics/returns.py`)
- [x] Rolling 52-week high / low on `adj_close`, plus `pct_off_52w_high` / `pct_off_52w_low` (`pipeline/metrics/rolling.py`)
- [x] Average volume 20d / 60d on raw `volume`, average turnover 20d on raw `turnover`
- [x] `tej-bazaar metrics build (--year YYYY | --all-years)`: writes `metrics/<ex>_<YYYY>.parquet`. Wired into the daily cron after `actions adjust`, before publish.
- [ ] Distance from VWAP / EMA (deferred; needs an intraday or weighted-bhavcopy input we do not have today)

## Phase 6 - SDKs & API handoff (NOT STARTED)

- [ ] Hand off serving to `tej-api` (REST + auth tiers) in a separate repo
- [ ] `tej-sdk-py`: thin Python client over API + parquet
- [ ] `tej-sdk-js`: TypeScript client

---

## Open questions

- Bhavcopy URL stability. NSE has changed paths historically. Pin specific fetcher + integration test against live URL.
- BSE bhavcopy format drift. Verify schema across years before extending backfill.
- Delisted symbols. Bhavcopy includes them on their last trading day. Decide retention policy.
- Cron failure visibility. Current `continue-on-error: true` on actions / adjust / metrics steps means a silent stale republish on transient breakage; pair with Phase 3 alerting.
