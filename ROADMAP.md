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
- [x] GitHub Actions cron in `.github/workflows/daily.yml`: 14:30 UTC (20:00 IST), Mon-Fri, holiday-safe (skip publish if no parquet written)
- [x] Backfill script (`tej-bazaar backfill --from D --to D --exchange both`)
- [ ] Sample data committed under `data/sample/`

## Phase 3 - Mirror & resilience (PARTIAL)

- [x] R2 mirror (`publish_r2.py`) - serves tej-api via Cloudflare R2 + DuckDB httpfs
- [ ] HF + R2 parity check: nightly diff to flag drift
- [x] Cron split into `prices` / `derived` jobs (2026-09-08). `continue-on-error` only on the NSE corp-actions website fetch; every other failure fails the run, so GitHub emails on it. Optional `ALERT_WEBHOOK_URL` secret posts to Discord/Slack.
- [x] `tej-bazaar status write` publishes `api/v1/status.json` (trading date, prices and derived publish times, run URL) from both cron jobs; served keyless at `api.tejhq.dev/v1/status` (2026-09-08)
- [ ] Health check beyond cron (probe R2/HF freshness from outside)
- [ ] Retry + backoff on transient bhavcopy 5xx (partially in place for fetch; extend to actions + publish)
- [ ] Source diff check: flag rows that change after publish (late corp action attribution, restatements)

## Phase 3.5 - Legacy historical data (DONE)

Extended coverage back to 2010 via the SEBI pre-cutover format:

- [x] Legacy NSE parser: SYMBOL/SERIES/TIMESTAMP layout, pre-2012 rows tolerate missing ISIN + TOTALTRADES
- [x] Format detection in `parse.py`: sniff header, dispatch to `_parse_modern` or `_parse_legacy_nse`
- [x] 2-digit year handling in TIMESTAMP (NSE 2020-07-13 shipped `13-Jul-20` instead of `13-JUL-2020`)
- [ ] Legacy BSE parser - skipped. Numeric SC_CODE vs alphanumeric modern ticker has no clean bridge; BSE scoped to 2024-07-08+
- [x] Sanity: 11/11 representative dates 2010-2026 return 200 from tej-api with correct row counts

## Phase 3.6 - R2 storage shape (DONE)

Backfill expanded R2 to ~4500 daily parquets, which destroyed query LIST cost on tej-api. Reshaped:

- [x] `tej-bazaar compact --year YYYY -e <ex>` writes one `year=YYYY/<ex>_<YYYY>.parquet` rollup per year
- [x] Rollup sorted by `(symbol, date)` so DuckDB row-group stats prune ~99% of bytes on single-symbol queries
- [x] `--from-r2` + `--refresh` flags so daily cron can rebuild current-year rollup from R2 (rollup + new dailies, dedup by `(symbol, date)`)
- [x] `tej-bazaar r2-prune --prefix <pfx>` deletes a key prefix safely (refuses bucket-root prefixes)
- [x] Daily cron extended: publish-r2 → compact --refresh current year → r2-prune day-keys, so each `year=YYYY/` stays at exactly 1 parquet all year round

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

- [x] Per (instrument chain, ex_date) compute multiplicative factor:
  - Split / face-value change: factor = `face_value_to / face_value_from`
  - Bonus N:M: factor = `M / (N + M)`
  - Dividend D on prior close C: factor = `(C - D) / C`, with C looked up across the full price history, not the year being adjusted
  - Rights, buyback, demerger, merger, AGM, other: factor `1.0`
- [x] Apply factors **backward** from latest date (reverse cumulative product, per instrument chain, numpy / `searchsorted`)
- [x] Instrument chains (2026-09-08, `pipeline/instrument.py`): ISIN intervals are linked when the same symbol moves to a new ISIN within 30 calendar days. Adjusted prices, metrics and universe partition by chain; the `isin` column in every published file stays as reported
- [x] Multi-part dividends (2026-09-08): `dividend_cash` sums several amounts on one ex date, except a distribution that states its total and then breaks it down. `tej-bazaar actions reparse` rewrites stored action files from `raw_subject` and runs in the cron before adjust
- [x] Emit `prices_adjusted/<ex>_<YYYY>.parquet` with `adj_factor_cumulative` and `adj_close` columns alongside raw OHLC

### 4c - Symbol continuity (DONE)

- [x] `symbol_history/<ex>.parquet`: per-ISIN intervals of `(symbol, valid_from, valid_to, trading_days)`
- [x] Helper APIs `lookup_isin(symbol, on_date)` and `lookup_current_symbol(isin)`
- [x] In-memory build also used by the adjust step to resolve stale post-merger / pre-split ISINs that NSE still tags to legacy identifiers (HDFCBANK / KOTAKBANK / BAJFINANCE / SHRIRAMFIN cases)

### 4d - Reconciliation (DONE)

- [x] `tej-bazaar reconcile` CLI: compares local adjusted close to Yahoo `Adj Close` over a date range and symbol set, fetched through `yfinance` (`pipeline/reconcile/yahoo.py`, lazy import, optional `[reconcile]` extra)
- [x] Headline (2026-09-08, `reconcile_report.md`, tracked): top 50 NSE by mean turnover, 2024-01-01 to 2026-05-06, **96.88% of 25,329 daily comparisons within +-1%** across 48 symbols, up from 88.79% before instrument chains, the full-history `prev_close` lookup and summed same-day dividends
- [x] Residual is Yahoo-side: TRENT (bonus applied on the wrong date) and ITC (Yahoo scales history for the hotels demerger, NSE convention does not). Yahoo's dividend factor equals the NSE convention, so dividends are not a source of difference

## Phase 5 - Derived metrics (DONE)

- [x] Returns at 1d / 5d / 21d / 63d / 126d / 252d horizons plus YTD anchored to first trading day of the calendar year (`pipeline/metrics/returns.py`)
- [x] Rolling 52-week high / low on `adj_close`, plus `pct_off_52w_high` / `pct_off_52w_low` (`pipeline/metrics/rolling.py`)
- [x] Average volume 20d / 60d on raw `volume`, average turnover 20d on raw `turnover`
- [x] `tej-bazaar metrics build (--year YYYY | --all-years)`: writes `metrics/<ex>_<YYYY>.parquet`. Wired into the daily cron after `actions adjust`, before publish.
- [x] `metrics/<ex>_latest.parquet` (newest day only) and, with `--slices-dir`, monthly `metrics/<ex>_YYYY-MM.parquet` slices written to R2 only (`data/api/metrics` in the cron) so the tej-api screener reads one small file per date (2026-09-08)
- [x] ISIN backfill across the 2011/2012 cutover for derived datasets (2026-09-08). Before this, 2010 and 2011 adjusted prices were unadjusted and metrics merged every ISIN-less symbol into one series.
- [ ] Distance from VWAP / EMA (deferred; needs an intraday or weighted-bhavcopy input we do not have today)

## Phase 5.5 - Liquidity universe (DONE)

- [x] `tej-bazaar universe build`: monthly top-500 by trailing 63-trading-day turnover per exchange, point-in-time symbols/ISINs, delisted names kept. `universe/<ex>_liquid.parquet`, wired into the derived cron job. Served by tej-api `/v1/universe/liquid{100,250,500}?as_of=`.
- [ ] Real index constituents (NIFTY50/500, SENSEX) from NSE index-change history

## Phase 6 - SDKs & API handoff (PARTIAL)

- [x] Hand off serving to `tej-api` (REST + auth tiers) in a separate repo, live at `api.tejhq.dev`
- [x] `tej-sdk-py`: thin Python client over the API, published as `tejhq` 0.3.0 on PyPI
- [ ] `tej-sdk-ts`: TypeScript client, built (0.1.0, same surface as the Python SDK), not yet published to npm

---

## Open questions

- Bhavcopy URL stability. NSE has changed paths historically. Pin specific fetcher + integration test against live URL.
- BSE bhavcopy format drift. Verify schema across years before extending backfill.
- Delisted symbols. Bhavcopy includes them on their last trading day. Decide retention policy.
- Cron failure visibility: resolved by the prices/derived split. Derived artifacts are built into `data/derived/` and published from there, so a failed step publishes nothing rather than yesterday's files.
