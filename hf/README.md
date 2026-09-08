---
license: mit
pretty_name: TejHQ Indian Markets
language:
  - en
tags:
  - finance
  - india
  - nse
  - bse
  - ohlcv
  - stocks
  - equity
  - corporate-actions
  - adjusted-close
  - backtesting
task_categories:
  - time-series-forecasting
  - tabular-classification
size_categories:
  - 10M<n<100M
configs:
  - config_name: nse
    data_files: "nse/year=*/nse_*.parquet"
    default: true
  - config_name: bse
    data_files: "bse/year=*/bse_*.parquet"
  - config_name: actions
    data_files: "actions/*_????.parquet"
  - config_name: prices_adjusted
    data_files: "prices_adjusted/*_????.parquet"
  - config_name: symbol_history
    data_files: "symbol_history/*.parquet"
  - config_name: metrics
    data_files: "metrics/*_????.parquet"
  - config_name: universe
    data_files: "universe/*_liquid.parquet"
---

# tejhq/indian-markets

End-of-day data for every **NSE** and **BSE** listed equity, built straight from the exchanges' official bhavcopy. Six parquet trees: raw prices, corporate actions, back-adjusted prices, symbol history, derived metrics, and a survivorship-bias-free liquidity universe. Refreshed every trading day at 20:00 IST by an open pipeline.

No broker, no auth, no scraping. The same data is served at [api.tejhq.dev](https://api.tejhq.dev) and mirrored on Cloudflare R2.

## Coverage

| Exchange | Series | Instruments | Coverage |
|----------|--------|-------------|----------|
| NSE | `EQ`, `BE`, `BZ` | ~2,300 / day | 2010-01-04 to today, ~4,100 trading days |
| BSE | `A`, `B`, `T` | ~2,200 / day | 2024-07-08 to today |

NSE and BSE both moved to the SEBI CMTS bhavcopy format on 2024-07-08. The pipeline parses both the modern CMTS layout and the legacy NSE layout, which is how NSE reaches back to 2010. Legacy BSE files identify instruments by numeric scrip code with no clean bridge to modern tickers, so BSE starts at the cutover.

Derived trees are computed per instrument chain, not per raw ISIN: when a face value split or re-listing issues a new ISIN under the same symbol within 30 days, adjustment factors, rolling windows and universe eligibility carry across the change. The `isin` column always holds the value the exchange reported on that day.

Pre-2012 NSE rows carry no `isin` and no `trades` because the source never had them. They are null in the raw tree rather than invented. The derived trees backfill the ISIN for symbols that traded continuously across the 2012 cutover so corporate actions and rolling windows apply back to 2010; symbols that did not survive the cutover stay ISIN-less and are windowed by symbol.

## The six trees

Pick a config in the viewer above to browse each one.

### 1. `nse/`, `bse/`: raw bhavcopy

One rollup per exchange per year: `<ex>/year=YYYY/<ex>_YYYY.parquet`. The current year's file is rewritten every trading day.

| Field | Type | Notes |
|-------|------|-------|
| `date` | date | Trading date |
| `symbol` | string | Ticker, e.g. `RELIANCE` |
| `series` | string | Exchange series code |
| `isin` | string | ISIN; null before 2012 on NSE |
| `name` | string | Full instrument name |
| `open`, `high`, `low`, `close` | float64 | Prices in INR |
| `last` | float64 | Last traded price |
| `prev_close` | float64 | Previous close |
| `volume` | int64 | Shares traded |
| `turnover` | float64 | Value traded, INR |
| `trades` | int64 | Trade count; null before 2012 on NSE |

### 2. `actions/`: corporate actions

One file per exchange per year: `actions/<ex>_YYYY.parquet`. NSE from 2010, BSE from 2024.

| Field | Type | Notes |
|-------|------|-------|
| `exchange` | string | `NSE` or `BSE` |
| `symbol` | string | Ticker on `ex_date` |
| `isin` | string | As reported by the source |
| `company` | string | Issuer name |
| `ex_date` | date | First day the price trades ex the action |
| `record_date` | date | Null when not reported |
| `type` | string | `dividend`, `split`, `bonus`, `rights`, `buyback`, `demerger`, `merger`, `agm`, `other` |
| `ratio_num`, `ratio_den` | int64 | Bonus and rights ratio |
| `cash_amount` | float64 | Per-share cash for dividends |
| `face_value_from`, `face_value_to` | float64 | For splits |
| `raw_subject` | string | Verbatim source text, kept for audit |

### 3. `prices_adjusted/`: back-adjusted prices

One file per exchange per year. Every raw bhavcopy column, plus:

| Field | Type | Notes |
|-------|------|-------|
| `adj_factor_cumulative` | float64 | Product of factors of all actions with `ex_date` after this row, per instrument chain. `1.0` when none. |
| `adj_close` | float64 | `close * adj_factor_cumulative`. Continuous through splits, bonuses and dividends. |

Dividend factor follows the NSE convention, `(prev_close - cash) / prev_close`, with `prev_close` taken from the session before the ex date across the full history; several payouts on one ex date are summed. Demergers, rights, buybacks and mergers are not scaled, which is where this series and Yahoo Finance differ by design. Reconciled at 96.88% within 1% of Yahoo Finance adjusted close across 25,329 daily comparisons on 48 symbols; the two outliers, TRENT and ITC, are Yahoo-side (a bonus applied on the wrong date, a demerger scaled into history).

### 4. `symbol_history/`: which symbol an ISIN traded under, and when

One file per exchange. Each row is one contiguous interval.

| Field | Type | Notes |
|-------|------|-------|
| `exchange` | string | `NSE` or `BSE` |
| `isin` | string | Stable instrument ID |
| `symbol` | string | Symbol active during the interval |
| `valid_from`, `valid_to` | date | First and last trading day |
| `trading_days` | int64 | Days in the interval |

### 5. `metrics/`: returns and rolling stats

One file per exchange per year, plus `<ex>_latest.parquet` with only the newest trading day for quick cross-sectional reads. Computed on `adj_close`; volume and turnover on raw values.

| Field | Type | Notes |
|-------|------|-------|
| `date`, `symbol`, `isin`, `adj_close` | | Carried for joining |
| `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d`, `ret_126d`, `ret_252d` | float64 | Simple returns over N trading days |
| `ret_ytd` | float64 | Since the first trading day of the year |
| `high_52w`, `low_52w` | float64 | Max and min `adj_close` over 252 trading days |
| `pct_off_52w_high`, `pct_off_52w_low` | float64 | `adj_close / high_52w - 1`, `adj_close / low_52w - 1` |
| `avg_vol_20d`, `avg_vol_60d` | float64 | Rolling mean of raw volume |
| `avg_turnover_20d` | float64 | Rolling mean of raw turnover |

Rolling fields are null until a full window of history exists. No partial windows.

### 6. `universe/`: point-in-time liquidity universe

One file per exchange: `universe/<ex>_liquid.parquet`. On the first trading day of each month, instruments are ranked by trailing 63-trading-day mean turnover and the top 500 are kept. Membership uses only data available on the rebalance date, so a name that later delisted still appears in the months it qualified for.

| Field | Type | Notes |
|-------|------|-------|
| `exchange` | string | `NSE` or `BSE` |
| `rebalance_date` | date | First trading day of the month |
| `valid_to` | date | Day before the next rebalance |
| `rank` | int64 | 1 = highest trailing turnover |
| `symbol`, `isin`, `name` | string | As in force on `rebalance_date` |
| `avg_turnover_63d` | float64 | Trailing mean turnover, INR |

`liquid100`, `liquid250` and `liquid500` are `rank <= N` filters over the same rows.

## Quickstart

### Polars, full history of one symbol

```python
import polars as pl

df = (
    pl.scan_parquet("hf://datasets/tejhq/indian-markets/nse/year=*/nse_*.parquet")
    .filter(pl.col("symbol") == "RELIANCE")
    .collect()
)
```

### DuckDB, adjusted close joined to the universe

```sql
SELECT p.date, p.symbol, p.adj_close
FROM read_parquet('hf://datasets/tejhq/indian-markets/prices_adjusted/nse_2024.parquet') p
JOIN read_parquet('hf://datasets/tejhq/indian-markets/universe/nse_liquid.parquet') u
  ON u.isin = p.isin
 AND p.date BETWEEN u.rebalance_date AND u.valid_to
WHERE u.rank <= 100
ORDER BY p.date, p.symbol;
```

### `datasets`

```python
from datasets import load_dataset

nse = load_dataset("tejhq/indian-markets", "nse")
actions = load_dataset("tejhq/indian-markets", "actions")
```

### REST

```bash
curl "https://api.tejhq.dev/v1/ohlcv/nse/RELIANCE?from=2024-01-01&to=2024-12-31"
```

No key needed for prices, snapshots, actions and the pipeline status at `/v1/status`. Adjusted prices and symbol history need a free key from [tejhq.dev/keys](https://tejhq.dev/keys). Metrics, universe, batch, resolve and the screener are on the Pro tier.

## Caveats

- Same-day rows land at about 20:00 IST. Fetching earlier returns the previous session.
- Bhavcopy occasionally carries anomalies (close outside `[low, high]`, zero-volume rows). The pipeline drops those before publishing.
- Corporate action feeds are messy at the source. `type = other` keeps events the classifier could not place, with the original text in `raw_subject`.
- `actions.isin` is as reported and can be a stale post-merger ISIN. The adjusted tree resolves it; join on `symbol_history` if you need the same.

## Source and license

- **Source:** NSE and BSE official EOD bhavcopy and corporate action feeds. Public, free, redistributable.
- **Pipeline:** [github.com/tejhq/tej-bazaar](https://github.com/tejhq/tej-bazaar), MIT. Every tree here is reproducible from it.
- **Data:** Exchange-published. Redistribution as cleaned parquet is permitted. Verify exchange terms before commercial use.

Part of [TejHQ](https://tejhq.dev): open, free, end-of-day market data for India. Build log on the [blog](https://tejhq.dev/blog).
