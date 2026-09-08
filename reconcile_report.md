# tej-bazaar vs Yahoo Finance: reconciliation report

- Run: **2026-09-08**, from a local rebuild of `prices_adjusted` with instrument chains, full-history `prev_close` lookup and summed same-day dividends (the three fixes shipped that day)
- Range: **2024-01-01 to 2026-05-06**
- Exchange: **NSE**
- Symbols requested: 50 (top by mean daily turnover)
- Symbols matched: 48 (TATAMOTORS and ZOMATO have no continuous Yahoo series across their renames)
- Total row-comparisons: 25329
- Tolerance: ±1.0%
- **Overall within tolerance: 96.88%** (previous report, same range and symbols: 88.79%)

## What the residual is

Forty-six of forty-eight symbols reconcile at 98.96% to 100% with mean differences under 0.05%. The two exceptions are Yahoo's, not the pipeline's:

- **TRENT** (14.26%): Yahoo applies the 1:2 bonus on 2026-01-01. The exchange ex-date is 2026-06-04, where the raw close drops 33% and Yahoo's does not. Every row between the two dates differs by the bonus factor.
- **ITC** (56.87%): Yahoo scales history for the January 2025 hotels demerger. tej-bazaar follows the NSE convention and leaves demergers unadjusted, so rows before 2025-01-06 differ by about 3.7%.

Yahoo's own `Close` series is split-adjusted; the comparison uses `Adj Close`, which adds dividends. Yahoo's dividend factor `1 - cash / prev_close` equals the NSE convention `(prev_close - cash) / prev_close` exactly, so dividend formula is not a source of difference.

## Per-symbol

| Symbol | Rows | Within tol | Max diff % | Mean diff % |
|--------|-----:|-----------:|-----------:|------------:|
| TRENT | 575 | 14.26% | 33.430 | 28.592 |
| ITC | 575 | 56.87% | 3.981 | 1.677 |
| ADANIENT | 575 | 98.96% | 3.148 | 0.033 |
| ETERNAL | 262 | 99.24% | 5.128 | 0.030 |
| M&M | 575 | 99.65% | 3.934 | 0.013 |
| MAZDOCK | 575 | 99.65% | 2.397 | 0.009 |
| JIOFIN | 575 | 99.65% | 4.587 | 0.013 |
| IRFC | 575 | 99.65% | 14.727 | 0.030 |
| PAYTM | 575 | 99.65% | 7.621 | 0.019 |
| SHRIRAMFIN | 575 | 99.65% | 3.057 | 0.008 |
| WAAREEENER | 373 | 99.73% | 3.021 | 0.010 |
| OLAELEC | 427 | 99.77% | 12.444 | 0.031 |
| HDFCBANK | 575 | 99.83% | 1.275 | 0.003 |
| ICICIBANK | 575 | 99.83% | 3.219 | 0.006 |
| INFY | 575 | 99.83% | 1.214 | 0.002 |
| SBIN | 575 | 99.83% | 1.874 | 0.004 |
| BSE | 575 | 99.83% | 4.653 | 0.008 |
| AXISBANK | 575 | 99.83% | 1.073 | 0.002 |
| TCS | 575 | 99.83% | 1.565 | 0.003 |
| KOTAKBANK | 575 | 99.83% | 2.052 | 0.004 |
| BAJFINANCE | 575 | 99.83% | 1.191 | 0.002 |
| LT | 575 | 99.83% | 3.064 | 0.006 |
| HAL | 575 | 99.83% | 4.071 | 0.007 |
| BEL | 575 | 99.83% | 1.922 | 0.003 |
| IDEA | 575 | 99.83% | 2.305 | 0.004 |
| VEDL | 575 | 99.83% | 2.920 | 0.005 |
| MARUTI | 575 | 99.83% | 1.392 | 0.002 |
| DIXON | 575 | 99.83% | 2.840 | 0.006 |
| TATASTEEL | 575 | 99.83% | 1.877 | 0.004 |
| RVNL | 575 | 99.83% | 1.431 | 0.003 |
| NTPC | 575 | 99.83% | 1.719 | 0.016 |
| SILVERBEES | 574 | 99.83% | 1.053 | 0.004 |
| RECLTD | 575 | 99.83% | 2.095 | 0.004 |
| CDSL | 575 | 99.83% | 5.816 | 0.011 |
| ADANIPORTS | 575 | 99.83% | 1.904 | 0.004 |
| BHEL | 575 | 99.83% | 4.324 | 0.008 |
| PFC | 575 | 99.83% | 3.159 | 0.020 |
| HINDCOPPER | 575 | 99.83% | 2.774 | 0.005 |
| HINDUNILVR | 575 | 99.83% | 1.393 | 0.007 |
| RELIANCE | 575 | 100.00% | 0.183 | 0.000 |
| BHARTIARTL | 575 | 100.00% | 0.723 | 0.002 |
| GROWW | 117 | 100.00% | 0.000 | 0.000 |
| INDIGO | 575 | 100.00% | 0.540 | 0.002 |
| MEESHO | 97 | 100.00% | 0.000 | 0.000 |
| SWIGGY | 362 | 100.00% | 0.428 | 0.002 |
| TMCV | 117 | 100.00% | 0.000 | 0.000 |
| INDUSINDBK | 575 | 100.00% | 0.598 | 0.002 |
| HCLTECH | 575 | 100.00% | 0.845 | 0.002 |

## How to reproduce

```bash
pip install -e '.[reconcile]'
tej-bazaar pull-r2 --prefix nse/ --prefix actions/ --data-dir data/out --bucket tej-bazaar
tej-bazaar actions reparse --actions-dir data/out/actions
tej-bazaar actions adjust --all-years -e nse --prices-dir data/out --actions-dir data/out/actions --out-dir data/derived/prices_adjusted
tej-bazaar reconcile --from 2024-01-01 --to 2026-05-06 -e nse --top 50 --adjusted-dir data/derived/prices_adjusted --tolerance 1.0
```
