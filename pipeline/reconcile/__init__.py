"""Reconcile our adjusted prices against an external truth source (Yahoo).

This module exists to give the dataset a verifiable quality claim: adjusted
closes from `tej-bazaar` should match Yahoo's adjusted closes within a small
tolerance. If they diverge meaningfully we know our split/dividend math, our
back-adjustment, or our raw bhavcopy ingestion has a bug.
"""

from pipeline.reconcile.compare import (
    ReconcileResult,
    SymbolReconcileStats,
    reconcile_symbol,
    summarize,
)
from pipeline.reconcile.yahoo import (
    YahooFetchError,
    fetch_yahoo_adjusted,
)

__all__ = [
    "ReconcileResult",
    "SymbolReconcileStats",
    "YahooFetchError",
    "fetch_yahoo_adjusted",
    "reconcile_symbol",
    "summarize",
]
