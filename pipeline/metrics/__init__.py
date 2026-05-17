"""Derived metrics built on top of back-adjusted prices.

Returns, rolling 52-week highs/lows, average volumes, and similar
window-based summaries. Inputs come from `pipeline.actions.back_adjust`
(adjusted close drives returns) and the raw bhavcopy parquet (volume +
turnover drive activity metrics).
"""

from pipeline.metrics.returns import RETURNS_SCHEMA, compute_returns

__all__ = ["RETURNS_SCHEMA", "compute_returns"]
