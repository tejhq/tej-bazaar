"""Corporate actions ingestion + parsing for NSE/BSE.

Bhavcopy publishes unadjusted prices. This sub-package fetches the raw
corporate-action feeds and normalizes them so a downstream adjustment layer
can build back-adjusted prices.
"""

from pipeline.actions.fetch import (
    ActionsFetchError,
    fetch_actions,
    fetch_bse_actions,
    fetch_nse_actions,
)
from pipeline.actions.parse import (
    parse_actions,
    parse_bse_record,
    parse_nse_record,
)
from pipeline.actions.schema import (
    ACTION_SCHEMA,
    ACTION_TYPES,
    CorporateAction,
    to_polars,
)

__all__ = [
    "ACTION_SCHEMA",
    "ACTION_TYPES",
    "ActionsFetchError",
    "CorporateAction",
    "fetch_actions",
    "fetch_bse_actions",
    "fetch_nse_actions",
    "parse_actions",
    "parse_bse_record",
    "parse_nse_record",
    "to_polars",
]
