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
from pipeline.actions.scrip_map import (
    build_scrip_to_isin,
    fetch_bse_scrip_master,
    load_bse_scrip_to_isin,
)

__all__ = [
    "ACTION_SCHEMA",
    "ACTION_TYPES",
    "ActionsFetchError",
    "CorporateAction",
    "build_scrip_to_isin",
    "fetch_actions",
    "fetch_bse_actions",
    "fetch_bse_scrip_master",
    "fetch_nse_actions",
    "load_bse_scrip_to_isin",
    "parse_actions",
    "parse_bse_record",
    "parse_nse_record",
    "to_polars",
]
