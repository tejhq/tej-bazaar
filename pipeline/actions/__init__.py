"""Corporate actions ingestion + parsing for NSE/BSE.

Bhavcopy publishes unadjusted prices. This sub-package fetches the raw
corporate-action feeds and normalizes them so a downstream adjustment layer
can build back-adjusted prices.
"""

from pipeline.actions.back_adjust import (
    back_adjust,
    compute_action_factors,
    resolve_isin_via_symbol_history,
)
from pipeline.actions.factors import compute_factor, needs_prev_close
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
    "back_adjust",
    "build_scrip_to_isin",
    "compute_action_factors",
    "compute_factor",
    "fetch_actions",
    "fetch_bse_actions",
    "fetch_bse_scrip_master",
    "fetch_nse_actions",
    "load_bse_scrip_to_isin",
    "needs_prev_close",
    "parse_actions",
    "parse_bse_record",
    "parse_nse_record",
    "resolve_isin_via_symbol_history",
    "to_polars",
]
