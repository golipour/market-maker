"""
Pricing: observed refs (ticks), simulated reps (torch), depth policy, valuation.

Observed reference math is implemented in ``mm.pricing``;
**import through this package** (e.g. ``reference_prices_from_orderbook_ticks``) so call sites
do not depend on the controller module path. Valuation and depth live under ``policies`` /
``valuation`` / ``depth``.
"""

from mm.layers.core.pricing.depth import target_depth_lots_from_price_model
from mm.layers.core.pricing.policies import (
    DEFAULT_REFERENCE_DEPTH_LOTS,
    DEFAULT_VALUATION_POLICY,
    ValuationPolicy,
    resolve_valuation_policy,
    valuation_policy_from_str,
)
from mm.layers.core.pricing.reference_prices import (
    mid_after_market_order_flow_ticks,
    reference_prices_from_transition_ticks,
    reference_prices_from_orderbook_ticks,
)
from mm.layers.core.pricing.valuation import (
    main_inventory_value_ticks,
    portfolio_wealth_quote,
    portfolio_wealth_ticks,
)

__all__ = [
    "ValuationPolicy",
    "DEFAULT_VALUATION_POLICY",
    "DEFAULT_REFERENCE_DEPTH_LOTS",
    "valuation_policy_from_str",
    "resolve_valuation_policy",
    "target_depth_lots_from_price_model",
    "portfolio_wealth_ticks",
    "portfolio_wealth_quote",
    "main_inventory_value_ticks",
    "mid_after_market_order_flow_ticks",
    "reference_prices_from_transition_ticks",
    "reference_prices_from_orderbook_ticks",
]
