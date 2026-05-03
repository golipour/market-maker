"""
Pricing: observed refs (ticks), simulated reps (torch), depth policy, valuation.

Observed reference math remains in ``market_making_controller.ob_reference_prices``; we re-export
here as the stable ``mm_core`` surface. Valuation and depth defaults are Phase 5.
"""

from mm_core.pricing.depth import target_depth_lots_from_price_model
from mm_core.pricing.policies import (
    DEFAULT_REFERENCE_DEPTH_LOTS,
    DEFAULT_VALUATION_POLICY,
    ValuationPolicy,
)
from mm_core.pricing.reference_prices import (
    mid_after_market_order_flow_ticks,
    reference_prices_from_transition_ticks,
    reference_prices_from_orderbook_ticks,
)
from mm_core.pricing.valuation import portfolio_wealth_quote, portfolio_wealth_ticks

__all__ = [
    "ValuationPolicy",
    "DEFAULT_VALUATION_POLICY",
    "DEFAULT_REFERENCE_DEPTH_LOTS",
    "target_depth_lots_from_price_model",
    "portfolio_wealth_ticks",
    "portfolio_wealth_quote",
    "mid_after_market_order_flow_ticks",
    "reference_prices_from_transition_ticks",
    "reference_prices_from_orderbook_ticks",
]
