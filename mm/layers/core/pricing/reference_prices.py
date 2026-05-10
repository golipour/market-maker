from __future__ import annotations

"""
Observed reference/representative prices (**ticks**) — canonical implementation in ``mm.pricing``.

Depth for VWAP is aligned via ``mm.layers.core.pricing.policies.DEFAULT_REFERENCE_DEPTH_LOTS`` and
``price_model._target_depth_lots``. Wealth marking lives in ``mm.layers.core.pricing.valuation``.
"""

from mm.pricing import (
    mid_after_market_order_flow_ticks,
    reference_prices_from_orderbook_ticks,
    reference_prices_from_transition_ticks,
)

