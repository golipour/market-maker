from __future__ import annotations

"""
Observed reference/representative prices (**ticks**) — single implementation in
``market_making_controller.ob_reference_prices``.

Depth for VWAP is aligned via ``mm_core.pricing.policies.DEFAULT_REFERENCE_DEPTH_LOTS`` and
``price_model._target_depth_lots``. Wealth marking lives in ``mm_core.pricing.valuation``.
"""

from market_making_controller.ob_reference_prices import (
    mid_ticks_after_market_order_flow as mid_after_market_order_flow_ticks,
    reference_prices_from_transition as reference_prices_from_transition_ticks,
    reference_prices_ticks_from_orderbook as reference_prices_from_orderbook_ticks,
)

