from __future__ import annotations

"""
Observed reference/representative price computations (ticks-based legacy).

These functions currently forward to `market_making_controller.ob_reference_prices`.
They exist to centralize imports and make future refactors localized.
"""

from market_making_controller.ob_reference_prices import (
    mid_ticks_after_market_order_flow as mid_after_market_order_flow_ticks,
    reference_prices_from_transition as reference_prices_from_transition_ticks,
    reference_prices_ticks_from_orderbook as reference_prices_from_orderbook_ticks,
)

