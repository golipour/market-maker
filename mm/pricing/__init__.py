"""Pricing helpers used by ``mm.mpc`` (tick VWAP refs and simulated reps)."""

from __future__ import annotations

from mm.pricing.constants import DEFAULT_REFERENCE_DEPTH_LOTS
from mm.pricing.depth import target_depth_lots_from_price_model
from mm.pricing.reference_prices_ticks import (
    mid_ticks_after_market_order_flow,
    reference_prices_from_transition_ticks,
    reference_prices_ticks_from_orderbook,
)
from mm.pricing.simulated_rep_prices import representative_prices_batched

# Tick/lot reference price helpers shared with ``mm.mpc``.
mid_after_market_order_flow_ticks = mid_ticks_after_market_order_flow
reference_prices_from_orderbook_ticks = reference_prices_ticks_from_orderbook

__all__ = [
    "DEFAULT_REFERENCE_DEPTH_LOTS",
    "mid_after_market_order_flow_ticks",
    "mid_ticks_after_market_order_flow",
    "reference_prices_from_orderbook_ticks",
    "reference_prices_from_transition_ticks",
    "reference_prices_ticks_from_orderbook",
    "representative_prices_batched",
    "target_depth_lots_from_price_model",
]
