"""Pricing helpers used by ``mm.mpc`` (tick VWAP, batched grid VWAP, raw-book VWAP)."""

from __future__ import annotations

from mm.pricing.constants import DEFAULT_VWAP_DEPTH_LOTS
from mm.pricing.depth import target_depth_lots_from_price_model
from mm.pricing.orderbook_ref import mid_raw_orderbook, vwap_from_raw_orderbook
from mm.pricing.reference_prices_ticks import (
    mid_ticks_after_market_order_flow,
    reference_prices_from_transition_ticks,
    reference_prices_ticks_from_orderbook,
)
from mm.pricing.simulated_rep_prices import vwap_prices_batched

# Tick/lot VWAP helpers (integer ``Orderbook``) shared with ``mm.mpc``.
mid_after_market_order_flow_ticks = mid_ticks_after_market_order_flow
reference_prices_from_orderbook_ticks = reference_prices_ticks_from_orderbook

__all__ = [
    "DEFAULT_VWAP_DEPTH_LOTS",
    "mid_raw_orderbook",
    "mid_after_market_order_flow_ticks",
    "mid_ticks_after_market_order_flow",
    "reference_prices_from_orderbook_ticks",
    "reference_prices_from_transition_ticks",
    "reference_prices_ticks_from_orderbook",
    "target_depth_lots_from_price_model",
    "vwap_from_raw_orderbook",
    "vwap_prices_batched",
]
