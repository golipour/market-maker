"""
Exchange-facing IO (REST, WebSockets, venue quirks).

Normalization helpers live here (canonical floats via ``mm_core``); raw REST/WS polling
still partly lives in ``trade_stats`` / ``mm_runtime.order_flow_stepper`` while we migrate.
"""

from mm_adapters.exchanges.execution import (
    execution_report_for_json,
    execution_report_from_place_response,
    place_limit_order_lots_with_report,
    place_order_intent_ticks,
)
from mm_adapters.exchanges.orderbook import to_canonical_orderbook

__all__ = [
    "to_canonical_orderbook",
    "place_limit_order_lots_with_report",
    "place_order_intent_ticks",
    "execution_report_from_place_response",
    "execution_report_for_json",
]
