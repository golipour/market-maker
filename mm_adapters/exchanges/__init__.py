"""
Exchange-facing IO (REST, WebSockets, venue quirks).

Normalization helpers live here (canonical floats via ``mm_core``); raw REST/WS polling
still partly lives in ``trade_stats`` / ``mm_runtime.order_flow_stepper`` while we migrate.
"""

from mm_adapters.exchanges.orderbook import to_canonical_orderbook

__all__ = ["to_canonical_orderbook"]
