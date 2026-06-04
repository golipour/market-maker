from __future__ import annotations

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.tabdeal.rest import TabdealRestMarketDataAdapter
from mm.exchanges.tabdeal.ws import TabdealWsMarketDataAdapter


def TabdealMarketDataAdapter(
    symbol: str,
    *,
    mode: TransportMode = "rest",
    trades_limit: int = 100,
    depth_limit: int = 50,
    poll_interval_ms: int = 1000,
) -> AbstractMarketDataAdapter:
    if mode == "rest":
        return TabdealRestMarketDataAdapter(
            symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            poll_interval_ms=poll_interval_ms,
        )
    if mode == "websocket":
        return TabdealWsMarketDataAdapter(symbol, depth_limit=depth_limit)
    raise ValueError(f"unsupported Tabdeal market-data mode: {mode}")
