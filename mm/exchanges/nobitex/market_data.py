from __future__ import annotations

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.exchanges.nobitex.rest import NobitexRestMarketDataAdapter
from mm.exchanges.nobitex.ws import NobitexWsMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode


def NobitexMarketDataAdapter(
        symbol: str,
        *,
        mode: TransportMode = "rest",
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
) -> AbstractMarketDataAdapter:
    if mode == "rest":
        return NobitexRestMarketDataAdapter(
            symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            poll_interval_ms=poll_interval_ms,
        )
    if mode == "websocket":
        return NobitexWsMarketDataAdapter(symbol, depth_limit=depth_limit)
    raise ValueError(f"unsupported Nobitex market-data mode: {mode}")
