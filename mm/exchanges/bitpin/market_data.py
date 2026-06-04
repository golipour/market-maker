from __future__ import annotations

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.exchanges.bitpin.rest import BitpinRestMarketDataAdapter
from mm.exchanges.bitpin.ws import BitpinWsMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode


def BitpinMarketDataAdapter(
    symbol: str,
    *,
    mode: TransportMode = "rest",
    trades_limit: int = 100,
    depth_limit: int = 50,
    poll_interval_ms: int = 1000,
) -> AbstractMarketDataAdapter:
    if mode == "rest":
        return BitpinRestMarketDataAdapter(
            symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            poll_interval_ms=poll_interval_ms,
        )
    if mode == "websocket":
        return BitpinWsMarketDataAdapter(symbol, depth_limit=depth_limit)
    raise ValueError(f"unsupported Bitpin market-data mode: {mode}")
