from __future__ import annotations

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.exchanges.kucoin.rest import KucoinRestMarketDataAdapter
from mm.exchanges.kucoin.ws import KucoinWsMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode


def KucoinMarketDataAdapter(
    symbol: str,
    *,
    mode: TransportMode = "rest",
    trades_limit: int = 100,
    depth_limit: int = 50,
    poll_interval_ms: int = 1000,
) -> AbstractMarketDataAdapter:
    if mode == "rest":
        return KucoinRestMarketDataAdapter(
            symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            poll_interval_ms=poll_interval_ms,
        )
    if mode == "websocket":
        return KucoinWsMarketDataAdapter(symbol, depth_limit=depth_limit)
    raise ValueError(f"unsupported KuCoin market-data mode: {mode}")
