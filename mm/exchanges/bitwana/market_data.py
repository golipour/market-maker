from __future__ import annotations

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.bitwana.rest import BitwanaRestMarketDataAdapter


def BitwanaMarketDataAdapter(
    symbol: str,
    *,
    mode: TransportMode = "rest",
    trades_limit: int = 100,
    depth_limit: int = 50,
    poll_interval_ms: int = 1000,
) -> AbstractMarketDataAdapter:
    if mode == "rest":
        return BitwanaRestMarketDataAdapter(
            symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            poll_interval_ms=poll_interval_ms,
        )
    raise ValueError(f"unsupported Bitwana market-data mode: {mode}")
