from __future__ import annotations

from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.exchanges.capabilities import validate_public_market_transport
from mm.exchanges.nobitex import NobitexMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.wallex import WallexMarketDataAdapter


def build_public_market_data_adapter(
    exchange: str,
    symbol: str,
    *,
    mode: TransportMode,
    depth_limit: int = 50,
    trades_limit: int = 100,
    poll_interval_ms: int = 1000,
) -> PublicMarketDataAdapter:
    exchange_key = exchange.lower().strip()
    validate_public_market_transport(exchange_key, mode)
    if exchange_key == "nobitex":
        return NobitexMarketDataAdapter(
            symbol,
            mode=mode,
            depth_limit=depth_limit,
            trades_limit=trades_limit,
            poll_interval_ms=poll_interval_ms,
        )
    if exchange_key == "wallex":
        return WallexMarketDataAdapter(
            symbol,
            mode=mode,
            depth_limit=depth_limit,
            trades_limit=trades_limit,
            poll_interval_ms=poll_interval_ms,
        )
    raise ValueError(f"unsupported exchange: {exchange}")
