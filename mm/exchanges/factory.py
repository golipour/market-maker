from __future__ import annotations

from dataclasses import dataclass

from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.exchanges.capabilities import resolve_public_market_transport
from mm.exchanges.nobitex import NobitexMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.wallex import WallexMarketDataAdapter


PublicMarketDataMode = TransportMode | str


@dataclass(frozen=True, slots=True)
class PublicMarketDataAdapterSelection:
    exchange: str
    symbol: str
    requested_mode: str
    transport: TransportMode
    adapter: PublicMarketDataAdapter


def build_public_market_data_adapter(
    exchange: str,
    symbol: str,
    *,
    mode: PublicMarketDataMode,
    depth_limit: int = 50,
    trades_limit: int = 100,
    poll_interval_ms: int = 1000,
) -> PublicMarketDataAdapter:
    return select_public_market_data_adapter(
        exchange,
        symbol,
        mode=mode,
        depth_limit=depth_limit,
        trades_limit=trades_limit,
        poll_interval_ms=poll_interval_ms,
    ).adapter


def select_public_market_data_adapter(
    exchange: str,
    symbol: str,
    *,
    mode: PublicMarketDataMode,
    depth_limit: int = 50,
    trades_limit: int = 100,
    poll_interval_ms: int = 1000,
) -> PublicMarketDataAdapterSelection:
    exchange_key = exchange.lower().strip()
    requested_mode = str(mode).lower().strip()
    transport = resolve_public_market_transport(exchange_key, requested_mode)
    if exchange_key == "nobitex":
        adapter = NobitexMarketDataAdapter(
            symbol,
            mode=transport,
            depth_limit=depth_limit,
            trades_limit=trades_limit,
            poll_interval_ms=poll_interval_ms,
        )
    elif exchange_key == "wallex":
        adapter = WallexMarketDataAdapter(
            symbol,
            mode=transport,
            depth_limit=depth_limit,
            trades_limit=trades_limit,
            poll_interval_ms=poll_interval_ms,
        )
    else:
        raise ValueError(f"unsupported exchange: {exchange}")

    return PublicMarketDataAdapterSelection(
        exchange=exchange_key,
        symbol=symbol,
        requested_mode=requested_mode,
        transport=transport,
        adapter=adapter,
    )
