from __future__ import annotations

from dataclasses import dataclass

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.exchanges.capabilities import (
    RequestedTransportMode,
    resolve_private_state_mode,
    resolve_public_market_transport,
)
from mm.exchanges.nobitex import NobitexMarketDataAdapter
from mm.exchanges.nobitex.private_state import NobitexRestPrivateStateAdapter
from mm.exchanges.nobitex.order_adapter import NobitexRestOrderCommandAdapter
from mm.exchanges.wallex.order_adapter import WallexRestOrderCommandAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.wallex import WallexMarketDataAdapter
from mm.exchanges.wallex.private_state import WallexRestPrivateStateAdapter


PublicMarketDataMode = RequestedTransportMode | str
PrivateStateAdapterMode = RequestedTransportMode | str


@dataclass(frozen=True, slots=True)
class PublicMarketDataAdapterSelection:
    exchange: str
    symbol: str
    requested_mode: str
    transport: TransportMode
    adapter: PublicMarketDataAdapter


@dataclass(frozen=True, slots=True)
class PrivateStateAdapterSelection:
    exchange: str
    symbol: str
    requested_mode: str
    mode: str
    adapter: AbstractPrivateStateAdapter


@dataclass(frozen=True, slots=True)
class OrderCommandAdapterSelection:
    exchange: str
    symbol: str
    transport: str
    adapter: AbstractOrderCommandAdapter


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


def build_private_state_adapter(
    exchange: str,
    symbol: str,
    *,
    mode: PrivateStateAdapterMode,
    api_token: str | None,
    poll_interval_ms: int = 5000,
) -> AbstractPrivateStateAdapter:
    return select_private_state_adapter(
        exchange,
        symbol,
        mode=mode,
        api_token=api_token,
        poll_interval_ms=poll_interval_ms,
    ).adapter


def select_private_state_adapter(
    exchange: str,
    symbol: str,
    *,
    mode: PrivateStateAdapterMode,
    api_token: str | None,
    poll_interval_ms: int = 5000,
) -> PrivateStateAdapterSelection:
    exchange_key = exchange.lower().strip()
    requested_mode = str(mode).lower().strip()
    selected_mode = resolve_private_state_mode(exchange_key, requested_mode)
    if selected_mode != "rest":
        raise ValueError(f"{exchange_key} private state mode is not implemented yet: {selected_mode}")

    if exchange_key == "nobitex":
        adapter = NobitexRestPrivateStateAdapter(
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
    elif exchange_key == "wallex":
        adapter = WallexRestPrivateStateAdapter(
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
    else:
        raise ValueError(f"unsupported exchange: {exchange}")

    return PrivateStateAdapterSelection(
        exchange=exchange_key,
        symbol=symbol,
        requested_mode=requested_mode,
        mode=selected_mode,
        adapter=adapter,
    )


def build_order_command_adapter(
    exchange: str,
    symbol: str,
    *,
    api_token: str | None,
) -> AbstractOrderCommandAdapter:
    return select_order_command_adapter(exchange, symbol, api_token=api_token).adapter


def select_order_command_adapter(
    exchange: str,
    symbol: str,
    *,
    api_token: str | None,
) -> OrderCommandAdapterSelection:
    exchange_key = exchange.lower().strip()
    capabilities = resolve_order_command_transport(exchange_key)
    if exchange_key == "nobitex":
        adapter = NobitexRestOrderCommandAdapter(symbol, api_token=api_token)
    elif exchange_key == "wallex":
        adapter = WallexRestOrderCommandAdapter(symbol, api_token=api_token)
    else:
        raise ValueError(f"unsupported exchange for REST order commands: {exchange_key!r}")
    return OrderCommandAdapterSelection(
        exchange=exchange_key,
        symbol=symbol,
        transport=capabilities,
        adapter=adapter,
    )


def resolve_order_command_transport(exchange: str) -> str:
    from mm.exchanges.capabilities import get_exchange_capabilities

    capabilities = get_exchange_capabilities(exchange).order_commands
    if capabilities.place_rest and capabilities.cancel_rest:
        return "rest"
    raise ValueError(f"{exchange} does not support complete REST order commands")
