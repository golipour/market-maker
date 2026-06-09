from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.exchanges.capabilities import (
    EXCHANGE_CAPABILITIES,
    RequestedTransportMode,
    get_exchange_capabilities,
    resolve_private_state_mode,
    resolve_public_market_transport,
)
from mm.exchanges.registry import get_exchange_registration, registered_exchanges
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.shared.venue_limits_trade_stats import TradeStatsVenueOrderLimitSource
from mm.execution.order_validation import NoOpVenueOrderLimitSource


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
    get_exchange_capabilities(exchange_key)
    registration = get_exchange_registration(exchange_key)
    transport = resolve_public_market_transport(exchange_key, requested_mode)
    adapter = registration.build_market_data(
        symbol,
        transport=transport,
        depth_limit=depth_limit,
        trades_limit=trades_limit,
        poll_interval_ms=poll_interval_ms,
    )
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
    api_token: Any,
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
    api_token: Any,
    poll_interval_ms: int = 5000,
) -> PrivateStateAdapterSelection:
    exchange_key = exchange.lower().strip()
    requested_mode = str(mode).lower().strip()
    get_exchange_capabilities(exchange_key)
    registration = get_exchange_registration(exchange_key)
    selected_mode = resolve_private_state_mode(exchange_key, requested_mode)
    adapter = registration.build_private_state(
        symbol,
        selected_mode=selected_mode,
        api_token=api_token,
        poll_interval_ms=poll_interval_ms,
    )
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
    api_token: Any,
) -> AbstractOrderCommandAdapter:
    return select_order_command_adapter(exchange, symbol, api_token=api_token).adapter


def select_order_command_adapter(
    exchange: str,
    symbol: str,
    *,
    api_token: Any,
    placement_max_concurrency: int = 8,
) -> OrderCommandAdapterSelection:
    exchange_key = exchange.lower().strip()
    capabilities = resolve_order_command_transport(exchange_key)
    registration = get_exchange_registration(exchange_key)
    adapter = registration.build_order_command(
        symbol,
        api_token=api_token,
        max_concurrency=max(1, int(placement_max_concurrency)),
    )
    return OrderCommandAdapterSelection(
        exchange=exchange_key,
        symbol=symbol,
        transport=capabilities,
        adapter=adapter,
    )


def select_venue_order_limit_source(exchange: str) -> AbstractVenueOrderLimitSource:
    """Venue min base / min notional for float ``Order`` validation before REST."""

    exchange_key = exchange.lower().strip()
    registration = get_exchange_registration(exchange_key)
    if registration.uses_trade_stats_limits:
        return TradeStatsVenueOrderLimitSource(exchange_key)
    return NoOpVenueOrderLimitSource()


def resolve_order_command_transport(exchange: str) -> str:
    capabilities = get_exchange_capabilities(exchange).order_commands
    if capabilities.place_rest and capabilities.cancel_rest:
        return "rest"
    raise ValueError(f"{exchange} does not support complete REST order commands")


def supported_exchanges() -> tuple[str, ...]:
    """Exchanges with both capability metadata and adapter registration."""

    capability_names = set(EXCHANGE_CAPABILITIES)
    return tuple(name for name in registered_exchanges() if name in capability_names)
