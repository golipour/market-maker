from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.exchanges.capabilities import PrivateStateMode
from mm.exchanges.shared.provider_adapter import TransportMode

MarketDataBuilder = Callable[..., PublicMarketDataAdapter]
PrivateStateBuilder = Callable[..., AbstractPrivateStateAdapter]
OrderCommandBuilder = Callable[..., AbstractOrderCommandAdapter]


@dataclass(frozen=True, slots=True)
class ExchangeRegistration:
    """Adapter builders for one supported exchange."""

    name: str
    build_market_data: MarketDataBuilder
    build_private_state: PrivateStateBuilder
    build_order_command: OrderCommandBuilder
    uses_trade_stats_limits: bool = False


_REGISTRY: dict[str, ExchangeRegistration] = {}


def register_exchange(registration: ExchangeRegistration) -> None:
    key = registration.name.lower().strip()
    if not key:
        raise ValueError("exchange registration requires a non-empty name")
    if key in _REGISTRY:
        raise ValueError(f"duplicate exchange registration: {key!r}")
    _REGISTRY[key] = registration


def get_exchange_registration(exchange: str) -> ExchangeRegistration:
    key = exchange.lower().strip()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unsupported exchange: {exchange!r} (supported: {supported})") from exc


def registered_exchanges() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _load_builtin_registrations() -> None:
    from mm.exchanges.bitpin.registration import REGISTRATION as bitpin
    from mm.exchanges.nobitex.registration import REGISTRATION as nobitex
    from mm.exchanges.tabdeal.registration import REGISTRATION as tabdeal
    from mm.exchanges.wallex.registration import REGISTRATION as wallex

    for registration in (nobitex, wallex, tabdeal, bitpin):
        register_exchange(registration)


_load_builtin_registrations()
