from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mm.core.models.instrument import SymbolBinding, build_instrument_spec

SourceRole = Literal["primary", "hedge"]


@dataclass(frozen=True, slots=True)
class MarketDataSourceConfig:
    """Public market data source for one strategy role."""

    role: SourceRole
    exchange: str
    symbol: str
    mode: str = "rest"
    depth_limit: int = 10
    trades_limit: int = 100
    poll_interval_ms: int = 1000

    def normalized_exchange(self) -> str:
        return self.exchange.lower().strip()

    def market_event_topic(self, interval_ms: int) -> str:
        return f"market:events:{self.normalized_exchange()}:{self.symbol}:{interval_ms}"

    def market_error_topic(self) -> str:
        return f"market:errors:{self.normalized_exchange()}:{self.symbol}"

    def raw_trade_topic(self) -> str:
        return f"market:raw:{self.normalized_exchange()}:{self.symbol}:trades"

    def raw_orderbook_topic(self) -> str:
        return f"market:raw:{self.normalized_exchange()}:{self.symbol}:orderbook"


@dataclass(frozen=True, slots=True)
class PrivateRuntimeConfig:
    """Private state and execution observation config for one exchange."""

    mode: str = "rest"
    api_token: str | None = None
    poll_interval_ms: int = 5000
    enable_execution: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.api_token)


@dataclass(frozen=True, slots=True)
class MarketMakerRuntimeConfig:
    """Config-driven wiring for primary plus optional hedge runtime.

    The hedge source is the only secondary source. It replaces the older
    anchor/hedge split and is used both as reference data and, when private
    credentials are configured, as an exchange that can expose private state.
    """

    primary: MarketDataSourceConfig
    hedge: MarketDataSourceConfig | None = None
    interval_ms: int = 5000
    max_hedge_lag_ms: int = 2000
    hedge_grace_ms: int = 250
    primary_private: PrivateRuntimeConfig | None = None
    hedge_private: PrivateRuntimeConfig | None = None

    def __post_init__(self) -> None:
        if self.primary.role != "primary":
            raise ValueError("primary source role must be 'primary'")
        if self.hedge is not None and self.hedge.role != "hedge":
            raise ValueError("hedge source role must be 'hedge'")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")

    @property
    def binding(self) -> SymbolBinding:
        return SymbolBinding(
            primary=build_instrument_spec(self.primary.normalized_exchange(), self.primary.symbol),
            hedge=(
                build_instrument_spec(self.hedge.normalized_exchange(), self.hedge.symbol)
                if self.hedge is not None
                else None
            ),
        )

    @property
    def frame_topic(self) -> str:
        return f"strategy:frames:{self.primary.normalized_exchange()}:{self.primary.symbol}:{self.interval_ms}"

    @property
    def sources(self) -> tuple[MarketDataSourceConfig, ...]:
        if self.hedge is None:
            return (self.primary,)
        return (self.primary, self.hedge)

    def private_config_for(self, role: SourceRole) -> PrivateRuntimeConfig | None:
        if role == "primary":
            return self.primary_private
        return self.hedge_private

    def as_contract(self) -> dict[str, Any]:
        return {
            "primary": _source_contract(self.primary, self.interval_ms),
            "hedge": _source_contract(self.hedge, self.interval_ms) if self.hedge else None,
            "strategy_input_frame": {
                "topic": self.frame_topic,
                "primary": "always present",
                "hedge": "None when no hedge source is configured, missing, or stale",
                "hedge_is_anchor": True,
                "max_hedge_lag_ms": self.max_hedge_lag_ms,
                "hedge_grace_ms": self.hedge_grace_ms,
            },
            "instrument_units": self.binding.as_contract(),
        }


def _source_contract(source: MarketDataSourceConfig | None, interval_ms: int) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "role": source.role,
        "exchange": source.normalized_exchange(),
        "symbol": source.symbol,
        "requested_public_market_transport": source.mode,
        "market_event_topic": source.market_event_topic(interval_ms),
        "market_error_topic": source.market_error_topic(),
        "raw_trade_topic": source.raw_trade_topic(),
        "raw_orderbook_topic": source.raw_orderbook_topic(),
    }
