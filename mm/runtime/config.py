from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
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
    api_token_env: str | None = None
    poll_interval_ms: int = 5000
    enable_execution: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.resolved_api_token())

    def resolved_api_token(self, exchange: str | None = None) -> str | None:
        if self.api_token:
            return self.api_token
        if self.api_token_env:
            return os.environ.get(self.api_token_env)
        if exchange:
            return os.environ.get(f"{exchange.upper()}_API_TOKEN")
        return None


@dataclass(frozen=True, slots=True)
class ControllerRuntimeConfig:
    """Config for the optional MPC/controller bridge.

    The controller consumes StrategyInputFrame objects and publishes controller
    diagnostics/actions. It is observe-only by default so live print_events runs
    never place or cancel orders unless config explicitly enables execution.
    """

    enabled: bool = False
    actor_id: str = "mpc-controller"
    observe_only: bool = True
    action_topic: str = "strategy:actions"
    diagnostics_topic: str = "strategy:controller"
    min_spread: float = 1.0
    max_spread: float = 100.0
    qty_step: float = 1.0
    min_order_qty: float = 0.0
    max_orders_per_side: int = 5
    price_model: dict[str, Any] | None = None
    market_flow_model: dict[str, Any] | None = None
    execution_model: dict[str, Any] | None = None
    optimizer: dict[str, Any] | None = None

    @property
    def normalized_action_topic(self) -> str:
        return self.action_topic.strip() or "strategy:actions"

    @property
    def normalized_diagnostics_topic(self) -> str:
        return self.diagnostics_topic.strip() or "strategy:controller"


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
    controller: ControllerRuntimeConfig | None = None

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
            "controller": (
                _controller_contract(self.controller)
                if self.controller is not None
                else {"enabled": False}
            ),
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


def load_runtime_config(path: str | Path) -> MarketMakerRuntimeConfig:
    """Load a full runtime config from JSON.

    JSON is used intentionally here because it has no additional runtime
    dependency and keeps deployment/configuration separate from CLI defaults.
    """

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("runtime config file must contain a JSON object")
    return runtime_config_from_mapping(payload)


def runtime_config_from_mapping(payload: dict[str, Any]) -> MarketMakerRuntimeConfig:
    primary_payload = payload.get("primary")
    if not isinstance(primary_payload, dict):
        raise ValueError("runtime config requires a 'primary' object")
    hedge_payload = payload.get("hedge")
    return MarketMakerRuntimeConfig(
        primary=_source_from_mapping(primary_payload, role="primary"),
        hedge=_source_from_mapping(hedge_payload, role="hedge") if isinstance(hedge_payload, dict) else None,
        interval_ms=int(payload.get("interval_ms", 5000)),
        max_hedge_lag_ms=int(payload.get("max_hedge_lag_ms", 2000)),
        hedge_grace_ms=int(payload.get("hedge_grace_ms", 250)),
        primary_private=_private_from_mapping(payload.get("primary_private")),
        hedge_private=_private_from_mapping(payload.get("hedge_private")),
        controller=_controller_from_mapping(payload.get("controller")),
    )


def _source_from_mapping(payload: dict[str, Any], *, role: SourceRole) -> MarketDataSourceConfig:
    return MarketDataSourceConfig(
        role=role,
        exchange=str(payload["exchange"]),
        symbol=str(payload["symbol"]),
        mode=str(payload.get("mode", "rest")),
        depth_limit=int(payload.get("depth_limit", 10)),
        trades_limit=int(payload.get("trades_limit", 100)),
        poll_interval_ms=int(payload.get("poll_interval_ms", 1000)),
    )


def _private_from_mapping(payload: Any) -> PrivateRuntimeConfig | None:
    if not isinstance(payload, dict):
        return None
    return PrivateRuntimeConfig(
        mode=str(payload.get("mode", "rest")),
        api_token=payload.get("api_token"),
        api_token_env=payload.get("api_token_env"),
        poll_interval_ms=int(payload.get("poll_interval_ms", 5000)),
        enable_execution=bool(payload.get("enable_execution", True)),
    )


def _controller_from_mapping(payload: Any) -> ControllerRuntimeConfig | None:
    if not isinstance(payload, dict):
        return None
    return ControllerRuntimeConfig(
        enabled=bool(payload.get("enabled", False)),
        actor_id=str(payload.get("actor_id", "mpc-controller")),
        observe_only=bool(payload.get("observe_only", True)),
        action_topic=str(payload.get("action_topic", "strategy:actions")),
        diagnostics_topic=str(payload.get("diagnostics_topic", "strategy:controller")),
        min_spread=float(payload.get("min_spread", 1.0)),
        max_spread=float(payload.get("max_spread", 100.0)),
        qty_step=float(payload.get("qty_step", 1.0)),
        min_order_qty=float(payload.get("min_order_qty", 0.0)),
        max_orders_per_side=int(payload.get("max_orders_per_side", 5)),
        price_model=_dict_or_none(payload.get("price_model")),
        market_flow_model=_dict_or_none(payload.get("market_flow_model")),
        execution_model=_dict_or_none(payload.get("execution_model")),
        optimizer=_dict_or_none(payload.get("optimizer")),
    )


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _controller_contract(config: ControllerRuntimeConfig | None) -> dict[str, Any]:
    if config is None:
        return {"enabled": False}
    return {
        "enabled": config.enabled,
        "actor_id": config.actor_id,
        "observe_only": config.observe_only,
        "diagnostics_topic": config.normalized_diagnostics_topic,
        "action_topic": config.normalized_action_topic,
        "model_inputs": {
            "price_model": config.price_model or {"kind": "gaussian"},
            "market_flow_model": config.market_flow_model or {"kind": "hurdle_lognormal"},
            "execution_model": config.execution_model or {"kind": "observed_step"},
            "optimizer": config.optimizer or {"kind": "monte_carlo_mpc_tensor_batched"},
        },
        "float_bridge": {
            "mid": "quote-per-base float",
            "relative_book_spread": "absolute quote-per-base distance from mid",
            "depth_qty": "base units",
            "cash": "quote units",
            "inventory": "base units",
        },
    }
