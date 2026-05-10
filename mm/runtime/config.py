from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mm.core.models.instrument import SymbolBinding, build_instrument_spec

SourceRole = Literal["primary", "hedge"]
BusKind = Literal["memory", "redis_streams"]


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
class BusRuntimeConfig:
    """Runtime bus selection.

    ``memory`` preserves the current single-process behavior. ``redis_streams``
    adds durable fan-out for persistence, replay, admin, and UI daemons.
    """

    kind: BusKind = "memory"
    redis_url: str = "redis://localhost:6379/0"
    stream_prefix: str = "mm"
    consumer_group: str = "mm-runtime"
    consumer_name: str = "runtime"
    max_stream_length: int | None = 100_000
    read_block_ms: int = 1_000
    max_queue_size: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("memory", "redis_streams"):
            raise ValueError("bus.kind must be 'memory' or 'redis_streams'")
        if self.max_queue_size < 0:
            raise ValueError("bus.max_queue_size must be non-negative")
        if self.max_stream_length is not None and self.max_stream_length <= 0:
            raise ValueError("bus.max_stream_length must be positive or null")


@dataclass(frozen=True, slots=True)
class PersistenceRuntimeConfig:
    enabled: bool = False
    database_url: str | None = None
    database_url_env: str = "MM_DATABASE_URL"
    parquet_root: str | None = None
    parquet_root_env: str = "MM_PARQUET_ROOT"
    topics: tuple[str, ...] = ()
    batch_size: int = 100
    flush_interval_ms: int = 1000

    def resolved_database_url(self) -> str | None:
        if self.database_url:
            return self.database_url
        return os.environ.get(self.database_url_env, "").strip() or None

    def resolved_parquet_root(self) -> str | None:
        if self.parquet_root:
            return self.parquet_root
        return os.environ.get(self.parquet_root_env, "").strip() or None


@dataclass(frozen=True, slots=True)
class ObservabilityRuntimeConfig:
    service_name: str = "market-maker"
    run_id: str = ""
    metrics_enabled: bool = True
    metrics_port: int = 8000
    json_logs: bool = True
    log_path: str | None = "logs/mm-runtime.jsonl"
    otel_enabled: bool = False
    otlp_endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class OrderbookGridConfig:
    """Projection grid for MPC orderbook arrays.

    ``price_step`` and ``max_spread`` are quote-per-base price distances from mid.
    The model array index is only a bucket number; it is converted back to quote
    spread via ``spread = index * price_step`` inside MPC/strategy bridging.
    """

    price_step: float = 1.0
    max_spread: float = 100.0

    def __post_init__(self) -> None:
        if self.price_step <= 0:
            raise ValueError("orderbook_grid.price_step must be positive")
        if self.max_spread <= 0:
            raise ValueError("orderbook_grid.max_spread must be positive")

    @property
    def levels(self) -> int:
        import math

        return max(1, int(math.ceil(self.max_spread / self.price_step)))

    def as_contract(self) -> dict[str, Any]:
        return {
            "price_step": self.price_step,
            "max_spread": self.max_spread,
            "levels": self.levels,
            "index_policy": "model index i represents quote spread (i + 1) * price_step from mid",
            "qty_policy": "array values are base-asset quantities",
        }


@dataclass(frozen=True, slots=True)
class ControllerRuntimeConfig:
    """Config for the optional MPC/controller bridge.

    The controller consumes StrategyInputFrame objects and publishes controller
    diagnostics/actions. It is observe-only by default so live print_events runs
    never place or cancel orders unless config explicitly enables execution.

    ``action_start_after_steps`` mirrors ``OrderFlowRunner.action_start_after_steps``:
    MPC still runs every frame, but live ``OrderManager.sync`` is deferred until
    ``frame_step > action_start_after_steps``.
    """

    enabled: bool = False
    actor_id: str = "mpc-controller"
    observe_only: bool = True
    # Warm-up: skip live OrderManager.sync while frame_step <= this value (MPC still runs).
    action_start_after_steps: int = 0
    action_topic: str = "strategy:actions"
    diagnostics_topic: str = "strategy:controller"
    min_spread: float = 1.0
    max_spread: float = 100.0
    orderbook_grid: OrderbookGridConfig = field(default_factory=OrderbookGridConfig)
    qty_step: float = 1.0
    min_order_qty: float = 0.0
    max_orders_per_side: int = 5
    vwap_depth: float | None = None
    price_model: dict[str, Any] | None = None
    market_flow_model: dict[str, Any] | None = None
    hedge_price_model: dict[str, Any] | None = None
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
    bus: BusRuntimeConfig = field(default_factory=BusRuntimeConfig)
    persistence: PersistenceRuntimeConfig = field(default_factory=PersistenceRuntimeConfig)
    observability: ObservabilityRuntimeConfig = field(default_factory=ObservabilityRuntimeConfig)

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
            "bus": {
                "kind": self.bus.kind,
                "stream_prefix": self.bus.stream_prefix,
                "consumer_group": self.bus.consumer_group,
                "consumer_name": self.bus.consumer_name,
                "max_stream_length": self.bus.max_stream_length,
            },
            "persistence": {
                "enabled": self.persistence.enabled,
                "database_url_configured": bool(self.persistence.resolved_database_url()),
                "parquet_root_configured": bool(self.persistence.resolved_parquet_root()),
                "topics": self.persistence.topics,
                "batch_size": self.persistence.batch_size,
            },
            "observability": {
                "service_name": self.observability.service_name,
                "run_id": self.observability.run_id,
                "metrics_enabled": self.observability.metrics_enabled,
                "metrics_port": self.observability.metrics_port,
                "json_logs": self.observability.json_logs,
                "log_path": self.observability.log_path,
                "otel_enabled": self.observability.otel_enabled,
            },
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
        bus=_bus_from_mapping(payload.get("bus")),
        persistence=_persistence_from_mapping(payload.get("persistence")),
        observability=_observability_from_mapping(payload.get("observability")),
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


def _bus_from_mapping(payload: Any) -> BusRuntimeConfig:
    if not isinstance(payload, dict):
        return BusRuntimeConfig()
    max_stream_length = payload.get("max_stream_length", 100_000)
    return BusRuntimeConfig(
        kind=str(payload.get("kind", "memory")),  # type: ignore[arg-type]
        redis_url=str(payload.get("redis_url", "redis://localhost:6379/0")),
        stream_prefix=str(payload.get("stream_prefix", "mm")),
        consumer_group=str(payload.get("consumer_group", "mm-runtime")),
        consumer_name=str(payload.get("consumer_name", "runtime")),
        max_stream_length=int(max_stream_length) if max_stream_length is not None else None,
        read_block_ms=int(payload.get("read_block_ms", 1000)),
        max_queue_size=int(payload.get("max_queue_size", 0)),
    )


def _persistence_from_mapping(payload: Any) -> PersistenceRuntimeConfig:
    if not isinstance(payload, dict):
        return PersistenceRuntimeConfig(
            enabled=bool(os.environ.get("MM_DATABASE_URL", "").strip() or os.environ.get("MM_PARQUET_ROOT", "").strip())
        )
    topics = payload.get("topics", ())
    if isinstance(topics, str):
        topics_tuple = (topics,)
    elif isinstance(topics, (list, tuple)):
        topics_tuple = tuple(str(topic) for topic in topics)
    else:
        topics_tuple = ()
    return PersistenceRuntimeConfig(
        enabled=bool(payload.get("enabled", bool(payload.get("database_url") or payload.get("parquet_root")))),
        database_url=payload.get("database_url"),
        database_url_env=str(payload.get("database_url_env", "MM_DATABASE_URL")),
        parquet_root=payload.get("parquet_root"),
        parquet_root_env=str(payload.get("parquet_root_env", "MM_PARQUET_ROOT")),
        topics=topics_tuple,
        batch_size=max(1, int(payload.get("batch_size", 100))),
        flush_interval_ms=max(50, int(payload.get("flush_interval_ms", 1000))),
    )


def _observability_from_mapping(payload: Any) -> ObservabilityRuntimeConfig:
    if not isinstance(payload, dict):
        return ObservabilityRuntimeConfig()
    return ObservabilityRuntimeConfig(
        service_name=str(payload.get("service_name", "market-maker")),
        run_id=str(payload.get("run_id", "")),
        metrics_enabled=bool(payload.get("metrics_enabled", True)),
        metrics_port=int(payload.get("metrics_port", 8000)),
        json_logs=bool(payload.get("json_logs", True)),
        log_path=payload.get("log_path", "logs/mm-runtime.jsonl"),
        otel_enabled=bool(payload.get("otel_enabled", False)),
        otlp_endpoint=payload.get("otlp_endpoint"),
    )


def _controller_from_mapping(payload: Any) -> ControllerRuntimeConfig | None:
    if not isinstance(payload, dict):
        return None
    return ControllerRuntimeConfig(
        enabled=bool(payload.get("enabled", False)),
        actor_id=str(payload.get("actor_id", "mpc-controller")),
        observe_only=bool(payload.get("observe_only", True)),
        action_start_after_steps=max(0, int(payload.get("action_start_after_steps", 0))),
        action_topic=str(payload.get("action_topic", "strategy:actions")),
        diagnostics_topic=str(payload.get("diagnostics_topic", "strategy:controller")),
        min_spread=float(payload.get("min_spread", 1.0)),
        max_spread=float(payload.get("max_spread", 100.0)),
        orderbook_grid=_orderbook_grid_from_mapping(payload),
        qty_step=float(payload.get("qty_step", 1.0)),
        min_order_qty=float(payload.get("min_order_qty", 0.0)),
        max_orders_per_side=int(payload.get("max_orders_per_side", 5)),
        vwap_depth=(
            float(payload["vwap_depth"])
            if payload.get("vwap_depth") is not None
            else None
        ),
        price_model=_dict_or_none(payload.get("price_model")),
        market_flow_model=_dict_or_none(payload.get("market_flow_model")),
        hedge_price_model=_dict_or_none(payload.get("hedge_price_model")),
        execution_model=_dict_or_none(payload.get("execution_model")),
        optimizer=_dict_or_none(payload.get("optimizer")),
    )


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _orderbook_grid_from_mapping(payload: dict[str, Any]) -> OrderbookGridConfig:
    grid = payload.get("orderbook_grid")
    if isinstance(grid, dict):
        return OrderbookGridConfig(
            price_step=float(grid.get("price_step", 1.0)),
            max_spread=float(grid.get("max_spread", payload.get("max_spread", 100.0))),
        )
    return OrderbookGridConfig(
        price_step=1.0,
        max_spread=float(payload.get("max_spread", 100.0)),
    )


def _controller_contract(config: ControllerRuntimeConfig | None) -> dict[str, Any]:
    if config is None:
        return {"enabled": False}
    return {
        "enabled": config.enabled,
        "actor_id": config.actor_id,
        "observe_only": config.observe_only,
        "action_start_after_steps": config.action_start_after_steps,
        "diagnostics_topic": config.normalized_diagnostics_topic,
        "action_topic": config.normalized_action_topic,
        "orderbook_grid": config.orderbook_grid.as_contract(),
        "vwap_depth": config.vwap_depth,
        "vwap_depth_note": (
            "cumulative base quantity cap for reference VWAP in MPC and residual price model; "
            "overrides price_model.representative_price_target_depth_lots when set"
        ),
        "model_inputs": {
            "price_model": config.price_model or {"kind": "gaussian"},
            "market_flow_model": config.market_flow_model or {"kind": "hurdle_lognormal"},
            "hedge_price_model": config.hedge_price_model
            or {
                "kind": "legacy",
                "note": "EWMA Gaussian deltas on hedge refs; set optimizer.hedge_coef non-zero for rollouts",
            },
            "execution_model": config.execution_model or {"kind": "real"},
            "optimizer": config.optimizer or {"kind": "monte_carlo_mpc_tensor_batched"},
        },
        "spread_depth_semantics": (
            "optimizer.spread_depth is cumulative base-asset quantity for the spread penalty "
            "(see MPC _action_spread_at_vwap_qty); not exchange lot size. "
            "Legacy key spread_target_lots is still accepted as an alias."
        ),
        "optimizer_inventory_and_depth_targets": {
            "target_inventory_level": (
                "maps to MPC target_inv; desired inventory level for inv_coef penalty (base units)"
            ),
            "max_total_qty": (
                "maps to MPC total_q_target; target cumulative quoted depth over [0, spread_max] "
                "per side (base units); legacy JSON key total_q_target still accepted"
            ),
            "project_max_total_qty": (
                "maps to MPC project_total_q_target; when true with positive max_total_qty, "
                "rescales depth parameters toward that target; legacy key project_total_q_target still accepted"
            ),
        },
        "float_bridge": {
            "mid": "quote-per-base float",
            "relative_book_spread": "absolute quote-per-base distance from mid",
            "depth_qty": "base units",
            "cash": "quote units",
            "inventory": "base units",
        },
    }
