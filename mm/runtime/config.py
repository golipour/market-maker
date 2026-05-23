from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mm.core.models.instrument import (
    InstrumentPrecisionConfig,
    SymbolBinding,
    build_instrument_spec,
    instrument_precision_from_mapping,
    merge_instrument_spec,
)
from mm.core.risk_limits import HedgeVenueRiskLimits, VenueRiskLimits

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
    instrument: InstrumentPrecisionConfig | None = None

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
    """Spread grid for relative orderbook projection and MPC optimization.

    Quote-per-base distances from mid are bucketed from 0 to ``spread_max`` in steps
    of ``spread_step``. Bucket index ``i`` covers spreads in
    ``((i) * spread_step, (i + 1) * spread_step]``; cumulative depth is evaluated at
    the upper edge ``(i + 1) * spread_step``.
    """

    spread_max: float = 100.0
    spread_step: float = 1.0

    def __post_init__(self) -> None:
        if self.spread_step <= 0:
            raise ValueError("orderbook_grid.spread_step must be positive")
        if self.spread_max <= 0:
            raise ValueError("orderbook_grid.spread_max must be positive")

    @property
    def levels(self) -> int:
        import math

        return max(1, int(math.ceil(self.spread_max / self.spread_step)))

    def spread_ladder(self) -> tuple[float, ...]:
        """Discrete quote spreads for limit orders: step, 2*step, …, up to spread_max."""
        return tuple(self.spread_step * float(i + 1) for i in range(self.levels))

    def as_contract(self) -> dict[str, Any]:
        return {
            "spread_max": self.spread_max,
            "spread_step": self.spread_step,
            "levels": self.levels,
            "spread_ladder": list(self.spread_ladder()),
            "index_policy": (
                "bucket i is cumulative depth through quote spread (i + 1) * spread_step from mid"
            ),
            "qty_policy": "array values are base-asset quantities",
        }


@dataclass(frozen=True, slots=True)
class HedgingRuntimeConfig:
    """Live fill-driven hedge execution (separate from MPC ``hedge_rollout_enabled``)."""

    enabled: bool = False
    actor_id: str = "hedge-executor"
    observe_only: bool = True
    intents_topic: str | None = None
    diagnostics_topic: str | None = None
    dedupe_window_ms: int = 300_000

    @property
    def normalized_intents_topic(self) -> str:
        explicit = (self.intents_topic or "").strip()
        if explicit:
            return explicit
        actor = self.actor_id.strip() or "hedge-executor"
        return f"hedge:intents:{actor}"

    @property
    def normalized_diagnostics_topic(self) -> str:
        explicit = (self.diagnostics_topic or "").strip()
        if explicit:
            return explicit
        actor = self.actor_id.strip() or "hedge-executor"
        return f"hedge:diagnostics:{actor}"


@dataclass(frozen=True, slots=True)
class RiskRuntimeConfig:
    """Hard pre-execution risk gates (separate from optimizer soft preferences)."""

    enabled: bool = True
    primary: VenueRiskLimits = field(default_factory=VenueRiskLimits)
    hedge: HedgeVenueRiskLimits = field(default_factory=HedgeVenueRiskLimits)

    @property
    def max_inventory(self) -> float:
        """Alias for legacy flat config and diagnostics."""
        return self.primary.max_inventory

    @property
    def max_data_age_ms(self) -> int:
        """Alias for legacy flat config."""
        return self.primary.max_market_data_age_ms


@dataclass(frozen=True, slots=True)
class ReconcileRuntimeConfig:
    """Tolerance when matching declarative target orders to live open orders.

    ``price_tol_bps``: max relative price gap in basis points (0 = exact match after quantization).
    ``qty_tol_rel``: max relative quantity gap vs live ``remaining_qty`` (0 = exact).
    """

    price_tol_bps: float = 0.0
    qty_tol_rel: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "price_tol_bps": float(self.price_tol_bps),
            "qty_tol_rel": float(self.qty_tol_rel),
        }


@dataclass(frozen=True, slots=True)
class ControllerRuntimeConfig:
    """Config for the optional MPC controller on the event bus.

    Consumes ``StrategyInputFrame`` and publishes ``StrategyDecision`` payloads and ``Action``.
    Observe-only by default: enable execution in config when placing live orders is intended.

    ``action_start_after_steps``: MPC runs every frame, but ``OrderManager.sync`` waits until
    ``frame_step > action_start_after_steps``.
    """

    enabled: bool = False
    actor_id: str = "mpc-controller"
    observe_only: bool = True
    # Warm-up: skip live OrderManager.sync while frame_step <= this value (MPC still runs).
    action_start_after_steps: int = 0
    # Max wall time for live order sync per frame; keeps diagnostics cadence within interval_ms.
    placement_timeout_ms: int = 3000
    # Max concurrent REST place/cancel calls per exchange adapter batch.
    placement_max_concurrency: int = 8
    action_topic: str = "strategy:actions"
    decisions_topic: str = "strategy:decisions"
    diagnostics_topic: str | None = None
    orderbook_grid: OrderbookGridConfig = field(default_factory=OrderbookGridConfig)
    max_orders_per_side: int = 5
    vwap_depth: float | None = None
    price_model: dict[str, Any] | None = None
    market_flow_model: dict[str, Any] | None = None
    hedge_price_model: dict[str, Any] | None = None
    execution_model: dict[str, Any] | None = None
    optimizer: dict[str, Any] | None = None
    risk: RiskRuntimeConfig = field(default_factory=RiskRuntimeConfig)
    reconcile: ReconcileRuntimeConfig = field(default_factory=ReconcileRuntimeConfig)

    @property
    def normalized_action_topic(self) -> str:
        return self.action_topic.strip() or "strategy:actions"

    @property
    def normalized_decisions_topic(self) -> str:
        return self.decisions_topic.strip() or "strategy:decisions"

    @property
    def normalized_diagnostics_topic(self) -> str:
        explicit = (self.diagnostics_topic or "").strip()
        if explicit:
            return explicit
        actor = self.actor_id.strip() or "mpc-controller"
        return f"strategy:diagnostics:{actor}"


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
    hedging: HedgingRuntimeConfig | None = None
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
            primary=_instrument_spec_for_source(self.primary),
            hedge=_instrument_spec_for_source(self.hedge) if self.hedge is not None else None,
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
        hedging=_hedging_from_mapping(payload.get("hedging")),
        bus=_bus_from_mapping(payload.get("bus")),
        persistence=_persistence_from_mapping(payload.get("persistence")),
        observability=_observability_from_mapping(payload.get("observability")),
    )


def _instrument_spec_for_source(source: MarketDataSourceConfig):
    from mm.exchanges.shared.instrument_spec_trade_stats import instrument_spec_from_trade_stats

    try:
        venue = instrument_spec_from_trade_stats(source.normalized_exchange(), source.symbol)
    except Exception:
        venue = None
    return merge_instrument_spec(
        source.normalized_exchange(),
        source.symbol,
        venue=venue,
        overrides=source.instrument,
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
        instrument=instrument_precision_from_mapping(payload.get("instrument")),
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
        placement_timeout_ms=max(0, int(payload.get("placement_timeout_ms", 3000))),
        placement_max_concurrency=max(1, int(payload.get("placement_max_concurrency", 8))),
        action_topic=str(payload.get("action_topic", "strategy:actions")),
        decisions_topic=str(payload.get("decisions_topic", "strategy:decisions")),
        diagnostics_topic=(
            str(payload["diagnostics_topic"]) if payload.get("diagnostics_topic") is not None else None
        ),
        orderbook_grid=_orderbook_grid_from_mapping(payload),
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
        risk=_risk_from_mapping(payload.get("risk")),
        reconcile=_reconcile_from_mapping(payload.get("reconcile")),
    )


def _reconcile_from_mapping(payload: Any) -> ReconcileRuntimeConfig:
    if not isinstance(payload, dict):
        return ReconcileRuntimeConfig()
    return ReconcileRuntimeConfig(
        price_tol_bps=max(0.0, float(payload.get("price_tol_bps", 0.0))),
        qty_tol_rel=max(0.0, float(payload.get("qty_tol_rel", 0.0))),
    )


def _hedging_from_mapping(payload: Any) -> HedgingRuntimeConfig | None:
    if not isinstance(payload, dict):
        return None
    return HedgingRuntimeConfig(
        enabled=bool(payload.get("enabled", False)),
        actor_id=str(payload.get("actor_id", "hedge-executor")),
        observe_only=bool(payload.get("observe_only", True)),
        intents_topic=(
            str(payload["intents_topic"]) if payload.get("intents_topic") is not None else None
        ),
        diagnostics_topic=(
            str(payload["diagnostics_topic"]) if payload.get("diagnostics_topic") is not None else None
        ),
        dedupe_window_ms=max(1000, int(payload.get("dedupe_window_ms", 300_000))),
    )


def _risk_from_mapping(payload: Any) -> RiskRuntimeConfig:
    if not isinstance(payload, dict):
        return RiskRuntimeConfig()
    enabled = bool(payload.get("enabled", True))
    primary_payload = payload.get("primary")
    hedge_payload = payload.get("hedge")
    if isinstance(primary_payload, dict):
        primary = _venue_risk_from_mapping(primary_payload)
    else:
        primary = _venue_risk_from_mapping(payload, legacy_flat=True)
    hedge = _hedge_risk_from_mapping(hedge_payload) if isinstance(hedge_payload, dict) else HedgeVenueRiskLimits()
    return RiskRuntimeConfig(enabled=enabled, primary=primary, hedge=hedge)


def _venue_risk_from_mapping(payload: dict[str, Any], *, legacy_flat: bool = False) -> VenueRiskLimits:
    """Parse primary venue limits; ``legacy_flat`` reads top-level migration aliases."""

    def _get(key: str, default: Any) -> Any:
        if legacy_flat and key in payload:
            return payload[key]
        return payload.get(key, default)

    max_inv = float(_get("max_inventory", float("inf")))
    min_inv = float(_get("min_inventory", 0.0))
    if min_inv < 0.0:
        min_inv = 0.0
    return VenueRiskLimits(
        min_inventory=min_inv,
        max_inventory=max_inv,
        min_cash=float(_get("min_cash", 0.0)),
        max_open_notional=float(_get("max_open_notional", float("inf"))),
        max_order_notional=float(_get("max_order_notional", float("inf"))),
        max_market_data_age_ms=int(_get("max_market_data_age_ms", _get("max_data_age_ms", 10_000))),
        max_private_state_age_ms=int(_get("max_private_state_age_ms", 60_000)),
    )


def _hedge_risk_from_mapping(payload: dict[str, Any] | None) -> HedgeVenueRiskLimits:
    if not isinstance(payload, dict):
        return HedgeVenueRiskLimits()
    min_inv = float(payload.get("min_inventory", 0.0))
    if min_inv < 0.0:
        min_inv = 0.0
    return HedgeVenueRiskLimits(
        min_inventory=min_inv,
        max_inventory=float(payload.get("max_inventory", float("inf"))),
        min_cash=float(payload.get("min_cash", 0.0)),
        max_market_order_notional=float(payload.get("max_market_order_notional", float("inf"))),
        max_slippage_bps=float(payload.get("max_slippage_bps", float("inf"))),
        max_market_data_age_ms=int(payload.get("max_market_data_age_ms", payload.get("max_data_age_ms", 10_000))),
        max_private_state_age_ms=int(payload.get("max_private_state_age_ms", 60_000)),
    )


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _orderbook_grid_from_mapping(payload: dict[str, Any]) -> OrderbookGridConfig:
    grid = payload.get("orderbook_grid")
    if grid is None:
        return OrderbookGridConfig()
    if not isinstance(grid, dict):
        raise ValueError("orderbook_grid must be an object")
    missing = [k for k in ("spread_max", "spread_step") if k not in grid]
    if missing:
        raise ValueError(f"orderbook_grid missing required keys: {', '.join(missing)}")
    return OrderbookGridConfig(
        spread_max=float(grid["spread_max"]),
        spread_step=float(grid["spread_step"]),
    )


def _controller_contract(config: ControllerRuntimeConfig | None) -> dict[str, Any]:
    if config is None:
        return {"enabled": False}
    return {
        "enabled": config.enabled,
        "actor_id": config.actor_id,
        "observe_only": config.observe_only,
        "action_start_after_steps": config.action_start_after_steps,
        "placement_timeout_ms": config.placement_timeout_ms,
        "placement_max_concurrency": config.placement_max_concurrency,
        "decisions_topic": config.normalized_decisions_topic,
        "action_topic": config.normalized_action_topic,
        "orderbook_grid": config.orderbook_grid.as_contract(),
        "max_orders_per_side": config.max_orders_per_side,
        "vwap_depth": config.vwap_depth,
        "vwap_depth_note": (
            "cumulative base quantity cap for reference VWAP in MPC and residual price model; "
            "overrides price_model.default_vwap_depth_lots when set"
        ),
        "model_inputs": {
            "price_model": config.price_model or {"kind": "gaussian"},
            "market_flow_model": config.market_flow_model or {"kind": "hurdle_lognormal"},
            "hedge_price_model": config.hedge_price_model
            or {
                "kind": "legacy",
                "note": "EWMA Gaussian deltas on hedge refs; set optimizer.hedge_rollout_enabled for rollouts",
            },
            "execution_model": config.execution_model or {"kind": "real"},
            "optimizer": config.optimizer or {"kind": "monte_carlo_mpc_tensor_batched"},
        },
        "spread_depth_semantics": (
            "quote_size_min_per_side and quote_size_max_per_side bound cumulative per-side "
            "depth at spread_max; default inventory target uses their midpoint when "
            "target_inventory is unset"
        ),
        "risk": {
            "enabled": config.risk.enabled,
            "primary": config.risk.primary.as_dict(),
            "hedge": config.risk.hedge.as_dict(),
            "note": "hard gates only; inventory_target_penalty_weight lives under optimizer",
        },
        "reconcile": {
            **config.reconcile.as_dict(),
            "note": (
                "OrderManager target-state matching: keep live orders within tolerance, "
                "cancel orphans, place unmatched targets"
            ),
        },
        "optimizer_objective": {
            "return_risk_aversion_beta": (
                "risk aversion beta in (-1/beta)*ln(mean(exp(-beta*R_i))) over MC path returns"
            ),
            "inventory_target_penalty_weight": (
                "lambda_inv weight on normalized quadratic inventory capacity penalty"
            ),
            "target_inventory": (
                "optional explicit inventory target in base units; "
                "defaults to target_inventory_multiplier * midpoint(quote_size_min, quote_size_max)"
            ),
            "target_inventory_multiplier": (
                "multiplier applied to quote-size midpoint when target_inventory is unset"
            ),
            "quote_size_min_per_side": (
                "minimum cumulative quote depth per side at spread_max (base units)"
            ),
            "quote_size_max_per_side": (
                "maximum cumulative quote depth per side at spread_max (base units)"
            ),
            "quote_spread_max": (
                "maximum allowed sum of first bid/ask spread centers (quote units)"
            ),
            "quote_spread_min_depth": (
                "minimum executable depth at each side's first spread center (base units)"
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
