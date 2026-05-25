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
from mm.core.risk_limits import GlobalVenueRiskLimits, VenueRiskLimits

SourceRole = Literal["local", "global"]
NeutralVenueRole = Literal["local", "global"]
LiquiditySourcingMode = Literal["single_venue", "liquidity_sourcing"]
ReferenceVenueRole = Literal["global", "local"]
BusKind = Literal["memory", "redis_streams"]


def _normalize_reference_price_source(value: Any) -> ReferenceVenueRole:
    source = str(value or "global").strip().lower()
    if source not in ("global", "local"):
        raise ValueError("controller.reference_price_source must be 'global' or 'local'")
    return source  # type: ignore[return-value]


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
class LiquiditySourcingRuntimeConfig:
    """Optional cross-venue liquidity sourcing (disabled by default)."""

    enabled: bool = False
    mode: LiquiditySourcingMode = "single_venue"
    reference_venue: ReferenceVenueRole = "global"
    require_global_private: bool = False
    global_live_orders: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "reference_venue": self.reference_venue,
            "require_global_private": self.require_global_private,
            "global_live_orders": self.global_live_orders,
        }


ExecutionPlannerKind = Literal["reconcile", "replace"]


@dataclass(frozen=True, slots=True)
class VenueExecutionConfig:
    """Per-venue execution policy and source-venue placement caps."""

    planner: ExecutionPlannerKind = "reconcile"
    observe_only: bool = False
    max_order_qty: float | None = None
    max_quote_distance: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "planner": self.planner,
            "observe_only": self.observe_only,
        }
        if self.max_order_qty is not None:
            payload["max_order_qty"] = float(self.max_order_qty)
        if self.max_quote_distance is not None:
            payload["max_quote_distance"] = float(self.max_quote_distance)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionRuntimeConfig:
    """Per-venue execution planner selection and live/observe overrides."""

    local: VenueExecutionConfig = field(default_factory=VenueExecutionConfig)
    global_: VenueExecutionConfig = field(
        default_factory=lambda: VenueExecutionConfig(planner="replace")
    )

    def for_role(self, role: NeutralVenueRole) -> VenueExecutionConfig:
        return self.local if role == "local" else self.global_

    def as_dict(self) -> dict[str, Any]:
        return {
            "local": self.local.as_dict(),
            "global": self.global_.as_dict(),
        }




RiskEngineKind = Literal["net_delta"]


@dataclass(frozen=True, slots=True)
class RiskRuntimeConfig:
    """Hard pre-execution risk gates (separate from optimizer soft preferences)."""

    enabled: bool = True
    engine: RiskEngineKind = "net_delta"
    local: VenueRiskLimits = field(default_factory=VenueRiskLimits)
    global_: GlobalVenueRiskLimits = field(default_factory=GlobalVenueRiskLimits)


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
    target_books_topic: str | None = None
    orderbook_grid: OrderbookGridConfig = field(default_factory=OrderbookGridConfig)
    max_orders_per_side: int = 5
    reference_price_source: ReferenceVenueRole = "global"
    local_price_model: dict[str, Any] | None = None
    local_market_flow_model: dict[str, Any] | None = None
    global_market_flow_model: dict[str, Any] | None = None
    global_price_model: dict[str, Any] | None = None
    execution_model: dict[str, Any] | None = None
    optimizer: dict[str, Any] | None = None
    risk: RiskRuntimeConfig = field(default_factory=RiskRuntimeConfig)
    reconcile: ReconcileRuntimeConfig = field(default_factory=ReconcileRuntimeConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_price_source",
            _normalize_reference_price_source(self.reference_price_source),
        )

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

    @property
    def normalized_target_books_topic(self) -> str:
        explicit = (self.target_books_topic or "").strip()
        if explicit:
            return explicit
        actor = self.actor_id.strip() or "mpc-controller"
        return f"strategy:target_books:{actor}"


@dataclass(frozen=True, slots=True)
class MarketMakerRuntimeConfig:
    """Config-driven wiring for local plus optional global runtime venues."""

    local: MarketDataSourceConfig
    global_: MarketDataSourceConfig | None = None
    interval_ms: int = 5000
    max_global_lag_ms: int = 2000
    global_grace_ms: int = 250
    local_private: PrivateRuntimeConfig | None = None
    global_private: PrivateRuntimeConfig | None = None
    controller: ControllerRuntimeConfig | None = None
    liquidity_sourcing: LiquiditySourcingRuntimeConfig = field(
        default_factory=LiquiditySourcingRuntimeConfig
    )
    execution: ExecutionRuntimeConfig = field(default_factory=ExecutionRuntimeConfig)
    bus: BusRuntimeConfig = field(default_factory=BusRuntimeConfig)
    persistence: PersistenceRuntimeConfig = field(default_factory=PersistenceRuntimeConfig)
    observability: ObservabilityRuntimeConfig = field(default_factory=ObservabilityRuntimeConfig)

    def __post_init__(self) -> None:
        if self.local.role != "local":
            raise ValueError("local source role must be 'local'")
        if self.global_ is not None and self.global_.role != "global":
            raise ValueError("global source role must be 'global'")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")

    @property
    def binding(self) -> SymbolBinding:
        return SymbolBinding(
            local=_instrument_spec_for_source(self.local),
            global_=_instrument_spec_for_source(self.global_) if self.global_ is not None else None,
        )

    @property
    def frame_topic(self) -> str:
        return f"strategy:frames:{self.local.normalized_exchange()}:{self.local.symbol}:{self.interval_ms}"

    @property
    def joint_frame_topic(self) -> str:
        return f"strategy:joint:frames:{self.local.normalized_exchange()}:{self.local.symbol}:{self.interval_ms}"

    @property
    def resolved_persistence_topics(self) -> tuple[str, ...]:
        """Configured persistence topics with runtime placeholders expanded."""

        if not self.persistence.topics:
            return ()
        context: dict[str, str | int] = {
            "interval_ms": self.interval_ms,
            "local_exchange": self.local.normalized_exchange(),
            "local_symbol": self.local.symbol,
            "local_market_event_topic": self.local.market_event_topic(self.interval_ms),
            "local_execution_orders_topic": f"execution:orders:{self.local.normalized_exchange()}",
            "local_execution_state_topic": f"execution:state:{self.local.normalized_exchange()}",
            "local_private_snapshot_topic": f"private:state:{self.local.normalized_exchange()}:snapshot",
            "local_private_fills_topic": f"private:state:{self.local.normalized_exchange()}:fills",
            "ledger_topic": "ledger:snapshot",
            "frame_topic": self.frame_topic,
            "joint_frame_topic": self.joint_frame_topic,
        }
        if self.global_ is not None:
            context.update(
                {
                    "global_exchange": self.global_.normalized_exchange(),
                    "global_symbol": self.global_.symbol,
                    "global_market_event_topic": self.global_.market_event_topic(self.interval_ms),
                    "global_execution_orders_topic": f"execution:orders:{self.global_.normalized_exchange()}",
                    "global_execution_state_topic": f"execution:state:{self.global_.normalized_exchange()}",
                    "global_private_snapshot_topic": f"private:state:{self.global_.normalized_exchange()}:snapshot",
                    "global_private_fills_topic": f"private:state:{self.global_.normalized_exchange()}:fills",
                }
            )
        if self.controller is not None:
            context.update(
                {
                    "strategy_actions_topic": self.controller.normalized_action_topic,
                    "strategy_decisions_topic": self.controller.normalized_decisions_topic,
                    "strategy_diagnostics_topic": self.controller.normalized_diagnostics_topic,
                    "strategy_target_books_topic": self.controller.normalized_target_books_topic,
                }
            )
        resolved: list[str] = []
        for topic in self.persistence.topics:
            try:
                expanded = topic.format_map(context)
            except KeyError as exc:
                key = str(exc).strip("'")
                raise ValueError(f"unknown persistence topic placeholder: {key}") from exc
            expanded = expanded.strip()
            if expanded:
                resolved.append(expanded)
        return tuple(dict.fromkeys(resolved))

    @property
    def default_persistence_topics(self) -> tuple[str, ...]:
        """Default durable event topics derived from the runtime wiring."""

        local_exchange = self.local.normalized_exchange()
        topics = [
            self.local.market_event_topic(self.interval_ms),
            self.joint_frame_topic,
            "ledger:snapshot",
            self.controller.normalized_action_topic if self.controller is not None else "strategy:actions",
            f"execution:orders:{local_exchange}",
            f"execution:state:{local_exchange}",
            f"private:state:{local_exchange}:snapshot",
        ]
        if self.controller is not None:
            topics.append(self.controller.normalized_decisions_topic)
            topics.append(self.controller.normalized_diagnostics_topic)
            topics.append(self.controller.normalized_target_books_topic)
        if self.global_ is not None:
            global_exchange = self.global_.normalized_exchange()
            topics.extend(
                [
                    self.global_.market_event_topic(self.interval_ms),
                    f"execution:orders:{global_exchange}",
                    f"execution:state:{global_exchange}",
                    f"private:state:{global_exchange}:snapshot",
                    f"private:state:{local_exchange}:fills",
                ]
            )
        return tuple(dict.fromkeys(topics))

    @property
    def effective_persistence_topics(self) -> tuple[str, ...]:
        """Topics used by persistence writers: configured override or runtime defaults."""

        return self.resolved_persistence_topics or self.default_persistence_topics

    @property
    def sources(self) -> tuple[MarketDataSourceConfig, ...]:
        if self.global_ is None:
            return (self.local,)
        return (self.local, self.global_)

    def private_config_for(self, role: SourceRole) -> PrivateRuntimeConfig | None:
        if role == "local":
            return self.local_private
        return self.global_private

    def as_contract(self) -> dict[str, Any]:
        return {
            "venues": {
                "local": _source_contract(self.local, self.interval_ms),
                "global": _source_contract(self.global_, self.interval_ms) if self.global_ else None,
            },
            "local_private": _private_contract(self.local_private),
            "global_private": _private_contract(self.global_private),
            "interval_ms": self.interval_ms,
            "max_global_lag_ms": self.max_global_lag_ms,
            "global_grace_ms": self.global_grace_ms,
            "liquidity_sourcing": self.liquidity_sourcing.as_dict(),
            "execution": self.execution.as_dict(),
            "joint_strategy_frame": {
                "topic": self.joint_frame_topic,
                "ledger_topic": "ledger:snapshot",
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
                "resolved_topics": self.resolved_persistence_topics,
                "effective_topics": self.effective_persistence_topics,
                "uses_default_topics": not bool(self.persistence.topics),
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


def _private_contract(private: PrivateRuntimeConfig | None) -> dict[str, Any] | None:
    if private is None:
        return None
    return {
        "mode": private.mode,
        "poll_interval_ms": private.poll_interval_ms,
        "enable_execution": private.enable_execution,
        "enabled": private.enabled,
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
    local_payload, global_payload = _resolve_public_venue_payloads(payload)
    local_private_payload, global_private_payload = _resolve_private_venue_payloads(payload)
    liquidity_sourcing = _liquidity_sourcing_from_mapping(payload.get("liquidity_sourcing"))
    return MarketMakerRuntimeConfig(
        local=_source_from_mapping(local_payload, role="local"),
        global_=_source_from_mapping(global_payload, role="global") if isinstance(global_payload, dict) else None,
        interval_ms=int(payload.get("interval_ms", 5000)),
        max_global_lag_ms=int(payload.get("max_global_lag_ms", 2000)),
        global_grace_ms=int(payload.get("global_grace_ms", 250)),
        local_private=_private_from_mapping(local_private_payload),
        global_private=_private_from_mapping(global_private_payload),
        controller=_controller_from_mapping(payload.get("controller")),
        liquidity_sourcing=liquidity_sourcing,
        execution=_execution_from_mapping(payload.get("execution")),
        bus=_bus_from_mapping(payload.get("bus")),
        persistence=_persistence_from_mapping(payload.get("persistence")),
        observability=_observability_from_mapping(payload.get("observability")),
    )


def _resolve_public_venue_payloads(payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    venues = payload.get("venues")
    if not isinstance(venues, dict):
        raise ValueError("runtime config requires venues object with local venue")
    local_payload = _venue_payload_from_mapping(venues.get("local"))
    global_payload = _venue_payload_from_mapping(venues.get("global"))
    if local_payload is None:
        raise ValueError("runtime config requires venues.local")
    return local_payload, global_payload


def _resolve_private_venue_payloads(payload: dict[str, Any]) -> tuple[Any, Any]:
    return payload.get("local_private"), payload.get("global_private")


def _venue_payload_from_mapping(payload: Any) -> dict[str, Any] | None:
    return dict(payload) if isinstance(payload, dict) else None


def _liquidity_sourcing_from_mapping(payload: Any) -> LiquiditySourcingRuntimeConfig:
    if not isinstance(payload, dict):
        return LiquiditySourcingRuntimeConfig()
    mode = str(payload.get("mode", "single_venue"))
    if mode not in ("single_venue", "liquidity_sourcing"):
        raise ValueError("liquidity_sourcing.mode must be 'single_venue' or 'liquidity_sourcing'")
    reference_venue = str(payload.get("reference_venue", "global"))
    if reference_venue not in ("global", "local"):
        raise ValueError("liquidity_sourcing.reference_venue must be 'global' or 'local'")
    return LiquiditySourcingRuntimeConfig(
        enabled=bool(payload.get("enabled", False)),
        mode=mode,  # type: ignore[arg-type]
        reference_venue=reference_venue,  # type: ignore[arg-type]
        require_global_private=bool(payload.get("require_global_private", False)),
        global_live_orders=bool(payload.get("global_live_orders", False)),
    )


def _venue_execution_from_mapping(payload: Any, *, default_planner: ExecutionPlannerKind) -> VenueExecutionConfig:
    if not isinstance(payload, dict):
        return VenueExecutionConfig(planner=default_planner)
    planner = str(payload.get("planner", default_planner)).strip().lower()
    if planner not in ("reconcile", "replace"):
        raise ValueError("execution planner must be 'reconcile' or 'replace'")
    max_order_qty = payload.get("max_order_qty")
    max_quote_distance = payload.get("max_quote_distance")
    return VenueExecutionConfig(
        planner=planner,  # type: ignore[arg-type]
        observe_only=bool(payload.get("observe_only", False)),
        max_order_qty=float(max_order_qty) if max_order_qty is not None else None,
        max_quote_distance=float(max_quote_distance) if max_quote_distance is not None else None,
    )


def _execution_from_mapping(payload: Any) -> ExecutionRuntimeConfig:
    if not isinstance(payload, dict):
        return ExecutionRuntimeConfig()
    return ExecutionRuntimeConfig(
        local=_venue_execution_from_mapping(payload.get("local"), default_planner="reconcile"),
        global_=_venue_execution_from_mapping(payload.get("global"), default_planner="replace"),
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
    removed_model_keys = [key for key in ("price_model", "market_flow_model") if key in payload]
    if removed_model_keys:
        raise ValueError(
            "controller model config fields were renamed: use local_price_model and "
            f"local_market_flow_model instead of {', '.join(removed_model_keys)}"
        )
    if "vwap_depth" in payload:
        raise ValueError(
            "controller.vwap_depth was removed; VWAP depth is derived from inventory"
        )
    optimizer_payload = payload.get("optimizer")
    if isinstance(optimizer_payload, dict):
        venue_optimizer_keys = {
            "return_risk_aversion_beta",
            "objective_weight",
            "inventory_target_penalty_weight",
            "target_inventory",
            "target_inventory_multiplier",
            "quote_size_min_per_side",
            "quote_size_max_per_side",
            "quote_spread_max",
            "quote_spread_min_depth",
            "quote_spread_k_epsilon",
            "soft_fill_temp",
            "soft_fill_activation",
        }
        removed_venue_optimizer_aliases = {
            "inv_coef": "inventory_target_penalty_weight",
            "target_inv": "target_inventory",
        }
        removed_optimizer_keys = [
            key for key in ("global_market_flow_model", "global_price_model") if key in optimizer_payload
        ]
        if removed_optimizer_keys:
            raise ValueError(
                "global model config belongs directly under controller: "
                f"{', '.join(removed_optimizer_keys)}"
            )
        removed_flat_optimizer_keys = [
            key
            for key in (
                "return_risk_aversion_beta",
                "objective_weight",
                "inventory_target_penalty_weight",
                "inv_coef",
                "target_inventory",
                "target_inv",
                "target_inventory_multiplier",
                "quote_size_min_per_side",
                "quote_size_max_per_side",
                "quote_spread_max",
                "quote_spread_min_depth",
                "quote_spread_k_epsilon",
                "soft_fill_temp",
                "soft_fill_activation",
            )
            if key in optimizer_payload
        ]
        if removed_flat_optimizer_keys:
            raise ValueError(
                "venue optimizer config belongs under controller.optimizer.local/global: "
                f"{', '.join(removed_flat_optimizer_keys)}"
            )
        for venue in ("local", "global"):
            venue_payload = optimizer_payload.get(venue)
            if venue_payload is None:
                continue
            if not isinstance(venue_payload, dict):
                raise ValueError(f"controller.optimizer.{venue} must be an object")
            removed_aliases = [
                key for key in removed_venue_optimizer_aliases if key in venue_payload
            ]
            if removed_aliases:
                replacements = ", ".join(
                    f"{key}->{removed_venue_optimizer_aliases[key]}"
                    for key in removed_aliases
                )
                raise ValueError(
                    f"removed venue optimizer aliases under controller.optimizer.{venue}: "
                    f"{replacements}"
                )
            unknown_keys = sorted(
                key for key in venue_payload if key not in venue_optimizer_keys
            )
            if unknown_keys:
                raise ValueError(
                    f"unknown controller.optimizer.{venue} keys: {', '.join(unknown_keys)}"
                )
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
        target_books_topic=(
            str(payload["target_books_topic"]) if payload.get("target_books_topic") is not None else None
        ),
        orderbook_grid=_orderbook_grid_from_mapping(payload),
        max_orders_per_side=int(payload.get("max_orders_per_side", 5)),
        reference_price_source=_normalize_reference_price_source(
            payload.get("reference_price_source", "global")
        ),
        local_price_model=_dict_or_none(payload.get("local_price_model")),
        local_market_flow_model=_dict_or_none(payload.get("local_market_flow_model")),
        global_market_flow_model=_dict_or_none(payload.get("global_market_flow_model")),
        global_price_model=_dict_or_none(payload.get("global_price_model")),
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



def _risk_from_mapping(payload: Any) -> RiskRuntimeConfig:
    if not isinstance(payload, dict):
        return RiskRuntimeConfig()
    enabled = bool(payload.get("enabled", True))
    local_payload = payload.get("local")
    global_payload = payload.get("global")
    local = _venue_risk_from_mapping(local_payload) if isinstance(local_payload, dict) else VenueRiskLimits()
    global_limits = _global_risk_from_mapping(global_payload) if isinstance(global_payload, dict) else GlobalVenueRiskLimits()
    engine_raw = str(payload.get("engine", "net_delta")).strip().lower()
    if engine_raw != "net_delta":
        raise ValueError("controller.risk.engine must be 'net_delta'")
    return RiskRuntimeConfig(
        enabled=enabled,
        engine="net_delta",
        local=local,
        global_=global_limits,
    )


def _venue_risk_from_mapping(payload: dict[str, Any]) -> VenueRiskLimits:
    """Parse local venue limits."""

    max_inv = float(payload.get("max_inventory", float("inf")))
    min_inv = float(payload.get("min_inventory", 0.0))
    if min_inv < 0.0:
        min_inv = 0.0
    return VenueRiskLimits(
        min_inventory=min_inv,
        max_inventory=max_inv,
        min_cash=float(payload.get("min_cash", 0.0)),
        max_open_notional=float(payload.get("max_open_notional", float("inf"))),
        max_order_notional=float(payload.get("max_order_notional", float("inf"))),
        max_market_data_age_ms=int(payload.get("max_market_data_age_ms", 10_000)),
        max_private_state_age_ms=int(payload.get("max_private_state_age_ms", 60_000)),
    )


def _global_risk_from_mapping(payload: dict[str, Any] | None) -> GlobalVenueRiskLimits:
    if not isinstance(payload, dict):
        return GlobalVenueRiskLimits()
    min_inv = float(payload.get("min_inventory", 0.0))
    if min_inv < 0.0:
        min_inv = 0.0
    return GlobalVenueRiskLimits(
        min_inventory=min_inv,
        max_inventory=float(payload.get("max_inventory", float("inf"))),
        min_cash=float(payload.get("min_cash", 0.0)),
        max_market_order_notional=float(payload.get("max_market_order_notional", float("inf"))),
        max_slippage_bps=float(payload.get("max_slippage_bps", float("inf"))),
        max_market_data_age_ms=int(payload.get("max_market_data_age_ms", 10_000)),
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
        "target_books_topic": config.normalized_target_books_topic,
        "orderbook_grid": config.orderbook_grid.as_contract(),
        "max_orders_per_side": config.max_orders_per_side,
        "vwap_depth_note": (
            "derived each frame from abs(local_inventory), or from "
            "abs(local_inventory + global_inventory) when liquidity sourcing is enabled; "
            "VWAP helpers cap to visible book depth when requested depth is larger"
        ),
        "reference_price_source": config.reference_price_source,
        "reference_price_source_note": (
            "selects which simulated sell-side VWAP path marks inventory and objective returns"
        ),
        "model_inputs": {
            "local_price_model": config.local_price_model or {"kind": "residual"},
            "local_market_flow_model": config.local_market_flow_model or {"kind": "hurdle_lognormal"},
            "global_market_flow_model": config.global_market_flow_model or {"kind": "hurdle_lognormal"},
            "global_price_model": config.global_price_model
            or {
                "kind": "ewma_gaussian",
                "note": "EWMA Gaussian deltas on global refs",
            },
            "execution_model": config.execution_model or {"kind": "real"},
            "optimizer": config.optimizer or {"kind": "monte_carlo_mpc_tensor_batched"},
        },
        "spread_depth_semantics": (
            "optimizer.local/global quote_size_min_per_side and quote_size_max_per_side "
            "bound venue cumulative per-side depth at spread_max; each venue default "
            "inventory target uses its own midpoint when target_inventory is unset"
        ),
        "risk": {
            "enabled": config.risk.enabled,
            "local": config.risk.local.as_dict(),
            "global": config.risk.global_.as_dict(),
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
            "shape": "controller.optimizer.local and controller.optimizer.global",
            "objective_weight": "venue contribution weight in the mixed joint objective",
            "return_risk_aversion_beta": (
                "venue risk aversion beta in (-1/beta)*ln(mean(exp(-beta*R_i)))"
            ),
            "inventory_target_penalty_weight": "venue lambda_inv inventory-capacity penalty weight",
            "target_inventory": "optional venue inventory target in base units",
            "target_inventory_multiplier": "venue multiplier when target_inventory is unset",
            "quote_size_min_per_side": "venue minimum cumulative quote depth per side at spread_max",
            "quote_size_max_per_side": "venue maximum cumulative quote depth per side at spread_max",
            "quote_spread_max": "venue maximum allowed sum of first bid/ask spread centers",
            "quote_spread_min_depth": "venue minimum executable depth at each side's first spread center",
            "soft_fill_temp": "venue soft-fill temperature",
            "soft_fill_activation": "venue soft-fill activation: sigmoid or hardsigmoid",
        },
        "float_bridge": {
            "mid": "quote-per-base float",
            "relative_book_spread": "absolute quote-per-base distance from mid",
            "depth_qty": "base units",
            "cash": "quote units",
            "inventory": "base units",
        },
    }
