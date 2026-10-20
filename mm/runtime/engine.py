from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mm.aggregation.scheduler import IntervalScheduler
from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.application.joint_frame_assembler import JointStrategyFrameAssembler
from mm.application.decision_service import build_strategy_runner
from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.core.clock import WallClock
from mm.core.topics import RuntimeTopics
from mm.exchanges.factory import (
    OrderCommandAdapterSelection,
    PrivateStateAdapterSelection,
    PublicMarketDataAdapterSelection,
    select_order_command_adapter,
    select_venue_order_limit_source,
    select_private_state_adapter,
    select_public_market_data_adapter,
)
from mm.application.execution_service import OrderManager
from mm.application.reconciliation_service import ExecutionStateReconciler, PrivateStateReconciler
from mm.ingestion.symbol_engine import SymbolEngine
from mm.observability import configure_observability, log_event
from mm.observability.subscribers import RuntimeObservabilitySubscriber
from mm.env_loader import load_mm_env
from mm.execution.order_attribution import OrderAttributionRegistry
from mm.execution.order_attribution_store import OrderAttributionService, build_order_attribution_service
from mm.execution.reconcile import ReconcileConfig
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, ReconcileRuntimeConfig, SourceRole
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.control import RuntimeControlPlane
from mm.ledger.service import build_portfolio_ledger_service
from mm.persistence.service import build_persistence_writers
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class RuntimeComponents:
    """Owns the in-process runtime built from MarketMakerRuntimeConfig."""

    config: MarketMakerRuntimeConfig
    bus: AbstractEventBus
    scheduler: IntervalScheduler
    engines: dict[SourceRole, SymbolEngine]
    frame_assembler: JointStrategyFrameAssembler | None
    public_market_selections: dict[SourceRole, PublicMarketDataAdapterSelection]
    private_state_selections: dict[SourceRole, PrivateStateAdapterSelection] = field(default_factory=dict)
    order_command_selections: dict[SourceRole, OrderCommandAdapterSelection] = field(default_factory=dict)
    order_managers: dict[str, OrderManager] = field(default_factory=dict)
    private_reconcilers: dict[SourceRole, PrivateStateReconciler] = field(default_factory=dict)
    execution_reconcilers: dict[SourceRole, ExecutionStateReconciler] = field(default_factory=dict)
    persistence_daemons: tuple[object, ...] = ()
    observability_daemons: tuple[object, ...] = ()
    control_plane: RuntimeControlPlane | None = None
    strategy_runner: object | None = None
    ledger_service: object | None = None
    attribution_services: dict[str, OrderAttributionService] = field(default_factory=dict)
    build_options: RuntimeBuildOptions = field(default_factory=RuntimeBuildOptions.monolith)
    _scheduler_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        for service in self.attribution_services.values():
            await service.start()
        for daemon in self.persistence_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
        for daemon in self.observability_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
        if self.frame_assembler is not None:
            await self.frame_assembler.start()
        if self.strategy_runner is not None:
            await self.strategy_runner.start()
        if self.ledger_service is not None:
            await self.ledger_service.start()
        if self.control_plane is not None:
            await self.control_plane.start()
        for engine in self.engines.values():
            await engine.start()
        for reconciler in self.execution_reconcilers.values():
            await reconciler.start()
        for reconciler in self.private_reconcilers.values():
            await reconciler.start()
        self._scheduler_task = asyncio.create_task(self.scheduler.run())
        self._started = True

    async def stop(self) -> None:
        if self._scheduler_task is not None:
            self.scheduler.stop()
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)
            self._scheduler_task = None
        if self.ledger_service is not None:
            await self.ledger_service.stop()
        if self.strategy_runner is not None:
            await self.strategy_runner.stop()
        if self.control_plane is not None:
            await self.control_plane.stop()
        for reconciler in reversed(tuple(self.private_reconcilers.values())):
            await reconciler.stop()
        for reconciler in reversed(tuple(self.execution_reconcilers.values())):
            await reconciler.stop()
        for service in self.attribution_services.values():
            await service.stop()
        for daemon in reversed(self.persistence_daemons):
            stop = getattr(daemon, "stop", None)
            if stop is not None:
                await stop()
        for daemon in reversed(self.observability_daemons):
            stop = getattr(daemon, "stop", None)
            if stop is not None:
                await stop()
        if self.frame_assembler is not None:
            await self.frame_assembler.stop()
        for engine in reversed(tuple(self.engines.values())):
            await engine.stop()
        await self.bus.close()
        self._started = False

    def market_event_topic(self, role: SourceRole) -> str:
        source = self.config.local if role == "local" else self.config.global_
        if source is None:
            raise ValueError("global source is not configured")
        return source.market_event_topic(self.config.interval_ms)

    def market_error_topic(self, role: SourceRole) -> str:
        source = self.config.local if role == "local" else self.config.global_
        if source is None:
            raise ValueError("global source is not configured")
        return source.market_error_topic()

    def private_snapshot_topic(self, role: SourceRole) -> str:
        source = self.config.local if role == "local" else self.config.global_
        if source is None:
            raise ValueError("source is not configured")
        return f"private:state:{source.normalized_exchange()}:snapshot"

    def execution_state_topic(self, role: SourceRole) -> str:
        source = self.config.local if role == "local" else self.config.global_
        if source is None:
            raise ValueError("source is not configured")
        return f"execution:state:{source.normalized_exchange()}"


def resolve_build_options(config: MarketMakerRuntimeConfig) -> RuntimeBuildOptions:
    """Pick runtime composition from config unless overridden by the caller."""

    if all(source.market_data_source == "bus" for source in config.sources):
        return RuntimeBuildOptions.for_mode("consumer_only")
    return RuntimeBuildOptions.monolith()


async def build_runtime_components(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus | None = None,
    build_options: RuntimeBuildOptions | None = None,
) -> RuntimeComponents:
    options = build_options or resolve_build_options(config)
    load_mm_env()
    configure_observability(config.observability)
    _validate_build_options(config, options)
    log_event(
        "runtime_build_options",
        {
            "mode": options.mode,
            "enable_public_ingestion": options.enable_public_ingestion,
            "enable_strategy": options.enable_strategy,
            "enable_private_state": options.enable_private_state,
            "enable_execution": options.enable_execution,
            "enable_persistence": options.enable_persistence,
            "enable_observability": options.enable_observability,
            "local_market_data_source": config.local.market_data_source,
            "global_market_data_source": (
                config.global_.market_data_source if config.global_ is not None else None
            ),
        },
    )
    if config.controller is not None:
        log_event(
            "runtime_controller_config",
            {
                "actor_id": config.controller.actor_id,
                "enabled": config.controller.enabled,
                "dry_run": config.controller.dry_run,
                "execution_after_steps": config.controller.execution_after_steps,
                "negative_objective_action": config.controller.negative_objective_action,
                "placement_timeout_ms": config.controller.placement_timeout_ms,
                "placement_max_concurrency": config.controller.placement_max_concurrency,
            },
        )
    runtime_bus = bus or build_event_bus(config.bus, run_id=config.observability.run_id)
    scheduler = IntervalScheduler(WallClock(), max_concurrency=config.scheduler.max_concurrency)
    public_selections: dict[SourceRole, PublicMarketDataAdapterSelection] = {}
    engines: dict[SourceRole, SymbolEngine] = {}

    for source in config.sources:
        if not _ingest_source_locally(source, options):
            continue
        selection = _select_public_source(source)
        public_selections[source.role] = selection
        engine = build_symbol_engine(
            exchange=source.normalized_exchange(),
            symbol=source.symbol,
            adapter=selection.adapter,
            bus=runtime_bus,
        )
        await engine.publish_on_intervals(scheduler, [config.interval_ms])
        engines[source.role] = engine

    private_selections: dict[SourceRole, PrivateStateAdapterSelection] = {}
    order_command_selections: dict[SourceRole, OrderCommandAdapterSelection] = {}
    order_managers: dict[str, OrderManager] = {}
    private_reconcilers: dict[SourceRole, PrivateStateReconciler] = {}
    execution_reconcilers: dict[SourceRole, ExecutionStateReconciler] = {}
    order_attribution: dict[str, OrderAttributionRegistry] = {}
    attribution_services: dict[str, OrderAttributionService] = {}

    def _attribution_for(exchange: str) -> OrderAttributionRegistry:
        key = exchange.lower().strip()
        registry = order_attribution.get(key)
        if registry is None:
            registry = OrderAttributionRegistry()
            attribution_services[key] = build_order_attribution_service(registry, exchange=key)
            order_attribution[key] = registry
        return registry

    if options.enable_private_state:
        for source in config.sources:
            private_config = config.private_config_for(source.role)
            if private_config is None:
                continue
            api_credentials = private_config.resolved_api_credentials(source.normalized_exchange())
            if not api_credentials:
                continue
            private_selection = select_private_state_adapter(
                source.normalized_exchange(),
                source.symbol,
                mode=private_config.mode,
                api_token=api_credentials,
                poll_interval_ms=private_config.poll_interval_ms,
            )
            private_selections[source.role] = private_selection
            attribution_registry = _attribution_for(source.normalized_exchange())
            private_selection.adapter.attribution_registry = attribution_registry
            private_reconcilers[source.role] = PrivateStateReconciler(
                exchange=source.normalized_exchange(),
                adapter=private_selection.adapter,
                bus=runtime_bus,
                fill_event_source=private_config.fill_event_source,
                attribution_registry=attribution_registry,
            )
            if not options.enable_execution:
                continue
            if config.execution.for_role(source.role).enabled:
                placement_max_concurrency = 32
                if config.controller is not None:
                    placement_max_concurrency = config.controller.placement_max_concurrency
                order_selection = select_order_command_adapter(
                    source.normalized_exchange(),
                    source.symbol,
                    api_token=api_credentials,
                    placement_max_concurrency=placement_max_concurrency,
                )
                order_command_selections[source.role] = order_selection
                manager = order_managers.get(source.normalized_exchange())
                if manager is None:
                    manager = OrderManager(
                        exchange=source.normalized_exchange(),
                        command_adapter=order_selection.adapter,
                        bus=runtime_bus,
                        limit_source=select_venue_order_limit_source(source.normalized_exchange()),
                        reconcile_config=_execution_reconcile_config(config),
                        attribution_registry=attribution_registry,
                    )
                    order_managers[source.normalized_exchange()] = manager
                execution_reconcilers[source.role] = ExecutionStateReconciler(
                    source.normalized_exchange(),
                    manager,
                    runtime_bus,
                )

    frame_assembler: JointStrategyFrameAssembler | None = None
    control_plane: RuntimeControlPlane | None = None
    strategy_runner: object | None = None
    ledger_service: object | None = None

    if options.enable_strategy:
        topics = _runtime_topics(config)
        local_topic = topics.local_market_events
        global_topic = topics.global_market_events
        frame_assembler_inner = StrategyInputFrameAssembler(
            local_queue=runtime_bus.subscribe_queue(local_topic),
            global_queue=runtime_bus.subscribe_queue(global_topic) if global_topic is not None else None,
            max_global_lag_ms=config.max_global_lag_ms,
            global_grace_ms=config.global_grace_ms,
        )

        ledger_service = build_portfolio_ledger_service(
            config,
            bus=runtime_bus,
            order_managers=order_managers,
        )

        max_local_private_age_ms = 120_000
        max_global_private_age_ms = 120_000
        if config.controller is not None:
            max_local_private_age_ms = config.controller.risk.local.max_private_state_age_ms
            max_global_private_age_ms = config.controller.risk.global_.max_private_state_age_ms

        frame_assembler = JointStrategyFrameAssembler(
            input_assembler=frame_assembler_inner,
            binding=config.binding,
            liquidity_sourcing=config.liquidity_sourcing,
            bus=runtime_bus,
            joint_output_topic=config.joint_frame_topic,
            ledger_service=ledger_service,
            max_local_private_age_ms=max_local_private_age_ms,
            max_global_private_age_ms=max_global_private_age_ms,
            max_local_market_age_ms=config.resolved_max_local_market_age_ms(),
            max_global_market_age_ms=config.resolved_max_global_market_age_ms(),
        )

        control_plane = RuntimeControlPlane(bus=runtime_bus)

        strategy_runner = build_strategy_runner(
            config,
            bus=runtime_bus,
            frame_topic=config.joint_frame_topic,
            order_managers=order_managers,
            control_plane=control_plane,
        )

    persistence_daemons = _build_persistence_daemons(config, runtime_bus, options)
    observability_daemons = _build_observability_daemons(config, runtime_bus, options)

    return RuntimeComponents(
        config=config,
        bus=runtime_bus,
        scheduler=scheduler,
        engines=engines,
        frame_assembler=frame_assembler,
        public_market_selections=public_selections,
        private_state_selections=private_selections,
        order_command_selections=order_command_selections,
        order_managers=order_managers,
        private_reconcilers=private_reconcilers,
        execution_reconcilers=execution_reconcilers,
        persistence_daemons=persistence_daemons,
        observability_daemons=observability_daemons,
        control_plane=control_plane,
        strategy_runner=strategy_runner,
        ledger_service=ledger_service,
        attribution_services=attribution_services,
        build_options=options,
    )


def _ingest_source_locally(source: MarketDataSourceConfig, options: RuntimeBuildOptions) -> bool:
    return options.enable_public_ingestion and source.market_data_source == "local"


def _validate_build_options(config: MarketMakerRuntimeConfig, options: RuntimeBuildOptions) -> None:
    bus_sources = [source for source in config.sources if source.market_data_source == "bus"]
    if not bus_sources:
        return
    if config.bus.kind != "redis_streams":
        exchanges = ", ".join(source.normalized_exchange() for source in bus_sources)
        raise ValueError(
            "market_data_source='bus' requires bus.kind='redis_streams' "
            f"for cross-process delivery (sources: {exchanges})"
        )


def _select_public_source(source: MarketDataSourceConfig) -> PublicMarketDataAdapterSelection:
    return select_public_market_data_adapter(
        source.normalized_exchange(),
        source.symbol,
        mode=source.mode,
        depth_limit=source.depth_limit,
        trades_limit=source.trades_limit,
        poll_interval_ms=source.poll_interval_ms,
    )


def _build_persistence_daemons(
    config: MarketMakerRuntimeConfig,
    bus: AbstractEventBus,
    options: RuntimeBuildOptions,
) -> tuple[object, ...]:
    if not options.enable_persistence:
        log_event("persistence_disabled", {"reason": "build_options"})
        return ()
    return build_persistence_writers(
        bus=bus,
        persistence=config.persistence,
        topics=config.effective_persistence_topics,
        run_id=config.observability.run_id,
    )


def _build_observability_daemons(
    config: MarketMakerRuntimeConfig,
    bus: AbstractEventBus,
    options: RuntimeBuildOptions,
) -> tuple[object, ...]:
    if not options.enable_observability:
        return ()
    topics = _runtime_topics(config)
    observed_topics = [
        config.joint_frame_topic,
        RuntimeTopics.ledger_snapshot(),
        topics.strategy_decisions,
        topics.strategy_actions,
        "risk:events",
        "execution:errors",
        RuntimeTopics.execution_orders(config.local.normalized_exchange()),
        RuntimeTopics.execution_state(config.local.normalized_exchange()),
        f"private:state:{config.local.normalized_exchange()}:fills",
        f"private:state:{config.local.normalized_exchange()}:errors",
        config.local.market_error_topic(),
        config.local.raw_orderbook_topic(),
        config.local.raw_trade_topic(),
    ]
    if topics.strategy_diagnostics is not None:
        observed_topics.append(topics.strategy_diagnostics)
    if config.global_ is not None:
        observed_topics.extend(
            [
                RuntimeTopics.execution_orders(config.global_.normalized_exchange()),
                RuntimeTopics.execution_state(config.global_.normalized_exchange()),
                f"private:state:{config.global_.normalized_exchange()}:fills",
                f"private:state:{config.global_.normalized_exchange()}:errors",
                config.global_.market_error_topic(),
                config.global_.raw_orderbook_topic(),
                config.global_.raw_trade_topic(),
            ]
        )
    return (RuntimeObservabilitySubscriber(bus=bus, topics=tuple(observed_topics)),)


def _default_persistence_topics(config: MarketMakerRuntimeConfig) -> tuple[str, ...]:
    return config.default_persistence_topics


def _runtime_topics(config: MarketMakerRuntimeConfig) -> RuntimeTopics:
    return RuntimeTopics(
        local=config.local,
        global_=config.global_,
        interval_ms=config.interval_ms,
        action_topic=config.controller.normalized_action_topic if config.controller is not None else "strategy:actions",
        decisions_topic=(
            config.controller.normalized_decisions_topic
            if config.controller is not None
            else "strategy:decisions"
        ),
        diagnostics_topic=(
            config.controller.normalized_diagnostics_topic
            if config.controller is not None
            else None
        ),
    )


def _execution_reconcile_config(config: MarketMakerRuntimeConfig) -> ReconcileConfig:
    runtime = (
        config.controller.reconcile
        if config.controller is not None
        else ReconcileRuntimeConfig()
    )
    return ReconcileConfig(
        price_tol_bps=runtime.price_tol_bps,
        qty_tol_rel=runtime.qty_tol_rel,
    )
