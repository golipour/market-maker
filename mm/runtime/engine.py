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
from mm.execution.reconcile import ReconcileConfig
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, ReconcileRuntimeConfig, SourceRole
from mm.runtime.control import RuntimeControlPlane
from mm.ledger.service import build_portfolio_ledger_service
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class RuntimeComponents:
    """Owns the in-process runtime built from MarketMakerRuntimeConfig."""

    config: MarketMakerRuntimeConfig
    bus: AbstractEventBus
    scheduler: IntervalScheduler
    engines: dict[SourceRole, SymbolEngine]
    frame_assembler: JointStrategyFrameAssembler
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
    _scheduler_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        for daemon in self.persistence_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
        for daemon in self.observability_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
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
        for daemon in reversed(self.persistence_daemons):
            stop = getattr(daemon, "stop", None)
            if stop is not None:
                await stop()
        for daemon in reversed(self.observability_daemons):
            stop = getattr(daemon, "stop", None)
            if stop is not None:
                await stop()
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


async def build_runtime_components(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus | None = None,
) -> RuntimeComponents:
    load_mm_env()
    configure_observability(config.observability)
    runtime_bus = bus or build_event_bus(config.bus, run_id=config.observability.run_id)
    scheduler = IntervalScheduler(WallClock())
    public_selections: dict[SourceRole, PublicMarketDataAdapterSelection] = {}
    engines: dict[SourceRole, SymbolEngine] = {}

    for source in config.sources:
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

    local_private_queue = None
    global_private_queue = None
    private_selections: dict[SourceRole, PrivateStateAdapterSelection] = {}
    order_command_selections: dict[SourceRole, OrderCommandAdapterSelection] = {}
    order_managers: dict[str, OrderManager] = {}
    private_reconcilers: dict[SourceRole, PrivateStateReconciler] = {}
    execution_reconcilers: dict[SourceRole, ExecutionStateReconciler] = {}

    for source in config.sources:
        private_config = config.private_config_for(source.role)
        if private_config is None:
            continue
        api_token = private_config.resolved_api_token(source.normalized_exchange())
        if not api_token:
            continue
        private_selection = select_private_state_adapter(
            source.normalized_exchange(),
            source.symbol,
            mode=private_config.mode,
            api_token=api_token,
            poll_interval_ms=private_config.poll_interval_ms,
        )
        private_selections[source.role] = private_selection
        private_reconcilers[source.role] = PrivateStateReconciler(
            exchange=source.normalized_exchange(),
            adapter=private_selection.adapter,
            bus=runtime_bus,
        )
        snapshot_queue = runtime_bus.subscribe_queue(f"private:state:{source.normalized_exchange()}:snapshot")
        if source.role == "local":
            local_private_queue = snapshot_queue
        else:
            global_private_queue = snapshot_queue

        if private_config.enable_execution:
            placement_max_concurrency = 8
            if config.controller is not None:
                placement_max_concurrency = config.controller.placement_max_concurrency
            order_selection = select_order_command_adapter(
                source.normalized_exchange(),
                source.symbol,
                api_token=api_token,
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
                )
                order_managers[source.normalized_exchange()] = manager
            execution_reconcilers[source.role] = ExecutionStateReconciler(
                source.normalized_exchange(),
                manager,
                runtime_bus,
            )

    topics = _runtime_topics(config)
    local_topic = topics.local_market_events
    global_topic = topics.global_market_events
    frame_assembler_inner = StrategyInputFrameAssembler(
        local_queue=runtime_bus.subscribe_queue(local_topic),
        global_queue=runtime_bus.subscribe_queue(global_topic) if global_topic is not None else None,
        local_private_queue=local_private_queue,
        global_private_queue=global_private_queue,
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
    )

    control_plane = RuntimeControlPlane(bus=runtime_bus)

    strategy_runner = build_strategy_runner(
        config,
        bus=runtime_bus,
        frame_topic=config.joint_frame_topic,
        order_managers=order_managers,
        control_plane=control_plane,
    )

    persistence_daemons = _build_persistence_daemons(config, runtime_bus)
    observability_daemons = _build_observability_daemons(config, runtime_bus)

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


def _build_persistence_daemons(config: MarketMakerRuntimeConfig, bus: AbstractEventBus) -> tuple[object, ...]:
    if not config.persistence.enabled:
        log_event("persistence_disabled", {"reason": "config"})
        return ()
    topics = config.persistence.topics or _default_persistence_topics(config)
    daemons: list[object] = []
    database_url = config.persistence.resolved_database_url()
    if database_url:
        from mm.persistence.postgres import PostgresEventWriter

        daemons.append(
            PostgresEventWriter(
                bus=bus,
                topics=topics,
                database_url=database_url,
                batch_size=config.persistence.batch_size,
                flush_interval_ms=config.persistence.flush_interval_ms,
                run_id=config.observability.run_id,
            )
        )
    else:
        log_event(
            "persistence_sink_missing",
            {
                "sink": "postgres",
                "database_url_env": config.persistence.database_url_env,
                "message": "persistence.enabled is true, but no database URL resolved",
            },
        )
    parquet_root = config.persistence.resolved_parquet_root()
    if parquet_root:
        from mm.persistence.parquet_writer import ParquetEventWriter

        daemons.append(
            ParquetEventWriter(
                bus=bus,
                topics=topics,
                root_path=parquet_root,
                batch_size=config.persistence.batch_size,
                flush_interval_ms=config.persistence.flush_interval_ms,
                run_id=config.observability.run_id,
            )
        )
    if not daemons:
        log_event(
            "persistence_no_writers_configured",
            {
                "database_url_env": config.persistence.database_url_env,
                "parquet_root_env": config.persistence.parquet_root_env,
                "topic_count": len(topics),
            },
        )
    else:
        log_event(
            "persistence_writers_configured",
            {
                "writers": [type(daemon).__name__ for daemon in daemons],
                "topic_count": len(topics),
            },
        )
    return tuple(daemons)


def _build_observability_daemons(config: MarketMakerRuntimeConfig, bus: AbstractEventBus) -> tuple[object, ...]:
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
        config.local.market_error_topic(),
    ]
    if topics.strategy_diagnostics is not None:
        observed_topics.append(topics.strategy_diagnostics)
    if config.global_ is not None:
        observed_topics.extend(
            [
                RuntimeTopics.execution_orders(config.global_.normalized_exchange()),
                RuntimeTopics.execution_state(config.global_.normalized_exchange()),
                config.global_.market_error_topic(),
            ]
        )
    return (RuntimeObservabilitySubscriber(bus=bus, topics=tuple(observed_topics)),)


def _default_persistence_topics(config: MarketMakerRuntimeConfig) -> tuple[str, ...]:
    runtime_topics = _runtime_topics(config)
    topics = [
        runtime_topics.local_market_events,
        config.joint_frame_topic,
        RuntimeTopics.ledger_snapshot(),
        runtime_topics.strategy_actions,
        RuntimeTopics.execution_orders(config.local.normalized_exchange()),
        RuntimeTopics.execution_state(config.local.normalized_exchange()),
        RuntimeTopics.private_snapshot(config.local.normalized_exchange()),
    ]
    if config.controller is not None:
        topics.append(runtime_topics.strategy_decisions)
        topics.append(config.controller.normalized_diagnostics_topic)
        topics.append(config.controller.normalized_target_books_topic)
    if config.global_ is not None:
        global_exchange = config.global_.normalized_exchange()
        topics.extend(
            [
                config.global_.market_event_topic(config.interval_ms),
                RuntimeTopics.execution_orders(global_exchange),
                RuntimeTopics.execution_state(global_exchange),
                RuntimeTopics.private_snapshot(global_exchange),
                f"private:state:{config.local.normalized_exchange()}:fills",
            ]
        )
    return tuple(dict.fromkeys(topics))


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
