from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mm.aggregation.scheduler import IntervalScheduler
from mm.actors.frame_assembler import StrategyInputFrameAssembler
from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.core.clock import WallClock
from mm.exchanges.factory import (
    OrderCommandAdapterSelection,
    PrivateStateAdapterSelection,
    PublicMarketDataAdapterSelection,
    select_order_command_adapter,
    select_private_state_adapter,
    select_public_market_data_adapter,
)
from mm.execution.order_manager import OrderManager
from mm.execution.reconciler import ExecutionStateReconciler
from mm.ingestion.symbol_engine import SymbolEngine
from mm.observability_setup import configure_observability
from mm.env_loader import load_mm_env
from mm.private_state.reconciler import PrivateStateReconciler
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, SourceRole
from mm.runtime.control import RuntimeControlPlane
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class RuntimeComponents:
    """Owns the in-process runtime built from MarketMakerRuntimeConfig."""

    config: MarketMakerRuntimeConfig
    bus: AbstractEventBus
    scheduler: IntervalScheduler
    engines: dict[SourceRole, SymbolEngine]
    frame_assembler: StrategyInputFrameAssembler
    public_market_selections: dict[SourceRole, PublicMarketDataAdapterSelection]
    private_state_selections: dict[SourceRole, PrivateStateAdapterSelection] = field(default_factory=dict)
    order_command_selections: dict[SourceRole, OrderCommandAdapterSelection] = field(default_factory=dict)
    order_managers: dict[str, OrderManager] = field(default_factory=dict)
    private_reconcilers: dict[SourceRole, PrivateStateReconciler] = field(default_factory=dict)
    execution_reconcilers: dict[SourceRole, ExecutionStateReconciler] = field(default_factory=dict)
    persistence_daemons: tuple[object, ...] = ()
    control_plane: RuntimeControlPlane | None = None
    controller_runner: object | None = None
    _scheduler_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        for daemon in self.persistence_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
        await self.frame_assembler.start()
        if self.controller_runner is not None:
            await self.controller_runner.start()
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
        if self.controller_runner is not None:
            await self.controller_runner.stop()
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
        await self.frame_assembler.stop()
        for engine in reversed(tuple(self.engines.values())):
            await engine.stop()
        await self.bus.close()
        self._started = False

    def market_event_topic(self, role: SourceRole) -> str:
        source = self.config.primary if role == "primary" else self.config.hedge
        if source is None:
            raise ValueError("hedge source is not configured")
        return source.market_event_topic(self.config.interval_ms)

    def market_error_topic(self, role: SourceRole) -> str:
        source = self.config.primary if role == "primary" else self.config.hedge
        if source is None:
            raise ValueError("hedge source is not configured")
        return source.market_error_topic()

    def private_snapshot_topic(self, role: SourceRole) -> str:
        source = self.config.primary if role == "primary" else self.config.hedge
        if source is None:
            raise ValueError("source is not configured")
        return f"private:state:{source.normalized_exchange()}:snapshot"

    def execution_state_topic(self, role: SourceRole) -> str:
        source = self.config.primary if role == "primary" else self.config.hedge
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

    primary_private_queue = None
    hedge_private_queue = None
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
        if source.role == "primary":
            primary_private_queue = snapshot_queue
        else:
            hedge_private_queue = snapshot_queue

        if private_config.enable_execution:
            order_selection = select_order_command_adapter(
                source.normalized_exchange(),
                source.symbol,
                api_token=api_token,
            )
            order_command_selections[source.role] = order_selection
            manager = order_managers.get(source.normalized_exchange())
            if manager is None:
                manager = OrderManager(
                    exchange=source.normalized_exchange(),
                    command_adapter=order_selection.adapter,
                    bus=runtime_bus,
                    cancel_missing=False,
                )
                order_managers[source.normalized_exchange()] = manager
            execution_reconcilers[source.role] = ExecutionStateReconciler(
                source.normalized_exchange(),
                manager,
                runtime_bus,
            )

    primary_topic = config.primary.market_event_topic(config.interval_ms)
    hedge_topic = config.hedge.market_event_topic(config.interval_ms) if config.hedge is not None else None
    frame_assembler = StrategyInputFrameAssembler(
        primary_queue=runtime_bus.subscribe_queue(primary_topic),
        hedge_queue=runtime_bus.subscribe_queue(hedge_topic) if hedge_topic is not None else None,
        primary_private_queue=primary_private_queue,
        hedge_private_queue=hedge_private_queue,
        bus=runtime_bus,
        output_topic=config.frame_topic,
        max_hedge_lag_ms=config.max_hedge_lag_ms,
        hedge_grace_ms=config.hedge_grace_ms,
    )

    controller_runner = None
    if config.controller is not None and config.controller.enabled:
        from mm.strategies.mpc_controller import MpcControllerRunner

        controller_runner = MpcControllerRunner(
            config=config.controller,
            binding=config.binding,
            frame_queue=runtime_bus.subscribe_queue(config.frame_topic),
            order_managers=order_managers,
            bus=runtime_bus,
        )

    persistence_daemons = _build_persistence_daemons(config, runtime_bus)
    control_plane = RuntimeControlPlane(bus=runtime_bus)

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
        control_plane=control_plane,
        controller_runner=controller_runner,
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
    return tuple(daemons)


def _default_persistence_topics(config: MarketMakerRuntimeConfig) -> tuple[str, ...]:
    topics = [
        config.primary.market_event_topic(config.interval_ms),
        config.frame_topic,
        "strategy:actions",
        "execution:orders:" + config.primary.normalized_exchange(),
        "execution:state:" + config.primary.normalized_exchange(),
        "private:state:" + config.primary.normalized_exchange() + ":snapshot",
    ]
    if config.controller is not None:
        topics.append(config.controller.normalized_diagnostics_topic)
    if config.hedge is not None:
        hedge_exchange = config.hedge.normalized_exchange()
        topics.extend(
            [
                config.hedge.market_event_topic(config.interval_ms),
                "execution:orders:" + hedge_exchange,
                "execution:state:" + hedge_exchange,
                "private:state:" + hedge_exchange + ":snapshot",
            ]
        )
    return tuple(dict.fromkeys(topics))
