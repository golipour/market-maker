"""Multi-strategy worker composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.env_loader import load_mm_env
from mm.observability import configure_observability, log_event
from mm.portfolio.allocation_service import PortfolioAllocationService
from mm.registry.memory import RegistryNotFoundError
from mm.registry.models import AllocationRecord, WorkerAssignmentRecord
from mm.registry.runtime_config import runtime_config_from_registry
from mm.registry.store import RegistryStore, build_registry_store
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import BusRuntimeConfig, MarketMakerRuntimeConfig, _bus_from_mapping
from mm.runtime.control import RuntimeControlPlane

from mm.worker.account_pool import SharedAccountPool, build_shared_account_pool
from mm.worker.strategy_slot import StrategySlot, build_strategy_slot
from mm.worker.supervisor import StrategyBackoffConfig, StrategySupervisor
from mm.worker.worker_config import WorkerRuntimeConfig


@dataclass(slots=True)
class LoadedStrategy:
    strategy_id: str
    account_id: str
    config: MarketMakerRuntimeConfig


@dataclass(slots=True)
class WorkerComponents:
    config: WorkerRuntimeConfig
    assignment: WorkerAssignmentRecord
    store: RegistryStore
    bus: AbstractEventBus
    account_pool: SharedAccountPool
    strategies: tuple[LoadedStrategy, ...] = ()
    slots: tuple[StrategySlot, ...] = ()
    supervisors: tuple[StrategySupervisor, ...] = ()
    control_plane: RuntimeControlPlane | None = None
    allocation_service: PortfolioAllocationService | None = None
    build_options: RuntimeBuildOptions = field(default_factory=RuntimeBuildOptions.for_mode)
    _heartbeat_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        if self.control_plane is not None:
            await self.control_plane.start()
        if self.allocation_service is not None:
            await self.allocation_service.start()
        await self.account_pool.start()
        for supervisor in self.supervisors:
            await supervisor.start()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._started = True

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        for supervisor in reversed(self.supervisors):
            await supervisor.stop()
        if self.allocation_service is not None:
            await self.allocation_service.stop()
        await self.account_pool.stop()
        if self.control_plane is not None:
            await self.control_plane.stop()
        await self.bus.close()
        self._started = False

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                try:
                    self.store.heartbeat_worker_assignment(
                        self.assignment.assignment_id,
                        lease_ttl_seconds=self.config.lease_ttl_seconds,
                    )
                except RegistryNotFoundError:
                    return
                await asyncio.sleep(self.config.heartbeat_interval_s)
        except asyncio.CancelledError:
            raise


def resolve_worker_assignment(
    store: RegistryStore,
    config: WorkerRuntimeConfig,
) -> WorkerAssignmentRecord:
    if config.assignment_id:
        return store.get_worker_assignment(config.assignment_id)
    return store.get_active_worker_assignment(config.worker_id)


def load_worker_strategies(
    store: RegistryStore,
    strategy_ids: tuple[str, ...],
) -> tuple[LoadedStrategy, ...]:
    loaded: list[LoadedStrategy] = []
    for strategy_id in strategy_ids:
        bundle = store.load_runtime_bundle(strategy_id)
        strategy = bundle["strategy"]
        if strategy.status != "active":
            log_event(
                "worker_strategy_skipped",
                {"strategy_id": strategy_id, "status": strategy.status},
            )
            continue
        runtime_config = runtime_config_from_registry(**bundle)
        loaded.append(
            LoadedStrategy(
                strategy_id=strategy_id,
                account_id=strategy.account_id,
                config=runtime_config,
            )
        )
    if not loaded:
        raise ValueError("worker has no active strategies to run")
    return tuple(loaded)


def load_worker_allocations(
    store: RegistryStore,
    strategies: tuple[LoadedStrategy, ...],
) -> tuple[AllocationRecord, ...]:
    allocations: list[AllocationRecord] = []
    for item in strategies:
        try:
            allocations.append(store.get_allocation_for_strategy(item.strategy_id))
        except RegistryNotFoundError:
            continue
    return tuple(allocations)


def _merge_bus_config(
    platform_bus: dict[str, Any],
    worker_bus: BusRuntimeConfig | None,
    *,
    worker_id: str,
) -> BusRuntimeConfig:
    payload: dict[str, Any] = dict(platform_bus)
    if worker_bus is not None:
        payload.update(
            {
                "kind": worker_bus.kind,
                "redis_url": worker_bus.redis_url,
                "stream_prefix": worker_bus.stream_prefix,
                "consumer_group": worker_bus.consumer_group,
                "consumer_name": worker_bus.consumer_name,
                "max_stream_length": worker_bus.max_stream_length,
                "read_block_ms": worker_bus.read_block_ms,
                "max_queue_size": worker_bus.max_queue_size,
                "backpressure_policy": worker_bus.backpressure_policy,
            }
        )
    payload.setdefault("kind", "redis_streams")
    payload.setdefault("consumer_group", "mm-worker")
    payload.setdefault("consumer_name", worker_id)
    return _bus_from_mapping(payload)


async def build_worker_components(
    config: WorkerRuntimeConfig,
    *,
    store: RegistryStore | None = None,
    bus: AbstractEventBus | None = None,
    build_options: RuntimeBuildOptions | None = None,
) -> WorkerComponents:
    load_mm_env()
    configure_observability(config.observability)
    options = build_options or RuntimeBuildOptions.for_mode("consumer_only")
    registry = store or build_registry_store(database_url=config.resolved_registry_database_url())

    assignment = resolve_worker_assignment(registry, config)
    if assignment.worker_id != config.worker_id:
        raise ValueError(
            f"assignment {assignment.assignment_id!r} belongs to worker "
            f"{assignment.worker_id!r}, not {config.worker_id!r}"
        )
    strategy_ids = config.resolved_strategy_ids(assignment.strategy_ids)
    if not strategy_ids:
        raise ValueError(f"worker assignment {assignment.assignment_id!r} has no strategy_ids")

    strategies = load_worker_strategies(registry, strategy_ids)
    platform = registry.get_platform_settings()
    bus_config = _merge_bus_config(platform.bus, config.bus, worker_id=config.worker_id)
    if bus_config.kind != "redis_streams":
        raise ValueError("multi-strategy worker requires bus.kind='redis_streams'")

    for strategy in strategies:
        for source in strategy.config.sources:
            if source.market_data_source != "bus":
                raise ValueError(
                    f"strategy {strategy.strategy_id} uses market_data_source="
                    f"{source.market_data_source!r}; worker requires 'bus'"
                )

    runtime_bus = bus or build_event_bus(bus_config, run_id=config.observability.run_id)
    strategy_triples = tuple(
        (item.strategy_id, item.account_id, item.config) for item in strategies
    )
    account_pool = build_shared_account_pool(strategy_triples, bus=runtime_bus, options=options)
    control_plane = RuntimeControlPlane(bus=runtime_bus)
    allocations = load_worker_allocations(registry, strategies)
    bindings = {
        item.config.controller.actor_id: item.config.binding
        for item in strategies
        if item.config.controller is not None
    }
    allocation_service: PortfolioAllocationService | None = None
    if allocations:
        allocation_service = PortfolioAllocationService(
            bus=runtime_bus,
            allocations=allocations,
            bindings=bindings,
            order_managers=account_pool.order_managers,
            control_plane=control_plane,
        )

    slots = tuple(
        build_strategy_slot(
            strategy_id=item.strategy_id,
            account_id=item.account_id,
            config=item.config,
            bus=runtime_bus,
            account_pool=account_pool,
            control_plane=control_plane,
            allocation_service=allocation_service,
        )
        for item in strategies
    )
    supervisors = tuple(
        StrategySupervisor(
            slot=slot,
            backoff=StrategyBackoffConfig(
                initial_s=config.backoff.initial_s,
                max_s=config.backoff.max_s,
            ),
        )
        for slot in slots
    )

    log_event(
        "worker_build_complete",
        {
            "worker_id": config.worker_id,
            "assignment_id": assignment.assignment_id,
            "strategy_ids": [item.strategy_id for item in strategies],
            "account_ids": list(dict.fromkeys(item.account_id for item in strategies)),
            "shared_exchanges": list(account_pool.order_managers.keys()),
            "allocation_count": len(allocations),
            "mode": config.mode,
            "bus_kind": bus_config.kind,
            "consumer_group": bus_config.consumer_group,
            "consumer_name": bus_config.consumer_name,
        },
    )

    return WorkerComponents(
        config=config,
        assignment=assignment,
        store=registry,
        bus=runtime_bus,
        account_pool=account_pool,
        strategies=strategies,
        slots=slots,
        supervisors=supervisors,
        control_plane=control_plane,
        allocation_service=allocation_service,
        build_options=options,
    )
