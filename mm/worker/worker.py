"""Multi-strategy worker composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.env_loader import load_mm_env
from mm.observability import configure_observability, log_event
from mm.portfolio.virtual_ledger_service import VirtualLedgerService
from mm.portfolio.wallet_store import build_wallet_store
from mm.registry.memory import RegistryNotFoundError
from mm.registry.models import WorkerAssignmentRecord
from mm.registry.runtime_config import runtime_config_from_registry
from mm.registry.store import RegistryStore, build_registry_store
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import BusRuntimeConfig, MarketMakerRuntimeConfig, _bus_from_mapping
from mm.runtime.control import RuntimeControlPlane, RuntimeControlState
from mm.runtime.control_restore import boot_paused_actors, load_latest_runtime_control_state

from mm.worker.account_pool import SharedAccountPool, build_shared_account_pool, summarise_capability
from mm.worker.strategy_slot import StrategySlot, build_strategy_slot
from mm.worker.supervisor import StrategyBackoffConfig, StrategySupervisor
from mm.worker.wallet_resolve import load_worker_virtual_allocations
from mm.worker.worker_config import WorkerRuntimeConfig


@dataclass(slots=True)
class LoadedStrategy:
    strategy_id: str
    account_id: str
    config: MarketMakerRuntimeConfig
    ledger_source: str = "virtual"


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
    virtual_ledger_service: VirtualLedgerService | None = None
    build_options: RuntimeBuildOptions = field(default_factory=RuntimeBuildOptions.for_mode)
    _heartbeat_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        if self.virtual_ledger_service is not None:
            await self.virtual_ledger_service.start()
        await self.account_pool.start()
        # Start slots while paused, then expose control plane so Start can reload+resume.
        for supervisor in self.supervisors:
            await supervisor.start()
        if self.control_plane is not None:
            await self.control_plane.start()
            boot_command = (
                "worker_boot_paused"
                if self.control_plane.state.paused_actors == self.control_plane.managed_actors
                else "worker_boot_restored"
            )
            await self.control_plane.publish_state(command=boot_command)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._started = True

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        for supervisor in reversed(self.supervisors):
            await supervisor.stop()
        if self.virtual_ledger_service is not None:
            await self.virtual_ledger_service.stop()
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
                    log_event(
                        "worker_heartbeat_assignment_missing",
                        {"assignment_id": self.assignment.assignment_id},
                    )
                except Exception as exc:
                    log_event(
                        "worker_heartbeat_failed",
                        {
                            "assignment_id": self.assignment.assignment_id,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                if self.control_plane is not None:
                    try:
                        await self.control_plane.publish_state(command="heartbeat", observe=False)
                    except Exception as exc:
                        log_event(
                            "worker_control_state_publish_failed",
                            {
                                "assignment_id": self.assignment.assignment_id,
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
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
                ledger_source=strategy.ledger_source,
            )
        )
    if not loaded:
        raise ValueError("worker has no active strategies to run")
    return tuple(loaded)


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
    actor_ids = {
        item.config.controller.actor_id
        for item in strategies
        if item.config.controller is not None and item.config.controller.actor_id
    }
    expected_actors = tuple(sorted(actor_ids))
    expected_accounts = tuple(dict.fromkeys(item.account_id for item in strategies))
    strategy_to_account = {
        item.config.controller.actor_id: item.account_id
        for item in strategies
        if item.config.controller is not None and item.config.controller.actor_id
    }
    capability_payload = summarise_capability(
        worker_id=config.worker_id,
        assignment_id=assignment.assignment_id,
        pool=account_pool,
        expected_actors=expected_actors,
        expected_accounts=expected_accounts,
        strategy_to_account=strategy_to_account,
    )
    try:
        await runtime_bus.publish("worker:capability", capability_payload)
    except Exception as exc:  # pragma: no cover - publication is best-effort
        log_event(
            "worker_capability_publish_failed",
            {
                "worker_id": config.worker_id,
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
    supervisors_by_actor: dict[str, StrategySupervisor] = {}

    async def _on_actor_start(actor_id: str) -> None:
        supervisor = supervisors_by_actor.get(actor_id)
        if supervisor is None:
            return
        await supervisor.reload()

    async def _on_actor_pause(actor_id: str) -> None:
        supervisor = supervisors_by_actor.get(actor_id)
        if supervisor is not None and supervisor.slot.strategy_runner is not None:
            supervisor.slot.strategy_runner.cancel_decide()

    snapshot = load_latest_runtime_control_state(
        config.resolved_registry_database_url() or None
    )
    paused_actors = boot_paused_actors(actor_ids, snapshot)
    control_plane = RuntimeControlPlane(
        bus=runtime_bus,
        state=RuntimeControlState(paused_actors=set(paused_actors)),
        on_actor_start=_on_actor_start,
        on_actor_pause=_on_actor_pause,
        managed_actors=actor_ids,
    )
    allocations = load_worker_virtual_allocations(
        registry,
        tuple(item.strategy_id for item in strategies),
    )
    bindings = {
        item.config.controller.actor_id: item.config.binding
        for item in strategies
        if item.config.controller is not None
    }
    virtual_ledger_service: VirtualLedgerService | None = None
    if allocations:
        virtual_ledger_service = VirtualLedgerService(
            bus=runtime_bus,
            allocations=allocations,
            bindings=bindings,
            order_managers=account_pool.order_managers,
            control_plane=control_plane,
            wallet_store=build_wallet_store(),
        )

    def _rebuild_slot(strategy_id: str) -> StrategySlot:
        bundle = registry.load_runtime_bundle(strategy_id)
        strategy = bundle["strategy"]
        runtime_config = runtime_config_from_registry(**bundle)
        return build_strategy_slot(
            strategy_id=strategy_id,
            account_id=strategy.account_id,
            config=runtime_config,
            bus=runtime_bus,
            account_pool=account_pool,
            control_plane=control_plane,
            virtual_ledger_service=virtual_ledger_service,
            ledger_source=strategy.ledger_source,
        )

    slots_list: list[StrategySlot] = []
    supervisors_list: list[StrategySupervisor] = []
    for item in strategies:
        # Subscribe only when the supervisor builds the live slot. A fully
        # wired constructor slot would stay registered after the first rebuild
        # and stall every publisher once its queue filled.
        placeholder = StrategySlot(
            strategy_id=item.strategy_id,
            account_id=item.account_id,
            config=item.config,
        )
        supervisor = StrategySupervisor(
            slot=placeholder,
            backoff=StrategyBackoffConfig(
                initial_s=config.backoff.initial_s,
                max_s=config.backoff.max_s,
            ),
            rebuild_slot=lambda strategy_id=item.strategy_id: _rebuild_slot(strategy_id),
        )
        slots_list.append(placeholder)
        supervisors_list.append(supervisor)
        if item.config.controller is not None and item.config.controller.actor_id:
            supervisors_by_actor[item.config.controller.actor_id] = supervisor
    slots = tuple(slots_list)
    supervisors = tuple(supervisors_list)

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
            "boot_paused_actors": sorted(paused_actors),
            "boot_restored": bool(snapshot),
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
        virtual_ledger_service=virtual_ledger_service,
        build_options=options,
    )
