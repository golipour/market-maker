"""Shared exchange-account resources for multi-strategy workers."""

from __future__ import annotations

from dataclasses import dataclass, field

from mm.application.execution_service import OrderManager
from mm.application.reconciliation_service import ExecutionStateReconciler, PrivateStateReconciler
from mm.bus.base import AbstractEventBus
from mm.exchanges.factory import (
    OrderCommandAdapterSelection,
    PrivateStateAdapterSelection,
    select_order_command_adapter,
    select_venue_order_limit_source,
    select_private_state_adapter,
)
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import MarketMakerRuntimeConfig, ReconcileRuntimeConfig
from mm.execution.order_attribution import OrderAttributionRegistry, load_attribution_registry
from mm.execution.reconcile import ReconcileConfig


@dataclass(slots=True)
class SharedAccountPool:
    """Account-scoped private state and execution resources shared across strategies."""

    order_managers: dict[str, OrderManager] = field(default_factory=dict)
    private_reconcilers: dict[str, PrivateStateReconciler] = field(default_factory=dict)
    execution_reconcilers: dict[str, ExecutionStateReconciler] = field(default_factory=dict)
    private_state_selections: dict[str, PrivateStateAdapterSelection] = field(default_factory=dict)
    order_command_selections: dict[str, OrderCommandAdapterSelection] = field(default_factory=dict)
    order_attribution: dict[str, OrderAttributionRegistry] = field(default_factory=dict)

    def attribution_for(self, exchange: str) -> OrderAttributionRegistry:
        key = exchange.lower().strip()
        registry = self.order_attribution.get(key)
        if registry is None:
            registry = load_attribution_registry(key)
            self.order_attribution[key] = registry
        return registry

    async def start(self) -> None:
        for reconciler in self.execution_reconcilers.values():
            await reconciler.start()
        for reconciler in self.private_reconcilers.values():
            await reconciler.start()

    async def stop(self) -> None:
        for reconciler in reversed(tuple(self.execution_reconcilers.values())):
            await reconciler.stop()
        for reconciler in reversed(tuple(self.private_reconcilers.values())):
            await reconciler.stop()


def validate_worker_account_exchange_mapping(
    strategies: tuple[tuple[str, str, MarketMakerRuntimeConfig], ...],
) -> None:
    """Ensure a worker does not host multiple accounts on the same exchange."""

    exchange_to_account: dict[str, str] = {}
    for strategy_id, account_id, config in strategies:
        exchange = config.local.normalized_exchange()
        previous = exchange_to_account.get(exchange)
        if previous is not None and previous != account_id:
            raise ValueError(
                f"worker cannot host multiple accounts on exchange {exchange!r} "
                f"({previous!r} vs {account_id!r} for strategy {strategy_id!r})"
            )
        exchange_to_account[exchange] = account_id


def extend_shared_account_pool(
    pool: SharedAccountPool,
    config: MarketMakerRuntimeConfig,
    *,
    account_id: str,
    bus: AbstractEventBus,
    options: RuntimeBuildOptions,
) -> None:
    """Attach account resources for one strategy config when not already present."""

    if not options.enable_private_state:
        return
    if account_id in pool.private_reconcilers:
        return

    private_reconciler: PrivateStateReconciler | None = None
    execution_reconciler: ExecutionStateReconciler | None = None

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
        pool.private_state_selections[account_id] = private_selection
        attribution_registry = pool.attribution_for(source.normalized_exchange())
        private_selection.adapter.attribution_registry = attribution_registry
        private_reconciler = PrivateStateReconciler(
            exchange=source.normalized_exchange(),
            adapter=private_selection.adapter,
            bus=bus,
            fill_event_source=private_config.fill_event_source,
            attribution_registry=attribution_registry,
        )
        if not options.enable_execution:
            break
        if not config.execution.for_role(source.role).enabled:
            break
        placement_max_concurrency = 8
        if config.controller is not None:
            placement_max_concurrency = config.controller.placement_max_concurrency
        order_selection = select_order_command_adapter(
            source.normalized_exchange(),
            source.symbol,
            api_token=api_credentials,
            placement_max_concurrency=placement_max_concurrency,
        )
        pool.order_command_selections[account_id] = order_selection
        manager = pool.order_managers.get(source.normalized_exchange())
        if manager is None:
            manager = OrderManager(
                exchange=source.normalized_exchange(),
                command_adapter=order_selection.adapter,
                bus=bus,
                limit_source=select_venue_order_limit_source(source.normalized_exchange()),
                reconcile_config=_execution_reconcile_config(config),
                attribution_registry=attribution_registry,
            )
            pool.order_managers[source.normalized_exchange()] = manager
        execution_reconciler = ExecutionStateReconciler(
            source.normalized_exchange(),
            manager,
            bus,
        )
        break

    if private_reconciler is not None:
        pool.private_reconcilers[account_id] = private_reconciler
    if execution_reconciler is not None:
        pool.execution_reconcilers[account_id] = execution_reconciler


def build_shared_account_pool(
    strategies: tuple[tuple[str, str, MarketMakerRuntimeConfig], ...],
    *,
    bus: AbstractEventBus,
    options: RuntimeBuildOptions,
) -> SharedAccountPool:
    validate_worker_account_exchange_mapping(strategies)
    pool = SharedAccountPool()
    for _strategy_id, account_id, config in strategies:
        extend_shared_account_pool(
            pool,
            config,
            account_id=account_id,
            bus=bus,
            options=options,
        )
    return pool


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
