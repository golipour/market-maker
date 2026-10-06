"""Shared exchange-account resources for multi-strategy workers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

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


CredentialStatus = Literal["resolved", "missing", "auth_failed", "skipped"]


@dataclass(slots=True)
class AccountAttachmentResult:
    """Per-account outcome of attaching a strategy to the shared pool."""

    account_id: str
    exchange: str
    status: CredentialStatus
    auth_mode: str
    credential_kind: str
    private_attached: bool
    execution_attached: bool
    error: str = ""


@dataclass(slots=True)
class SharedAccountPool:
    """Account-scoped private state and execution resources shared across strategies."""

    order_managers: dict[str, OrderManager] = field(default_factory=dict)
    private_reconcilers: dict[str, PrivateStateReconciler] = field(default_factory=dict)
    execution_reconcilers: dict[str, ExecutionStateReconciler] = field(default_factory=dict)
    private_state_selections: dict[str, PrivateStateAdapterSelection] = field(default_factory=dict)
    order_command_selections: dict[str, OrderCommandAdapterSelection] = field(default_factory=dict)
    order_attribution: dict[str, OrderAttributionRegistry] = field(default_factory=dict)
    attachment_results: dict[str, AccountAttachmentResult] = field(default_factory=dict)

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
) -> AccountAttachmentResult:
    """Attach account resources for one strategy config when not already present.

    Returns an :class:`AccountAttachmentResult` describing the outcome so the
    worker can publish a ``WorkerCapability`` event. Callers must not silently
    skip on missing credentials — the result is the only thing the admin API
    uses to decide whether an actor has the resources it needs to trade.
    """

    exchange = config.local.normalized_exchange()

    if not options.enable_private_state:
        result = AccountAttachmentResult(
            account_id=account_id,
            exchange=exchange,
            status="skipped",
            auth_mode="",
            credential_kind="",
            private_attached=False,
            execution_attached=False,
            error="private_state_disabled",
        )
        pool.attachment_results[account_id] = result
        return result
    if account_id in pool.private_reconcilers:
        return pool.attachment_results[account_id]

    private_reconciler: PrivateStateReconciler | None = None
    execution_reconciler: ExecutionStateReconciler | None = None
    credential_status: CredentialStatus = "missing"
    auth_mode = ""
    credential_kind = ""
    error_message = ""
    last_exchange = exchange

    for source in config.sources:
        last_exchange = source.normalized_exchange()
        private_config = config.private_config_for(source.role)
        if private_config is None:
            credential_status = "skipped"
            error_message = "no_private_config"
            _log_attachment_skip(account_id, source.normalized_exchange(), credential_status, error_message)
            continue
        api_credentials = private_config.resolved_api_credentials(source.normalized_exchange())
        if not api_credentials:
            credential_status = "missing"
            error_message = "no_resolved_credentials"
            _log_attachment_skip(account_id, source.normalized_exchange(), credential_status, error_message)
            continue
        auth_mode = private_config.resolved_auth_mode(source.normalized_exchange())
        credential_kind = type(api_credentials).__name__ if not isinstance(api_credentials, str) else "token"
        credential_status = "resolved"
        print(
            f"[account-pool] {account_id} {source.normalized_exchange()} "
            f"private auth_mode={auth_mode} credential={credential_kind}",
            flush=True,
        )
        try:
            private_selection = select_private_state_adapter(
                source.normalized_exchange(),
                source.symbol,
                mode=private_config.mode,
                api_token=api_credentials,
                poll_interval_ms=private_config.poll_interval_ms,
            )
        except Exception as exc:
            credential_status = "auth_failed"
            error_message = f"{type(exc).__name__}: {exc}"
            _log_attachment_skip(account_id, source.normalized_exchange(), credential_status, error_message)
            continue
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
        try:
            order_selection = select_order_command_adapter(
                source.normalized_exchange(),
                source.symbol,
                api_token=api_credentials,
                placement_max_concurrency=placement_max_concurrency,
            )
        except Exception as exc:
            credential_status = "auth_failed"
            error_message = f"order_adapter_init: {type(exc).__name__}: {exc}"
            private_reconciler = None
            break
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

    if private_reconciler is not None and credential_status == "resolved":
        pool.private_reconcilers[account_id] = private_reconciler
    if execution_reconciler is not None:
        pool.execution_reconcilers[account_id] = execution_reconciler

    result = AccountAttachmentResult(
        account_id=account_id,
        exchange=last_exchange,
        status=credential_status,
        auth_mode=auth_mode,
        credential_kind=credential_kind,
        private_attached=private_reconciler is not None,
        execution_attached=execution_reconciler is not None,
        error=error_message,
    )
    pool.attachment_results[account_id] = result
    return result


def _log_attachment_skip(account_id: str, exchange: str, status: str, error: str) -> None:
    """Make a dropped private/execution source visible instead of skipping it silently.

    A source that never resolves credentials leaves the actor running with an empty
    private-state store, which surfaces downstream as fabricated zero balances rather
    than as the credential fault that actually caused it.
    """

    print(
        f"[account-pool] {account_id} {exchange} private NOT attached "
        f"status={status} error={error}",
        flush=True,
    )


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


def summarise_capability(
    *,
    worker_id: str,
    assignment_id: str,
    pool: SharedAccountPool,
    expected_actors: tuple[str, ...],
    expected_accounts: tuple[str, ...],
    strategy_to_account: dict[str, str],
) -> dict:
    """Build a ``WorkerCapability`` payload from current pool state.

    The admin API persists this and the cluster health monitor uses it to
    detect missing or misconfigured actors before they become silent stalls.
    """

    attachments = [
        {
            "account_id": r.account_id,
            "exchange": r.exchange,
            "status": r.status,
            "auth_mode": r.auth_mode,
            "credential_kind": r.credential_kind,
            "private_attached": r.private_attached,
            "execution_attached": r.execution_attached,
            "error": r.error,
        }
        for r in pool.attachment_results.values()
    ]
    account_status = {r.account_id: r.status for r in pool.attachment_results.values()}

    actor_capability: list[dict] = []
    for actor_id in expected_actors:
        account_id = strategy_to_account.get(actor_id, "")
        status = account_status.get(account_id, "missing")
        actor_capability.append(
            {
                "actor_id": actor_id,
                "account_id": account_id,
                "status": status,
                "tradeable": status == "resolved",
            }
        )

    return {
        "type": "WorkerCapability",
        "payload_schema": "mm.worker_capability",
        "payload_version": 1,
        "worker_id": worker_id,
        "assignment_id": assignment_id,
        "expected_actors": list(expected_actors),
        "expected_accounts": list(expected_accounts),
        "shared_exchanges": list(pool.order_managers.keys()),
        "attachments": attachments,
        "actors": actor_capability,
        "ts_ms": int(time.time() * 1000),
    }


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
