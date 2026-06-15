"""Resolve registry wallets for worker strategies."""

from __future__ import annotations

from mm.registry.memory import RegistryNotFoundError
from mm.registry.models import AllocationRecord, StrategyRecord
from mm.registry.store import RegistryStore


def resolve_strategy_wallet(
    store: RegistryStore,
    strategy: StrategyRecord,
) -> AllocationRecord | None:
    """Return the virtual wallet for a strategy when ledger_source is virtual."""

    if strategy.ledger_source != "virtual":
        return None
    wallet_id = (strategy.wallet_id or "").strip()
    if wallet_id:
        return store.get_allocation(wallet_id)
    try:
        return store.get_allocation_for_strategy(strategy.strategy_id)
    except RegistryNotFoundError:
        return None


def load_worker_virtual_allocations(
    store: RegistryStore,
    strategy_ids: tuple[str, ...],
) -> tuple[AllocationRecord, ...]:
    """Collect distinct virtual-wallet allocations for active worker strategies."""

    by_actor: dict[str, AllocationRecord] = {}
    for strategy_id in strategy_ids:
        strategy = store.get_strategy(strategy_id)
        if strategy.status != "active":
            continue
        allocation = resolve_strategy_wallet(store, strategy)
        if allocation is None or allocation.status != "active":
            continue
        actor_id = strategy.actor_id.strip()
        if allocation.actor_id != actor_id:
            allocation = AllocationRecord(
                allocation_id=allocation.allocation_id,
                strategy_id=allocation.strategy_id,
                actor_id=actor_id,
                account_id=allocation.account_id,
                quote_asset=allocation.quote_asset,
                base_asset=allocation.base_asset,
                allocated_quote=allocation.allocated_quote,
                allocated_base=allocation.allocated_base,
                halt_on_breach=allocation.halt_on_breach,
                status=allocation.status,
                created_at=allocation.created_at,
                updated_at=allocation.updated_at,
            )
        by_actor[actor_id] = allocation
    return tuple(by_actor.values())
