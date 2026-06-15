from __future__ import annotations

from mm.registry.memory import MemoryRegistryStore
from mm.worker.wallet_resolve import resolve_strategy_wallet


def test_resolve_strategy_wallet_prefers_strategy_wallet_id() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_account(account_id="nobitex-main", exchange="nobitex", credential_ref="TOKEN")
    store.upsert_preset(preset_id="p1", config={"controller": {"enabled": True}})
    wallet = store.upsert_allocation(
        allocation_id="pool-a",
        account_id="nobitex-main",
        allocated_quote=1.0,
        allocated_base=1.0,
    )
    store.upsert_strategy(
        strategy_id="s1",
        actor_id="actor-1",
        account_id="nobitex-main",
        symbol="USDT_IRT",
        preset_id="p1",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        wallet_id=wallet.allocation_id,
        ledger_source="virtual",
    )
    strategy = store.get_strategy("s1")
    resolved = resolve_strategy_wallet(store, strategy)
    assert resolved is not None
    assert resolved.allocation_id == "pool-a"
    assert resolved.actor_id == "actor-1"


def test_resolve_strategy_wallet_returns_none_for_physical_ledger() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_account(account_id="nobitex-main", exchange="nobitex", credential_ref="TOKEN")
    store.upsert_preset(preset_id="p1", config={"controller": {"enabled": True}})
    wallet = store.upsert_allocation(
        allocation_id="pool-a",
        account_id="nobitex-main",
    )
    store.upsert_strategy(
        strategy_id="s1",
        actor_id="actor-1",
        account_id="nobitex-main",
        symbol="USDT_IRT",
        preset_id="p1",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        wallet_id=wallet.allocation_id,
        ledger_source="physical",
    )
    strategy = store.get_strategy("s1")
    assert resolve_strategy_wallet(store, strategy) is None
