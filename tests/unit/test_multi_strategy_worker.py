from __future__ import annotations

import pytest

from mm.bus.memory import InMemoryBus
from mm.registry.memory import MemoryRegistryStore
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
from mm.worker.account_pool import build_shared_account_pool, validate_worker_account_exchange_mapping
from mm.worker.worker import build_worker_components, load_worker_strategies
from mm.worker.worker_config import worker_config_from_mapping


def _seed_worker_registry(store: MemoryRegistryStore) -> tuple[str, tuple[str, ...]]:
    store.upsert_platform_settings(
        bus={"kind": "redis_streams", "consumer_group": "mm-worker"},
        persistence={"enabled": False},
        observability={"metrics_enabled": False},
    )
    store.upsert_exchange(
        exchange="nobitex",
        config={
            "market_data": {"mode": "rest", "market_data_source": "bus"},
            "private": {"mode": "rest", "poll_interval_ms": 5000},
            "fees": {"maker_fee": 0.0001},
        },
    )
    store.upsert_account(
        account_id="nobitex-main",
        exchange="nobitex",
        credential_ref="NOBITEX_API_TOKEN",
    )
    store.upsert_preset(
        preset_id="test_preset",
        config={
            "interval_ms": 5000,
            "liquidity_sourcing": {"mode": "off"},
            "controller": {"enabled": True, "dry_run": True, "actor_id": "placeholder"},
        },
    )
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        symbol="BTC_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
    )
    store.upsert_strategy(
        strategy_id="usdt-maker",
        actor_id="usdt-maker-actor",
        account_id="nobitex-main",
        symbol="USDT_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
    )
    assignment = store.upsert_worker_assignment(
        worker_id="worker-1",
        strategy_ids=["btc-maker", "usdt-maker"],
    )
    return assignment.assignment_id, ("btc-maker", "usdt-maker")


@pytest.mark.asyncio
async def test_build_worker_components_loads_registry_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    assignment_id, strategy_ids = _seed_worker_registry(store)

    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=InMemoryBus(),
    )

    assert len(worker.strategies) == 2
    assert {item.strategy_id for item in worker.strategies} == set(strategy_ids)
    assert len(worker.supervisors) == 2
    assert len(worker.account_pool.private_reconcilers) == 1
    assert "nobitex" in worker.account_pool.order_managers


@pytest.mark.asyncio
async def test_build_worker_components_single_mode_runs_one_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)

    worker = await build_worker_components(
        worker_config_from_mapping(
            {
                "worker_id": "worker-1",
                "assignment_id": assignment_id,
                "mode": "single",
            }
        ),
        store=store,
        bus=InMemoryBus(),
    )

    assert len(worker.strategies) == 1
    assert worker.strategies[0].strategy_id == "btc-maker"


@pytest.mark.asyncio
async def test_build_worker_components_rejects_local_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    store = MemoryRegistryStore()
    store.upsert_platform_settings(bus={"kind": "redis_streams"})
    store.upsert_exchange(
        exchange="nobitex",
        config={"market_data": {"mode": "rest", "market_data_source": "local"}},
    )
    store.upsert_account(account_id="nobitex-main", exchange="nobitex", credential_ref="NOBITEX_API_TOKEN")
    store.upsert_preset(
        preset_id="test_preset",
        config={"interval_ms": 5000, "controller": {"enabled": True, "actor_id": "x"}},
    )
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        symbol="BTC_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
    )
    assignment = store.upsert_worker_assignment(worker_id="worker-1", strategy_ids=["btc-maker"])

    with pytest.raises(ValueError, match="worker requires 'bus'"):
        await build_worker_components(
            worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment.assignment_id}),
            store=store,
            bus=InMemoryBus(),
        )


def test_validate_worker_account_exchange_mapping_rejects_multiple_accounts() -> None:
    config_a = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="a"),
    )
    config_b = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="ETH_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="b"),
    )
    with pytest.raises(ValueError, match="multiple accounts"):
        validate_worker_account_exchange_mapping(
            (
                ("s1", "account-a", config_a),
                ("s2", "account-b", config_b),
            )
        )


@pytest.mark.asyncio
async def test_shared_account_pool_reuses_resources_for_same_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    _seed_worker_registry(store)
    strategies = load_worker_strategies(store, ("btc-maker", "usdt-maker"))
    triples = tuple((item.strategy_id, item.account_id, item.config) for item in strategies)
    pool = build_shared_account_pool(
        triples,
        bus=InMemoryBus(),
        options=RuntimeBuildOptions.for_mode("consumer_only"),
    )
    assert len(pool.private_reconcilers) == 1
    assert len(pool.order_managers) == 1


def test_registry_get_active_worker_assignment() -> None:
    store = MemoryRegistryStore()
    store.upsert_worker_assignment(worker_id="worker-1", strategy_ids=["btc-maker"])
    assignment = store.get_active_worker_assignment("worker-1")
    assert assignment.worker_id == "worker-1"
