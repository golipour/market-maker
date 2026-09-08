from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.registry.memory import MemoryRegistryStore
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
from mm.worker.account_pool import build_shared_account_pool, validate_worker_account_exchange_mapping
from mm.worker.supervisor import StrategyBackoffConfig, StrategySupervisor
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
    assert worker.control_plane is not None
    assert worker.control_plane.state.paused_actors == {
        "btc-maker-actor",
        "usdt-maker-actor",
    }
    assert all(supervisor._rebuild_slot is not None for supervisor in worker.supervisors)


@pytest.mark.asyncio
async def test_build_worker_restores_last_control_pause_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setattr(
        "mm.worker.worker.load_latest_runtime_control_state",
        lambda _url: {
            "type": "RuntimeControlState",
            "actors": ["btc-maker-actor", "usdt-maker-actor"],
            "paused_actors": ["btc-maker-actor"],
        },
    )
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)

    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=InMemoryBus(),
    )

    assert worker.control_plane is not None
    assert worker.control_plane.state.paused_actors == {"btc-maker-actor"}
    assert worker.control_plane.managed_actors == {"btc-maker-actor", "usdt-maker-actor"}


@pytest.mark.asyncio
async def test_worker_start_publishes_boot_paused_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)
    bus = InMemoryBus()
    events: list[dict] = []
    original_publish = bus.publish

    async def capture(topic: str, event: object) -> None:
        if topic == "admin:events" and isinstance(event, dict):
            events.append(event)
        await original_publish(topic, event)

    bus.publish = capture  # type: ignore[method-assign]
    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=bus,
    )
    await worker.start()
    try:
        assert events
        assert events[-1]["paused_actors"] == ["btc-maker-actor", "usdt-maker-actor"]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_resume_reloads_strategy_slot_from_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)
    bus = InMemoryBus()
    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=bus,
    )
    await worker.start()
    try:
        assert "btc-maker-actor" in worker.control_plane.state.paused_actors
        rebuilds = 0
        supervisor = next(
            item
            for item in worker.supervisors
            if item.slot.strategy_id == "btc-maker"
        )
        original_rebuild = supervisor._rebuild_slot

        def counting_rebuild():
            nonlocal rebuilds
            rebuilds += 1
            return original_rebuild()

        supervisor._rebuild_slot = counting_rebuild
        await worker.control_plane.apply(
            {
                "command": "resume",
                "actor_id": "btc-maker-actor",
                "command_id": "test-resume",
            }
        )
        assert "btc-maker-actor" not in worker.control_plane.state.paused_actors
        assert rebuilds >= 1
    finally:
        await worker.stop()


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


class _FakeSlot:
    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id
        self.account_id = "account"
        self.starts = 0
        self.stops = 0
        self.started = asyncio.Event()

    async def start(self) -> None:
        self.starts += 1
        self.started.set()

    async def stop(self) -> None:
        self.stops += 1

    def monitor_tasks(self) -> list[object]:
        return []


@pytest.mark.asyncio
async def test_supervisor_stops_previous_slot_when_rebuilding() -> None:
    first = _FakeSlot("placeholder")
    rebuilt = _FakeSlot("live")
    rebuilds: list[_FakeSlot] = []

    def rebuild() -> _FakeSlot:
        rebuilds.append(rebuilt)
        return rebuilt

    supervisor = StrategySupervisor(
        slot=first,
        backoff=StrategyBackoffConfig(initial_s=0.01, max_s=0.01),
        rebuild_slot=rebuild,
    )
    await supervisor.start()
    try:
        await asyncio.wait_for(rebuilt.started.wait(), timeout=1.0)
        assert rebuilds == [rebuilt]
        assert first.stops >= 1
        assert first.starts == 0
        assert supervisor.slot is rebuilt
    finally:
        await supervisor.stop()


def _subscriber_count(bus: InMemoryBus, topic: str) -> int:
    return len(bus._subscribers.get(topic, ()))


@pytest.mark.asyncio
async def test_worker_start_does_not_leak_orphan_subscribers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)
    bus = InMemoryBus()
    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=bus,
    )
    slot = next(item for item in worker.slots if item.strategy_id == "btc-maker")
    market_topic = slot.config.local.market_event_topic(slot.config.interval_ms)
    joint_topic = slot.config.joint_frame_topic
    assert _subscriber_count(bus, market_topic) == 0
    assert _subscriber_count(bus, joint_topic) == 0

    await worker.start()
    try:
        live = next(
            item for item in worker.supervisors if item.slot.strategy_id == "btc-maker"
        )
        await asyncio.sleep(0.05)
        assert live.slot.frame_assembler is not None
        assert _subscriber_count(bus, market_topic) == 1
        assert _subscriber_count(bus, joint_topic) == 1

        original_rebuild = live._rebuild_slot
        rebuilds = 0

        def counting_rebuild():
            nonlocal rebuilds
            rebuilds += 1
            return original_rebuild()

        live._rebuild_slot = counting_rebuild
        await worker.control_plane.apply(
            {
                "command": "resume",
                "actor_id": "btc-maker-actor",
                "command_id": "test-resume-no-leak",
            }
        )
        assert rebuilds >= 1
        assert _subscriber_count(bus, market_topic) == 1
        assert _subscriber_count(bus, joint_topic) == 1
    finally:
        await worker.stop()
