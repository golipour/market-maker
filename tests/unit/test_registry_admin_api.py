from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mm.api.admin import create_admin_app
from mm.bus.memory import InMemoryBus
from mm.registry.memory import MemoryRegistryStore


@pytest.fixture
def registry_client() -> tuple[TestClient, MemoryRegistryStore, InMemoryBus]:
    bus = InMemoryBus()
    store = MemoryRegistryStore()
    store.upsert_exchange(
        exchange="nobitex",
        config={"market_data": {"mode": "rest"}, "private": {"mode": "rest"}},
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
            "controller": {"enabled": False},
        },
    )
    app = create_admin_app(bus=bus, registry_store=store)
    return TestClient(app), store, bus


def _seed_strategy(store: MemoryRegistryStore) -> str:
    store.upsert_platform_settings(bus={"kind": "memory"})
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        symbol="BTC_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
    )
    return "btc-maker"


def test_registry_admin_stores_account_credentials(registry_client) -> None:
    client, store, _bus = registry_client
    store.upsert_exchange(exchange="wallex", config={"private": {"mode": "rest"}})

    response = client.post(
        "/api/admin/registry/accounts",
        json={
            "account_id": "wallex-main",
            "exchange": "wallex",
            "credentials": {
                "auth_mode": "api_key",
                "api_key": "wallex-secret-key",
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["credentials"]["auth_mode"] == "api_key"
    assert body["credentials"]["api_key"] == "wallex-secret-key"
    assert body["credential_storage"] == "stored"

    record = store.get_account("wallex-main")
    assert '"v":2' in record.credential_ref
    assert "wallex-secret-key" in record.credential_ref


def test_registry_admin_crud_and_runtime_config(registry_client) -> None:
    client, store, _bus = registry_client
    strategy_id = _seed_strategy(store)

    listed = client.get("/api/admin/registry/strategies")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["strategy_id"] == strategy_id

    runtime = client.get(f"/api/admin/registry/strategies/{strategy_id}/runtime-config")
    assert runtime.status_code == 200
    body = runtime.json()
    assert body["runtime_config"]["bots"][strategy_id]["symbol"] == "BTC_IRT"
    assert body["resolved_contract"]["bot"]["bot_id"] == strategy_id


def test_registry_strategy_start_publishes_admin_command(registry_client) -> None:
    client, store, bus = registry_client
    strategy_id = _seed_strategy(store)
    published: list[tuple[str, object]] = []
    original_publish = bus.publish

    async def capture_publish(topic: str, event: object) -> None:
        published.append((topic, event))
        await original_publish(topic, event)

    bus.publish = capture_publish  # type: ignore[method-assign]

    response = client.post(f"/api/admin/registry/strategies/{strategy_id}/start")
    assert response.status_code == 200
    assert response.json()["actor_id"] == "btc-maker-actor"
    assert published
    topic, event = published[0]
    assert topic == "admin:commands"
    assert event["command"] == "resume"
    assert event["actor_id"] == "btc-maker-actor"


def test_registry_strategy_stop_publishes_pause(registry_client) -> None:
    client, store, bus = registry_client
    strategy_id = _seed_strategy(store)
    published: list[tuple[str, object]] = []
    original_publish = bus.publish

    async def capture_publish(topic: str, event: object) -> None:
        published.append((topic, event))
        await original_publish(topic, event)

    bus.publish = capture_publish  # type: ignore[method-assign]

    response = client.post(f"/api/admin/registry/strategies/{strategy_id}/stop")
    assert response.status_code == 200
    assert response.json()["actor_id"] == "btc-maker-actor"
    topic, event = published[0]
    assert topic == "admin:commands"
    assert event["command"] == "pause"


def test_runtime_overview_endpoint(registry_client) -> None:
    client, store, _bus = registry_client
    strategy_id = _seed_strategy(store)
    store.upsert_worker_assignment(
        assignment_id="wa-test",
        worker_id="worker-1",
        strategy_ids=[strategy_id],
        lease_ttl_seconds=300,
        status="active",
    )
    response = client.get("/api/admin/runtime/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["strategies"][0]["strategy_id"] == strategy_id
    assert body["workers"][0]["worker_id"] == "worker-1"
    assert body["control"]["connected"] is False
