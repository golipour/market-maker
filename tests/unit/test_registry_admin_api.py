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
