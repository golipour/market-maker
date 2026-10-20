from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mm.core.models.execution import Action, Fill, Order
from mm.core.models.private import OpenOrder
from mm.execution.order_attribution import (
    AttributionRecord,
    OrderAttributionRegistry,
    _attribution_cache_dir,
    load_attribution_registry,
    wire_client_order_id_for_exchange,
)
from mm.execution.order_attribution_store import MemoryAttributionDatabase, OrderAttributionService
from mm.execution.order_manager import OrderManager
from tests.unit.test_execution_manager import FakeOrderCommandAdapter
from mm.exchanges.nobitex.private_rest import nobitex_wire_client_order_id
from mm.exchanges.nobitex.private_state import NobitexRestPrivateStateAdapter


def test_order_attribution_registry_resolves_exchange_and_client_ids() -> None:
    registry = OrderAttributionRegistry()
    registry.register(
        "nobitex",
        "12345",
        "mpc-controller",
        client_order_id="mpc-controller:buy:0",
        wire_client_order_id=nobitex_wire_client_order_id("mpc-controller:buy:0"),
    )

    assert registry.resolve_actor_id("nobitex", exchange_order_id="12345") == "mpc-controller"
    assert (
        registry.resolve_actor_id(
            "nobitex",
            client_order_id=nobitex_wire_client_order_id("mpc-controller:buy:0") or "",
        )
        == "mpc-controller"
    )


def test_nobitex_fill_normalization_uses_attribution_registry() -> None:
    registry = OrderAttributionRegistry()
    wire_id = nobitex_wire_client_order_id("mpc-controller:buy:0")
    registry.register(
        "nobitex",
        "999",
        "mpc-controller",
        client_order_id="mpc-controller:buy:0",
        wire_client_order_id=wire_id,
    )
    adapter = NobitexRestPrivateStateAdapter("USDT_IRT", api_token="token")
    adapter.attribution_registry = registry

    fill = adapter.normalize_fill(
        {
            "id": "999",
            "type": "buy",
            "clientOrderId": wire_id,
            "matchedAmount": "1.0",
            "averagePrice": "158000",
            "fee": "0",
        },
        timestamp_ms=1000,
    )

    assert fill is not None
    assert fill.actor_id == "mpc-controller"


def test_wire_client_order_id_for_exchange_uses_registration_hook() -> None:
    assert (
        wire_client_order_id_for_exchange("nobitex", "mpc-controller:buy:0")
        == nobitex_wire_client_order_id("mpc-controller:buy:0")
    )
    assert wire_client_order_id_for_exchange("wallex", "actor:buy:0") == "actor:buy:0"


def test_register_does_not_write_attribution_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", str(tmp_path))
    registry = OrderAttributionRegistry()
    registry.register("bitwana", "974646013", "bitwana-controller")

    assert registry.resolve_actor_id("bitwana", exchange_order_id="974646013") == "bitwana-controller"
    assert list(tmp_path.iterdir()) == []


def test_legacy_cache_loads_without_rewriting(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "attribution"
    cache_dir.mkdir()
    path = cache_dir / "bitwana.json"
    path.write_text(
        '{"by_exchange_order_id":{"bitwana:1":"bitwana-controller"},'
        '"by_client_order_id":{"bitwana:bitwana-controller:buy:0":"bitwana-controller"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", str(cache_dir))
    registry = load_attribution_registry("bitwana")
    registry.register("bitwana", "2", "bitwana-controller")

    restored = load_attribution_registry("bitwana")
    assert restored.resolve_actor_id("bitwana", exchange_order_id="1") == "bitwana-controller"
    assert (
        restored.resolve_actor_id("bitwana", client_order_id="bitwana-controller:buy:0")
        == "bitwana-controller"
    )
    assert "2" not in path.read_text(encoding="utf-8")


def test_attribution_cache_dir_prefers_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", str(tmp_path / "custom"))
    assert _attribution_cache_dir() == tmp_path / "custom"


def test_attribution_cache_dir_falls_back_when_data_root_not_writable(monkeypatch) -> None:
    monkeypatch.delenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", raising=False)
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    assert _attribution_cache_dir() == Path("data/attribution")


def test_enrich_fill_preserves_existing_actor_id() -> None:
    registry = OrderAttributionRegistry()
    fill = Fill(
        exchange_order_id="1",
        actor_id="already-set",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="buy",
        price=1.0,
        qty=1.0,
        fee=0.0,
        timestamp_ms=1,
    )
    enriched = registry.enrich_fill(fill)
    assert enriched.actor_id == "already-set"


_FIXED = datetime(2026, 10, 20, tzinfo=timezone.utc)


def test_unchanged_registration_is_not_flushed_again() -> None:
    registry = OrderAttributionRegistry()
    database = MemoryAttributionDatabase()
    service = OrderAttributionService(
        registry,
        database,
        exchange="bitwana",
        clock=lambda: _FIXED,
    )
    registry.register(
        "bitwana",
        "1",
        "bitwana-controller",
        client_order_id="bitwana-controller:buy:0",
    )
    registry.register("bitwana", "1", "bitwana-controller")
    service.flush()
    service.flush()

    assert len(database.rows) == 1
    stored = database.rows[("bitwana", "1")]
    assert stored.actor_id == "bitwana-controller"
    assert stored.client_order_id == "bitwana-controller:buy:0"
    assert stored.closed_at is None


def test_closed_mappings_expire_and_open_mappings_remain() -> None:
    registry = OrderAttributionRegistry()
    database = MemoryAttributionDatabase()
    now = _FIXED
    service = OrderAttributionService(
        registry,
        database,
        exchange="bitwana",
        retention_hours=24,
        clock=lambda: now,
    )
    registry.register("bitwana", "open", "bitwana-controller")
    registry.register("bitwana", "done", "bitwana-controller")
    registry.release("bitwana", "done", closed_at=_FIXED)
    service.flush()

    service._clock = lambda: _FIXED + timedelta(hours=25)
    expired = service.retain_expired()

    assert expired == ["done"]
    assert registry.resolve_actor_id("bitwana", exchange_order_id="done") == ""
    assert registry.resolve_actor_id("bitwana", exchange_order_id="open") == "bitwana-controller"
    assert ("bitwana", "open") in database.rows
    assert ("bitwana", "done") not in database.rows


def test_reregister_reopens_a_closed_mapping() -> None:
    registry = OrderAttributionRegistry()
    database = MemoryAttributionDatabase()
    service = OrderAttributionService(registry, database, exchange="bitwana", clock=lambda: _FIXED)
    registry.register("bitwana", "1", "bitwana-controller")
    registry.release("bitwana", "1", closed_at=_FIXED)
    registry.register("bitwana", "1", "bitwana-controller")
    service.flush()

    assert database.rows[("bitwana", "1")].closed_at is None
    service._clock = lambda: _FIXED + timedelta(hours=25)
    assert service.retain_expired() == []
    assert registry.resolve_actor_id("bitwana", exchange_order_id="1") == "bitwana-controller"


def test_failed_flush_keeps_the_pending_record() -> None:
    class FailingDatabase(MemoryAttributionDatabase):
        def apply(self, records) -> None:
            raise RuntimeError("db down")

    registry = OrderAttributionRegistry()
    database = FailingDatabase()
    service = OrderAttributionService(registry, database, exchange="bitwana", clock=lambda: _FIXED)
    registry.register("bitwana", "1", "bitwana-controller")
    service.flush()
    assert database.rows == {}

    database.apply = MemoryAttributionDatabase.apply.__get__(database, FailingDatabase)
    service.flush()
    assert database.rows[("bitwana", "1")].actor_id == "bitwana-controller"


@pytest.mark.asyncio
async def test_empty_database_imports_legacy_cache_as_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", str(tmp_path))
    (tmp_path / "bitwana.json").write_text(
        '{"by_exchange_order_id":{"bitwana:1":"bitwana-controller"},'
        '"by_client_order_id":{"bitwana:bitwana-controller:buy:0":"bitwana-controller"}}',
        encoding="utf-8",
    )
    registry = OrderAttributionRegistry()
    database = MemoryAttributionDatabase()
    service = OrderAttributionService(registry, database, exchange="bitwana", clock=lambda: _FIXED)
    await service.start()
    await service.stop()

    assert registry.resolve_actor_id("bitwana", exchange_order_id="1") == "bitwana-controller"
    assert (
        registry.resolve_actor_id("bitwana", client_order_id="bitwana-controller:buy:0")
        == "bitwana-controller"
    )
    assert database.rows[("bitwana", "1")].closed_at == _FIXED

    restored = OrderAttributionRegistry()
    again = OrderAttributionService(restored, database, exchange="bitwana", clock=lambda: _FIXED)
    await again.start()
    await again.stop()
    assert len(database.rows) == 1
    assert restored.resolve_actor_id("bitwana", exchange_order_id="1") == "bitwana-controller"


@pytest.mark.asyncio
async def test_empty_store_seeds_recent_placements_as_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MM_ORDER_ATTRIBUTION_CACHE_DIR", str(tmp_path))
    class SeededDatabase(MemoryAttributionDatabase):
        def load_recent_placements(self, exchange: str, placed_after: datetime):
            del placed_after
            return [
                AttributionRecord(
                    exchange=exchange,
                    exchange_order_id="seed-1",
                    actor_id="bitwana-controller",
                    client_order_id="bitwana-controller:buy:0",
                )
            ]

    registry = OrderAttributionRegistry()
    database = SeededDatabase()
    service = OrderAttributionService(registry, database, exchange="bitwana", clock=lambda: _FIXED)
    await service.start()
    await service.stop()

    assert registry.resolve_actor_id("bitwana", exchange_order_id="seed-1") == "bitwana-controller"
    assert database.rows[("bitwana", "seed-1")].closed_at == _FIXED


@pytest.mark.asyncio
async def test_terminal_order_closes_attribution_and_keeps_it_resolvable() -> None:
    registry = OrderAttributionRegistry()
    manager = OrderManager(
        "nobitex",
        FakeOrderCommandAdapter(),
        clock_ms=lambda: 1_000,
        attribution_registry=registry,
    )
    await manager.sync(
        [
            Action(
                actor_id="mpc-controller",
                orders=(Order("nobitex", "USDT_IRT", "buy", 100.0, 1.0, client_order_id="mpc-controller:buy:0"),),
            )
        ]
    )
    assert registry.resolve_actor_id("nobitex", exchange_order_id="ex-1") == "mpc-controller"

    manager.apply_open_order(
        OpenOrder(
            exchange_order_id="ex-1",
            actor_id="mpc-controller",
            order=Order("nobitex", "USDT_IRT", "buy", 100.0, 1.0),
            remaining_qty=0.0,
            filled_qty=1.0,
            status="filled",
            timestamp_ms=1_000,
        )
    )

    assert registry.resolve_actor_id("nobitex", exchange_order_id="ex-1") == "mpc-controller"
    stored = registry._records["nobitex:ex-1"]
    assert stored.closed_at == datetime.fromtimestamp(1.0, timezone.utc)
