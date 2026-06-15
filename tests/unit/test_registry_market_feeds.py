from __future__ import annotations

import pytest

from mm.registry.feeds import ingestor_feeds_from_registry, merge_registry_feeds_into_mapping
from mm.registry.memory import MemoryRegistryStore, RegistryNotFoundError
from mm.persistence.service_config import persistence_service_config_from_mapping


def test_upsert_market_feed_requires_registered_exchange() -> None:
    store = MemoryRegistryStore()
    with pytest.raises(RegistryNotFoundError, match="exchange 'nobitex' not found"):
        store.upsert_market_feed(
            exchange="nobitex",
            symbol="USDT_IRT",
            interval_ms=10_000,
        )


def test_registry_market_feeds_crud_and_ingestor_mapping() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    record = store.upsert_market_feed(
        feed_id="nobitex-usdt-irt-10s",
        exchange="nobitex",
        symbol="USDT_IRT",
        interval_ms=10_000,
    )
    assert store.get_market_feed(record.feed_id).symbol == "USDT_IRT"
    feeds = ingestor_feeds_from_registry(store)
    assert len(feeds) == 1
    assert feeds[0].market_event_topic() == "market:events:nobitex:USDT_IRT:10000"


def test_registry_feeds_expand_persistence_topics() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_exchange(exchange="tabdeal")
    store.upsert_market_feed(
        feed_id="nobitex-usdt-irt-10s",
        exchange="nobitex",
        symbol="USDT_IRT",
        interval_ms=10_000,
    )
    store.upsert_market_feed(
        feed_id="tabdeal-usdt-irt-10s",
        exchange="tabdeal",
        symbol="USDT_IRT",
        interval_ms=10_000,
    )
    payload = merge_registry_feeds_into_mapping(
        {
            "load_feeds_from_registry": True,
            "feeds": [{"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10_000}],
            "bus": {"kind": "redis_streams"},
            "persistence": {"enabled": True},
        },
        store,
    )
    config = persistence_service_config_from_mapping(payload)
    topics = config.effective_topics
    assert "market:events:nobitex:USDT_IRT:10000" in topics
    assert "market:events:tabdeal:USDT_IRT:10000" in topics
