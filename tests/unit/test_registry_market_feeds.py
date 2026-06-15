from __future__ import annotations

import pytest

from mm.registry.feeds import ingestor_feeds_from_registry
from mm.registry.memory import MemoryRegistryStore, RegistryNotFoundError


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
