from __future__ import annotations

import pytest

from mm.registry.feeds import ingestor_feeds_from_registry, merge_registry_feeds_into_mapping
from mm.registry.market_data import (
    DEFAULT_MARKET_DATA_POLL_INTERVAL_MS,
    market_data_settings_from_exchange,
    market_data_settings_from_mapping,
    market_source_as_dict,
)
from mm.registry.memory import MemoryRegistryStore, RegistryNotFoundError
from mm.persistence.service_config import persistence_service_config_from_mapping


def test_market_data_settings_from_exchange_config() -> None:
    settings = market_data_settings_from_mapping(
        {
            "market_data": {
                "mode": "rest",
                "poll_interval_ms": 2000,
                "depth_limit": 100,
                "trades_limit": 200,
            }
        }
    )
    assert settings["poll_interval_ms"] == 2000
    assert settings["depth_limit"] == 100
    assert settings["trades_limit"] == 200


def test_market_data_settings_use_defaults_when_missing() -> None:
    settings = market_data_settings_from_mapping({})
    assert settings["poll_interval_ms"] == DEFAULT_MARKET_DATA_POLL_INTERVAL_MS


def test_upsert_market_source_requires_registered_exchange() -> None:
    store = MemoryRegistryStore()
    with pytest.raises(RegistryNotFoundError, match="exchange 'nobitex' not found"):
        store.upsert_market_source(
            exchange="nobitex",
            symbol="USDT_IRT",
        )


def test_upsert_market_feed_requires_market_source() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    with pytest.raises(RegistryNotFoundError, match="market source"):
        store.upsert_market_feed(
            source_id="missing",
            interval_ms=10_000,
        )


def test_registry_market_sources_inherit_exchange_market_data() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(
        exchange="nobitex",
        config={
            "market_data": {
                "mode": "rest",
                "poll_interval_ms": 2000,
                "depth_limit": 100,
                "trades_limit": 200,
            }
        },
    )
    source = store.upsert_market_source(
        source_id="nobitex-usdt-irt",
        exchange="nobitex",
        symbol="USDT_IRT",
    )
    resolved = market_data_settings_from_exchange(store.get_exchange(source.exchange))
    assert resolved["poll_interval_ms"] == 2000

    payload = market_source_as_dict(store.get_exchange(source.exchange), source)
    assert payload["poll_interval_ms"] == 2000
    assert payload["depth_limit"] == 100

    record = store.upsert_market_feed(
        feed_id="nobitex-usdt-irt-10s",
        source_id=source.source_id,
        interval_ms=10_000,
    )
    assert store.get_market_feed(record.feed_id).poll_interval_ms == 2000
    feeds = ingestor_feeds_from_registry(store)
    assert len(feeds) == 1
    assert feeds[0].market_event_topic() == "market:events:nobitex:USDT_IRT:10000"
    assert feeds[0].poll_interval_ms == 2000


def test_registry_feeds_expand_persistence_topics() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_exchange(exchange="tabdeal")
    nobitex = store.upsert_market_source(exchange="nobitex", symbol="USDT_IRT")
    tabdeal = store.upsert_market_source(exchange="tabdeal", symbol="USDT_IRT")
    store.upsert_market_feed(source_id=nobitex.source_id, interval_ms=10_000)
    store.upsert_market_feed(source_id=tabdeal.source_id, interval_ms=10_000)
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


def test_registry_feeds_exclude_exchanges() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_exchange(exchange="kucoin")
    nobitex = store.upsert_market_source(exchange="nobitex", symbol="USDT_IRT")
    kucoin = store.upsert_market_source(exchange="kucoin", symbol="BTC_USDT")
    store.upsert_market_feed(source_id=nobitex.source_id, interval_ms=10_000)
    store.upsert_market_feed(source_id=kucoin.source_id, interval_ms=10_000)
    feeds = ingestor_feeds_from_registry(store, exclude_exchanges=frozenset({"kucoin"}))
    assert len(feeds) == 1
    assert feeds[0].exchange == "nobitex"


def test_merge_registry_feeds_honors_exclude_exchanges() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(exchange="nobitex")
    store.upsert_exchange(exchange="kucoin")
    nobitex = store.upsert_market_source(exchange="nobitex", symbol="USDT_IRT")
    kucoin = store.upsert_market_source(exchange="kucoin", symbol="BTC_USDT")
    store.upsert_market_feed(source_id=nobitex.source_id, interval_ms=10_000)
    store.upsert_market_feed(source_id=kucoin.source_id, interval_ms=10_000)
    payload = merge_registry_feeds_into_mapping(
        {"load_feeds_from_registry": True, "exclude_exchanges": ["kucoin"], "feeds": []},
        store,
    )
    assert len(payload["feeds"]) == 1
    assert payload["feeds"][0]["exchange"] == "nobitex"


def test_multiple_intervals_share_one_market_source() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(
        exchange="nobitex",
        config={"market_data": {"poll_interval_ms": 2000}},
    )
    source = store.upsert_market_source(exchange="nobitex", symbol="USDT_IRT")
    for interval_ms in (10_000, 20_000, 30_000):
        store.upsert_market_feed(source_id=source.source_id, interval_ms=interval_ms)
    feeds = ingestor_feeds_from_registry(store)
    assert len(feeds) == 3
    assert {feed.interval_ms for feed in feeds} == {10_000, 20_000, 30_000}
    assert all(feed.poll_interval_ms == 2000 for feed in feeds)
