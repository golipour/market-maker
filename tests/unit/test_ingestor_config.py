from __future__ import annotations

import json

import pytest

from mm.runtime.ingestor_config import ingestor_config_from_mapping, load_ingestor_config


def test_ingestor_config_from_mapping_allows_inline_feed_without_exchanges_block() -> None:
    config = ingestor_config_from_mapping(
        {
            "feeds": [
                {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "interval_ms": 1000,
                    "mode": "rest",
                }
            ],
            "bus": {"kind": "redis_streams"},
        }
    )
    assert config.feeds[0].mode == "rest"


def test_ingestor_config_from_mapping_requires_redis_bus() -> None:
    with pytest.raises(ValueError, match="redis_streams"):
        ingestor_config_from_mapping(
            {
                "feeds": [{"exchange": "nobitex", "symbol": "BTC_IRT", "interval_ms": 5000}],
                "bus": {"kind": "memory"},
            }
        )


def test_ingestor_config_from_mapping_merges_exchange_market_data() -> None:
    config = ingestor_config_from_mapping(
        {
            "feeds": [{"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10000}],
            "exchanges": {
                "nobitex": {
                    "market_data": {
                        "mode": "websocket",
                        "depth_limit": 50,
                        "poll_interval_ms": 1500,
                    }
                }
            },
            "bus": {"kind": "redis_streams"},
        }
    )
    feed = config.feeds[0]
    assert feed.mode == "websocket"
    assert feed.depth_limit == 50
    assert feed.poll_interval_ms == 1500
    assert feed.market_event_topic() == "market:events:nobitex:USDT_IRT:10000"


def test_load_ingestor_example_config() -> None:
    config = load_ingestor_config("mm/configs/market_maker.ingestor.example.json")
    assert config.bus.kind == "redis_streams"
    assert len(config.feeds) == 1
    assert config.as_contract()["feeds"][0]["exchange"] == "nobitex"
