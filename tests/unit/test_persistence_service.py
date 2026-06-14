from __future__ import annotations

import pytest

from mm.bus.memory import InMemoryBus
from mm.persistence.service import build_persistence_service, build_persistence_writers
from mm.persistence.service_config import (
    fleet_persistence_topics,
    persistence_service_config_from_mapping,
)
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import PersistenceRuntimeConfig
from mm.runtime.engine import resolve_build_options
from mm.runtime.config import BusRuntimeConfig, ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig


def test_fleet_persistence_topics_expands_feeds_exchanges_and_actors() -> None:
    topics = fleet_persistence_topics(
        feeds=(),
        exchanges=("nobitex", "wallex"),
        actors=("mpc-controller", "btc-maker"),
        joint_frames=(),
    )
    assert "execution:orders:nobitex" in topics
    assert "execution:orders:wallex" in topics
    assert "private:state:nobitex:fills" in topics
    assert "strategy:target_books:mpc-controller" in topics
    assert "strategy:diagnostics:btc-maker" in topics
    assert "ledger:snapshot" in topics
    assert "strategy:actions" in topics


def test_persistence_service_config_derives_topics_from_feeds() -> None:
    config = persistence_service_config_from_mapping(
        {
            "feeds": [{"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10000}],
            "exchanges": ["nobitex"],
            "actors": ["mpc-controller"],
            "bus": {"kind": "redis_streams"},
            "persistence": {"enabled": True, "parquet_root": "/tmp/mm-parquet"},
        }
    )
    topics = config.effective_topics
    assert "market:events:nobitex:USDT_IRT:10000" in topics
    assert "strategy:joint:frames:nobitex:USDT_IRT:10000" in topics
    assert "strategy:target_books:mpc-controller" in topics
    assert config.bus.consumer_group == "persistence"


def test_persistence_service_config_honors_explicit_topic_override() -> None:
    config = persistence_service_config_from_mapping(
        {
            "bus": {"kind": "redis_streams"},
            "persistence": {
                "enabled": True,
                "topics": ["ledger:snapshot", "strategy:actions"],
            },
        }
    )
    assert config.effective_topics == ("ledger:snapshot", "strategy:actions")


def test_persistence_service_config_requires_redis_bus() -> None:
    with pytest.raises(ValueError, match="redis_streams"):
        persistence_service_config_from_mapping(
            {
                "bus": {"kind": "memory"},
                "persistence": {"enabled": True, "parquet_root": "/tmp/mm-parquet"},
            }
        )


def test_build_persistence_writers_requires_enabled_config() -> None:
    writers = build_persistence_writers(
        bus=InMemoryBus(),
        persistence=PersistenceRuntimeConfig(enabled=False),
        topics=("ledger:snapshot",),
    )
    assert writers == ()


def test_build_persistence_writers_creates_parquet_writer(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_PARQUET_ROOT", str(tmp_path))
    writers = build_persistence_writers(
        bus=InMemoryBus(),
        persistence=PersistenceRuntimeConfig(enabled=True),
        topics=("ledger:snapshot",),
    )
    assert len(writers) == 1
    assert type(writers[0]).__name__ == "ParquetEventWriter"


@pytest.mark.asyncio
async def test_build_persistence_service_wires_writers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.persistence.service.load_mm_env", lambda: False)
    monkeypatch.setenv("MM_PARQUET_ROOT", str(tmp_path))
    config = persistence_service_config_from_mapping(
        {
            "feeds": [{"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10000}],
            "exchanges": ["nobitex"],
            "actors": ["mpc-controller", "eth-maker"],
            "bus": {"kind": "redis_streams"},
            "persistence": {"enabled": True},
        }
    )
    service = await build_persistence_service(config, bus=InMemoryBus())
    assert len(service.writers) == 1
    assert "strategy:target_books:eth-maker" in config.effective_topics


def test_consumer_only_mode_disables_persistence() -> None:
    options = RuntimeBuildOptions.for_mode("consumer_only")
    assert options.enable_persistence is False


@pytest.mark.asyncio
async def test_resolve_build_options_bus_consumer_disables_persistence() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
        bus=BusRuntimeConfig(kind="redis_streams"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="test-actor"),
    )
    options = resolve_build_options(config)
    assert options.mode == "consumer_only"
    assert options.enable_persistence is False


def test_multi_strategy_fleet_topics_include_all_actors() -> None:
    config = persistence_service_config_from_mapping(
        {
            "feeds": [
                {"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10000},
                {"exchange": "wallex", "symbol": "BTC_IRT", "interval_ms": 5000},
            ],
            "exchanges": ["nobitex", "wallex"],
            "actors": ["mpc-controller", "btc-maker", "eth-maker"],
            "bus": {"kind": "redis_streams"},
            "persistence": {"enabled": True, "parquet_root": "/tmp/mm-parquet"},
        }
    )
    topics = set(config.effective_topics)
    assert "market:events:nobitex:USDT_IRT:10000" in topics
    assert "market:events:wallex:BTC_IRT:5000" in topics
    assert "strategy:joint:frames:wallex:BTC_IRT:5000" in topics
    assert "strategy:target_books:btc-maker" in topics
    assert "strategy:diagnostics:eth-maker" in topics
    assert "execution:orders:wallex" in topics
