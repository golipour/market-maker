from __future__ import annotations

import pytest

from mm.bus.factory import build_event_bus
from mm.core.events import decode_envelope, encode_envelope, envelope_event
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.runtime.config import BusRuntimeConfig, runtime_config_from_mapping


def _event() -> MarketEvent:
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1,
        t_end_ms=1001,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=1000,
            bids=(),
            asks=(),
            is_snapshot=True,
        ),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=0.0,
    )


def test_event_envelope_round_trips_market_event() -> None:
    envelope = envelope_event("market:events:nobitex:BTC_IRT:1000", _event(), run_id="run-1")

    decoded = decode_envelope(encode_envelope(envelope))

    assert decoded.schema == "mm.event"
    assert decoded.version == 1
    assert decoded.run_id == "run-1"
    assert decoded.exchange == "nobitex"
    assert decoded.symbol == "BTC_IRT"
    assert decoded.payload["t_end_ms"] == 1001


@pytest.mark.asyncio
async def test_memory_bus_factory_preserves_publish_subscribe() -> None:
    bus = build_event_bus(BusRuntimeConfig(kind="memory"))
    queue = bus.subscribe_queue("strategy:actions")

    await bus.publish("strategy:actions", {"type": "Action", "actor_id": "a1"})

    assert await queue.get() == {"type": "Action", "actor_id": "a1"}


def test_runtime_config_parses_bus_persistence_and_observability() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"exchange": "nobitex", "symbol": "BTC_IRT"},
            "bus": {"kind": "redis_streams", "stream_prefix": "test-mm", "max_stream_length": 10},
            "persistence": {"enabled": True, "topics": ["strategy:controller"], "batch_size": 5},
            "observability": {"run_id": "run-1", "otel_enabled": True},
        }
    )

    assert config.bus.kind == "redis_streams"
    assert config.bus.stream_prefix == "test-mm"
    assert config.persistence.enabled is True
    assert config.persistence.topics == ("strategy:controller",)
    assert config.observability.run_id == "run-1"
    assert config.observability.otel_enabled is True
