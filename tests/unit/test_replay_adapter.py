from __future__ import annotations

import asyncio
import json

import pytest

from mm.bus.factory import build_event_bus
from mm.core.events import envelope_event
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.ingestion.replay_adapter import ReplayMarketDataAdapter, market_event_from_payload
from mm.runtime.config import BusRuntimeConfig


def _minimal_market_event() -> MarketEvent:
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


def test_market_event_from_payload_round_trip_dict() -> None:
    ev = _minimal_market_event()
    from mm.core.events import to_jsonable

    d = to_jsonable(ev)
    out = market_event_from_payload(d)
    assert out is not None
    assert out.exchange == ev.exchange
    assert out.t_end_ms == ev.t_end_ms


@pytest.mark.asyncio
async def test_replay_adapter_publishes_market_topic(tmp_path) -> None:
    topic = "market:events:nobitex:BTC_IRT:1000"
    env = envelope_event(topic, _minimal_market_event())
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps(env.as_dict(), ensure_ascii=False) + "\n", encoding="utf-8")

    bus = build_event_bus(BusRuntimeConfig(kind="memory"))
    queue = bus.subscribe_queue(topic)
    adapter = ReplayMarketDataAdapter(bus, path)
    n_m, n_p = await adapter.publish_all()
    assert n_m == 1
    assert n_p == 0
    got = await asyncio.wait_for(queue.get(), timeout=2.0)
    assert isinstance(got, MarketEvent)
    assert got.symbol == "BTC_IRT"
