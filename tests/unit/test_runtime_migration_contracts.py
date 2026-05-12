from __future__ import annotations

import pytest

from mm.bus.factory import build_event_bus
from mm.core.events import decode_envelope, encode_envelope, envelope_event
from mm.core.models.execution import Action
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.core.models.strategy import StrategyDecision
from mm.runtime.config import BusRuntimeConfig, runtime_config_from_mapping
from mm.strategies import mpc_engine


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
            "persistence": {"enabled": True, "topics": ["strategy:decisions"], "batch_size": 5},
            "observability": {"run_id": "run-1", "otel_enabled": True},
        }
    )

    assert config.bus.kind == "redis_streams"
    assert config.bus.stream_prefix == "test-mm"
    assert config.persistence.enabled is True
    assert config.persistence.topics == ("strategy:decisions",)
    assert config.observability.run_id == "run-1"
    assert config.observability.otel_enabled is True


def test_strategy_decision_payload_is_canonical_controller_event() -> None:
    decision = StrategyDecision(
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=1001,
        frame_step=1,
        action=Action(actor_id="mpc-controller"),
        latency_ms=2.5,
        decision_hash="abc",
    )

    payload = decision.as_payload()

    assert payload["type"] == "StrategyDecision"
    assert payload["actor_id"] == "mpc-controller"
    assert payload["order_count"] == 0
    assert payload["unit_policy"] == "prices=quote_per_base, quantities=base, cash/notional=quote"
    assert "economics" not in payload


def test_mpc_market_flow_step_tuple_from_frame() -> None:
    from mm.core.models.strategy import StrategyInputFrame

    frame = StrategyInputFrame(primary=_event())
    step = mpc_engine._market_flow_step_tuple(frame)

    assert len(step) == 15
    assert step[0] == 1001
    assert step[2] == 0.0
    assert step[3] == 0.0


def test_histogram_registry_emits_prometheus_histogram() -> None:
    from mm.observability_metrics import MetricsRegistry

    r = MetricsRegistry()
    r.histogram_observe("test_latency_ms", 42.0, labels={"pool": "a"}, help="unit test histogram")
    text = r.render()
    assert "test_latency_ms_bucket" in text
    assert "test_latency_ms_sum" in text
    assert "test_latency_ms_count" in text
    assert "histogram" in text


def test_strategy_decision_payload_includes_economics_when_set() -> None:
    decision = StrategyDecision(
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=1001,
        frame_step=1,
        action=Action(actor_id="mpc-controller"),
        economics={
            "t_end_ms": 1001,
            "step_counter": 7,
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "d_w": 1.0,
            "spread_pnl": 0.1,
            "inventory_pnl": 0.2,
            "cross_pnl": 0.0,
            "fee_pnl": -0.05,
            "wealth_pnl": 0.25,
        },
    )
    payload = decision.as_payload()
    assert payload["economics"]["d_w"] == 1.0
