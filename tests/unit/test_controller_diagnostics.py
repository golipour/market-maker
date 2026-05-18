from __future__ import annotations

import asyncio
import json

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.controller.context import DecisionContext
from mm.core.events import decode_envelope, encode_envelope, envelope_event
from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.diagnostics.models import CONTROLLER_DIAGNOSTICS_TYPE, CONTROLLER_DIAGNOSTICS_VERSION, ControllerDiagnostics
from mm.execution.memory import RecordingOrderManager
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    ObservabilityRuntimeConfig,
    RiskRuntimeConfig,
)
from mm.runtime.engine import _default_persistence_topics


def _market_event() -> MarketEvent:
    book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=book,
        trades=(),
        buy_volume=1.0,
        sell_volume=2.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )


def _minimal_config(*, actor_id: str = "test-runner") -> ControllerRuntimeConfig:
    return ControllerRuntimeConfig(
        enabled=True,
        actor_id=actor_id,
        observe_only=True,
        action_start_after_steps=0,
        risk=RiskRuntimeConfig(enabled=True),
    )


def test_controller_diagnostics_payload_has_stable_shape() -> None:
    diagnostics = ControllerDiagnostics(
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2000,
        frame_step=3,
        created_at_ms=9_999,
        run_id="run-abc",
        warnings=("optimizer_failed",),
    )
    payload = diagnostics.as_payload()

    assert payload["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert payload["version"] == CONTROLLER_DIAGNOSTICS_VERSION
    assert payload["actor_id"] == "mpc-controller"
    assert payload["exchange"] == "nobitex"
    assert payload["symbol"] == "BTC_IRT"
    assert payload["t_end_ms"] == 2000
    assert payload["frame_step"] == 3
    assert payload["created_at_ms"] == 9_999
    assert payload["run_id"] == "run-abc"
    assert set(payload) >= {
        "frame",
        "state",
        "models",
        "optimizer",
        "action_pipeline",
        "risk",
        "execution_gates",
        "warnings",
    }
    assert payload["execution_gates"]["executed"] is False
    assert payload["warnings"] == ["optimizer_failed"]


def test_controller_diagnostics_envelope_round_trips() -> None:
    diagnostics = ControllerDiagnostics(
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2000,
        frame_step=1,
        created_at_ms=1_000,
    )
    topic = "strategy:diagnostics:mpc-controller"
    envelope = envelope_event(topic, diagnostics.as_payload(), run_id="run-1")

    decoded = decode_envelope(encode_envelope(envelope))

    assert decoded.schema == "mm.event"
    assert decoded.version == 1
    assert decoded.run_id == "run-1"
    assert decoded.topic == topic
    assert decoded.event_type == CONTROLLER_DIAGNOSTICS_TYPE
    assert decoded.exchange == "nobitex"
    assert decoded.symbol == "BTC_IRT"
    assert decoded.actor_id == "mpc-controller"
    assert decoded.payload["version"] == CONTROLLER_DIAGNOSTICS_VERSION
    json.dumps(decoded.payload)


def test_normalized_diagnostics_topic_defaults_to_actor_id() -> None:
    config = _minimal_config(actor_id="actor-42")
    assert config.normalized_diagnostics_topic == "strategy:diagnostics:actor-42"


def test_default_persistence_topics_include_diagnostics() -> None:
    runtime = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        controller=_minimal_config(actor_id="mpc-controller"),
        observability=ObservabilityRuntimeConfig(),
    )
    topics = _default_persistence_topics(runtime)
    assert "strategy:diagnostics:mpc-controller" in topics


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_on_controller_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        raise RuntimeError("mpc blew up")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", boom)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runtime = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_minimal_config(),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": RecordingOrderManager()},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
        run_id="run-test",
    )
    frame = StrategyInputFrame(primary=_market_event(), created_at_ms=1_500)
    approved = await runner.on_frame(frame)

    assert approved == []
    diagnostics = await diag_queue.get()
    assert diagnostics["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert diagnostics["version"] == CONTROLLER_DIAGNOSTICS_VERSION
    assert diagnostics["frame_step"] == 1
    assert diagnostics["run_id"] == "run-test"
    assert any("mpc blew up" in w for w in diagnostics["warnings"])


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    order = Order("nobitex", "BTC_IRT", "buy", 99.0, 1.0, client_order_id="c1")

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.primary.exchange,
            symbol=frame.primary.symbol,
            t_end_ms=frame.primary.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders_to_create=(order,)),
            mid=100.0,
            inventory=5.0,
            cash=1000.0,
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runtime = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_minimal_config(),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": RecordingOrderManager()},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )
    await runner.on_frame(StrategyInputFrame(primary=_market_event()))

    diagnostics = await diag_queue.get()
    assert diagnostics["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert diagnostics["state"]["mid"] == 100.0
    assert diagnostics["state"]["primary_inventory"] == 5.0
    assert "observe_only_config" in diagnostics["execution_gates"]["blocked_reasons"]


def test_decision_context_maps_frame_fields() -> None:
    frame = StrategyInputFrame(
        primary=_market_event(),
        hedge_status="ready",
        hedge_lag_ms=12,
        created_at_ms=1_000,
    )
    runtime = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
    )
    from mm.application.trading_context import trading_state_from_frame

    trading_state = trading_state_from_frame(
        frame,
        runtime.binding,
        actor_id="test-runner",
        open_orders={},
    )
    ctx = DecisionContext(
        config=_minimal_config(),
        frame=frame,
        clean_frame=frame,
        trading_state=trading_state,
        frame_step=2,
        now_ms=2_500,
    )
    diag = ctx.to_diagnostics()
    assert diag.frame.hedge_status == "ready"
    assert diag.frame.hedge_lag_ms == 12
    assert diag.frame.frame_age_ms == 1500
