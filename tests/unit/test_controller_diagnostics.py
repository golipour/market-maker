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
from mm.core.payload_schema import (
    CONTROLLER_DIAGNOSTICS_SCHEMA,
    CONTROLLER_DIAGNOSTICS_VERSION,
    assert_payload_schema,
)
from mm.diagnostics.models import CONTROLLER_DIAGNOSTICS_TYPE, ControllerDiagnostics
from mm.execution.memory import RecordingOrderManager
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    ObservabilityRuntimeConfig,
    RiskRuntimeConfig,
)
from mm.runtime.engine import _build_observability_daemons, _default_persistence_topics
from tests.unit.conftest import joint_frame_for_input, strategy_decision_with_joint


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
        dry_run=True,
        execution_after_steps=0,
        risk=RiskRuntimeConfig(enabled=True),
        optimizer={
            "local": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
            "global": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
        },
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
    assert_payload_schema(payload, CONTROLLER_DIAGNOSTICS_SCHEMA, CONTROLLER_DIAGNOSTICS_VERSION)
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
        "cleared_orderbook",
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
    assert_payload_schema(
        decoded.payload,
        CONTROLLER_DIAGNOSTICS_SCHEMA,
        CONTROLLER_DIAGNOSTICS_VERSION,
    )
    json.dumps(decoded.payload)


def test_normalized_diagnostics_topic_defaults_to_actor_id() -> None:
    config = _minimal_config(actor_id="actor-42")
    assert config.normalized_diagnostics_topic == "strategy:diagnostics:actor-42"


def test_default_persistence_topics_include_diagnostics() -> None:
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=_minimal_config(actor_id="mpc-controller"),
        observability=ObservabilityRuntimeConfig(),
    )
    topics = _default_persistence_topics(runtime)
    assert "strategy:diagnostics:mpc-controller" in topics
    assert "execution:errors" in topics
    assert "private:state:nobitex:errors" in topics


def test_observability_topics_include_diagnostics_risk_and_execution_errors() -> None:
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=_minimal_config(actor_id="mpc-controller"),
        observability=ObservabilityRuntimeConfig(),
    )
    daemons = _build_observability_daemons(runtime, InMemoryBus())

    topics = daemons[0].topics

    assert "strategy:diagnostics:mpc-controller" in topics
    assert "risk:events" in topics
    assert "execution:errors" in topics
    assert "private:state:nobitex:fills" in topics
    assert "private:state:nobitex:errors" in topics
    assert "market:raw:nobitex:BTC_IRT:orderbook" in topics
    assert "market:raw:nobitex:BTC_IRT:trades" in topics


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_on_controller_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        raise RuntimeError("mpc blew up")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", boom)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
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
    frame = StrategyInputFrame(local=_market_event(), created_at_ms=1_500)
    approved = await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    assert approved == []
    diagnostics = await diag_queue.get()
    assert diagnostics["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert_payload_schema(
        diagnostics,
        CONTROLLER_DIAGNOSTICS_SCHEMA,
        CONTROLLER_DIAGNOSTICS_VERSION,
    )
    assert diagnostics["frame_step"] == 1
    assert diagnostics["run_id"] == "run-test"
    assert any("mpc blew up" in w for w in diagnostics["warnings"])


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    order = Order("nobitex", "BTC_IRT", "buy", 99.0, 1.0, client_order_id="c1")

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        decision = strategy_decision_with_joint(frame.local, frame_step=frame_step, actor_id="test-runner")
        return StrategyDecision(
            actor_id=decision.actor_id,
            exchange=decision.exchange,
            symbol=decision.symbol,
            t_end_ms=decision.t_end_ms,
            frame_step=frame_step,
            action=decision.action,
            joint_target_book=decision.joint_target_book,
            mid=100.0,
            inventory=5.0,
            cash=1000.0,
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_minimal_config(),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": RecordingOrderManager()},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )
    frame = StrategyInputFrame(local=_market_event())
    await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    diagnostics = await diag_queue.get()
    assert diagnostics["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert diagnostics["state"]["mid"] == 100.0
    assert diagnostics["state"]["local_inventory"] == 5.0
    assert "dry_run_config" in diagnostics["execution_gates"]["blocked_reasons"]


def test_decision_context_maps_frame_fields() -> None:
    global_book = RawOrderbook(
        exchange="wallex",
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(99.0, 7.0),),
        asks=(PriceLevel(102.0, 6.0),),
        is_snapshot=True,
    )
    global_event = MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=global_book,
        trades=(),
        buy_volume=1.0,
        sell_volume=2.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )
    frame = StrategyInputFrame(local=_market_event(),
        global_=global_event,
        global_status="ready",
        global_lag_ms=12,
        created_at_ms=1_000,
    )
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
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
    assert diag.frame.global_status == "ready"
    assert diag.frame.global_lag_ms == 12
    assert diag.frame.frame_age_ms == 1500
    assert diag.state.mid == pytest.approx(100.5)
    assert diag.state.local_buy_vwap == pytest.approx(100.5)
    assert diag.state.local_sell_vwap == pytest.approx(100.5)
    assert diag.state.global_buy_vwap == pytest.approx(100.5)
    assert diag.state.global_sell_vwap == pytest.approx(100.5)
    assert diag.cleared_orderbook["bids"] == [{"price": 100.0, "qty": 5.0}]
    assert diag.cleared_orderbook["asks"] == [{"price": 101.0, "qty": 4.0}]
    assert diag.global_cleared_orderbook["bids"] == [{"price": 99.0, "qty": 7.0}]
    assert diag.global_cleared_orderbook["asks"] == [{"price": 102.0, "qty": 6.0}]


def test_sync_status_marks_stale_global_market_not_ready() -> None:
    frame = StrategyInputFrame(
        local=_market_event(),
        global_=None,
        global_status="stale",
        global_lag_ms=1500,
        created_at_ms=1_000,
    )
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
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

    sync_status = ctx.to_diagnostics().as_payload()["frame"]["sync_status"]

    assert sync_status["global_market_ready"] is False
    assert sync_status["global_private_ready"] is False
    assert sync_status["ledger_fresh"] is False


def test_optimizer_diagnostics_include_execution_model_config() -> None:
    frame = StrategyInputFrame(local=_market_event(), created_at_ms=1_000)
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    from mm.application.trading_context import trading_state_from_frame

    config = ControllerRuntimeConfig(
        enabled=True,
        actor_id="test-runner",
        dry_run=True,
        execution_after_steps=0,
        risk=RiskRuntimeConfig(enabled=True),
        execution_model={
            "kind": "real",
            "local_maker_fee": 0.0001,
            "global_maker_fee": 0.0002,
        },
        optimizer={
            "local": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
            "global": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
        },
    )
    ctx = DecisionContext(
        config=config,
        frame=frame,
        clean_frame=frame,
        trading_state=trading_state_from_frame(
            frame,
            runtime.binding,
            actor_id="test-runner",
            open_orders={},
        ),
        frame_step=2,
        now_ms=2_500,
    )

    optimizer = ctx.to_diagnostics().as_payload()["optimizer"]

    assert optimizer["config"]["execution_model"] == {
        "kind": "real",
        "local_maker_fee": 0.0001,
        "global_maker_fee": 0.0002,
    }


def test_decision_context_clears_own_orders_from_diagnostics_orderbook() -> None:
    from mm.application.orderbook_view import subtract_own_orders

    frame = StrategyInputFrame(local=_market_event())
    clean_book = subtract_own_orders(
        frame.local.orderbook,
        {
            "o1": Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0, client_order_id="c1"),
        },
    )
    clean_frame = StrategyInputFrame(local=MarketEvent(
            exchange="nobitex",
            symbol="BTC_IRT",
            interval_ms=1000,
            t_start_ms=1000,
            t_end_ms=2000,
            orderbook=clean_book,
            trades=(),
            buy_volume=1.0,
            sell_volume=2.0,
            vwap=0.0,
            imbalance=0.0,
            microprice=100.5,
        )
    )
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
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
        clean_frame=clean_frame,
        trading_state=trading_state,
        frame_step=1,
    )
    payload = ctx.to_diagnostics().as_payload()
    assert payload["cleared_orderbook"]["bids"] == [{"price": 100.0, "qty": 3.0}]
