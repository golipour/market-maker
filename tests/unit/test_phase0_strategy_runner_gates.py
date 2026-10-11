"""Phase 0 regression tests: StrategyRunner observe-only, warmup, and live execution gates."""

from __future__ import annotations

import asyncio

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.execution.memory import RecordingOrderManager
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    RiskRuntimeConfig,
)
from mm.runtime.control import RuntimeControlPlane, RuntimeControlState
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


def _decision(event: MarketEvent, *, frame_step: int) -> StrategyDecision:
    return strategy_decision_with_joint(event, frame_step=frame_step)


def _runner(
    *,
    config: ControllerRuntimeConfig,
    manager: RecordingOrderManager,
    bus: InMemoryBus,
    control_plane: RuntimeControlPlane | None = None,
) -> StrategyRunner:
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    return StrategyRunner(
        config=config,
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
        control_plane=control_plane,
    )


def _runner_config(**overrides: object) -> ControllerRuntimeConfig:
    base = {
        "enabled": True,
        "actor_id": "test-runner",
        "dry_run": False,
        "execution_after_steps": 0,
        "placement_timeout_ms": 1000,
        "decision_timeout_ms": 0,
        "risk": RiskRuntimeConfig(enabled=True),
        "optimizer": {
            "local": {"cumulative_qty_floor_at_spread_max": 1.0, "cumulative_qty_cap_at_spread_max": 10.0},
            "global": {"cumulative_qty_floor_at_spread_max": 1.0, "cumulative_qty_cap_at_spread_max": 10.0},
        },
    }
    base.update(overrides)
    return ControllerRuntimeConfig(**base)


async def _run_frame(runner: StrategyRunner) -> None:
    await runner.on_joint_frame(joint_frame_for_input(StrategyInputFrame(local=_market_event())))


@pytest.mark.asyncio
async def test_strategy_runner_dry_run_config_blocks_live_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local, frame_step=frame_step),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    runner = _runner(
        config=_runner_config(dry_run=True),
        manager=manager,
        bus=bus,
    )

    await _run_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert manager.synced == []
    assert diagnostics["execution_gates"]["executed"] is False
    assert "dry_run_config" in diagnostics["execution_gates"]["blocked_reasons"]


@pytest.mark.asyncio
async def test_strategy_runner_dry_run_control_plane_blocks_live_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local, frame_step=frame_step),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    control_plane = RuntimeControlPlane(
        bus=bus,
        state=RuntimeControlState(dry_run=True),
    )
    runner = _runner(
        config=_runner_config(dry_run=False),
        manager=manager,
        bus=bus,
        control_plane=control_plane,
    )

    await _run_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert manager.synced == []
    assert diagnostics["execution_gates"]["executed"] is False
    assert "dry_run_control" in diagnostics["execution_gates"]["blocked_reasons"]


@pytest.mark.asyncio
async def test_strategy_runner_warmup_blocks_live_sync_until_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local, frame_step=frame_step),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    runner = _runner(
        config=_runner_config(execution_after_steps=2),
        manager=manager,
        bus=bus,
    )

    await _run_frame(runner)
    await _run_frame(runner)
    await _run_frame(runner)

    assert len(manager.synced) == 1

    first = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    second = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    third = await asyncio.wait_for(diag_queue.get(), timeout=1.0)

    assert first["execution_gates"]["executed"] is False
    assert "warmup" in first["execution_gates"]["blocked_reasons"]
    assert second["execution_gates"]["executed"] is False
    assert "warmup" in second["execution_gates"]["blocked_reasons"]
    assert third["execution_gates"]["executed"] is True
    assert third["execution_gates"]["blocked_reasons"] == []


@pytest.mark.asyncio
async def test_strategy_runner_live_execution_syncs_when_gates_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local, frame_step=frame_step),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    runner = _runner(
        config=_runner_config(dry_run=False, execution_after_steps=0),
        manager=manager,
        bus=bus,
    )

    await _run_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert len(manager.synced) == 1
    assert diagnostics["execution_gates"]["executed"] is True
    assert diagnostics["execution_gates"]["blocked_reasons"] == []


@pytest.mark.asyncio
async def test_strategy_runner_no_manager_blocks_live_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local, frame_step=frame_step),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(dry_run=False),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    await _run_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert diagnostics["execution_gates"]["executed"] is False
    assert "no_manager" in diagnostics["execution_gates"]["blocked_reasons"]


@pytest.mark.asyncio
async def test_strategy_runner_skips_optimization_when_actor_is_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decide_called = False

    def _decide_fail(*args: object, **kwargs: object) -> StrategyDecision:
        nonlocal decide_called
        decide_called = True
        raise AssertionError("decide() must NOT be called when actor is paused!")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", _decide_fail)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    heartbeat_queue = bus.subscribe_queue("strategy:heartbeat:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    control_plane = RuntimeControlPlane(
        bus=bus,
        state=RuntimeControlState(paused_actors={"test-runner"}),
    )
    runner = _runner(
        config=_runner_config(dry_run=False),
        manager=manager,
        bus=bus,
        control_plane=control_plane,
    )

    await _run_frame(runner)

    assert decide_called is False
    assert manager.synced == []

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert diagnostics["execution_gates"]["executed"] is False
    assert "paused_actor" in diagnostics["execution_gates"]["blocked_reasons"]

    heartbeat = await asyncio.wait_for(heartbeat_queue.get(), timeout=1.0)
    assert heartbeat["decided"] is False
    assert heartbeat["decision_latency_ms"] is None


@pytest.mark.asyncio
async def test_strategy_runner_skips_optimization_when_kill_switch_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decide_called = False

    def _decide_fail(*args: object, **kwargs: object) -> StrategyDecision:
        nonlocal decide_called
        decide_called = True
        raise AssertionError("decide() must NOT be called when kill switch is enabled!")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", _decide_fail)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = RecordingOrderManager(exchange="nobitex")
    control_plane = RuntimeControlPlane(
        bus=bus,
        state=RuntimeControlState(kill_switch=True),
    )
    runner = _runner(
        config=_runner_config(dry_run=False),
        manager=manager,
        bus=bus,
        control_plane=control_plane,
    )

    await _run_frame(runner)

    assert decide_called is False
    assert manager.synced == []

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert diagnostics["execution_gates"]["executed"] is False
    assert "kill_switch" in diagnostics["execution_gates"]["blocked_reasons"]


@pytest.mark.asyncio
async def test_control_plane_pause_invokes_on_actor_pause_and_cancels_runner() -> None:
    bus = InMemoryBus()
    manager = RecordingOrderManager(exchange="nobitex")
    runner = _runner(
        config=_runner_config(dry_run=False),
        manager=manager,
        bus=bus,
    )

    paused_actor: str | None = None

    def _on_pause(actor_id: str) -> None:
        nonlocal paused_actor
        paused_actor = actor_id
        runner.cancel_decide()

    control_plane = RuntimeControlPlane(
        bus=bus,
        on_actor_pause=_on_pause,
    )
    runner.control_plane = control_plane

    assert not runner._decide_cancel_event.is_set()

    await control_plane.apply({"command": "pause", "actor_id": "test-runner"})

    assert paused_actor == "test-runner"
    assert "test-runner" in control_plane.state.paused_actors
    assert runner._decide_cancel_event.is_set()

