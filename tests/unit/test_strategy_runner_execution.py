from __future__ import annotations

import asyncio

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.execution.plan import ExecutionSyncReport, summarize_orders
from mm.risk.engine import PassThroughRiskEngine
from tests.unit.conftest import strategy_decision_with_joint
from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig, RiskRuntimeConfig


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


def _runner_config(**overrides: object) -> ControllerRuntimeConfig:
    base = {
        "enabled": True,
        "actor_id": "test-runner",
        "observe_only": False,
        "action_start_after_steps": 0,
        "placement_timeout_ms": 50,
        "risk": RiskRuntimeConfig(enabled=True),
        "optimizer": {
            "quote_size_min_per_side": 1.0,
            "quote_size_max_per_side": 10.0,
        },
    }
    base.update(overrides)
    return ControllerRuntimeConfig(**base)


def _decision(event: MarketEvent) -> StrategyDecision:
    return strategy_decision_with_joint(event, frame_step=1)


class _SlowOrderManager:
    exchange = "nobitex"

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.sync_calls = 0

    @property
    def live_orders(self) -> dict[str, Order]:
        return {}

    async def sync_with_report(self, actions: list[Action]) -> ExecutionSyncReport:
        self.sync_calls += 1
        await asyncio.sleep(self.delay_s)
        orders = tuple(order for action in actions for order in action.orders)
        summary = summarize_orders(orders)
        return ExecutionSyncReport(
            target=summary,
            kept=summary,
            limit_valid=summary,
            accepted=summary,
            placed=True,
        )

    async def sync_venue_target_book(self, target):
        from mm.core.adapters.joint import venue_target_book_to_action

        return await self.sync_with_report([venue_target_book_to_action(target)])



class _FailingOrderManager:
    exchange = "nobitex"

    def __init__(self) -> None:
        self.sync_calls = 0

    @property
    def live_orders(self) -> dict[str, Order]:
        return {}

    async def sync_with_report(self, actions: list[Action]) -> ExecutionSyncReport:
        self.sync_calls += 1
        raise RuntimeError("place order failed on nobitex: simulated")

    async def sync_venue_target_book(self, target):
        from mm.core.adapters.joint import venue_target_book_to_action

        return await self.sync_with_report([venue_target_book_to_action(target)])



@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_after_sync_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _SlowOrderManager(delay_s=0.2)
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=50),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    await runner.on_frame(StrategyInputFrame(local=_market_event()))

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert diagnostics["type"] == "ControllerDiagnostics"
    assert any("ExecutionSyncTimeout" in warning for warning in diagnostics["warnings"])
    assert "execution_sync_timeout" in diagnostics["execution_gates"]["blocked_reasons"]
    assert diagnostics["execution_gates"]["executed"] is False
    assert manager.sync_calls == 1


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_after_sync_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _FailingOrderManager()
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    await runner.on_frame(StrategyInputFrame(local=_market_event()))

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert any("ExecutionSyncError" in warning for warning in diagnostics["warnings"])
    assert "execution_sync_error" in diagnostics["execution_gates"]["blocked_reasons"]
    assert manager.sync_calls == 1


@pytest.mark.asyncio
async def test_strategy_runner_continues_after_sync_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        calls["count"] += 1
        return _decision(frame.local)

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _FailingOrderManager()
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    await runner.on_frame(StrategyInputFrame(local=_market_event()))
    await runner.on_frame(StrategyInputFrame(local=_market_event()))

    assert calls["count"] == 2
    assert manager.sync_calls == 2
    first = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    second = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert first["frame_step"] == 1
    assert second["frame_step"] == 2
