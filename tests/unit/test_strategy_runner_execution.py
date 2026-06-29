from __future__ import annotations

import asyncio
import time

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Action, Order
from mm.core.models.joint_strategy import FrameSyncStatus, JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.risk_limits import GlobalVenueRiskLimits, VenueRiskLimits
from mm.execution.plan import (
    ActionPipelineReport,
    ActionStageReport,
    ExecutionSyncReport,
    RiskResult,
    summarize_orders,
)
from mm.risk.engine import PassThroughRiskEngine, RiskLimits
from mm.risk.models import TargetBookRiskResult
from mm.risk.net_delta import NetDeltaRiskEngine
from tests.unit.conftest import joint_frame_for_input, strategy_decision_with_joint
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    RiskRuntimeConfig,
    VenueExecutionConfig,
)


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
        "dry_run": False,
        "execution_after_steps": 0,
        "decision_timeout_ms": 0,
        "placement_timeout_ms": 50,
        "risk": RiskRuntimeConfig(enabled=True),
        "optimizer": {
            "local": {"cumulative_qty_floor_at_spread_max": 1.0, "cumulative_qty_cap_at_spread_max": 10.0},
            "global": {"cumulative_qty_floor_at_spread_max": 1.0, "cumulative_qty_cap_at_spread_max": 10.0},
        },
    }
    base.update(overrides)
    return ControllerRuntimeConfig(**base)


def _decision(event: MarketEvent) -> StrategyDecision:
    return strategy_decision_with_joint(event, frame_step=1)


async def _run_market_frame(runner: StrategyRunner) -> list[Action]:
    return await runner.on_joint_frame(
        joint_frame_for_input(StrategyInputFrame(local=_market_event()))
    )


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
            to_place=summary,
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


class _EmptyApprovingRiskEngine:
    def validate_frame_with_result(
        self,
        actions: list[Action],
        state,
        frame: StrategyInputFrame,
    ) -> RiskResult:
        _ = actions, state, frame
        return RiskResult(approved=(), rejection_counts={})


class _RecordingVenueOrderManager:
    exchange = "nobitex"

    def __init__(self) -> None:
        self.sync_calls = 0
        self.targets = []

    @property
    def live_orders(self) -> dict[str, Order]:
        return {}

    async def sync_venue_target_book(self, target) -> ExecutionSyncReport:
        self.sync_calls += 1
        self.targets.append(target)
        return ExecutionSyncReport(
            canceled=ActionStageReport(order_count=1),
            cancel_count=1,
        )


@pytest.mark.asyncio
async def test_strategy_runner_propagates_lineage_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        target = JointTargetBook(
            actor_id="test-runner",
            local=VenueTargetBook(
                exchange="nobitex",
                symbol="BTC_IRT",
                actor_id="test-runner",
                orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 0.5),),
            ),
        )
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(
                actor_id="test-runner",
                orders=(Order("nobitex", "BTC_IRT", "buy", 100.0, 0.5),),
            ),
            joint_target_book=target,
            action_pipeline_report=ActionPipelineReport(target_book=ActionStageReport(order_count=1)),
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    decision_queue = bus.subscribe_queue("strategy:decisions")
    target_queue = bus.subscribe_queue("strategy:target_books:test-runner")
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _RecordingVenueOrderManager()
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    runner = StrategyRunner(
        config=_runner_config(),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    await _run_market_frame(runner)

    decision = await asyncio.wait_for(decision_queue.get(), timeout=1.0)
    target = await asyncio.wait_for(target_queue.get(), timeout=1.0)
    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert decision["frame_id"] == "frame:test-runner:nobitex:btc_irt:2000:1"
    assert decision["decision_id"] == "decision:frame_test-runner_nobitex_btc_irt_2000_1"
    assert decision["target_book_id"] == "target:frame_test-runner_nobitex_btc_irt_2000_1"
    assert target["frame_id"] == decision["frame_id"]
    assert target["decision_id"] == decision["decision_id"]
    assert target["target_book_id"] == decision["target_book_id"]
    assert target["local"]["execution_plan_id"] == (
        "execplan:target_frame_test-runner_nobitex_btc_irt_2000_1:local:nobitex"
    )
    assert diagnostics["frame_id"] == decision["frame_id"]
    assert diagnostics["decision_id"] == decision["decision_id"]
    assert diagnostics["target_book_id"] == decision["target_book_id"]
    [manager_target] = manager.targets
    assert manager_target.frame_id == decision["frame_id"]
    assert manager_target.decision_id == decision["decision_id"]
    assert manager_target.target_book_id == decision["target_book_id"]
    assert manager_target.execution_plan_id == target["local"]["execution_plan_id"]


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_after_sync_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    error_queue = bus.subscribe_queue("execution:errors")
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

    await _run_market_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert diagnostics["type"] == "ControllerDiagnostics"
    assert any("ExecutionSyncTimeout" in warning for warning in diagnostics["warnings"])
    assert "execution_sync_timeout" in diagnostics["execution_gates"]["blocked_reasons"]
    assert diagnostics["execution_gates"]["executed"] is False
    assert manager.sync_calls == 1
    error = await asyncio.wait_for(error_queue.get(), timeout=1.0)
    assert error["type"] == "ExecutionSyncFailed"
    assert error["error"] == "ExecutionSyncTimeout"
    assert error["latency_ms"] >= 50.0
    assert "timestamp_ms" in error


@pytest.mark.asyncio
async def test_strategy_runner_holds_current_orders_after_decision_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        time.sleep(0.05)
        return strategy_decision_with_joint(frame.local, frame_step=frame_step, actor_id="test-runner")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", slow_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    decision_queue = bus.subscribe_queue("strategy:decisions")
    action_queue = bus.subscribe_queue("strategy:actions")
    target_queue = bus.subscribe_queue("strategy:target_books:test-runner")
    manager = _SlowOrderManager(delay_s=0.0)
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(decision_timeout_ms=10),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    approved = await _run_market_frame(runner)

    fallback = await asyncio.wait_for(decision_queue.get(), timeout=1.0)
    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert approved == []
    assert fallback["type"] == "ControllerFallback"
    assert fallback["reason"] == "optimizer_decision_timeout"
    assert fallback["fallback_policy"] == "hold_current_orders"
    assert manager.sync_calls == 0
    assert action_queue.empty()
    assert target_queue.empty()
    assert any("OptimizerDecisionTimeout" in warning for warning in diagnostics["warnings"])

    await asyncio.sleep(0.06)


@pytest.mark.asyncio
async def test_strategy_runner_publishes_diagnostics_after_sync_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mm.application.strategy_runner.MpcStrategyEngine.decide",
        lambda self, frame, *, frame_step: _decision(frame.local),
    )

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    error_queue = bus.subscribe_queue("execution:errors")
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

    await _run_market_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert any("ExecutionSyncError" in warning for warning in diagnostics["warnings"])
    assert "execution_sync_error" in diagnostics["execution_gates"]["blocked_reasons"]
    assert manager.sync_calls == 1
    error = await asyncio.wait_for(error_queue.get(), timeout=1.0)
    assert error["type"] == "ExecutionSyncFailed"
    assert error["error"] == "ExecutionSyncError"
    assert error["latency_ms"] >= 0.0
    assert "timestamp_ms" in error


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

    await _run_market_frame(runner)
    await _run_market_frame(runner)

    assert calls["count"] == 2
    assert manager.sync_calls == 2
    first = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    second = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert first["frame_step"] == 1
    assert second["frame_step"] == 2


@pytest.mark.asyncio
async def test_strategy_runner_syncs_empty_replace_book_when_risk_has_no_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=()),
            joint_target_book=JointTargetBook(
                actor_id="test-runner",
                local=VenueTargetBook(
                    exchange="nobitex",
                    symbol="BTC_IRT",
                    actor_id="test-runner",
                    orders=(),
                    policy="replace",
                ),
            ),
            action_pipeline_report=ActionPipelineReport(
                target_book=ActionStageReport(order_count=1),
            ),
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _SlowOrderManager(delay_s=0.0)
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"))
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=_EmptyApprovingRiskEngine(),
        execution=ExecutionRuntimeConfig(local=VenueExecutionConfig(planner="replace")),
        bus=bus,
    )

    approved = await _run_market_frame(runner)

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert approved == []
    assert manager.sync_calls == 1
    assert diagnostics["execution_gates"]["executed"] is True
    assert diagnostics["execution_gates"]["venue_gates"]["nobitex"]["can_sync"] is True
    assert diagnostics["execution_gates"]["venue_gates"]["nobitex"]["has_orders"] is False


@pytest.mark.asyncio
async def test_strategy_runner_syncs_cancel_only_replace_book_when_risk_rejects_new_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        target_order = TargetOrder(
            exchange="nobitex",
            symbol="BTC_IRT",
            side="buy",
            price=100.0,
            qty=2.0,
        )
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(
                actor_id="test-runner",
                orders=(Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0),),
            ),
            joint_target_book=JointTargetBook(
                actor_id="test-runner",
                local=VenueTargetBook(
                    exchange="nobitex",
                    symbol="BTC_IRT",
                    actor_id="test-runner",
                    orders=(target_order,),
                    policy="replace",
                ),
            ),
            action_pipeline_report=ActionPipelineReport(
                target_book=ActionStageReport(order_count=1),
            ),
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    risk_engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=runtime.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 2_000,
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        local_private=PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=2_000,
            balances={
                "BTC": Balance("nobitex", "BTC", 10.0, 0.0, 10.0, 2_000),
                "IRT": Balance("nobitex", "IRT", 50.0, 0.0, 50.0, 2_000),
            },
            open_orders={},
        ),
        created_at_ms=2_000,
    )
    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    manager = _RecordingVenueOrderManager()
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=risk_engine,
        execution=ExecutionRuntimeConfig(local=VenueExecutionConfig(planner="replace")),
        bus=bus,
    )

    approved = await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert approved == []
    assert manager.sync_calls == 1
    assert manager.targets[0].orders == ()
    assert diagnostics["execution_gates"]["executed"] is True
    assert diagnostics["risk"]["rejection_counts"]["insufficient_quote_for_bid_orders"] == 1
    assert diagnostics["action_pipeline"]["risk_approved_target"]["order_count"] == 0
    assert diagnostics["action_pipeline"]["risk_approved_target"]["rejections"]
    assert (
        diagnostics["action_pipeline"]["risk_approved_target"]["rejections"][0]["reason"]
        == "insufficient_quote_for_bid_orders"
    )
    assert diagnostics["action_pipeline"]["planned_cancel"]["order_count"] == 1
    assert diagnostics["action_pipeline"]["planned_place"]["order_count"] == 0


@pytest.mark.asyncio
async def test_strategy_runner_records_local_risk_approved_target_for_net_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0)

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        target_order = TargetOrder(
            exchange=order.exchange,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            qty=order.qty,
        )
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=(order,)),
            joint_target_book=JointTargetBook(
                actor_id="test-runner",
                local=VenueTargetBook(
                    exchange="nobitex",
                    symbol="BTC_IRT",
                    actor_id="test-runner",
                    orders=(target_order,),
                    policy="reconcile",
                ),
            ),
            action_pipeline_report=ActionPipelineReport(
                capped=summarize_orders((order,)),
                target_book=summarize_orders((order,)),
            ),
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    risk_engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=runtime.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 2_000,
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        local_private=PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=2_000,
            balances={
                "BTC": Balance("nobitex", "BTC", 10.0, 0.0, 10.0, 2_000),
                "IRT": Balance("nobitex", "IRT", 1_000.0, 0.0, 1_000.0, 2_000),
            },
            open_orders={},
        ),
        created_at_ms=2_000,
    )
    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": _SlowOrderManager(delay_s=0.0)},
        risk_engine=risk_engine,
        execution=ExecutionRuntimeConfig(local=VenueExecutionConfig(planner="reconcile")),
        bus=bus,
    )

    approved = await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert len(approved) == 1
    assert diagnostics["action_pipeline"]["risk_approved_target"]["order_count"] == 1
    assert diagnostics["action_pipeline"]["risk_approved_target"]["orders"][0]["exchange"] == "nobitex"
    assert diagnostics["action_pipeline"]["planned_place"]["order_count"] == 1


@pytest.mark.asyncio
async def test_strategy_runner_records_global_risk_approved_target_for_net_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_order = Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)
    global_order = Order("wallex", "BTC_IRT", "sell", 102.0, 2.0)

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=(local_order, global_order)),
            joint_target_book=JointTargetBook(
                actor_id="test-runner",
                local=VenueTargetBook(
                    "nobitex",
                    "BTC_IRT",
                    "test-runner",
                    (TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 1.0),),
                    policy="replace",
                ),
                global_=VenueTargetBook(
                    "wallex",
                    "BTC_IRT",
                    "test-runner",
                    (TargetOrder("wallex", "BTC_IRT", "sell", 102.0, 2.0),),
                    policy="replace",
                ),
                mode="liquidity_sourcing",
            ),
            action_pipeline_report=ActionPipelineReport(
                capped=summarize_orders((local_order, global_order)),
                target_book=summarize_orders((local_order, global_order)),
            ),
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT"),
    )
    risk_engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            ),
            global_=GlobalVenueRiskLimits(
                min_cash=0.0,
                min_inventory=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            ),
        ),
        binding=runtime.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        clock_ms=lambda: 2_000,
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        global_=MarketEvent(
            exchange="wallex",
            symbol="BTC_IRT",
            interval_ms=1000,
            t_start_ms=1000,
            t_end_ms=2000,
            orderbook=RawOrderbook("wallex", "BTC_IRT", 2000, (), (), True),
            trades=(),
            buy_volume=0.0,
            sell_volume=0.0,
            vwap=0.0,
            imbalance=0.0,
            microprice=100.5,
        ),
        local_private=PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=2_000,
            balances={
                "BTC": Balance("nobitex", "BTC", 10.0, 0.0, 10.0, 2_000),
                "IRT": Balance("nobitex", "IRT", 1_000.0, 0.0, 1_000.0, 2_000),
            },
            open_orders={},
        ),
        global_private=PrivateStateSnapshot(
            exchange="wallex",
            timestamp_ms=2_000,
            balances={
                "BTC": Balance("wallex", "BTC", 10.0, 0.0, 10.0, 2_000),
                "IRT": Balance("wallex", "IRT", 0.0, 0.0, 0.0, 2_000),
            },
            open_orders={},
        ),
        global_status="ready",
        created_at_ms=2_000,
    )
    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={},
        risk_engine=risk_engine,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        execution=ExecutionRuntimeConfig(local=VenueExecutionConfig(planner="replace")),
        bus=bus,
    )

    approved = await runner.on_joint_frame(
        joint_frame_for_input(
            frame,
            binding=runtime.binding,
            liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        )
    )

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert len(approved) == 1
    risk_orders = diagnostics["action_pipeline"]["risk_approved_target"]["orders"]
    assert {(order["exchange"], order["side"]) for order in risk_orders} == {
        ("nobitex", "sell"),
        ("wallex", "sell"),
    }


@pytest.mark.asyncio
async def test_strategy_runner_keeps_global_risk_rejections_out_of_local_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid_order = Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0)
    ask_order = Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        target_orders = (
            TargetOrder(
                exchange=bid_order.exchange,
                symbol=bid_order.symbol,
                side=bid_order.side,
                price=bid_order.price,
                qty=bid_order.qty,
            ),
            TargetOrder(
                exchange=ask_order.exchange,
                symbol=ask_order.symbol,
                side=ask_order.side,
                price=ask_order.price,
                qty=ask_order.qty,
            ),
        )
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=(bid_order, ask_order)),
            joint_target_book=JointTargetBook(
                actor_id="test-runner",
                local=VenueTargetBook(
                    exchange="nobitex",
                    symbol="BTC_IRT",
                    actor_id="test-runner",
                    orders=target_orders,
                    policy="replace",
                ),
            ),
            action_pipeline_report=ActionPipelineReport(
                capped=summarize_orders((bid_order, ask_order)),
                target_book=summarize_orders((bid_order, ask_order)),
            ),
        )

    def fake_validate(self, target_book, ledger, frame, **kwargs) -> TargetBookRiskResult:
        _ = self, target_book, ledger, frame, kwargs
        return TargetBookRiskResult(
            approved=None,
            rejection_counts={
                "insufficient_quote_for_bid_orders": 1,
                "min_cash": 1,
                "global_min_cash": 1,
            },
            exposure_before={"local_available_quote": 50.0},
            exposure_after={"local_available_quote": 50.0},
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)
    monkeypatch.setattr(NetDeltaRiskEngine, "validate_target_book_with_result", fake_validate)

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    risk_engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=runtime.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 2_000,
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        local_private=PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=2_000,
            balances={
                "BTC": Balance("nobitex", "BTC", 10.0, 0.0, 10.0, 2_000),
                "IRT": Balance("nobitex", "IRT", 50.0, 0.0, 50.0, 2_000),
            },
            open_orders={},
        ),
        created_at_ms=2_000,
    )
    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    runner = StrategyRunner(
        config=_runner_config(placement_timeout_ms=1000),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": _RecordingVenueOrderManager()},
        risk_engine=risk_engine,
        execution=ExecutionRuntimeConfig(local=VenueExecutionConfig(planner="replace")),
        bus=bus,
    )

    approved = await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert approved == []
    assert diagnostics["risk"]["rejection_counts"] == {
        "insufficient_quote_for_bid_orders": 1,
        "min_cash": 1,
        "global_min_cash": 1,
    }
    pipeline_risk = diagnostics["action_pipeline"]["risk_approved_target"]
    assert pipeline_risk["rejection_counts"] == {
        "insufficient_quote_for_bid_orders": 1,
        "min_cash": 1,
    }
    assert {row["reason"] for row in pipeline_risk["rejections"]} == {
        "insufficient_quote_for_bid_orders",
        "min_cash",
    }
    assert {row["side"] for row in pipeline_risk["rejections"]} == {"buy"}


@pytest.mark.asyncio
async def test_strategy_runner_records_pipeline_error_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=()),
            joint_target_book=None,
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    diag_queue = bus.subscribe_queue("strategy:diagnostics:test-runner")
    decision_queue = bus.subscribe_queue("strategy:decisions")
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    approved = await _run_market_frame(runner)

    assert approved == []
    controller_error = await asyncio.wait_for(decision_queue.get(), timeout=1.0)
    diagnostics = await asyncio.wait_for(diag_queue.get(), timeout=1.0)
    assert controller_error["type"] == "ControllerError"
    assert controller_error["error"] == "ValueError"
    assert any("joint_target_book" in warning for warning in diagnostics["warnings"])


@pytest.mark.asyncio
async def test_joint_frame_uses_ledger_live_orders_for_clean_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        seen["best_bid_qty"] = frame.local.orderbook.bids[0].qty
        assert frame.local_private is None
        seen["ledger_order_ids"] = tuple(self._ledger_context.local.live_orders)
        return strategy_decision_with_joint(frame.local, frame_step=frame_step, actor_id="test-runner")

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    order = Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0, client_order_id="ledger-order")
    open_order = OpenOrder(
        exchange_order_id="ledger-1",
        actor_id="test-runner",
        order=order,
        remaining_qty=2.0,
        filled_qty=0.0,
        status="open",
        timestamp_ms=2000,
    )
    input_frame = StrategyInputFrame(local=_market_event(), created_at_ms=2000)
    ledger = JointLedgerSnapshot(
        local=VenueLedger(
            exchange="nobitex",
            symbol="BTC_IRT",
            base_asset="BTC",
            quote_asset="IRT",
            live_orders={"ledger-1": open_order},
            last_snapshot_ms=2000,
        ),
        created_at_ms=2000,
    )
    joint_frame = JointStrategyFrame(
        local_market=input_frame.local,
        ledger=ledger,
        sync_status=FrameSyncStatus(
            local_market_ready=True,
            local_private_ready=True,
            ledger_fresh=True,
        ),
        created_at_ms=2000,
        input_frame=input_frame,
    )
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_runner_config(dry_run=True),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={},
        risk_engine=PassThroughRiskEngine(),
        bus=InMemoryBus(),
    )

    await runner.on_joint_frame(joint_frame)

    assert seen["best_bid_qty"] == pytest.approx(3.0)
    assert seen["ledger_order_ids"] == ("ledger-1",)
