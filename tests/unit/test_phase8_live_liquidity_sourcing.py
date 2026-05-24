"""Phase 8 tests: live liquidity sourcing mode and per-venue execution."""

from __future__ import annotations

import asyncio

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.application.venue_execution import (
    apply_venue_execution_caps,
    compute_venue_execution_gates,
    execution_book_from_joint,
    prepare_execution_joint_book,
)
from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.execution.memory import RecordingOrderManager
from mm.execution.order_manager import OrderManager
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    RiskRuntimeConfig,
    VenueExecutionConfig,
    runtime_config_from_mapping,
)
from mm.runtime.control import RuntimeControlPlane, RuntimeControlState
from tests.unit.conftest import venues_payload, strategy_decision_with_joint
from tests.unit.test_execution_manager import FakeOrderCommandAdapter


def _market_event(*, exchange: str = "nobitex", microprice: float = 100.0) -> MarketEvent:
    book = RawOrderbook(
        exchange=exchange,
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(microprice - 0.5, 5.0),),
        asks=(PriceLevel(microprice + 0.5, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange=exchange,
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
        microprice=microprice,
    )


def _joint_book(*, include_global: bool = True) -> JointTargetBook:
    local = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="test-runner",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 99.0, 1.0),),
        policy="reconcile",
    )
    global_ = None
    if include_global:
        global_ = VenueTargetBook(
            exchange="wallex",
            symbol="BTC_IRT",
            actor_id="test-runner",
            orders=(TargetOrder("wallex", "BTC_IRT", "sell", 101.0, 0.5),),
            policy="replace",
        )
    return JointTargetBook(
        actor_id="test-runner",
        local=local,
        global_=global_,
        mode="liquidity_sourcing",
    )


def _runtime_binding() -> MarketMakerRuntimeConfig:
    return MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT"),
    )


def test_execution_config_parses_per_venue_policies() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}, global_=False),
            "execution": {
                "local": {"planner": "reconcile"},
                "global": {"planner": "replace", "max_order_qty": 0.25, "max_quote_distance": 5.0},
            },
            "liquidity_sourcing": {
                "enabled": True,
                "mode": "liquidity_sourcing",
                "global_live_orders": False,
            },
        }
    )

    assert config.execution.local.planner == "reconcile"
    assert config.execution.global_.planner == "replace"
    assert config.execution.global_.max_order_qty == 0.25
    assert config.liquidity_sourcing.global_live_orders is False


def test_global_live_orders_disabled_strips_global_from_execution_book() -> None:
    book = execution_book_from_joint(
        _joint_book(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
            global_live_orders=False,
        ),
    )

    assert len(book.local.orders) == 1
    assert book.global_ is not None
    assert book.global_.orders == ()


def test_apply_venue_execution_caps_filters_by_qty_and_distance() -> None:
    book = VenueTargetBook(
        exchange="wallex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(
            TargetOrder("wallex", "BTC_IRT", "sell", 106.0, 1.0),
            TargetOrder("wallex", "BTC_IRT", "sell", 101.0, 0.1),
        ),
        policy="replace",
    )
    filtered = apply_venue_execution_caps(
        book,
        mid=100.0,
        config=VenueExecutionConfig(max_order_qty=0.5, max_quote_distance=2.0),
    )

    assert len(filtered.orders) == 1
    assert filtered.orders[0].qty == 0.1


def test_compute_venue_execution_gates_independent_local_and_global() -> None:
    runtime = _runtime_binding()
    frame = StrategyInputFrame(local=_market_event(exchange="nobitex"),
        global_=_market_event(exchange="wallex"),
        global_status="ready",
    )
    gates = compute_venue_execution_gates(
        config=ControllerRuntimeConfig(
            enabled=True,
            actor_id="test-runner",
            observe_only=False,
            action_start_after_steps=0,
        ),
        execution=ExecutionRuntimeConfig(
            local=VenueExecutionConfig(planner="reconcile"),
            global_=VenueExecutionConfig(planner="replace", observe_only=True),
        ),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
            global_live_orders=True,
        ),
        binding=runtime.binding,
        frame_step=1,
        joint_target_book=_joint_book(),
        order_managers={"nobitex": RecordingOrderManager("nobitex"), "wallex": RecordingOrderManager("wallex")},
        control_plane=None,
        frame=frame,
    )

    assert gates["nobitex"].live is True
    assert gates["wallex"].live is False
    assert "venue_observe_only" in gates["wallex"].blocked_reasons


def test_global_stale_market_blocks_only_global_gate() -> None:
    runtime = _runtime_binding()
    frame = StrategyInputFrame(local=_market_event(exchange="nobitex"),
        global_=None,
        global_status="stale",
    )
    gates = compute_venue_execution_gates(
        config=ControllerRuntimeConfig(
            enabled=True,
            actor_id="test-runner",
            observe_only=False,
            action_start_after_steps=0,
        ),
        execution=ExecutionRuntimeConfig(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
            global_live_orders=True,
        ),
        binding=runtime.binding,
        frame_step=1,
        joint_target_book=_joint_book(),
        order_managers={"nobitex": RecordingOrderManager("nobitex"), "wallex": RecordingOrderManager("wallex")},
        control_plane=None,
        frame=frame,
    )

    assert gates["nobitex"].live is True
    assert gates["wallex"].live is False
    assert "stale_global_market_data" in gates["wallex"].blocked_reasons


@pytest.mark.asyncio
async def test_order_manager_sync_venue_target_book_uses_replace_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class TrackingPlanner:
        def plan(self, target, ledger, *, config=None):
            calls.append(target.policy)
            from mm.execution.target_book import ExecutionPlan

            return ExecutionPlan(
                exchange=target.exchange,
                symbol=target.symbol,
                actor_id=target.actor_id,
                policy=target.policy,
            )

    monkeypatch.setattr("mm.execution.order_manager.select_planner", lambda policy: TrackingPlanner())

    manager = OrderManager("wallex", FakeOrderCommandAdapter(), clock_ms=lambda: 1000)
    await manager.sync_venue_target_book(
        VenueTargetBook(
            exchange="wallex",
            symbol="BTC_IRT",
            actor_id="actor-1",
            orders=(TargetOrder("wallex", "BTC_IRT", "sell", 101.0, 0.5),),
            policy="replace",
        )
    )

    assert calls == ["replace"]


@pytest.mark.asyncio
async def test_strategy_runner_routes_both_venues_when_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingManager(RecordingOrderManager):
        def __init__(self, exchange: str) -> None:
            super().__init__(exchange=exchange)
            self.venue_syncs = 0

        async def sync_venue_target_book(self, target):
            self.venue_syncs += 1
            return await super().sync_with_report(
                [Action(actor_id=target.actor_id, orders=tuple())]
            )

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        book = _joint_book()
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=()),
            joint_target_book=book,
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    runtime = _runtime_binding()
    local_manager = TrackingManager("nobitex")
    global_manager = TrackingManager("wallex")
    runner = StrategyRunner(
        config=ControllerRuntimeConfig(
            enabled=True,
            actor_id="test-runner",
            observe_only=False,
            action_start_after_steps=0,
            risk=RiskRuntimeConfig(enabled=True),
            optimizer={"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
        ),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": local_manager, "wallex": global_manager},
        risk_engine=PassThroughRiskEngine(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
            global_live_orders=True,
        ),
        execution=ExecutionRuntimeConfig(
            local=VenueExecutionConfig(planner="reconcile"),
            global_=VenueExecutionConfig(planner="replace"),
        ),
        bus=InMemoryBus(),
    )

    frame = StrategyInputFrame(local=_market_event(exchange="nobitex"),
        global_=_market_event(exchange="wallex"),
        global_status="ready",
    )
    await runner.on_frame(frame)

    assert local_manager.venue_syncs == 1
    assert global_manager.venue_syncs == 1


def test_legacy_hedging_key_ignored_when_live_sourcing_active() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(
                local={"exchange": "nobitex", "symbol": "BTC_IRT"},
                global_={"exchange": "wallex", "symbol": "BTC_IRT"},
            ),
            "hedging": {"enabled": True, "observe_only": False},
            "liquidity_sourcing": {
                "enabled": True,
                "mode": "liquidity_sourcing",
                "global_live_orders": True,
            },
        }
    )

    assert config.liquidity_sourcing.enabled is True
    assert config.liquidity_sourcing.global_live_orders is True


def test_prepare_execution_joint_book_applies_configured_policies() -> None:
    runtime = _runtime_binding()
    frame = StrategyInputFrame(local=_market_event(exchange="nobitex"),
        global_=_market_event(exchange="wallex"),
        global_status="ready",
    )
    prepared = prepare_execution_joint_book(
        _joint_book(),
        execution=ExecutionRuntimeConfig(
            local=VenueExecutionConfig(planner="reconcile"),
            global_=VenueExecutionConfig(planner="replace"),
        ),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
            global_live_orders=True,
        ),
        frame=frame,
        binding=runtime.binding,
    )

    assert prepared.local.policy == "reconcile"
    assert prepared.global_ is not None
    assert prepared.global_.policy == "replace"
