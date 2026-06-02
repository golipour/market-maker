"""Phase 4 tests: target-book strategy output, pipeline stages, and persistence."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.core.adapters.joint import joint_target_book_to_action
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.private import OpenOrder, PrivateStateSnapshot
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.order_constraints import OrderConstraints
from mm.core.wallet import make_balance, normalize_snapshot_balances_for_open_orders
from mm.execution.memory import RecordingOrderManager
from mm.execution.plan import apply_execution_stages_to_pipeline
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    OrderbookGridConfig,
    RiskRuntimeConfig,
)
from mm.runtime.engine import _default_persistence_topics
from mm.strategies.target_book_builder import (
    build_joint_target_book_from_sigmoid,
    build_venue_target_book_from_orders,
)
from tests.unit.conftest import joint_frame_for_input


def _market_event(*, exchange: str = "nobitex", symbol: str = "BTC_IRT") -> MarketEvent:
    book = RawOrderbook(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange=exchange,
        symbol=symbol,
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


def test_mpc_state_uses_normalized_wallet_total_including_locked_collateral_for_objective() -> None:
    from mm.strategies.mpc_engine import _state_from_frame

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    snapshot = normalize_snapshot_balances_for_open_orders(
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=2_000,
            balances={
                "BTC": make_balance("nobitex", "BTC", 1.0, 0.0, 2_000, total=1.0),
                "IRT": make_balance("nobitex", "IRT", 10_000.0, 0.0, 2_000),
            },
            open_orders={
                "sell-1": OpenOrder(
                    exchange_order_id="sell-1",
                    actor_id="mpc-controller",
                    order=Order("nobitex", "BTC_IRT", "sell", 100.0, 2.0),
                    remaining_qty=2.0,
                    filled_qty=0.0,
                    status="open",
                    timestamp_ms=2_000,
                )
            },
        ),
        base_asset="BTC",
        quote_asset="IRT",
        trading_symbol="BTC_IRT",
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        local_private=snapshot,
    )

    state = _state_from_frame(frame, runtime.binding, spread_max=100.0)

    assert state.inventory == pytest.approx(2.0)
    assert state.cash == pytest.approx(10_000.0)


def _sigmoid_action(*, bid_qty: float = 1.0, ask_qty: float = 0.5) -> SimpleNamespace:
    def depth_bid(spread: float) -> float:
        return bid_qty * (1.0 + spread / 20.0) if spread <= 20.0 else bid_qty * 2.0

    def depth_ask(spread: float) -> float:
        return ask_qty * (1.0 + spread / 20.0) if spread <= 20.0 else ask_qty * 2.0

    return SimpleNamespace(
        depth_bid=depth_bid,
        depth_ask=depth_ask,
        effective_depth_bid=lambda spread: max(0.0, depth_bid(spread) - depth_bid(0.0)),
        effective_depth_ask=lambda spread: max(0.0, depth_ask(spread) - depth_ask(0.0)),
    )


def _grid() -> SimpleNamespace:
    return SimpleNamespace(
        spread_ladder=lambda: (20.0,),
    )


def _constraints() -> OrderConstraints:
    return OrderConstraints.from_spec(
        InstrumentSpec(
            exchange="nobitex",
            symbol="BTC_IRT",
            base_asset="BTC",
            quote_asset="IRT",
        ),
        max_orders_per_side=4,
    )


def test_build_joint_target_book_from_sigmoid_emits_local_book() -> None:
    joint_book, pipeline, ladder, local_orders = build_joint_target_book_from_sigmoid(
        actor_id="actor-1",
        local_event=_market_event(),
        sigmoid_action=_sigmoid_action(),
        orderbook_grid=_grid(),
        local_constraints=_constraints(),
    )

    assert isinstance(joint_book, JointTargetBook)
    assert joint_book.mode == "single_venue"
    assert joint_book.global_ is None
    assert joint_book.local.exchange == "nobitex"
    assert len(joint_book.local.orders) == len(local_orders)
    assert len(ladder) >= 1
    assert pipeline.target_book.order_count == len(local_orders)
    assert pipeline.target_book.orders
    assert pipeline.venue_quantized.order_count == 0


def test_build_joint_target_book_from_sigmoid_merges_global_action_diagnostics() -> None:
    joint_book, pipeline, ladder, local_orders = build_joint_target_book_from_sigmoid(
        actor_id="actor-1",
        local_event=_market_event(exchange="nobitex", symbol="BTC_IRT"),
        sigmoid_action=_sigmoid_action(bid_qty=1.0, ask_qty=0.5),
        orderbook_grid=_grid(),
        local_constraints=_constraints(),
        mode="liquidity_sourcing",
        global_event=_market_event(exchange="binance", symbol="BTC_USDT"),
        global_constraints=OrderConstraints.from_spec(
            InstrumentSpec(
                exchange="binance",
                symbol="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
            ),
            max_orders_per_side=4,
        ),
        global_sigmoid_action=_sigmoid_action(bid_qty=0.25, ask_qty=0.75),
    )

    assert joint_book.global_ is not None
    assert len(joint_book.local.orders) == len(local_orders)
    assert len(ladder) > len(local_orders)
    assert pipeline.target_book.order_count == len(joint_book.local.orders) + len(joint_book.global_.orders)
    assert {order["exchange"] for order in pipeline.ladder.orders} == {"nobitex", "binance"}
    assert {order["exchange"] for order in pipeline.capped.orders} == {"nobitex", "binance"}


def test_target_book_builder_does_not_apply_venue_minimums() -> None:
    constraints = OrderConstraints(
        price_step=10.0,
        qty_step=10.0,
        min_qty=10.0,
        min_notional=1_000_000.0,
        max_orders_per_side=4,
    )
    joint_book, pipeline, ladder, local_orders = build_joint_target_book_from_sigmoid(
        actor_id="actor-1",
        local_event=_market_event(),
        sigmoid_action=_sigmoid_action(bid_qty=0.01, ask_qty=0.01),
        orderbook_grid=_grid(),
        local_constraints=constraints,
    )

    assert ladder
    assert local_orders
    assert joint_book.local.orders
    assert pipeline.venue_quantized.order_count == 0
    assert pipeline.target_book.order_count == len(local_orders)


def test_joint_target_book_converts_to_action() -> None:
    book = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook(
            exchange="nobitex",
            symbol="BTC_IRT",
            actor_id="actor-1",
            orders=(
                TargetOrder("nobitex", "BTC_IRT", "buy", 99.0, 1.0),
                TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 0.5),
            ),
        ),
    )
    action = joint_target_book_to_action(book)

    assert isinstance(action, Action)
    assert len(action.orders) == 2
    assert joint_target_book_to_action(book).orders == action.orders


def test_action_pipeline_phase4_stages() -> None:
    from mm.execution.plan import ActionPipelineReport, ActionStageReport, ExecutionSyncReport

    report = ActionPipelineReport(
        target_book=ActionStageReport(order_count=2, buy_qty=1.0, sell_qty=1.0),
        risk_approved_target=ActionStageReport(order_count=2),
        planned_cancel=ActionStageReport(order_count=1),
        planned_place=ActionStageReport(order_count=1),
        rules_valid=ActionStageReport(order_count=1),
        router_accepted=ActionStageReport(order_count=1),
    )

    assert report.target_book.order_count == 2
    assert report.risk_approved_target.order_count == 2
    assert report.planned_cancel.order_count == 1
    assert report.planned_place.order_count == 1
    assert report.rules_valid.order_count == 1
    assert report.router_accepted.order_count == 1

    original_target = report.target_book
    execution = ExecutionSyncReport(
        target=ActionStageReport(order_count=99),
        kept=ActionStageReport(order_count=1),
        canceled=ActionStageReport(order_count=1),
        to_place=ActionStageReport(order_count=1),
        limit_valid=ActionStageReport(
            order_count=1,
            orders=(
                {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "side": "buy",
                    "price": 99.0,
                    "qty": 1.0,
                    "notional": 99.0,
                    "order_type": "limit",
                    "client_order_id": None,
                },
            ),
            rejections=(
                {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "side": "sell",
                    "price": 101.0,
                    "qty": 0.0,
                    "notional": 0.0,
                    "order_type": "limit",
                    "client_order_id": None,
                    "reason": "rules_invalid_price_or_qty",
                    "stage": "rules",
                },
            ),
        ),
        accepted=ActionStageReport(order_count=1),
        venue_rejection_counts={"rules_invalid_price_or_qty": 1},
    )
    apply_execution_stages_to_pipeline(report, execution, executed=True)
    assert report.target_book == original_target
    assert report.planned_cancel.order_count == 1
    assert report.planned_place.order_count == 1
    assert report.rules_valid.order_count == 1
    assert report.rules_valid.orders == execution.limit_valid.orders
    assert report.rules_valid.rejections == execution.limit_valid.rejections
    assert report.rules_valid.rejection_counts == {"rules_invalid_price_or_qty": 1}
    assert report.router_accepted.order_count == 1


def test_execution_report_filter_keeps_local_symbol_only() -> None:
    from mm.execution.plan import ActionStageReport, ExecutionSyncReport, filter_execution_sync_report

    local_order = {
        "exchange": "nobitex",
        "symbol": "USDT_IRT",
        "side": "buy",
        "price": 100.0,
        "qty": 2.0,
        "notional": 200.0,
        "order_type": "limit",
        "client_order_id": None,
    }
    global_order = {
        "exchange": "wallex",
        "symbol": "USDT_IRT",
        "side": "sell",
        "price": 110.0,
        "qty": 3.0,
        "notional": 330.0,
        "order_type": "limit",
        "client_order_id": None,
    }
    report = ExecutionSyncReport(
        to_place=ActionStageReport(order_count=2, orders=(local_order, global_order)),
        limit_valid=ActionStageReport(
            order_count=1,
            orders=(local_order,),
            rejections=(
                {
                    **global_order,
                    "reason": "rules_min_notional",
                    "stage": "rules",
                },
            ),
            rejection_counts={"rules_min_notional": 1},
        ),
        canceled=ActionStageReport(order_count=2, orders=(local_order, global_order)),
        cancel_count=2,
        placed=True,
    )

    filtered = filter_execution_sync_report(report, exchange="nobitex", symbol="USDT_IRT")

    assert filtered.to_place.order_count == 1
    assert filtered.to_place.orders == (local_order,)
    assert filtered.canceled.order_count == 1
    assert filtered.cancel_count == 1
    assert filtered.limit_valid.order_count == 1
    assert filtered.limit_valid.rejections == ()
    assert filtered.venue_rejection_counts == {}


def test_default_persistence_topics_include_target_books() -> None:
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="mpc-controller"),
    )
    topics = _default_persistence_topics(runtime)
    assert "strategy:target_books:mpc-controller" in topics


@pytest.mark.asyncio
async def test_mpc_engine_emits_joint_target_book(monkeypatch: pytest.MonkeyPatch) -> None:
    from mm.strategies.mpc_engine import MpcStrategyEngine

    class _FakeController:
        last_diagnostics = None

        def optimize(self, state):
            return _sigmoid_action()

    monkeypatch.setattr(
        "mm.strategies.mpc_engine._build_controller",
        lambda config, binding, **kwargs: _FakeController(),
    )
    monkeypatch.setattr("mm.strategies.mpc_engine.score_frame_models", lambda **kwargs: {})
    monkeypatch.setattr("mm.strategies.mpc_engine._update_global_vwap_price_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("mm.strategies.mpc_engine._update_market_flow_model", lambda *args: None)
    monkeypatch.setattr("mm.strategies.mpc_engine._update_residual_price_model", lambda *args: None)
    monkeypatch.setattr(
        "mm.strategies.mpc_engine.local_constraints_from_spec",
        lambda spec, *, max_orders_per_side: _constraints(),
    )

    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            actor_id="mpc-controller",
            orderbook_grid=OrderbookGridConfig(spread_max=200.0, spread_step=20.0),
            optimizer={
                "local": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 10.0},
                "global": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 10.0},
            },
        ),
    )
    engine = MpcStrategyEngine(config=runtime.controller, binding=runtime.binding)
    decision = engine.decide(StrategyInputFrame(local=_market_event()), frame_step=1)

    assert decision.joint_target_book is not None
    assert decision.joint_target_book.local.exchange == "nobitex"
    assert len(decision.action.orders) == len(decision.joint_target_book.local.orders)
    assert decision.action_pipeline_report is not None
    assert decision.action_pipeline_report.target_book.order_count >= 1


@pytest.mark.asyncio
async def test_strategy_runner_publishes_target_book_event(monkeypatch: pytest.MonkeyPatch) -> None:
    order = Order("nobitex", "BTC_IRT", "buy", 99.0, 1.0, client_order_id="c1")
    joint_book = JointTargetBook(
        actor_id="test-runner",
        local=build_venue_target_book_from_orders(actor_id="test-runner", orders=(order,)),
    )

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return StrategyDecision(
            actor_id="test-runner",
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=Action(actor_id="test-runner", orders=(order,)),
            joint_target_book=joint_book,
        )

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    target_queue = bus.subscribe_queue("strategy:target_books:test-runner")
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=ControllerRuntimeConfig(
            enabled=True,
            actor_id="test-runner",
            observe_only=True,
            risk=RiskRuntimeConfig(enabled=True),
            optimizer={
                "local": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
                "global": {"quote_size_min_per_side": 1.0, "quote_size_max_per_side": 10.0},
            },
        ),
        binding=runtime.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": RecordingOrderManager(exchange="nobitex")},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )

    frame = StrategyInputFrame(local=_market_event())
    await runner.on_joint_frame(joint_frame_for_input(frame, binding=runtime.binding))

    event = await asyncio.wait_for(target_queue.get(), timeout=1.0)
    assert event["type"] == "JointTargetBook"
    assert event["actor_id"] == "test-runner"
    assert event["local"]["orders"][0]["side"] == "buy"
