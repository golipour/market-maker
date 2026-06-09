from __future__ import annotations

import pytest

from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.models.execution import Action, Order
from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness, VenueLedger
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
from mm.core.models.strategy import StrategyDecision
from mm.core.payload_schema import (
    CONTROLLER_DIAGNOSTICS_SCHEMA,
    CONTROLLER_DIAGNOSTICS_VERSION,
    EXECUTION_LIFECYCLE_SCHEMA,
    EXECUTION_LIFECYCLE_VERSION,
    EXECUTION_SNAPSHOT_SCHEMA,
    EXECUTION_SNAPSHOT_VERSION,
    JOINT_LEDGER_SNAPSHOT_SCHEMA,
    JOINT_LEDGER_SNAPSHOT_VERSION,
    JOINT_LEDGER_SNAPSHOT_TYPE,
    JOINT_TARGET_BOOK_SCHEMA,
    JOINT_TARGET_BOOK_VERSION,
    MARKET_EVENT_SCHEMA,
    MARKET_EVENT_VERSION,
    PRIVATE_STATE_SNAPSHOT_SCHEMA,
    PRIVATE_STATE_SNAPSHOT_VERSION,
    PRIVATE_STATE_SNAPSHOT_TYPE,
    STRATEGY_DECISION_SCHEMA,
    STRATEGY_DECISION_VERSION,
    STRATEGY_DECISION_TYPE,
    TARGET_ORDER_SCHEMA,
    TARGET_ORDER_VERSION,
    VENUE_LEDGER_SNAPSHOT_SCHEMA,
    VENUE_LEDGER_SNAPSHOT_VERSION,
    VENUE_LEDGER_SNAPSHOT_TYPE,
    VENUE_TARGET_BOOK_SCHEMA,
    VENUE_TARGET_BOOK_VERSION,
    assert_payload_schema,
    execution_lifecycle_event,
    execution_snapshot_event,
    joint_target_book_event,
    market_event_event,
    market_event_from_payload,
    private_state_snapshot_from_payload,
)
from mm.diagnostics.models import CONTROLLER_DIAGNOSTICS_TYPE, ControllerDiagnostics
from mm.execution.order_manager import ExecutionSnapshot
from mm.execution.router import OrderRouter
from mm.execution.target_book import CancelCommand, ExecutionPlan, PlaceCommand
from tests.unit.test_execution_manager import FakeOrderCommandAdapter


from tests.unit.conftest import joint_frame_for_input


def test_market_event_payload_is_versioned_and_round_trips() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook

    event = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=2000,
            bids=(PriceLevel(100.0, 1.0),),
            asks=(PriceLevel(101.0, 2.0),),
            is_snapshot=True,
        ),
        trades=(),
        buy_volume=0.1,
        sell_volume=0.2,
        vwap=100.66,
        imbalance=-0.1,
        microprice=100.5,
    )
    payload = market_event_event(event)

    assert_payload_schema(payload, MARKET_EVENT_SCHEMA, MARKET_EVENT_VERSION)
    round_tripped = market_event_from_payload(payload)
    assert round_tripped is not None
    assert round_tripped.exchange == event.exchange
    assert round_tripped.t_end_ms == event.t_end_ms
    assert round_tripped.orderbook.bids[0].price == 100.0


def test_target_book_models_emit_versioned_payloads() -> None:
    order = TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 0.5)
    venue = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(order,),
        execution_plan_id="plan-1",
    )
    joint = JointTargetBook(actor_id="actor-1", local=venue, target_book_id="target-1")

    order_payload = order.as_payload()
    venue_payload = venue.as_payload()
    joint_payload = joint.as_payload()

    assert_payload_schema(order_payload, TARGET_ORDER_SCHEMA, TARGET_ORDER_VERSION)
    assert_payload_schema(venue_payload, VENUE_TARGET_BOOK_SCHEMA, VENUE_TARGET_BOOK_VERSION)
    assert_payload_schema(joint_payload, JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION)
    assert_payload_schema(venue_payload["orders"][0], TARGET_ORDER_SCHEMA, TARGET_ORDER_VERSION)


def test_joint_target_book_bus_event_includes_type_and_schema() -> None:
    venue = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 0.5),),
    )
    event = joint_target_book_event(
        JointTargetBook(actor_id="actor-1", local=venue),
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2_000,
    )

    assert event["type"] == "JointTargetBook"
    assert_payload_schema(event, JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION)


def test_execution_snapshot_payload_is_versioned() -> None:
    payload = ExecutionSnapshot(
        exchange="nobitex",
        timestamp_ms=1_000,
        live_order_count=0,
        live_orders={},
    ).as_payload()

    assert payload["type"] == "ExecutionSnapshot"
    assert_payload_schema(payload, EXECUTION_SNAPSHOT_SCHEMA, EXECUTION_SNAPSHOT_VERSION)


@pytest.mark.asyncio
async def test_router_events_emit_execution_lifecycle_schema() -> None:
    router = OrderRouter("nobitex", FakeOrderCommandAdapter())
    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="replace",
        cancel=(CancelCommand("live-1", "actor-1", "replace"),),
        place=(PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)),),
        frame_id="frame-1",
        decision_id="decision-1",
        target_book_id="target-1",
        execution_plan_id="plan-1",
    )

    events: list[dict] = []

    async def capture(_topic: str, event: dict) -> None:
        events.append(event)

    router._publish = capture  # type: ignore[method-assign]
    await router.execute(plan)

    assert events
    for event in events:
        assert_payload_schema(event, EXECUTION_LIFECYCLE_SCHEMA, EXECUTION_LIFECYCLE_VERSION)


def test_replay_rejects_unversioned_execution_payload() -> None:
    from mm.persistence.replay import _MutableExecutionPlanState

    plan = _MutableExecutionPlanState("plan-1")
    with pytest.raises(ValueError, match="expected payload mm.execution_lifecycle"):
        plan.apply(
            "PlaceOrdersRequested",
            {"type": "PlaceOrdersRequested", "orders": []},
        )


def test_replay_accepts_versioned_execution_payload() -> None:
    from mm.persistence.replay import _MutableExecutionPlanState

    plan = _MutableExecutionPlanState("plan-1")
    plan.apply(
        "PlaceOrdersRequested",
        execution_lifecycle_event(
            "PlaceOrdersRequested",
            exchange="nobitex",
            orders=[{"side": "buy", "price": 100.0, "qty": 0.5}],
        ),
    )
    frozen = plan.freeze()
    assert frozen.place_requested_orders == ({"side": "buy", "price": 100.0, "qty": 0.5},)


def test_replay_rejects_unversioned_target_book_payload() -> None:
    from mm.persistence.replay import reconstruct_lineage_state
    from mm.core.events import EventEnvelope

    chain = type(
        "Chain",
        (),
        {
            "frame_id": "frame-1",
            "decisions": (),
            "diagnostics": (),
            "target_books": (
                EventEnvelope(
                    topic="strategy:target_books:actor-1",
                    event_type="JointTargetBook",
                    payload={"type": "JointTargetBook", "actor_id": "actor-1"},
                ),
            ),
            "execution_events": (),
            "other_events": (),
            "reconstruct_state": lambda self: reconstruct_lineage_state(self),
        },
    )()

    with pytest.raises(ValueError, match="expected payload mm.joint_target_book"):
        reconstruct_lineage_state(chain)


def test_execution_snapshot_helper_sets_type() -> None:
    payload = execution_snapshot_event(exchange="nobitex", timestamp_ms=1_000, live_order_count=0, live_orders={})
    assert payload["type"] == "ExecutionSnapshot"
    assert_payload_schema(payload, EXECUTION_SNAPSHOT_SCHEMA, EXECUTION_SNAPSHOT_VERSION)


def test_strategy_decision_payload_is_versioned() -> None:
    decision = StrategyDecision(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2_000,
        frame_step=1,
        action=Action(actor_id="actor-1", orders=()),
        frame_id="frame-1",
        decision_id="decision-1",
        target_book_id="target-1",
    )
    payload = decision.as_payload()

    assert payload["type"] == STRATEGY_DECISION_TYPE
    assert_payload_schema(payload, STRATEGY_DECISION_SCHEMA, STRATEGY_DECISION_VERSION)


def test_private_state_snapshot_round_trips_through_versioned_payload() -> None:
    order = Order("nobitex", "BTC_IRT", "buy", 100.0, 0.5, client_order_id="cid-1")
    open_order = OpenOrder(
        exchange_order_id="ex-1",
        actor_id="actor-1",
        order=order,
        remaining_qty=0.5,
        filled_qty=0.0,
        status="open",
        timestamp_ms=2_000,
    )
    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2_000,
        balances={"IRT": Balance("nobitex", "IRT", 1_000.0, 0.0, 1_000.0, 2_000)},
        open_orders={"ex-1": open_order},
    )

    payload = snapshot.as_payload()

    assert payload["type"] == PRIVATE_STATE_SNAPSHOT_TYPE
    assert_payload_schema(payload, PRIVATE_STATE_SNAPSHOT_SCHEMA, PRIVATE_STATE_SNAPSHOT_VERSION)
    restored = private_state_snapshot_from_payload(payload)
    assert restored.exchange == "nobitex"
    assert restored.open_orders["ex-1"].order.price == 100.0


def test_ledger_snapshot_payloads_are_versioned() -> None:
    venue = VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        available_quote=1_000.0,
    )
    joint = JointLedgerSnapshot(
        local=venue,
        reference_price=100.0,
        created_at_ms=2_000,
        freshness=LedgerFreshness(ledger_fresh=True),
    )

    venue_payload = venue.as_payload()
    joint_payload = joint.as_payload()

    assert venue_payload["type"] == VENUE_LEDGER_SNAPSHOT_TYPE
    assert_payload_schema(venue_payload, VENUE_LEDGER_SNAPSHOT_SCHEMA, VENUE_LEDGER_SNAPSHOT_VERSION)
    assert joint_payload["type"] == JOINT_LEDGER_SNAPSHOT_TYPE
    assert_payload_schema(joint_payload, JOINT_LEDGER_SNAPSHOT_SCHEMA, JOINT_LEDGER_SNAPSHOT_VERSION)


def test_controller_diagnostics_payload_is_versioned() -> None:
    payload = ControllerDiagnostics(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2_000,
        frame_step=1,
        created_at_ms=1_500,
    ).as_payload()

    assert payload["type"] == CONTROLLER_DIAGNOSTICS_TYPE
    assert_payload_schema(payload, CONTROLLER_DIAGNOSTICS_SCHEMA, CONTROLLER_DIAGNOSTICS_VERSION)
