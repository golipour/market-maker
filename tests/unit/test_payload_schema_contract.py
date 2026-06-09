from __future__ import annotations

import pytest

from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.payload_schema import (
    EXECUTION_LIFECYCLE_SCHEMA,
    EXECUTION_LIFECYCLE_VERSION,
    EXECUTION_SNAPSHOT_SCHEMA,
    EXECUTION_SNAPSHOT_VERSION,
    JOINT_TARGET_BOOK_SCHEMA,
    JOINT_TARGET_BOOK_VERSION,
    TARGET_ORDER_SCHEMA,
    TARGET_ORDER_VERSION,
    VENUE_TARGET_BOOK_SCHEMA,
    VENUE_TARGET_BOOK_VERSION,
    assert_payload_schema,
    execution_lifecycle_event,
    execution_snapshot_event,
    joint_target_book_event,
)
from mm.execution.order_manager import ExecutionSnapshot
from mm.execution.router import OrderRouter
from mm.core.models.execution import Order
from mm.execution.target_book import CancelCommand, ExecutionPlan, PlaceCommand
from tests.unit.test_execution_manager import FakeOrderCommandAdapter


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
