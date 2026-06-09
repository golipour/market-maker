from __future__ import annotations

import json

from mm.core.events import envelope_event
from mm.core.payload_schema import (
    EXECUTION_LIFECYCLE_SCHEMA,
    EXECUTION_LIFECYCLE_VERSION,
    EXECUTION_SNAPSHOT_SCHEMA,
    EXECUTION_SNAPSHOT_VERSION,
    JOINT_TARGET_BOOK_SCHEMA,
    JOINT_TARGET_BOOK_VERSION,
    execution_lifecycle_event,
    execution_snapshot_event,
    stamped,
)
from mm.persistence.replay import read_lineage_chains


def test_replay_lineage_reconstructs_decision_to_execution_chain(tmp_path) -> None:
    frame_id = "frame:test-runner:nobitex:btc_irt:2000:1"
    decision_id = "decision:frame_test-runner_nobitex_btc_irt_2000_1"
    target_book_id = "target:frame_test-runner_nobitex_btc_irt_2000_1"
    execution_plan_id = "execplan:target_frame_test-runner_nobitex_btc_irt_2000_1:local:nobitex"
    rows = [
        _row(
            "strategy:decisions",
            {
                "type": "StrategyDecision",
                "actor_id": "test-runner",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "t_end_ms": 2_000,
                "frame_step": 1,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
            },
        ),
        _row(
            "strategy:target_books:test-runner",
            {
                "type": "JointTargetBook",
                "actor_id": "test-runner",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "t_end_ms": 2_000,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "local": {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "actor_id": "test-runner",
                    "execution_plan_id": execution_plan_id,
                    "orders": [{"side": "buy", "price": 100.0, "qty": 0.5}],
                },
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersRequested",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "orders": [{"side": "buy", "price": 100.0, "qty": 0.5}],
                "timestamp_ms": 2_100,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersAccepted",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["ex-1"],
                "orders": [{"side": "buy", "price": 100.0, "qty": 0.5, "exchange_order_id": "ex-1"}],
                "timestamp_ms": 2_110,
            },
        ),
        _row(
            "strategy:diagnostics:test-runner",
            {
                "type": "ControllerDiagnostics",
                "actor_id": "test-runner",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "t_end_ms": 2_000,
                "frame_step": 1,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
            },
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    chains = read_lineage_chains(path)

    assert set(chains) == {frame_id}
    chain = chains[frame_id]
    assert chain.decision_ids == (decision_id,)
    assert chain.target_book_ids == (target_book_id,)
    assert chain.execution_plan_ids == (execution_plan_id,)
    assert [event.event_type for event in chain.decisions] == ["StrategyDecision"]
    assert [event.event_type for event in chain.target_books] == ["JointTargetBook"]
    assert [event.event_type for event in chain.diagnostics] == ["ControllerDiagnostics"]
    plan_events = chain.execution_events_for_plan(execution_plan_id)
    assert [event.event_type for event in plan_events] == [
        "PlaceOrdersRequested",
        "PlaceOrdersAccepted",
    ]
    assert plan_events[-1].payload["order_ids"] == ["ex-1"]
    state = chain.reconstruct_state()
    assert state.frame_id == frame_id
    assert state.decision_id == decision_id
    assert state.target_book_id == target_book_id
    assert state.latest_decision is not None
    assert set(state.execution_plans) == {execution_plan_id}
    plan_state = state.execution_plans[execution_plan_id]
    assert plan_state.exchange == "nobitex"
    assert plan_state.symbol == "BTC_IRT"
    assert plan_state.target_orders == ({"side": "buy", "price": 100.0, "qty": 0.5},)
    assert plan_state.place_requested_orders == ({"side": "buy", "price": 100.0, "qty": 0.5},)
    assert plan_state.place_accepted_order_ids == ("ex-1",)
    assert not state.confirmed_live_orders
    assert not state.pending_places


def test_replay_lineage_distinguishes_pending_from_confirmed_execution_state(tmp_path) -> None:
    frame_id = "frame:test-runner:nobitex:btc_irt:2000:1"
    decision_id = "decision:frame_test-runner_nobitex_btc_irt_2000_1"
    target_book_id = "target:frame_test-runner_nobitex_btc_irt_2000_1"
    execution_plan_id = "execplan:target_frame_test-runner_nobitex_btc_irt_2000_1:local:nobitex"
    path = tmp_path / "events.jsonl"
    rows = [
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersAccepted",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["ex-1"],
                "timestamp_ms": 2_100,
            },
        ),
        _row(
            "execution:state:nobitex",
            {
                "type": "ExecutionSnapshot",
                "exchange": "nobitex",
                "timestamp_ms": 2_110,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "live_order_count": 0,
                "live_orders": {},
                "pending_place_count": 1,
                "pending_places": {
                    "ex-1": {
                        "operation": "place",
                        "exchange_order_id": "ex-1",
                        "actor_id": "test-runner",
                        "accepted_ms": 2_100,
                    }
                },
                "pending_cancel_count": 0,
                "pending_cancels": {},
            },
        ),
        _row(
            "execution:state:nobitex",
            {
                "type": "ExecutionSnapshot",
                "exchange": "nobitex",
                "timestamp_ms": 2_200,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "live_order_count": 1,
                "live_orders": {
                    "ex-1": {
                        "exchange_order_id": "ex-1",
                        "actor_id": "test-runner",
                        "remaining_qty": 0.5,
                        "filled_qty": 0.0,
                        "status": "open",
                    }
                },
                "pending_place_count": 0,
                "pending_places": {},
                "pending_cancel_count": 0,
                "pending_cancels": {},
            },
        ),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    [chain] = read_lineage_chains(path).values()

    snapshots = [event.payload for event in chain.execution_events if event.event_type == "ExecutionSnapshot"]
    assert snapshots[0]["pending_place_count"] == 1
    assert snapshots[0]["live_order_count"] == 0
    assert "ex-1" in snapshots[0]["pending_places"]
    assert snapshots[1]["pending_place_count"] == 0
    assert snapshots[1]["live_order_count"] == 1
    assert "ex-1" in snapshots[1]["live_orders"]
    state = chain.reconstruct_state()
    plan_state = state.execution_plans[execution_plan_id]
    assert plan_state.place_accepted_order_ids == ("ex-1",)
    assert not plan_state.pending_places
    assert not state.pending_places
    assert list(plan_state.confirmed_live_orders) == ["ex-1"]
    assert list(state.confirmed_live_orders) == ["ex-1"]


def test_replay_state_reconstructs_cancel_and_failure_outcomes(tmp_path) -> None:
    frame_id = "frame:test-runner:nobitex:btc_irt:3000:2"
    decision_id = "decision:frame_test-runner_nobitex_btc_irt_3000_2"
    target_book_id = "target:frame_test-runner_nobitex_btc_irt_3000_2"
    execution_plan_id = "execplan:target_frame_test-runner_nobitex_btc_irt_3000_2:local:nobitex"
    rows = [
        _row(
            "execution:orders:nobitex",
            {
                "type": "CancelOrdersRequested",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["live-1"],
                "timestamp_ms": 3_010,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "CancelOrdersAccepted",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["live-1"],
                "timestamp_ms": 3_020,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersFailed",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "orders": [{"side": "sell", "price": 110.0, "qty": 0.2}],
                "error": "exchange timeout",
                "timestamp_ms": 3_030,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "OrderValidationRejected",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "side": "buy",
                "price": 1.0,
                "qty": 0.001,
                "reason": "min_notional",
                "timestamp_ms": 3_040,
            },
        ),
        _row(
            "execution:state:nobitex",
            {
                "type": "ExecutionSnapshot",
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "frame_id": frame_id,
                "decision_id": decision_id,
                "target_book_id": target_book_id,
                "execution_plan_id": execution_plan_id,
                "live_order_count": 1,
                "live_orders": {"live-1": {"exchange_order_id": "live-1", "status": "open"}},
                "pending_place_count": 0,
                "pending_places": {},
                "pending_cancel_count": 1,
                "pending_cancels": {
                    "live-1": {
                        "operation": "cancel",
                        "exchange_order_id": "live-1",
                        "accepted_ms": 3_020,
                    }
                },
                "timestamp_ms": 3_050,
            },
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    [chain] = read_lineage_chains(path).values()
    state = chain.reconstruct_state()
    plan_state = state.execution_plans[execution_plan_id]

    assert plan_state.cancel_requested_order_ids == ("live-1",)
    assert plan_state.cancel_accepted_order_ids == ("live-1",)
    assert plan_state.place_failed_errors == ("exchange timeout",)
    assert plan_state.validation_rejections[0]["reason"] == "min_notional"
    assert plan_state.has_failures
    assert state.has_execution_failures
    assert list(state.pending_cancels) == ["live-1"]
    assert list(state.confirmed_live_orders) == ["live-1"]


def _row(topic: str, payload: dict) -> dict:
    envelope = envelope_event(topic, _versioned_payload(payload))
    return {"envelope_json": envelope.as_dict()}


def _versioned_payload(payload: dict) -> dict:
    event_type = payload.get("type")
    fields = {key: value for key, value in payload.items() if key != "type"}
    if event_type == "JointTargetBook":
        return stamped(JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION, type=event_type, **fields)
    if event_type == "ExecutionSnapshot":
        return execution_snapshot_event(**fields)
    if event_type:
        return execution_lifecycle_event(str(event_type), **fields)
    return payload
