from __future__ import annotations

import json

from mm.core.events import decode_envelope, envelope_event
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
from mm.persistence.replay import (
    lineage_chains_from_envelopes,
    verify_lineage_fidelity,
    verify_persisted_lineage_fidelity,
)


def test_verify_lineage_fidelity_passes_for_coherent_frame(tmp_path) -> None:
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
                "timestamp_ms": 2_110,
            },
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    reports = verify_persisted_lineage_fidelity(path)

    assert set(reports) == {frame_id}
    assert reports[frame_id].ok


def test_verify_lineage_fidelity_detects_decision_id_mismatch() -> None:
    frame_id = "frame:test:nobitex:btc_irt:1000:1"
    rows = [
        _row(
            "strategy:decisions",
            {
                "type": "StrategyDecision",
                "frame_id": frame_id,
                "decision_id": "decision:a",
                "target_book_id": "target:a",
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersRequested",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "decision_id": "decision:b",
                "target_book_id": "target:a",
                "execution_plan_id": "execplan:1",
                "orders": [],
                "timestamp_ms": 1_100,
            },
        ),
    ]
    [chain] = _chains_from_rows(rows).values()

    report = verify_lineage_fidelity(chain)

    assert not report.ok
    assert any(v.code == "decision_id_mismatch" for v in report.violations)


def test_verify_lineage_fidelity_detects_place_request_not_from_target() -> None:
    frame_id = "frame:test:nobitex:btc_irt:1000:1"
    execution_plan_id = "execplan:1"
    rows = [
        _row(
            "strategy:target_books:test-runner",
            {
                "type": "JointTargetBook",
                "frame_id": frame_id,
                "decision_id": "decision:a",
                "target_book_id": "target:a",
                "local": {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
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
                "decision_id": "decision:a",
                "target_book_id": "target:a",
                "execution_plan_id": execution_plan_id,
                "orders": [{"side": "sell", "price": 110.0, "qty": 0.2}],
                "timestamp_ms": 1_100,
            },
        ),
    ]
    [chain] = _chains_from_rows(rows).values()

    report = verify_lineage_fidelity(chain)

    assert not report.ok
    assert any(v.code == "place_request_not_from_target" for v in report.violations)


def test_verify_lineage_fidelity_detects_snapshot_count_mismatch() -> None:
    frame_id = "frame:test:nobitex:btc_irt:1000:1"
    execution_plan_id = "execplan:1"
    rows = [
        _row(
            "execution:state:nobitex",
            {
                "type": "ExecutionSnapshot",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "live_order_count": 2,
                "live_orders": {"ex-1": {"exchange_order_id": "ex-1"}},
                "pending_place_count": 0,
                "pending_places": {},
                "pending_cancel_count": 0,
                "pending_cancels": {},
                "timestamp_ms": 1_100,
            },
        ),
    ]
    [chain] = _chains_from_rows(rows).values()

    report = verify_lineage_fidelity(chain)

    assert not report.ok
    assert any(v.code == "execution_snapshot_count_mismatch" for v in report.violations)


def test_verify_lineage_fidelity_detects_lifecycle_order_violation() -> None:
    frame_id = "frame:test:nobitex:btc_irt:1000:1"
    execution_plan_id = "execplan:1"
    rows = [
        _row(
            "strategy:target_books:test-runner",
            {
                "type": "JointTargetBook",
                "frame_id": frame_id,
                "decision_id": "decision:a",
                "target_book_id": "target:a",
                "local": {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "execution_plan_id": execution_plan_id,
                    "orders": [{"side": "buy", "price": 100.0, "qty": 0.5}],
                },
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersAccepted",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["ex-1"],
                "timestamp_ms": 1_000,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersRequested",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "orders": [{"side": "buy", "price": 100.0, "qty": 0.5}],
                "timestamp_ms": 1_100,
            },
        ),
    ]
    [chain] = _chains_from_rows(rows).values()

    report = verify_lineage_fidelity(chain)

    assert not report.ok
    assert any(v.code == "execution_lifecycle_order" for v in report.violations)


def test_verify_lineage_fidelity_allows_validation_rejection_to_remove_target_order() -> None:
    frame_id = "frame:test:nobitex:btc_irt:1000:1"
    execution_plan_id = "execplan:1"
    rows = [
        _row(
            "strategy:target_books:test-runner",
            {
                "type": "JointTargetBook",
                "frame_id": frame_id,
                "decision_id": "decision:a",
                "target_book_id": "target:a",
                "local": {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "execution_plan_id": execution_plan_id,
                    "orders": [
                        {"side": "buy", "price": 100.0, "qty": 0.5},
                        {"side": "buy", "price": 1.0, "qty": 0.001},
                    ],
                },
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "OrderValidationRejected",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "side": "buy",
                "price": 1.0,
                "qty": 0.001,
                "reason": "min_notional",
                "timestamp_ms": 1_050,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersRequested",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "orders": [{"side": "buy", "price": 100.0, "qty": 0.5}],
                "timestamp_ms": 1_100,
            },
        ),
        _row(
            "execution:orders:nobitex",
            {
                "type": "PlaceOrdersAccepted",
                "exchange": "nobitex",
                "frame_id": frame_id,
                "execution_plan_id": execution_plan_id,
                "order_ids": ["ex-1"],
                "timestamp_ms": 1_110,
            },
        ),
    ]
    [chain] = _chains_from_rows(rows).values()

    report = verify_lineage_fidelity(chain)

    assert report.ok


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


def _chains_from_rows(rows: list[dict]):
    envelopes = [decode_envelope(row["envelope_json"]) for row in rows]
    return lineage_chains_from_envelopes(envelopes)
