from __future__ import annotations

import pytest

from mm.core.payload_schema import (
    EXECUTION_LIFECYCLE_SCHEMA,
    EXECUTION_LIFECYCLE_VERSION,
    assert_payload_schema,
    cancel_orders_requested_event,
    execution_lineage_fields,
    execution_order_summary,
    execution_sync_completed_event,
    place_orders_requested_event,
)


def test_execution_lineage_fields_omit_empty_values() -> None:
    assert execution_lineage_fields(
        frame_id="frame:1",
        decision_id="",
        target_book_id="target:1",
    ) == {
        "frame_id": "frame:1",
        "target_book_id": "target:1",
    }


def test_place_orders_requested_event_is_versioned() -> None:
    payload = place_orders_requested_event(
        exchange="nobitex",
        orders=[{"side": "buy", "price": 100.0, "qty": 0.5}],
        policy="replace",
        timestamp_ms=1_000,
        frame_id="frame:1",
        execution_plan_id="execplan:1",
    )

    assert_payload_schema(payload, EXECUTION_LIFECYCLE_SCHEMA, EXECUTION_LIFECYCLE_VERSION)
    assert payload["type"] == "PlaceOrdersRequested"
    assert payload["execution_plan_id"] == "execplan:1"
    assert payload["orders"][0]["side"] == "buy"


def test_execution_order_summary_from_order_object() -> None:
    class _Order:
        exchange = "nobitex"
        symbol = "BTC_IRT"
        side = "buy"
        price = 100.0
        qty = 0.5
        order_type = "limit"
        client_order_id = "cid-1"

    summary = execution_order_summary(_Order())

    assert summary["exchange"] == "nobitex"
    assert summary["client_order_id"] == "cid-1"


def test_execution_sync_completed_event_requires_core_counts() -> None:
    payload = execution_sync_completed_event(
        exchange="nobitex",
        actor_id="actor",
        symbol="BTC_IRT",
        target_count=2,
        keep_count=1,
        cancel_count=0,
        place_count=1,
        accepted_count=1,
        rejected_count=0,
        placed=True,
        pending_place_count=0,
        pending_cancel_count=0,
        pending_max_age_ms=0,
        latency_ms=12.5,
        timestamp_ms=2_000,
        frame_id="frame:1",
    )

    assert payload["latency_ms"] == 12.5
    assert payload["frame_id"] == "frame:1"


def test_cancel_orders_requested_event_rejects_unknown_fields_via_builder() -> None:
    with pytest.raises(TypeError):
        cancel_orders_requested_event(
            exchange="nobitex",
            order_ids=["x"],
            policy="replace",
            timestamp_ms=1,
            unknown_field=True,  # type: ignore[call-arg]
        )
