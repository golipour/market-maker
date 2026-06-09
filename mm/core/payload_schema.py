"""Versioned payload contracts for durable high-value bus events."""

from __future__ import annotations

from typing import Any

TARGET_ORDER_SCHEMA = "mm.target_order"
TARGET_ORDER_VERSION = 1

VENUE_TARGET_BOOK_SCHEMA = "mm.venue_target_book"
VENUE_TARGET_BOOK_VERSION = 1

JOINT_TARGET_BOOK_SCHEMA = "mm.joint_target_book"
JOINT_TARGET_BOOK_VERSION = 1

EXECUTION_LIFECYCLE_SCHEMA = "mm.execution_lifecycle"
EXECUTION_LIFECYCLE_VERSION = 1

EXECUTION_SNAPSHOT_SCHEMA = "mm.execution_snapshot"
EXECUTION_SNAPSHOT_VERSION = 1


def stamped(schema: str, version: int, **fields: Any) -> dict[str, Any]:
    return {"payload_schema": schema, "payload_version": version, **fields}


def assert_payload_schema(payload: dict[str, Any], schema: str, version: int) -> None:
    actual_schema = payload.get("payload_schema")
    actual_version = payload.get("payload_version")
    if actual_schema != schema or actual_version != version:
        raise ValueError(
            f"expected payload {schema} v{version}, got {actual_schema!r} v{actual_version!r}"
        )


def execution_lifecycle_event(event_type: str, **fields: Any) -> dict[str, Any]:
    return stamped(
        EXECUTION_LIFECYCLE_SCHEMA,
        EXECUTION_LIFECYCLE_VERSION,
        type=event_type,
        **fields,
    )


def execution_snapshot_event(**fields: Any) -> dict[str, Any]:
    return stamped(
        EXECUTION_SNAPSHOT_SCHEMA,
        EXECUTION_SNAPSHOT_VERSION,
        type="ExecutionSnapshot",
        **fields,
    )


def execution_lineage_fields(
    *,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, str]:
    fields = {
        "frame_id": str(frame_id or "").strip(),
        "decision_id": str(decision_id or "").strip(),
        "target_book_id": str(target_book_id or "").strip(),
        "execution_plan_id": str(execution_plan_id or "").strip(),
    }
    return {key: value for key, value in fields.items() if value}


def execution_order_summary(order: Any) -> dict[str, Any]:
    if isinstance(order, dict):
        return {
            "exchange": order.get("exchange"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "price": order.get("price"),
            "qty": order.get("qty"),
            "order_type": order.get("order_type"),
            "client_order_id": order.get("client_order_id"),
        }
    return {
        "exchange": order.exchange,
        "symbol": order.symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "order_type": order.order_type,
        "client_order_id": order.client_order_id,
    }


def place_orders_requested_event(
    *,
    exchange: str,
    orders: list[dict[str, Any]],
    policy: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "PlaceOrdersRequested",
        exchange=exchange,
        orders=orders,
        policy=policy,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def place_orders_accepted_event(
    *,
    exchange: str,
    order_ids: list[str],
    orders: list[dict[str, Any]],
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "PlaceOrdersAccepted",
        exchange=exchange,
        order_ids=order_ids,
        orders=orders,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def place_orders_failed_event(
    *,
    exchange: str,
    orders: list[dict[str, Any]],
    error: str,
    message: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "PlaceOrdersFailed",
        exchange=exchange,
        orders=orders,
        error=error,
        message=message,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def cancel_orders_requested_event(
    *,
    exchange: str,
    order_ids: list[str],
    policy: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "CancelOrdersRequested",
        exchange=exchange,
        order_ids=order_ids,
        policy=policy,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def cancel_orders_accepted_event(
    *,
    exchange: str,
    order_ids: list[str],
    timestamp_ms: int,
    actor_id: str = "",
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    payload = execution_lifecycle_event(
        "CancelOrdersAccepted",
        exchange=exchange,
        order_ids=order_ids,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )
    if actor_id:
        payload["actor_id"] = actor_id
    return payload


def cancel_orders_failed_event(
    *,
    exchange: str,
    order_ids: list[str],
    error: str,
    message: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "CancelOrdersFailed",
        exchange=exchange,
        order_ids=order_ids,
        error=error,
        message=message,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def cancel_actor_orders_requested_event(
    *,
    exchange: str,
    actor_id: str,
    reason: str,
    count: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "CancelActorOrdersRequested",
        exchange=exchange,
        actor_id=actor_id,
        reason=reason,
        count=count,
        timestamp_ms=timestamp_ms,
    )


def cancel_all_requested_event(
    *,
    exchange: str,
    reason: str,
    count: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "CancelAllRequested",
        exchange=exchange,
        reason=reason,
        count=count,
        timestamp_ms=timestamp_ms,
    )


def order_validation_rejected_event(
    *,
    exchange: str,
    symbol: str,
    reason: str,
    side: str,
    price: float,
    qty: float,
    stage: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "OrderValidationRejected",
        exchange=exchange,
        symbol=symbol,
        reason=reason,
        side=side,
        price=price,
        qty=qty,
        stage=stage,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def execution_sync_requested_event(
    *,
    exchange: str,
    actor_id: str,
    symbol: str,
    action_count: int,
    target_count: int,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "ExecutionSyncRequested",
        exchange=exchange,
        actor_id=actor_id,
        symbol=symbol,
        action_count=action_count,
        target_count=target_count,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def execution_sync_completed_event(
    *,
    exchange: str,
    actor_id: str,
    symbol: str,
    target_count: int,
    keep_count: int,
    cancel_count: int,
    place_count: int,
    accepted_count: int,
    rejected_count: int,
    placed: bool,
    pending_place_count: int,
    pending_cancel_count: int,
    pending_max_age_ms: int,
    latency_ms: float,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    execution_plan_id: str = "",
) -> dict[str, Any]:
    return execution_lifecycle_event(
        "ExecutionSyncCompleted",
        exchange=exchange,
        actor_id=actor_id,
        symbol=symbol,
        target_count=target_count,
        keep_count=keep_count,
        cancel_count=cancel_count,
        place_count=place_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        placed=placed,
        pending_place_count=pending_place_count,
        pending_cancel_count=pending_cancel_count,
        pending_max_age_ms=pending_max_age_ms,
        latency_ms=max(0.0, float(latency_ms)),
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
            execution_plan_id=execution_plan_id,
        ),
    )


def execution_sync_failed_event(
    *,
    actor_id: str,
    exchange: str,
    symbol: str,
    t_end_ms: int,
    frame_step: int,
    error: str,
    message: str,
    timestamp_ms: int,
    frame_id: str = "",
    decision_id: str = "",
    target_book_id: str = "",
    retryable: bool = True,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    payload = execution_lifecycle_event(
        "ExecutionSyncFailed",
        actor_id=actor_id,
        exchange=exchange,
        symbol=symbol,
        t_end_ms=t_end_ms,
        frame_step=frame_step,
        error=error,
        message=message,
        retryable=retryable,
        timestamp_ms=timestamp_ms,
        **execution_lineage_fields(
            frame_id=frame_id,
            decision_id=decision_id,
            target_book_id=target_book_id,
        ),
    )
    if latency_ms is not None:
        payload["latency_ms"] = max(0.0, float(latency_ms))
    return payload


def joint_target_book_event(book: Any, **metadata: Any) -> dict[str, Any]:
    """Build a versioned JointTargetBook bus payload from a model instance."""

    return {"type": "JointTargetBook", **book.as_payload(), **metadata}
