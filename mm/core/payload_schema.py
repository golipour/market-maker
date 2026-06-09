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


def joint_target_book_event(book: Any, **metadata: Any) -> dict[str, Any]:
    """Build a versioned JointTargetBook bus payload from a model instance."""

    return {"type": "JointTargetBook", **book.as_payload(), **metadata}
