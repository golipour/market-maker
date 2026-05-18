"""JSON-safe conversion for diagnostics payloads."""

from __future__ import annotations

from typing import Any

from mm.core.events import to_jsonable


def diagnostics_to_jsonable(value: Any) -> Any:
    """Convert diagnostics dataclasses and nested runtime values to JSON-compatible data."""

    return to_jsonable(value)
