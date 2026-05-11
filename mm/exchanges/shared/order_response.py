"""Shared helpers for parsing venue REST responses (order ids, nested shapes)."""

from __future__ import annotations

from typing import Any


def order_id_from_response(resp: dict[str, Any]) -> str:
    """Best-effort order id from a venue JSON dict (Nobitex / Wallex shapes)."""

    for key in ("orderId", "order_id", "id", "clientOrderId", "client_order_id"):
        value = resp.get(key)
        if value is not None and str(value).strip():
            return str(value)
    inner = resp.get("order") if isinstance(resp.get("order"), dict) else resp.get("data")
    if isinstance(inner, dict):
        nested = order_id_from_response(inner)
        if nested:
            return nested
    result = resp.get("result")
    if isinstance(result, dict):
        nested = order_id_from_response(result)
        if nested:
            return nested
    return ""
