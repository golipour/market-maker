"""Project domain events (e.g. strategy decisions) into diagnostics sinks."""

from __future__ import annotations

from typing import Any

DIAGNOSTICS_FRAME_SCHEMA = "DiagnosticsFrame"
DIAGNOSTICS_FRAME_VERSION = 1


def controller_decision_to_diagnostics_frame(payload: dict[str, Any]) -> dict[str, Any]:
    """Map controller decision payloads into a compact DiagnosticsFrame-like dict."""

    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else []
    return {
        "schema": DIAGNOSTICS_FRAME_SCHEMA,
        "schema_version": DIAGNOSTICS_FRAME_VERSION,
        "runtime": "mm",
        "point": {
            "step_counter": payload.get("frame_step"),
            "t_end_ms": payload.get("t_end_ms"),
            "exchange": payload.get("exchange"),
            "symbol": payload.get("symbol"),
            "mid": payload.get("mid"),
            "inventory": payload.get("inventory"),
            "cash": payload.get("cash"),
            "buy_volume": payload.get("buy_volume"),
            "sell_volume": payload.get("sell_volume"),
            "latency_ms": payload.get("latency_ms"),
            "hedge_status": payload.get("hedge_status"),
            "has_hedge": payload.get("has_hedge"),
            "observe_only": payload.get("observe_only"),
        },
        "controller": {
            "actor_id": payload.get("actor_id"),
            "order_count": payload.get("order_count"),
            "orders": orders,
            "price_model": payload.get("price_model"),
            "market_flow_model": payload.get("market_flow_model"),
            "hedge_price_model": payload.get("hedge_price_model"),
            "optimizer": payload.get("optimizer"),
            "orderbook_grid": payload.get("orderbook_grid"),
            "unit_policy": payload.get("unit_policy"),
        },
        "series": {
            "order_count": payload.get("order_count"),
            "latency_ms": payload.get("latency_ms"),
        },
    }


def emit_controller_diagnostics(payload: dict[str, Any]) -> None:
    """Best-effort projection from controller decisions to diagnostics sinks."""

    try:
        from mm.diagnostics_emitter import emit

        emit(controller_decision_to_diagnostics_frame(payload))
    except Exception:
        return
