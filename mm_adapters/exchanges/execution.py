"""
Live limit placement with ``ExecutionReport`` — canonical floats at the boundary, ticks on the wire.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mm.integrations.trade_stats import place_limit_order_lots

from mm_core.domain.orders import ExecutionReport, OrderIntent
from mm_core.domain.types import InstrumentId, Side
from mm_core.units.converter import UnitConverter


def _side_to_str(side: Side) -> str:
    return str(side.value)


def _order_id_from_response(resp: dict[str, Any]) -> str:
    for k in ("orderId", "order_id", "id", "clientOrderId", "client_order_id"):
        v = resp.get(k)
        if v is not None and str(v).strip():
            return str(v)
    inner = resp.get("order") if isinstance(resp.get("order"), dict) else resp.get("data")
    if isinstance(inner, dict):
        return _order_id_from_response(inner)
    return ""


def _venue_metadata_snippet(resp: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "state",
        "orderId",
        "order_id",
        "id",
        "clientOrderId",
        "client_order_id",
        "msg",
        "message",
        "code",
    )
    out: dict[str, Any] = {k: resp[k] for k in keys if k in resp and resp[k] is not None}
    inner = resp.get("order") if isinstance(resp.get("order"), dict) else resp.get("data")
    if isinstance(inner, dict):
        sub = {k: inner[k] for k in ("id", "status", "price", "amount") if k in inner}
        if sub:
            out["_order"] = sub
    return out


def execution_report_from_place_response(
    resp: dict[str, Any],
    *,
    instrument: InstrumentId,
    filled_qty: float = 0.0,
    avg_price: float | None = None,
) -> ExecutionReport:
    """Map a ``trade_stats`` / adapter place response to a domain report (best-effort)."""
    raw = str(resp.get("status", "") or resp.get("state", "") or "").strip().lower()
    if raw in ("ok", "success", "filled", "new", "open"):
        status = raw if raw else "ok"
    elif not raw and resp.get("id") is not None:
        status = "ok"
    else:
        status = raw or "unknown"
    oid = _order_id_from_response(resp)
    return ExecutionReport(
        instrument=instrument,
        order_id=oid or "unknown",
        status=status,
        filled_qty=filled_qty,
        avg_price=avg_price,
        venue_metadata=_venue_metadata_snippet(resp),
    )


def place_limit_order_lots_with_report(
    *,
    symbol: str,
    exchange: str,
    side: str,
    price_ticks: int,
    qty_lots: int,
    api_token: str | None,
    instrument: InstrumentId,
) -> tuple[dict[str, Any], ExecutionReport]:
    """Place via ``place_limit_order_lots`` and attach an ``ExecutionReport``."""
    resp = place_limit_order_lots(
        symbol=symbol,
        exchange=exchange,
        side=side,
        price_ticks=price_ticks,
        qty_lots=qty_lots,
        api_token=api_token,
    )
    return resp, execution_report_from_place_response(resp, instrument=instrument)


def place_order_intent_ticks(
    intent: OrderIntent,
    *,
    api_token: str | None,
    uc: UnitConverter,
    symbol: str,
    exchange: str,
) -> tuple[dict[str, Any], ExecutionReport]:
    """
    Canonical-float intent → tick/lot → venue.

    Use when callers already work in quote/base and base qty (e.g. rebalancing).
    """
    px = uc.price_ticks_from_quote(intent.price)
    ql = uc.qty_lots_from_base(intent.qty)
    side = _side_to_str(intent.side)
    return place_limit_order_lots_with_report(
        symbol=symbol,
        exchange=exchange,
        side=side,
        price_ticks=px,
        qty_lots=ql,
        api_token=api_token,
        instrument=intent.instrument,
    )


def execution_report_for_json(report: ExecutionReport) -> dict[str, Any]:
    """Flatten for JSON logs (instrument as strings)."""
    d = asdict(report)
    inst = report.instrument
    d["instrument"] = str(inst)
    d["exchange"] = inst.exchange
    d["symbol"] = inst.symbol
    return d
