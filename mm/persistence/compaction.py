from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Literal

from mm.core.events import EventEnvelope, to_jsonable

PersistenceSink = Literal["postgres", "parquet"]


def compact_envelope_for_persistence(
    envelope: EventEnvelope,
    *,
    sink: PersistenceSink,
) -> EventEnvelope:
    """Return a smaller durable envelope without changing live bus payloads."""

    payload = compact_payload_for_persistence(envelope.event_type, envelope.payload, sink=sink)
    if payload is envelope.payload:
        return envelope
    return dataclasses.replace(envelope, payload=payload)


def compact_payload_for_persistence(event_type: str, payload: Any, *, sink: PersistenceSink) -> Any:
    value = to_jsonable(payload)
    if not isinstance(value, dict):
        return value
    if event_type == "MarketEvent":
        return _compact_market_event_payload(value)
    if event_type == "JointStrategyFrame":
        return _compact_joint_frame_payload(value)
    if event_type == "ControllerDiagnostics":
        return _compact_controller_diagnostics_payload(value)
    if sink == "parquet" and event_type == "PrivateStateSnapshot":
        return _compact_private_snapshot_payload(value)
    return value


def _compact_market_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    trades = out.get("trades")
    if isinstance(trades, list) and trades:
        out["trade_count"] = len(trades)
        first_ts = _first_numeric(trades, "timestamp_ms")
        last_ts = _last_numeric(trades, "timestamp_ms")
        if first_ts is not None:
            out["first_trade_timestamp_ms"] = first_ts
        if last_ts is not None:
            out["last_trade_timestamp_ms"] = last_ts
        out["trades"] = []
        out["trades_compacted"] = True
    return out


def _compact_joint_frame_payload(payload: dict[str, Any]) -> dict[str, Any]:
    ledger = payload.get("ledger")
    if not isinstance(ledger, dict):
        return payload
    out = dict(payload)
    out["ledger"] = _ledger_summary(ledger)
    out["ledger_compacted"] = True
    out["ledger_ref"] = {
        "topic": "ledger:snapshot",
        "created_at_ms": ledger.get("created_at_ms"),
        "t_end_ms": payload.get("t_end_ms"),
    }
    return out


def _ledger_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "local": _venue_ledger_summary(ledger.get("local")),
        "global": _venue_ledger_summary(ledger.get("global")),
        "reference_price": ledger.get("reference_price"),
        "consolidated_base_inventory": ledger.get("consolidated_base_inventory"),
        "consolidated_quote_wealth": ledger.get("consolidated_quote_wealth"),
        "created_at_ms": ledger.get("created_at_ms"),
        "freshness": dict(ledger.get("freshness") or {}),
    }


def _venue_ledger_summary(venue: Any) -> dict[str, Any] | None:
    if not isinstance(venue, dict):
        return None
    live_orders = venue.get("live_orders")
    balances = venue.get("balances")
    return {
        "exchange": venue.get("exchange"),
        "symbol": venue.get("symbol"),
        "base_asset": venue.get("base_asset"),
        "quote_asset": venue.get("quote_asset"),
        "locked_base": venue.get("locked_base"),
        "locked_quote": venue.get("locked_quote"),
        "available_base": venue.get("available_base"),
        "available_quote": venue.get("available_quote"),
        "last_private_event_ms": venue.get("last_private_event_ms"),
        "last_snapshot_ms": venue.get("last_snapshot_ms"),
        "balance_count": len(balances) if isinstance(balances, dict) else 0,
        "live_order_count": len(live_orders) if isinstance(live_orders, dict) else 0,
    }


def _compact_private_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    balances = payload.get("balances")
    if not isinstance(balances, dict) or not balances:
        return payload
    nonzero_balances = {
        asset: row
        for asset, row in balances.items()
        if not _balance_payload_is_zero(row)
    }
    if len(nonzero_balances) == len(balances):
        return payload
    out = dict(payload)
    out["balance_count"] = len(balances)
    out["nonzero_balance_count"] = len(nonzero_balances)
    out["balances"] = nonzero_balances
    out["balances_compacted"] = True
    return out


def _compact_controller_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pipeline = payload.get("action_pipeline")
    if not isinstance(pipeline, dict):
        return payload
    compacted_pipeline: dict[str, Any] = {}
    changed = False
    for stage, stage_payload in pipeline.items():
        if isinstance(stage_payload, dict):
            compacted_stage, stage_changed = _compact_action_pipeline_stage(stage_payload)
            compacted_pipeline[str(stage)] = compacted_stage
            changed = changed or stage_changed
        else:
            compacted_pipeline[str(stage)] = stage_payload
    if not changed:
        return payload
    out = dict(payload)
    out["action_pipeline"] = compacted_pipeline
    out["action_pipeline_compacted"] = True
    return out


def _compact_action_pipeline_stage(stage: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(stage)
    changed = False
    orders = out.get("orders")
    if isinstance(orders, list) and orders:
        out.setdefault("buy_count", _count_side(orders, "buy"))
        out.setdefault("sell_count", _count_side(orders, "sell"))
        out["orders_digest"] = _stable_digest(orders)
        out["orders_sample"] = [_sample_order(order) for order in orders[:2] if isinstance(order, dict)]
        out["orders_compacted_count"] = len(orders)
        out["orders_compacted"] = True
        out["orders"] = []
        changed = True
    rejections = out.get("rejections")
    if isinstance(rejections, list) and rejections:
        out["rejections_digest"] = _stable_digest(rejections)
        out["rejections_sample"] = [
            _sample_rejection(rejection)
            for rejection in rejections[:3]
            if isinstance(rejection, dict)
        ]
        out["rejections_compacted_count"] = len(rejections)
        out["rejections_compacted"] = True
        out["rejections"] = []
        changed = True
    return out, changed


def _count_side(rows: list[Any], side: str) -> int:
    return sum(
        1
        for row in rows
        if isinstance(row, dict)
        and str(row.get("side", "")).lower().strip() == side
    )


def _sample_order(order: dict[str, Any]) -> dict[str, Any]:
    keys = ("exchange", "symbol", "side", "price", "qty", "notional", "order_type")
    return {key: order.get(key) for key in keys if key in order}


def _sample_rejection(rejection: dict[str, Any]) -> dict[str, Any]:
    keys = ("exchange", "symbol", "side", "reason", "price", "qty", "notional")
    return {key: rejection.get(key) for key in keys if key in rejection}


def _stable_digest(value: Any) -> str:
    raw = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _balance_payload_is_zero(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    try:
        return (
            abs(float(row.get("total", 0.0) or 0.0)) <= 1e-12
            and abs(float(row.get("free", 0.0) or 0.0)) <= 1e-12
            and abs(float(row.get("locked", 0.0) or 0.0)) <= 1e-12
        )
    except Exception:
        return False


def _first_numeric(rows: list[Any], key: str) -> int | float | None:
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None


def _last_numeric(rows: list[Any], key: str) -> int | float | None:
    for row in reversed(rows):
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None
