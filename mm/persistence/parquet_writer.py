from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import EventEnvelope, envelope_event, to_jsonable
from mm.observability import counter_inc, gauge, histogram_observe, log_event
from mm.observability import schema as obs_schema
from mm.observability.event_payloads import observe_envelope
from mm.persistence.compaction import compact_envelope_for_persistence


class ParquetEventWriter:
    """Append runtime events to partitioned Parquet files when pyarrow is present.

    If pyarrow is not installed, the writer falls back to JSONL files in the same
    partition layout so replay data is still captured during development.
    """

    def __init__(
        self,
        *,
        bus: AbstractEventBus,
        topics: tuple[str, ...],
        root_path: str,
        batch_size: int = 100,
        flush_interval_ms: int = 1000,
        run_id: str = "",
        compact_payloads: bool = True,
    ):
        self.bus = bus
        self.topics = topics
        self.root_path = Path(root_path).expanduser()
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_ms = max(50, int(flush_interval_ms))
        self.run_id = run_id
        self.compact_payloads = bool(compact_payloads)
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=self.batch_size * 10)
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._tasks = [asyncio.create_task(self._consume(topic)) for topic in self.topics]
        self._tasks.append(asyncio.create_task(self._write_loop()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _consume(self, topic: str) -> None:
        try:
            async for event in self.bus.subscribe(topic):
                envelope = envelope_event(topic, event, run_id=self.run_id, source="parquet-writer")
                observe_envelope(envelope, stage="parquet_enqueue")
                await self._queue.put(envelope)
                gauge(
                    obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                    float(self._queue.qsize()),
                    labels={"writer": "parquet"},
                    help="Persistence writer queue depth",
                )
        except asyncio.CancelledError:
            raise

    async def _write_loop(self) -> None:
        batch: list[EventEnvelope] = []
        timeout = self.flush_interval_ms / 1000.0
        try:
            while not self._stopped.is_set():
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    pass
                if batch and (len(batch) >= self.batch_size or self._queue.empty()):
                    to_write = batch
                    batch = []
                    queue_depth = self._queue.qsize()
                    gauge(
                        obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                        float(queue_depth),
                        labels={"writer": "parquet"},
                        help="Persistence writer queue depth",
                    )
                    await asyncio.to_thread(self._write_batch, to_write, queue_depth)
        except asyncio.CancelledError:
            if batch:
                await asyncio.to_thread(self._write_batch, batch, self._queue.qsize())
            raise

    def _write_batch(self, batch: list[EventEnvelope], queue_depth: int = 0) -> None:
        if not batch:
            return
        started_ns = time.monotonic_ns()
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for envelope in batch:
            stored = (
                compact_envelope_for_persistence(envelope, sink="parquet")
                if self.compact_payloads
                else envelope
            )
            partition = self._partition_key(stored)
            grouped.setdefault(partition, []).append(_row(stored))
        sink = "parquet"
        written_count = 0
        for partition, rows in grouped.items():
            path = self._part_path(partition)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                schema = _parquet_schema(pa)
                table = pa.Table.from_pylist(rows, schema=schema)
                tmp = path.with_suffix(path.suffix + ".tmp")
                pq.write_table(table, tmp)
                os.replace(tmp, path)
                written_count += len(rows)
            except ImportError:
                sink = "jsonl"
                fallback = path.with_suffix(".jsonl")
                try:
                    with fallback.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_count += len(rows)
                except Exception as exc:
                    counter_inc(
                        "mm_persistence_errors_total",
                        labels={"writer": "parquet", "error": type(exc).__name__},
                    )
                    log_event(
                        obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                        {
                            "writer": "parquet",
                            "batch_size": len(batch),
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "path": str(fallback),
                            "retryable": True,
                        },
                    )
            except Exception as exc:
                counter_inc(
                    "mm_persistence_errors_total",
                    labels={"writer": "parquet", "error": type(exc).__name__},
                )
                log_event(
                    obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                    {
                        "writer": "parquet",
                        "batch_size": len(batch),
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "path": str(path),
                        "retryable": True,
                    },
                )
                continue
        latency_ms = _elapsed_ms(started_ns)
        if written_count <= 0:
            return
        counter_inc("mm_persistence_events_written_total", written_count, labels={"writer": "parquet"})
        gauge("mm_persistence_batch_size", written_count, labels={"writer": "parquet"})
        gauge(
            obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
            float(queue_depth),
            labels={"writer": "parquet"},
            help="Persistence writer queue depth",
        )
        histogram_observe(
            obs_schema.MM_PERSISTENCE_WRITE_LATENCY_MS,
            latency_ms,
            labels={"writer": "parquet"},
            help="Persistence batch write latency",
        )
        log_event(
            obs_schema.EVENT_PERSISTENCE_BATCH_WRITTEN,
            {
                "writer": "parquet",
                "sink": sink,
                "batch_size": written_count,
                "queue_depth": int(queue_depth),
                "latency_ms": latency_ms,
            },
        )

    def _partition_key(self, envelope: EventEnvelope) -> tuple[str, str, str, str, str]:
        dt = datetime.fromtimestamp(envelope.timestamp_ms / 1000.0, tz=timezone.utc)
        event_date = dt.strftime("%Y-%m-%d")
        event_hour = dt.strftime("%H")
        exchange = envelope.exchange or "none"
        symbol = envelope.symbol or "none"
        event_type = envelope.event_type or "event"
        return event_date, event_hour, exchange, symbol, event_type

    def _part_path(self, partition: tuple[str, str, str, str, str]) -> Path:
        event_date, event_hour, exchange, symbol, event_type = partition
        filename = f"part-{time.time_ns()}-{uuid.uuid4().hex}.parquet"
        return (
            self.root_path
            / f"event_date={event_date}"
            / f"exchange={exchange}"
            / f"symbol={symbol}"
            / f"event_type={event_type}"
            / f"hour={event_hour}"
            / filename
        )


def _row(envelope: EventEnvelope) -> dict[str, Any]:
    payload = to_jsonable(envelope.payload)
    if not isinstance(payload, dict):
        payload = {}
    row = {
        "event_id": envelope.event_id,
        "schema": envelope.schema,
        "version": envelope.version,
        "run_id": envelope.run_id,
        "source": envelope.source,
        "topic": envelope.topic,
        "event_type": envelope.event_type,
        "exchange": envelope.exchange,
        "symbol": envelope.symbol,
        "actor_id": envelope.actor_id,
        "timestamp_ms": envelope.timestamp_ms,
        "frame_id": _optional_str(payload.get("frame_id")),
        "decision_id": _optional_str(payload.get("decision_id")),
        "target_book_id": _optional_str(payload.get("target_book_id")),
        "execution_plan_id": _optional_str(payload.get("execution_plan_id")),
        "envelope_json": json.dumps(envelope.as_dict(), ensure_ascii=False),
    }
    row.update(_market_columns(envelope, payload))
    row.update(_controller_columns(envelope, payload))
    row.update(_target_book_columns(envelope, payload))
    row.update(_execution_columns(envelope, payload))
    return row


def _market_columns(envelope: EventEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
    if envelope.event_type != "MarketEvent":
        return {}
    orderbook = payload.get("orderbook")
    bids = _levels(orderbook, "bids")
    asks = _levels(orderbook, "asks")
    best_bid = bids[0][0] if bids else None
    best_bid_qty = bids[0][1] if bids else None
    best_ask = asks[0][0] if asks else None
    best_ask_qty = asks[0][1] if asks else None
    buy_volume = _float_or_none(payload.get("buy_volume"))
    sell_volume = _float_or_none(payload.get("sell_volume"))
    return {
        "interval_ms": _int_or_none(payload.get("interval_ms")),
        "t_start_ms": _int_or_none(payload.get("t_start_ms")),
        "t_end_ms": _int_or_none(payload.get("t_end_ms")),
        "orderbook_timestamp_ms": _dict_int(orderbook, "timestamp_ms"),
        "best_bid": best_bid,
        "best_bid_qty": best_bid_qty,
        "best_ask": best_ask,
        "best_ask_qty": best_ask_qty,
        "mid": (
            (best_bid + best_ask) / 2.0
            if best_bid is not None and best_ask is not None
            else None
        ),
        "spread": (
            best_ask - best_bid
            if best_bid is not None and best_ask is not None
            else None
        ),
        "microprice": _float_or_none(payload.get("microprice")),
        "vwap": _float_or_none(payload.get("vwap")),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "total_trade_volume": (
            (buy_volume or 0.0) + (sell_volume or 0.0)
            if buy_volume is not None or sell_volume is not None
            else None
        ),
        "imbalance": _float_or_none(payload.get("imbalance")),
        "trade_count": _int_or_none(payload.get("trade_count")) or _list_len(payload.get("trades")),
        "bid_depth_levels": len(bids),
        "ask_depth_levels": len(asks),
        "orderbook_is_snapshot": _dict_bool(orderbook, "is_snapshot"),
    }


def _controller_columns(envelope: EventEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
    if envelope.event_type != "ControllerDiagnostics":
        return {}
    state = _dict_or_empty(payload.get("state"))
    models = _dict_or_empty(payload.get("models"))
    optimizer = _dict_or_empty(payload.get("optimizer"))
    timing = _dict_or_empty(optimizer.get("timing"))
    objective = _dict_or_empty(optimizer.get("objective"))
    initial = _dict_or_empty(objective.get("initial"))
    final = _dict_or_empty(objective.get("final"))
    best = _dict_or_empty(objective.get("best"))
    improvement = _dict_or_empty(objective.get("improvement"))
    config = _dict_or_empty(optimizer.get("config"))
    warnings = payload.get("warnings")
    updated_count = 0
    skipped_count = 0
    for score in models.values():
        if not isinstance(score, dict):
            continue
        if bool(score.get("updated")):
            updated_count += 1
        if score.get("skip_reason"):
            skipped_count += 1
    initial_value = _objective_value(initial)
    final_value = _objective_value(final)
    best_value = _objective_value(best)
    return {
        "created_at_ms": _int_or_none(payload.get("created_at_ms")),
        "frame_step": _int_or_none(payload.get("frame_step")),
        "local_inventory": _float_or_none(state.get("local_inventory")),
        "global_inventory": _float_or_none(state.get("global_inventory")),
        "local_cash": _float_or_none(state.get("local_cash")),
        "global_cash": _float_or_none(state.get("global_cash")),
        "local_wealth": _float_or_none(state.get("local_wealth")),
        "global_wealth": _float_or_none(state.get("global_wealth")),
        "model_count": len(models),
        "model_updated_count": updated_count,
        "model_skipped_count": skipped_count,
        "optimizer_kind": _optional_str(config.get("kind") or optimizer.get("kind")),
        "controller_total_ms": _timing_float(timing, "controller_total_ms"),
        "optimize_ms": _timing_float(timing, "optimize_ms", "optimizer_ms", "total_ms"),
        "setup_ms": _timing_float(timing, "setup_ms"),
        "sample_ms": _timing_float(timing, "sample_ms"),
        "forward_ms": _timing_float(timing, "forward_ms"),
        "objective_ms": _timing_float(timing, "objective_ms"),
        "backward_ms": _timing_float(timing, "backward_ms"),
        "adam_ms": _timing_float(timing, "adam_ms"),
        "diagnostics_ms": _timing_float(timing, "diagnostics_ms"),
        "epochs_run": _int_or_none(timing.get("epochs_run")),
        "initial_value": initial_value,
        "final_value": final_value,
        "best_value": best_value,
        "best_minus_initial": _coalesce_float(
            _float_or_none(improvement.get("best_minus_initial")),
            _float_diff(best_value, initial_value),
        ),
        "final_minus_initial": _coalesce_float(
            _float_or_none(improvement.get("final_minus_initial")),
            _float_diff(final_value, initial_value),
        ),
        "best_expected_total_reward": _float_or_none(best.get("expected_total_reward")),
        "best_expected_total_return": _float_or_none(best.get("expected_total_return")),
        "risk_approved": _dict_bool(payload.get("risk"), "approved"),
        "warnings_count": len(warnings) if isinstance(warnings, list) else 0,
    }


def _target_book_columns(envelope: EventEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
    if envelope.event_type != "JointTargetBook":
        return {}
    local = _dict_or_empty(payload.get("local"))
    global_ = _dict_or_empty(payload.get("global"))
    local_orders = local.get("orders")
    global_orders = global_.get("orders")
    return {
        "mode": _optional_str(payload.get("mode")),
        "local_target_order_count": _list_len(local_orders),
        "global_target_order_count": _list_len(global_orders),
        "target_order_count": _list_len(local_orders) + _list_len(global_orders),
        "local_target_notional": _orders_notional(local_orders),
        "global_target_notional": _orders_notional(global_orders),
    }


def _execution_columns(envelope: EventEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
    event_type = envelope.event_type
    if not _is_execution_event(envelope.topic, event_type):
        return {}
    timestamp_ms = _int_or_none(payload.get("timestamp_ms")) or envelope.timestamp_ms
    action = None
    lifecycle = None
    success = None
    state_kind = None
    order = payload
    exchange_order_id = _optional_str(payload.get("exchange_order_id"))
    count_value = _int_or_none(payload.get("count"))
    if event_type.startswith("ExecutionSync"):
        action = "sync"
        lifecycle = _execution_lifecycle(event_type)
        success = _success_for_lifecycle(lifecycle)
        count_value = _int_or_none(payload.get("target_count"))
    elif event_type in {"PlaceOrdersRequested", "PlaceOrdersAccepted", "PlaceOrdersFailed"}:
        action = "place"
        lifecycle = _order_lifecycle(event_type)
        success = _success_for_lifecycle(lifecycle)
        state_kind = "requested" if lifecycle == "requested" else "pending" if lifecycle == "accepted" else None
        orders = payload.get("orders")
        order = orders[0] if isinstance(orders, list) and orders and isinstance(orders[0], dict) else payload
        order_ids = payload.get("order_ids")
        exchange_order_id = _order_id_at(order_ids, 0) or _optional_str(order.get("exchange_order_id"))
        count_value = _list_len(orders) or count_value
    elif event_type in {"CancelOrdersRequested", "CancelOrdersAccepted", "CancelOrdersFailed", "CancelActorOrdersRequested"}:
        action = "cancel"
        lifecycle = _order_lifecycle(event_type)
        success = _success_for_lifecycle(lifecycle)
        state_kind = "requested" if lifecycle == "requested" else "pending" if lifecycle == "accepted" else None
        order_ids = payload.get("order_ids")
        exchange_order_id = _order_id_at(order_ids, 0)
        count_value = _list_len(order_ids) or count_value
    elif event_type == "OrderValidationRejected":
        action = "validation"
        lifecycle = "rejected"
        success = False
    elif event_type == "Fill":
        action = "fill"
        lifecycle = "filled"
        success = True
        state_kind = "confirmed"
    elif event_type == "ExecutionSnapshot":
        action = "snapshot"
        lifecycle = "snapshot"
        success = True
        state_kind = "aggregate"
        count_value = _int_or_none(payload.get("live_order_count"))
    else:
        return {}
    return {
        "execution_timestamp_ms": timestamp_ms,
        "execution_action": action,
        "execution_lifecycle": lifecycle,
        "execution_success": success,
        "execution_state_kind": state_kind,
        "client_order_id": _optional_str(order.get("client_order_id")) if isinstance(order, dict) else None,
        "exchange_order_id": exchange_order_id,
        "order_side": _optional_str(order.get("side")) if isinstance(order, dict) else None,
        "order_price": _float_or_none(order.get("price")) if isinstance(order, dict) else None,
        "order_qty": _float_or_none(order.get("qty")) if isinstance(order, dict) else None,
        "order_status": _optional_str(payload.get("status") or payload.get("reason")),
        "error_type": _optional_str(payload.get("error")),
        "error_message": _optional_str(payload.get("message")),
        "latency_ms": _float_or_none(payload.get("latency_ms")),
        "count_value": count_value,
    }


def _parquet_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("event_id", pa.string()),
            ("schema", pa.string()),
            ("version", pa.int64()),
            ("run_id", pa.string()),
            ("source", pa.string()),
            ("topic", pa.string()),
            ("event_type", pa.string()),
            ("exchange", pa.string()),
            ("symbol", pa.string()),
            ("actor_id", pa.string()),
            ("timestamp_ms", pa.int64()),
            ("frame_id", pa.string()),
            ("decision_id", pa.string()),
            ("target_book_id", pa.string()),
            ("execution_plan_id", pa.string()),
            ("envelope_json", pa.string()),
            ("interval_ms", pa.int64()),
            ("t_start_ms", pa.int64()),
            ("t_end_ms", pa.int64()),
            ("orderbook_timestamp_ms", pa.int64()),
            ("best_bid", pa.float64()),
            ("best_bid_qty", pa.float64()),
            ("best_ask", pa.float64()),
            ("best_ask_qty", pa.float64()),
            ("mid", pa.float64()),
            ("spread", pa.float64()),
            ("microprice", pa.float64()),
            ("vwap", pa.float64()),
            ("buy_volume", pa.float64()),
            ("sell_volume", pa.float64()),
            ("total_trade_volume", pa.float64()),
            ("imbalance", pa.float64()),
            ("trade_count", pa.int64()),
            ("bid_depth_levels", pa.int64()),
            ("ask_depth_levels", pa.int64()),
            ("orderbook_is_snapshot", pa.bool_()),
            ("created_at_ms", pa.int64()),
            ("frame_step", pa.int64()),
            ("local_inventory", pa.float64()),
            ("global_inventory", pa.float64()),
            ("local_cash", pa.float64()),
            ("global_cash", pa.float64()),
            ("local_wealth", pa.float64()),
            ("global_wealth", pa.float64()),
            ("model_count", pa.int64()),
            ("model_updated_count", pa.int64()),
            ("model_skipped_count", pa.int64()),
            ("optimizer_kind", pa.string()),
            ("controller_total_ms", pa.float64()),
            ("optimize_ms", pa.float64()),
            ("setup_ms", pa.float64()),
            ("sample_ms", pa.float64()),
            ("forward_ms", pa.float64()),
            ("objective_ms", pa.float64()),
            ("backward_ms", pa.float64()),
            ("adam_ms", pa.float64()),
            ("diagnostics_ms", pa.float64()),
            ("epochs_run", pa.int64()),
            ("initial_value", pa.float64()),
            ("final_value", pa.float64()),
            ("best_value", pa.float64()),
            ("best_minus_initial", pa.float64()),
            ("final_minus_initial", pa.float64()),
            ("best_expected_total_reward", pa.float64()),
            ("best_expected_total_return", pa.float64()),
            ("risk_approved", pa.bool_()),
            ("warnings_count", pa.int64()),
            ("mode", pa.string()),
            ("local_target_order_count", pa.int64()),
            ("global_target_order_count", pa.int64()),
            ("target_order_count", pa.int64()),
            ("local_target_notional", pa.float64()),
            ("global_target_notional", pa.float64()),
            ("execution_timestamp_ms", pa.int64()),
            ("execution_action", pa.string()),
            ("execution_lifecycle", pa.string()),
            ("execution_success", pa.bool_()),
            ("execution_state_kind", pa.string()),
            ("client_order_id", pa.string()),
            ("exchange_order_id", pa.string()),
            ("order_side", pa.string()),
            ("order_price", pa.float64()),
            ("order_qty", pa.float64()),
            ("order_status", pa.string()),
            ("error_type", pa.string()),
            ("error_message", pa.string()),
            ("latency_ms", pa.float64()),
            ("count_value", pa.int64()),
        ]
    )


def _levels(orderbook: Any, side: str) -> list[tuple[float, float]]:
    if not isinstance(orderbook, dict):
        return []
    raw = orderbook.get(side)
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for level in raw:
        price = qty = None
        if isinstance(level, dict):
            price = _float_or_none(level.get("price"))
            qty = _float_or_none(level.get("qty"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _float_or_none(level[0])
            qty = _float_or_none(level[1])
        if price is not None and qty is not None:
            out.append((price, qty))
    return out


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_int(value: Any, key: str) -> int | None:
    return _int_or_none(value.get(key)) if isinstance(value, dict) else None


def _dict_bool(value: Any, key: str) -> bool | None:
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    if raw is None:
        return None
    return bool(raw)


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _orders_notional(orders: Any) -> float | None:
    if not isinstance(orders, list):
        return None
    total = 0.0
    found = False
    for order in orders:
        if not isinstance(order, dict):
            continue
        price = _float_or_none(order.get("price"))
        qty = _float_or_none(order.get("qty"))
        notional = _float_or_none(order.get("notional"))
        if notional is None and price is not None and qty is not None:
            notional = price * qty
        if notional is not None:
            total += notional
            found = True
    return total if found else None


def _objective_value(snapshot: dict[str, Any]) -> float | None:
    for key in ("value", "objective_value"):
        value = _float_or_none(snapshot.get(key))
        if value is not None:
            return value
    return None


def _timing_float(timing: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(timing.get(key))
        if value is not None:
            return value
    return None


def _float_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _coalesce_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _is_execution_event(topic: str, event_type: str) -> bool:
    return (
        topic.startswith("execution:orders:")
        or topic.startswith("execution:state:")
        or topic.startswith("private:state:") and event_type == "Fill"
        or event_type in {"Fill", "ExecutionSnapshot"}
    )


def _execution_lifecycle(event_type: str) -> str:
    if event_type.endswith("Requested"):
        return "requested"
    if event_type.endswith("Completed"):
        return "completed"
    if event_type.endswith("Failed"):
        return "failed"
    return "event"


def _order_lifecycle(event_type: str) -> str:
    if event_type.endswith("Requested"):
        return "requested"
    if event_type.endswith("Accepted"):
        return "accepted"
    if event_type.endswith("Failed"):
        return "failed"
    return "event"


def _success_for_lifecycle(lifecycle: str) -> bool | None:
    if lifecycle in {"accepted", "completed", "filled"}:
        return True
    if lifecycle in {"failed", "rejected"}:
        return False
    return None


def _order_id_at(order_ids: Any, index: int) -> str | None:
    if not isinstance(order_ids, list) or index >= len(order_ids):
        return None
    return _optional_str(order_ids[index])


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000
