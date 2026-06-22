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

STRATEGY_DECISION_SCHEMA = "mm.strategy_decision"
STRATEGY_DECISION_VERSION = 1
STRATEGY_DECISION_TYPE = "StrategyDecision"

CONTROLLER_DIAGNOSTICS_SCHEMA = "mm.controller_diagnostics"
CONTROLLER_DIAGNOSTICS_VERSION = 1
CONTROLLER_DIAGNOSTICS_TYPE = "ControllerDiagnostics"

PRIVATE_STATE_SNAPSHOT_SCHEMA = "mm.private_state_snapshot"
PRIVATE_STATE_SNAPSHOT_VERSION = 1
PRIVATE_STATE_SNAPSHOT_TYPE = "PrivateStateSnapshot"

VENUE_LEDGER_SNAPSHOT_SCHEMA = "mm.venue_ledger_snapshot"
VENUE_LEDGER_SNAPSHOT_VERSION = 1
VENUE_LEDGER_SNAPSHOT_TYPE = "VenueLedgerSnapshot"

JOINT_LEDGER_SNAPSHOT_SCHEMA = "mm.joint_ledger_snapshot"
JOINT_LEDGER_SNAPSHOT_VERSION = 1
JOINT_LEDGER_SNAPSHOT_TYPE = "JointLedgerSnapshot"

MARKET_EVENT_SCHEMA = "mm.market_event"
MARKET_EVENT_VERSION = 1
MARKET_EVENT_TYPE = "MarketEvent"

JOINT_STRATEGY_FRAME_SCHEMA = "mm.joint_strategy_frame"
JOINT_STRATEGY_FRAME_VERSION = 1
JOINT_STRATEGY_FRAME_TYPE = "JointStrategyFrame"


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


def market_event_event(event: Any) -> dict[str, Any]:
    """Build a versioned ``MarketEvent`` bus payload."""

    from mm.core.models.market import MarketEvent

    if isinstance(event, dict):
        if event.get("payload_schema") == MARKET_EVENT_SCHEMA:
            assert_payload_schema(event, MARKET_EVENT_SCHEMA, MARKET_EVENT_VERSION)
            return dict(event)
        parsed = market_event_from_payload(event)
        if parsed is not None:
            event = parsed
    if not isinstance(event, MarketEvent):
        raise TypeError(f"expected MarketEvent, got {type(event)!r}")
    return stamped(
        MARKET_EVENT_SCHEMA,
        MARKET_EVENT_VERSION,
        type=MARKET_EVENT_TYPE,
        exchange=event.exchange,
        symbol=event.symbol,
        interval_ms=int(event.interval_ms),
        t_start_ms=int(event.t_start_ms),
        t_end_ms=int(event.t_end_ms),
        orderbook=_raw_orderbook_payload(event.orderbook),
        trades=[_raw_trade_payload(trade) for trade in event.trades],
        buy_volume=float(event.buy_volume),
        sell_volume=float(event.sell_volume),
        vwap=float(event.vwap),
        imbalance=float(event.imbalance),
        microprice=float(event.microprice),
        book_buy_vwap=float(event.book_buy_vwap),
        book_sell_vwap=float(event.book_sell_vwap),
        book_vwap_depth=float(event.book_vwap_depth),
    )


def market_event_from_payload(payload: Any) -> Any | None:
    """Parse a versioned ``MarketEvent`` payload back into a domain struct."""

    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade

    if not isinstance(payload, dict):
        return None
    if payload.get("payload_schema") != MARKET_EVENT_SCHEMA:
        return None
    try:
        assert_payload_schema(payload, MARKET_EVENT_SCHEMA, MARKET_EVENT_VERSION)
        orderbook = _raw_orderbook_from_payload(payload["orderbook"])
        trades = tuple(
            trade
            for trade in (_raw_trade_from_payload(row) for row in payload.get("trades") or ())
            if trade is not None
        )
        return MarketEvent(
            exchange=str(payload["exchange"]),
            symbol=str(payload["symbol"]),
            interval_ms=int(payload["interval_ms"]),
            t_start_ms=int(payload["t_start_ms"]),
            t_end_ms=int(payload["t_end_ms"]),
            orderbook=orderbook,
            trades=trades,
            buy_volume=float(payload.get("buy_volume") or 0.0),
            sell_volume=float(payload.get("sell_volume") or 0.0),
            vwap=float(payload.get("vwap") or 0.0),
            imbalance=float(payload.get("imbalance") or 0.0),
            microprice=float(payload.get("microprice") or 0.0),
            book_buy_vwap=float(payload.get("book_buy_vwap") or 0.0),
            book_sell_vwap=float(payload.get("book_sell_vwap") or 0.0),
            book_vwap_depth=float(payload.get("book_vwap_depth") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _raw_orderbook_payload(orderbook: Any) -> dict[str, Any]:
    return {
        "exchange": orderbook.exchange,
        "symbol": orderbook.symbol,
        "timestamp_ms": int(orderbook.timestamp_ms),
        "bids": [_price_level_payload(level) for level in orderbook.bids],
        "asks": [_price_level_payload(level) for level in orderbook.asks],
        "is_snapshot": bool(orderbook.is_snapshot),
    }


def _price_level_payload(level: Any) -> dict[str, Any]:
    return {"price": float(level.price), "qty": float(level.qty)}


def _raw_trade_payload(trade: Any) -> dict[str, Any]:
    return {
        "trade_id": trade.trade_id,
        "timestamp_ms": int(trade.timestamp_ms),
        "price": float(trade.price),
        "qty": float(trade.qty),
        "side": str(trade.side),
        "exchange": str(trade.exchange),
        "symbol": str(trade.symbol),
    }


def _raw_orderbook_from_payload(payload: Any) -> Any:
    from mm.core.models.market import PriceLevel, RawOrderbook

    if isinstance(payload, RawOrderbook):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected orderbook payload dict, got {type(payload)!r}")
    return RawOrderbook(
        exchange=str(payload["exchange"]),
        symbol=str(payload["symbol"]),
        timestamp_ms=int(payload["timestamp_ms"]),
        bids=_price_levels_from_payload(payload.get("bids")),
        asks=_price_levels_from_payload(payload.get("asks")),
        is_snapshot=bool(payload.get("is_snapshot", True)),
    )


def _price_levels_from_payload(levels: Any) -> tuple[Any, ...]:
    from mm.core.models.market import PriceLevel

    if not levels:
        return ()
    out: list[PriceLevel] = []
    for row in levels:
        if isinstance(row, PriceLevel):
            out.append(row)
        elif isinstance(row, dict):
            out.append(PriceLevel(price=float(row["price"]), qty=float(row["qty"])))
    return tuple(out)


def _raw_trade_from_payload(row: Any) -> Any | None:
    from mm.core.models.market import RawTrade

    if isinstance(row, RawTrade):
        return row
    if not isinstance(row, dict):
        return None
    try:
        return RawTrade(
            trade_id=row.get("trade_id"),
            timestamp_ms=int(row["timestamp_ms"]),
            price=float(row["price"]),
            qty=float(row["qty"]),
            side=str(row["side"]),
            exchange=str(row["exchange"]),
            symbol=str(row["symbol"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
    cancel_latency_ms: float = 0.0,
    place_latency_ms: float = 0.0,
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
        cancel_latency_ms=max(0.0, float(cancel_latency_ms)),
        place_latency_ms=max(0.0, float(place_latency_ms)),
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


def strategy_decision_event(
    decision: Any,
    *,
    executed_orders: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a versioned ``StrategyDecision`` bus payload."""

    executed = list(executed_orders if executed_orders is not None else ())
    fields: dict[str, Any] = {
        "type": STRATEGY_DECISION_TYPE,
        "actor_id": decision.actor_id,
        "exchange": decision.exchange,
        "symbol": decision.symbol,
        "t_end_ms": decision.t_end_ms,
        "frame_id": decision.frame_id,
        "decision_id": decision.decision_id,
        "target_book_id": decision.target_book_id,
        "dry_run": decision.dry_run,
        "frame_step": decision.frame_step,
        "execution_after_steps": decision.execution_after_steps,
        "live_execution_skipped_warmup": decision.live_execution_skipped_warmup,
        "global_status": decision.global_status,
        "has_global": decision.has_global,
        "mid": decision.mid,
        "local_buy_vwap": decision.local_buy_vwap,
        "local_sell_vwap": decision.local_sell_vwap,
        "global_buy_vwap": decision.global_buy_vwap,
        "global_sell_vwap": decision.global_sell_vwap,
        "inventory": decision.inventory,
        "cash": decision.cash,
        "buy_volume": decision.buy_volume,
        "sell_volume": decision.sell_volume,
        "order_count": decision.order_count,
        "orders": list(decision.orders),
        "joint_target_book": (
            decision.joint_target_book.as_payload()
            if decision.joint_target_book is not None
            else None
        ),
        "ladder_orders": list(decision.ladder_orders),
        "executed_orders": executed,
        "local_price_model": decision.model_updates.local_price,
        "local_market_flow_model": decision.model_updates.local_market_flow,
        "global_market_flow_model": decision.model_updates.global_market_flow,
        "global_price_model": decision.model_updates.global_price,
        "optimizer": decision.diagnostics.optimizer,
        "continuous_action_depth": decision.diagnostics.continuous_action_depth,
        "orderbook_grid": decision.diagnostics.orderbook_grid,
        "latency_ms": decision.latency_ms,
        "unit_policy": decision.diagnostics.unit_policy,
        "decision_hash": decision.decision_hash,
    }
    if decision.economics:
        fields["economics"] = dict(decision.economics)
    return stamped(STRATEGY_DECISION_SCHEMA, STRATEGY_DECISION_VERSION, **fields)


def controller_diagnostics_event(diagnostics: Any) -> dict[str, Any]:
    """Build a versioned ``ControllerDiagnostics`` bus payload."""

    return stamped(
        CONTROLLER_DIAGNOSTICS_SCHEMA,
        CONTROLLER_DIAGNOSTICS_VERSION,
        type=CONTROLLER_DIAGNOSTICS_TYPE,
        actor_id=diagnostics.actor_id,
        exchange=diagnostics.exchange,
        symbol=diagnostics.symbol,
        t_end_ms=diagnostics.t_end_ms,
        frame_step=diagnostics.frame_step,
        created_at_ms=diagnostics.created_at_ms,
        run_id=diagnostics.run_id,
        frame_id=diagnostics.frame_id,
        decision_id=diagnostics.decision_id,
        target_book_id=diagnostics.target_book_id,
        frame=_controller_frame_payload(diagnostics.frame),
        state=_controller_state_payload(diagnostics.state),
        models={
            name: _controller_model_score_payload(score)
            for name, score in diagnostics.models.items()
        },
        optimizer=_controller_optimizer_payload(diagnostics.optimizer),
        action_pipeline=_controller_action_pipeline_payload(diagnostics.action_pipeline),
        risk=_controller_risk_payload(diagnostics.risk),
        execution_gates={
            "executed": diagnostics.execution_gates.executed,
            "blocked_reasons": list(diagnostics.execution_gates.blocked_reasons),
            "venue_gates": dict(diagnostics.execution_gates.venue_gates),
        },
        warnings=list(diagnostics.warnings),
        cleared_orderbook=dict(diagnostics.cleared_orderbook),
        global_cleared_orderbook=dict(diagnostics.global_cleared_orderbook),
        ledger_source=diagnostics.ledger_source,
        physical_state=(
            _controller_physical_state_payload(diagnostics.physical_state)
            if diagnostics.physical_state is not None
            else None
        ),
    )


def _controller_frame_payload(frame: Any) -> dict[str, Any]:
    return {
        "global_status": frame.global_status,
        "global_lag_ms": frame.global_lag_ms,
        "frame_age_ms": frame.frame_age_ms,
        "frame_created_at_ms": frame.frame_created_at_ms,
        "local_orderbook_timestamp_ms": frame.local_orderbook_timestamp_ms,
        "local_private_age_ms": frame.local_private_age_ms,
        "global_private_age_ms": frame.global_private_age_ms,
        "sync_status": dict(frame.sync_status),
    }


def _controller_state_payload(state: Any) -> dict[str, Any]:
    return {
        "mid": state.mid,
        "local_inventory": state.local_inventory,
        "local_cash": state.local_cash,
        "global_inventory": state.global_inventory,
        "global_cash": state.global_cash,
        "local_buy_vwap": state.local_buy_vwap,
        "local_sell_vwap": state.local_sell_vwap,
        "global_buy_vwap": state.global_buy_vwap,
        "global_sell_vwap": state.global_sell_vwap,
        "local_base_wallet": dict(state.local_base_wallet),
        "local_quote_wallet": dict(state.local_quote_wallet),
        "global_base_wallet": dict(state.global_base_wallet),
        "global_quote_wallet": dict(state.global_quote_wallet),
        "local_wealth": state.local_wealth,
        "global_wealth": state.global_wealth,
        "ledger_source": state.ledger_source,
    }


def _controller_physical_state_payload(state: Any) -> dict[str, Any]:
    return {
        "local_inventory": state.local_inventory,
        "local_cash": state.local_cash,
        "global_inventory": state.global_inventory,
        "global_cash": state.global_cash,
        "consolidated_base_inventory": state.consolidated_base_inventory,
        "consolidated_quote_wealth": state.consolidated_quote_wealth,
    }


def _controller_model_score_payload(score: Any) -> dict[str, Any]:
    return {
        "model": score.model,
        "kind": score.kind,
        "updated": score.updated,
        "skip_reason": score.skip_reason,
        "observation": dict(score.observation),
        "params_before": dict(score.params_before),
        "params_after": dict(score.params_after),
        "log_likelihood": score.log_likelihood,
        "nll": score.nll,
        "standardized_residual": score.standardized_residual,
    }


def _controller_optimizer_payload(optimizer: Any) -> dict[str, Any]:
    return {
        "config": dict(optimizer.config),
        "timing": dict(optimizer.timing),
        "objective": dict(optimizer.objective),
        "action": dict(optimizer.action),
        "gradients": dict(optimizer.gradients),
        "rollout": dict(optimizer.rollout),
    }


def _controller_action_pipeline_payload(pipeline: Any) -> dict[str, Any]:
    return {
        "continuous": dict(pipeline.continuous),
        "ladder": dict(pipeline.ladder),
        "capped": dict(pipeline.capped),
        "target_book": dict(pipeline.target_book),
        "risk_approved_target": dict(pipeline.risk_approved_target),
        "planned_cancel": dict(pipeline.planned_cancel),
        "planned_place": dict(pipeline.planned_place),
        "rules_valid": dict(pipeline.rules_valid),
        "router_accepted": dict(pipeline.router_accepted),
        "kept": dict(pipeline.kept),
    }


def _controller_risk_payload(risk: Any) -> dict[str, Any]:
    return {
        "approved": risk.approved,
        "rejection_counts": dict(risk.rejection_counts),
        "exposure_before": dict(risk.exposure_before),
        "exposure_after": dict(risk.exposure_after),
        "limits": dict(risk.limits),
    }


def private_state_snapshot_event(snapshot: Any) -> dict[str, Any]:
    """Build a versioned private exchange snapshot bus payload."""

    return stamped(
        PRIVATE_STATE_SNAPSHOT_SCHEMA,
        PRIVATE_STATE_SNAPSHOT_VERSION,
        type=PRIVATE_STATE_SNAPSHOT_TYPE,
        exchange=snapshot.exchange,
        timestamp_ms=int(snapshot.timestamp_ms),
        balances={
            asset: _private_balance_payload(balance)
            for asset, balance in snapshot.balances.items()
        },
        open_orders={
            order_id: _private_open_order_payload(open_order)
            for order_id, open_order in snapshot.open_orders.items()
        },
    )


def private_state_snapshot_from_payload(payload: Any) -> Any:
    """Parse a versioned private snapshot payload back into ``PrivateStateSnapshot``."""

    from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
    from mm.core.models.execution import Order

    if isinstance(payload, PrivateStateSnapshot):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected private snapshot payload dict, got {type(payload)!r}")
    assert_payload_schema(payload, PRIVATE_STATE_SNAPSHOT_SCHEMA, PRIVATE_STATE_SNAPSHOT_VERSION)
    exchange = str(payload["exchange"])
    timestamp_ms = int(payload["timestamp_ms"])
    balances: dict[str, Balance] = {}
    for asset, row in dict(payload.get("balances") or {}).items():
        balances[str(asset)] = _private_balance_from_payload(row, exchange=exchange, timestamp_ms=timestamp_ms)
    open_orders: dict[str, OpenOrder] = {}
    for order_id, row in dict(payload.get("open_orders") or {}).items():
        open_orders[str(order_id)] = _private_open_order_from_payload(
            row,
            exchange=exchange,
            fallback_order_id=str(order_id),
            timestamp_ms=timestamp_ms,
        )
    return PrivateStateSnapshot(
        exchange=exchange,
        timestamp_ms=timestamp_ms,
        balances=balances,
        open_orders=open_orders,
    )


def venue_ledger_snapshot_event(ledger: Any) -> dict[str, Any]:
    """Build a versioned venue ledger snapshot bus payload."""

    return stamped(
        VENUE_LEDGER_SNAPSHOT_SCHEMA,
        VENUE_LEDGER_SNAPSHOT_VERSION,
        type=VENUE_LEDGER_SNAPSHOT_TYPE,
        exchange=ledger.exchange,
        symbol=ledger.symbol,
        base_asset=ledger.base_asset,
        quote_asset=ledger.quote_asset,
        balances={
            asset: _private_balance_payload(balance)
            for asset, balance in ledger.balances.items()
        },
        live_orders={
            order_id: _private_open_order_payload(open_order)
            for order_id, open_order in ledger.live_orders.items()
        },
        locked_base=float(ledger.locked_base),
        locked_quote=float(ledger.locked_quote),
        available_base=float(ledger.available_base),
        available_quote=float(ledger.available_quote),
        last_private_event_ms=int(ledger.last_private_event_ms),
        last_snapshot_ms=int(ledger.last_snapshot_ms),
    )


def joint_ledger_snapshot_event(joint: Any) -> dict[str, Any]:
    """Build a versioned joint ledger snapshot bus payload."""

    payload = stamped(
        JOINT_LEDGER_SNAPSHOT_SCHEMA,
        JOINT_LEDGER_SNAPSHOT_VERSION,
        type=JOINT_LEDGER_SNAPSHOT_TYPE,
        local=venue_ledger_snapshot_event(joint.local),
        reference_price=float(joint.reference_price),
        consolidated_base_inventory=float(joint.consolidated_base_inventory),
        consolidated_quote_wealth=float(joint.consolidated_quote_wealth),
        created_at_ms=int(joint.created_at_ms),
        freshness=_ledger_freshness_payload(joint.freshness),
    )
    payload["global"] = (
        venue_ledger_snapshot_event(joint.global_) if joint.global_ is not None else None
    )
    return payload


def joint_strategy_frame_event(frame: Any) -> dict[str, Any]:
    """Build a versioned joint strategy frame bus payload with full replay data."""

    from mm.core.models.joint_strategy import JointStrategyFrame

    if isinstance(frame, dict):
        if frame.get("payload_schema") == JOINT_STRATEGY_FRAME_SCHEMA:
            assert_payload_schema(frame, JOINT_STRATEGY_FRAME_SCHEMA, JOINT_STRATEGY_FRAME_VERSION)
            return dict(frame)
        parsed = joint_strategy_frame_from_payload(frame)
        if parsed is not None:
            frame = parsed
    if not isinstance(frame, JointStrategyFrame):
        raise TypeError(f"expected JointStrategyFrame, got {type(frame)!r}")
    if frame.input_frame is None:
        raise ValueError("JointStrategyFrame.input_frame is required for durable bus payloads")
    if frame.ledger is None:
        raise ValueError("JointStrategyFrame.ledger is required for durable bus payloads")
    return stamped(
        JOINT_STRATEGY_FRAME_SCHEMA,
        JOINT_STRATEGY_FRAME_VERSION,
        type=JOINT_STRATEGY_FRAME_TYPE,
        local_market=market_event_event(frame.local_market),
        global_market=(
            market_event_event(frame.global_market) if frame.global_market is not None else None
        ),
        ledger=joint_ledger_snapshot_event(frame.ledger),
        physical_ledger=(
            joint_ledger_snapshot_event(frame.physical_ledger)
            if frame.physical_ledger is not None
            else None
        ),
        ledger_source=str(frame.ledger_source or "physical"),
        mode=str(frame.mode),
        sync_status=frame.sync_status.as_payload(),
        created_at_ms=int(frame.created_at_ms),
        input_frame=_strategy_input_frame_event(frame.input_frame),
        frame_id=str(frame.frame_id or ""),
    )


def joint_strategy_frame_from_payload(payload: Any) -> Any | None:
    """Parse a versioned joint strategy frame payload back into a domain struct."""

    from mm.core.models.joint_strategy import FrameSyncStatus, JointStrategyFrame

    if isinstance(payload, JointStrategyFrame):
        return payload
    if not isinstance(payload, dict):
        return None
    if payload.get("payload_schema") != JOINT_STRATEGY_FRAME_SCHEMA:
        return None
    try:
        assert_payload_schema(payload, JOINT_STRATEGY_FRAME_SCHEMA, JOINT_STRATEGY_FRAME_VERSION)
        input_payload = payload.get("input_frame")
        if not isinstance(input_payload, dict):
            return None
        local_market = market_event_from_payload(payload["local_market"])
        if local_market is None:
            return None
        global_payload = payload.get("global_market")
        global_market = (
            market_event_from_payload(global_payload)
            if isinstance(global_payload, dict)
            else None
        )
        ledger = joint_ledger_snapshot_from_payload(payload["ledger"])
        physical_payload = payload.get("physical_ledger")
        physical_ledger = (
            joint_ledger_snapshot_from_payload(physical_payload)
            if isinstance(physical_payload, dict)
            else None
        )
        input_frame = _strategy_input_frame_from_payload(input_payload)
        sync_payload = payload.get("sync_status")
        sync_status = (
            _frame_sync_status_from_payload(sync_payload)
            if isinstance(sync_payload, dict)
            else FrameSyncStatus()
        )
        return JointStrategyFrame(
            local_market=local_market,
            global_market=global_market,
            ledger=ledger,
            physical_ledger=physical_ledger,
            ledger_source=str(payload.get("ledger_source") or "physical"),
            mode=payload.get("mode") or "single_venue",
            sync_status=sync_status,
            created_at_ms=int(payload.get("created_at_ms") or 0),
            input_frame=input_frame,
            frame_id=str(payload.get("frame_id") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _strategy_input_frame_event(frame: Any) -> dict[str, Any]:
    from mm.core.models.strategy import StrategyInputFrame

    if not isinstance(frame, StrategyInputFrame):
        raise TypeError(f"expected StrategyInputFrame, got {type(frame)!r}")
    return {
        "local": market_event_event(frame.local),
        "global_": market_event_event(frame.global_) if frame.global_ is not None else None,
        "local_private": (
            private_state_snapshot_event(frame.local_private)
            if frame.local_private is not None
            else None
        ),
        "global_private": (
            private_state_snapshot_event(frame.global_private)
            if frame.global_private is not None
            else None
        ),
        "global_status": frame.global_status,
        "global_lag_ms": frame.global_lag_ms,
        "max_global_lag_ms": int(frame.max_global_lag_ms),
        "created_at_ms": int(frame.created_at_ms),
        "frame_id": str(frame.frame_id or ""),
    }


def _strategy_input_frame_from_payload(payload: dict[str, Any]) -> Any:
    from mm.core.models.strategy import StrategyInputFrame

    local = market_event_from_payload(payload["local"])
    if local is None:
        raise ValueError("StrategyInputFrame.local is required")
    global_payload = payload.get("global_")
    global_market = (
        market_event_from_payload(global_payload)
        if isinstance(global_payload, dict)
        else None
    )
    local_private_payload = payload.get("local_private")
    local_private = (
        private_state_snapshot_from_payload(local_private_payload)
        if isinstance(local_private_payload, dict)
        else None
    )
    global_private_payload = payload.get("global_private")
    global_private = (
        private_state_snapshot_from_payload(global_private_payload)
        if isinstance(global_private_payload, dict)
        else None
    )
    return StrategyInputFrame(
        local=local,
        global_=global_market,
        local_private=local_private,
        global_private=global_private,
        global_status=payload.get("global_status") or "disabled",
        global_lag_ms=payload.get("global_lag_ms"),
        max_global_lag_ms=int(payload.get("max_global_lag_ms") or 0),
        created_at_ms=int(payload.get("created_at_ms") or 0),
        frame_id=str(payload.get("frame_id") or ""),
    )


def _frame_sync_status_from_payload(payload: dict[str, Any]) -> Any:
    from mm.core.models.joint_strategy import FrameSyncStatus

    return FrameSyncStatus(
        local_market_ready=bool(payload.get("local_market_ready", False)),
        global_market_ready=bool(payload.get("global_market_ready", False)),
        local_private_ready=bool(payload.get("local_private_ready", False)),
        global_private_ready=bool(payload.get("global_private_ready", False)),
        ledger_fresh=bool(payload.get("ledger_fresh", True)),
        latency_skew_ms=payload.get("latency_skew_ms"),
        global_status=payload.get("global_status") or "disabled",
        ledger_ahead_of_market_ms=payload.get("ledger_ahead_of_market_ms"),
    )


def joint_ledger_snapshot_from_payload(payload: Any) -> Any:
    """Parse a versioned joint ledger payload back into ``JointLedgerSnapshot``."""

    from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness

    if isinstance(payload, JointLedgerSnapshot):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected joint ledger payload dict, got {type(payload)!r}")
    assert_payload_schema(payload, JOINT_LEDGER_SNAPSHOT_SCHEMA, JOINT_LEDGER_SNAPSHOT_VERSION)
    global_payload = payload.get("global")
    return JointLedgerSnapshot(
        local=venue_ledger_snapshot_from_payload(payload["local"]),
        global_=venue_ledger_snapshot_from_payload(global_payload) if isinstance(global_payload, dict) else None,
        reference_price=float(payload.get("reference_price") or 0.0),
        consolidated_base_inventory=float(payload.get("consolidated_base_inventory") or 0.0),
        consolidated_quote_wealth=float(payload.get("consolidated_quote_wealth") or 0.0),
        created_at_ms=int(payload.get("created_at_ms") or 0),
        freshness=_ledger_freshness_from_payload(payload.get("freshness")),
    )


def venue_ledger_snapshot_from_payload(payload: Any) -> Any:
    """Parse a versioned venue ledger payload back into ``VenueLedger``."""

    from mm.core.models.ledger import VenueLedger

    if isinstance(payload, VenueLedger):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected venue ledger payload dict, got {type(payload)!r}")
    assert_payload_schema(payload, VENUE_LEDGER_SNAPSHOT_SCHEMA, VENUE_LEDGER_SNAPSHOT_VERSION)
    exchange = str(payload["exchange"])
    timestamp_ms = int(payload.get("last_snapshot_ms") or payload.get("last_private_event_ms") or 0)
    balances = {
        str(asset): _private_balance_from_payload(row, exchange=exchange, timestamp_ms=timestamp_ms)
        for asset, row in dict(payload.get("balances") or {}).items()
    }
    live_orders = {
        str(order_id): _private_open_order_from_payload(
            row,
            exchange=exchange,
            fallback_order_id=str(order_id),
            timestamp_ms=timestamp_ms,
        )
        for order_id, row in dict(payload.get("live_orders") or {}).items()
    }
    return VenueLedger(
        exchange=exchange,
        symbol=str(payload["symbol"]),
        base_asset=str(payload["base_asset"]),
        quote_asset=str(payload["quote_asset"]),
        balances=balances,
        live_orders=live_orders,
        locked_base=float(payload.get("locked_base") or 0.0),
        locked_quote=float(payload.get("locked_quote") or 0.0),
        available_base=float(payload.get("available_base") or 0.0),
        available_quote=float(payload.get("available_quote") or 0.0),
        last_private_event_ms=int(payload.get("last_private_event_ms") or 0),
        last_snapshot_ms=int(payload.get("last_snapshot_ms") or 0),
    )


def joint_target_book_from_payload(payload: Any) -> Any:
    """Parse a versioned joint target-book payload back into ``JointTargetBook``."""

    from mm.core.models.target_book import JointTargetBook, LiquiditySourcingMode

    if isinstance(payload, JointTargetBook):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected joint target-book payload dict, got {type(payload)!r}")
    assert_payload_schema(payload, JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION)
    global_payload = payload.get("global")
    mode = str(payload.get("mode") or "single_venue")
    if mode not in {"single_venue", "liquidity_sourcing"}:
        mode = "single_venue"
    return JointTargetBook(
        actor_id=str(payload.get("actor_id") or ""),
        local=venue_target_book_from_payload(payload["local"]),
        global_=venue_target_book_from_payload(global_payload) if isinstance(global_payload, dict) else None,
        mode=mode,  # type: ignore[arg-type]
        diagnostics=dict(payload.get("diagnostics") or {}),
        frame_id=str(payload.get("frame_id") or ""),
        decision_id=str(payload.get("decision_id") or ""),
        target_book_id=str(payload.get("target_book_id") or ""),
    )


def venue_target_book_from_payload(payload: Any) -> Any:
    """Parse a versioned venue target-book payload back into ``VenueTargetBook``."""

    from mm.core.models.target_book import VenueTargetBook

    if isinstance(payload, VenueTargetBook):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected venue target-book payload dict, got {type(payload)!r}")
    assert_payload_schema(payload, VENUE_TARGET_BOOK_SCHEMA, VENUE_TARGET_BOOK_VERSION)
    policy = str(payload.get("policy") or "reconcile")
    if policy not in {"reconcile", "replace"}:
        policy = "reconcile"
    return VenueTargetBook(
        exchange=str(payload["exchange"]),
        symbol=str(payload["symbol"]),
        actor_id=str(payload.get("actor_id") or ""),
        orders=tuple(target_order_from_payload(row) for row in payload.get("orders") or ()),
        policy=policy,  # type: ignore[arg-type]
        frame_id=str(payload.get("frame_id") or ""),
        decision_id=str(payload.get("decision_id") or ""),
        target_book_id=str(payload.get("target_book_id") or ""),
        execution_plan_id=str(payload.get("execution_plan_id") or ""),
    )


def target_order_from_payload(row: Any) -> Any:
    """Parse a versioned target-order payload back into ``TargetOrder``."""

    from mm.core.models.target_book import TargetOrder

    if isinstance(row, TargetOrder):
        return row
    if not isinstance(row, dict):
        raise TypeError(f"expected target-order payload dict, got {type(row)!r}")
    if row.get("payload_schema") == TARGET_ORDER_SCHEMA:
        assert_payload_schema(row, TARGET_ORDER_SCHEMA, TARGET_ORDER_VERSION)
    return TargetOrder(
        exchange=str(row["exchange"]),
        symbol=str(row["symbol"]),
        side=str(row["side"]),
        price=float(row["price"]),
        qty=float(row["qty"]),
        order_type=str(row.get("order_type") or "limit"),
        tags=tuple(str(tag) for tag in row.get("tags") or ()),
    )


def _private_balance_payload(balance: Any) -> dict[str, Any]:
    return {
        "exchange": balance.exchange,
        "asset": balance.asset,
        "free": float(balance.free),
        "locked": float(balance.locked),
        "total": float(balance.total),
        "timestamp_ms": int(balance.timestamp_ms),
    }


def _private_open_order_payload(open_order: Any) -> dict[str, Any]:
    order = open_order.order
    return {
        "exchange_order_id": open_order.exchange_order_id,
        "actor_id": open_order.actor_id,
        "remaining_qty": float(open_order.remaining_qty),
        "filled_qty": float(open_order.filled_qty),
        "status": open_order.status,
        "timestamp_ms": int(open_order.timestamp_ms),
        "order": {
            "exchange": order.exchange,
            "symbol": order.symbol,
            "side": order.side,
            "price": float(order.price),
            "qty": float(order.qty),
            "order_type": order.order_type,
            "client_order_id": order.client_order_id,
        },
    }


def _private_balance_from_payload(
    row: Any,
    *,
    exchange: str,
    timestamp_ms: int,
) -> Any:
    from mm.core.models.private import Balance

    if isinstance(row, Balance):
        return row
    if not isinstance(row, dict):
        raise TypeError(f"expected balance payload dict, got {type(row)!r}")
    asset = str(row.get("asset") or "")
    total = row.get("total")
    return Balance(
        exchange=str(row.get("exchange") or exchange),
        asset=asset,
        free=float(row.get("free") or 0.0),
        locked=float(row.get("locked") or 0.0),
        total=float(total if total is not None else float(row.get("free") or 0.0) + float(row.get("locked") or 0.0)),
        timestamp_ms=int(row.get("timestamp_ms") or timestamp_ms),
    )


def _private_open_order_from_payload(
    row: Any,
    *,
    exchange: str,
    fallback_order_id: str,
    timestamp_ms: int,
) -> Any:
    from mm.core.models.execution import Order
    from mm.core.models.private import OpenOrder

    if isinstance(row, OpenOrder):
        return row
    if not isinstance(row, dict):
        raise TypeError(f"expected open-order payload dict, got {type(row)!r}")
    order_row = row.get("order")
    if not isinstance(order_row, dict):
        raise ValueError("open-order payload requires nested order dict")
    order = Order(
        exchange=str(order_row.get("exchange") or exchange),
        symbol=str(order_row["symbol"]),
        side=str(order_row["side"]),
        price=float(order_row["price"]),
        qty=float(order_row["qty"]),
        order_type=str(order_row.get("order_type") or "limit"),
        client_order_id=order_row.get("client_order_id"),
    )
    return OpenOrder(
        exchange_order_id=str(row.get("exchange_order_id") or fallback_order_id),
        actor_id=row.get("actor_id"),
        order=order,
        remaining_qty=float(row.get("remaining_qty") or 0.0),
        filled_qty=float(row.get("filled_qty") or 0.0),
        status=str(row.get("status") or "open"),
        timestamp_ms=int(row.get("timestamp_ms") or timestamp_ms),
    )


def _ledger_freshness_payload(freshness: Any) -> dict[str, Any]:
    return {
        "local_private_age_ms": freshness.local_private_age_ms,
        "global_private_age_ms": freshness.global_private_age_ms,
        "local_market_age_ms": freshness.local_market_age_ms,
        "global_market_age_ms": freshness.global_market_age_ms,
        "ledger_fresh": freshness.ledger_fresh,
    }


def _ledger_freshness_from_payload(payload: Any) -> Any:
    from mm.core.models.ledger import LedgerFreshness

    if isinstance(payload, LedgerFreshness):
        return payload
    if not isinstance(payload, dict):
        return LedgerFreshness()
    return LedgerFreshness(
        local_private_age_ms=payload.get("local_private_age_ms"),
        global_private_age_ms=payload.get("global_private_age_ms"),
        local_market_age_ms=payload.get("local_market_age_ms"),
        global_market_age_ms=payload.get("global_market_age_ms"),
        ledger_fresh=bool(payload.get("ledger_fresh", True)),
    )
