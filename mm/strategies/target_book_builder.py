"""Build venue-scoped and joint target books from MPC sigmoid actions."""

from __future__ import annotations

from typing import Any

from mm.core.adapters.joint import joint_target_book_to_action
from mm.core.models.execution import Order
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.market import MarketEvent
from mm.core.models.target_book import (
    ExecutionPolicy,
    JointTargetBook,
    LiquiditySourcingMode,
    TargetOrder,
    VenueTargetBook,
)
from mm.core.order_constraints import OrderConstraints, build_orders_from_sigmoid_with_report
from mm.execution.plan import ActionPipelineReport, summarize_orders


def _order_to_target_order(order: Order) -> TargetOrder:
    return TargetOrder(
        exchange=order.exchange,
        symbol=order.symbol,
        side=order.side,
        price=float(order.price),
        qty=float(order.qty),
        order_type=order.order_type,
    )


def _orders_to_target_orders(orders: list[Order] | tuple[Order, ...]) -> tuple[TargetOrder, ...]:
    return tuple(_order_to_target_order(order) for order in orders)


def build_venue_target_book_from_orders(
    *,
    actor_id: str,
    orders: list[Order] | tuple[Order, ...],
    policy: ExecutionPolicy = "reconcile",
) -> VenueTargetBook:
    """Wrap discretized orders as a venue-scoped target book."""

    if not orders:
        return VenueTargetBook(
            exchange="",
            symbol="",
            actor_id=actor_id,
            policy=policy,
        )
    first = orders[0]
    return VenueTargetBook(
        exchange=first.exchange,
        symbol=first.symbol,
        actor_id=actor_id,
        orders=_orders_to_target_orders(orders),
        policy=policy,
    )


def build_joint_target_book_from_sigmoid(
    *,
    actor_id: str,
    local_event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
    local_constraints: OrderConstraints,
    continuous: dict[str, Any] | None = None,
    mode: LiquiditySourcingMode = "single_venue",
    local_policy: ExecutionPolicy = "reconcile",
    global_policy: ExecutionPolicy = "replace",
    global_event: MarketEvent | None = None,
    global_constraints: OrderConstraints | None = None,
    global_sigmoid_action: Any | None = None,
) -> tuple[JointTargetBook, ActionPipelineReport, list[Order], list[Order]]:
    """Discretize MPC output into a joint target book and legacy pipeline report."""

    local_orders, pipeline, ladder = build_orders_from_sigmoid_with_report(
        actor_id=actor_id,
        event=local_event,
        sigmoid_action=sigmoid_action,
        orderbook_grid=orderbook_grid,
        constraints=local_constraints,
        continuous=continuous,
    )
    local_book = build_venue_target_book_from_orders(
        actor_id=actor_id,
        orders=local_orders,
        policy=local_policy,
    )

    global_book: VenueTargetBook | None = None
    if (
        mode == "liquidity_sourcing"
        and global_event is not None
        and global_constraints is not None
        and global_sigmoid_action is not None
    ):
        global_orders, _global_pipeline, _ = build_orders_from_sigmoid_with_report(
            actor_id=actor_id,
            event=global_event,
            sigmoid_action=global_sigmoid_action,
            orderbook_grid=orderbook_grid,
            constraints=global_constraints,
        )
        global_book = build_venue_target_book_from_orders(
            actor_id=actor_id,
            orders=global_orders,
            policy=global_policy,
        )

    target_book_report = summarize_orders(local_orders)
    if global_book is not None:
        global_orders_flat = tuple(
            Order(
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                qty=order.qty,
                order_type=order.order_type,
            )
            for order in global_book.orders
        )
        merged = summarize_orders([*local_orders, *global_orders_flat])
        target_book_report = merged

    pipeline.target_book = target_book_report

    joint_book = JointTargetBook(
        actor_id=actor_id,
        local=local_book,
        global_=global_book,
        mode=mode,
        diagnostics={
            "local_order_count": len(local_book.orders),
            "global_order_count": len(global_book.orders) if global_book is not None else 0,
        },
    )
    return joint_book, pipeline, ladder, local_orders


def joint_target_book_to_legacy_action(book: JointTargetBook):
    """Convert a joint target book to the legacy ``Action`` execution shape."""

    return joint_target_book_to_action(book)


def local_constraints_from_spec(
    spec: InstrumentSpec,
    *,
    max_orders_per_side: int,
) -> OrderConstraints:
    return OrderConstraints.from_spec(spec, max_orders_per_side=max_orders_per_side)
