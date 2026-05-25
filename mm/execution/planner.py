"""Execution planners: diff approved target books against ledger live orders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Order
from mm.core.models.ledger import VenueLedger
from mm.core.models.target_book import TargetOrder, VenueTargetBook
from mm.execution.reconcile import ReconcileConfig, reconcile_target_orders
from mm.execution.target_book import CancelCommand, ExecutionPlan, LiveOrderRef, PlaceCommand


def _target_order_to_order(order: TargetOrder) -> Order:
    return Order(
        exchange=order.exchange,
        symbol=order.symbol,
        side=order.side,
        price=float(order.price),
        qty=float(order.qty),
        order_type=order.order_type,
    )


def target_orders_to_domain_orders(target: VenueTargetBook) -> tuple[Order, ...]:
    """Convert venue target orders into ``Order`` objects for reconcile."""

    return tuple(_target_order_to_order(order) for order in target.orders)


class AbstractExecutionPlanner(ABC):
    """Plan cancel/place commands from an approved venue target book and ledger snapshot."""

    @abstractmethod
    def plan(
        self,
        target: VenueTargetBook,
        ledger: VenueLedger,
        *,
        config: ReconcileConfig | None = None,
    ) -> ExecutionPlan:
        ...


class ToleranceReconcilePlanner(AbstractExecutionPlanner):
    """Keep live orders within tolerance, cancel orphans, place unmatched targets."""

    def plan(
        self,
        target: VenueTargetBook,
        ledger: VenueLedger,
        *,
        config: ReconcileConfig | None = None,
    ) -> ExecutionPlan:
        desired = target_orders_to_domain_orders(target)
        reconcile_plan = reconcile_target_orders(
            actor_id=target.actor_id,
            exchange=target.exchange,
            symbol=target.symbol,
            desired=desired,
            live_orders=ledger.live_orders,
            config=config,
        )
        return ExecutionPlan(
            exchange=target.exchange.lower().strip(),
            symbol=target.symbol,
            actor_id=target.actor_id,
            policy="reconcile",
            keep=tuple(
                LiveOrderRef(open_order.exchange_order_id, open_order)
                for open_order in reconcile_plan.keep
            ),
            cancel=tuple(
                CancelCommand(
                    exchange_order_id=order_id,
                    actor_id=target.actor_id,
                    reason="orphan",
                )
                for order_id in reconcile_plan.cancel_ids
            ),
            place=tuple(
                PlaceCommand(actor_id=target.actor_id, order=order)
                for order in reconcile_plan.place
            ),
            diagnostics={
                "planner": "tolerance_reconcile",
                "keep_count": len(reconcile_plan.keep),
                "cancel_count": len(reconcile_plan.cancel_ids),
                "place_count": len(reconcile_plan.place),
            },
        )


class AtomicReplacePlanner(AbstractExecutionPlanner):
    """Cancel all scoped live orders, then place the full target book."""

    def plan(
        self,
        target: VenueTargetBook,
        ledger: VenueLedger,
        *,
        config: ReconcileConfig | None = None,
    ) -> ExecutionPlan:
        _ = config
        ex = target.exchange.lower().strip()
        sym = target.symbol.strip()
        actor = target.actor_id.strip()
        scoped_live = [
            open_order
            for open_order in ledger.live_orders.values()
            if _live_matches_scope(open_order, actor_id=actor, exchange=ex, symbol=sym)
        ]
        desired = target_orders_to_domain_orders(target)
        return ExecutionPlan(
            exchange=ex,
            symbol=target.symbol,
            actor_id=target.actor_id,
            policy="replace",
            keep=(),
            cancel=tuple(
                CancelCommand(
                    exchange_order_id=open_order.exchange_order_id,
                    actor_id=target.actor_id,
                    reason="replace",
                )
                for open_order in scoped_live
            ),
            place=tuple(
                PlaceCommand(actor_id=target.actor_id, order=order)
                for order in desired
            ),
            diagnostics={
                "planner": "atomic_replace",
                "cancel_count": len(scoped_live),
                "place_count": len(desired),
            },
        )


def select_planner(policy: str) -> AbstractExecutionPlanner:
    """Return the planner for a venue execution policy."""

    if policy == "replace":
        return AtomicReplacePlanner()
    return ToleranceReconcilePlanner()


def _live_matches_scope(
    open_order,
    *,
    actor_id: str,
    exchange: str,
    symbol: str,
) -> bool:
    order_actor = (open_order.actor_id or "").strip()
    if order_actor != actor_id:
        return False
    order = open_order.order
    if order.exchange.lower().strip() != exchange:
        return False
    if order.symbol.strip() != symbol:
        return False
    return True
