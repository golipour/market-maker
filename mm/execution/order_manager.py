from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource, order_passes_limits
from mm.core.models.execution import Action, Fill, Order
from mm.execution.plan import (
    ActionStageReport,
    ExecutionSyncReport,
    summarize_open_orders,
    summarize_orders,
)
from mm.execution.order_validation import NoOpVenueOrderLimitSource
from mm.execution.reconcile import ReconcileConfig, ReconcilePlan, reconcile_target_orders
from mm.core.models.private import OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.provider_adapter import now_ms


TERMINAL_ORDER_STATUSES = {"done", "filled", "closed", "canceled", "cancelled", "rejected"}


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    exchange: str
    timestamp_ms: int
    live_order_count: int
    live_orders: dict[str, OpenOrder]


class OrderManager(AbstractOrderManager):
    """Reconciles declarative target orders from each actor to exchange live orders."""

    def __init__(
        self,
        exchange: str,
        command_adapter: AbstractOrderCommandAdapter,
        *,
        bus: AbstractEventBus | None = None,
        clock_ms: Callable[[], int] = now_ms,
        limit_source: AbstractVenueOrderLimitSource | None = None,
        reconcile_config: ReconcileConfig | None = None,
    ):
        self.exchange = exchange.lower().strip()
        self.command_adapter = command_adapter
        self.bus = bus
        self.clock_ms = clock_ms
        self._limit_source: AbstractVenueOrderLimitSource = limit_source or NoOpVenueOrderLimitSource()
        self._reconcile_config = reconcile_config or ReconcileConfig()
        self._live_orders: dict[str, OpenOrder] = {}
        self._lock = asyncio.Lock()

    @property
    def live_orders(self) -> dict[str, OpenOrder]:
        return dict(self._live_orders)

    async def sync(self, actions: list[Action]) -> None:
        await self.sync_with_report(actions)

    async def sync_with_report(self, actions: list[Action]) -> ExecutionSyncReport:
        async with self._lock:
            return await self._sync_locked(actions)

    async def _sync_locked(self, actions: list[Action]) -> ExecutionSyncReport:
        target_count = sum(len(action.orders) for action in actions)
        await self._publish(
            "execution:orders",
            {
                "type": "ExecutionSyncRequested",
                "exchange": self.exchange,
                "action_count": len(actions),
                "target_count": target_count,
                "timestamp_ms": self.clock_ms(),
            },
        )
        combined = ExecutionSyncReport()
        for action in actions:
            combined = combined.merge(await self._reconcile_action(action, execute=True))
        await self._publish_snapshot()
        return combined

    def preview_placement(self, actions: list[Action]) -> ExecutionSyncReport:
        """Dry-run reconcile without exchange cancel/place API calls."""

        combined = ExecutionSyncReport()
        for action in actions:
            for symbol in self._symbols_for_action(action):
                plan = self._plan_for_scope(action, symbol=symbol)
                partial = self._report_from_plan(action, plan, symbol=symbol)
                filtered, venue_rejection_counts, _rejected = self._validate_orders_for_action(
                    action, plan
                )
                limit_valid = summarize_orders([order for _, order in filtered])
                combined = combined.merge(
                    ExecutionSyncReport(
                        target=partial.target,
                        kept=partial.kept,
                        canceled=partial.canceled,
                        to_place=partial.to_place,
                        limit_valid=limit_valid,
                        accepted=limit_valid,
                        venue_rejection_counts=venue_rejection_counts,
                        cancel_count=partial.cancel_count,
                    )
                )
        return combined

    def apply_private_snapshot(self, snapshot: PrivateStateSnapshot) -> ExecutionSnapshot:
        self._live_orders = {
            order_id: order
            for order_id, order in snapshot.open_orders.items()
            if order.order.exchange.lower().strip() == self.exchange
        }
        return self.snapshot(timestamp_ms=snapshot.timestamp_ms)

    def apply_open_order(self, open_order: OpenOrder) -> ExecutionSnapshot:
        status = open_order.status.lower().strip()
        if status in TERMINAL_ORDER_STATUSES or open_order.remaining_qty <= 0:
            self._live_orders.pop(open_order.exchange_order_id, None)
        else:
            self._live_orders[open_order.exchange_order_id] = open_order
        return self.snapshot(timestamp_ms=open_order.timestamp_ms)

    def apply_order_update(self, update: OrderUpdate) -> ExecutionSnapshot:
        status = update.status.lower().strip()
        if status in TERMINAL_ORDER_STATUSES or update.remaining_qty <= 0:
            self._live_orders.pop(update.exchange_order_id, None)
            return self.snapshot(timestamp_ms=update.timestamp_ms)
        current = self._live_orders.get(update.exchange_order_id)
        if current is not None:
            self._live_orders[update.exchange_order_id] = OpenOrder(
                exchange_order_id=current.exchange_order_id,
                actor_id=update.actor_id if update.actor_id is not None else current.actor_id,
                order=current.order,
                remaining_qty=update.remaining_qty,
                filled_qty=update.filled_qty,
                status=update.status,
                timestamp_ms=update.timestamp_ms,
            )
        return self.snapshot(timestamp_ms=update.timestamp_ms)

    def apply_fill(self, fill: Fill) -> ExecutionSnapshot:
        current = self._live_orders.get(fill.exchange_order_id)
        if current is not None:
            remaining = max(0.0, current.remaining_qty - fill.qty)
            filled = current.filled_qty + fill.qty
            self.apply_open_order(
                OpenOrder(
                    exchange_order_id=current.exchange_order_id,
                    actor_id=fill.actor_id or current.actor_id,
                    order=current.order,
                    remaining_qty=remaining,
                    filled_qty=filled,
                    status="filled" if remaining <= 0 else current.status,
                    timestamp_ms=fill.timestamp_ms,
                )
            )
        return self.snapshot(timestamp_ms=fill.timestamp_ms)

    def snapshot(self, *, timestamp_ms: int | None = None) -> ExecutionSnapshot:
        return ExecutionSnapshot(
            exchange=self.exchange,
            timestamp_ms=self.clock_ms() if timestamp_ms is None else int(timestamp_ms),
            live_order_count=len(self._live_orders),
            live_orders=dict(self._live_orders),
        )

    async def publish_snapshot(self) -> None:
        await self._publish_snapshot()

    async def cancel_all(self, *, reason: str = "") -> None:
        order_ids = list(self._live_orders)
        if not order_ids:
            return
        await self._publish(
            "execution:orders",
            {
                "type": "CancelAllRequested",
                "exchange": self.exchange,
                "reason": reason,
                "count": len(order_ids),
                "timestamp_ms": self.clock_ms(),
            },
        )
        await self._cancel_order_ids(order_ids)
        await self._publish_snapshot()

    async def _cancel_order_ids(self, order_ids: list[str]) -> None:
        if not order_ids:
            return
        await self._publish(
            "execution:orders",
            {
                "type": "CancelOrdersRequested",
                "exchange": self.exchange,
                "order_ids": order_ids,
                "timestamp_ms": self.clock_ms(),
            },
        )
        await self.command_adapter.cancel_orders(order_ids)
        for order_id in order_ids:
            self._live_orders.pop(order_id, None)
        await self._publish(
            "execution:orders",
            {
                "type": "CancelOrdersAccepted",
                "exchange": self.exchange,
                "order_ids": order_ids,
                "timestamp_ms": self.clock_ms(),
            },
        )

    async def _reconcile_action(self, action: Action, *, execute: bool) -> ExecutionSyncReport:
        combined = ExecutionSyncReport()
        for symbol in self._symbols_for_action(action):
            plan = self._plan_for_scope(action, symbol=symbol)
            partial = self._report_from_plan(action, plan, symbol=symbol)
            if execute and plan.cancel_ids:
                await self._cancel_order_ids(list(plan.cancel_ids))
            filtered, venue_rejection_counts, rejected = self._validate_orders_for_action(action, plan)
            if execute:
                for order, limits, reason in rejected:
                    await self._publish(
                        "execution:orders",
                        {
                            "type": "OrderValidationRejected",
                            "exchange": self.exchange,
                            "symbol": order.symbol,
                            "reason": reason,
                            "side": order.side,
                            "price": float(order.price),
                            "qty": float(order.qty),
                            "min_base_qty": float(limits.min_base_qty),
                            "min_notional_quote": float(limits.min_notional_quote),
                            "timestamp_ms": self.clock_ms(),
                        },
                    )
            limit_valid = summarize_orders([order for _, order in filtered])
            partial_report = ExecutionSyncReport(
                target=partial.target,
                kept=partial.kept,
                canceled=partial.canceled,
                to_place=partial.to_place,
                limit_valid=limit_valid,
                accepted=limit_valid,
                venue_rejection_counts=venue_rejection_counts,
                cancel_count=partial.cancel_count,
                placed=False,
            )
            if execute and filtered:
                partial_report = await self._place_filtered(
                    to_place=filtered,
                    report=partial_report,
                    limit_valid=limit_valid,
                    venue_rejection_counts=venue_rejection_counts,
                )
            combined = combined.merge(partial_report)
        return combined

    def _symbols_for_action(self, action: Action) -> list[str | None]:
        symbols = {
            order.symbol
            for order in action.orders
            if order.exchange.lower().strip() == self.exchange
        }
        if not symbols:
            return [None]
        return sorted(symbols)

    def _plan_for_scope(self, action: Action, *, symbol: str | None) -> ReconcilePlan:
        return reconcile_target_orders(
            actor_id=action.actor_id,
            exchange=self.exchange,
            symbol=symbol,
            desired=action.orders,
            live_orders=self._live_orders,
            config=self._reconcile_config,
        )

    def _report_from_plan(
        self, action: Action, plan: ReconcilePlan, *, symbol: str | None
    ) -> ExecutionSyncReport:
        scoped_desired = [
            o
            for o in action.orders
            if o.exchange.lower().strip() == self.exchange
            and (symbol is None or o.symbol == symbol)
        ]
        canceled_orders = [
            self._live_orders[oid] for oid in plan.cancel_ids if oid in self._live_orders
        ]
        return ExecutionSyncReport(
            target=summarize_orders(scoped_desired),
            kept=summarize_open_orders(list(plan.keep)),
            canceled=summarize_open_orders(canceled_orders),
            to_place=summarize_orders(plan.place),
            cancel_count=len(plan.cancel_ids),
        )

    def _validate_orders_for_action(
        self, action: Action, plan: ReconcilePlan
    ) -> tuple[list[tuple[str, Order]], dict[str, int], list[tuple[Order, Any, str]]]:
        venue_rejection_counts: dict[str, int] = {}
        filtered: list[tuple[str, Order]] = []
        rejected: list[tuple[Order, Any, str]] = []
        for order in plan.place:
            limits = self._limit_source.limits_for(order.symbol)
            ok, reason = order_passes_limits(order, limits)
            if ok:
                filtered.append((action.actor_id, order))
            else:
                venue_rejection_counts[reason] = venue_rejection_counts.get(reason, 0) + 1
                rejected.append((order, limits, reason))
        return filtered, venue_rejection_counts, rejected

    async def _place_filtered(
        self,
        *,
        to_place: list[tuple[str, Order]],
        report: ExecutionSyncReport,
        limit_valid: ActionStageReport,
        venue_rejection_counts: dict[str, int],
    ) -> ExecutionSyncReport:
        orders = [order for _, order in to_place]
        await self._publish(
            "execution:orders",
            {
                "type": "PlaceOrdersRequested",
                "exchange": self.exchange,
                "orders": [_order_summary(order) for order in orders],
                "timestamp_ms": self.clock_ms(),
            },
        )
        order_ids = await self.command_adapter.place_orders(orders)
        timestamp_ms = self.clock_ms()
        accepted_orders: list[Order] = []
        for (actor_id, order), order_id in zip(to_place, order_ids):
            accepted_orders.append(order)
            self._live_orders[order_id] = OpenOrder(
                exchange_order_id=order_id,
                actor_id=actor_id,
                order=order,
                remaining_qty=order.qty,
                filled_qty=0.0,
                status="open",
                timestamp_ms=timestamp_ms,
            )
        await self._publish(
            "execution:orders",
            {
                "type": "PlaceOrdersAccepted",
                "exchange": self.exchange,
                "order_ids": order_ids,
                "timestamp_ms": timestamp_ms,
            },
        )
        return ExecutionSyncReport(
            target=report.target,
            kept=report.kept,
            canceled=report.canceled,
            to_place=report.to_place,
            limit_valid=limit_valid,
            accepted=summarize_orders(accepted_orders),
            venue_rejection_counts=venue_rejection_counts,
            cancel_count=report.cancel_count,
            placed=True,
        )

    async def _publish_snapshot(self) -> None:
        await self._publish("execution:state", self.snapshot())

    async def _publish(self, topic_prefix: str, event) -> None:
        if self.bus is not None:
            await self.bus.publish(f"{topic_prefix}:{self.exchange}", event)


def _order_summary(order: Order) -> dict:
    return {
        "exchange": order.exchange,
        "symbol": order.symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "order_type": order.order_type,
        "client_order_id": order.client_order_id,
    }
