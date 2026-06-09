from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.adapters.joint import _order_to_target_order
from mm.core.events import to_jsonable
from mm.core.payload_schema import (
    cancel_actor_orders_requested_event,
    cancel_all_requested_event,
    cancel_orders_accepted_event,
    execution_snapshot_event,
    execution_sync_completed_event,
    execution_sync_requested_event,
)
from mm.execution.target_book import ExecutionPlan
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.execution import Action, Fill, Order
from mm.core.models.ledger import VenueLedger
from mm.core.models.target_book import VenueTargetBook
from mm.execution.plan import ExecutionSyncReport
from mm.execution.order_validation import NoOpVenueOrderLimitSource
from mm.execution.planning import plan_venue_target_book
from mm.execution.reporting import execution_plan_to_sync_report
from mm.execution.reconcile import ReconcileConfig
from mm.execution.rules_gateway import ExchangeRulesGateway
from mm.execution.router import OrderRouter
from mm.core.models.private import OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.provider_adapter import now_ms


TERMINAL_ORDER_STATUSES = {"done", "filled", "closed", "canceled", "cancelled", "rejected"}


@dataclass(frozen=True, slots=True)
class PendingOrderChange:
    """Router-accepted order change waiting for private exchange confirmation."""

    operation: str
    exchange_order_id: str
    actor_id: str | None
    accepted_ms: int
    order: Order | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    exchange: str
    timestamp_ms: int
    live_order_count: int
    live_orders: dict[str, OpenOrder]
    pending_place_count: int = 0
    pending_cancel_count: int = 0
    pending_max_age_ms: int = 0
    pending_places: dict[str, PendingOrderChange] = field(default_factory=dict)
    pending_cancels: dict[str, PendingOrderChange] = field(default_factory=dict)
    frame_id: str = ""
    decision_id: str = ""
    target_book_id: str = ""
    execution_plan_id: str = ""

    def as_payload(self) -> dict[str, Any]:
        return execution_snapshot_event(
            exchange=self.exchange,
            timestamp_ms=self.timestamp_ms,
            live_order_count=self.live_order_count,
            live_orders=to_jsonable(self.live_orders),
            pending_place_count=self.pending_place_count,
            pending_cancel_count=self.pending_cancel_count,
            pending_max_age_ms=self.pending_max_age_ms,
            pending_places=to_jsonable(self.pending_places),
            pending_cancels=to_jsonable(self.pending_cancels),
            frame_id=self.frame_id,
            decision_id=self.decision_id,
            target_book_id=self.target_book_id,
            execution_plan_id=self.execution_plan_id,
        )


class OrderManager(AbstractOrderManager):
    """Compatibility facade over planner, rules gateway, and stateless router."""

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
        self._rules_gateway = ExchangeRulesGateway(self._limit_source)
        self._router = OrderRouter(
            self.exchange,
            command_adapter,
            bus=bus,
            clock_ms=clock_ms,
        )
        self._live_orders: dict[str, OpenOrder] = {}
        self._pending_places: dict[str, PendingOrderChange] = {}
        self._pending_cancels: dict[str, PendingOrderChange] = {}
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
        started_ns = time.monotonic_ns()
        target_count = sum(len(action.orders) for action in actions)
        timestamp_ms = self.clock_ms()
        await self._publish(
            "execution:orders",
            execution_sync_requested_event(
                exchange=self.exchange,
                actor_id=_single_action_attr(actions, "actor_id"),
                symbol=_single_action_symbol(actions),
                action_count=len(actions),
                target_count=target_count,
                timestamp_ms=timestamp_ms,
            ),
        )
        combined = ExecutionSyncReport()
        for action in actions:
            combined = combined.merge(await self._reconcile_action(action, execute=True))
        await self._publish_sync_completed(
            combined,
            actor_id=_single_action_attr(actions, "actor_id"),
            symbol=_single_action_symbol(actions),
            target_count=target_count,
            latency_ms=_elapsed_ms(started_ns),
        )
        await self._publish_snapshot()
        return combined

    def preview_placement(self, actions: list[Action]) -> ExecutionSyncReport:
        """Dry-run reconcile without exchange cancel/place API calls."""

        combined = ExecutionSyncReport()
        for action in actions:
            for symbol in self._symbols_for_action(action):
                plan = self._plan_for_scope(action, symbol=symbol)
                scoped_desired = self._scoped_desired(action, symbol=symbol)
                combined = combined.merge(
                    execution_plan_to_sync_report(
                        plan,
                        scoped_desired=scoped_desired,
                        live_orders=self._live_orders,
                        placed=False,
                    )
                )
        return combined

    def preview_venue_target_book(self, target: VenueTargetBook) -> ExecutionSyncReport:
        plan = self._plan_for_venue_target(target)
        scoped_desired = [
            self._target_order_to_order(order) for order in target.orders
        ]
        return execution_plan_to_sync_report(
            plan,
            scoped_desired=scoped_desired,
            live_orders=self._live_orders,
            placed=False,
        )

    async def sync_venue_target_book(self, target: VenueTargetBook) -> ExecutionSyncReport:
        async with self._lock:
            return await self._sync_venue_target_locked(target)

    async def _sync_venue_target_locked(self, target: VenueTargetBook) -> ExecutionSyncReport:
        started_ns = time.monotonic_ns()
        target_count = len(target.orders)
        await self._publish(
            "execution:orders",
            execution_sync_requested_event(
                exchange=self.exchange,
                actor_id=target.actor_id,
                symbol=target.symbol,
                action_count=1,
                target_count=target_count,
                frame_id=target.frame_id,
                decision_id=target.decision_id,
                target_book_id=target.target_book_id,
                execution_plan_id=target.execution_plan_id,
                timestamp_ms=self.clock_ms(),
            ),
        )
        plan = self._plan_for_venue_target(target)
        scoped_desired = [self._target_order_to_order(order) for order in target.orders]
        accepted_orders: list[Order] = []
        placed = False
        live_snapshot = dict(self._live_orders)
        if plan.cancel or plan.place:
            router_result = await self._router.execute(plan)
            accepted_orders = [order for _, order in router_result.placed]
            placed = bool(router_result.placed)
            self._apply_router_result(
                Action(actor_id=target.actor_id, orders=tuple(scoped_desired)),
                plan,
                router_result,
            )
        report = execution_plan_to_sync_report(
            plan,
            scoped_desired=scoped_desired,
            live_orders=live_snapshot,
            placed=placed,
            accepted_orders=accepted_orders,
        )
        await self._publish_sync_completed(
            report,
            actor_id=target.actor_id,
            symbol=target.symbol,
            target_count=target_count,
            latency_ms=_elapsed_ms(started_ns),
            frame_id=target.frame_id,
            decision_id=target.decision_id,
            target_book_id=target.target_book_id,
            execution_plan_id=target.execution_plan_id,
        )
        await self._publish_snapshot()
        return report

    async def cancel_actor_orders(self, actor_id: str, *, reason: str = "") -> None:
        actor = str(actor_id or "").strip()
        order_ids = [
            order_id
            for order_id, open_order in self._live_orders.items()
            if (open_order.actor_id or "").strip() == actor
        ]
        if not order_ids:
            return
        await self._publish(
            "execution:orders",
            cancel_actor_orders_requested_event(
                exchange=self.exchange,
                actor_id=actor,
                reason=reason,
                count=len(order_ids),
                timestamp_ms=self.clock_ms(),
            ),
        )
        await self.command_adapter.cancel_orders(order_ids)
        timestamp_ms = self.clock_ms()
        for order_id in order_ids:
            open_order = self._live_orders.get(order_id)
            self._pending_cancels[order_id] = PendingOrderChange(
                operation="cancel",
                exchange_order_id=order_id,
                actor_id=open_order.actor_id if open_order is not None else actor,
                accepted_ms=timestamp_ms,
                order=open_order.order if open_order is not None else None,
                reason=reason,
            )
        await self._publish(
            "execution:orders",
            cancel_orders_accepted_event(
                exchange=self.exchange,
                actor_id=actor,
                order_ids=order_ids,
                timestamp_ms=self.clock_ms(),
            ),
        )
        await self._publish_snapshot()

    def apply_private_snapshot(self, snapshot: PrivateStateSnapshot) -> ExecutionSnapshot:
        self._live_orders = {
            order_id: order
            for order_id, order in snapshot.open_orders.items()
            if order.order.exchange.lower().strip() == self.exchange
        }
        self._reconcile_pending_with_confirmed()
        return self.snapshot(timestamp_ms=snapshot.timestamp_ms)

    def apply_open_order(self, open_order: OpenOrder) -> ExecutionSnapshot:
        status = open_order.status.lower().strip()
        if status in TERMINAL_ORDER_STATUSES or open_order.remaining_qty <= 0:
            self._live_orders.pop(open_order.exchange_order_id, None)
            self._pending_places.pop(open_order.exchange_order_id, None)
            self._pending_cancels.pop(open_order.exchange_order_id, None)
        else:
            self._live_orders[open_order.exchange_order_id] = open_order
            self._pending_places.pop(open_order.exchange_order_id, None)
        return self.snapshot(timestamp_ms=open_order.timestamp_ms)

    def apply_order_update(self, update: OrderUpdate) -> ExecutionSnapshot:
        status = update.status.lower().strip()
        if status in TERMINAL_ORDER_STATUSES or update.remaining_qty <= 0:
            self._live_orders.pop(update.exchange_order_id, None)
            self._pending_places.pop(update.exchange_order_id, None)
            self._pending_cancels.pop(update.exchange_order_id, None)
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
        else:
            pending = self._pending_places.get(update.exchange_order_id)
            if pending is not None and pending.order is not None:
                self._live_orders[update.exchange_order_id] = OpenOrder(
                    exchange_order_id=update.exchange_order_id,
                    actor_id=update.actor_id if update.actor_id is not None else pending.actor_id,
                    order=pending.order,
                    remaining_qty=update.remaining_qty,
                    filled_qty=update.filled_qty,
                    status=update.status,
                    timestamp_ms=update.timestamp_ms,
                )
                self._pending_places.pop(update.exchange_order_id, None)
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
        else:
            pending = self._pending_places.pop(fill.exchange_order_id, None)
            if pending is not None:
                self._pending_cancels.pop(fill.exchange_order_id, None)
        return self.snapshot(timestamp_ms=fill.timestamp_ms)

    def snapshot(self, *, timestamp_ms: int | None = None) -> ExecutionSnapshot:
        ts = self.clock_ms() if timestamp_ms is None else int(timestamp_ms)
        return ExecutionSnapshot(
            exchange=self.exchange,
            timestamp_ms=ts,
            live_order_count=len(self._live_orders),
            live_orders=dict(self._live_orders),
            pending_place_count=len(self._pending_places),
            pending_cancel_count=len(self._pending_cancels),
            pending_max_age_ms=self._pending_max_age_ms(ts),
            pending_places=dict(self._pending_places),
            pending_cancels=dict(self._pending_cancels),
        )

    async def publish_snapshot(self) -> None:
        await self._publish_snapshot()

    async def cancel_all(self, *, reason: str = "") -> None:
        order_ids = list(self._live_orders)
        if not order_ids:
            return
        await self._publish(
            "execution:orders",
            cancel_all_requested_event(
                exchange=self.exchange,
                reason=reason,
                count=len(order_ids),
                timestamp_ms=self.clock_ms(),
            ),
        )
        await self._router.command_adapter.cancel_orders(order_ids)
        timestamp_ms = self.clock_ms()
        for order_id in order_ids:
            open_order = self._live_orders.get(order_id)
            self._pending_cancels[order_id] = PendingOrderChange(
                operation="cancel",
                exchange_order_id=order_id,
                actor_id=open_order.actor_id if open_order is not None else None,
                accepted_ms=timestamp_ms,
                order=open_order.order if open_order is not None else None,
                reason=reason,
            )
        await self._publish(
            "execution:orders",
            cancel_orders_accepted_event(
                exchange=self.exchange,
                order_ids=order_ids,
                timestamp_ms=self.clock_ms(),
            ),
        )
        await self._publish_snapshot()

    async def _reconcile_action(self, action: Action, *, execute: bool) -> ExecutionSyncReport:
        combined = ExecutionSyncReport()
        for symbol in self._symbols_for_action(action):
            plan = self._plan_for_scope(action, symbol=symbol)
            scoped_desired = self._scoped_desired(action, symbol=symbol)
            accepted_orders: list[Order] = []
            placed = False
            live_snapshot = dict(self._live_orders)
            if execute and (plan.cancel or plan.place):
                router_result = await self._router.execute(plan)
                accepted_orders = [order for _, order in router_result.placed]
                placed = bool(router_result.placed)
                self._apply_router_result(action, plan, router_result)
            partial = execution_plan_to_sync_report(
                plan,
                scoped_desired=scoped_desired,
                live_orders=live_snapshot,
                placed=placed,
                accepted_orders=accepted_orders,
            )
            combined = combined.merge(partial)
        return combined

    def _apply_router_result(self, action: Action, plan: ExecutionPlan, router_result) -> None:
        """Record router-accepted changes without mutating confirmed live orders."""

        timestamp_ms = self.clock_ms()
        cancel_commands = {command.exchange_order_id: command for command in plan.cancel}
        for order_id in router_result.cancelled_ids:
            open_order = self._live_orders.get(order_id)
            command = cancel_commands.get(order_id)
            self._pending_cancels[order_id] = PendingOrderChange(
                operation="cancel",
                exchange_order_id=order_id,
                actor_id=(
                    open_order.actor_id
                    if open_order is not None
                    else command.actor_id if command is not None else action.actor_id
                ),
                accepted_ms=timestamp_ms,
                order=open_order.order if open_order is not None else None,
                reason=command.reason if command is not None else "",
            )
        for order_id, order in router_result.placed:
            self._pending_places[order_id] = PendingOrderChange(
                operation="place",
                exchange_order_id=order_id,
                actor_id=action.actor_id,
                order=order,
                accepted_ms=timestamp_ms,
            )

    def _symbols_for_action(self, action: Action) -> list[str | None]:
        symbols = {
            order.symbol
            for order in action.orders
            if order.exchange.lower().strip() == self.exchange
        }
        if not symbols:
            return [None]
        return sorted(symbols)

    def _scoped_desired(self, action: Action, *, symbol: str | None) -> list[Order]:
        return [
            order
            for order in action.orders
            if order.exchange.lower().strip() == self.exchange
            and (symbol is None or order.symbol == symbol)
        ]

    def _plan_for_scope(self, action: Action, *, symbol: str | None):
        scoped_desired = self._scoped_desired(action, symbol=symbol)
        if scoped_desired:
            target = VenueTargetBook(
                exchange=self.exchange,
                symbol=symbol or scoped_desired[0].symbol,
                actor_id=action.actor_id,
                orders=tuple(_order_to_target_order(order) for order in scoped_desired),
                policy="reconcile",
            )
        else:
            target = VenueTargetBook(
                exchange=self.exchange,
                symbol=symbol or "",
                actor_id=action.actor_id,
                orders=(),
                policy="reconcile",
            )
        return self._plan_for_venue_target(target)

    def _plan_for_venue_target(self, target: VenueTargetBook):
        pending_place_count, pending_cancel_count = self._pending_counts_for_scope(target)
        if pending_place_count or pending_cancel_count:
            return ExecutionPlan(
                exchange=target.exchange.lower().strip(),
                symbol=target.symbol,
                actor_id=target.actor_id,
                policy=target.policy,
                diagnostics={
                    "planner": "pending_confirmation_hold",
                    "pending_place_count": pending_place_count,
                    "pending_cancel_count": pending_cancel_count,
                },
                frame_id=target.frame_id,
                decision_id=target.decision_id,
                target_book_id=target.target_book_id,
                execution_plan_id=target.execution_plan_id,
            )
        ledger = VenueLedger(
            exchange=self.exchange,
            symbol=target.symbol,
            base_asset="",
            quote_asset="",
            live_orders=dict(self._live_orders),
        )
        validated_plan = plan_venue_target_book(
            target,
            ledger,
            rules_gateway=self._rules_gateway,
            reconcile_config=self._reconcile_config,
        )
        return ExecutionPlan(
            exchange=validated_plan.exchange,
            symbol=validated_plan.symbol,
            actor_id=validated_plan.actor_id,
            policy=validated_plan.policy,
            keep=validated_plan.keep,
            cancel=validated_plan.cancel,
            place=validated_plan.place,
            rejected=validated_plan.rejected,
            diagnostics={
                **validated_plan.diagnostics,
                "planned_place_orders": tuple(command.order for command in validated_plan.place),
            },
            frame_id=validated_plan.frame_id,
            decision_id=validated_plan.decision_id,
            target_book_id=validated_plan.target_book_id,
            execution_plan_id=validated_plan.execution_plan_id,
        )

    def _pending_counts_for_scope(self, target: VenueTargetBook) -> tuple[int, int]:
        ex = target.exchange.lower().strip()
        sym = target.symbol.lower().strip()
        actor = target.actor_id.strip()
        return (
            sum(
                1
                for pending in self._pending_places.values()
                if self._pending_matches_scope(pending, exchange=ex, symbol=sym, actor_id=actor)
            ),
            sum(
                1
                for pending in self._pending_cancels.values()
                if self._pending_matches_scope(pending, exchange=ex, symbol=sym, actor_id=actor)
            ),
        )

    @staticmethod
    def _pending_matches_scope(
        pending: PendingOrderChange,
        *,
        exchange: str,
        symbol: str,
        actor_id: str,
    ) -> bool:
        if pending.actor_id is not None and pending.actor_id.strip() != actor_id:
            return False
        if pending.order is None:
            return True
        return (
            pending.order.exchange.lower().strip() == exchange
            and pending.order.symbol.lower().strip() == symbol
        )

    def _pending_max_age_ms(self, timestamp_ms: int) -> int:
        accepted = [
            pending.accepted_ms
            for pending in (*self._pending_places.values(), *self._pending_cancels.values())
        ]
        if not accepted:
            return 0
        return max(0, int(timestamp_ms) - min(int(ts) for ts in accepted))

    def _reconcile_pending_with_confirmed(self) -> None:
        for order_id in list(self._pending_places):
            if order_id in self._live_orders:
                self._pending_places.pop(order_id, None)
        for order_id in list(self._pending_cancels):
            if order_id not in self._live_orders:
                self._pending_cancels.pop(order_id, None)

    @staticmethod
    def _target_order_to_order(order) -> Order:
        from mm.core.models.target_book import TargetOrder

        assert isinstance(order, TargetOrder)
        return Order(
            exchange=order.exchange,
            symbol=order.symbol,
            side=order.side,
            price=float(order.price),
            qty=float(order.qty),
            order_type=order.order_type,
        )

    async def _publish_snapshot(self) -> None:
        await self._publish("execution:state", self.snapshot().as_payload())

    async def _publish_sync_completed(
        self,
        report: ExecutionSyncReport,
        *,
        actor_id: str,
        symbol: str,
        target_count: int,
        latency_ms: float,
        frame_id: str = "",
        decision_id: str = "",
        target_book_id: str = "",
        execution_plan_id: str = "",
    ) -> None:
        timestamp_ms = self.clock_ms()
        await self._publish(
            "execution:orders",
            execution_sync_completed_event(
                exchange=self.exchange,
                actor_id=actor_id,
                symbol=symbol,
                target_count=int(target_count),
                frame_id=frame_id,
                decision_id=decision_id,
                target_book_id=target_book_id,
                execution_plan_id=execution_plan_id,
                keep_count=int(report.kept.order_count),
                cancel_count=int(report.cancel_count),
                place_count=int(report.to_place.order_count),
                accepted_count=int(report.accepted.order_count),
                rejected_count=int(sum(report.venue_rejection_counts.values())),
                placed=bool(report.placed),
                pending_place_count=len(self._pending_places),
                pending_cancel_count=len(self._pending_cancels),
                pending_max_age_ms=self._pending_max_age_ms(timestamp_ms),
                latency_ms=latency_ms,
                timestamp_ms=timestamp_ms,
            ),
        )

    async def _publish(self, topic_prefix: str, event) -> None:
        if self.bus is not None:
            await self.bus.publish(f"{topic_prefix}:{self.exchange}", event)


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000


def _single_action_attr(actions: list[Action], attr: str) -> str:
    values = {str(getattr(action, attr, "")).strip() for action in actions}
    values.discard("")
    return next(iter(values)) if len(values) == 1 else ""


def _single_action_symbol(actions: list[Action]) -> str:
    symbols = {
        order.symbol
        for action in actions
        for order in action.orders
    }
    return next(iter(symbols)) if len(symbols) == 1 else ""
