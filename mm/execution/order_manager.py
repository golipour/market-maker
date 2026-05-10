from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.actor import AbstractOrderManager
from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Action, Fill, Order
from mm.core.models.private import OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.provider_adapter import now_ms
from mm.observability import counter_inc, gauge, latency_metric, log_event, span


TERMINAL_ORDER_STATUSES = {"done", "filled", "closed", "canceled", "cancelled", "rejected"}


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    exchange: str
    timestamp_ms: int
    live_order_count: int
    live_orders: dict[str, OpenOrder]


class OrderManager(AbstractOrderManager):
    """Owns exchange order commands and the local live-order view.

    The manager accepts float-domain strategy actions and delegates all exchange
    wire details to an AbstractOrderCommandAdapter. Private REST/WS state can be
    applied separately so own-order subtraction and observability use exchange
    truth, not only optimistic local command results.
    """

    def __init__(
        self,
        exchange: str,
        command_adapter: AbstractOrderCommandAdapter,
        *,
        bus: AbstractEventBus | None = None,
        clock_ms: Callable[[], int] = now_ms,
        cancel_missing: bool = False,
    ):
        self.exchange = exchange.lower().strip()
        self.command_adapter = command_adapter
        self.bus = bus
        self.clock_ms = clock_ms
        self.cancel_missing = bool(cancel_missing)
        self._live_orders: dict[str, OpenOrder] = {}
        self._lock = asyncio.Lock()

    @property
    def live_orders(self) -> dict[str, OpenOrder]:
        return dict(self._live_orders)

    async def sync(self, actions: list[Action]) -> None:
        labels = {"exchange": self.exchange}
        with span("mm.execution.sync", exchange=self.exchange), latency_metric(
            "mm_order_sync_latency_ms",
            labels=labels,
            help="Order manager sync latency",
        ):
            async with self._lock:
                create_count = sum(len(action.orders_to_create) for action in actions)
                cancel_count = sum(len(action.orders_to_cancel) for action in actions)
                counter_inc("mm_order_sync_total", labels=labels, help="Order manager sync calls")
                gauge("mm_order_sync_create_count", create_count, labels=labels, help="Orders requested for create")
                gauge("mm_order_sync_cancel_count", cancel_count, labels=labels, help="Orders requested for cancel")
                log_event(
                    "execution_sync_requested",
                    {
                        "exchange": self.exchange,
                        "action_count": len(actions),
                        "create_count": create_count,
                        "cancel_count": cancel_count,
                    },
                )
                await self._sync_locked(actions)

    async def _sync_locked(self, actions: list[Action]) -> None:
        await self._publish(
            "execution:orders",
            {
                "type": "ExecutionSyncRequested",
                "exchange": self.exchange,
                "action_count": len(actions),
                "create_count": sum(len(action.orders_to_create) for action in actions),
                "cancel_count": sum(len(action.orders_to_cancel) for action in actions),
                "cancel_missing": self.cancel_missing,
                "timestamp_ms": self.clock_ms(),
            },
        )
        await self._cancel_explicit(actions)
        if self.cancel_missing:
            await self._cancel_missing(actions)
        await self._place_new(actions)
        await self._publish_snapshot()

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
        log_event("execution_cancel_all_requested", {"exchange": self.exchange, "reason": reason, "count": len(order_ids)})
        await self._cancel_explicit([Action(actor_id="admin", orders_to_cancel=tuple(order_ids))])
        await self._publish_snapshot()

    async def _cancel_explicit(self, actions: list[Action]) -> None:
        cancel_ids: list[str] = []
        seen: set[str] = set()
        for action in actions:
            for order_id in action.orders_to_cancel:
                if order_id not in seen:
                    seen.add(order_id)
                    cancel_ids.append(order_id)
        if not cancel_ids:
            return
        await self._publish(
            "execution:orders",
            {
                "type": "CancelOrdersRequested",
                "exchange": self.exchange,
                "order_ids": cancel_ids,
                "timestamp_ms": self.clock_ms(),
            },
        )
        await self.command_adapter.cancel_orders(cancel_ids)
        for order_id in cancel_ids:
            self._live_orders.pop(order_id, None)
        await self._publish(
            "execution:orders",
            {
                "type": "CancelOrdersAccepted",
                "exchange": self.exchange,
                "order_ids": cancel_ids,
                "timestamp_ms": self.clock_ms(),
            },
        )

    async def _cancel_missing(self, actions: list[Action]) -> None:
        desired = {
            _order_key(action.actor_id, order)
            for action in actions
            for order in action.orders_to_create
            if order.exchange.lower().strip() == self.exchange
        }
        to_cancel = [
            order_id
            for order_id, open_order in self._live_orders.items()
            if _open_order_key(open_order) not in desired
        ]
        if to_cancel:
            await self._cancel_explicit([Action(actor_id="execution-manager", orders_to_cancel=tuple(to_cancel))])

    async def _place_new(self, actions: list[Action]) -> None:
        current_keys = {_open_order_key(open_order) for open_order in self._live_orders.values()}
        to_place: list[tuple[str, Order]] = []
        for action in actions:
            for order in action.orders_to_create:
                if order.exchange.lower().strip() != self.exchange:
                    continue
                key = _order_key(action.actor_id, order)
                if key in current_keys:
                    continue
                to_place.append((action.actor_id, order))
                current_keys.add(key)
        if not to_place:
            return

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
        for (actor_id, order), order_id in zip(to_place, order_ids):
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

    async def _publish_snapshot(self) -> None:
        gauge(
            "mm_live_order_count",
            len(self._live_orders),
            labels={"exchange": self.exchange},
            help="Live order count known to the order manager",
        )
        await self._publish("execution:state", self.snapshot())

    async def _publish(self, topic_prefix: str, event) -> None:
        if self.bus is not None:
            await self.bus.publish(f"{topic_prefix}:{self.exchange}", event)


def _order_key(actor_id: str | None, order: Order) -> tuple:
    return (
        actor_id,
        order.client_order_id,
        order.exchange.lower().strip(),
        order.symbol,
        order.side.lower().strip(),
        float(order.price),
        float(order.qty),
        order.order_type.lower().strip(),
    )


def _open_order_key(open_order: OpenOrder) -> tuple:
    return _order_key(open_order.actor_id, open_order.order)


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
