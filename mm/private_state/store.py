from __future__ import annotations

from mm.core.models.execution import Fill
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot

# Match `mm.exchanges.shared.private_state._BALANCE_TOTAL_EPS` so zero rows drop out of snapshots.
_BALANCE_TOTAL_EPS = 1e-10


class PrivateStateStore:
    """Current private exchange state in normalized float domain values."""

    def __init__(self, exchange: str):
        self.exchange = exchange.lower().strip()
        self._timestamp_ms = 0
        self._balances: dict[str, Balance] = {}
        self._open_orders: dict[str, OpenOrder] = {}
        self._fills: dict[str, Fill] = {}

    def apply_snapshot(self, snapshot: PrivateStateSnapshot) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(snapshot.timestamp_ms))
        self._balances = dict(snapshot.balances)
        self._open_orders = dict(snapshot.open_orders)
        return self.snapshot()

    def apply_balance(self, balance: Balance) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(balance.timestamp_ms))
        key = balance.asset.upper()
        if abs(float(balance.free)) + abs(float(balance.locked)) <= _BALANCE_TOTAL_EPS:
            self._balances.pop(key, None)
        else:
            self._balances[key] = balance
        return self.snapshot()

    def apply_open_order(self, open_order: OpenOrder) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(open_order.timestamp_ms))
        status = open_order.status.lower().strip()
        if status in {"done", "filled", "closed", "canceled", "cancelled", "rejected"}:
            self._open_orders.pop(open_order.exchange_order_id, None)
        elif open_order.remaining_qty <= 0:
            self._open_orders.pop(open_order.exchange_order_id, None)
        else:
            self._open_orders[open_order.exchange_order_id] = open_order
        return self.snapshot()

    def apply_order_update(self, update: OrderUpdate) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(update.timestamp_ms))
        status = update.status.lower().strip()
        if status in {"done", "filled", "closed", "canceled", "cancelled", "rejected"}:
            self._open_orders.pop(update.exchange_order_id, None)
            return self.snapshot()
        current = self._open_orders.get(update.exchange_order_id)
        if current is not None:
            self._open_orders[update.exchange_order_id] = OpenOrder(
                exchange_order_id=current.exchange_order_id,
                actor_id=update.actor_id if update.actor_id is not None else current.actor_id,
                order=current.order,
                remaining_qty=update.remaining_qty,
                filled_qty=update.filled_qty,
                status=update.status,
                timestamp_ms=update.timestamp_ms,
            )
        return self.snapshot()

    def apply_fill(self, fill: Fill) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(fill.timestamp_ms))
        self._fills[fill.exchange_order_id] = fill
        current = self._open_orders.get(fill.exchange_order_id)
        if current is not None:
            remaining = max(0.0, current.remaining_qty - fill.qty)
            filled = current.filled_qty + fill.qty
            status = "filled" if remaining <= 0 else current.status
            self.apply_open_order(
                OpenOrder(
                    exchange_order_id=current.exchange_order_id,
                    actor_id=current.actor_id,
                    order=current.order,
                    remaining_qty=remaining,
                    filled_qty=filled,
                    status=status,
                    timestamp_ms=fill.timestamp_ms,
                )
            )
        return self.snapshot()

    def snapshot(self) -> PrivateStateSnapshot:
        return PrivateStateSnapshot(
            exchange=self.exchange,
            timestamp_ms=self._timestamp_ms,
            balances=dict(self._balances),
            open_orders=dict(self._open_orders),
        )
