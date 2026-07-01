from __future__ import annotations

from mm.core.wallet import (
    make_balance,
    normalize_snapshot_balances_for_open_orders,
    reconcile_inflated_locked_wallet_balance,
    reported_wallet_total,
)
from mm.core.asset_aliases import canonical_asset_code
from mm.core.models.execution import Fill
from mm.core.models.instrument import split_symbol
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot

# Match `mm.exchanges.shared.private_state._BALANCE_TOTAL_EPS` so zero rows drop out of snapshots.
_BALANCE_TOTAL_EPS = 1e-10


class PrivateStateStore:
    """Current private exchange state in normalized float domain values."""

    def __init__(self, exchange: str, *, trading_symbol: str | None = None):
        self.exchange = exchange.lower().strip()
        self._trading_symbol = (trading_symbol or "").strip() or None
        self._economy_assets = self._economy_canonical_assets()
        self._timestamp_ms = 0
        self._balances: dict[str, Balance] = {}
        self._open_orders: dict[str, OpenOrder] = {}
        self._fills: dict[str, Fill] = {}

    def _economy_canonical_assets(self) -> frozenset[str]:
        if not self._trading_symbol:
            return frozenset()
        try:
            base, quote = split_symbol(self._trading_symbol)
        except ValueError:
            return frozenset()
        b = canonical_asset_code(base)
        q = canonical_asset_code(quote)
        if not b or not q:
            return frozenset()
        return frozenset({b, q})

    def _ensure_economy_balances(self, balances: dict[str, Balance], *, timestamp_ms: int) -> None:
        for asset in self._economy_assets:
            if asset not in balances:
                balances[asset] = make_balance(
                    self.exchange,
                    asset,
                    0.0,
                    0.0,
                    timestamp_ms,
                    total=0.0,
                )

    def apply_snapshot(self, snapshot: PrivateStateSnapshot) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(snapshot.timestamp_ms))
        reconciled: dict[str, Balance] = {}
        for asset, balance in snapshot.balances.items():
            key = canonical_asset_code(balance.asset)
            previous = self._balances.get(key)
            reconciled[key] = reconcile_inflated_locked_wallet_balance(previous, balance)
        self._balances = reconciled
        self._ensure_economy_balances(self._balances, timestamp_ms=self._timestamp_ms)
        self._open_orders = dict(snapshot.open_orders)
        return self.snapshot()

    def apply_balance(self, balance: Balance) -> PrivateStateSnapshot:
        self._timestamp_ms = max(self._timestamp_ms, int(balance.timestamp_ms))
        key = canonical_asset_code(balance.asset)
        total = reported_wallet_total(balance)
        if total <= _BALANCE_TOTAL_EPS:
            if key in self._economy_assets:
                self._balances[key] = make_balance(
                    self.exchange,
                    key,
                    0.0,
                    0.0,
                    int(balance.timestamp_ms),
                    total=0.0,
                )
            else:
                self._balances.pop(key, None)
        else:
            previous = self._balances.get(key)
            reconciled = reconcile_inflated_locked_wallet_balance(previous, balance)
            self._balances[key] = make_balance(
                self.exchange,
                key,
                float(reconciled.free),
                float(reconciled.locked),
                int(reconciled.timestamp_ms),
                total=float(reconciled.total),
            )
        self._ensure_economy_balances(self._balances, timestamp_ms=self._timestamp_ms)
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
        balances = dict(self._balances)
        self._ensure_economy_balances(balances, timestamp_ms=self._timestamp_ms)
        snapshot = PrivateStateSnapshot(
            exchange=self.exchange,
            timestamp_ms=self._timestamp_ms,
            balances=balances,
            open_orders=dict(self._open_orders),
        )
        if self._trading_symbol:
            base, quote = split_symbol(self._trading_symbol)
            return normalize_snapshot_balances_for_open_orders(
                snapshot,
                base_asset=canonical_asset_code(base),
                quote_asset=canonical_asset_code(quote),
                trading_symbol=self._trading_symbol,
            )
        return snapshot
