"""Deterministic venue ledger state machine driven by private exchange events."""

from __future__ import annotations

import time

from mm.core.adapters.joint import venue_ledger_from_private_snapshot
from mm.core.models.execution import Fill
from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness, VenueLedger
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.core.wallet import inventory_cash_from_snapshot
from mm.private_state.store import PrivateStateStore


class VenueLedgerStateMachine:
    """Apply private events and expose exchange-confirmed ``VenueLedger`` snapshots."""

    def __init__(self, exchange: str, spec: InstrumentSpec) -> None:
        self.exchange = exchange.lower().strip()
        self.spec = spec
        self._store = PrivateStateStore(self.exchange, trading_symbol=spec.symbol)

    def apply_snapshot(self, snapshot: PrivateStateSnapshot) -> VenueLedger:
        self._store.apply_snapshot(snapshot)
        return self.snapshot()

    def apply_balance(self, balance: Balance) -> VenueLedger:
        self._store.apply_balance(balance)
        return self.snapshot()

    def apply_open_order(self, open_order: OpenOrder) -> VenueLedger:
        self._store.apply_open_order(open_order)
        return self.snapshot()

    def apply_order_update(self, update: OrderUpdate) -> VenueLedger:
        self._store.apply_order_update(update)
        return self.snapshot()

    def apply_fill(self, fill: Fill) -> VenueLedger:
        self._store.apply_fill(fill)
        return self.snapshot()

    def private_snapshot(self) -> PrivateStateSnapshot:
        return self._store.snapshot()

    def snapshot(self) -> VenueLedger:
        return venue_ledger_from_private_snapshot(self._store.snapshot(), self.spec)


def build_joint_ledger_snapshot(
    local: VenueLedger,
    global_: VenueLedger | None,
    binding: SymbolBinding,
    *,
    reference_price: float = 0.0,
    created_at_ms: int | None = None,
) -> JointLedgerSnapshot:
    """Build a joint ledger snapshot from per-venue ledgers."""

    local_inventory, local_cash = _inventory_cash_from_venue_ledger(local)
    global_inventory = 0.0
    global_cash = 0.0
    if global_ is not None and binding.global_ is not None:
        global_inventory, global_cash = _inventory_cash_from_venue_ledger(global_)

    now_ms = int(created_at_ms if created_at_ms is not None else time.time() * 1000)
    ref_price = float(reference_price)
    consolidated_base = local_inventory + global_inventory
    consolidated_quote = local_cash + global_cash

    local_private_age = (
        max(0, now_ms - int(local.last_private_event_ms))
        if local.last_private_event_ms > 0
        else None
    )
    global_private_age = (
        max(0, now_ms - int(global_.last_private_event_ms))
        if global_ is not None and global_.last_private_event_ms > 0
        else None
    )

    return JointLedgerSnapshot(
        local=local,
        global_=global_,
        reference_price=ref_price,
        consolidated_base_inventory=consolidated_base,
        consolidated_quote_wealth=consolidated_quote + consolidated_base * ref_price,
        created_at_ms=now_ms,
        freshness=LedgerFreshness(
            local_private_age_ms=local_private_age,
            global_private_age_ms=global_private_age,
            ledger_fresh=local.last_snapshot_ms > 0,
        ),
    )


def _inventory_cash_from_venue_ledger(ledger: VenueLedger) -> tuple[float, float]:
    private = PrivateStateSnapshot(
        exchange=ledger.exchange,
        timestamp_ms=ledger.last_snapshot_ms,
        balances=dict(ledger.balances),
        open_orders=dict(ledger.live_orders),
    )
    return inventory_cash_from_snapshot(
        private,
        ledger.base_asset,
        ledger.quote_asset,
        trading_symbol=ledger.symbol,
    )


def compare_ledger_to_private_snapshot(
    ledger: VenueLedger,
    snapshot: PrivateStateSnapshot,
) -> bool:
    """Return True when balances and open orders match a private snapshot."""

    if set(ledger.balances.keys()) != set(snapshot.balances.keys()):
        return False
    for asset, balance in snapshot.balances.items():
        ledger_balance = ledger.balances.get(asset)
        if ledger_balance is None:
            return False
        if (
            ledger_balance.free != balance.free
            or ledger_balance.locked != balance.locked
            or ledger_balance.total != balance.total
        ):
            return False

    if set(ledger.live_orders.keys()) != set(snapshot.open_orders.keys()):
        return False
    for order_id, open_order in snapshot.open_orders.items():
        ledger_order = ledger.live_orders.get(order_id)
        if ledger_order is None:
            return False
        if (
            ledger_order.remaining_qty != open_order.remaining_qty
            or ledger_order.filled_qty != open_order.filled_qty
            or ledger_order.status != open_order.status
        ):
            return False
    return True
