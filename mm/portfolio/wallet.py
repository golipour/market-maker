"""Virtual strategy wallet: isolated capital pool per strategy instance."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Mapping

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Fill
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.private import OpenOrder
from mm.core.wallet import make_balance, order_collateral_from_open_orders
from mm.ledger.state_machine import joint_consolidated_wealth
from mm.registry.models import AllocationRecord


@dataclass(slots=True)
class StrategyWallet:
    """Isolated base/quote pool the strategy may spend and rebalance within."""

    actor_id: str
    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    initial_base: float
    initial_quote: float
    held_base: float
    held_quote: float
    locked_base: float = 0.0
    locked_quote: float = 0.0
    isolate_global_balances: bool = False
    coverage_reasons: tuple[str, ...] = ()

    @classmethod
    def from_allocation(
        cls,
        record: AllocationRecord,
        binding: SymbolBinding,
    ) -> StrategyWallet:
        local = binding.local
        return cls(
            actor_id=record.actor_id.strip(),
            exchange=local.exchange.lower().strip(),
            symbol=local.symbol.strip(),
            base_asset=(record.base_asset or local.base_asset).strip(),
            quote_asset=(record.quote_asset or local.quote_asset).strip(),
            initial_base=max(0.0, float(record.allocated_base)),
            initial_quote=max(0.0, float(record.allocated_quote)),
            held_base=max(0.0, float(record.allocated_base)),
            held_quote=max(0.0, float(record.allocated_quote)),
        )

    @property
    def free_base(self) -> float:
        return self.held_base - self.locked_base

    @property
    def free_quote(self) -> float:
        return self.held_quote - self.locked_quote

    def apply_fill(self, fill: Fill) -> None:
        if fill.actor_id.strip() != self.actor_id:
            return
        qty = max(0.0, float(fill.qty))
        price = max(0.0, float(fill.price))
        fee = max(0.0, float(fill.fee))
        side = str(fill.side).lower().strip()
        if side == "buy":
            self.held_base += qty
            self.held_quote -= price * qty + fee
        elif side == "sell":
            self.held_base -= qty
            self.held_quote += price * qty - fee

    def restore_held(
        self,
        *,
        exchange: str,
        symbol: str,
        initial_base: float,
        initial_quote: float,
        held_base: float,
        held_quote: float,
    ) -> bool:
        """Keep saved holdings when the grant still matches. A changed grant starts over."""

        if exchange.lower().strip() != self.exchange or symbol.strip() != self.symbol:
            return False
        if not _allocation_matches(self.initial_base, initial_base):
            return False
        if not _allocation_matches(self.initial_quote, initial_quote):
            return False
        self.held_base = float(held_base)
        self.held_quote = float(held_quote)
        return True

    def refresh_open_order_locks(self, order_managers: Mapping[str, AbstractOrderManager]) -> None:
        manager = order_managers.get(self.exchange)
        if manager is None:
            self.locked_base = 0.0
            self.locked_quote = 0.0
            return
        actor_orders = {
            order_id: open_order
            for order_id, open_order in manager.live_orders.items()
            if (open_order.actor_id or "").strip() == self.actor_id
        }
        pending_place_orders = getattr(manager, "pending_place_orders", None)
        if callable(pending_place_orders):
            for order_id, open_order in pending_place_orders().items():
                if order_id in actor_orders:
                    continue
                if (open_order.actor_id or "").strip() != self.actor_id:
                    continue
                actor_orders[order_id] = open_order
        locked_base, locked_quote = order_collateral_from_open_orders(
            actor_orders,
            trading_symbol=self.symbol,
            base_asset=self.base_asset,
            quote_asset=self.quote_asset,
        )
        self.locked_base = locked_base
        self.locked_quote = locked_quote

    def breach_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.held_base < -1e-9:
            reasons.append("wallet_base_deficit")
        if self.held_quote < -1e-9:
            reasons.append("wallet_quote_deficit")
        if self.free_base < -1e-9:
            reasons.append("wallet_base_locked_exceeds_held")
        if self.free_quote < -1e-9:
            reasons.append("wallet_quote_locked_exceeds_held")
        reasons.extend(self.coverage_reasons)
        return tuple(reasons)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "actor_id": self.actor_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "initial_base": self.initial_base,
            "initial_quote": self.initial_quote,
            "held_base": self.held_base,
            "held_quote": self.held_quote,
            "free_base": self.free_base,
            "free_quote": self.free_quote,
            "locked_base": self.locked_base,
            "locked_quote": self.locked_quote,
        }


def venue_ledger_from_strategy_wallet(
    wallet: StrategyWallet,
    *,
    live_orders: Mapping[str, OpenOrder] | None = None,
) -> VenueLedger:
    """Build a venue ledger view backed only by the strategy wallet."""

    scoped_orders = {
        order_id: open_order
        for order_id, open_order in (live_orders or {}).items()
        if (open_order.actor_id or "").strip() == wallet.actor_id
    }
    timestamp_ms = int(time.time() * 1000)
    free_base = max(0.0, wallet.free_base)
    free_quote = max(0.0, wallet.free_quote)
    locked_base = max(0.0, wallet.locked_base)
    locked_quote = max(0.0, wallet.locked_quote)
    return VenueLedger(
        exchange=wallet.exchange,
        symbol=wallet.symbol,
        base_asset=wallet.base_asset,
        quote_asset=wallet.quote_asset,
        available_base=free_base,
        available_quote=free_quote,
        locked_base=locked_base,
        locked_quote=locked_quote,
        balances={
            wallet.base_asset: make_balance(
                wallet.exchange,
                wallet.base_asset,
                free_base,
                locked_base,
                timestamp_ms,
                total=wallet.held_base,
            ),
            wallet.quote_asset: make_balance(
                wallet.exchange,
                wallet.quote_asset,
                free_quote,
                locked_quote,
                timestamp_ms,
                total=wallet.held_quote,
            ),
        },
        live_orders=dict(scoped_orders),
        last_private_event_ms=timestamp_ms,
        last_snapshot_ms=timestamp_ms,
    )


def isolated_global_venue(venue: VenueLedger, *, actor_id: str) -> VenueLedger:
    """Global book limited to this actor's resting orders, with no free balance.

    Several strategies can share one global account, and that account has no
    per-strategy grant. Free cash on that book would be the whole physical pile.
    """

    actor = actor_id.strip()
    scoped_orders = {
        order_id: open_order
        for order_id, open_order in venue.live_orders.items()
        if (open_order.actor_id or "").strip() == actor
    }
    locked_base, locked_quote = order_collateral_from_open_orders(
        scoped_orders,
        trading_symbol=venue.symbol,
        base_asset=venue.base_asset,
        quote_asset=venue.quote_asset,
    )
    timestamp_ms = int(time.time() * 1000)
    return replace(
        venue,
        available_base=0.0,
        available_quote=0.0,
        locked_base=locked_base,
        locked_quote=locked_quote,
        balances={
            venue.base_asset: make_balance(
                venue.exchange,
                venue.base_asset,
                0.0,
                locked_base,
                timestamp_ms,
                total=locked_base,
            ),
            venue.quote_asset: make_balance(
                venue.exchange,
                venue.quote_asset,
                0.0,
                locked_quote,
                timestamp_ms,
                total=locked_quote,
            ),
        },
        live_orders=dict(scoped_orders),
    )


def joint_ledger_with_virtual_local(
    joint: JointLedgerSnapshot,
    wallet: StrategyWallet,
) -> JointLedgerSnapshot:
    """Return a joint ledger whose local venue uses the strategy virtual wallet.

    Live orders are taken from the physical ledger so locks stay aligned with the
    shared order manager; only this actor's orders are included in the virtual view.
    A shared global account is reduced to this actor's own orders.
    """

    virtual_local = venue_ledger_from_strategy_wallet(
        wallet,
        live_orders=joint.local.live_orders,
    )
    global_ledger = joint.global_
    if wallet.isolate_global_balances and global_ledger is not None:
        global_ledger = isolated_global_venue(global_ledger, actor_id=wallet.actor_id)
    consolidated_base, consolidated_quote = joint_consolidated_wealth(
        virtual_local,
        global_ledger,
        reference_price=joint.reference_price,
    )
    return replace(
        joint,
        local=virtual_local,
        global_=global_ledger,
        consolidated_base_inventory=consolidated_base,
        consolidated_quote_wealth=consolidated_quote,
    )


def _allocation_matches(current: float, saved: float) -> bool:
    scale = max(1.0, abs(current), abs(saved))
    return abs(current - saved) <= scale * 1e-8
