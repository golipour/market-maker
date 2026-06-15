"""Virtual strategy wallet: isolated capital pool per strategy instance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Fill
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import VenueLedger
from mm.core.models.private import OpenOrder
from mm.core.wallet import make_balance, order_collateral_from_open_orders
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
