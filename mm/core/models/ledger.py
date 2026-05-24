"""Portfolio ledger snapshot models for joint-venue strategy and risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.private import Balance, OpenOrder


@dataclass(frozen=True, slots=True)
class VenueLedger:
    """Exchange-confirmed portfolio state for one venue and symbol."""

    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    balances: dict[str, Balance] = field(default_factory=dict)
    live_orders: dict[str, OpenOrder] = field(default_factory=dict)
    locked_base: float = 0.0
    locked_quote: float = 0.0
    available_base: float = 0.0
    available_quote: float = 0.0
    last_private_event_ms: int = 0
    last_snapshot_ms: int = 0

    def as_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "balances": {
                asset: {
                    "exchange": balance.exchange,
                    "asset": balance.asset,
                    "free": float(balance.free),
                    "locked": float(balance.locked),
                    "total": float(balance.total),
                    "timestamp_ms": int(balance.timestamp_ms),
                }
                for asset, balance in self.balances.items()
            },
            "live_orders": {
                order_id: {
                    "exchange_order_id": open_order.exchange_order_id,
                    "actor_id": open_order.actor_id,
                    "remaining_qty": float(open_order.remaining_qty),
                    "filled_qty": float(open_order.filled_qty),
                    "status": open_order.status,
                    "timestamp_ms": int(open_order.timestamp_ms),
                    "order": {
                        "exchange": open_order.order.exchange,
                        "symbol": open_order.order.symbol,
                        "side": open_order.order.side,
                        "price": float(open_order.order.price),
                        "qty": float(open_order.order.qty),
                        "order_type": open_order.order.order_type,
                        "client_order_id": open_order.order.client_order_id,
                    },
                }
                for order_id, open_order in self.live_orders.items()
            },
            "locked_base": float(self.locked_base),
            "locked_quote": float(self.locked_quote),
            "available_base": float(self.available_base),
            "available_quote": float(self.available_quote),
            "last_private_event_ms": int(self.last_private_event_ms),
            "last_snapshot_ms": int(self.last_snapshot_ms),
        }


@dataclass(frozen=True, slots=True)
class LedgerFreshness:
    """Freshness markers for a joint ledger snapshot."""

    local_private_age_ms: int | None = None
    global_private_age_ms: int | None = None
    local_market_age_ms: int | None = None
    global_market_age_ms: int | None = None
    ledger_fresh: bool = True

    def as_payload(self) -> dict[str, Any]:
        return {
            "local_private_age_ms": self.local_private_age_ms,
            "global_private_age_ms": self.global_private_age_ms,
            "local_market_age_ms": self.local_market_age_ms,
            "global_market_age_ms": self.global_market_age_ms,
            "ledger_fresh": self.ledger_fresh,
        }


@dataclass(frozen=True, slots=True)
class JointLedgerSnapshot:
    """Consolidated portfolio view across local and optional global venues."""

    local: VenueLedger
    global_: VenueLedger | None = None
    reference_price: float = 0.0
    consolidated_base_inventory: float = 0.0
    consolidated_quote_wealth: float = 0.0
    created_at_ms: int = 0
    freshness: LedgerFreshness = field(default_factory=LedgerFreshness)

    def as_payload(self) -> dict[str, Any]:
        return {
            "local": self.local.as_payload(),
            "global": self.global_.as_payload() if self.global_ is not None else None,
            "reference_price": float(self.reference_price),
            "consolidated_base_inventory": float(self.consolidated_base_inventory),
            "consolidated_quote_wealth": float(self.consolidated_quote_wealth),
            "created_at_ms": int(self.created_at_ms),
            "freshness": self.freshness.as_payload(),
        }
