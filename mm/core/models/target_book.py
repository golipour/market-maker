"""Target-book strategy output models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExecutionPolicy = Literal["reconcile", "replace"]
LiquiditySourcingMode = Literal["single_venue", "liquidity_sourcing"]


@dataclass(frozen=True, slots=True)
class TargetOrder:
    """Desired resting order at one venue."""

    exchange: str
    symbol: str
    side: str
    price: float
    qty: float
    order_type: str = "limit"
    tags: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "price": float(self.price),
            "qty": float(self.qty),
            "order_type": self.order_type,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class VenueTargetBook:
    """Venue-scoped desired resting book for one actor."""

    exchange: str
    symbol: str
    actor_id: str
    orders: tuple[TargetOrder, ...] = ()
    policy: ExecutionPolicy = "reconcile"
    frame_id: str = ""
    decision_id: str = ""
    target_book_id: str = ""
    execution_plan_id: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "actor_id": self.actor_id,
            "frame_id": self.frame_id,
            "decision_id": self.decision_id,
            "target_book_id": self.target_book_id,
            "execution_plan_id": self.execution_plan_id,
            "orders": [order.as_payload() for order in self.orders],
            "policy": self.policy,
        }


@dataclass(frozen=True, slots=True)
class JointTargetBook:
    """Joint strategy intent across local and optional global venues."""

    actor_id: str
    local: VenueTargetBook
    global_: VenueTargetBook | None = None
    mode: LiquiditySourcingMode = "single_venue"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    frame_id: str = ""
    decision_id: str = ""
    target_book_id: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "frame_id": self.frame_id,
            "decision_id": self.decision_id,
            "target_book_id": self.target_book_id,
            "local": self.local.as_payload(),
            "global": self.global_.as_payload() if self.global_ is not None else None,
            "mode": self.mode,
            "diagnostics": dict(self.diagnostics),
        }
