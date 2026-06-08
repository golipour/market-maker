"""Execution planning models: diff output from approved target books and ledger state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.execution import Order
from mm.core.models.private import OpenOrder
from mm.core.models.target_book import ExecutionPolicy


@dataclass(frozen=True, slots=True)
class LiveOrderRef:
    """Exchange-confirmed live order retained to reach the target book."""

    exchange_order_id: str
    open_order: OpenOrder


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Cancel one scoped live order."""

    exchange_order_id: str
    actor_id: str
    reason: str = "orphan"


@dataclass(frozen=True, slots=True)
class PlaceCommand:
    """Place one new resting order."""

    actor_id: str
    order: Order


@dataclass(frozen=True, slots=True)
class CommandRejection:
    """Rejected place command with a stable reason code."""

    order: Order
    reason: str
    stage: str = "rules"


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Venue-scoped cancel/place diff for one actor and symbol."""

    exchange: str
    symbol: str
    actor_id: str
    policy: ExecutionPolicy
    keep: tuple[LiveOrderRef, ...] = ()
    cancel: tuple[CancelCommand, ...] = ()
    place: tuple[PlaceCommand, ...] = ()
    rejected: tuple[CommandRejection, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    frame_id: str = ""
    decision_id: str = ""
    target_book_id: str = ""
    execution_plan_id: str = ""

    @property
    def cancel_ids(self) -> tuple[str, ...]:
        return tuple(command.exchange_order_id for command in self.cancel)

    def with_rejected(self, rejected: tuple[CommandRejection, ...]) -> ExecutionPlan:
        return ExecutionPlan(
            exchange=self.exchange,
            symbol=self.symbol,
            actor_id=self.actor_id,
            policy=self.policy,
            keep=self.keep,
            cancel=self.cancel,
            place=self.place,
            rejected=rejected,
            diagnostics=dict(self.diagnostics),
            frame_id=self.frame_id,
            decision_id=self.decision_id,
            target_book_id=self.target_book_id,
            execution_plan_id=self.execution_plan_id,
        )

    def with_places(self, place: tuple[PlaceCommand, ...]) -> ExecutionPlan:
        return ExecutionPlan(
            exchange=self.exchange,
            symbol=self.symbol,
            actor_id=self.actor_id,
            policy=self.policy,
            keep=self.keep,
            cancel=self.cancel,
            place=place,
            rejected=self.rejected,
            diagnostics=dict(self.diagnostics),
            frame_id=self.frame_id,
            decision_id=self.decision_id,
            target_book_id=self.target_book_id,
            execution_plan_id=self.execution_plan_id,
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "actor_id": self.actor_id,
            "policy": self.policy,
            "keep_count": len(self.keep),
            "cancel_count": len(self.cancel),
            "place_count": len(self.place),
            "rejected_count": len(self.rejected),
            "frame_id": self.frame_id,
            "decision_id": self.decision_id,
            "target_book_id": self.target_book_id,
            "execution_plan_id": self.execution_plan_id,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class RouterResult:
    """Adapter outcome for one executed plan (router does not mutate live-order truth)."""

    cancelled_ids: tuple[str, ...] = ()
    placed: tuple[tuple[str, Order], ...] = ()
    failed_cancel_ids: tuple[str, ...] = ()
    failed_places: tuple[CommandRejection, ...] = ()
