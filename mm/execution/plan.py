"""Action pipeline stage reports for controller diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.execution import Action, Order


@dataclass(frozen=True, slots=True)
class ActionStageReport:
    """Per-side order totals and rejection counts at one pipeline stage."""

    order_count: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_count": self.order_count,
            "buy_qty": self.buy_qty,
            "sell_qty": self.sell_qty,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "rejection_counts": dict(self.rejection_counts),
        }


@dataclass(slots=True)
class ActionPipelineReport:
    """Full discretization → risk → execution funnel for one frame."""

    continuous: dict[str, Any] = field(default_factory=dict)
    ladder: ActionStageReport = field(default_factory=ActionStageReport)
    venue_quantized: ActionStageReport = field(default_factory=ActionStageReport)
    capped: ActionStageReport = field(default_factory=ActionStageReport)
    risk_approved: ActionStageReport = field(default_factory=ActionStageReport)
    execution_deduped: ActionStageReport = field(default_factory=ActionStageReport)
    venue_limit_valid: ActionStageReport = field(default_factory=ActionStageReport)
    venue_accepted: ActionStageReport = field(default_factory=ActionStageReport)

    def as_dict(self) -> dict[str, Any]:
        return {
            "continuous": dict(self.continuous),
            "ladder": self.ladder.as_dict(),
            "venue_quantized": self.venue_quantized.as_dict(),
            "capped": self.capped.as_dict(),
            "risk_approved": self.risk_approved.as_dict(),
            "execution_deduped": self.execution_deduped.as_dict(),
            "venue_limit_valid": self.venue_limit_valid.as_dict(),
            "venue_accepted": self.venue_accepted.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Risk gate outcome with structured rejection reasons."""

    approved: tuple[Action, ...] = ()
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def approved_list(self) -> list[Action]:
        return list(self.approved)


@dataclass(frozen=True, slots=True)
class ExecutionSyncReport:
    """Order-manager placement funnel (dedupe, venue limits, accept)."""

    dedupe_skipped: int = 0
    venue_rejection_counts: dict[str, int] = field(default_factory=dict)
    after_dedupe: ActionStageReport = field(default_factory=ActionStageReport)
    limit_valid: ActionStageReport = field(default_factory=ActionStageReport)
    accepted: ActionStageReport = field(default_factory=ActionStageReport)
    placed: bool = False

    def merge(self, other: ExecutionSyncReport) -> ExecutionSyncReport:
        merged_rejections = dict(self.venue_rejection_counts)
        for key, count in other.venue_rejection_counts.items():
            merged_rejections[key] = merged_rejections.get(key, 0) + int(count)
        return ExecutionSyncReport(
            dedupe_skipped=self.dedupe_skipped + other.dedupe_skipped,
            venue_rejection_counts=merged_rejections,
            after_dedupe=_merge_stage_reports(self.after_dedupe, other.after_dedupe),
            limit_valid=_merge_stage_reports(self.limit_valid, other.limit_valid),
            accepted=_merge_stage_reports(self.accepted, other.accepted),
            placed=self.placed or other.placed,
        )


def summarize_orders(orders: list[Order] | tuple[Order, ...]) -> ActionStageReport:
    buy_qty = sell_qty = 0.0
    buy_notional = sell_notional = 0.0
    for order in orders:
        qty = float(order.qty)
        price = float(order.price)
        side = order.side.lower().strip()
        if side == "buy":
            buy_qty += qty
            buy_notional += price * qty
        elif side == "sell":
            sell_qty += qty
            sell_notional += price * qty
    return ActionStageReport(
        order_count=len(orders),
        buy_qty=buy_qty,
        sell_qty=sell_qty,
        buy_notional=buy_notional,
        sell_notional=sell_notional,
    )


def stage_report_with_rejections(
    orders: list[Order] | tuple[Order, ...],
    rejection_counts: dict[str, int],
) -> ActionStageReport:
    base = summarize_orders(orders)
    return ActionStageReport(
        order_count=base.order_count,
        buy_qty=base.buy_qty,
        sell_qty=base.sell_qty,
        buy_notional=base.buy_notional,
        sell_notional=base.sell_notional,
        rejection_counts=dict(rejection_counts),
    )


def _merge_stage_reports(left: ActionStageReport, right: ActionStageReport) -> ActionStageReport:
    rejections = dict(left.rejection_counts)
    for key, count in right.rejection_counts.items():
        rejections[key] = rejections.get(key, 0) + int(count)
    return ActionStageReport(
        order_count=left.order_count + right.order_count,
        buy_qty=left.buy_qty + right.buy_qty,
        sell_qty=left.sell_qty + right.sell_qty,
        buy_notional=left.buy_notional + right.buy_notional,
        sell_notional=left.sell_notional + right.sell_notional,
        rejection_counts=rejections,
    )
