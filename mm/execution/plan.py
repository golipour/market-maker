"""Action pipeline stage reports for controller diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.execution import Action, Order


@dataclass(frozen=True, slots=True)
class ActionStageReport:
    """Per-side order totals, orders, and rejection details at one pipeline stage."""

    order_count: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    orders: tuple[dict[str, Any], ...] = ()
    rejections: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_count": self.order_count,
            "buy_qty": self.buy_qty,
            "sell_qty": self.sell_qty,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "rejection_counts": dict(self.rejection_counts),
            "orders": [dict(order) for order in self.orders],
            "rejections": [dict(rejection) for rejection in self.rejections],
        }


@dataclass(slots=True)
class ActionPipelineReport:
    """Full discretization → target book → risk → execution funnel for one frame."""

    continuous: dict[str, Any] = field(default_factory=dict)
    ladder: ActionStageReport = field(default_factory=ActionStageReport)
    venue_quantized: ActionStageReport = field(default_factory=ActionStageReport)
    capped: ActionStageReport = field(default_factory=ActionStageReport)
    target_book: ActionStageReport = field(default_factory=ActionStageReport)
    risk_approved_target: ActionStageReport = field(default_factory=ActionStageReport)
    planned_cancel: ActionStageReport = field(default_factory=ActionStageReport)
    planned_place: ActionStageReport = field(default_factory=ActionStageReport)
    rules_valid: ActionStageReport = field(default_factory=ActionStageReport)
    router_accepted: ActionStageReport = field(default_factory=ActionStageReport)
    kept: ActionStageReport = field(default_factory=ActionStageReport)

    def as_dict(self) -> dict[str, Any]:
        return {
            "continuous": dict(self.continuous),
            "ladder": self.ladder.as_dict(),
            "venue_quantized": self.venue_quantized.as_dict(),
            "capped": self.capped.as_dict(),
            "target_book": self.target_book.as_dict(),
            "risk_approved_target": self.risk_approved_target.as_dict(),
            "planned_cancel": self.planned_cancel.as_dict(),
            "planned_place": self.planned_place.as_dict(),
            "rules_valid": self.rules_valid.as_dict(),
            "router_accepted": self.router_accepted.as_dict(),
            "kept": self.kept.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Risk gate outcome with structured rejection reasons."""

    approved: tuple[Action, ...] = ()
    rejection_counts: dict[str, int] = field(default_factory=dict)
    exposure_before: dict[str, float] = field(default_factory=dict)
    exposure_after: dict[str, float] = field(default_factory=dict)

    @property
    def approved_list(self) -> list[Action]:
        return list(self.approved)


@dataclass(frozen=True, slots=True)
class ExecutionSyncReport:
    """Order-manager reconcile funnel (target → keep/cancel/place → venue limits → accept)."""

    target: ActionStageReport = field(default_factory=ActionStageReport)
    kept: ActionStageReport = field(default_factory=ActionStageReport)
    canceled: ActionStageReport = field(default_factory=ActionStageReport)
    to_place: ActionStageReport = field(default_factory=ActionStageReport)
    limit_valid: ActionStageReport = field(default_factory=ActionStageReport)
    accepted: ActionStageReport = field(default_factory=ActionStageReport)
    venue_rejection_counts: dict[str, int] = field(default_factory=dict)
    cancel_count: int = 0
    placed: bool = False

    def merge(self, other: ExecutionSyncReport) -> ExecutionSyncReport:
        merged_rejections = dict(self.venue_rejection_counts)
        for key, count in other.venue_rejection_counts.items():
            merged_rejections[key] = merged_rejections.get(key, 0) + int(count)
        return ExecutionSyncReport(
            target=_merge_stage_reports(self.target, other.target),
            kept=_merge_stage_reports(self.kept, other.kept),
            canceled=_merge_stage_reports(self.canceled, other.canceled),
            to_place=_merge_stage_reports(self.to_place, other.to_place),
            limit_valid=_merge_stage_reports(self.limit_valid, other.limit_valid),
            accepted=_merge_stage_reports(self.accepted, other.accepted),
            venue_rejection_counts=merged_rejections,
            cancel_count=self.cancel_count + other.cancel_count,
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
        orders=tuple(order_summary(order) for order in orders),
    )


def stage_report_with_rejections(
    orders: list[Order] | tuple[Order, ...],
    rejection_counts: dict[str, int],
    *,
    rejections: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> ActionStageReport:
    base = summarize_orders(orders)
    return ActionStageReport(
        order_count=base.order_count,
        buy_qty=base.buy_qty,
        sell_qty=base.sell_qty,
        buy_notional=base.buy_notional,
        sell_notional=base.sell_notional,
        rejection_counts=dict(rejection_counts),
        orders=base.orders,
        rejections=tuple(dict(rejection) for rejection in rejections),
    )


def summarize_open_orders(open_orders: list) -> ActionStageReport:
    """Summarize live open orders (for canceled-stage diagnostics)."""
    orders = [o.order for o in open_orders]
    return summarize_orders(orders)


def apply_execution_stages_to_pipeline(
    report: ActionPipelineReport,
    execution_report: ExecutionSyncReport,
    *,
    executed: bool,
) -> None:
    """Map order-manager reconcile output onto pipeline stage names."""

    report.kept = execution_report.kept
    report.planned_cancel = execution_report.canceled
    report.planned_place = execution_report.to_place
    report.rules_valid = ActionStageReport(
        order_count=execution_report.limit_valid.order_count,
        buy_qty=execution_report.limit_valid.buy_qty,
        sell_qty=execution_report.limit_valid.sell_qty,
        buy_notional=execution_report.limit_valid.buy_notional,
        sell_notional=execution_report.limit_valid.sell_notional,
        rejection_counts=dict(execution_report.venue_rejection_counts),
        orders=execution_report.limit_valid.orders,
        rejections=execution_report.limit_valid.rejections,
    )
    report.router_accepted = (
        execution_report.accepted if executed else report.rules_valid
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
        orders=(*left.orders, *right.orders),
        rejections=(*left.rejections, *right.rejections),
    )


def order_summary(order: Order) -> dict[str, Any]:
    """JSON-friendly order details for stage-level diagnostics."""

    return {
        "exchange": order.exchange,
        "symbol": order.symbol,
        "side": order.side,
        "price": float(order.price),
        "qty": float(order.qty),
        "notional": float(order.price) * float(order.qty),
        "order_type": order.order_type,
        "client_order_id": order.client_order_id,
    }
