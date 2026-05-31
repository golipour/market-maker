"""Convert execution plans into execution sync reports for diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mm.core.models.execution import Order
from mm.core.models.private import OpenOrder
from mm.execution.plan import (
    ActionStageReport,
    ExecutionSyncReport,
    order_summary,
    summarize_open_orders,
    summarize_orders,
)
from mm.execution.target_book import ExecutionPlan


def execution_plan_to_sync_report(
    plan: ExecutionPlan,
    *,
    scoped_desired: Sequence[Order],
    live_orders: Mapping[str, OpenOrder],
    placed: bool = False,
    accepted_orders: Sequence[Order] = (),
) -> ExecutionSyncReport:
    """Map a venue execution plan onto the ``ExecutionSyncReport`` funnel."""

    canceled_orders = [
        live_orders[command.exchange_order_id]
        for command in plan.cancel
        if command.exchange_order_id in live_orders
    ]
    planned_place_orders = _planned_place_orders(plan)
    validated_place_orders = [command.order for command in plan.place]
    rejection_counts = _rejection_counts(plan)

    limit_valid = summarize_orders(validated_place_orders)
    limit_valid = ActionStageReport(
        order_count=limit_valid.order_count,
        buy_qty=limit_valid.buy_qty,
        sell_qty=limit_valid.sell_qty,
        buy_notional=limit_valid.buy_notional,
        sell_notional=limit_valid.sell_notional,
        rejection_counts=rejection_counts,
        orders=limit_valid.orders,
        rejections=tuple(
            {
                **order_summary(rejection.order),
                "reason": rejection.reason,
                "stage": rejection.stage,
            }
            for rejection in plan.rejected
        ),
    )

    return ExecutionSyncReport(
        target=summarize_orders(scoped_desired),
        kept=summarize_open_orders([ref.open_order for ref in plan.keep]),
        canceled=summarize_open_orders(canceled_orders),
        to_place=summarize_orders(planned_place_orders),
        limit_valid=limit_valid,
        accepted=summarize_orders(accepted_orders if placed else ()),
        venue_rejection_counts=rejection_counts,
        cancel_count=len(plan.cancel),
        placed=placed,
    )


def _planned_place_orders(plan: ExecutionPlan) -> list[Order]:
    diagnostics = plan.diagnostics.get("planned_place_orders")
    if diagnostics:
        return list(diagnostics)
    return [command.order for command in plan.place]


def _rejection_counts(plan: ExecutionPlan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in plan.rejected:
        counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
    return counts
