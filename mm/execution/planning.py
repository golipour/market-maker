"""Pure venue execution planning from target books and ledger snapshots."""

from __future__ import annotations

from mm.core.models.ledger import VenueLedger
from mm.core.models.target_book import VenueTargetBook
from mm.execution.planner import select_planner
from mm.execution.reconcile import ReconcileConfig
from mm.execution.rules_gateway import ExchangeRulesGateway
from mm.execution.target_book import ExecutionPlan


def plan_venue_target_book(
    target: VenueTargetBook,
    ledger: VenueLedger,
    *,
    rules_gateway: ExchangeRulesGateway,
    reconcile_config: ReconcileConfig | None = None,
    pending_place_count: int = 0,
    pending_cancel_count: int = 0,
) -> ExecutionPlan:
    """Plan cancel/place commands without router calls or optimistic pending state."""

    if pending_place_count or pending_cancel_count:
        return ExecutionPlan(
            exchange=target.exchange.lower().strip(),
            symbol=target.symbol,
            actor_id=target.actor_id,
            policy=target.policy,
            diagnostics={
                "planner": "pending_confirmation_hold",
                "pending_place_count": int(pending_place_count),
                "pending_cancel_count": int(pending_cancel_count),
            },
            frame_id=target.frame_id,
            decision_id=target.decision_id,
            target_book_id=target.target_book_id,
            execution_plan_id=target.execution_plan_id,
        )

    planner = select_planner(target.policy)
    raw_plan = planner.plan(target, ledger, config=reconcile_config)
    validated_plan = rules_gateway.validate(raw_plan)
    return ExecutionPlan(
        exchange=validated_plan.exchange,
        symbol=validated_plan.symbol,
        actor_id=validated_plan.actor_id,
        policy=validated_plan.policy,
        keep=validated_plan.keep,
        cancel=validated_plan.cancel,
        place=validated_plan.place,
        rejected=validated_plan.rejected,
        diagnostics={
            **validated_plan.diagnostics,
            "planned_place_count": len(raw_plan.place),
        },
        frame_id=validated_plan.frame_id,
        decision_id=validated_plan.decision_id,
        target_book_id=validated_plan.target_book_id,
        execution_plan_id=validated_plan.execution_plan_id,
    )
