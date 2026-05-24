"""Exchange rules gateway: hard venue formatting and limit validation for place commands."""

from __future__ import annotations

from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource, order_passes_limits
from mm.execution.target_book import CommandRejection, ExecutionPlan, PlaceCommand

RULES_MIN_QTY = "rules_min_qty"
RULES_MIN_NOTIONAL = "rules_min_notional"
RULES_TICK_SIZE = "rules_tick_size"
RULES_QTY_STEP = "rules_qty_step"
RULES_PRICE_BAND = "rules_price_band"

_LEGACY_REJECTION_MAP = {
    "below_min_base_qty": RULES_MIN_QTY,
    "below_min_notional_quote": RULES_MIN_NOTIONAL,
}


def normalize_rules_reason(reason: str) -> str:
    """Map legacy limit-check reasons onto the stable rules taxonomy."""

    if reason in _LEGACY_REJECTION_MAP:
        return _LEGACY_REJECTION_MAP[reason]
    if reason.startswith("rules_"):
        return reason
    return reason


def legacy_rejection_label(reason: str) -> str:
    """Map rules taxonomy back to legacy OrderManager report labels."""

    reverse = {value: key for key, value in _LEGACY_REJECTION_MAP.items()}
    return reverse.get(reason, reason)


class ExchangeRulesGateway:
    """Validate ``PlaceCommand`` objects before they reach the order router."""

    def __init__(self, limit_source: AbstractVenueOrderLimitSource) -> None:
        self._limit_source = limit_source

    def validate(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Return a plan with invalid place commands moved to ``rejected``."""

        valid: list[PlaceCommand] = []
        rejected: list[CommandRejection] = list(plan.rejected)
        for command in plan.place:
            limits = self._limit_source.limits_for(command.order.symbol)
            ok, reason = order_passes_limits(command.order, limits)
            if ok:
                valid.append(command)
            else:
                rejected.append(
                    CommandRejection(
                        order=command.order,
                        reason=normalize_rules_reason(reason),
                        stage="rules",
                    )
                )
        return plan.with_places(tuple(valid)).with_rejected(tuple(rejected))
