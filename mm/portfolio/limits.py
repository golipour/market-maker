"""Portfolio allocation models and limits."""

from __future__ import annotations

from dataclasses import dataclass

from mm.core.risk_limits import VenueRiskLimits
from mm.registry.models import AllocationRecord


@dataclass(slots=True)
class AllocationUsage:
    actor_id: str
    attributed_base: float = 0.0
    attributed_quote: float = 0.0
    locked_base: float = 0.0
    locked_quote: float = 0.0
    realized_pnl_quote: float = 0.0

    @property
    def used_base(self) -> float:
        return max(0.0, self.attributed_base + self.locked_base)

    @property
    def used_quote(self) -> float:
        return max(0.0, max(0.0, -self.attributed_quote) + self.locked_quote)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "actor_id": self.actor_id,
            "attributed_base": self.attributed_base,
            "attributed_quote": self.attributed_quote,
            "locked_base": self.locked_base,
            "locked_quote": self.locked_quote,
            "used_base": self.used_base,
            "used_quote": self.used_quote,
            "realized_pnl_quote": self.realized_pnl_quote,
        }


@dataclass(frozen=True, slots=True)
class AllocationCaps:
    max_base_inventory: float | None = None
    max_quote_exposure: float | None = None
    attributed_base: float = 0.0
    attributed_quote_used: float = 0.0

    @classmethod
    def from_record(cls, record: AllocationRecord, usage: AllocationUsage) -> AllocationCaps:
        max_base = record.allocated_base if record.allocated_base > 0 else None
        max_quote = record.allocated_quote if record.allocated_quote > 0 else None
        return cls(
            max_base_inventory=max_base,
            max_quote_exposure=max_quote,
            attributed_base=usage.attributed_base,
            attributed_quote_used=usage.used_quote,
        )


def merge_allocation_into_venue_limits(
    limits: VenueRiskLimits,
    record: AllocationRecord | None,
) -> VenueRiskLimits:
    if record is None or record.status != "active":
        return limits
    max_inventory = limits.max_inventory
    if record.allocated_base > 0:
        max_inventory = min(max_inventory, record.allocated_base)
    max_open_notional = limits.max_open_notional
    if record.allocated_quote > 0:
        max_open_notional = min(max_open_notional, record.allocated_quote)
    return VenueRiskLimits(
        min_inventory=limits.min_inventory,
        max_inventory=max_inventory,
        min_cash=limits.min_cash,
        max_open_notional=max_open_notional,
        max_order_notional=limits.max_order_notional,
        max_market_data_age_ms=limits.max_market_data_age_ms,
        max_private_state_age_ms=limits.max_private_state_age_ms,
    )
