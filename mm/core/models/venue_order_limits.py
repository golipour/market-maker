"""Venue-agnostic numeric limits for normalizing/validating ``Order`` before REST."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VenueOrderLimits:
    """Exchange formatting and minimum-size checks in domain order units.

    ``price * qty`` is treated as **quote notional** (standard spot convention).
    Zero means \"do not apply this constraint\" (unknown or not applicable).
    """

    min_base_qty: float = 0.0
    min_notional_quote: float = 0.0
    price_step: float = 0.0
    qty_step: float = 0.0

    @classmethod
    def disabled(cls) -> VenueOrderLimits:
        return cls(0.0, 0.0, 0.0, 0.0)
