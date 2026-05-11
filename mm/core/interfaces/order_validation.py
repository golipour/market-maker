"""Pluggable venue limits for pre-flight order validation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Order
from mm.core.models.venue_order_limits import VenueOrderLimits


class AbstractVenueOrderLimitSource(ABC):
    """Exchange-specific provider of ``VenueOrderLimits`` keyed by trading symbol."""

    @abstractmethod
    def limits_for(self, symbol: str) -> VenueOrderLimits:
        """Return limits for ``symbol`` (cached inside implementations as needed)."""
        ...


def order_passes_limits(order: Order, limits: VenueOrderLimits, *, rel_tol: float = 1e-9) -> tuple[bool, str]:
    """Return (True, \"\") if ``order`` satisfies ``limits``; else (False, short reason)."""

    if limits.min_base_qty > 0.0 and float(order.qty) + rel_tol < float(limits.min_base_qty):
        return False, "below_min_base_qty"
    notional = float(order.price) * float(order.qty)
    if limits.min_notional_quote > 0.0 and notional + rel_tol < float(limits.min_notional_quote):
        return False, "below_min_notional_quote"
    return True, ""
