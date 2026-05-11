"""Venue limit source helpers (default no-op)."""

from __future__ import annotations

from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.venue_order_limits import VenueOrderLimits


class NoOpVenueOrderLimitSource(AbstractVenueOrderLimitSource):
    """No limits (always passes)."""

    def limits_for(self, symbol: str) -> VenueOrderLimits:
        return VenueOrderLimits.disabled()
