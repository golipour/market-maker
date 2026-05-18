"""Hard risk limit structs (runtime gates, not optimizer preferences)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VenueRiskLimits:
    """Hard limits for one venue role (primary quote or hedge)."""

    min_inventory: float = 0.0
    max_inventory: float = float("inf")
    min_cash: float = 0.0
    max_open_notional: float = float("inf")
    max_order_notional: float = float("inf")
    max_market_data_age_ms: int = 10_000
    max_private_state_age_ms: int = 60_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_inventory": self.min_inventory,
            "max_inventory": self.max_inventory,
            "min_cash": self.min_cash,
            "max_open_notional": self.max_open_notional,
            "max_order_notional": self.max_order_notional,
            "max_market_data_age_ms": self.max_market_data_age_ms,
            "max_private_state_age_ms": self.max_private_state_age_ms,
        }


@dataclass(frozen=True, slots=True)
class HedgeVenueRiskLimits:
    """Hard limits for hedge venue (live hedge orders in Phase 6)."""

    min_inventory: float = 0.0
    max_inventory: float = float("inf")
    min_cash: float = 0.0
    max_market_order_notional: float = float("inf")
    max_slippage_bps: float = float("inf")
    max_market_data_age_ms: int = 10_000
    max_private_state_age_ms: int = 60_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_inventory": self.min_inventory,
            "max_inventory": self.max_inventory,
            "min_cash": self.min_cash,
            "max_market_order_notional": self.max_market_order_notional,
            "max_slippage_bps": self.max_slippage_bps,
            "max_market_data_age_ms": self.max_market_data_age_ms,
            "max_private_state_age_ms": self.max_private_state_age_ms,
        }
