"""Portfolio ledger snapshot models for joint-venue strategy and risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.payload_schema import joint_ledger_snapshot_event, venue_ledger_snapshot_event
from mm.core.models.private import Balance, OpenOrder


@dataclass(frozen=True, slots=True)
class VenueLedger:
    """Exchange-confirmed portfolio state for one venue and symbol."""

    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    balances: dict[str, Balance] = field(default_factory=dict)
    live_orders: dict[str, OpenOrder] = field(default_factory=dict)
    locked_base: float = 0.0
    locked_quote: float = 0.0
    available_base: float = 0.0
    available_quote: float = 0.0
    last_private_event_ms: int = 0
    last_snapshot_ms: int = 0

    def as_payload(self) -> dict[str, Any]:
        return venue_ledger_snapshot_event(self)


@dataclass(frozen=True, slots=True)
class LedgerFreshness:
    """Freshness markers for a joint ledger snapshot."""

    local_private_age_ms: int | None = None
    global_private_age_ms: int | None = None
    local_market_age_ms: int | None = None
    global_market_age_ms: int | None = None
    ledger_fresh: bool = True

    def as_payload(self) -> dict[str, Any]:
        return {
            "local_private_age_ms": self.local_private_age_ms,
            "global_private_age_ms": self.global_private_age_ms,
            "local_market_age_ms": self.local_market_age_ms,
            "global_market_age_ms": self.global_market_age_ms,
            "ledger_fresh": self.ledger_fresh,
        }


@dataclass(frozen=True, slots=True)
class JointLedgerSnapshot:
    """Consolidated portfolio view across local and optional global venues."""

    local: VenueLedger
    global_: VenueLedger | None = None
    reference_price: float = 0.0
    consolidated_base_inventory: float = 0.0
    consolidated_quote_wealth: float = 0.0
    created_at_ms: int = 0
    freshness: LedgerFreshness = field(default_factory=LedgerFreshness)

    def as_payload(self) -> dict[str, Any]:
        return joint_ledger_snapshot_event(self)
