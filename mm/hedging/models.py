"""Hedge intent and execution diagnostic events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HEDGE_INTENT_TYPE = "HedgeIntent"
HEDGE_EXECUTION_DIAGNOSTIC_TYPE = "HedgeExecutionDiagnostic"
HEDGE_DIAGNOSTICS_VERSION = 1


@dataclass(frozen=True, slots=True)
class HedgeIntent:
    """Request to offset a primary fill on the hedge venue."""

    actor_id: str
    source_fill_id: str
    primary_exchange: str
    hedge_exchange: str
    symbol: str
    primary_side: str
    hedge_side: str
    qty: float
    reference_price: float | None
    timestamp_ms: int
    run_id: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": HEDGE_INTENT_TYPE,
            "version": HEDGE_DIAGNOSTICS_VERSION,
            "actor_id": self.actor_id,
            "source_fill_id": self.source_fill_id,
            "primary_exchange": self.primary_exchange,
            "hedge_exchange": self.hedge_exchange,
            "symbol": self.symbol,
            "primary_side": self.primary_side,
            "hedge_side": self.hedge_side,
            "qty": self.qty,
            "reference_price": self.reference_price,
            "timestamp_ms": self.timestamp_ms,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class HedgeExecutionDiagnostic:
    """Outcome of attempting to hedge one primary fill."""

    actor_id: str
    status: str
    source_fill_id: str
    primary_exchange: str
    hedge_exchange: str
    symbol: str
    timestamp_ms: int
    run_id: str = ""
    reason: str | None = None
    intent: dict[str, Any] = field(default_factory=dict)
    hedge_order: dict[str, Any] = field(default_factory=dict)
    residual_unhedged_qty: float = 0.0

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": HEDGE_EXECUTION_DIAGNOSTIC_TYPE,
            "version": HEDGE_DIAGNOSTICS_VERSION,
            "actor_id": self.actor_id,
            "status": self.status,
            "reason": self.reason,
            "source_fill_id": self.source_fill_id,
            "primary_exchange": self.primary_exchange,
            "hedge_exchange": self.hedge_exchange,
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "run_id": self.run_id,
            "intent": dict(self.intent),
            "hedge_order": dict(self.hedge_order),
            "residual": {"unhedged_qty": self.residual_unhedged_qty},
        }
