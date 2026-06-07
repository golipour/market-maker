"""Ledger-specific diagnostics and re-exports of core snapshot models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness, VenueLedger

__all__ = [
    "JointLedgerSnapshot",
    "LedgerConsistencyDiagnostic",
    "LedgerFreshness",
    "LedgerOrderMismatch",
    "LedgerOrderManagerMismatch",
    "VenueLedger",
]


@dataclass(frozen=True, slots=True)
class LedgerOrderMismatch:
    """One divergent live order between ledger and OrderManager."""

    exchange_order_id: str
    reason: str
    ledger_value: str | float | None = None
    manager_value: str | float | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "exchange_order_id": self.exchange_order_id,
            "reason": self.reason,
            "ledger_value": self.ledger_value,
            "manager_value": self.manager_value,
        }


@dataclass(frozen=True, slots=True)
class LedgerConsistencyDiagnostic:
    """Compare ledger live orders to OrderManager confirmed state."""

    exchange: str
    timestamp_ms: int
    ledger_order_count: int
    manager_order_count: int
    matches: bool
    mismatches: tuple[LedgerOrderMismatch, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": "LedgerConsistencyDiagnostic",
            "exchange": self.exchange,
            "timestamp_ms": int(self.timestamp_ms),
            "ledger_order_count": int(self.ledger_order_count),
            "manager_order_count": int(self.manager_order_count),
            "matches": self.matches,
            "mismatches": [m.as_payload() for m in self.mismatches],
        }


@dataclass(frozen=True, slots=True)
class LedgerOrderManagerMismatch:
    """Persistent divergence between ledger truth and OrderManager state."""

    exchange: str
    timestamp_ms: int
    first_seen_ms: int
    consecutive_count: int
    ledger_order_count: int
    manager_order_count: int
    mismatches: tuple[LedgerOrderMismatch, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        timestamp_ms = int(self.timestamp_ms)
        first_seen_ms = int(self.first_seen_ms)
        return {
            "type": "LedgerOrderManagerMismatch",
            "exchange": self.exchange,
            "timestamp_ms": timestamp_ms,
            "first_seen_ms": first_seen_ms,
            "age_ms": max(0, timestamp_ms - first_seen_ms),
            "consecutive_count": int(self.consecutive_count),
            "ledger_order_count": int(self.ledger_order_count),
            "manager_order_count": int(self.manager_order_count),
            "mismatches": [m.as_payload() for m in self.mismatches],
        }
