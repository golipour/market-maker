"""Target-book risk outcome models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.target_book import JointTargetBook


@dataclass(frozen=True, slots=True)
class VenueTransitionExposure:
    """Collateral and capacity snapshot for one venue transition."""

    exchange: str
    symbol: str
    actor_locked_base: float = 0.0
    actor_locked_quote: float = 0.0
    target_locked_base: float = 0.0
    target_locked_quote: float = 0.0
    available_base: float = 0.0
    available_quote: float = 0.0
    open_notional: float = 0.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "actor_locked_base": float(self.actor_locked_base),
            "actor_locked_quote": float(self.actor_locked_quote),
            "target_locked_base": float(self.target_locked_base),
            "target_locked_quote": float(self.target_locked_quote),
            "available_base": float(self.available_base),
            "available_quote": float(self.available_quote),
            "open_notional": float(self.open_notional),
        }


@dataclass(frozen=True, slots=True)
class TargetBookRiskResult:
    """Net-delta risk outcome for a joint target book transition."""

    approved: JointTargetBook | None = None
    frame_gate_reasons: tuple[str, ...] = ()
    target_book_gate_reasons: tuple[str, ...] = ()
    command_gate_reasons: tuple[str, ...] = ()
    rejection_counts: dict[str, int] = field(default_factory=dict)
    exposure_before: dict[str, float] = field(default_factory=dict)
    exposure_after: dict[str, float] = field(default_factory=dict)
    transition_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def approved_book(self) -> JointTargetBook | None:
        return self.approved

    @property
    def rejected(self) -> bool:
        return self.approved is None and (
            bool(self.frame_gate_reasons)
            or bool(self.target_book_gate_reasons)
            or bool(self.command_gate_reasons)
            or bool(self.rejection_counts)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved.as_payload() if self.approved is not None else None,
            "frame_gate_reasons": list(self.frame_gate_reasons),
            "target_book_gate_reasons": list(self.target_book_gate_reasons),
            "command_gate_reasons": list(self.command_gate_reasons),
            "rejection_counts": dict(self.rejection_counts),
            "exposure_before": dict(self.exposure_before),
            "exposure_after": dict(self.exposure_after),
            "transition_diagnostics": dict(self.transition_diagnostics),
        }
