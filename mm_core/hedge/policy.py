"""
Multi-instrument hedge policy (Phase 6).

The stepper and execution path can record **one** hedge venue. ``MonteCarloMPCTensorBatched``
simulates **at most one** hedge leg (anchor inventory coupled to main fills). If ``HedgePolicy``
ever carries multiple ``hedge`` legs, **only the first** is honored by MPC until tensors are extended.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mm_core.domain.types import InstrumentId


class HedgeLegRole(str, Enum):
    """Trading venue role in the hedge graph."""

    primary = "primary"
    hedge = "hedge"


@dataclass(frozen=True, slots=True)
class HedgeInstrumentLeg:
    instrument: InstrumentId
    role: HedgeLegRole


@dataclass(frozen=True, slots=True)
class HedgePolicy:
    """
    Declarative hedge setup: which instruments participate and how MPC weights hedge PnL.

    - ``anchor_execution_fee``: proportional fee on simulated hedge venue trades (MPC ``anchor_hedge_fee``).
    - ``hedge_reward_weight``: scales hedge-leg Δwealth inside the MPC objective (0 = ignore hedge PnL).
    """

    legs: tuple[HedgeInstrumentLeg, ...]
    anchor_execution_fee: float
    hedge_reward_weight: float

    @property
    def primary(self) -> InstrumentId | None:
        for leg in self.legs:
            if leg.role == HedgeLegRole.primary:
                return leg.instrument
        return None

    @property
    def hedge_legs(self) -> tuple[HedgeInstrumentLeg, ...]:
        return tuple(l for l in self.legs if l.role == HedgeLegRole.hedge)

    @property
    def has_hedge_venue(self) -> bool:
        return len(self.hedge_legs) > 0

    @property
    def include_anchor_in_portfolio(self) -> bool:
        """Runner/engine: treat anchor balances/refs as part of strategy state (matches MPC hedge dynamics)."""
        return self.hedge_reward_weight > 0.0 and self.has_hedge_venue

    def mpc_optimizer_kwargs(self) -> dict[str, float]:
        """
        Keyword arguments for ``MonteCarloMPCTensorBatched``.

        ``anchor_hedge_coef`` remains the legacy on/off for **simulating** hedge inventory in rollouts.
        ``anchor_hedge_reward_weight`` scales hedge Δwealth in the objective.
        """
        dyn = self.include_anchor_in_portfolio
        w = float(self.hedge_reward_weight) if dyn else 0.0
        return {
            "anchor_hedge_coef": 1.0 if dyn else 0.0,
            "anchor_hedge_fee": float(self.anchor_execution_fee),
            "anchor_hedge_reward_weight": w,
        }

    @classmethod
    def from_mpc_and_instruments(
        cls,
        mpc_kw: dict,
        *,
        primary: InstrumentId,
        anchor_exchange: str | None,
        anchor_symbol: str | None,
    ) -> HedgePolicy:
        """
        Build policy from ``mpc_kwargs`` plus venue wiring (primary + optional anchor).

        Legacy: ``anchor_hedge_enabled`` / ``anchor_hedge_coef`` gate hedge; optional
        ``anchor_hedge_reward_weight`` scales hedge contribution (default 1.0 when enabled).
        """
        fee = float(mpc_kw.get("anchor_hedge_fee", 0.0))
        legacy_on = bool(
            mpc_kw.get("anchor_hedge_enabled", mpc_kw.get("anchor_hedge_coef", 0.0))
        )
        if "anchor_hedge_reward_weight" in mpc_kw:
            w = float(mpc_kw["anchor_hedge_reward_weight"])
        else:
            w = 1.0 if legacy_on else 0.0

        legs: list[HedgeInstrumentLeg] = [
            HedgeInstrumentLeg(primary, HedgeLegRole.primary),
        ]
        ax = str(anchor_exchange or "").strip()
        if ax:
            sym_a = str(anchor_symbol or "").strip() or primary.symbol
            legs.append(
                HedgeInstrumentLeg(InstrumentId(exchange=ax, symbol=sym_a), HedgeLegRole.hedge)
            )

        # No anchor venue → cannot simulate hedge; force weight 0 for consistency.
        if not ax:
            w = 0.0

        return cls(
            legs=tuple(legs),
            anchor_execution_fee=fee,
            hedge_reward_weight=max(0.0, w),
        )
