"""
Pricing and valuation **policies** (defaults and enums).

Reference VWAP uses a **target depth in lots** (see ``DEFAULT_REFERENCE_DEPTH_LOTS`` and
``mm.pricing`` reference VWAP helpers). Inventory is marked using ``ValuationPolicy`` (project default:
**sell_ref** = mark-to-bid / sell-side representative price in ticks or quote).

Override at runtime with env ``MM_VALUATION_POLICY`` (``sell_ref`` or ``mid``), or pass an
explicit policy into ``OrderFlowRunner``.
"""

from __future__ import annotations

import os

from mm_core.domain.types import ValuationPolicy

# Single defaults for the whole stack (stepper, price model, MPC, runner MC).
DEFAULT_REFERENCE_DEPTH_LOTS: float = 1000.0
DEFAULT_VALUATION_POLICY: ValuationPolicy = ValuationPolicy.sell_ref


def valuation_policy_from_str(s: str) -> ValuationPolicy:
    key = str(s).strip().lower().replace("-", "_")
    if key in ("mid", "m"):
        return ValuationPolicy.mid
    if key in ("sell_ref", "sellref", "mark_to_bid", "option_b"):
        return ValuationPolicy.sell_ref
    raise ValueError(
        f"Unknown valuation policy {s!r}; expected 'sell_ref' or 'mid'."
    )


def resolve_valuation_policy(explicit: ValuationPolicy | str | None = None) -> ValuationPolicy:
    """
    Resolution order: explicit constructor argument, then ``MM_VALUATION_POLICY``, then default.
    """
    if explicit is not None:
        if isinstance(explicit, ValuationPolicy):
            return explicit
        return valuation_policy_from_str(str(explicit))
    env = os.environ.get("MM_VALUATION_POLICY", "").strip()
    if env:
        return valuation_policy_from_str(env)
    return DEFAULT_VALUATION_POLICY


__all__ = [
    "ValuationPolicy",
    "DEFAULT_REFERENCE_DEPTH_LOTS",
    "DEFAULT_VALUATION_POLICY",
    "valuation_policy_from_str",
    "resolve_valuation_policy",
]
