"""
Pricing and valuation **policies** (defaults and enums).

Reference VWAP uses a **target depth in lots** (see ``DEFAULT_REFERENCE_DEPTH_LOTS`` and
``ob_reference_prices``). Inventory is marked using ``ValuationPolicy`` (project default:
**sell_ref** = mark-to-bid / sell-side representative price in ticks or quote).
"""

from __future__ import annotations

from mm_core.domain.types import ValuationPolicy

# Single defaults for the whole stack (stepper, price model, MPC, runner MC).
DEFAULT_REFERENCE_DEPTH_LOTS: float = 1000.0
DEFAULT_VALUATION_POLICY: ValuationPolicy = ValuationPolicy.sell_ref

__all__ = [
    "ValuationPolicy",
    "DEFAULT_REFERENCE_DEPTH_LOTS",
    "DEFAULT_VALUATION_POLICY",
]
