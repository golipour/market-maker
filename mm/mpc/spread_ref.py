"""Architectural spread-at-reference constraints for sigmoid-mixture quotes.

When enabled, the first bid/ask sigmoid centers and scales are reparameterized so that:

- ``x0_1_bid + x0_1_ask <= S`` (``spread_max_at_ref``), with optional slack via ``sigmoid(u_tight)``
- Effective cumulative depth at each first center is at least ``Q`` (``spread_ref_quantity``):
  ``D_eff_side(x0_1) >= Q`` from component 1 alone via ``L1 >= Q / margin_1`` where
  ``margin_1 = 0.5 - sigmoid(-k_1 * x0_1)`` (baseline-shift single-component contribution).

Production uses ``D_eff(spread) = max(0, D(spread) - D(0))``; the L floor is stated in that convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SpreadRefConfig:
    """Spread-at-reference architectural constraints (no penalty weights)."""

    spread_ref_quantity: float
    spread_max_at_ref: float

    @classmethod
    def from_values(
        cls,
        *,
        spread_ref_quantity: float | None,
        spread_max_at_ref: float | None,
        spread_ref_constrain: bool = False,
    ) -> SpreadRefConfig | None:
        if not spread_ref_constrain:
            return None
        q = float(spread_ref_quantity or 0.0)
        s = float(spread_max_at_ref or 0.0)
        if q <= 0.0 or s <= 0.0:
            raise ValueError(
                "spread_ref_constrain requires positive spread_ref_quantity and spread_max_at_ref"
            )
        return cls(spread_ref_quantity=q, spread_max_at_ref=s)


def _sigmoid_np(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def first_center_margin_np(k: float, x0: float) -> float:
    """Single-component baseline-shift margin at spread ``x0``: ``0.5 - sigmoid(-k*x0)``."""
    return max(1e-12, 0.5 - _sigmoid_np(-float(k) * float(x0)))


def decode_first_centers_np(
    raw_bid_x0_slot: float,
    raw_ask_x0_slot: float,
    spread_ref: SpreadRefConfig,
) -> tuple[float, float]:
    """
    Map raw optimizer slots to first-component centers.

    ``raw_bid_x0_slot`` = ``u_skew``, ``raw_ask_x0_slot`` = ``u_tight``:
      tight = sigmoid(u_tight) in (0, 1)
      skew = sigmoid(u_skew) in (0, 1)
      x0_bid = S * tight * skew
      x0_ask = S * tight * (1 - skew)
    so ``x0_bid + x0_ask = S * tight <= S``.
    """
    s = float(spread_ref.spread_max_at_ref)
    tight = _sigmoid_np(float(raw_ask_x0_slot))
    skew = _sigmoid_np(float(raw_bid_x0_slot))
    return s * tight * skew, s * tight * (1.0 - skew)


def decode_x0_side_np(
    x0_raw: np.ndarray,
    *,
    side: str,
    spread_ref: SpreadRefConfig | None,
    raw_other_x0_slot: float | None = None,
) -> np.ndarray:
    """Decode all ``x0`` for one side; component 0 uses spread-ref reparam when configured."""
    out = np.asarray(x0_raw, dtype=np.float64).copy()
    if spread_ref is None or out.size == 0:
        return out
    if side == "bid":
        if raw_other_x0_slot is None:
            raise ValueError("raw_other_x0_slot required for bid spread-ref decode")
        x0_bid, _ = decode_first_centers_np(float(out[0]), float(raw_other_x0_slot), spread_ref)
        out[0] = x0_bid
    else:
        if raw_other_x0_slot is None:
            raise ValueError("raw_other_x0_slot required for ask spread-ref decode")
        _, x0_ask = decode_first_centers_np(float(raw_other_x0_slot), float(out[0]), spread_ref)
        out[0] = x0_ask
    return out


def decode_k_side_np(
    k_raw: np.ndarray,
    *,
    spread_ref: SpreadRefConfig | None = None,
) -> np.ndarray:
    del spread_ref  # k is always softplus(raw); spread_ref does not fix k
    return np.array(
        [float(np.log1p(np.exp(-abs(float(v)))) + max(float(v), 0.0) + 1e-8) for v in k_raw]
    )


def apply_spread_ref_L_floor_np(
    L: np.ndarray,
    k: np.ndarray,
    x0: np.ndarray,
    spread_ref: SpreadRefConfig,
) -> np.ndarray:
    """Raise ``L[0]`` so effective depth at ``x0[0]`` from component 0 alone is at least ``Q``."""
    out = np.asarray(L, dtype=np.float64).copy()
    if out.size == 0:
        return out
    q = float(spread_ref.spread_ref_quantity)
    margin = first_center_margin_np(float(k[0]), float(x0[0]))
    out[0] = max(float(out[0]), q / margin)
    return out


def decode_first_centers_torch(
    raw_bid_x0_slot: Any,
    raw_ask_x0_slot: Any,
    spread_ref: SpreadRefConfig,
) -> tuple[Any, Any]:
    torch = __import__("torch")
    s = float(spread_ref.spread_max_at_ref)
    tight = torch.sigmoid(raw_ask_x0_slot)
    skew = torch.sigmoid(raw_bid_x0_slot)
    x0_bid = s * tight * skew
    x0_ask = s * tight * (1.0 - skew)
    return x0_bid, x0_ask


def decode_x0_side_torch(
    x0_raw: Any,
    *,
    side: str,
    spread_ref: SpreadRefConfig | None,
    raw_other_x0_slot: Any | None = None,
) -> Any:
    torch = __import__("torch")
    if spread_ref is None or x0_raw.numel() == 0:
        return x0_raw
    out = x0_raw.clone()
    if side == "bid":
        if raw_other_x0_slot is None:
            raise ValueError("raw_other_x0_slot required for bid spread-ref decode")
        x0_bid, _ = decode_first_centers_torch(out[0], raw_other_x0_slot, spread_ref)
        out[0] = x0_bid
    else:
        if raw_other_x0_slot is None:
            raise ValueError("raw_other_x0_slot required for ask spread-ref decode")
        _, x0_ask = decode_first_centers_torch(raw_other_x0_slot, out[0], spread_ref)
        out[0] = x0_ask
    return out


def decode_k_side_torch(
    k_raw: Any,
    *,
    spread_ref: SpreadRefConfig | None = None,
    use_softplus: bool = True,
) -> Any:
    del spread_ref
    torch = __import__("torch")
    if use_softplus:
        return torch.nn.functional.softplus(k_raw) + 1e-8
    return k_raw


def first_center_margin_torch(k: Any, x0: Any) -> Any:
    torch = __import__("torch")
    return (0.5 - torch.sigmoid(-k * x0)).clamp(min=1e-12)


def apply_spread_ref_L_floor_torch(
    L: Any,
    k: Any,
    x0: Any,
    spread_ref: SpreadRefConfig,
) -> Any:
    torch = __import__("torch")
    if L.numel() == 0:
        return L
    q = float(spread_ref.spread_ref_quantity)
    margin = first_center_margin_torch(k[0], x0[0])
    l0 = torch.maximum(L[0], torch.tensor(q, dtype=L.dtype, device=L.device) / margin)
    out = L.clone()
    out[0] = l0
    return out


def effective_depth_at_first_center_np(
    L: np.ndarray,
    k: np.ndarray,
    x0: np.ndarray,
    *,
    comp: int = 0,
) -> float:
    """Effective depth at ``x0[comp]`` from that component alone (baseline shift)."""
    margin = first_center_margin_np(float(k[comp]), float(x0[comp]))
    return float(L[comp]) * margin
