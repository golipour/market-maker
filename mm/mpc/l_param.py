"""Depth-scale (L) parameterization for sigmoid-mixture MPC actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction, SigmoidMixtureAction
    from mm.mpc.spread_ref import SpreadRefConfig


@dataclass(frozen=True, slots=True)
class LParamConfig:
    """
    When ``use_softmax`` is true, raw L slots in the flat action are unconstrained logits ``l_i``.
    Effective depth scales are

        L_i = Q * softmax(l)_i / sum_j(softmax(l)_j * delta_j)

    where ``delta_j = sigmoid(k_j * spread_max - x0_j) - sigmoid(-k_j * x0_j)`` and
    ``Q`` is ``effective_quote_size_target()``. This enforces baseline-shift total quoted
    depth ``D(spread_max) - D(0) == Q`` for the sigmoid mixture (up to numerical tolerance).
    """

    use_softmax: bool
    quote_size_target: float | None
    spread_max: float
    quote_size_min: float | None = None
    quote_size_max: float | None = None
    apply_quote_size_range: bool = False

    def effective_quote_size_target(self) -> float | None:
        if not self.use_softmax:
            return None
        if self.quote_size_target is None or float(self.quote_size_target) <= 0.0:
            return None
        q = float(self.quote_size_target)
        if not self.apply_quote_size_range:
            return q
        lo = (
            float(self.quote_size_min)
            if self.quote_size_min is not None
            else -float("inf")
        )
        hi = (
            float(self.quote_size_max)
            if self.quote_size_max is not None
            else float("inf")
        )
        return float(min(max(q, lo), hi))


def _softplus_np(x: float) -> float:
    x = float(x)
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0) + 1e-8)


def _sigmoid_np(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def component_delta_spread_range_np(
    k_eff: np.ndarray,
    x0: np.ndarray,
    spread_max: float,
) -> np.ndarray:
    spread_hi = max(0.0, float(spread_max))
    s_hi = np.array([_sigmoid_np(float(k) * (spread_hi - float(x))) for k, x in zip(k_eff, x0)])
    s_lo = np.array([_sigmoid_np(float(k) * (0.0 - float(x))) for k, x in zip(k_eff, x0)])
    return s_hi - s_lo


def positive_L_numpy(
    L_raw: np.ndarray,
    k_raw: np.ndarray,
    x0: np.ndarray,
    *,
    l_param: LParamConfig | None,
    use_softplus: bool = True,
) -> np.ndarray:
    k_eff = np.array([_softplus_np(float(v)) for v in k_raw], dtype=np.float64)
    if l_param is None or not l_param.use_softmax:
        if use_softplus:
            return np.array([_softplus_np(float(v)) for v in L_raw], dtype=np.float64)
        return np.asarray(L_raw, dtype=np.float64)
    q_target = l_param.effective_quote_size_target()
    if q_target is None:
        return np.array([_softplus_np(float(v)) for v in L_raw], dtype=np.float64)
    logits = np.asarray(L_raw, dtype=np.float64)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / max(float(w.sum()), 1e-12)
    delta = component_delta_spread_range_np(k_eff, x0, l_param.spread_max)
    z = max(1e-12, float(np.dot(w, delta)))
    return q_target * w / z


def effective_qrange_from_flat_np(
    flat: np.ndarray,
    n_components: int,
    *,
    side_off: int,
    spread_max: float,
    l_param: LParamConfig | None,
    use_softplus: bool = True,
) -> float:
    """Baseline-shift quoted size D(spread_max) - D(0) for one side."""
    n = n_components
    L_raw = np.array([flat[side_off + 3 * i] for i in range(n)], dtype=np.float64)
    k_raw = np.array([flat[side_off + 3 * i + 1] for i in range(n)], dtype=np.float64)
    x0 = np.array([flat[side_off + 3 * i + 2] for i in range(n)], dtype=np.float64)
    L_eff = positive_L_numpy(L_raw, k_raw, x0, l_param=l_param, use_softplus=use_softplus)
    k_eff = np.array([_softplus_np(float(v)) for v in k_raw], dtype=np.float64)
    delta = component_delta_spread_range_np(k_eff, x0, spread_max)
    return float(np.dot(L_eff, delta))


def sigmoid_action_from_flat_raw(
    flat: tuple[float, ...] | list[float] | np.ndarray,
    n_components: int,
    *,
    l_param: LParamConfig | None,
    spread_ref: SpreadRefConfig | None = None,
    use_softplus: bool = True,
) -> EffectiveSigmoidMixtureAction:
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction
    from mm.mpc.spread_ref import (
        apply_spread_ref_L_floor_np,
        decode_k_side_np,
        decode_x0_side_np,
    )

    arr = np.asarray(flat, dtype=np.float64)
    n = n_components
    bid_x0 = decode_x0_side_np(
        arr[2 : 3 * n : 3],
        side="bid",
        spread_ref=spread_ref,
        raw_other_x0_slot=float(arr[3 * n + 2]) if spread_ref is not None else None,
    )
    bid_k = decode_k_side_np(arr[1 : 3 * n : 3], spread_ref=spread_ref)
    bid_L = positive_L_numpy(
        arr[0 : 3 * n : 3],
        arr[1 : 3 * n : 3],
        bid_x0,
        l_param=l_param,
        use_softplus=use_softplus,
    )
    if spread_ref is not None:
        bid_L = apply_spread_ref_L_floor_np(bid_L, bid_k, bid_x0, spread_ref)
    base = 3 * n
    ask_x0 = decode_x0_side_np(
        arr[base + 2 : base + 3 * n : 3],
        side="ask",
        spread_ref=spread_ref,
        raw_other_x0_slot=float(arr[2]) if spread_ref is not None else None,
    )
    ask_k = decode_k_side_np(arr[base + 1 : base + 3 * n : 3], spread_ref=spread_ref)
    ask_L = positive_L_numpy(
        arr[base + 0 : base + 3 * n : 3],
        arr[base + 1 : base + 3 * n : 3],
        ask_x0,
        l_param=l_param,
        use_softplus=use_softplus,
    )
    if spread_ref is not None:
        ask_L = apply_spread_ref_L_floor_np(ask_L, ask_k, ask_x0, spread_ref)
    return EffectiveSigmoidMixtureAction(
        bid_L=tuple(float(x) for x in bid_L),
        bid_k=tuple(float(x) for x in bid_k),
        bid_x0=tuple(float(x) for x in bid_x0),
        ask_L=tuple(float(x) for x in ask_L),
        ask_k=tuple(float(x) for x in ask_k),
        ask_x0=tuple(float(x) for x in ask_x0),
    )


def component_delta_spread_range_torch(
    k_eff: Any,
    x0: Any,
    spread_max: float,
) -> Any:
    torch = __import__("torch")
    spread_hi = float(spread_max)
    s_hi = torch.sigmoid(k_eff * (spread_hi - x0))
    s_lo = torch.sigmoid(k_eff * (-x0))
    return s_hi - s_lo


def positive_L_torch(
    L_raw: Any,
    k_raw: Any,
    x0: Any,
    *,
    l_param: LParamConfig | None,
    use_softplus: bool = True,
) -> Any:
    torch = __import__("torch")
    if use_softplus:
        k_eff = torch.nn.functional.softplus(k_raw) + 1e-8
    else:
        k_eff = k_raw
    if l_param is None or not l_param.use_softmax:
        if use_softplus:
            return torch.nn.functional.softplus(L_raw) + 1e-8
        return L_raw
    q_target = l_param.effective_quote_size_target()
    if q_target is None:
        return torch.nn.functional.softplus(L_raw) + 1e-8
    w = torch.softmax(L_raw, dim=-1)
    delta = component_delta_spread_range_torch(k_eff, x0, l_param.spread_max)
    z = (w * delta).sum(dim=-1).clamp(min=1e-12)
    return float(q_target) * w / z
