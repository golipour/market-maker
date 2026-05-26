"""Structured sigmoid-mixture parameter decoding for MPC quote actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction


def action_param_len(n_components: int) -> int:
    """Flat optimizer vector length: sigmoid mixture params plus per-side depth logits."""
    return 6 * max(1, int(n_components)) + 2


def joint_action_param_len(n_components: int) -> int:
    """Flat optimizer vector length for local + global sigmoid curves."""
    return 2 * action_param_len(n_components)


def depth_logit_index(*, side_off: int, n_components: int) -> int:
    base = 6 * n_components
    return base if side_off == 0 else base + 1


@dataclass(frozen=True, slots=True)
class LParamConfig:
    """
    Raw optimizer params decode into effective ``(L, k, x0)`` with structural constraints.

    Without spread geometry: global softmax on ``L`` with ``δ(spread_max)`` normalization.

    With spread geometry (``quote_spread_max`` and ``quote_spread_min_depth``):
    - first centers satisfy ``x0_1_bid + x0_1_ask <= quote_spread_max`` via a scaled softmax split
    - component 1 carries at least ``quote_spread_min_depth`` at its center
    - tail components fill remaining depth up to a chosen total in
      ``[quote_size_min, quote_size_max]`` at ``spread_max``

    Total per-side depth at ``spread_max`` is selected by a depth logit:
    ``quote_size_min + sigmoid(u_depth) * (quote_size_max - quote_size_min)``.
    """

    quote_size_min: float
    quote_size_max: float
    spread_max: float
    n_components: int = 1
    quote_spread_max: float | None = None
    quote_spread_min_depth: float | None = None
    quote_spread_k_epsilon: float = 0.01

    @property
    def spread_structured(self) -> bool:
        if self.quote_spread_max is None or float(self.quote_spread_max) <= 0.0:
            return False
        if self.quote_spread_min_depth is None or float(self.quote_spread_min_depth) <= 0.0:
            return False
        return True

    def quote_size_midpoint(self) -> float:
        return 0.5 * (float(self.quote_size_min) + float(self.quote_size_max))

    def quote_size_from_depth_logit(self, u_depth: float) -> float:
        lo = float(self.quote_size_min)
        hi = float(self.quote_size_max)
        return lo + _sigmoid_np(u_depth) * (hi - lo)

    def quote_size_from_flat(self, flat: np.ndarray, *, side_off: int, n_components: int) -> float:
        idx = depth_logit_index(side_off=side_off, n_components=n_components)
        return self.quote_size_from_depth_logit(float(flat[idx]))

    def quote_size_from_depth_logit_torch(self, u_depth: Any) -> Any:
        torch = __import__("torch")
        lo = torch.tensor(float(self.quote_size_min), dtype=u_depth.dtype, device=u_depth.device)
        hi = torch.tensor(float(self.quote_size_max), dtype=u_depth.dtype, device=u_depth.device)
        return lo + torch.sigmoid(u_depth) * (hi - lo)


def validate_l_param_config(l_param: LParamConfig) -> None:
    lo = float(l_param.quote_size_min)
    hi = float(l_param.quote_size_max)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("quote_size_min_per_side and quote_size_max_per_side must be positive")
    if lo > hi:
        raise ValueError(
            f"quote_size_min_per_side ({lo}) must be <= quote_size_max_per_side ({hi})"
        )
    if l_param.spread_structured:
        s_max = float(l_param.quote_spread_max or 0.0)
        q_spread = float(l_param.quote_spread_min_depth or 0.0)
        if s_max > float(l_param.spread_max):
            raise ValueError(
                f"quote_spread_max ({s_max}) must be <= spread_max ({l_param.spread_max})"
            )
        if q_spread > hi:
            raise ValueError(
                f"quote_spread_min_depth ({q_spread}) must be <= quote_size_max_per_side ({hi})"
            )
        if int(l_param.n_components) < 2:
            raise ValueError(
                "spread-structured softmax requires n_components >= 2 "
                "(spread leg + tail leg)"
            )


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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    return w / max(float(w.sum()), 1e-12)


def _min_structured_center(spread_cap: float) -> float:
    """Small positive center floor for spread-structured quote legs."""

    cap = max(0.0, float(spread_cap))
    if cap <= 0.0:
        return 0.0
    return min(max(1e-6, cap * 1e-6), cap * 0.5)


def k1_min_np(x0_1: float, epsilon: float) -> float:
    """Minimum k so sigmoid(-k * x0_1) <= epsilon."""
    x0_1 = max(float(x0_1), 1e-6)
    eps = float(min(max(epsilon, 1e-6), 0.49))
    logit_eps = math.log(eps / (1.0 - eps))
    return float(-logit_eps / x0_1)


def component_delta_at_np(
    k_eff: np.ndarray,
    x0: np.ndarray,
    spread: float,
) -> np.ndarray:
    s = float(spread)
    return np.array(
        [_sigmoid_np(float(k) * (s - float(x))) - _sigmoid_np(float(k) * (0.0 - float(x)))
         for k, x in zip(k_eff, x0)],
        dtype=np.float64,
    )


def component_delta_spread_range_np(
    k_eff: np.ndarray,
    x0: np.ndarray,
    spread_max: float,
) -> np.ndarray:
    return component_delta_at_np(k_eff, x0, spread_max)


def _spread_centers_from_flat(
    flat: np.ndarray,
    n_components: int,
    quote_spread_max: float,
) -> tuple[float, float]:
    """
    Decode first bid/ask centers with an architectural upper bound on their sum.

    ``u_scale`` lives in the bid-side first ``L`` slot (``flat[0]``). Then::

        scale = sigmoid(u_scale) in (0, 1)
        [x0_1_bid, x0_1_ask] = scale * quote_spread_max * softmax([u_bid, u_ask])
    """
    n = n_components
    u_scale = float(flat[0])
    u_bid = float(flat[2])
    u_ask = float(flat[3 * n + 2])
    scale = _sigmoid_np(u_scale)
    w = _softmax_np(np.array([u_bid, u_ask], dtype=np.float64))
    spread_cap = float(quote_spread_max)
    centers = np.maximum(scale * spread_cap * w, _min_structured_center(spread_cap))
    return float(centers[0]), float(centers[1])


def _solve_spread_leg_L_np(
    *,
    q_total: float,
    q_spread: float,
    delta_center: np.ndarray,
    delta_max: np.ndarray,
    tail_v: np.ndarray,
) -> np.ndarray:
    n = len(delta_center)
    L_eff = np.zeros(n, dtype=np.float64)
    d1_c = float(delta_center[0])
    d1_m = float(delta_max[0])
    d1_c = max(d1_c, 1e-12)
    if n == 1:
        L_eff[0] = q_total / max(float(delta_max[0]), 1e-12)
        return L_eff
    z_tm = max(float(np.dot(tail_v, delta_max[1:])), 1e-12)
    tail_c = float(np.dot(tail_v, delta_center[1:]))
    denom = d1_c - d1_m * tail_c / z_tm
    numer = q_spread - q_total * tail_c / z_tm
    if abs(denom) < 1e-12:
        L1 = q_spread / d1_c
    else:
        L1 = numer / denom
    L1 = max(L1, q_spread / d1_c)
    q_tail = max(0.0, q_total - L1 * d1_m)
    L_eff[0] = L1
    L_eff[1:] = q_tail * tail_v / z_tm
    return L_eff


def materialize_side_np(
    flat: np.ndarray,
    *,
    side_off: int,
    n_components: int,
    l_param: LParamConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one side of a flat action into effective (L, k, x0)."""
    n = n_components
    L_raw = np.array([flat[side_off + 3 * i] for i in range(n)], dtype=np.float64)
    k_raw = np.array([flat[side_off + 3 * i + 1] for i in range(n)], dtype=np.float64)
    x0_raw = np.array([flat[side_off + 3 * i + 2] for i in range(n)], dtype=np.float64)
    spread_max = float(l_param.spread_max)

    if l_param.spread_structured:
        x0_bid_1, x0_ask_1 = _spread_centers_from_flat(
            flat,
            n,
            float(l_param.quote_spread_max),
        )
        x0_first = x0_bid_1 if side_off == 0 else x0_ask_1
        x0_eff = np.zeros(n, dtype=np.float64)
        x0_eff[0] = x0_first
        for i in range(1, n):
            frac = _sigmoid_np(float(x0_raw[i]))
            x0_eff[i] = x0_first + (spread_max - x0_first) * frac
    else:
        x0_eff = spread_max * np.array([_sigmoid_np(float(v)) for v in x0_raw], dtype=np.float64)

    k_eff = np.zeros(n, dtype=np.float64)
    eps = float(l_param.quote_spread_k_epsilon)
    if l_param.spread_structured:
        k_eff[0] = k1_min_np(float(x0_eff[0]), eps) + _softplus_np(float(k_raw[0]))
        for i in range(1, n):
            k_eff[i] = _softplus_np(float(k_raw[i]))
    else:
        k_eff = np.array([_softplus_np(float(v)) for v in k_raw], dtype=np.float64)

    q_total = l_param.quote_size_from_flat(flat, side_off=side_off, n_components=n)
    if l_param.spread_structured:
        q_spread = float(l_param.quote_spread_min_depth)
        delta_at_center = component_delta_at_np(k_eff, x0_eff, float(x0_eff[0]))
        delta_at_max = component_delta_at_np(k_eff, x0_eff, spread_max)
        if n > 1:
            tail_logits = L_raw[1:]
            tail_v = _softmax_np(tail_logits)
            L_eff = _solve_spread_leg_L_np(
                q_total=q_total,
                q_spread=q_spread,
                delta_center=delta_at_center,
                delta_max=delta_at_max,
                tail_v=tail_v,
            )
        else:
            L_eff = np.zeros(n, dtype=np.float64)
            L_eff[0] = q_total / max(float(delta_at_max[0]), 1e-12)
        return L_eff, k_eff, x0_eff

    logits = L_raw
    w = _softmax_np(logits)
    delta = component_delta_spread_range_np(k_eff, x0_eff, spread_max)
    z = max(1e-12, float(np.dot(w, delta)))
    L_eff = q_total * w / z
    return L_eff, k_eff, x0_eff


def positive_L_numpy(
    L_raw: np.ndarray,
    k_raw: np.ndarray,
    x0_raw: np.ndarray,
    *,
    flat: np.ndarray | None,
    side_off: int,
    l_param: LParamConfig | None,
    use_softplus: bool = True,
) -> np.ndarray:
    del k_raw, x0_raw, use_softplus
    if l_param is None:
        raise ValueError("l_param is required for structured softmax decode")
    if flat is None:
        raise ValueError("flat action required for structured softmax decode")
    L_eff, _, _ = materialize_side_np(flat, side_off=side_off, n_components=len(L_raw), l_param=l_param)
    return L_eff


def effective_qrange_from_flat_np(
    flat: np.ndarray,
    n_components: int,
    *,
    side_off: int,
    spread_max: float,
    l_param: LParamConfig,
    use_softplus: bool = True,
) -> float:
    del use_softplus, spread_max
    L_eff, k_eff, x0_eff = materialize_side_np(
        flat, side_off=side_off, n_components=n_components, l_param=l_param
    )
    delta = component_delta_spread_range_np(k_eff, x0_eff, l_param.spread_max)
    return float(np.dot(L_eff, delta))


def effective_depth_at_center_from_flat_np(
    flat: np.ndarray,
    n_components: int,
    *,
    side_off: int,
    l_param: LParamConfig,
) -> float:
    L_eff, k_eff, x0_eff = materialize_side_np(
        flat, side_off=side_off, n_components=n_components, l_param=l_param
    )
    s_ref = float(x0_eff[0])
    delta = component_delta_at_np(k_eff, x0_eff, s_ref)
    return float(np.dot(L_eff, delta))


def sigmoid_action_from_flat_raw(
    flat: tuple[float, ...] | list[float] | np.ndarray,
    n_components: int,
    *,
    l_param: LParamConfig,
    use_softplus: bool = True,
) -> EffectiveSigmoidMixtureAction:
    del use_softplus
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction

    arr = np.asarray(flat, dtype=np.float64)
    n = n_components
    bid_L, bid_k, bid_x0 = materialize_side_np(arr, side_off=0, n_components=n, l_param=l_param)
    ask_L, ask_k, ask_x0 = materialize_side_np(arr, side_off=3 * n, n_components=n, l_param=l_param)
    return EffectiveSigmoidMixtureAction(
        bid_L=tuple(float(x) for x in bid_L),
        bid_k=tuple(float(x) for x in bid_k),
        bid_x0=tuple(float(x) for x in bid_x0),
        ask_L=tuple(float(x) for x in ask_L),
        ask_k=tuple(float(x) for x in ask_k),
        ask_x0=tuple(float(x) for x in ask_x0),
    )


def component_delta_at_torch(
    k_eff: Any,
    x0: Any,
    spread: float,
) -> Any:
    torch = __import__("torch")
    s = float(spread)
    return torch.sigmoid(k_eff * (s - x0)) - torch.sigmoid(k_eff * (-x0))


def component_delta_spread_range_torch(
    k_eff: Any,
    x0: Any,
    spread_max: float,
) -> Any:
    return component_delta_at_torch(k_eff, x0, spread_max)


def _solve_spread_leg_L_torch(
    *,
    q_total: Any,
    q_spread: float,
    delta_center: Any,
    delta_max: Any,
    tail_v: Any,
) -> Any:
    torch = __import__("torch")
    n = int(delta_center.numel())
    d1_c = delta_center[0].clamp(min=1e-12)
    d1_m = delta_max[0]
    if n == 1:
        L_eff = torch.zeros(1, dtype=d1_c.dtype, device=d1_c.device)
        L_eff[0] = q_total / delta_max[0].clamp(min=1e-12)
        return L_eff
    z_tm = (tail_v * delta_max[1:]).sum().clamp(min=1e-12)
    tail_c = (tail_v * delta_center[1:]).sum()
    denom = d1_c - d1_m * tail_c / z_tm
    numer = torch.tensor(q_spread, dtype=d1_c.dtype, device=d1_c.device) - q_total * tail_c / z_tm
    L1 = torch.where(
        denom.abs() < 1e-12,
        torch.tensor(q_spread, dtype=d1_c.dtype, device=d1_c.device) / d1_c,
        numer / denom,
    )
    L1 = torch.maximum(L1, torch.tensor(q_spread, dtype=d1_c.dtype, device=d1_c.device) / d1_c)
    q_tail = (q_total - L1 * d1_m).clamp(min=0.0)
    L_eff = torch.zeros(n, dtype=d1_c.dtype, device=d1_c.device)
    L_eff[0] = L1
    L_eff[1:] = q_tail * tail_v / z_tm
    return L_eff


def materialize_side_torch(
    params: Any,
    *,
    side_off: int,
    n_components: int,
    l_param: LParamConfig,
) -> tuple[Any, Any, Any]:
    torch = __import__("torch")
    n = n_components
    L_raw = params[side_off + 0 : side_off + 3 * n : 3]
    k_raw = params[side_off + 1 : side_off + 3 * n : 3]
    x0_raw = params[side_off + 2 : side_off + 3 * n : 3]
    spread_max = float(l_param.spread_max)

    min_len = action_param_len(n)
    if params.numel() < min_len:
        raise ValueError(
            f"structured softmax requires action params len>={min_len}, got {params.numel()}"
        )

    u_bid = params[2]
    u_ask = params[3 * n + 2]
    u_scale = params[0]
    u_depth = params[depth_logit_index(side_off=side_off, n_components=n)]

    if l_param.spread_structured:
        spread_cap = float(l_param.quote_spread_max)
        scale = torch.sigmoid(u_scale)
        spread_w = torch.softmax(torch.stack([u_bid, u_ask]), dim=0)
        min_center = torch.tensor(
            _min_structured_center(spread_cap),
            dtype=params.dtype,
            device=params.device,
        )
        centers = torch.clamp(scale * spread_cap * spread_w, min=min_center)
        x0_first = centers[0] if side_off == 0 else centers[1]
        x0_eff = torch.zeros(n, dtype=params.dtype, device=params.device)
        x0_eff[0] = x0_first
        for i in range(1, n):
            frac = torch.sigmoid(x0_raw[i])
            x0_eff[i] = x0_first + (spread_max - x0_first) * frac
    else:
        x0_eff = spread_max * torch.sigmoid(x0_raw)

    if l_param.spread_structured:
        eps_val = min(max(float(l_param.quote_spread_k_epsilon), 1e-6), 0.49)
        logit_eps = torch.log(
            torch.tensor(eps_val / (1.0 - eps_val), dtype=params.dtype, device=params.device)
        )
        k_min = -logit_eps / x0_eff[0].clamp(min=1e-6)
        k_eff = torch.zeros(n, dtype=params.dtype, device=params.device)
        k_eff[0] = k_min + torch.nn.functional.softplus(k_raw[0]) + 1e-8
        if n > 1:
            k_eff[1:] = torch.nn.functional.softplus(k_raw[1:]) + 1e-8
    else:
        k_eff = torch.nn.functional.softplus(k_raw) + 1e-8

    q_total = l_param.quote_size_from_depth_logit_torch(u_depth)
    if l_param.spread_structured:
        q_spread = float(l_param.quote_spread_min_depth)
        delta_at_center = component_delta_at_torch(k_eff, x0_eff, float(x0_eff[0].item()))
        delta_at_max = component_delta_at_torch(k_eff, x0_eff, spread_max)
        if n > 1:
            tail_v = torch.softmax(L_raw[1:], dim=0)
            L_eff = _solve_spread_leg_L_torch(
                q_total=q_total,
                q_spread=q_spread,
                delta_center=delta_at_center,
                delta_max=delta_at_max,
                tail_v=tail_v,
            )
        else:
            L_eff = torch.zeros(n, dtype=params.dtype, device=params.device)
            L_eff[0] = q_total / delta_at_max[0].clamp(min=1e-12)
        return L_eff, k_eff, x0_eff

    w = torch.softmax(L_raw, dim=-1)
    delta = component_delta_spread_range_torch(k_eff, x0_eff, spread_max)
    z = (w * delta).sum().clamp(min=1e-12)
    L_eff = q_total * w / z
    return L_eff, k_eff, x0_eff


def positive_L_torch(
    L_raw: Any,
    k_raw: Any,
    x0_raw: Any,
    *,
    params: Any | None,
    side_off: int,
    l_param: LParamConfig | None,
    use_softplus: bool = True,
) -> Any:
    del k_raw, x0_raw, use_softplus
    if l_param is None:
        raise ValueError("l_param is required for structured softmax decode")
    if params is None:
        raise ValueError("full params tensor required for structured softmax decode")
    L_eff, _, _ = materialize_side_torch(
        params, side_off=side_off, n_components=L_raw.numel(), l_param=l_param
    )
    return L_eff
