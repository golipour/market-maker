"""Structured sigmoid-mixture parameter decoding for MPC quote actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mm.mpc.optimizer_config_keys import (
    CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    MAX_ABS_SKEW,
    RESERVATION_SKEW,
    reject_removed_reservation_skew_keys,
)

if TYPE_CHECKING:
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction


def _torch() -> Any:
    """Return the ``torch`` module, importing it lazily on first use.

    Torch is an optional dependency for parts of this package, so it is not
    imported at module scope. The result is memoized so the hot torch paths
    resolve to a plain module-level reference after the first call.

    This indirection exists instead of a bare ``__import__("torch")`` inside
    each function because a dynamic import cannot be traced by
    ``torch.compile`` (Dynamo raises "Attempted to call function marked as
    skipped"), which makes every function that decodes parameters
    un-compilable. A memoized accessor is Dynamo-friendly and also removes
    the repeated import-machinery lookup from the innermost decode path.
    """
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        import torch as _torch_mod

        _TORCH_MODULE = _torch_mod
    return _TORCH_MODULE


_TORCH_MODULE: Any | None = None

# Scalar bound tensors are rebuilt on every depth decode (80x per optimizer
# decide at n_samples=64, horizon=2) even though the configured quote-size
# bounds and their dtype/device are fixed for a run. Memoizing them turns five
# tensor constructions plus a clamp into a dict lookup: 6.6us -> 2.5us per call
# at (64,) float64, bit-identical. Keyed by (value, dtype, device, clamped) so a
# change of bound, precision or device still produces a fresh tensor.
_SCALAR_TENSOR_CACHE: dict[tuple[Any, ...], Any] = {}


def _scalar_tensor(value: Any, default: Any, like: Any, *, clamp_zero: bool = False) -> Any:
    """Return a cached 0-d tensor for ``value`` (or ``default`` if None).

    Mirrors the original inline logic: a tensor input is converted and used
    as-is (never cached, since it can carry gradients or vary per call), and a
    scalar/None input resolves to a constant tensor. ``clamp_zero`` reproduces
    the ``torch.clamp(..., min=0.0)`` that the scalar paths previously applied.

    The cache is only populated while *not* tracing. Dynamo guards on the
    globals a traced frame reads, so mutating ``_SCALAR_TENSOR_CACHE`` from
    inside a compiled region installs a ``len(G['_SCALAR_TENSOR_CACHE']) == N``
    guard; every new bound inserts a key and invalidates it, forcing a
    recompile. Production sweeps many distinct bound values, hit
    ``config.recompile_limit (8)`` on ``materialize_side_torch``, and were
    silently demoted to eager execution. Under tracing the tensors are rebuilt
    as compile-time constants anyway (inductor folds them), so skipping the
    cache there costs nothing and keeps the graph stable.
    """
    torch = _torch()
    if torch.is_tensor(value):
        out = value.to(dtype=like.dtype, device=like.device)
        return torch.clamp(out, min=0.0) if clamp_zero else out
    if value is None:
        value = default
    scalar = float(value)
    if torch.compiler.is_compiling():
        # Constant-folded by inductor; do not touch the traced global.
        out = torch.tensor(scalar, dtype=like.dtype, device=like.device)
        return torch.clamp(out, min=0.0) if clamp_zero else out
    key = (scalar, like.dtype, str(like.device), clamp_zero)
    cached = _SCALAR_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(scalar, dtype=like.dtype, device=like.device)
        if clamp_zero:
            cached = torch.clamp(cached, min=0.0)
        _SCALAR_TENSOR_CACHE[key] = cached
    return cached


@dataclass(frozen=True, slots=True)
class SpreadStructureParams:
    """Structured first-touch spread geometry for a venue optimizer block."""

    center_sum_max: float
    center_min_depth: float = 0.0
    center_k_epsilon: float = 0.01

    @classmethod
    def from_mapping(cls, payload: Any) -> SpreadStructureParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("book_quality.hard_structure must be an object")
        center_sum_max_raw = payload.get("action_max_spread_bps")
        if center_sum_max_raw is not None:
            center_sum_max = float(center_sum_max_raw)
        else:
            center_sum_max = float(
                payload.get("action_max_spread", payload.get("center_sum_max", 0.0))
            )
        if center_sum_max <= 0.0:
            raise ValueError("book_quality.hard_structure.action_max_spread must be > 0")
        return cls(
            center_sum_max=center_sum_max,
            center_min_depth=max(
                0.0,
                float(payload.get("action_min_depth", payload.get("center_min_depth", 0.0))),
            ),
            center_k_epsilon=float(
                payload.get("action_k_epsilon", payload.get("center_k_epsilon", 0.01))
            ),
        )

    @classmethod
    def from_book_quality_hard_structure(cls, hard: Any) -> SpreadStructureParams | None:
        if hard is None:
            return None
        payload: dict[str, Any] = {
            "action_max_spread": hard.action_max_spread,
            "action_min_depth": hard.action_min_depth,
            "action_k_epsilon": hard.action_k_epsilon,
        }
        if getattr(hard, "action_max_spread_bps", None) is not None:
            payload["action_max_spread_bps"] = hard.action_max_spread_bps
        return cls.from_mapping(payload)

    def as_mapping(self) -> dict[str, float]:
        out: dict[str, float] = {"center_sum_max": self.center_sum_max}
        if self.center_min_depth > 0.0:
            out["center_min_depth"] = self.center_min_depth
        if self.center_k_epsilon != 0.01:
            out["center_k_epsilon"] = self.center_k_epsilon
        return out

    def to_l_param_fields(self) -> dict[str, Any]:
        return {
            "quote_spread_max": self.center_sum_max,
            "quote_spread_min_depth": (
                self.center_min_depth if self.center_min_depth > 0.0 else None
            ),
            "quote_spread_k_epsilon": self.center_k_epsilon,
        }


def action_param_len(n_components: int) -> int:
    """Flat optimizer vector length: mixture params, depth logits, reservation offset."""
    return 6 * max(1, int(n_components)) + 3


def reservation_logit_index(n_components: int) -> int:
    """Raw logit index for dimensionless reservation offset ``u_r``."""

    return action_param_len(n_components) - 1


@dataclass(frozen=True, slots=True)
class ReservationSkewParams:
    """Bounded learnable reservation skew on the spread grid."""

    max_abs_skew: float

    @classmethod
    def from_mapping(cls, payload: Any) -> ReservationSkewParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError(f"{RESERVATION_SKEW} must be an object")
        reject_removed_reservation_skew_keys(payload)
        if "max_abs_skew_bps" in payload:
            max_abs_skew = max(0.0, float(payload["max_abs_skew_bps"]) / 10000.0)
        else:
            max_abs_skew = max(0.0, float(payload.get(MAX_ABS_SKEW, 0.0)))
        if max_abs_skew <= 0.0:
            return None
        return cls(max_abs_skew=max_abs_skew)

    def as_mapping(self) -> dict[str, float]:
        return {
            MAX_ABS_SKEW: self.max_abs_skew,
            "max_abs_skew_bps": self.max_abs_skew * 10000.0,
        }


def joint_action_param_len(n_components: int, global_n_components: int | None = None) -> int:
    """Flat optimizer vector length for local + global sigmoid curves."""

    local_n = max(1, int(n_components))
    global_n = local_n if global_n_components is None else max(1, int(global_n_components))
    return action_param_len(local_n) + action_param_len(global_n)


def depth_logit_index(*, side_off: int, n_components: int) -> int:
    base = 6 * n_components
    return base if side_off == 0 else base + 1


@dataclass(frozen=True, slots=True)
class LParamConfig:
    """
    Raw optimizer params decode into effective ``(L, k, x0)`` with structural constraints.

    Without spread geometry: global softmax on ``L`` with ``δ(spread_max)`` normalization.

    With spread geometry (``book_quality.hard_structure.action_max_spread`` mapped to ``quote_spread_max``):
    - first centers satisfy ``x0_1_bid + x0_1_ask <= center_sum_max`` via a scaled softmax split
    - component 1 may carry at least ``center_min_depth`` at its center when configured
    - tail components fill remaining depth up to a chosen total in
      ``[quote_size_min, quote_size_max]`` at ``spread_max``

    Total per-side depth at ``spread_max`` is selected by a depth logit:
    ``quote_size_min + sigmoid(u_depth) * (quote_size_max - quote_size_min)``.
    Runtime callers may pass dynamic per-side min/max bounds; when
    ``quote_size_max`` is not configured, the dynamic max is the structural cap.
    """

    quote_size_min: float
    quote_size_max: float | None
    spread_max: float
    n_components: int = 1
    quote_spread_max: float | None = None
    quote_spread_min_depth: float | None = None
    quote_spread_k_epsilon: float = 0.01
    reservation_r_max: float = 0.0

    @property
    def spread_structured(self) -> bool:
        return self.quote_spread_max is not None and float(self.quote_spread_max) > 0.0

    def quote_size_midpoint(self) -> float:
        hi = float(self.quote_size_max) if self.quote_size_max is not None else float(self.quote_size_min)
        return 0.5 * (float(self.quote_size_min) + hi)

    def quote_size_from_depth_logit(
        self,
        u_depth: float,
        *,
        quote_size_min: float | None = None,
        quote_size_max: float | None = None,
    ) -> float:
        lo = max(0.0, float(self.quote_size_min if quote_size_min is None else quote_size_min))
        configured_hi = (
            float(self.quote_size_max)
            if self.quote_size_max is not None
            else float("inf")
        )
        if quote_size_max is None and self.quote_size_max is None:
            hi_raw = lo
        else:
            hi_raw = configured_hi if quote_size_max is None else max(0.0, float(quote_size_max))
        hi = max(lo, min(configured_hi, hi_raw))
        return lo + _sigmoid_np(u_depth) * (hi - lo)

    def quote_size_from_flat(
        self,
        flat: np.ndarray,
        *,
        side_off: int,
        n_components: int,
        quote_size_min: float | None = None,
        quote_size_max: float | None = None,
    ) -> float:
        idx = depth_logit_index(side_off=side_off, n_components=n_components)
        return self.quote_size_from_depth_logit(
            float(flat[idx]),
            quote_size_min=quote_size_min,
            quote_size_max=quote_size_max,
        )

    def quote_size_from_depth_logit_torch(
        self,
        u_depth: Any,
        *,
        quote_size_min: Any | None = None,
        quote_size_max: Any | None = None,
    ) -> Any:
        """Differentiable quote size for a depth logit.

        ``self.quote_size_max is None`` used to be a Python-level branch here,
        and this function is traced by dynamo inside ``soft_fill_batched``.
        Each distinct answer forced another specialization of
        ``materialize_side_torch``; production runs the local (4 components)
        and global (2 components) venues plus a bitwana override shape, and
        that exhausted ``torch._dynamo``'s default ``cache_size_limit`` of 8.
        Once the limit is hit dynamo falls back to eager *permanently* for the
        function, so the compiled kernel silently stopped being used.

        The bounds are now resolved *outside* the traced arithmetic by
        :meth:`_quote_size_bounds`, which returns plain floats. The traced
        body then has a single shape regardless of configuration:

        * capped (configured ``quote_size_max``): ``hi`` is that bound;
        * uncapped: ``+inf``, so the minimum clamp is a no-op and the sigmoid
          spans the full range, exactly as before.

        Per-sample tensor bounds stay on their own path because they are data
        rather than configuration, and production always passes tensors.

        No behavior change; only the number of traced specializations.
        """
        torch = _torch()
        lo = _scalar_tensor(quote_size_min, self.quote_size_min, u_depth, clamp_zero=True)
        configured_hi_val, hi_val = self._quote_size_bounds(quote_size_max)
        if torch.is_tensor(quote_size_max):
            # A per-sample bound is data, not configuration, so it keeps its
            # own traced path.
            hi = torch.clamp(quote_size_max.to(dtype=u_depth.dtype, device=u_depth.device), min=0.0)
        elif hi_val is None:
            # Degenerate case: neither a configured cap nor an override, so
            # the original used ``hi = lo`` and the size collapses to ``lo``.
            hi = lo
        else:
            hi = _scalar_tensor(hi_val, None, u_depth)
        hi = torch.maximum(lo, torch.minimum(_scalar_tensor(configured_hi_val, None, u_depth), hi))
        return lo + torch.sigmoid(u_depth) * (hi - lo)

    def _quote_size_bounds(self, quote_size_max: Any | None) -> tuple[float, float | None]:
        """Resolve quote-size bounds to plain floats, outside the traced body.

        Returns ``(configured_hi, hi_val)`` mirroring the original branches:

        * configured cap, no override -> ``(cap, cap)``
        * no cap, no override -> ``(inf, None)``; the ``None`` here means
          "degenerate, use ``lo``", and is resolved by the caller to a
          *tensor* so the traced body never branches on it
        * explicit non-tensor override ``v`` -> ``(min(cap, v), v)``,
          because the caller clamps by ``configured_hi_val`` afterwards
        * tensor override -> handled on its own traced path; only
          ``configured_hi_val`` is reported

        Returning floats unconditionally keeps the traced region free of the
        ``quote_size_max is None`` branch that exhausted dynamo's cache.
        """
        configured = self.quote_size_max
        configured_hi_val = (
            float(configured) if configured is not None else float("inf")
        )
        if quote_size_max is None:
            if configured is None:
                return (configured_hi_val, None)
            return (configured_hi_val, float(configured))
        if _torch().is_tensor(quote_size_max):
            return (configured_hi_val, 0.0)
        value = max(0.0, float(quote_size_max))
        return (configured_hi_val, value)


def validate_l_param_config(l_param: LParamConfig) -> None:
    lo = float(l_param.quote_size_min)
    hi = float(l_param.quote_size_max) if l_param.quote_size_max is not None else None
    if lo < 0.0:
        raise ValueError(f"{CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX} must be >= 0")
    if hi is not None and hi <= 0.0:
        raise ValueError(f"{CUMULATIVE_QTY_CAP_AT_SPREAD_MAX} must be positive when configured")
    if hi is not None and lo > hi:
        raise ValueError(
            f"{CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX} ({lo}) must be <= {CUMULATIVE_QTY_CAP_AT_SPREAD_MAX} ({hi})"
        )
    if l_param.spread_structured:
        s_max = float(l_param.quote_spread_max or 0.0)
        q_spread = float(l_param.quote_spread_min_depth or 0.0)
        if s_max > float(l_param.spread_max):
            raise ValueError(
                f"book_quality.hard_structure.action_max_spread ({s_max}) must be <= spread_max ({l_param.spread_max})"
            )
        if hi is not None and q_spread > hi:
            raise ValueError(
                f"book_quality.hard_structure.action_min_depth ({q_spread}) must be <= {CUMULATIVE_QTY_CAP_AT_SPREAD_MAX} ({hi})"
            )
        if int(l_param.n_components) < 2:
            raise ValueError(
                "spread-structured softmax requires n_components >= 2 "
                "(spread leg + tail leg)"
            )
    if float(l_param.reservation_r_max) > 0.0 and float(l_param.reservation_r_max) >= 1.0:
        raise ValueError(f"{RESERVATION_SKEW}.{MAX_ABS_SKEW} must be < 1.0 for relative mid shift")


def reservation_r_offset_from_flat_np(
    flat: np.ndarray,
    *,
    n_components: int,
    r_max: float,
) -> float:
    if float(r_max) <= 0.0:
        return 0.0
    u_r = float(flat[reservation_logit_index(n_components)])
    return float(r_max) * float(np.tanh(u_r))


def reservation_r_offset_from_params(
    params: Any,
    *,
    n_components: int,
    r_max: float,
) -> Any:
    torch = _torch()
    if float(r_max) <= 0.0:
        return torch.zeros((), dtype=params.dtype, device=params.device)
    idx = reservation_logit_index(n_components)
    u_r = params[idx]
    return float(r_max) * torch.tanh(u_r)


def curve_spread_for_market_spread_np(
    market_spread: float,
    *,
    side: str,
    mid_price: float,
    reservation_r_offset: float,
    spread_max: float | None = None,
) -> float:
    """Numpy map from market spread to sigmoid curve argument."""

    if mid_price <= 0.0 or abs(float(reservation_r_offset)) <= 0.0:
        return float(market_spread)
    shift = float(mid_price) * float(reservation_r_offset)
    if spread_max is not None and float(spread_max) > 0.0:
        max_shift = 0.8 * float(spread_max)
        shift = max(-max_shift, min(max_shift, shift))
    if side == "bid":
        return max(0.0, float(market_spread) + shift)
    return max(0.0, float(market_spread) - shift)


def curve_spread_for_market_spread(
    market_spread: Any,
    *,
    side: str,
    mid_price: Any | None,
    reservation_r_offset: Any | None,
    spread_max: Any | None = None,
) -> Any:
    """Map market-grid spread to sigmoid argument with relative reservation shift."""
    if mid_price is None or reservation_r_offset is None:
        return market_spread
    torch = _torch()
    r_off = reservation_r_offset
    if not torch.is_tensor(r_off):
        r_off = torch.as_tensor(r_off, dtype=market_spread.dtype, device=market_spread.device)
    mid = mid_price
    if not torch.is_tensor(mid):
        mid = torch.as_tensor(mid, dtype=market_spread.dtype, device=market_spread.device)
    shift = mid * r_off
    if spread_max is not None:
        s_max = spread_max
        if not torch.is_tensor(s_max):
            s_max = torch.as_tensor(s_max, dtype=market_spread.dtype, device=market_spread.device)
        max_shift = 0.8 * s_max
        shift = shift.clamp(-max_shift, max_shift)
    if side == "bid":
        return (market_spread + shift).clamp(min=0.0)
    return (market_spread - shift).clamp(min=0.0)


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
    total_at_max = float(np.dot(L_eff, delta_max))
    if total_at_max > q_total > 0.0:
        L_eff *= q_total / max(total_at_max, 1e-12)
    return L_eff


def materialize_side_np(
    flat: np.ndarray,
    *,
    side_off: int,
    n_components: int,
    l_param: LParamConfig,
    quote_size_min: float | None = None,
    quote_size_max: float | None = None,
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

    q_total = l_param.quote_size_from_flat(
        flat,
        side_off=side_off,
        n_components=n,
        quote_size_min=quote_size_min,
        quote_size_max=quote_size_max,
    )
    if l_param.spread_structured:
        q_spread = float(l_param.quote_spread_min_depth or 0.0)
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
    bid_quote_size_max: float | None = None,
    ask_quote_size_max: float | None = None,
    bid_quote_size_min: float | None = None,
    ask_quote_size_min: float | None = None,
    mid_price: float | None = None,
    use_softplus: bool = True,
) -> EffectiveSigmoidMixtureAction:
    del use_softplus
    from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction

    arr = np.asarray(flat, dtype=np.float64)
    n = n_components
    r_offset = reservation_r_offset_from_flat_np(
        arr,
        n_components=n,
        r_max=float(l_param.reservation_r_max),
    )
    r_idx = reservation_logit_index(n)
    raw_logit = float(arr[r_idx]) if len(arr) > r_idx else 0.0
    bid_L, bid_k, bid_x0 = materialize_side_np(
        arr,
        side_off=0,
        n_components=n,
        l_param=l_param,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
    )
    ask_L, ask_k, ask_x0 = materialize_side_np(
        arr,
        side_off=3 * n,
        n_components=n,
        l_param=l_param,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
    )
    return EffectiveSigmoidMixtureAction(
        bid_L=tuple(float(x) for x in bid_L),
        bid_k=tuple(float(x) for x in bid_k),
        bid_x0=tuple(float(x) for x in bid_x0),
        ask_L=tuple(float(x) for x in ask_L),
        ask_k=tuple(float(x) for x in ask_k),
        ask_x0=tuple(float(x) for x in ask_x0),
        reservation_r_offset=r_offset,
        reservation_mid=float(mid_price or 0.0),
        raw_reservation_logit=raw_logit,
        spread_max=float(l_param.spread_max) if l_param is not None else 0.0,
    )


def component_delta_at_torch(
    k_eff: Any,
    x0: Any,
    spread: Any,
) -> Any:
    torch = _torch()
    s = spread if torch.is_tensor(spread) else float(spread)
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
    torch = _torch()
    n = int(delta_center.numel())
    d1_c = delta_center[0].clamp(min=1e-12)
    d1_m = delta_max[0]
    q_total_t = q_total
    q_is_vector = torch.is_tensor(q_total_t) and q_total_t.ndim > 0
    if not q_is_vector and n == 1:
        return (q_total / delta_max[0].clamp(min=1e-12)).reshape(1)
    if q_is_vector and n == 1:
        return q_total_t.reshape(-1, 1) / delta_max[0].clamp(min=1e-12)
    z_tm = (tail_v * delta_max[1:]).sum().clamp(min=1e-12)
    tail_c = (tail_v * delta_center[1:]).sum()
    q_spread_t = _scalar_tensor(q_spread, q_spread, d1_c, clamp_zero=False)
    denom = d1_c - d1_m * tail_c / z_tm
    if q_is_vector:
        q_flat = q_total_t.reshape(-1)
        numer = q_spread_t - q_flat * tail_c / z_tm
        L1 = torch.where(
            denom.abs() < 1e-12,
            q_spread_t / d1_c,
            numer / denom,
        )
        L1 = torch.maximum(L1, q_spread_t / d1_c)
        q_tail = (q_flat - L1 * d1_m).clamp(min=0.0)
        tail = q_tail.unsqueeze(1) * tail_v.unsqueeze(0) / z_tm
        L_eff = torch.cat([L1.reshape(-1, 1), tail], dim=1)
        total_at_max = (L_eff * delta_max.unsqueeze(0)).sum(dim=1).clamp(min=1e-12)
        scale = torch.minimum(torch.ones_like(total_at_max), q_flat / total_at_max)
        L_eff = L_eff * scale.unsqueeze(1)
        return L_eff
    numer = q_spread_t - q_total * tail_c / z_tm
    L1 = torch.where(denom.abs() < 1e-12, q_spread_t / d1_c, numer / denom)
    L1 = torch.maximum(L1, q_spread_t / d1_c)
    q_tail = (q_total - L1 * d1_m).clamp(min=0.0)
    L_eff = torch.cat([L1.reshape(1), q_tail * tail_v / z_tm])
    total_at_max = (L_eff * delta_max).sum().clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(total_at_max), q_total / total_at_max)
    L_eff = L_eff * scale
    return L_eff


def materialize_side_torch(
    params: Any,
    *,
    side_off: int,
    n_components: int,
    l_param: LParamConfig,
    quote_size_min: Any | None = None,
    quote_size_max: Any | None = None,
) -> tuple[Any, Any, Any]:
    torch = _torch()
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
        min_center = _scalar_tensor(
            _min_structured_center(spread_cap), None, params, clamp_zero=False
        )
        centers = torch.clamp(scale * spread_cap * spread_w, min=min_center)
        x0_first = centers[0] if side_off == 0 else centers[1]
        # Tail centers are ``x0_first + (spread_max - x0_first) * sigmoid(x0_raw)``.
        # Built vectorized and concatenated with the head, instead of filling a
        # zeros tensor element by element in a Python loop: the same values with
        # one op instead of n slice-assignments, and a cheaper backward.
        if n > 1:
            tail_centers = x0_first + (spread_max - x0_first) * torch.sigmoid(x0_raw[1:])
            x0_eff = torch.cat([x0_first.reshape(1), tail_centers])
        else:
            x0_eff = x0_first.reshape(1)
    else:
        x0_eff = spread_max * torch.sigmoid(x0_raw)

    if l_param.spread_structured:
        eps_val = min(max(float(l_param.quote_spread_k_epsilon), 1e-6), 0.49)
        logit_eps = torch.log(
            _scalar_tensor(
                eps_val / (1.0 - eps_val), None, params, clamp_zero=False
            )
        )
        k_min = -logit_eps / x0_eff[0].clamp(min=1e-6)
        k_head = k_min + torch.nn.functional.softplus(k_raw[0]) + 1e-8
        if n > 1:
            k_eff = torch.cat(
                [k_head.reshape(1), torch.nn.functional.softplus(k_raw[1:]) + 1e-8]
            )
        else:
            k_eff = k_head.reshape(1)
    else:
        k_eff = torch.nn.functional.softplus(k_raw) + 1e-8

    q_total = l_param.quote_size_from_depth_logit_torch(
        u_depth,
        quote_size_min=quote_size_min,
        quote_size_max=quote_size_max,
    )
    if l_param.spread_structured:
        q_spread = float(l_param.quote_spread_min_depth or 0.0)
        delta_at_center = component_delta_at_torch(k_eff, x0_eff, x0_eff[0])
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
            L_eff = (q_total / delta_at_max[0].clamp(min=1e-12)).reshape(n)
        return L_eff, k_eff, x0_eff

    w = torch.softmax(L_raw, dim=-1)
    delta = component_delta_spread_range_torch(k_eff, x0_eff, spread_max)
    z = (w * delta).sum().clamp(min=1e-12)
    if torch.is_tensor(q_total) and q_total.ndim > 0:
        L_eff = q_total.reshape(-1, 1) * w.unsqueeze(0) / z
    else:
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
