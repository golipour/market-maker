"""Warm-start initialization for MPC Adam optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_WARM_START_ALPHA_MODES = frozenset({"fixed", "adaptive"})


@dataclass(frozen=True, slots=True)
class WarmStartParams:
    """
    Warm-start controls for MPC action initialization and Adam state carry-over.

    ``enabled=false`` or ``alpha=0`` uses the default initializer only (cold start).
    ``enabled=true`` and ``alpha=1`` matches legacy binary warm start.
    ``preserve_optimizer_state`` keeps Adam moments across decisions when true.
    """

    enabled: bool = False
    preserve_optimizer_state: bool = True
    alpha: float = 1.0
    alpha_mode: str = "fixed"
    alpha_max: float = 0.95
    grad_norm_eps: float = 1e-12

    @classmethod
    def from_mapping(cls, payload: Any) -> WarmStartParams:
        if payload is None:
            return cls()
        if isinstance(payload, bool):
            enabled = bool(payload)
            return cls(
                enabled=enabled,
                preserve_optimizer_state=enabled,
                alpha=1.0 if enabled else 0.0,
            )
        if not isinstance(payload, dict):
            raise ValueError("warm_start must be a bool or object")
        enabled = bool(payload.get("enabled", True))
        alpha_mode = str(payload.get("alpha_mode", "fixed")).strip().lower()
        if alpha_mode not in _WARM_START_ALPHA_MODES:
            raise ValueError(
                f"warm_start.alpha_mode must be one of {sorted(_WARM_START_ALPHA_MODES)}"
            )
        return cls(
            enabled=enabled,
            preserve_optimizer_state=bool(
                payload.get("preserve_optimizer_state", enabled)
            ),
            alpha=float(payload.get("alpha", 1.0)),
            alpha_mode=alpha_mode,
            alpha_max=float(payload.get("alpha_max", 0.95)),
            grad_norm_eps=float(payload.get("grad_norm_eps", 1e-12)),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "preserve_optimizer_state": self.preserve_optimizer_state,
            "alpha": self.alpha,
            "alpha_mode": self.alpha_mode,
            "alpha_max": self.alpha_max,
            "grad_norm_eps": self.grad_norm_eps,
        }

    def resolved_alpha(
        self,
        *,
        grad_norm_warm: float | None = None,
        grad_norm_default: float | None = None,
    ) -> float:
        if not self.enabled:
            return 0.0
        if self.alpha_mode == "adaptive":
            if grad_norm_warm is None or grad_norm_default is None:
                return _clamp_unit(self.alpha)
            gw = max(0.0, float(grad_norm_warm))
            gd = max(0.0, float(grad_norm_default))
            eps = max(1e-300, float(self.grad_norm_eps))
            return min(float(self.alpha_max), gw / (gw + gd + eps))
        return _clamp_unit(self.alpha)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def blend_action_flat(
    u_warm: np.ndarray,
    u_default: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Convex blend ``alpha * u_warm + (1 - alpha) * u_default`` in raw action space."""

    a = _clamp_unit(alpha)
    warm = np.asarray(u_warm, dtype=np.float64)
    default = np.asarray(u_default, dtype=np.float64)
    if warm.shape != default.shape:
        raise ValueError("warm_start blend requires matching action shapes")
    return (a * warm + (1.0 - a) * default).astype(np.float64, copy=False)


def resolve_warm_start_one_action(
    *,
    params: WarmStartParams,
    u_default: np.ndarray,
    u_warm: np.ndarray | None,
    grad_norm_warm: float | None = None,
    grad_norm_default: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pick per-horizon-step flat action before tiling across the horizon."""

    meta: dict[str, float] = {
        "warm_start_enabled": 1.0 if params.enabled else 0.0,
        "warm_start_preserve_optimizer_state": (
            1.0 if params.enabled and params.preserve_optimizer_state else 0.0
        ),
    }
    if not params.enabled or u_warm is None:
        meta["warm_start_alpha"] = 0.0
        return np.asarray(u_default, dtype=np.float64).copy(), meta

    alpha = params.resolved_alpha(
        grad_norm_warm=grad_norm_warm,
        grad_norm_default=grad_norm_default,
    )
    meta["warm_start_alpha"] = alpha
    if grad_norm_warm is not None:
        meta["warm_start_grad_norm_warm"] = float(grad_norm_warm)
    if grad_norm_default is not None:
        meta["warm_start_grad_norm_default"] = float(grad_norm_default)
    if alpha <= 0.0:
        return np.asarray(u_default, dtype=np.float64).copy(), meta
    if alpha >= 1.0:
        return np.asarray(u_warm, dtype=np.float64).copy(), meta
    return blend_action_flat(u_warm, u_default, alpha), meta


def prepare_warm_start_initial_flat(
    *,
    params: WarmStartParams,
    u_default: np.ndarray,
    u_warm: np.ndarray | None,
    horizon: int,
    grad_norm_probe: Callable[[np.ndarray], float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Tile blended per-step action across the MPC horizon."""

    grad_warm: float | None = None
    grad_default: float | None = None
    if (
        params.enabled
        and u_warm is not None
        and params.alpha_mode == "adaptive"
        and grad_norm_probe is not None
    ):
        grad_warm = float(grad_norm_probe(u_warm))
        grad_default = float(grad_norm_probe(u_default))
        if not math.isfinite(grad_warm):
            grad_warm = 0.0
        if not math.isfinite(grad_default):
            grad_default = 0.0

    one_action, meta = resolve_warm_start_one_action(
        params=params,
        u_default=u_default,
        u_warm=u_warm,
        grad_norm_warm=grad_warm,
        grad_norm_default=grad_default,
    )
    h = max(1, int(horizon))
    return np.tile(one_action, h), meta


def warm_action_from_mpc_state(
    *,
    params: WarmStartParams,
    last_raw_flat: np.ndarray | None,
    last_action_flat: np.ndarray | None,
    expected_len: int,
    finite_flat_or_none: Callable[[Any, int], np.ndarray | None],
) -> np.ndarray | None:
    if not params.enabled:
        return None
    if last_raw_flat is not None:
        warm = finite_flat_or_none(last_raw_flat, expected_len)
        if warm is not None:
            return warm
    if last_action_flat is not None:
        return finite_flat_or_none(last_action_flat, expected_len)
    return None
