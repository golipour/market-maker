"""Post-optimizer policy when expected risk-adjusted return is negative."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from mm.core.models.instrument import SymbolBinding
from mm.mpc.l_param import action_param_len, depth_logit_index, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction, SigmoidMixtureAction, State
from mm.mpc.mpc_tensor_batched import VenueOptimizerParams, _quote_capacity_bounds_scalar
from mm.mpc.optimizer_diagnostics import OptimizerRunDiagnostics
from mm.runtime.config import ControllerRuntimeConfig

NegativeObjectiveActionMode = Literal["optimizer_action", "floor_action", "zero_action"]

_LEGACY_NEGATIVE_OBJECTIVE_ACTION_ALIASES = {
    "off": "optimizer_action",
    "min": "floor_action",
    "max": "zero_action",
    "zero": "zero_action",
}

_MIN_DEPTH_LOGIT = -20.0


def normalize_negative_objective_action_mode(value: Any) -> NegativeObjectiveActionMode:
    mode = str(value or "optimizer_action").strip().lower()
    mode = _LEGACY_NEGATIVE_OBJECTIVE_ACTION_ALIASES.get(mode, mode)
    if mode not in ("optimizer_action", "floor_action", "zero_action"):
        raise ValueError(
            "controller.negative_objective_action must be 'optimizer_action', "
            "'floor_action', or 'zero_action'"
        )
    return mode  # type: ignore[return-value]


def best_risk_adjusted_return(optimizer_run: OptimizerRunDiagnostics | None) -> float | None:
    """Return the best closure risk-adjusted return, if optimizer diagnostics exist."""

    if optimizer_run is None or optimizer_run.best is None:
        return None
    components = optimizer_run.best.components
    if "risk_adjusted_return" in components:
        value = float(components["risk_adjusted_return"])
        return value if math.isfinite(value) else None
    value = float(optimizer_run.best.objective_value)
    return value if math.isfinite(value) else None


def zero_sigmoid_action(n_components: int) -> EffectiveSigmoidMixtureAction:
    n = max(1, int(n_components))
    z = tuple(0.0 for _ in range(n))
    k = tuple(1.0 for _ in range(n))
    return EffectiveSigmoidMixtureAction(
        bid_L=z,
        bid_k=k,
        bid_x0=z,
        ask_L=z,
        ask_k=k,
        ask_x0=z,
    )


def _venue_optimizer_params(
    config: ControllerRuntimeConfig,
    binding: SymbolBinding,
    *,
    venue: str,
) -> VenueOptimizerParams:
    optimizer = dict(config.optimizer or {})
    root_n = optimizer.get("n_components")
    defaults = VenueOptimizerParams(
        n_components=max(1, int(root_n)) if root_n is not None else 1,
    )
    if venue == "local":
        min_qty = float(binding.local.min_qty or 1.0)
    else:
        spec = binding.global_ or binding.local
        min_qty = float(spec.min_qty or 1.0)
    venue_defaults = VenueOptimizerParams.from_mapping(
        {CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX: min_qty},
        defaults=defaults,
    )
    venue_cfg = dict(optimizer.get(venue, {}) or {})
    return VenueOptimizerParams.from_mapping(venue_cfg, defaults=venue_defaults)


def _ensure_optimizer_flat(flat: np.ndarray, *, n_components: int) -> np.ndarray:
    need = action_param_len(n_components)
    arr = np.asarray(flat, dtype=np.float64)
    if arr.size >= need:
        return arr[:need].copy()
    out = np.zeros(need, dtype=np.float64)
    out[: arr.size] = arr
    return out


def _flat_at_min_depth(flat: np.ndarray, *, n_components: int) -> np.ndarray:
    arr = _ensure_optimizer_flat(flat, n_components=n_components)
    for side_off in (0, 3 * n_components):
        idx = depth_logit_index(side_off=side_off, n_components=n_components)
        arr[idx] = _MIN_DEPTH_LOGIT
    return arr


def _action_flat(action: SigmoidMixtureAction, optimizer_run: OptimizerRunDiagnostics | None) -> np.ndarray:
    if optimizer_run is not None and optimizer_run.action_best is not None:
        raw = optimizer_run.action_best.raw_params.get("flat")
        if isinstance(raw, list) and raw:
            return np.asarray(raw, dtype=np.float64)
    return np.asarray(action.to_flat(), dtype=np.float64)


def _min_sigmoid_action(
    *,
    action: SigmoidMixtureAction,
    optimizer_run: OptimizerRunDiagnostics | None,
    venue_params: VenueOptimizerParams,
    spread_max: float,
    cash: float,
    inventory: float,
    mid_price: float,
) -> EffectiveSigmoidMixtureAction:
    n_components = action.n_components
    venue_n = venue_params.n_components or n_components
    n = max(1, int(venue_n))
    l_param = venue_params.l_param_config(spread_max=spread_max, n_components=n)
    bid_min, bid_max, ask_min, ask_max = _quote_capacity_bounds_scalar(
        cash=cash,
        inventory=inventory,
        mid_price=mid_price,
        optimizer=venue_params,
    )
    flat = _flat_at_min_depth(_action_flat(action, optimizer_run), n_components=n)
    return sigmoid_action_from_flat_raw(
        flat,
        n,
        l_param=l_param,
        mid_price=mid_price,
        bid_quote_size_min=bid_min,
        bid_quote_size_max=bid_max,
        ask_quote_size_min=ask_min,
        ask_quote_size_max=ask_max,
    )


def apply_negative_objective_policy(
    *,
    mode: NegativeObjectiveActionMode,
    zero_threshold: float = 0.0,
    local_action: SigmoidMixtureAction,
    global_action: SigmoidMixtureAction | None,
    optimizer_run: OptimizerRunDiagnostics | None,
    config: ControllerRuntimeConfig,
    binding: SymbolBinding,
    state: State,
    global_active: bool,
) -> tuple[SigmoidMixtureAction, SigmoidMixtureAction | None, dict[str, Any]]:
    """Adjust MPC sigmoid actions when best risk-adjusted return is negative."""

    risk_adjusted_return = best_risk_adjusted_return(optimizer_run)
    threshold = max(0.0, float(zero_threshold))
    meta: dict[str, Any] = {
        "mode": mode,
        "zero_threshold": threshold,
        "risk_adjusted_return": risk_adjusted_return,
        "applied": False,
        "resolved_action": "optimizer_action",
    }
    if risk_adjusted_return is None:
        return local_action, global_action, meta
    if mode == "optimizer_action":
        return local_action, global_action, meta
    if mode == "zero_action":
        if risk_adjusted_return >= -threshold:
            return local_action, global_action, meta
    elif risk_adjusted_return >= 0.0:
        return local_action, global_action, meta

    spread_max = float(config.orderbook_grid.spread_max)
    local_params = _venue_optimizer_params(config, binding, venue="local")
    if mode == "zero_action":
        local_out = zero_sigmoid_action(local_action.n_components)
        global_out = None
        if global_active and global_action is not None:
            global_params = _venue_optimizer_params(config, binding, venue="global")
            global_n = global_params.n_components or global_action.n_components
            global_out = zero_sigmoid_action(global_n)
        meta.update(applied=True, resolved_action="zero_action")
        return local_out, global_out, meta

    local_out = _min_sigmoid_action(
        action=local_action,
        optimizer_run=optimizer_run,
        venue_params=local_params,
        spread_max=spread_max,
        cash=float(state.cash),
        inventory=float(state.inventory),
        mid_price=float(state.mid),
    )
    global_out = global_action
    if global_active and global_action is not None:
        global_params = _venue_optimizer_params(config, binding, venue="global")
        global_mid = float(state.global_mid if state.global_mid is not None else state.mid)
        global_out = _min_sigmoid_action(
            action=global_action,
            optimizer_run=None,
            venue_params=global_params,
            spread_max=spread_max,
            cash=float(state.global_cash),
            inventory=float(state.global_inventory),
            mid_price=global_mid,
        )
    meta.update(applied=True, resolved_action="floor_action")
    return local_out, global_out, meta
