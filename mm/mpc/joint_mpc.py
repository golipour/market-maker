"""Joint-venue MPC helpers: four-curve actions, dual-venue rollout, consolidated wealth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mm.mpc.l_param import action_param_len, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import OrderbookRelative, SigmoidMixtureAction, State
from mm.mpc.reward_tensor import objective_entropic_return, objective_threshold_shortfall_penalty


def joint_action_param_len(n_components: int, global_n_components: int | None = None) -> int:
    """Flat optimizer vector length for local + global sigmoid curves per horizon step."""

    local_n = max(1, int(n_components))
    global_n = local_n if global_n_components is None else max(1, int(global_n_components))
    return action_param_len(local_n) + action_param_len(global_n)


def split_venue_action_flat(
    flat: np.ndarray | list[float],
    n_components: int,
    global_n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    local_n = max(1, int(n_components))
    global_n = local_n if global_n_components is None else max(1, int(global_n_components))
    local_len = action_param_len(local_n)
    global_len = action_param_len(global_n)
    arr = np.asarray(flat, dtype=np.float64)
    required = local_len + global_len
    if arr.size < required:
        raise ValueError(f"joint action requires {required} params, got {arr.size}")
    return arr[:local_len], arr[local_len:required]


@dataclass(frozen=True, slots=True)
class JointSigmoidMixtureAction:
    """Local and optional global sigmoid quote curves from joint MPC."""

    local: SigmoidMixtureAction
    global_: SigmoidMixtureAction | None = None
    global_active: bool = False

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray | list[float],
        *,
        n_components: int,
        global_n_components: int | None = None,
        l_param: Any,
        global_l_param: Any | None = None,
        global_active: bool,
    ) -> JointSigmoidMixtureAction:
        global_n = n_components if global_n_components is None else global_n_components
        local_flat, global_flat = split_venue_action_flat(flat, n_components, global_n)
        local = sigmoid_action_from_flat_raw(local_flat, n_components, l_param=l_param)
        global_action = sigmoid_action_from_flat_raw(
            global_flat,
            global_n,
            l_param=global_l_param or l_param,
        )
        return cls(local=local, global_=global_action, global_active=global_active)


def reference_price_from_state(state: State, reference_price_source: str | None = None) -> float:
    """Return the selected sell-side VWAP reference used to mark inventory."""

    source = str(reference_price_source or "").strip().lower()
    if source == "local":
        return float(state.sell_vwap)
    if source == "global":
        if state.global_sell_vwap is not None and float(state.global_sell_vwap) > 0.0:
            return float(state.global_sell_vwap)
        return float(state.sell_vwap)
    if state.reference_price > 0.0:
        return float(state.reference_price)
    if state.global_sell_vwap is not None and float(state.global_sell_vwap) > 0.0:
        return float(state.global_sell_vwap)
    return float(state.sell_vwap)


def consolidated_wealth_scalar(
    *,
    local_cash: float,
    local_inventory: float,
    global_cash: float,
    global_inventory: float,
    reference_price: float,
) -> float:
    """W = local_quote + global_quote + (local_base + global_base) * reference_price."""

    return float(local_cash) + float(global_cash) + (
        float(local_inventory) + float(global_inventory)
    ) * float(reference_price)


def consolidated_wealth_batched(
    local_cash: Any,
    local_inventory: Any,
    global_cash: Any,
    global_inventory: Any,
    reference_price: Any,
) -> Any:
    """Batched consolidated wealth tensor."""

    return local_cash + global_cash + (local_inventory + global_inventory) * reference_price


def delta_consolidated_wealth_batched(
    *,
    local_cash: Any,
    local_inventory: Any,
    global_cash: Any,
    global_inventory: Any,
    reference_price: Any,
    local_cash_next: Any,
    local_inventory_next: Any,
    global_cash_next: Any,
    global_inventory_next: Any,
    reference_price_next: Any,
) -> Any:
    wealth_t = consolidated_wealth_batched(
        local_cash, local_inventory, global_cash, global_inventory, reference_price
    )
    wealth_next = consolidated_wealth_batched(
        local_cash_next,
        local_inventory_next,
        global_cash_next,
        global_inventory_next,
        reference_price_next,
    )
    return (wealth_next - wealth_t).squeeze(1)


def joint_terminal_objective(
    local_returns: Any,
    global_returns: Any,
    local_cash_path: Any,
    local_inventory_path: Any,
    global_cash_path: Any,
    global_inventory_path: Any,
    *,
    local_return_risk_aversion_beta: float,
    global_return_risk_aversion_beta: float,
    local_objective_weight: float,
    global_objective_weight: float,
    local_inventory_threshold_penalty_weight: float,
    global_inventory_threshold_penalty_weight: float,
    local_cash_threshold: float,
    local_inventory_threshold: float,
    global_cash_threshold: float,
    global_inventory_threshold: float,
    global_active: float,
) -> tuple[Any, dict[str, Any]]:
    """Terminal objective with venue-scoped return and inventory terms."""

    import torch

    local_returns = torch.nan_to_num(local_returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    global_returns = torch.nan_to_num(global_returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_local_return = float(local_objective_weight) * objective_entropic_return(
        local_returns,
        beta=local_return_risk_aversion_beta,
    )
    term_global_return = (
        float(global_active)
        * float(global_objective_weight)
        * objective_entropic_return(global_returns, beta=global_return_risk_aversion_beta)
    )
    local_pen = objective_threshold_shortfall_penalty(
        local_cash_path,
        local_inventory_path,
        cash_threshold=local_cash_threshold,
        inventory_threshold=local_inventory_threshold,
    )
    global_pen = objective_threshold_shortfall_penalty(
        global_cash_path,
        global_inventory_path,
        cash_threshold=global_cash_threshold,
        inventory_threshold=global_inventory_threshold,
    )
    term_local_threshold = -float(local_inventory_threshold_penalty_weight) * local_pen
    term_global_threshold = (
        -float(global_active) * float(global_inventory_threshold_penalty_weight) * global_pen
    )
    term_threshold = term_local_threshold + term_global_threshold
    components = {
        "risk_adjusted_return": term_local_return + term_global_return,
        "local_risk_adjusted_return": term_local_return,
        "global_risk_adjusted_return": term_global_return,
        "threshold_shortfall_penalty": term_threshold,
        "local_threshold_shortfall_penalty": term_local_threshold,
        "global_threshold_shortfall_penalty": term_global_threshold,
    }
    return term_local_return + term_global_return + term_threshold, components


def ob_rel_from_market_event(
    orderbook: Any,
    *,
    spread_max: float,
) -> OrderbookRelative:
    from mm.core.order_constraints import mid_from_orderbook

    mid = mid_from_orderbook(orderbook)
    bids = tuple(
        (spread, level.qty)
        for level in orderbook.bids
        if (spread := mid - level.price) >= 0 and spread <= spread_max and level.qty > 0
    )
    asks = tuple(
        (spread, level.qty)
        for level in orderbook.asks
        if (spread := level.price - mid) >= 0 and spread <= spread_max and level.qty > 0
    )
    return OrderbookRelative(bids=bids, asks=asks)


def enrich_state_for_joint(
    state: State,
    *,
    global_active: bool,
    reference_price: float | None = None,
) -> State:
    """Attach joint-venue fields on ``State`` for the optimizer rollout."""

    ref = float(reference_price if reference_price is not None else reference_price_from_state(state))
    return State(
        mid=state.mid,
        ob_rel=state.ob_rel,
        buy_vol=state.buy_vol,
        sell_vol=state.sell_vol,
        inventory=state.inventory,
        cash=state.cash,
        buy_vwap=state.buy_vwap,
        sell_vwap=state.sell_vwap,
        vwap_depth=state.vwap_depth,
        global_buy_vwap=state.global_buy_vwap,
        global_sell_vwap=state.global_sell_vwap,
        global_mid=state.global_mid,
        global_inventory=state.global_inventory,
        global_cash=state.global_cash,
        global_ob_rel=state.global_ob_rel,
        global_buy_vol=state.global_buy_vol,
        global_sell_vol=state.global_sell_vol,
        reference_price=ref,
        global_active=bool(global_active),
    )


def initial_joint_action_flat(
    *,
    n_components: int,
    spread_max: float,
    l_param: Any,
    initializer: Any,
) -> np.ndarray:
    """Build initial flat joint action as local init concatenated with global init."""

    one = initializer(n_components=n_components, spread_max=spread_max, l_param=l_param)
    return np.concatenate([one, one], dtype=np.float64)


def wealth_0_from_state(state: State, reference_price_source: str | None = None) -> float:
    ref = reference_price_from_state(state, reference_price_source)
    wealth = consolidated_wealth_scalar(
        local_cash=state.cash,
        local_inventory=state.inventory,
        global_cash=state.global_cash,
        global_inventory=state.global_inventory,
        reference_price=ref,
    )
    return max(1e-8, float(wealth))


def _finite_item(t) -> float:
    try:
        x = float(t.detach().item())
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0
