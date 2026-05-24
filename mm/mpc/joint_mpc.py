"""Joint-venue MPC helpers: four-curve actions, dual-venue rollout, consolidated wealth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mm.mpc.l_param import action_param_len, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import OrderbookRelative, SigmoidMixtureAction, State
from mm.mpc.reward_tensor import objective_entropic_return, objective_inventory_capacity_penalty


def joint_action_param_len(n_components: int) -> int:
    """Flat optimizer vector length for local + global sigmoid curves per horizon step."""

    return 2 * action_param_len(n_components)


def split_venue_action_flat(
    flat: np.ndarray | list[float],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    single = action_param_len(n_components)
    arr = np.asarray(flat, dtype=np.float64)
    if arr.size < 2 * single:
        raise ValueError(f"joint action requires {2 * single} params, got {arr.size}")
    return arr[:single], arr[single : 2 * single]


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
        l_param: Any,
        global_active: bool,
    ) -> JointSigmoidMixtureAction:
        local_flat, global_flat = split_venue_action_flat(flat, n_components)
        local = sigmoid_action_from_flat_raw(local_flat, n_components, l_param=l_param)
        global_action = sigmoid_action_from_flat_raw(global_flat, n_components, l_param=l_param)
        return cls(local=local, global_=global_action, global_active=global_active)


def reference_price_from_state(state: State) -> float:
    """Mark consolidated wealth using the global sell-side reference when available."""

    if state.reference_price > 0.0:
        return float(state.reference_price)
    if state.global_sell_vwap > 0.0:
        return float(state.global_sell_vwap)
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
    returns: Any,
    local_inventory_path: Any,
    global_inventory_path: Any,
    *,
    return_risk_aversion_beta: float,
    inventory_target_penalty_weight: float,
    local_target_inventory: float,
    global_target_inventory: float,
    local_inventory_scale: float,
    global_inventory_scale: float,
    global_active: float,
) -> tuple[Any, dict[str, Any]]:
    """Terminal objective with joint inventory penalty across venues."""

    import torch

    returns = torch.nan_to_num(returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_return = objective_entropic_return(returns, beta=return_risk_aversion_beta)
    local_pen = objective_inventory_capacity_penalty(
        local_inventory_path,
        target_inventory=local_target_inventory,
        inventory_scale=local_inventory_scale,
    )
    global_pen = objective_inventory_capacity_penalty(
        global_inventory_path,
        target_inventory=global_target_inventory,
        inventory_scale=global_inventory_scale,
    )
    inv_penalty = local_pen + float(global_active) * global_pen
    term_inv = -float(inventory_target_penalty_weight) * inv_penalty
    components = {
        "risk_adjusted_return": term_return,
        "inventory_capacity_penalty": term_inv,
        "local_inventory_penalty": -float(inventory_target_penalty_weight) * local_pen,
        "global_inventory_penalty": -float(inventory_target_penalty_weight)
        * float(global_active)
        * global_pen,
    }
    return term_return + term_inv, components


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
        global_buy_vwap=state.global_buy_vwap,
        global_sell_vwap=state.global_sell_vwap,
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


def wealth_0_from_state(state: State) -> float:
    ref = reference_price_from_state(state)
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
