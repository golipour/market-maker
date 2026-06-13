"""Joint-venue MPC helpers: four-curve actions, dual-venue rollout, consolidated wealth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mm.mpc.l_param import action_param_len, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import OrderbookRelative, SigmoidMixtureAction, State
from mm.mpc.reward_tensor import objective_entropic_return


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
    """Return the selected liquidation VWAP reference used to mark inventory."""

    source = str(reference_price_source or "").strip().lower()
    net_inventory = float(state.inventory) + float(state.global_inventory)
    if source == "local":
        return _inventory_mark_price_from_vwaps(
            net_inventory,
            buy_vwap=float(state.buy_vwap),
            sell_vwap=float(state.sell_vwap),
            flat_price=float(state.reference_price or state.mid),
        )
    if source == "global":
        if state.global_sell_vwap is not None and float(state.global_sell_vwap) > 0.0:
            return _inventory_mark_price_from_vwaps(
                net_inventory,
                buy_vwap=float(state.global_buy_vwap or state.buy_vwap),
                sell_vwap=float(state.global_sell_vwap),
                flat_price=float(state.reference_price or state.global_mid or state.mid),
            )
        return _inventory_mark_price_from_vwaps(
            net_inventory,
            buy_vwap=float(state.buy_vwap),
            sell_vwap=float(state.sell_vwap),
            flat_price=float(state.reference_price or state.mid),
        )
    if state.reference_price > 0.0:
        return float(state.reference_price)
    if state.global_sell_vwap is not None and float(state.global_sell_vwap) > 0.0:
        return _inventory_mark_price_from_vwaps(
            net_inventory,
            buy_vwap=float(state.global_buy_vwap or state.buy_vwap),
            sell_vwap=float(state.global_sell_vwap),
            flat_price=float(state.global_mid or state.mid),
        )
    return _inventory_mark_price_from_vwaps(
        net_inventory,
        buy_vwap=float(state.buy_vwap),
        sell_vwap=float(state.sell_vwap),
        flat_price=float(state.mid),
    )


def _inventory_mark_price_from_vwaps(
    net_inventory: float,
    *,
    buy_vwap: float,
    sell_vwap: float,
    flat_price: float,
) -> float:
    if net_inventory > 0.0:
        return float(sell_vwap)
    if net_inventory < 0.0:
        return float(buy_vwap)
    return float(flat_price)


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
    *,
    local_returns: Any | None = None,
    global_returns: Any | None = None,
    return_risk_aversion_beta: float,
    global_active: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Terminal objective on consolidated portfolio return with optional venue diagnostics."""

    import torch

    returns = torch.nan_to_num(returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_return = objective_entropic_return(
        returns,
        beta=return_risk_aversion_beta,
    )
    components: dict[str, Any] = {
        "risk_adjusted_return": term_return,
    }
    if local_returns is not None:
        local_returns = torch.nan_to_num(
            local_returns.view(-1),
            nan=0.0,
            posinf=1e30,
            neginf=-1e30,
        )
        components["local_risk_adjusted_return"] = objective_entropic_return(
            local_returns,
            beta=return_risk_aversion_beta,
        )
    if global_returns is not None and float(global_active) > 0.0:
        global_returns = torch.nan_to_num(
            global_returns.view(-1),
            nan=0.0,
            posinf=1e30,
            neginf=-1e30,
        )
        components["global_risk_adjusted_return"] = objective_entropic_return(
            global_returns,
            beta=return_risk_aversion_beta,
        )
    return term_return, components


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
        local_book_mid=state.local_book_mid,
        liquidity_book_mid=state.liquidity_book_mid,
        quote_anchor_price=state.quote_anchor_price,
        inventory_mark_price=state.inventory_mark_price,
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
