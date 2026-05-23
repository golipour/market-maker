"""Differentiable PyTorch tensor equivalents of State and Action.

State: mid, fixed-length orderbook arrays (configured price buckets per side), buy_vol, sell_vol,
       inventory, cash — all as tensors so gradients can flow.
Action: flat tensor of sigmoid-mixture parameters (6 * n_components) with differentiable
        depth and depth_derivative for use in reward/optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from mm.mpc.l_param import LParamConfig
    from mm.mpc.mpc_state import State, SigmoidMixtureAction

# Default number of orderbook buckets per side.
DEFAULT_OB_LEVELS = 100


def _get_torch():
    import torch
    return torch


@dataclass
class StateTensor:
    """
    Differentiable state as PyTorch tensors.

    - mid: scalar or (1,) — mid price.
    - ob_bids: (n_levels,) — base qty in configured price bucket below mid.
    - ob_asks: (n_levels,) — base qty in configured price bucket above mid.
    - buy_vol, sell_vol, inventory, cash: scalar or (1,) — flow and position.
    All tensors are float (inventory as float for gradient flow).
    """

    mid: Tensor
    ob_bids: Tensor
    ob_asks: Tensor
    buy_vol: Tensor
    sell_vol: Tensor
    inventory: Tensor
    cash: Tensor

    @property
    def device(self) -> "torch.device":
        return self.mid.device

    @property
    def dtype(self):
        return self.mid.dtype

    def scalar_mid(self) -> float:
        return float(self.mid.view(-1)[0].detach().item())

    def scalar_inventory(self) -> float:
        return float(self.inventory.view(-1)[0].detach().item())

    def scalar_cash(self) -> float:
        return float(self.cash.view(-1)[0].detach().item())


@dataclass
class ActionTensor:
    """
    Differentiable action: flat tensor of sigmoid-mixture parameters.

    params: (6 * n_components,) with layout
      [bid_L1, bid_k1, bid_x0_1, ..., bid_Ln, bid_kn, bid_x0_n,
       ask_L1, ask_k1, ask_x0_1, ..., ask_Ln, ask_kn, ask_x0_n].

    L and k are interpreted as positive via softplus in depth computations so
    unconstrained optimization is valid; x0 is free.
    """

    params: Tensor
    n_components: int

    def __post_init__(self):
        n = self.n_components
        if self.params.numel() != 6 * n:
            raise ValueError(f"ActionTensor params must have 6*n_components={6*n} elements, got {self.params.numel()}")

    @property
    def device(self) -> "torch.device":
        return self.params.device

    @property
    def dtype(self):
        return self.params.dtype


# -----------------------------------------------------------------------------
# Differentiable depth from action params (L, k > 0 via softplus)
# -----------------------------------------------------------------------------


def _softplus_positive(x: "Tensor") -> "Tensor":
    """Return positive values from unconstrained x (for L, k)."""
    torch = _get_torch()
    return torch.nn.functional.softplus(x) + 1e-8


def depth_bid_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """
    Cumulative bid depth at quote-price spread: D(δ) = sum_i L_i * sigmoid(k_i * (δ - x0_i)).
    L and k taken as softplus(params) when use_softplus=True so gradients flow.
    spread: any shape; output has same shape.
    """
    from mm.mpc.l_param import materialize_side_torch

    torch = _get_torch()
    if l_param is not None:
        L, k, x0 = materialize_side_torch(
            params, side_off=0, n_components=n_components, l_param=l_param
        )
    else:
        from mm.mpc.l_param import positive_L_torch

        L_raw = params[0 : 3 * n_components : 3]
        k_raw = params[1 : 3 * n_components : 3]
        x0 = params[2 : 3 * n_components : 3]
        L = positive_L_torch(
            L_raw,
            k_raw,
            x0,
            params=params,
            side_off=0,
            l_param=l_param,
            use_softplus=use_softplus,
        )
        if use_softplus:
            k = _softplus_positive(k_raw)
        else:
            k = k_raw
    # spread: (...,); L, k, x0: (n_components,)
    # z_i = k_i * (spread - x0_i) -> (..., n_components)
    z = (spread.unsqueeze(-1) - x0) * k
    s = torch.sigmoid(z)
    depth = (L * s).sum(dim=-1)
    return depth


def depth_ask_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """Cumulative ask depth at quote-price spread (same formula as bid, ask-side params)."""
    from mm.mpc.l_param import materialize_side_torch

    torch = _get_torch()
    base = 3 * n_components
    if l_param is not None:
        L, k, x0 = materialize_side_torch(
            params, side_off=base, n_components=n_components, l_param=l_param
        )
    else:
        from mm.mpc.l_param import positive_L_torch

        L_raw = params[base + 0 : base + 3 * n_components : 3]
        k_raw = params[base + 1 : base + 3 * n_components : 3]
        x0 = params[base + 2 : base + 3 * n_components : 3]
        L = positive_L_torch(
            L_raw,
            k_raw,
            x0,
            params=params,
            side_off=base,
            l_param=l_param,
            use_softplus=use_softplus,
        )
        if use_softplus:
            k = _softplus_positive(k_raw)
        else:
            k = k_raw
    z = (spread.unsqueeze(-1) - x0) * k
    s = torch.sigmoid(z)
    depth = (L * s).sum(dim=-1)
    return depth


def effective_depth_bid_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """Executable bid depth ``D_bid(spread) - D_bid(0)`` with non-negative clamp."""
    torch = _get_torch()
    raw = depth_bid_tensor(params, spread, n_components, use_softplus=use_softplus, l_param=l_param)
    s0 = torch.zeros(1, dtype=spread.dtype, device=spread.device)
    base = depth_bid_tensor(params, s0, n_components, use_softplus=use_softplus, l_param=l_param)[0]
    return torch.clamp(raw - base, min=0.0)


def effective_depth_ask_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """Executable ask depth ``D_ask(spread) - D_ask(0)`` with non-negative clamp."""
    torch = _get_torch()
    raw = depth_ask_tensor(params, spread, n_components, use_softplus=use_softplus, l_param=l_param)
    s0 = torch.zeros(1, dtype=spread.dtype, device=spread.device)
    base = depth_ask_tensor(params, s0, n_components, use_softplus=use_softplus, l_param=l_param)[0]
    return torch.clamp(raw - base, min=0.0)


def depth_bid_derivative_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """Marginal bid depth d/dδ D(δ) = sum_i L_i * k_i * s_i * (1 - s_i). Same shape as spread."""
    from mm.mpc.l_param import materialize_side_torch

    torch = _get_torch()
    if l_param is not None:
        L, k, x0 = materialize_side_torch(
            params, side_off=0, n_components=n_components, l_param=l_param
        )
    else:
        from mm.mpc.l_param import positive_L_torch

        L_raw = params[0 : 3 * n_components : 3]
        k_raw = params[1 : 3 * n_components : 3]
        x0 = params[2 : 3 * n_components : 3]
        L = positive_L_torch(
            L_raw,
            k_raw,
            x0,
            params=params,
            side_off=0,
            l_param=l_param,
            use_softplus=use_softplus,
        )
        if use_softplus:
            k = _softplus_positive(k_raw)
        else:
            k = k_raw
    z = (spread.unsqueeze(-1) - x0) * k
    s = torch.sigmoid(z)
    der = (L * k * s * (1.0 - s)).sum(dim=-1)
    return der


def depth_ask_derivative_tensor(
    params: "Tensor",
    spread: "Tensor",
    n_components: int,
    use_softplus: bool = True,
    l_param: "LParamConfig | None" = None,
) -> "Tensor":
    """Marginal ask depth d/dδ D(δ). Same shape as spread."""
    from mm.mpc.l_param import materialize_side_torch

    torch = _get_torch()
    base = 3 * n_components
    if l_param is not None:
        L, k, x0 = materialize_side_torch(
            params, side_off=base, n_components=n_components, l_param=l_param
        )
    else:
        from mm.mpc.l_param import positive_L_torch

        L_raw = params[base + 0 : base + 3 * n_components : 3]
        k_raw = params[base + 1 : base + 3 * n_components : 3]
        x0 = params[base + 2 : base + 3 * n_components : 3]
        L = positive_L_torch(
            L_raw,
            k_raw,
            x0,
            params=params,
            side_off=base,
            l_param=l_param,
            use_softplus=use_softplus,
        )
        if use_softplus:
            k = _softplus_positive(k_raw)
        else:
            k = k_raw
    z = (spread.unsqueeze(-1) - x0) * k
    s = torch.sigmoid(z)
    der = (L * k * s * (1.0 - s)).sum(dim=-1)
    return der


# -----------------------------------------------------------------------------
# Slippage term: -‖∇d‖₂² over a spread grid (differentiable)
# -----------------------------------------------------------------------------


def slippage_penalty_tensor(
    params: "Tensor",
    n_components: int,
    spread_bid: "Tensor | None" = None,
    spread_ask: "Tensor | None" = None,
    spread_bid_ticks: "Tensor | None" = None,
    spread_ask_ticks: "Tensor | None" = None,
    use_softplus: bool = True,
) -> "Tensor":
    """
    Slippage penalty -‖∇d‖₂²: negative sum of squares of marginal depth at grid points.
    Returns a scalar tensor; gradients flow through params.
    """
    torch = _get_torch()
    device = params.device
    dtype = params.dtype
    if spread_bid is None:
        spread_bid = spread_bid_ticks
    if spread_ask is None:
        spread_ask = spread_ask_ticks
    if spread_bid is None:
        spread_bid = torch.arange(0.0, 51.0, dtype=dtype, device=device)
    if spread_ask is None:
        spread_ask = torch.arange(0.0, 51.0, dtype=dtype, device=device)
    db = depth_bid_derivative_tensor(params, spread_bid, n_components, use_softplus)
    da = depth_ask_derivative_tensor(params, spread_ask, n_components, use_softplus)
    sq = db.square().sum() + da.square().sum()
    return -sq


# -----------------------------------------------------------------------------
# Conversion: State <-> StateTensor
# -----------------------------------------------------------------------------


def state_to_tensor(
    state: State,
    device: "str | torch.device | None" = None,
    n_levels: int = DEFAULT_OB_LEVELS,
    spread_step: float = 1.0,
    dtype: "torch.dtype | None" = None,
) -> StateTensor:
    """
    Convert State to StateTensor. Orderbook is expanded to fixed-length arrays:
    ob_bids[i] = base qty in spread bucket with upper edge (i+1) * spread_step from mid.
    """
    from mm.mpc.mpc_state import State, OrderbookRelative

    torch = _get_torch()
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64

    mid = torch.tensor([state.mid], dtype=dtype, device=device)
    buy_vol = torch.tensor([state.buy_vol], dtype=dtype, device=device)
    sell_vol = torch.tensor([state.sell_vol], dtype=dtype, device=device)
    inventory = torch.tensor([float(state.inventory)], dtype=dtype, device=device)
    cash = torch.tensor([state.cash], dtype=dtype, device=device)

    ob_bids = torch.zeros(n_levels, dtype=dtype, device=device)
    ob_asks = torch.zeros(n_levels, dtype=dtype, device=device)
    step = max(1e-12, float(spread_step))
    for (spread, qty) in state.ob_rel.bids:
        idx = min(max(int(math.ceil(float(spread) / step)) - 1, 0), n_levels - 1)
        if 0 <= idx < n_levels:
            ob_bids[idx] = float(qty)
    for (spread, qty) in state.ob_rel.asks:
        idx = min(max(int(math.ceil(float(spread) / step)) - 1, 0), n_levels - 1)
        if 0 <= idx < n_levels:
            ob_asks[idx] = float(qty)

    return StateTensor(
        mid=mid,
        ob_bids=ob_bids,
        ob_asks=ob_asks,
        buy_vol=buy_vol,
        sell_vol=sell_vol,
        inventory=inventory,
        cash=cash,
    )


def state_tensor_to_state(state_t: StateTensor, ob_rel_from_bids_asks: bool = False) -> State:
    """
    Convert StateTensor back to State. Requires OrderbookRelative from elsewhere unless
    we reconstruct from ob_bids/ob_asks (sparse levels with qty > 0).
    """
    from mm.mpc.mpc_state import State, OrderbookRelative

    mid = state_t.scalar_mid()
    buy_vol = float(state_t.buy_vol.view(-1)[0].detach().item())
    sell_vol = float(state_t.sell_vol.view(-1)[0].detach().item())
    inventory = state_t.scalar_inventory()
    cash = state_t.scalar_cash()

    if ob_rel_from_bids_asks:
        bids = tuple(
            (float(i + 1), int(round(float(q))))
            for i, q in enumerate(state_t.ob_bids)
            if q.detach().item() > 0
        )
        asks = tuple(
            (float(i + 1), int(round(float(q))))
            for i, q in enumerate(state_t.ob_asks)
            if q.detach().item() > 0
        )
        ob_rel = OrderbookRelative(bids=bids, asks=asks)
    else:
        ob_rel = OrderbookRelative(bids=(), asks=())

    return State(
        mid=mid,
        ob_rel=ob_rel,
        buy_vol=buy_vol,
        sell_vol=sell_vol,
        inventory=inventory,
        cash=cash,
        buy_vwap=mid,
        sell_vwap=mid,
    )


# -----------------------------------------------------------------------------
# Conversion: SigmoidMixtureAction <-> ActionTensor
# -----------------------------------------------------------------------------


def action_tensor_from_sigmoid(
    sigmoid_action: SigmoidMixtureAction,
    device: "str | torch.device | None" = None,
    dtype: "torch.dtype | None" = None,
    requires_grad: bool = False,
) -> ActionTensor:
    """Build ActionTensor from SigmoidMixtureAction (flat param vector)."""
    from mm.mpc.mpc_state import SigmoidMixtureAction

    torch = _get_torch()
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64

    flat = sigmoid_action.to_flat()
    params = torch.tensor(flat, dtype=dtype, device=device, requires_grad=requires_grad)
    return ActionTensor(params=params, n_components=sigmoid_action.n_components)


def sigmoid_from_action_tensor(
    action_t: ActionTensor,
    use_softplus: bool = True,
) -> SigmoidMixtureAction:
    """Build SigmoidMixtureAction from ActionTensor; uses softplus for L,k if use_softplus."""
    from mm.mpc.mpc_state import SigmoidMixtureAction

    torch = _get_torch()
    n = action_t.n_components
    p = action_t.params.detach()
    if use_softplus:
        L_bid = _softplus_positive(p[0 : 3 * n : 3]).tolist()
        k_bid = _softplus_positive(p[1 : 3 * n : 3]).tolist()
        x0_bid = p[2 : 3 * n : 3].tolist()
        base = 3 * n
        L_ask = _softplus_positive(p[base + 0 : base + 3 * n : 3]).tolist()
        k_ask = _softplus_positive(p[base + 1 : base + 3 * n : 3]).tolist()
        x0_ask = p[base + 2 : base + 3 * n : 3].tolist()
    else:
        L_bid = p[0 : 3 * n : 3].tolist()
        k_bid = p[1 : 3 * n : 3].tolist()
        x0_bid = p[2 : 3 * n : 3].tolist()
        base = 3 * n
        L_ask = p[base + 0 : base + 3 * n : 3].tolist()
        k_ask = p[base + 1 : base + 3 * n : 3].tolist()
        x0_ask = p[base + 2 : base + 3 * n : 3].tolist()

    return SigmoidMixtureAction(
        bid_L=tuple(L_bid),
        bid_k=tuple(k_bid),
        bid_x0=tuple(x0_bid),
        ask_L=tuple(L_ask),
        ask_k=tuple(k_ask),
        ask_x0=tuple(x0_ask),
    )
