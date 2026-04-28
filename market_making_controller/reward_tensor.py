"""
Differentiable (PyTorch) reward functions that take StateTensor and ActionTensor
and return a scalar tensor so gradients flow for training or gradient-based optimization.

Terminal objective (single mode):
  J = reward_coef * J_mean - inv_coef * (J_inv - target_inv)^2 - risk_coef * J_std
      - uniformity_coef * J_uniformity - size_coef * J_size - spread_coef * J_spread
where:
  J_mean = E[ΔW], J_inv = E[q_{t+H}], J_std = Std(ΔW),
  J_uniformity = E[L2-to-uniform marginal penalty], J_size/J_spread are action penalties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from .state_tensor import StateTensor, ActionTensor

from .state_tensor import (
    StateTensor,
    ActionTensor,
    depth_bid_derivative_tensor,
    depth_ask_derivative_tensor,
    slippage_penalty_tensor,
)


def _get_torch():
    import torch
    return torch


def delta_pnl_detailed_tensor(state: StateTensor, next_state: StateTensor) -> Tensor:
    """
    Delta PnL from ΔW = ΔX + Q·ΔP + ΔQ·P + ΔQ·ΔP.
    Returns a scalar tensor; gradients flow through state/next_state tensors.
    """
    torch = _get_torch()
    dX = next_state.cash - state.cash
    dQ = next_state.inventory - state.inventory
    dP = next_state.mid - state.mid
    Q = state.inventory
    P = state.mid
    # Flatten to scalars if (1,) shaped
    dX = dX.view(-1)[0]
    dQ = dQ.view(-1)[0]
    dP = dP.view(-1)[0]
    Q = Q.view(-1)[0]
    P = P.view(-1)[0]
    return dX + Q * dP + dQ * P + dQ * dP


def inventory_pnl_term_tensor(state: StateTensor, next_state: StateTensor) -> Tensor:
    """
    Inventory penalty -(Q + ΔQ)² = -(next_state.inventory)².
    Returns a scalar tensor.
    """
    q_next = next_state.inventory.view(-1)[0]
    return -(q_next * q_next)


def slippage_pnl_term_tensor(
    action: ActionTensor,
    spread_bid_ticks: Tensor | None = None,
    spread_ask_ticks: Tensor | None = None,
    use_softplus: bool = True,
) -> Tensor:
    """
    Slippage penalty -‖∇d‖₂² (L2² of marginal depth over grid).
    Returns a scalar tensor; gradients flow through action.params.
    """
    return slippage_penalty_tensor(
        action.params,
        action.n_components,
        spread_bid_ticks=spread_bid_ticks,
        spread_ask_ticks=spread_ask_ticks,
        use_softplus=use_softplus,
    )


def reward_delta_inventory_slippage_tensor(
    state: StateTensor,
    action: ActionTensor,
    next_state: StateTensor,
    w_delta: float = 1.0,
    w_inv: float = 0.01,
    w_slip: float = 0.0,
    spread_bid_ticks: Tensor | None = None,
    spread_ask_ticks: Tensor | None = None,
    use_softplus: bool = True,
) -> Tensor:
    """
    Composite differentiable reward:
      w_delta * delta_pnl_detailed_tensor(s, s') + w_inv * inventory_pnl_term_tensor(s, s') + w_slip * slippage_pnl_term_tensor(a).
    Returns a scalar tensor; gradients flow through state, action.params, next_state.
    """
    torch = _get_torch()
    r_delta = delta_pnl_detailed_tensor(state, next_state)
    r_inv = inventory_pnl_term_tensor(state, next_state)
    r_slip = slippage_pnl_term_tensor(
        action,
        spread_bid_ticks=spread_bid_ticks,
        spread_ask_ticks=spread_ask_ticks,
        use_softplus=use_softplus,
    )
    w_d = torch.tensor(w_delta, dtype=state.mid.dtype, device=state.mid.device)
    w_i = torch.tensor(w_inv, dtype=state.mid.dtype, device=state.mid.device)
    w_s = torch.tensor(w_slip, dtype=state.mid.dtype, device=state.mid.device)
    return w_d * r_delta + w_i * r_inv + w_s * r_slip


# -----------------------------------------------------------------------------
# Terminal objective components (batched: per-trajectory ΔW and terminal inventory)
# All modes use ΔW = W(s_{t+H}) - W(s_t); W(s) = c + q*p.
# -----------------------------------------------------------------------------


def objective_mean(delta_W: "Tensor") -> "Tensor":
    """J_mean = E[ΔW]. delta_W: (n_samples,) or (n_samples, 1)."""
    torch = _get_torch()
    d = delta_W.view(-1)
    return d.mean()


def objective_inv(inv_terminal: "Tensor") -> "Tensor":
    """J_inv = E[q_{t+H}]. inv_terminal: (n_samples,) or (n_samples, 1)."""
    torch = _get_torch()
    q = inv_terminal.view(-1)
    return q.mean()


def objective_var(delta_W: "Tensor") -> "Tensor":
    """J_var = Var(ΔW) = E[ΔW^2] - (E[ΔW])^2. delta_W: (n_samples,) or (n_samples, 1)."""
    torch = _get_torch()
    d = delta_W.view(-1)
    return d.var(unbiased=False)


def objective_std(delta_W: "Tensor") -> "Tensor":
    """J_std = Std(ΔW) = sqrt(Var(ΔW)). delta_W: (n_samples,) or (n_samples, 1)."""
    torch = _get_torch()
    v = objective_var(delta_W)
    return torch.sqrt(v.clamp(min=1e-20))


def terminal_objective(
    delta_W: "Tensor",
    inv_terminal: "Tensor",
    reward_coef: float = 1.0,
    inv_coef: float = 0.0,
    target_inv: float = 0.0,
    risk_coef: float = 0.0,
    uniformity: "Tensor | None" = None,
    uniformity_coef: float = 0.0,
    size: "Tensor | None" = None,
    size_coef: float = 0.0,
    spread: "Tensor | None" = None,
    spread_coef: float = 0.0,
) -> tuple["Tensor", dict[str, Any]]:
    """
    Build scalar objective J from terminal ΔW and optional action regularizers.

    delta_W: (n_samples,) wealth change per trajectory, ΔW = W(s_{t+H}) - W(s_t).
    inv_terminal: (n_samples,) or (n_samples, 1) terminal inventory per trajectory.
    reward_coef: multiplicative weight for mean reward term J_mean.
    inv_coef: weight for inventory target penalty.
    target_inv: desired terminal inventory target.
    risk_coef: weight for ΔW volatility penalty (J_std).
    uniformity: differentiable L2-to-uniform regularizer term from marginal action distributions.
    uniformity_coef: additive weight for uniformity penalty.
    size: differentiable action-size penalty term.
    size_coef: additive weight for size penalty.
    spread: differentiable spread-at-depth penalty term.
    spread_coef: additive weight for spread penalty.

    Returns:
        J: scalar tensor to maximize (loss = -J).
        components: dict with weighted objective terms (ready for plotting/summing):
            - term_reward = reward_coef * J_mean
            - term_inv = -inv_coef * (J_inv - target_inv)^2
            - term_risk = -risk_coef * J_std
            - term_uniformity = -uniformity_coef * J_uniformity
            - term_size = -size_coef * J_size
            - term_spread = -spread_coef * J_spread
    """
    torch = _get_torch()
    # Batched rollouts can produce inf/nan in ΔW when mid×inv overflows; poison the whole objective otherwise.
    delta_W = torch.nan_to_num(delta_W, nan=0.0, posinf=1e30, neginf=-1e30)
    inv_terminal = torch.nan_to_num(inv_terminal, nan=0.0, posinf=1e30, neginf=-1e30)
    J_mean = objective_mean(delta_W)
    J_inv = objective_inv(inv_terminal)
    J_std = objective_std(delta_W)
    if uniformity is None:
        J_uniformity = J_mean * 0.0
    else:
        J_uniformity = torch.nan_to_num(
            uniformity.view(-1).mean(), nan=0.0, posinf=1e6, neginf=-1e6
        )
    if size is None:
        J_size = J_mean * 0.0
    else:
        J_size = torch.nan_to_num(size.view(-1).mean(), nan=0.0, posinf=1e6, neginf=-1e6)
    if spread is None:
        J_spread = J_mean * 0.0
    else:
        J_spread = torch.nan_to_num(spread.view(-1).mean(), nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.tensor(float(target_inv), dtype=J_inv.dtype, device=J_inv.device)
    J_inv_penalty = (J_inv - target) * (J_inv - target)

    term_reward = float(reward_coef) * J_mean
    term_inv = -float(inv_coef) * J_inv_penalty
    term_risk = -float(risk_coef) * J_std
    term_uniformity = -float(uniformity_coef) * J_uniformity
    term_size = -float(size_coef) * J_size
    term_spread = -float(spread_coef) * J_spread
    components = {
        "term_reward": term_reward,
        "term_inv": term_inv,
        "term_risk": term_risk,
        "term_uniformity": term_uniformity,
        "term_size": term_size,
        "term_spread": term_spread,
    }
    J = term_reward + term_inv + term_risk + term_uniformity + term_size + term_spread

    return J, components
