"""
Differentiable (PyTorch) reward functions that take StateTensor and ActionTensor
and return a scalar tensor so gradients flow for training or gradient-based optimization.

Terminal objective (single mode):
  J = J_mean - inv_coef * (J_inv - target_inv)^2 - risk_coef * J_std + action_coef * J_action_entropy
where:
  J_mean = E[ΔW], J_inv = E[q_{t+H}], J_std = Std(ΔW), J_action_entropy = E[normalized action entropy]
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
    inv_coef: float = 0.0,
    target_inv: float = 0.0,
    risk_coef: float = 0.0,
    action_entropy: "Tensor | None" = None,
    action_coef: float = 0.0,
) -> tuple["Tensor", dict[str, Any]]:
    """
    Build scalar objective J from terminal ΔW and optional action entropy regularizer.

    delta_W: (n_samples,) wealth change per trajectory, ΔW = W(s_{t+H}) - W(s_t).
    inv_terminal: (n_samples,) or (n_samples, 1) terminal inventory per trajectory.
    inv_coef: weight for inventory target penalty.
    target_inv: desired terminal inventory target.
    risk_coef: weight for ΔW volatility penalty (J_std).
    action_entropy: differentiable entropy regularizer term from action distribution.
    action_coef: additive weight for action entropy regularization.

    Returns:
        J: scalar tensor to maximize (loss = -J).
        components: dict with J_mean, J_inv, J_std, J_action_entropy (as scalars) for logging.
    """
    torch = _get_torch()
    # Batched rollouts can produce inf/nan in ΔW when mid×inv overflows; poison the whole objective otherwise.
    delta_W = torch.nan_to_num(delta_W, nan=0.0, posinf=1e30, neginf=-1e30)
    inv_terminal = torch.nan_to_num(inv_terminal, nan=0.0, posinf=1e30, neginf=-1e30)
    J_mean = objective_mean(delta_W)
    J_inv = objective_inv(inv_terminal)
    J_std = objective_std(delta_W)
    if action_entropy is None:
        J_action_entropy = J_mean * 0.0
    else:
        J_action_entropy = torch.nan_to_num(
            action_entropy.view(-1).mean(), nan=0.0, posinf=1e6, neginf=-1e6
        )
    target = torch.tensor(float(target_inv), dtype=J_inv.dtype, device=J_inv.device)
    J_inv_penalty = (J_inv - target) * (J_inv - target)

    components = {
        "J_mean": J_mean,
        "J_inv": J_inv,
        "J_inv_penalty": J_inv_penalty,
        "J_std": J_std,
        "J_action_entropy": J_action_entropy,
    }
    J = (
        J_mean
        - float(inv_coef) * J_inv_penalty
        - float(risk_coef) * J_std
        + float(action_coef) * J_action_entropy
    )

    return J, components
