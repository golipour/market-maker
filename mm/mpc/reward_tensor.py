"""
Differentiable (PyTorch) reward functions that take StateTensor and ActionTensor
and return a scalar tensor so gradients flow for training or gradient-based optimization.

Terminal objective (two terms):
  J = risk_adjusted_return
      + inventory_capacity_penalty

where:
  risk_adjusted_return = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))
  inventory_capacity_penalty = -lambda_inv * mean_{i,t} ((I_{i,t} - I_target) / I_scale)^2
  R_i = horizon return per Monte Carlo path (dimensionless)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from mm.mpc.state_tensor import StateTensor, ActionTensor

from mm.mpc.state_tensor import (
    StateTensor,
    ActionTensor,
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
    dX = dX.view(-1)[0]
    dQ = dQ.view(-1)[0]
    dP = dP.view(-1)[0]
    Q = Q.view(-1)[0]
    P = P.view(-1)[0]
    return dX + Q * dP + dQ * P + dQ * dP


def inventory_pnl_term_tensor(state: StateTensor, next_state: StateTensor) -> Tensor:
    """Inventory penalty -(Q + ΔQ)² = -(next_state.inventory)²."""
    q_next = next_state.inventory.view(-1)[0]
    return -(q_next * q_next)


def slippage_pnl_term_tensor(
    action: ActionTensor,
    spread_bid_ticks: Tensor | None = None,
    spread_ask_ticks: Tensor | None = None,
    use_softplus: bool = True,
) -> Tensor:
    """Slippage penalty -‖∇d‖₂² (L2² of marginal depth over grid)."""
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
    """Composite differentiable reward for legacy training helpers."""
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


def objective_mean(values: "Tensor") -> "Tensor":
    """E[x] over the leading sample dimension."""
    torch = _get_torch()
    return values.view(-1).mean()


def objective_entropic_return(returns: "Tensor", *, beta: float) -> "Tensor":
    """
    Risk-adjusted return (certainty equivalent):

        CE = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))

    Shifting by ``min(R)`` keeps every exponent <= 0, so ``exp`` stays in (0, 1]
    and the formula matches ``(-1/beta) * ln(mean_i exp(-beta * R_i))`` without overflow.
    """
    torch = _get_torch()
    r = returns.view(-1)
    if r.numel() <= 0:
        return r.sum()
    b = max(1e-6, float(beta))
    r_min = r.min()
    shifted = r - r_min
    log_mean_exp = torch.logsumexp(-b * shifted, dim=0) - torch.log(
        torch.tensor(float(r.numel()), dtype=r.dtype, device=r.device)
    )
    return r_min - log_mean_exp / b


def objective_inventory_capacity_penalty(
    inventory_path: "Tensor",
    *,
    target_inventory: float,
    inventory_scale: float,
) -> "Tensor":
    """
    Normalized quadratic inventory target miss averaged over MC paths and horizon steps.

    R_inv(I) = ((I - I_target) / I_scale)^2
    """
    torch = _get_torch()
    inv = inventory_path.view(inventory_path.shape[0], -1)
    inv = torch.nan_to_num(inv, nan=0.0, posinf=1e30, neginf=-1e30)
    target = torch.tensor(float(target_inventory), dtype=inv.dtype, device=inv.device)
    scale = max(1e-8, float(inventory_scale))
    return (((inv - target) / scale) ** 2).mean()


def terminal_objective(
    returns: "Tensor",
    inventory_path: "Tensor",
    *,
    return_risk_aversion_beta: float = 1.0,
    inventory_target_penalty_weight: float = 0.0,
    target_inventory: float = 0.0,
    inventory_scale: float = 1.0,
) -> tuple["Tensor", dict[str, Any]]:
    """
    Build scalar objective J from per-path returns and simulated inventory path.

    returns: (n_samples,) horizon return R_i = ΔW_i / W_0 per trajectory.
    inventory_path: (n_samples, horizon) simulated inventory after each rollout step.

    Returns:
        J: scalar tensor to maximize (loss = -J).
        components:
            - risk_adjusted_return
            - inventory_capacity_penalty
    """
    torch = _get_torch()
    returns = torch.nan_to_num(returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_return = objective_entropic_return(returns, beta=return_risk_aversion_beta)
    inv_penalty = objective_inventory_capacity_penalty(
        inventory_path,
        target_inventory=target_inventory,
        inventory_scale=inventory_scale,
    )
    term_inv = -float(inventory_target_penalty_weight) * inv_penalty
    components = {
        "risk_adjusted_return": term_return,
        "inventory_capacity_penalty": term_inv,
    }
    J = term_return + term_inv
    return J, components
