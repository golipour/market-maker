"""
Differentiable (PyTorch) reward functions that take StateTensor and ActionTensor
and return a scalar tensor so gradients flow for training or gradient-based optimization.

Terminal objective (two terms):
  J = risk_adjusted_return
      + threshold_shortfall_penalty

where:
  risk_adjusted_return = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))
  threshold_shortfall_penalty = -lambda_thr * mean_{i,t} (
      relu(C_threshold - C_{i,t}) / C_scale
      + relu(I_threshold - I_{i,t}) / I_scale
  )
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
    """Composite differentiable reward for single-venue training helpers."""
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


def objective_threshold_shortfall_penalty(
    cash_path: "Tensor",
    inventory_path: "Tensor",
    *,
    cash_threshold: float,
    inventory_threshold: float,
) -> "Tensor":
    """
    One-sided normalized cash/inventory threshold shortfall penalty.

    R = relu(C_threshold - C) / C_scale
      + relu(I_threshold - I) / I_scale

    The penalty is zero when both cash and inventory are above their thresholds.
    """
    torch = _get_torch()
    cash = cash_path.view(cash_path.shape[0], -1)
    inv = inventory_path.view(inventory_path.shape[0], -1)
    cash = torch.nan_to_num(cash, nan=0.0, posinf=1e30, neginf=-1e30)
    inv = torch.nan_to_num(inv, nan=0.0, posinf=1e30, neginf=-1e30)
    cash_thr = torch.tensor(float(cash_threshold), dtype=cash.dtype, device=cash.device)
    inv_thr = torch.tensor(float(inventory_threshold), dtype=inv.dtype, device=inv.device)
    cash_scale = max(1.0, abs(float(cash_threshold)))
    inv_scale = max(1.0, abs(float(inventory_threshold)))
    cash_shortfall = torch.clamp(cash_thr - cash, min=0.0) / cash_scale
    inv_shortfall = torch.clamp(inv_thr - inv, min=0.0) / inv_scale
    return (cash_shortfall + inv_shortfall).mean()


def terminal_objective(
    returns: "Tensor",
    cash_path: "Tensor",
    inventory_path: "Tensor",
    *,
    return_risk_aversion_beta: float = 1.0,
    inventory_threshold_penalty_weight: float = 0.0,
    cash_threshold: float = 0.0,
    inventory_threshold: float = 0.0,
) -> tuple["Tensor", dict[str, Any]]:
    """
    Build scalar objective J from per-path returns and simulated inventory path.

    returns: (n_samples,) horizon return R_i = ΔW_i / W_0 per trajectory.
    cash_path: (n_samples, horizon) simulated cash after each rollout step.
    inventory_path: (n_samples, horizon) simulated inventory after each rollout step.

    Returns:
        J: scalar tensor to maximize (loss = -J).
        components:
            - risk_adjusted_return
            - threshold_shortfall_penalty
    """
    torch = _get_torch()
    returns = torch.nan_to_num(returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_return = objective_entropic_return(returns, beta=return_risk_aversion_beta)
    threshold_penalty = objective_threshold_shortfall_penalty(
        cash_path,
        inventory_path,
        cash_threshold=cash_threshold,
        inventory_threshold=inventory_threshold,
    )
    term_threshold = -float(inventory_threshold_penalty_weight) * threshold_penalty
    components = {
        "risk_adjusted_return": term_return,
        "threshold_shortfall_penalty": term_threshold,
    }
    J = term_return + term_threshold
    return J, components
