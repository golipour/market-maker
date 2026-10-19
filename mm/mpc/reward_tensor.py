"""
Differentiable (PyTorch) reward functions that take StateTensor and ActionTensor
and return a scalar tensor so gradients flow for training or gradient-based optimization.

Terminal objective:
  J = risk_adjusted_return

where:
  risk_adjusted_return = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))
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


def objective_entropic_return(
    returns: "Tensor",
    *,
    beta: float,
    weights: "Tensor | None" = None,
) -> "Tensor":
    """
    Risk-adjusted return (certainty equivalent):

        CE = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))

    When weighted scenarios are provided:
        CE = min(R) - (1/beta) * ln(sum_i w_i * exp(-beta * (R_i - min(R))))
    """
    torch = _get_torch()
    r = returns.view(-1)
    if r.numel() <= 0:
        return r.sum()
    b = max(1e-6, float(beta))
    r_min = r.min()
    shifted = r - r_min
    if weights is not None:
        w = torch.clamp(weights.view(-1), min=1e-12)
        w = w / w.sum()
        log_w = torch.log(w)
        log_mean_exp = torch.logsumexp(-b * shifted + log_w, dim=0)
    else:
        log_mean_exp = torch.logsumexp(-b * shifted, dim=0) - torch.log(
            torch.tensor(float(r.numel()), dtype=r.dtype, device=r.device)
        )
    return r_min - log_mean_exp / b


def terminal_objective(
    returns: "Tensor",
    *,
    return_risk_aversion_beta: float = 1.0,
    weights: "Tensor | None" = None,
) -> tuple["Tensor", dict[str, Any]]:
    """
    Build scalar objective J from per-path returns.

    returns: (n_samples,) horizon return R_i = ΔW_i / W_0 per trajectory.
    weights: optional (n_samples,) probability weights summing to 1.0.

    Returns:
        J: scalar tensor to maximize (loss = -J).
        components:
            - risk_adjusted_return
    """
    torch = _get_torch()
    returns = torch.nan_to_num(returns.view(-1), nan=0.0, posinf=1e30, neginf=-1e30)
    term_return = objective_entropic_return(
        returns, beta=return_risk_aversion_beta, weights=weights
    )
    components = {
        "risk_adjusted_return": term_return,
    }
    return term_return, components
