"""
Batched Monte Carlo MPC: batches over n_samples in a single forward/backward for performance.
Single MPC implementation; MPCOptimizer protocol and helpers live here.
"""

from __future__ import annotations

import math
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from mm.mpc.l_param import (
    LParamConfig,
    action_param_len,
    sigmoid_action_from_flat_raw,
    validate_l_param_config,
)
from mm.mpc.mpc_state import State
from mm.core.wallet import portfolio_wealth
from mm.mpc.torch_device import resolve_torch_device
from mm.mpc.state_tensor import (
    effective_depth_bid_tensor,
    effective_depth_ask_tensor,
)
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.optimizer_diagnostics import (
    EpochObjectiveMetrics,
    OptimizerRunDiagnostics,
    build_action_snapshot,
)
from mm.observability.logs import log_event

from mm.pricing import target_depth_lots_from_price_model, vwap_prices_batched


# -----------------------------------------------------------------------------
# MPCOptimizer protocol (controller and package surface)
# -----------------------------------------------------------------------------


@runtime_checkable
class MPCOptimizer(Protocol):
    """Given state, return optimal action. Estimators provided at construction."""

    def optimize(self, state: State) -> SigmoidMixtureAction:
        ...


# -----------------------------------------------------------------------------
# Helpers: bounds, quote-size projection, book cumulative
# -----------------------------------------------------------------------------


def _get_torch():
    return torch


def _coalesce_param(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _bounds_effective_to_raw(bounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Convert optimizer bounds from effective parameter space to raw tensor space.

    Mixture ``L`` slots are logits; ``k`` bounds are inverse-softplus mapped; ``x0`` and
    trailing per-side depth logits pass through unchanged.
    """
    out: list[tuple[float, float]] = []
    for i, (lo, hi) in enumerate(bounds):
        if i >= len(bounds) - 2:
            out.append((float(lo), float(hi)))
        elif i % 3 == 0:
            out.append((float(lo), float(hi)))
        elif i % 3 == 1:
            out.append((_inv_softplus_np(float(lo)), _inv_softplus_np(float(hi))))
        else:
            out.append((float(lo), float(hi)))
    return out


def _inv_softplus_np(y: float) -> float:
    """Inverse softplus for positive target y (numpy scalar)."""
    y_f = float(max(1e-12, y))
    if y_f > 50.0:
        return y_f
    return float(np.log(np.expm1(y_f)))


def _initial_action_uniform_marginal_qrange(
    n_components: int,
    spread_max: float,
    l_param: LParamConfig,
) -> np.ndarray:
    """
    Build an initial flat action (raw optimizer params) using a simple linear-range
    sigmoid approximation. Depth logits start at zero (midpoint of configured range).
    """
    n = max(1, int(n_components))
    sm = max(1, int(round(float(spread_max))))
    q_ref = l_param.quote_size_midpoint()
    q_level = float(q_ref) / float(sm)
    x0_center = 0.5 * float(sm)
    k_base = max(1e-3, min(0.05, 2.0 / max(1.0, float(sm))))
    k_eff = np.full(n, k_base, dtype=np.float64)
    L_total = max(1e-6, 4.0 * q_level / max(k_base, 1e-12))
    L_eff = np.full(n, max(1e-6, L_total / float(n)), dtype=np.float64)
    L_raw = np.zeros(n, dtype=np.float64)
    k_raw = np.array([_inv_softplus_np(v) for v in k_eff], dtype=np.float64)

    if l_param.spread_structured:
        u_scale = 0.0
        u_init = 0.0
        x0_raw = np.zeros(n, dtype=np.float64)
        if n > 1:
            fracs = np.linspace(0.25, 0.85, n - 1)
            for j, frac in enumerate(fracs):
                frac = min(max(float(frac), 1e-4), 1.0 - 1e-4)
                x0_raw[j + 1] = float(np.log(frac / (1.0 - frac)))
        bid_part: list[float] = []
        ask_part: list[float] = []
        bid_part.extend([u_scale, float(k_raw[0]), u_init])
        for i in range(1, n):
            bid_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        ask_part.extend([0.0, float(k_raw[0]), u_init])
        for i in range(1, n):
            ask_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        core = np.array(bid_part + ask_part, dtype=np.float64)
    else:
        x0_raw = np.zeros(n, dtype=np.float64)
        bid_part = []
        ask_part = []
        for i in range(n):
            bid_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
            ask_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        core = np.array(bid_part + ask_part, dtype=np.float64)

    return np.concatenate([core, np.array([0.0, 0.0], dtype=np.float64)])


def _format_action_verbose(flat: np.ndarray, n_components: int) -> str:
    """Format first action (flat 6*n) for verbose logging."""
    n = n_components
    if len(flat) < 6 * n:
        return str(flat)
    bid_L = [flat[3 * i] for i in range(n)]
    bid_k = [flat[3 * i + 1] for i in range(n)]
    bid_x0 = [flat[3 * i + 2] for i in range(n)]
    ask_L = [flat[3 * n + 3 * i] for i in range(n)]
    ask_k = [flat[3 * n + 3 * i + 1] for i in range(n)]
    ask_x0 = [flat[3 * n + 3 * i + 2] for i in range(n)]
    return (
        f"bid L={[round(x, 2) for x in bid_L]} k={[round(x, 3) for x in bid_k]} x0={[round(x, 2) for x in bid_x0]} | "
        f"ask L={[round(x, 2) for x in ask_L]} k={[round(x, 3) for x in ask_k]} x0={[round(x, 2) for x in ask_x0]}"
    )


def _grid_levels(spread_max: float, spread_step: float) -> int:
    step = max(1e-12, float(spread_step))
    return max(2, int(math.ceil(float(spread_max) / step)))


def _spread_grid_tensor(torch_mod: Any, grid_levels: int, spread_step: float, device: Any, dtype: Any) -> Any:
    """Upper bucket edges from spread_step through grid_levels * spread_step (domain [0, spread_max])."""
    return torch_mod.arange(1.0, float(grid_levels) + 1.0, dtype=dtype, device=device) * float(spread_step)


def _book_cumulative_from_ob_rel(
    ob_rel: OrderbookRelative,
    grid_levels: int,
    spread_step: float,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """Build cumulative qty tensors on quote spread buckets through each bucket upper edge."""
    torch_mod = _get_torch()
    bid_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    ask_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    step = max(1e-12, float(spread_step))
    b_sum = 0.0
    for (spread, qty) in sorted(ob_rel.bids, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            b_sum += float(qty)
            bid_cum[idx] = b_sum
    for i in range(1, grid_levels):
        if bid_cum[i].item() == 0:
            bid_cum[i] = bid_cum[i - 1]
    a_sum = 0.0
    for (spread, qty) in sorted(ob_rel.asks, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            a_sum += float(qty)
            ask_cum[idx] = a_sum
    for i in range(1, grid_levels):
        if ask_cum[i].item() == 0:
            ask_cum[i] = ask_cum[i - 1]
    return bid_cum, ask_cum


def _spread_bucket_index(spread: float, spread_step: float, grid_levels: int) -> int:
    if spread <= 0.0:
        return 0
    idx = int(math.ceil(spread / spread_step)) - 1
    return min(max(idx, 0), grid_levels - 1)


def _book_cumulative_from_arrays(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched marginal per-bucket qty -> cumulative depth along the spread grid."""
    return torch.cumsum(ob_bids, dim=1), torch.cumsum(ob_asks, dim=1)



def _mid_from_book_best_levels_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Mid implied by post-flow best bid/ask from level volumes.
    center_mid sets the absolute price reference; spread_grid is quote-price distance from mid.
    """
    big = float(spread_grid.max().item()) + 1000.0
    mask_b = (ob_bids > 0).to(center_mid.dtype)
    mask_a = (ob_asks > 0).to(center_mid.dtype)
    spread_b = spread_grid.unsqueeze(0) * mask_b + (1.0 - mask_b) * big
    spread_a = spread_grid.unsqueeze(0) * mask_a + (1.0 - mask_a) * big
    best_bid_spread = spread_b.min(dim=1, keepdim=True)[0]
    best_ask_spread = spread_a.min(dim=1, keepdim=True)[0]
    best_bid = center_mid - best_bid_spread
    best_ask = center_mid + best_ask_spread
    # If one side is empty, keep center_mid for that side (same fallback as previous logic).
    no_bid = best_bid_spread >= big
    no_ask = best_ask_spread >= big
    best_bid = torch.where(no_bid, center_mid, best_bid)
    best_ask = torch.where(no_ask, center_mid, best_ask)
    return 0.5 * (best_bid + best_ask)


_SOFT_FILL_ACTIVATIONS = frozenset({"sigmoid", "hardsigmoid"})


def _normalize_soft_fill_activation(activation: str) -> str:
    kind = str(activation).lower().strip()
    if kind not in _SOFT_FILL_ACTIVATIONS:
        supported = ", ".join(sorted(_SOFT_FILL_ACTIVATIONS))
        raise ValueError(f"unsupported soft_fill_activation {activation!r}; expected one of: {supported}")
    return kind


def _soft_fill_fraction(
    reach: torch.Tensor,
    soft_temp: float,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """
    Map normalized reach to a per-level consumption fraction in [0, 1].

    ``soft_temp`` scales the input before the activation:
    ``scaled_reach = reach / soft_temp``.

    ``hardsigmoid`` is exactly zero when ``scaled_reach <= -3``, eliminating
    asymptotic ghost fills from far-from-mid quote levels.
    """
    scaled = reach / max(float(soft_temp), 1e-12)
    kind = _normalize_soft_fill_activation(activation)
    if kind == "hardsigmoid":
        return F.hardsigmoid(scaled)
    return torch.sigmoid(scaled)


def soft_fill_batched(
    params: torch.Tensor,
    n_components: int,
    sell_vwap: torch.Tensor,
    buy_vwap: torch.Tensor,
    book_cum_bid: torch.Tensor,
    book_cum_ask: torch.Tensor,
    sell_vol: torch.Tensor,
    buy_vol: torch.Tensor,
    spread_grid: torch.Tensor,
    maker_fee: float,
    soft_temp: float = 1.0,
    soft_fill_activation: str = "sigmoid",
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
    depth_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched soft transition using combined (book + quoted) depth:
    - soft-consume against combined cumulative book
    - split consumed volume between market book and our quotes proportionally per level
    - return next market-only orderbook levels plus cash/inventory deltas from our fills

    Inputs:
      book_cum_*: market-book cumulative depth on spread grid (n_samples, grid_levels)
      sell_vol/buy_vol: market order flow per sample (n_samples, 1)

    Returns:
      ob_bids_next, ob_asks_next: next market-only level volumes (n_samples, grid_levels)
      cash_delta, inv_delta: deltas from our fills only (n_samples,)
    """
    depth_bid = effective_depth_bid_tensor(
        params, spread_grid, n_components, use_softplus=use_softplus, l_param=l_param
    )
    depth_ask = effective_depth_ask_tensor(
        params, spread_grid, n_components, use_softplus=use_softplus, l_param=l_param
    )
    eps = 1e-12
    # Convert cumulative quoted depth to marginal level sizes.
    depth_bid_prev = torch.cat([torch.zeros_like(depth_bid[:1]), depth_bid[:-1]], dim=0)
    depth_ask_prev = torch.cat([torch.zeros_like(depth_ask[:1]), depth_ask[:-1]], dim=0)
    marg_bid = torch.clamp(depth_bid - depth_bid_prev, min=0.0)
    marg_ask = torch.clamp(depth_ask - depth_ask_prev, min=0.0)

    # Recover market-book level volumes from cumulative market book.
    book_bid_prev = torch.cat([torch.zeros_like(book_cum_bid[:, :1]), book_cum_bid[:, :-1]], dim=1)
    book_ask_prev = torch.cat([torch.zeros_like(book_cum_ask[:, :1]), book_cum_ask[:, :-1]], dim=1)
    book_bid_level = torch.clamp(book_cum_bid - book_bid_prev, min=0.0)
    book_ask_level = torch.clamp(book_cum_ask - book_ask_prev, min=0.0)

    # Combine market book and our quotes on each level.
    scale = max(0.0, float(depth_scale))
    our_bid_level = marg_bid.unsqueeze(0).expand_as(book_bid_level) * scale
    our_ask_level = marg_ask.unsqueeze(0).expand_as(book_ask_level) * scale
    comb_bid_level = book_bid_level + our_bid_level
    comb_ask_level = book_ask_level + our_ask_level

    # Soft consumption against combined cumulative depth.
    comb_cum_bid = torch.cumsum(comb_bid_level, dim=1)
    comb_cum_ask = torch.cumsum(comb_ask_level, dim=1)
    reach_bid_raw = sell_vol - comb_cum_bid
    reach_ask_raw = buy_vol - comb_cum_ask
    bid_pos_scale = torch.clamp(reach_bid_raw.max(dim=1, keepdim=True).values, min=eps)
    bid_neg_scale = torch.clamp((-reach_bid_raw).max(dim=1, keepdim=True).values, min=eps)
    ask_pos_scale = torch.clamp(reach_ask_raw.max(dim=1, keepdim=True).values, min=eps)
    ask_neg_scale = torch.clamp((-reach_ask_raw).max(dim=1, keepdim=True).values, min=eps)
    reach_bid = torch.where(
        reach_bid_raw >= 0.0,
        reach_bid_raw / bid_pos_scale,
        reach_bid_raw / bid_neg_scale,
    )
    reach_ask = torch.where(
        reach_ask_raw >= 0.0,
        reach_ask_raw / ask_pos_scale,
        reach_ask_raw / ask_neg_scale,
    )
    fill_frac_bid = _soft_fill_fraction(reach_bid, soft_temp, soft_fill_activation)
    fill_frac_ask = _soft_fill_fraction(reach_ask, soft_temp, soft_fill_activation)
    consumed_bid_level = comb_bid_level * fill_frac_bid
    consumed_ask_level = comb_ask_level * fill_frac_ask

    # Split consumed level volumes proportionally between market book and our quotes.
    bid_share_our = torch.where(
        comb_bid_level > eps, our_bid_level / torch.clamp(comb_bid_level, min=eps), torch.zeros_like(comb_bid_level)
    )
    ask_share_our = torch.where(
        comb_ask_level > eps, our_ask_level / torch.clamp(comb_ask_level, min=eps), torch.zeros_like(comb_ask_level)
    )
    fill_bid_level = consumed_bid_level * bid_share_our
    fill_ask_level = consumed_ask_level * ask_share_our
    fill_bid = fill_bid_level.sum(dim=1)
    fill_ask = fill_ask_level.sum(dim=1)

    # Market-book next state after market-flow consumption (our resting remainder is canceled).
    consumed_bid_market = consumed_bid_level * (1.0 - bid_share_our)
    consumed_ask_market = consumed_ask_level * (1.0 - ask_share_our)
    ob_bids_next = torch.clamp(book_bid_level - consumed_bid_market, min=0.0)
    ob_asks_next = torch.clamp(book_ask_level - consumed_ask_market, min=0.0)
    sell_exp = sell_vwap.expand(-1, spread_grid.shape[0])
    buy_exp = buy_vwap.expand(-1, spread_grid.shape[0])
    cash_bid = (fill_bid_level * (sell_exp - spread_grid.unsqueeze(0))).sum(dim=1)
    cash_ask = (fill_ask_level * (buy_exp + spread_grid.unsqueeze(0))).sum(dim=1)
    one_minus_f = 1.0 - maker_fee
    cash_delta = cash_ask * one_minus_f - cash_bid
    inv_delta = fill_bid * one_minus_f - fill_ask
    return ob_bids_next, ob_asks_next, cash_delta, inv_delta


def _delta_wealth_mtm_batched(
    sell_vwap: torch.Tensor,
    inv: torch.Tensor,
    cash: torch.Tensor,
    sell_vwap_next: torch.Tensor,
    inv_next: torch.Tensor,
    cash_next: torch.Tensor,
) -> torch.Tensor:
    """Per-sample ΔW with inventory marked at sell-side VWAP: W = cash + inv * sell_vwap."""
    W = cash + inv * sell_vwap
    W_next = cash_next + inv_next * sell_vwap_next
    return (W_next - W).squeeze(1)


class MonteCarloMPCTensorBatched:
    """
    Batched MC MPC: same API as MonteCarloMPCTensor but batches over n_samples
    in a single forward/backward per epoch for better performance.
    """

    def __init__(
        self,
        price_model: Any,
        market_flow_model: Any,
        execution_model: Any,
        reference_global_price_model: Any = None,
        n_components: int = 2,
        n_samples: int = 32,
        horizon: int = 4,
        spread_max: float = 100.0,
        spread_step: float = 1.0,
        soft_fill_temp: float = 0.1,
        soft_fill_activation: str = "sigmoid",
        adam_lr: float = 0.05,
        adam_epochs: int = 80,
        adam_grad_clip: float = 10.0,
        device: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        warm_start: bool = False,
        fixed_mc_samples: bool = False,
        verbose: bool = False,
        inv_coef: float | None = None,
        target_inv: float | None = None,
        global_coef: float = 0.0,
        global_fee: float = 0.0,
        global_reward_weight: float = 1.0,
        return_risk_aversion_beta: float = 1.0,
        inventory_target_penalty_weight: float | None = None,
        target_inventory: float | None = None,
        target_inventory_multiplier: float = 1.5,
        liquidity_sourcing_enabled: bool = False,
        global_market_flow_model: Any = None,
        global_price_model: Any = None,
        quote_size_min_per_side: float | int | None = None,
        quote_size_max_per_side: float | int | None = None,
        quote_spread_max: float | int | None = None,
        quote_spread_min_depth: float | int | None = None,
        quote_spread_k_epsilon: float | None = None,
    ):
        self.price_model = price_model
        self.market_flow_model = market_flow_model
        self.execution_model = execution_model
        self.reference_global_price_model = reference_global_price_model
        self.n_components = max(1, n_components)
        self.n_samples = max(1, n_samples)
        self.horizon = max(1, horizon)
        self.spread_max = float(spread_max)
        self.spread_step = max(1e-12, float(spread_step))
        self.grid_levels = _grid_levels(self.spread_max, self.spread_step)
        self.soft_fill_temp = soft_fill_temp
        self.soft_fill_activation = _normalize_soft_fill_activation(soft_fill_activation)
        self.adam_lr = adam_lr
        self.adam_epochs = adam_epochs
        self.adam_grad_clip = adam_grad_clip
        self.device = device
        self.warm_start = warm_start
        self.quote_size_min_per_side = (
            float(quote_size_min_per_side) if quote_size_min_per_side is not None else None
        )
        self.quote_size_max_per_side = (
            float(quote_size_max_per_side) if quote_size_max_per_side is not None else None
        )
        if self.quote_size_min_per_side is None or self.quote_size_max_per_side is None:
            raise ValueError("quote_size_min_per_side and quote_size_max_per_side are required")
        if self.quote_size_min_per_side <= 0.0 or self.quote_size_max_per_side <= 0.0:
            raise ValueError("quote_size_min_per_side and quote_size_max_per_side must be positive")
        if self.quote_size_min_per_side > self.quote_size_max_per_side:
            raise ValueError(
                "quote_size_min_per_side must be <= quote_size_max_per_side"
            )
        self.fixed_mc_samples = fixed_mc_samples
        self.verbose = verbose
        self.return_risk_aversion_beta = max(1e-6, float(return_risk_aversion_beta))
        self.inventory_target_penalty_weight = max(
            0.0, float(_coalesce_param(inventory_target_penalty_weight, inv_coef, default=0.0))
        )
        explicit_target = _coalesce_param(target_inventory, target_inv, default=None)
        self.target_inventory = float(explicit_target) if explicit_target is not None else None
        self.target_inventory_multiplier = max(1e-6, float(target_inventory_multiplier))
        self.global_enabled = bool(float(global_coef) != 0.0)
        self.global_fee = max(0.0, min(1.0 - 1e-9, float(global_fee)))
        self.global_reward_weight = max(0.0, float(global_reward_weight))
        self.liquidity_sourcing_enabled = bool(liquidity_sourcing_enabled)
        self.global_market_flow_model = global_market_flow_model
        self.global_price_model = global_price_model
        self.quote_spread_max = (
            float(quote_spread_max) if quote_spread_max is not None else None
        )
        self.quote_spread_min_depth = (
            float(quote_spread_min_depth) if quote_spread_min_depth is not None else None
        )
        self.quote_spread_k_epsilon = (
            float(quote_spread_k_epsilon)
            if quote_spread_k_epsilon is not None
            else 0.01
        )
        lp = self._l_param_config()
        validate_l_param_config(lp)
        self.inv_coef = self.inventory_target_penalty_weight
        self.target_inv = self._resolve_inventory_target()
        self._last_action: SigmoidMixtureAction | None = None
        self._last_raw_flat: np.ndarray | None = None
        self._last_J_components: dict | None = None
        self._last_J_components_initial: dict | None = None
        self._last_J_mean_raw: float = 0.0
        self._last_E_dW_next: float = 0.0
        self._last_main_reward_mean: float = 0.0
        self._last_global_reward_mean: float = 0.0
        self._last_main_reward_mean_initial: float = 0.0
        self._last_global_reward_mean_initial: float = 0.0
        self._last_sigmoid_action: SigmoidMixtureAction | None = None
        self._last_global_sigmoid_action: SigmoidMixtureAction | None = None
        self._last_joint_action: Any = None
        self._last_diagnostics: OptimizerRunDiagnostics | None = None
        self._x_torch: Any = None
        self._optimizer: Any = None

        if bounds is None:
            n = self.n_components
            logit_bound = (-30.0, 30.0)
            k_bound = (1e-6, 100.0)
            x0_bound = (-self.spread_max, self.spread_max)
            self.bounds = [logit_bound, k_bound, x0_bound] * (2 * n) + [logit_bound, logit_bound]
        else:
            self.bounds = bounds

    @property
    def last_diagnostics(self) -> OptimizerRunDiagnostics | None:
        return self._last_diagnostics

    def _l_param_config(self) -> LParamConfig:
        return LParamConfig(
            quote_size_min=float(self.quote_size_min_per_side),
            quote_size_max=float(self.quote_size_max_per_side),
            spread_max=self.spread_max,
            n_components=self.n_components,
            quote_spread_max=self.quote_spread_max,
            quote_spread_min_depth=self.quote_spread_min_depth,
            quote_spread_k_epsilon=self.quote_spread_k_epsilon,
        )

    def _resolve_inventory_target(self) -> float:
        if self.target_inventory is not None and float(self.target_inventory) > 0.0:
            return float(self.target_inventory)
        q_mid = self._l_param_config().quote_size_midpoint()
        return float(self.target_inventory_multiplier) * q_mid

    def _resolve_inventory_scale(self) -> float:
        return max(1e-8, self._resolve_inventory_target())

    def _depth_call_kwargs(self) -> dict[str, Any]:
        return {"use_softplus": True, "l_param": self._l_param_config()}

    def _flat_to_effective_action(self, flat: np.ndarray) -> SigmoidMixtureAction:
        return sigmoid_action_from_flat_raw(
            flat, self.n_components, l_param=self._l_param_config()
        )

    def optimize(self, state: State) -> SigmoidMixtureAction:
        action, _ = self.optimize_with_diagnostics(state)
        return action

    def optimize_with_diagnostics(self, state: State) -> tuple[SigmoidMixtureAction, OptimizerRunDiagnostics]:
        if self.liquidity_sourcing_enabled:
            from mm.mpc.joint_optimizer import optimize_joint_with_diagnostics

            return optimize_joint_with_diagnostics(self, state)
        torch_mod = _get_torch()
        if isinstance(self.device, str) and self.device.lower() == "cpu":
            _device = torch_mod.device("cpu")
        elif self.device is None:
            _device = resolve_torch_device(None)
        else:
            _device = resolve_torch_device(str(self.device).strip() or None)
        dtype = torch_mod.float64

        book_cum_bid, book_cum_ask = _book_cumulative_from_ob_rel(
            state.ob_rel, self.grid_levels, self.spread_step, _device, dtype
        )
        spread_grid = _spread_grid_tensor(torch_mod, self.grid_levels, self.spread_step, _device, dtype)
        mid_0 = float(state.mid)
        inv_0 = float(state.inventory)
        cash_0 = float(state.cash)
        sell_vwap_0 = float(state.sell_vwap)
        wealth_0 = portfolio_wealth(cash_0, inv_0, sell_vwap_0)
        obj_inventory_target = self._resolve_inventory_target()
        obj_inventory_scale = self._resolve_inventory_scale()

        ob_bids_0 = torch_mod.zeros(self.n_samples, self.grid_levels, dtype=dtype, device=_device)
        ob_asks_0 = torch_mod.zeros(self.n_samples, self.grid_levels, dtype=dtype, device=_device)
        for (s, q) in state.ob_rel.bids:
            i = _spread_bucket_index(float(s), self.spread_step, self.grid_levels)
            if 0 <= i < self.grid_levels:
                ob_bids_0[:, i] += float(q)
        for (s, q) in state.ob_rel.asks:
            i = _spread_bucket_index(float(s), self.spread_step, self.grid_levels)
            if 0 <= i < self.grid_levels:
                ob_asks_0[:, i] += float(q)
        n = self.n_components
        single_len = action_param_len(n)
        param_len = self.horizon * single_len

        if self.warm_start and self._last_raw_flat is not None:
            one_action = np.array(self._last_raw_flat[:single_len], dtype=np.float64)
        elif self.warm_start and self._last_action is not None:
            one_action = np.array(self._last_action.to_flat(), dtype=np.float64)
            if one_action.size < single_len:
                one_action = np.concatenate(
                    [one_action, np.zeros(single_len - one_action.size, dtype=np.float64)]
                )
        else:
            one_action = _initial_action_uniform_marginal_qrange(
                n_components=n,
                spread_max=self.spread_max,
                l_param=self._l_param_config(),
            )
        x0_arr = np.tile(one_action, self.horizon)

        bounds_raw = _bounds_effective_to_raw(self.bounds)
        bounds_full = bounds_raw * self.horizon
        lower = torch_mod.tensor([b[0] for b in bounds_full], dtype=dtype, device=_device)
        upper = torch_mod.tensor([b[1] for b in bounds_full], dtype=dtype, device=_device)

        if self._x_torch is None or self._x_torch.numel() != param_len:
            self._x_torch = torch_mod.tensor(x0_arr, dtype=dtype, device=_device, requires_grad=True)
            self._optimizer = torch_mod.optim.Adam([self._x_torch], lr=self.adam_lr)
        else:
            with torch_mod.no_grad():
                self._x_torch.data = torch_mod.tensor(x0_arr, dtype=dtype, device=_device)
            # If warm start is disabled, do not carry Adam moments across optimize()
            # calls; otherwise one-step updates can be dominated by stale momentum.
            if not self.warm_start:
                self._optimizer = torch_mod.optim.Adam([self._x_torch], lr=self.adam_lr)
        x_torch = self._x_torch
        optimizer = self._optimizer
        maker_fee = getattr(self.execution_model, "maker_fee", 0.0)
        global_fee_rollout = getattr(self.execution_model, "global_fee", self.global_fee)
        global_fee_rollout = max(0.0, min(1.0 - 1e-9, float(global_fee_rollout)))
        vwap_depth = target_depth_lots_from_price_model(self.price_model)

        # Pre-draw MC samples: (n_samples, horizon) for MOF, price residuals,
        # and hedge delta shocks (used when hedge dynamics are enabled).
        def draw_samples():
            sample_shape = (self.n_samples, self.horizon)
            if hasattr(self.market_flow_model, "sample"):
                buy_vol, sell_vol = self.market_flow_model.sample(
                    sample_shape, device=_device, dtype=dtype
                )
            else:
                buy_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                sell_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                for s in range(self.n_samples):
                    for h in range(self.horizon):
                        bv, sv = self.market_flow_model.sample((1,), device=_device, dtype=dtype)
                        buy_vol[s, h] = bv.squeeze()
                        sell_vol[s, h] = sv.squeeze()
            if hasattr(self.price_model, "sample_residual"):
                eps_buy, eps_sell = self.price_model.sample_residual(
                    sample_shape, device=_device, dtype=dtype
                )
            else:
                eps_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                eps_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            if self.reference_global_price_model is not None and hasattr(self.reference_global_price_model, "sample_deltas"):
                d_hedge_buy, d_hedge_sell = self.reference_global_price_model.sample_deltas(
                    sample_shape,
                    device=_device,
                    dtype=dtype,
                    torch_mod=torch_mod,
                )
            else:
                d_hedge_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                d_hedge_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            return buy_vol, sell_vol, eps_buy, eps_sell, d_hedge_buy, d_hedge_sell

        batch_time_sec = 0.0
        if self.fixed_mc_samples:
            t_batch = time.perf_counter()
            (
                fixed_buy_vol,
                fixed_sell_vol,
                fixed_eps_buy,
                fixed_eps_sell,
                fixed_d_hedge_buy,
                fixed_d_hedge_sell,
            ) = draw_samples()
            batch_time_sec = time.perf_counter() - t_batch

        spread_levels = tuple(
            round(i * self.spread_step, 12) for i in range(self.grid_levels + 1)
        )
        initial_flat = np.array(x_torch.detach().cpu().numpy()[:single_len], dtype=np.float64)
        initial_action = self._flat_to_effective_action(initial_flat)
        action_initial_snap = build_action_snapshot(
            initial_action, mid=mid_0, spread_levels=spread_levels
        )

        best_loss = float("inf")
        best_x = x_torch.detach().clone()
        best_J_mean: float | None = None
        best_J_mean_next: float | None = None
        best_main_reward_mean: float | None = None
        best_global_reward_mean: float | None = None
        best_epoch: int | None = None
        initial_metrics: EpochObjectiveMetrics | None = None
        final_metrics: EpochObjectiveMetrics | None = None
        best_metrics: EpochObjectiveMetrics | None = None
        self._last_J_components_initial = None
        captured_initial_components = False
        nonfinite_loss_count = 0
        nonfinite_grad_count = 0
        t0 = time.perf_counter()

        for epoch in range(self.adam_epochs):
            if self.verbose:
                cur_flat = x_torch.detach().cpu().numpy()[:single_len]
                print(f"  Epoch {epoch}: action  {_format_action_verbose(cur_flat, n)}")

            def closure():
                nonlocal batch_time_sec, captured_initial_components, initial_metrics, final_metrics
                nonlocal nonfinite_loss_count, nonfinite_grad_count
                optimizer.zero_grad()
                with torch_mod.no_grad():
                    x_torch.data = torch_mod.clamp(x_torch.data, lower, upper)

                if self.fixed_mc_samples:
                    buy_vol, sell_vol, eps_buy, eps_sell, d_hedge_buy, d_hedge_sell = (
                        fixed_buy_vol,
                        fixed_sell_vol,
                        fixed_eps_buy,
                        fixed_eps_sell,
                        fixed_d_hedge_buy,
                        fixed_d_hedge_sell,
                    )
                else:
                    t_batch = time.perf_counter()
                    buy_vol, sell_vol, eps_buy, eps_sell, d_hedge_buy, d_hedge_sell = draw_samples()
                    batch_time_sec += time.perf_counter() - t_batch

                inv_t = torch_mod.full((self.n_samples, 1), inv_0, dtype=dtype, device=_device)
                cash_t = torch_mod.full((self.n_samples, 1), cash_0, dtype=dtype, device=_device)
                ob_bids = ob_bids_0.clone()
                ob_asks = ob_asks_0.clone()
                mid_fallback_t = torch_mod.full((self.n_samples, 1), mid_0, dtype=dtype, device=_device)
                global_buy_vwap_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(state.global_buy_vwap if state.global_buy_vwap is not None else state.buy_vwap),
                    dtype=dtype,
                    device=_device,
                )
                global_sell_vwap_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(state.global_sell_vwap if state.global_sell_vwap is not None else state.sell_vwap),
                    dtype=dtype,
                    device=_device,
                )
                hedge_inv_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(getattr(state, "global_inventory", 0)),
                    dtype=dtype,
                    device=_device,
                )
                hedge_cash_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(getattr(state, "hedge_cash", 0.0)),
                    dtype=dtype,
                    device=_device,
                )

                book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(self.n_samples, -1)
                book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(self.n_samples, -1)

                total_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                total_main_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                total_hedge_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                inv_path = torch_mod.zeros(self.n_samples, self.horizon, dtype=dtype, device=_device)
                step_reward_0 = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)

                for h in range(self.horizon):
                    if h == 0:
                        book_cum_bid_h = book_cum_bid_0
                        book_cum_ask_h = book_cum_ask_0
                    else:
                        book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks)

                    action_h = x_torch[h * single_len : (h + 1) * single_len]
                    depth_kw = self._depth_call_kwargs()
                    sell_h = sell_vol[:, h : h + 1]
                    buy_h = buy_vol[:, h : h + 1]
                    eps_buy_h = eps_buy[:, h : h + 1]
                    eps_sell_h = eps_sell[:, h : h + 1]
                    if h + 1 < self.horizon:
                        eps_buy_next = eps_buy[:, h + 1 : h + 2]
                        eps_sell_next = eps_sell[:, h + 1 : h + 2]
                    else:
                        eps_buy_next = eps_buy_h
                        eps_sell_next = eps_sell_h

                    mid_t = _mid_from_book_best_levels_batched(
                        ob_bids,
                        ob_asks,
                        mid_fallback_t,
                        spread_grid,
                    )
                    if h == 0:
                        # Use upstream VWAPs (single vwap_depth policy),
                        # instead of recomputing from the book arrays.
                        buy_vwap_book_t = torch_mod.full(
                            (self.n_samples, 1),
                            float(state.buy_vwap),
                            dtype=dtype,
                            device=_device,
                        )
                        sell_vwap_book_t = torch_mod.full(
                            (self.n_samples, 1),
                            float(state.sell_vwap),
                            dtype=dtype,
                            device=_device,
                        )
                    else:
                        buy_vwap_book_t, sell_vwap_book_t = vwap_prices_batched(
                            ob_bids,
                            ob_asks,
                            mid_t,
                            spread_grid,
                            target_depth=vwap_depth,
                        )
                    buy_vwap_t = buy_vwap_book_t + eps_buy_h
                    sell_vwap_t = sell_vwap_book_t + eps_sell_h

                    ob_bids_next, ob_asks_next, cash_delta, inv_delta = soft_fill_batched(
                        action_h,
                        n,
                        sell_vwap_t,
                        buy_vwap_t,
                        book_cum_bid_h,
                        book_cum_ask_h,
                        sell_h,
                        buy_h,
                        spread_grid,
                        maker_fee,
                        soft_temp=self.soft_fill_temp,
                        soft_fill_activation=self.soft_fill_activation,
                        **depth_kw,
                    )
                    inv_next = inv_t + inv_delta.unsqueeze(1)
                    cash_next = cash_t + cash_delta.unsqueeze(1)
                    inv_path[:, h] = inv_next.squeeze(1)
                    if self.global_enabled:
                        d_buy = d_hedge_buy[:, h : h + 1]
                        d_sell = d_hedge_sell[:, h : h + 1]
                        global_buy_vwap_next = global_buy_vwap_t + d_buy
                        global_sell_vwap_next = global_sell_vwap_t + d_sell
                        d_inv = inv_delta.unsqueeze(1)  # fee-adjusted executed inventory change
                        # Hedge trade at t+1 on hedge venue:
                        # - d_inv>0 (longer main) -> sell hedge inventory at hedge bid
                        # - d_inv<0 (shorter main) -> buy hedge inventory at hedge ask
                        trade_cash_hedge = torch_mod.where(
                            d_inv >= 0.0,
                            d_inv * global_sell_vwap_next,
                            d_inv * global_buy_vwap_next,
                        )
                        one_minus_hedge_fee = float(1.0 - global_fee_rollout)
                        hedge_cash_next = hedge_cash_t + trade_cash_hedge * one_minus_hedge_fee
                        hedge_inv_next = hedge_inv_t - d_inv
                    else:
                        global_buy_vwap_next = global_buy_vwap_t
                        global_sell_vwap_next = global_sell_vwap_t
                        hedge_cash_next = hedge_cash_t
                        hedge_inv_next = hedge_inv_t
                    mid_next = _mid_from_book_best_levels_batched(
                        ob_bids_next,
                        ob_asks_next,
                        mid_t,
                        spread_grid,
                    )
                    buy_vwap_book_next, sell_vwap_book_next = vwap_prices_batched(
                        ob_bids_next,
                        ob_asks_next,
                        mid_next,
                        spread_grid,
                        target_depth=vwap_depth,
                    )
                    buy_vwap_next = buy_vwap_book_next + eps_buy_next
                    sell_vwap_next = sell_vwap_book_next + eps_sell_next
                    
                    ob_bids, ob_asks = ob_bids_next, ob_asks_next

                    step_reward_main = _delta_wealth_mtm_batched(
                        sell_vwap_t, inv_t, cash_t, sell_vwap_next, inv_next, cash_next
                    )
                    hedge_reward = torch_mod.zeros_like(step_reward_main)
                    if self.global_enabled:
                        hedge_wealth_t = hedge_cash_t + hedge_inv_t * global_sell_vwap_t
                        hedge_wealth_next = hedge_cash_next + hedge_inv_next * global_sell_vwap_next
                        hedge_reward = (hedge_wealth_next - hedge_wealth_t).squeeze(1)
                    step_reward = step_reward_main + float(self.global_reward_weight) * hedge_reward
                    if h == 0:
                        step_reward_0 = step_reward
                    total_reward = total_reward + step_reward
                    total_main_reward = total_main_reward + step_reward_main
                    total_hedge_reward = total_hedge_reward + hedge_reward

                    mid_fallback_t = mid_next
                    inv_t = inv_next
                    cash_t = cash_next
                    global_buy_vwap_t = global_buy_vwap_next
                    global_sell_vwap_t = global_sell_vwap_next
                    hedge_cash_t = hedge_cash_next
                    hedge_inv_t = hedge_inv_next

                returns = total_reward / wealth_0
                J_mean_raw = torch_mod.nan_to_num(
                    total_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J_return_raw = torch_mod.nan_to_num(
                    returns.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J_mean_next_raw = torch_mod.nan_to_num(
                    step_reward_0.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                main_reward_mean_raw = torch_mod.nan_to_num(
                    total_main_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                global_reward_mean_raw = torch_mod.nan_to_num(
                    total_hedge_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J, components = terminal_objective(
                    returns,
                    inv_path,
                    return_risk_aversion_beta=self.return_risk_aversion_beta,
                    inventory_target_penalty_weight=self.inventory_target_penalty_weight,
                    target_inventory=obj_inventory_target,
                    inventory_scale=obj_inventory_scale,
                )

                def _finite_item(t) -> float:
                    try:
                        x = float(t.detach().item())
                        return x if math.isfinite(x) else 0.0
                    except Exception:
                        return 0.0

                obj_value = _finite_item(J)
                epoch_metrics = EpochObjectiveMetrics.from_tensors(
                    objective_value=obj_value,
                    components={k: _finite_item(v) for k, v in components.items()},
                    expected_total_reward=_finite_item(J_mean_raw),
                    expected_total_return=_finite_item(J_return_raw),
                    expected_next_reward=_finite_item(J_mean_next_raw),
                    main_reward_mean=_finite_item(main_reward_mean_raw),
                    global_reward_mean=_finite_item(global_reward_mean_raw),
                )
                self._last_J_components = dict(epoch_metrics.components)
                if not captured_initial_components:
                    self._last_J_components_initial = dict(epoch_metrics.components)
                    self._last_main_reward_mean_initial = epoch_metrics.main_reward_mean
                    self._last_global_reward_mean_initial = epoch_metrics.global_reward_mean
                    initial_metrics = epoch_metrics
                    captured_initial_components = True
                final_metrics = epoch_metrics
                self._last_J_mean_raw = epoch_metrics.expected_total_reward
                self._last_E_dW_next = epoch_metrics.expected_next_reward
                self._last_main_reward_mean = epoch_metrics.main_reward_mean
                self._last_global_reward_mean = epoch_metrics.global_reward_mean
                loss = -J
                if not torch_mod.isfinite(loss):
                    nonfinite_loss_count += 1
                if torch_mod.isfinite(loss):
                    loss.backward()
                    if x_torch.grad is not None:
                        if not torch_mod.isfinite(x_torch.grad).all():
                            nonfinite_grad_count += 1
                        torch_mod.nn.utils.clip_grad_norm_([x_torch], self.adam_grad_clip)
                return loss

            loss = optimizer.step(closure)
            if loss is not None and torch_mod.isfinite(loss) and loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x_torch.detach().clone()
                best_epoch = epoch
                if final_metrics is not None:
                    best_metrics = final_metrics
                best_J_mean = final_metrics.expected_total_reward if final_metrics else 0.0
                best_J_mean_next = final_metrics.expected_next_reward if final_metrics else 0.0
                best_main_reward_mean = final_metrics.main_reward_mean if final_metrics else 0.0
                best_global_reward_mean = final_metrics.global_reward_mean if final_metrics else 0.0

        loop_elapsed = time.perf_counter() - t0
        if self.fixed_mc_samples:
            train_time_sec = loop_elapsed
            total_sec = batch_time_sec + loop_elapsed
        else:
            train_time_sec = loop_elapsed - batch_time_sec
            total_sec = loop_elapsed
        best_flat = best_x.cpu().numpy()
        best_flat_slice = best_flat[:single_len]
        best_action = self._flat_to_effective_action(best_flat_slice)
        final_flat_slice = np.array(x_torch.detach().cpu().numpy()[:single_len], dtype=np.float64)
        final_action = self._flat_to_effective_action(final_flat_slice)
        self._last_sigmoid_action = best_action
        if best_J_mean is not None and math.isfinite(float(best_J_mean)):
            self._last_E_dW = float(best_J_mean)
        else:
            bl = -float(best_loss)
            self._last_E_dW = bl if math.isfinite(bl) else 0.0
        if best_J_mean_next is not None and math.isfinite(float(best_J_mean_next)):
            self._last_E_dW_next = float(best_J_mean_next)
        if best_main_reward_mean is not None and math.isfinite(float(best_main_reward_mean)):
            self._last_main_reward_mean = float(best_main_reward_mean)
        if best_global_reward_mean is not None and math.isfinite(float(best_global_reward_mean)):
            self._last_global_reward_mean = float(best_global_reward_mean)
        if self.warm_start:
            self._last_action = best_action
            self._last_raw_flat = np.array(best_flat_slice, dtype=np.float64)
        if best_metrics is None and final_metrics is not None:
            best_metrics = final_metrics
        if initial_metrics is None and final_metrics is not None:
            initial_metrics = final_metrics
        diagnostics = OptimizerRunDiagnostics(
            config={
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "adam_epochs": self.adam_epochs,
                "fixed_mc_samples": self.fixed_mc_samples,
                "spread_max": self.spread_max,
                "spread_step": self.spread_step,
                "soft_fill_temp": self.soft_fill_temp,
                "soft_fill_activation": self.soft_fill_activation,
                "return_risk_aversion_beta": self.return_risk_aversion_beta,
                "inventory_target_penalty_weight": self.inventory_target_penalty_weight,
                "target_inventory": obj_inventory_target,
                "target_inventory_multiplier": self.target_inventory_multiplier,
                "quote_size_min_per_side": self.quote_size_min_per_side,
                "quote_size_max_per_side": self.quote_size_max_per_side,
                "quote_spread_max": self.quote_spread_max,
                "quote_spread_min_depth": self.quote_spread_min_depth,
            },
            timing={
                "sample_draw_sec": float(batch_time_sec),
                "train_sec": float(train_time_sec),
                "total_sec": float(total_sec),
                "best_epoch": best_epoch,
            },
            initial=initial_metrics,
            final=final_metrics,
            best=best_metrics,
            best_epoch=best_epoch,
            action_initial=action_initial_snap,
            action_final=build_action_snapshot(final_action, mid=mid_0, spread_levels=spread_levels),
            action_best=build_action_snapshot(best_action, mid=mid_0, spread_levels=spread_levels),
            gradients={
                "nonfinite_loss_count": nonfinite_loss_count,
                "nonfinite_grad_count": nonfinite_grad_count,
            },
            rollout={
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "global_enabled": self.global_enabled,
                "soft_fill_temp": self.soft_fill_temp,
                "soft_fill_activation": self.soft_fill_activation,
            },
        )
        self._last_diagnostics = diagnostics
        log_event(
            "mpc_optimizer_timing",
            {
                "batch_sec": batch_time_sec,
                "train_sec": train_time_sec,
                "total_sec": total_sec,
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "best_epoch": best_epoch,
            },
        )
        return best_action, diagnostics

    def reset_warm_start(self) -> None:
        self._last_action = None
        self._x_torch = None
        self._optimizer = None
