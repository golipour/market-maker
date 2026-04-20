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

from .state import State, Action, OrderbookRelative, SigmoidMixtureAction, sigmoid_mixture_action_bounds
from .state_tensor import (
    depth_bid_tensor,
    depth_ask_tensor,
)
from .reward_tensor import terminal_objective


# -----------------------------------------------------------------------------
# MPCOptimizer protocol (controller and package surface)
# -----------------------------------------------------------------------------


@runtime_checkable
class MPCOptimizer(Protocol):
    """Given state, return optimal action. Estimators provided at construction."""

    def optimize(self, state: State) -> Action:
        ...


# -----------------------------------------------------------------------------
# Helpers: normalization, scaling, bounds, total_q projection, book cumulative
# -----------------------------------------------------------------------------


def _get_torch():
    return torch


def _normalization_volume_scale(state: State) -> float:
    """Volume scale from order book for numerical stability."""
    total_book_vol = sum(q for (_, q) in state.ob_rel.bids) + sum(q for (_, q) in state.ob_rel.asks)
    return max(1.0, total_book_vol / 1000.0)


def _scale_flat_params_to_normalized(flat: np.ndarray, volume_scale: float) -> np.ndarray:
    """Scale L components (param index 0 in each 3-tuple) to normalized space."""
    out = np.array(flat, dtype=np.float64, copy=True)
    for i in range(len(out)):
        if i % 3 == 0:
            out[i] /= volume_scale
    return out


def _unscale_flat_params_from_normalized(flat: np.ndarray, volume_scale: float) -> np.ndarray:
    """Unscale L components from normalized space back to original scale."""
    out = np.array(flat, dtype=np.float64, copy=True)
    for i in range(len(out)):
        if i % 3 == 0:
            out[i] *= volume_scale
    return out


def _scale_bounds_for_normalized(
    bounds: list[tuple[float, float]], volume_scale: float
) -> list[tuple[float, float]]:
    """Scale L bounds (first of each 3-tuple) for optimization in normalized space."""
    result = []
    for i, (lo, hi) in enumerate(bounds):
        if i % 3 == 0:
            result.append((lo / volume_scale, hi / volume_scale))
        else:
            result.append((lo, hi))
    return result


def _project_total_q_torch(
    x_torch: Any,
    single_len: int,
    n_components: int,
    total_q_min: float | None,
    total_q_max: float | None,
    torch_mod: Any,
) -> None:
    """In-place project L so sum(bid_L) and sum(ask_L) in [total_q_min, total_q_max]."""
    if total_q_min is None or total_q_max is None:
        return
    total_q_min_f = float(total_q_min)
    total_q_max_f = float(total_q_max)
    n = n_components
    num_slices = x_torch.numel() // single_len
    for start in range(0, num_slices * single_len, single_len):
        bid_L_sum = x_torch[start + 0] * 0.0
        for i in range(n):
            bid_L_sum = bid_L_sum + x_torch[start + 3 * i]
        eps = 1e-12
        bid_L_sum = bid_L_sum + eps
        target_bid = torch_mod.clamp(bid_L_sum, total_q_min_f, total_q_max_f)
        factor_bid = target_bid / bid_L_sum
        for i in range(n):
            x_torch.data[start + 3 * i] = x_torch.data[start + 3 * i] * factor_bid
        ask_L_sum = x_torch[start + 3 * n] * 0.0
        for i in range(n):
            ask_L_sum = ask_L_sum + x_torch[start + 3 * n + 3 * i]
        ask_L_sum = ask_L_sum + eps
        target_ask = torch_mod.clamp(ask_L_sum, total_q_min_f, total_q_max_f)
        factor_ask = target_ask / ask_L_sum
        for i in range(n):
            x_torch.data[start + 3 * n + 3 * i] = x_torch.data[start + 3 * n + 3 * i] * factor_ask


def _project_total_q_flat(
    flat: np.ndarray,
    n_components: int,
    total_q_min: float | None,
    total_q_max: float | None,
) -> np.ndarray:
    """Project flat param vector so sum(bid_L) and sum(ask_L) in [total_q_min, total_q_max]."""
    if total_q_min is None or total_q_max is None:
        return flat
    out = np.array(flat, dtype=np.float64, copy=True)
    n = n_components
    bid_L_sum = sum(out[3 * i] for i in range(n)) + 1e-12
    target_bid = np.clip(bid_L_sum, total_q_min, total_q_max)
    for i in range(n):
        out[3 * i] *= target_bid / bid_L_sum
    ask_L_sum = sum(out[3 * n + 3 * i] for i in range(n)) + 1e-12
    target_ask = np.clip(ask_L_sum, total_q_min, total_q_max)
    for i in range(n):
        out[3 * n + 3 * i] *= target_ask / ask_L_sum
    return out


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


def _book_cumulative_from_ob_rel(
    ob_rel: OrderbookRelative,
    n_levels: int,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """Build (bid_cum, ask_cum) tensors of length n_levels: cumulative qty at spread 1..n_levels."""
    torch_mod = _get_torch()
    bid_cum = torch_mod.zeros(n_levels, dtype=dtype, device=device)
    ask_cum = torch_mod.zeros(n_levels, dtype=dtype, device=device)
    b_sum = 0.0
    for (spread_ticks, qty) in sorted(ob_rel.bids, key=lambda x: x[0]):
        idx = int(round(spread_ticks)) - 1
        if 0 <= idx < n_levels:
            b_sum += float(qty)
            bid_cum[idx] = b_sum
    for i in range(1, n_levels):
        if bid_cum[i].item() == 0:
            bid_cum[i] = bid_cum[i - 1]
    a_sum = 0.0
    for (spread_ticks, qty) in sorted(ob_rel.asks, key=lambda x: x[0]):
        idx = int(round(spread_ticks)) - 1
        if 0 <= idx < n_levels:
            a_sum += float(qty)
            ask_cum[idx] = a_sum
    for i in range(1, n_levels):
        if ask_cum[i].item() == 0:
            ask_cum[i] = ask_cum[i - 1]
    return bid_cum, ask_cum


def _book_cumulative_from_arrays(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    n_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched: ob_bids (n_samples, n_levels) -> cum (n_samples, n_grid)."""
    n_levels = ob_bids.shape[1]
    k = min(n_grid, n_levels)
    cum_b = torch.cumsum(ob_bids[:, :k], dim=1)
    cum_a = torch.cumsum(ob_asks[:, :k], dim=1)
    if k < n_grid:
        cum_b = torch.nn.functional.pad(cum_b, (0, n_grid - k), value=cum_b[:, -1:])
        cum_a = torch.nn.functional.pad(cum_a, (0, n_grid - k), value=cum_a[:, -1:])
    return cum_b, cum_a


def _consume_mof_only_torch(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    buy_vol: torch.Tensor,
    sell_vol: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MOF consumption only (no LOF residual) to level volumes."""
    cum_bid = torch.cumsum(ob_bids.float(), dim=1)
    cum_ask = torch.cumsum(ob_asks.float(), dim=1)
    sell_vol = sell_vol.clamp(min=0.0)
    buy_vol = buy_vol.clamp(min=0.0)
    consumed_b = torch.minimum(sell_vol, cum_bid)
    consumed_a = torch.minimum(buy_vol, cum_ask)
    new_cum_bid = (cum_bid - consumed_b).clamp(min=0.0)
    new_cum_ask = (cum_ask - consumed_a).clamp(min=0.0)
    ob_bids_after = torch.cat(
        [new_cum_bid[:, :1], new_cum_bid[:, 1:] - new_cum_bid[:, :-1]],
        dim=1,
    ).clamp(min=0.0)
    ob_asks_after = torch.cat(
        [new_cum_ask[:, :1], new_cum_ask[:, 1:] - new_cum_ask[:, :-1]],
        dim=1,
    ).clamp(min=0.0)
    return ob_bids_after, ob_asks_after


def _representative_prices_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
    target_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VWAP representative buy/sell prices to shared depth from level volumes."""
    eps = 1e-12
    cum_bid = torch.cumsum(ob_bids, dim=1)
    cum_ask = torch.cumsum(ob_asks, dim=1)
    total_bid = cum_bid[:, -1:]
    total_ask = cum_ask[:, -1:]
    depth_used = torch.minimum(torch.minimum(total_bid, total_ask), torch.full_like(total_bid, float(target_depth)))
    depth_used_safe = torch.clamp(depth_used, min=eps)

    cum_bid_prev = torch.cat([torch.zeros_like(cum_bid[:, :1]), cum_bid[:, :-1]], dim=1)
    cum_ask_prev = torch.cat([torch.zeros_like(cum_ask[:, :1]), cum_ask[:, :-1]], dim=1)
    take_bid = torch.clamp(depth_used - cum_bid_prev, min=0.0)
    take_bid = torch.minimum(take_bid, ob_bids)
    take_ask = torch.clamp(depth_used - cum_ask_prev, min=0.0)
    take_ask = torch.minimum(take_ask, ob_asks)

    bid_prices = center_mid - spread_grid.unsqueeze(0)
    ask_prices = center_mid + spread_grid.unsqueeze(0)
    sell_rep = (take_bid * bid_prices).sum(dim=1, keepdim=True) / depth_used_safe
    buy_rep = (take_ask * ask_prices).sum(dim=1, keepdim=True) / depth_used_safe

    no_depth = depth_used <= eps
    sell_rep = torch.where(no_depth, center_mid, sell_rep)
    buy_rep = torch.where(no_depth, center_mid, buy_rep)
    return buy_rep, sell_rep


def _mid_from_book_best_levels_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Mid implied by post-flow best bid/ask from level volumes.
    center_mid anchors absolute prices; spread_grid is 1..n_levels.
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


def _fill_and_cash_batched(
    params: torch.Tensor,
    n_components: int,
    sell_ref: torch.Tensor,
    buy_ref: torch.Tensor,
    book_cum_bid: torch.Tensor,
    book_cum_ask: torch.Tensor,
    sell_vol: torch.Tensor,
    buy_vol: torch.Tensor,
    spread_grid: torch.Tensor,
    maker_fee: float,
    soft_temp: float = 1.0,
    use_softplus: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched fill/cash: mid, book_cum_* (n_samples, n_grid), sell_vol, buy_vol (n_samples, 1).
    Returns fill_bid, fill_ask, cash_delta, inv_delta (n_samples,) or (n_samples, 1).
    """
    depth_bid = depth_bid_tensor(params, spread_grid, n_components, use_softplus=use_softplus)
    depth_ask = depth_ask_tensor(params, spread_grid, n_components, use_softplus=use_softplus)
    reach_bid = (sell_vol - book_cum_bid).clamp(min=0.0)
    reach_ask = (buy_vol - book_cum_ask).clamp(min=0.0)
    fill_bid_level = depth_bid.unsqueeze(0) * torch.sigmoid(reach_bid / soft_temp)
    fill_ask_level = depth_ask.unsqueeze(0) * torch.sigmoid(reach_ask / soft_temp)
    fill_bid = fill_bid_level.sum(dim=1)
    fill_ask = fill_ask_level.sum(dim=1)
    sell_exp = sell_ref.expand(-1, spread_grid.shape[0])
    buy_exp = buy_ref.expand(-1, spread_grid.shape[0])
    cash_bid = (fill_bid_level * (sell_exp - spread_grid.unsqueeze(0))).sum(dim=1)
    cash_ask = (fill_ask_level * (buy_exp + spread_grid.unsqueeze(0))).sum(dim=1)
    one_minus_f = 1.0 - maker_fee
    cash_delta = cash_ask * one_minus_f - cash_bid
    inv_delta = fill_bid * one_minus_f - fill_ask
    return fill_bid, fill_ask, cash_delta, inv_delta


def _action_entropy_from_flat_params(
    params: torch.Tensor,
    n_components: int,
    *,
    use_softplus: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Entropy regularizer from bid/ask action size distributions.
    Uses L parameters per side, normalizes to probabilities, and returns
    normalized entropy in [0, 1] averaged over bid and ask.
    """
    if n_components <= 1:
        return params.sum() * 0.0
    n = int(n_components)
    bid_L = params[0 : 3 * n : 3]
    ask_L = params[3 * n : 6 * n : 3]
    if use_softplus:
        bid_w = torch.nn.functional.softplus(bid_L)
        ask_w = torch.nn.functional.softplus(ask_L)
    else:
        bid_w = torch.clamp(bid_L, min=eps)
        ask_w = torch.clamp(ask_L, min=eps)
    bid_p = bid_w / torch.clamp(bid_w.sum(), min=eps)
    ask_p = ask_w / torch.clamp(ask_w.sum(), min=eps)
    bid_H = -(bid_p * torch.log(torch.clamp(bid_p, min=eps))).sum()
    ask_H = -(ask_p * torch.log(torch.clamp(ask_p, min=eps))).sum()
    norm = math.log(float(n))
    return 0.5 * (bid_H + ask_H) / max(norm, eps)


def _delta_wealth_mtm_batched(
    sell_ref: torch.Tensor,
    inv: torch.Tensor,
    cash: torch.Tensor,
    sell_ref_next: torch.Tensor,
    inv_next: torch.Tensor,
    cash_next: torch.Tensor,
) -> torch.Tensor:
    """Per-sample ΔW with inventory marked at sell-side ref: W = cash + inv * sell_ref."""
    W = cash + inv * sell_ref
    W_next = cash_next + inv_next * sell_ref_next
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
        n_components: int = 2,
        n_samples: int = 32,
        horizon: int = 4,
        spread_max: float = 100.0,
        n_grid: int = 51,
        ob_levels: int = 100,
        soft_fill_temp: float = 1.0,
        adam_lr: float = 0.05,
        adam_epochs: int = 80,
        adam_grad_clip: float = 10.0,
        device: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        warm_start: bool = False,
        adam_normalize_inputs: bool = False,
        total_q_min: float | int | None = None,
        total_q_max: float | int | None = None,
        fixed_mc_samples: bool = False,
        verbose: bool = False,
        inv_coef: float = 0.0,
        target_inv: float = 0.0,
        risk_coef: float = 0.0,
        action_coef: float = 0.0,
    ):
        self.price_model = price_model
        self.market_flow_model = market_flow_model
        self.execution_model = execution_model
        self.n_components = max(1, n_components)
        self.n_samples = max(1, n_samples)
        self.horizon = max(1, horizon)
        self.spread_max = float(spread_max)
        self.n_grid = max(2, n_grid)
        self.ob_levels = max(1, ob_levels)
        self.soft_fill_temp = soft_fill_temp
        self.adam_lr = adam_lr
        self.adam_epochs = adam_epochs
        self.adam_grad_clip = adam_grad_clip
        self.device = device
        self.warm_start = warm_start
        self.adam_normalize_inputs = adam_normalize_inputs
        self.total_q_min = float(total_q_min) if total_q_min is not None else None
        self.total_q_max = float(total_q_max) if total_q_max is not None else None
        self.fixed_mc_samples = fixed_mc_samples
        self.verbose = verbose
        self.inv_coef = max(0.0, float(inv_coef))
        self.target_inv = float(target_inv)
        self.risk_coef = max(0.0, float(risk_coef))
        self.action_coef = max(0.0, float(action_coef))
        self._last_action: SigmoidMixtureAction | None = None
        self._last_J_components: dict | None = None
        self._last_sigmoid_action: SigmoidMixtureAction | None = None
        self._x_torch: Any = None
        self._optimizer: Any = None

        if bounds is None:
            L_max = 1e5
            if self.total_q_max is not None:
                L_max = max(L_max, float(self.total_q_max))
            self.bounds = sigmoid_mixture_action_bounds(
                n_components=self.n_components,
                L_max=L_max,
                k_max=100.0,
                x0_max=self.spread_max,
            )
        else:
            self.bounds = bounds

    def optimize(self, state: State) -> SigmoidMixtureAction:
        torch_mod = _get_torch()
        if self.device is None or (isinstance(self.device, str) and self.device.lower() == "cuda"):
            _device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        else:
            _device = torch_mod.device("cpu" if (isinstance(self.device, str) and self.device.lower() == "cpu") else self.device)
        dtype = torch_mod.float64

        book_cum_bid, book_cum_ask = _book_cumulative_from_ob_rel(
            state.ob_rel, self.n_grid, _device, dtype
        )
        spread_grid = torch_mod.arange(1.0, self.n_grid + 1.0, dtype=dtype, device=_device)
        mid_0 = float(state.mid)
        inv_0 = float(state.inventory)
        cash_0 = float(state.cash)

        ob_bids_0 = torch_mod.zeros(self.n_samples, self.ob_levels, dtype=dtype, device=_device)
        ob_asks_0 = torch_mod.zeros(self.n_samples, self.ob_levels, dtype=dtype, device=_device)
        for (s, q) in state.ob_rel.bids:
            i = int(round(s)) - 1
            if 0 <= i < self.ob_levels:
                ob_bids_0[:, i] += float(q)
        for (s, q) in state.ob_rel.asks:
            i = int(round(s)) - 1
            if 0 <= i < self.ob_levels:
                ob_asks_0[:, i] += float(q)
        n = self.n_components
        single_len = 6 * n
        param_len = self.horizon * single_len

        volume_scale = 1.0
        if self.adam_normalize_inputs:
            volume_scale = _normalization_volume_scale(state)

        if self.warm_start and self._last_action is not None:
            one_action = np.array(self._last_action.to_flat(), dtype=np.float64)
        else:
            sm = self.spread_max
            bid_part = []
            ask_part = []
            for i in range(n):
                x0_i = 1.0 + (sm - 1.0) * (i + 1) / (n + 1)
                bid_part.extend([10.0 + 10.0 * i, 0.5 - 0.1 * i, x0_i])
                ask_part.extend([10.0 + 10.0 * i, 0.5 - 0.1 * i, x0_i])
            one_action = np.array(bid_part + ask_part, dtype=np.float64)
        if self.adam_normalize_inputs:
            one_action = _scale_flat_params_to_normalized(one_action, volume_scale)
        x0_arr = np.tile(one_action, self.horizon)

        bounds_used = (
            _scale_bounds_for_normalized(self.bounds, volume_scale)
            if self.adam_normalize_inputs
            else self.bounds
        )
        bounds_full = bounds_used * self.horizon
        lower = torch_mod.tensor([b[0] for b in bounds_full], dtype=dtype, device=_device)
        upper = torch_mod.tensor([b[1] for b in bounds_full], dtype=dtype, device=_device)

        if self._x_torch is None or self._x_torch.numel() != param_len:
            self._x_torch = torch_mod.tensor(x0_arr, dtype=dtype, device=_device, requires_grad=True)
            self._optimizer = torch_mod.optim.Adam([self._x_torch], lr=self.adam_lr)
        else:
            with torch_mod.no_grad():
                self._x_torch.data = torch_mod.tensor(x0_arr, dtype=dtype, device=_device)
        x_torch = self._x_torch
        optimizer = self._optimizer
        maker_fee = getattr(self.execution_model, "maker_fee", 0.0)
        spread_grid_rep = torch_mod.arange(1.0, float(self.ob_levels) + 1.0, dtype=dtype, device=_device)
        target_depth = float(getattr(self.price_model, "_target_depth_lots", 1000.0))

        # Pre-draw MC samples: (n_samples, horizon) for MOF and price residuals.
        def draw_samples():
            buy_vol = torch_mod.zeros(self.n_samples, self.horizon, dtype=dtype, device=_device)
            sell_vol = torch_mod.zeros(self.n_samples, self.horizon, dtype=dtype, device=_device)
            eps_buy = torch_mod.zeros(self.n_samples, self.horizon, dtype=dtype, device=_device)
            eps_sell = torch_mod.zeros(self.n_samples, self.horizon, dtype=dtype, device=_device)
            for s in range(self.n_samples):
                for h in range(self.horizon):
                    bv, sv = self.market_flow_model.sample((1,), device=_device, dtype=dtype)
                    buy_vol[s, h] = bv.squeeze()
                    sell_vol[s, h] = sv.squeeze()
                    eb, es = self.price_model.sample_residual((), device=_device, dtype=dtype)
                    eps_buy[s, h] = eb.squeeze()
                    eps_sell[s, h] = es.squeeze()
            return buy_vol, sell_vol, eps_buy, eps_sell

        batch_time_sec = 0.0
        if self.fixed_mc_samples:
            t_batch = time.perf_counter()
            fixed_buy_vol, fixed_sell_vol, fixed_eps_buy, fixed_eps_sell = draw_samples()
            batch_time_sec = time.perf_counter() - t_batch

        best_loss = float("inf")
        best_x = x_torch.detach().clone()
        best_J_mean: float | None = None  # E[ΔW] for the best action (for E_dW plot)
        t0 = time.perf_counter()

        for epoch in range(self.adam_epochs):
            if self.verbose:
                cur_flat = x_torch.detach().cpu().numpy()[:single_len]
                if self.adam_normalize_inputs:
                    cur_flat = _unscale_flat_params_from_normalized(cur_flat, volume_scale)
                cur_flat = _project_total_q_flat(cur_flat, n, self.total_q_min, self.total_q_max)
                print(f"  Epoch {epoch}: action  {_format_action_verbose(cur_flat, n)}")

            def closure():
                nonlocal batch_time_sec
                optimizer.zero_grad()
                with torch_mod.no_grad():
                    x_torch.data = torch_mod.clamp(x_torch.data, lower, upper)
                    _project_total_q_torch(
                        x_torch, single_len, n,
                        self.total_q_min, self.total_q_max, torch_mod,
                    )

                if self.fixed_mc_samples:
                    buy_vol, sell_vol, eps_buy, eps_sell = (
                        fixed_buy_vol,
                        fixed_sell_vol,
                        fixed_eps_buy,
                        fixed_eps_sell,
                    )
                else:
                    t_batch = time.perf_counter()
                    buy_vol, sell_vol, eps_buy, eps_sell = draw_samples()
                    batch_time_sec += time.perf_counter() - t_batch

                inv_t = torch_mod.full((self.n_samples, 1), inv_0, dtype=dtype, device=_device)
                cash_t = torch_mod.full((self.n_samples, 1), cash_0, dtype=dtype, device=_device)
                ob_bids = ob_bids_0.clone()
                ob_asks = ob_asks_0.clone()
                mid_anchor_t = torch_mod.full((self.n_samples, 1), mid_0, dtype=dtype, device=_device)

                book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(self.n_samples, -1)
                book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(self.n_samples, -1)

                total_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                action_entropy_sum = torch_mod.zeros((), dtype=dtype, device=_device)

                for h in range(self.horizon):
                    if h == 0:
                        book_cum_bid_h = book_cum_bid_0
                        book_cum_ask_h = book_cum_ask_0
                    else:
                        book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks, self.n_grid)

                    action_h = x_torch[h * single_len : (h + 1) * single_len]
                    action_entropy_sum = action_entropy_sum + _action_entropy_from_flat_params(
                        action_h, n, use_softplus=True
                    )
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
                        mid_anchor_t,
                        spread_grid_rep,
                    )
                    buy_cmp_t, sell_cmp_t = _representative_prices_batched(
                        ob_bids,
                        ob_asks,
                        mid_t,
                        spread_grid_rep,
                        target_depth=target_depth,
                    )
                    buy_ref_t = buy_cmp_t + eps_buy_h
                    sell_ref_t = sell_cmp_t + eps_sell_h

                    fill_bid, fill_ask, cash_delta, inv_delta = _fill_and_cash_batched(
                        action_h,
                        n,
                        sell_ref_t,
                        buy_ref_t,
                        book_cum_bid_h,
                        book_cum_ask_h,
                        sell_h,
                        buy_h,
                        spread_grid,
                        maker_fee,
                        soft_temp=self.soft_fill_temp,
                        use_softplus=True,
                    )
                    inv_next = inv_t + inv_delta.unsqueeze(1)
                    cash_next = cash_t + cash_delta.unsqueeze(1)
                    ob_bids_next, ob_asks_next = _consume_mof_only_torch(
                        ob_bids, ob_asks, buy_h, sell_h
                    )
                    mid_next = _mid_from_book_best_levels_batched(
                        ob_bids_next,
                        ob_asks_next,
                        mid_t,
                        spread_grid_rep,
                    )
                    buy_cmp_h, sell_cmp_h = _representative_prices_batched(
                        ob_bids_next,
                        ob_asks_next,
                        mid_next,
                        spread_grid_rep,
                        target_depth=target_depth,
                    )
                    buy_ref_next = buy_cmp_h + eps_buy_next
                    sell_ref_next = sell_cmp_h + eps_sell_next
                    
                    ob_bids, ob_asks = ob_bids_next, ob_asks_next

                    step_reward = _delta_wealth_mtm_batched(
                        sell_ref_t, inv_t, cash_t, sell_ref_next, inv_next, cash_next
                    )
                    total_reward = total_reward + step_reward

                    mid_anchor_t = mid_next
                    inv_t = inv_next
                    cash_t = cash_next

                # total_reward = ΔW per trajectory (same as W(s_{t+H}) - W(s_t)); inv_t = terminal inventory
                action_entropy_mean = action_entropy_sum / float(self.horizon)
                J, components = terminal_objective(
                    total_reward,
                    inv_t,
                    inv_coef=self.inv_coef,
                    target_inv=self.target_inv,
                    risk_coef=self.risk_coef,
                    action_entropy=action_entropy_mean,
                    action_coef=self.action_coef,
                )

                def _finite_item(t) -> float:
                    try:
                        x = float(t.detach().item())
                        return x if math.isfinite(x) else 0.0
                    except Exception:
                        return 0.0

                self._last_J_components = {k: _finite_item(v) for k, v in components.items()}
                loss = -J
                if torch_mod.isfinite(loss):
                    loss.backward()
                    if x_torch.grad is not None:
                        torch_mod.nn.utils.clip_grad_norm_([x_torch], self.adam_grad_clip)
                return loss

            loss = optimizer.step(closure)
            if loss is not None and torch_mod.isfinite(loss) and loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x_torch.detach().clone()
                comp = self._last_J_components
                if comp is not None and "J_mean" in comp:
                    jm = float(comp["J_mean"])
                    best_J_mean = jm if math.isfinite(jm) else 0.0

        loop_elapsed = time.perf_counter() - t0
        if self.fixed_mc_samples:
            train_time_sec = loop_elapsed
            total_sec = batch_time_sec + loop_elapsed
        else:
            train_time_sec = loop_elapsed - batch_time_sec
            total_sec = loop_elapsed
        best_flat = best_x.cpu().numpy()
        best_flat_slice = best_flat[:single_len]
        if self.adam_normalize_inputs:
            best_flat_slice = _unscale_flat_params_from_normalized(best_flat_slice, volume_scale)
        best_flat_slice = _project_total_q_flat(
            best_flat_slice, n, self.total_q_min, self.total_q_max,
        )
        best_action = SigmoidMixtureAction.from_flat_positive(best_flat_slice, n_components=self.n_components)
        self._last_sigmoid_action = best_action
        # E_dW plot shows E[ΔW] (J_mean) for the best action, not the full objective J
        if best_J_mean is not None and math.isfinite(float(best_J_mean)):
            self._last_E_dW = float(best_J_mean)
        else:
            bl = -float(best_loss)
            self._last_E_dW = bl if math.isfinite(bl) else 0.0
        if self.warm_start:
            self._last_action = best_action
        print(
            f"batch={batch_time_sec:.3f}s train={train_time_sec:.3f}s total={total_sec:.3f}s "
            f"MC batched (n={self.n_samples}, H={self.horizon})"
        )
        return best_action

    def reset_warm_start(self) -> None:
        self._last_action = None
        self._x_torch = None
        self._optimizer = None
