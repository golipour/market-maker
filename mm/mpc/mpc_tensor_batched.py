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

from mm.mpc.l_param import LParamConfig, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import State, Action, OrderbookRelative, SigmoidMixtureAction, sigmoid_mixture_action_bounds
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


def _bounds_effective_to_raw(
    bounds: list[tuple[float, float]],
    *,
    l_softmax: bool = False,
) -> list[tuple[float, float]]:
    """
    Convert optimizer bounds from effective parameter space to raw tensor space.

    L and k (index 0/1 of each 3-tuple) are represented in raw space and mapped via
    softplus during model evaluation, so their bounds must be inverse-softplus mapped.
    When ``l_softmax`` is true, L slots are unconstrained logits and bounds pass through.
    x0 (index 2) is used directly and left unchanged.
    """
    out: list[tuple[float, float]] = []
    for i, (lo, hi) in enumerate(bounds):
        if i % 3 == 0 and l_softmax:
            out.append((float(lo), float(hi)))
        elif i % 3 in (0, 1):  # L (softplus) or k
            out.append((_inv_softplus_np(float(lo)), _inv_softplus_np(float(hi))))
        else:  # x0
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
    quote_size_target_per_side: float | None,
    *,
    use_softmax_L: bool = False,
) -> np.ndarray | None:
    """
    Build an initial flat action (raw optimizer params) using a simple linear-range
    sigmoid approximation over s in [0..N], N=round(spread_max).

    We place x0 near N/2 and choose a small k so most of [0..N] lies in the near-linear
    part of sigmoid. Then pick L so baseline-shift marginal is approximately uniform:
      m(s) = D(s)-D(s-1) ~= quote_size_target_per_side / N.

    Finally we rescale L for exact baseline-shift total:
      D(N) - D(0) = quote_size_target_per_side.
    """
    if quote_size_target_per_side is None or float(quote_size_target_per_side) <= 0.0:
        return None
    n = max(1, int(n_components))
    sm = max(1, int(round(float(spread_max))))
    q_level = float(quote_size_target_per_side) / float(sm)
    x0_center = 0.5 * float(sm)
    # Keep each component centered in the same linear region; tiny jitter avoids exact symmetry.
    if n == 1:
        x0 = np.array([x0_center], dtype=np.float64)
    else:
        x0 = np.linspace(x0_center - 0.01 * sm, x0_center + 0.01 * sm, n, dtype=np.float64)

    # Linear-range choice: keep |k*(s-x0)| relatively small on [0..N].
    k_base = max(1e-3, min(0.05, 2.0 / max(1.0, float(sm))))
    k_eff = np.full(n, k_base, dtype=np.float64)

    # Linearized slope: sigmoid'(z) ~ 1/4 around z~0 => marginal ~ sum_i(L_i*k_i/4).
    L_total = max(1e-6, 4.0 * q_level / max(k_base, 1e-12))
    L_eff = np.full(n, max(1e-6, L_total / float(n)), dtype=np.float64)

    # Final exact normalization on D(N)-D(0) (discretizer convention total).
    s_hi = 1.0 / (1.0 + np.exp(-(k_eff * (float(sm) - x0))))
    s_lo = 1.0 / (1.0 + np.exp(-(k_eff * (0.0 - x0))))
    d_n = float(np.sum(L_eff * (s_hi - s_lo)))
    if d_n > 1e-12 and not use_softmax_L:
        L_eff *= float(quote_size_target_per_side) / d_n

    if use_softmax_L:
        L_raw = np.zeros(n, dtype=np.float64)
    else:
        L_raw = np.array([_inv_softplus_np(v) for v in L_eff], dtype=np.float64)
    k_raw = np.array([_inv_softplus_np(v) for v in k_eff], dtype=np.float64)

    bid_part: list[float] = []
    ask_part: list[float] = []
    for i in range(n):
        bid_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0[i])])
        ask_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0[i])])
    return np.array(bid_part + ask_part, dtype=np.float64)


def _clamp_quote_size_per_side(
    cur_q: float,
    *,
    quote_size_min_per_side: float | None,
    quote_size_max_per_side: float | None,
) -> float:
    lo = float(quote_size_min_per_side) if quote_size_min_per_side is not None else -float("inf")
    hi = float(quote_size_max_per_side) if quote_size_max_per_side is not None else float("inf")
    return float(min(max(cur_q, lo), hi))


def _desired_quote_size_per_side(
    cur_q: float,
    *,
    quote_size_target_per_side: float | None,
    hard_project_quote_size_target: bool,
    quote_size_min_per_side: float | None,
    quote_size_max_per_side: float | None,
    hard_project_quote_size_range: bool,
) -> float | None:
    """Option A: hard target projection first, then hard range projection."""

    eps = 1e-12
    if cur_q <= eps:
        return None
    use_target = bool(
        hard_project_quote_size_target
        and quote_size_target_per_side is not None
        and float(quote_size_target_per_side) > 0.0
    )
    has_range = (quote_size_min_per_side is not None) or (quote_size_max_per_side is not None)
    use_range = bool(hard_project_quote_size_range and has_range)
    if not use_target and not use_range:
        return None

    q = cur_q
    if use_target:
        q = float(quote_size_target_per_side)
    if use_range:
        q = _clamp_quote_size_per_side(
            q,
            quote_size_min_per_side=quote_size_min_per_side,
            quote_size_max_per_side=quote_size_max_per_side,
        )
    if abs(q - cur_q) <= eps * max(1.0, abs(cur_q)):
        return None
    return q


def _project_quote_size_torch(
    x_torch: Any,
    single_len: int,
    n_components: int,
    *,
    quote_size_target_per_side: float | None,
    hard_project_quote_size_target: bool,
    quote_size_min_per_side: float | None,
    quote_size_max_per_side: float | None,
    hard_project_quote_size_range: bool,
    spread_max: float,
    torch_mod: Any,
) -> None:
    """
    In-place project L by quoted size over discretizer range s=0..N (N=spread_max),
    under baseline-shift convention Q_range = D(N) - D(0).

    Target and range projection are independent toggles. When both are enabled, target
    projection runs first and range projection clamps the result.
    """
    spread_hi = max(0.0, float(spread_max))
    spread_lo = 0.0
    eps = 1e-12
    n = n_components

    def _effective_qrange(side_off: int, start: int) -> float:
        q = 0.0
        for i in range(n):
            L_raw = x_torch[start + side_off + 3 * i]
            k_raw = x_torch[start + side_off + 3 * i + 1]
            x0 = x_torch[start + side_off + 3 * i + 2]
            L_eff = torch_mod.nn.functional.softplus(L_raw) + 1e-8
            k_eff = torch_mod.nn.functional.softplus(k_raw) + 1e-8
            s_hi = torch_mod.sigmoid(k_eff * (spread_hi - x0))
            s_lo = torch_mod.sigmoid(k_eff * (spread_lo - x0))
            q = q + L_eff * (s_hi - s_lo)
        try:
            return float(q.detach().item())
        except Exception:
            return 0.0

    num_slices = x_torch.numel() // single_len
    for start in range(0, num_slices * single_len, single_len):
        cur_bid = _effective_qrange(0, start)
        tgt_bid = _desired_quote_size_per_side(
            cur_bid,
            quote_size_target_per_side=quote_size_target_per_side,
            hard_project_quote_size_target=hard_project_quote_size_target,
            quote_size_min_per_side=quote_size_min_per_side,
            quote_size_max_per_side=quote_size_max_per_side,
            hard_project_quote_size_range=hard_project_quote_size_range,
        )
        if tgt_bid is not None and cur_bid > eps:
            factor_bid = float(tgt_bid) / max(cur_bid, eps)
            with torch_mod.no_grad():
                for i in range(n):
                    idx = start + 3 * i
                    x_torch[idx].copy_(x_torch[idx] * factor_bid)

        cur_ask = _effective_qrange(3 * n, start)
        tgt_ask = _desired_quote_size_per_side(
            cur_ask,
            quote_size_target_per_side=quote_size_target_per_side,
            hard_project_quote_size_target=hard_project_quote_size_target,
            quote_size_min_per_side=quote_size_min_per_side,
            quote_size_max_per_side=quote_size_max_per_side,
            hard_project_quote_size_range=hard_project_quote_size_range,
        )
        if tgt_ask is not None and cur_ask > eps:
            factor_ask = float(tgt_ask) / max(cur_ask, eps)
            with torch_mod.no_grad():
                for i in range(n):
                    idx = start + 3 * n + 3 * i
                    x_torch[idx].copy_(x_torch[idx] * factor_ask)


def _project_quote_size_flat(
    flat: np.ndarray,
    n_components: int,
    *,
    quote_size_target_per_side: float | None,
    hard_project_quote_size_target: bool,
    quote_size_min_per_side: float | None,
    quote_size_max_per_side: float | None,
    hard_project_quote_size_range: bool,
    spread_max: float,
) -> np.ndarray:
    """Project one flat param vector using the same target-then-range semantics."""
    out = np.array(flat, dtype=np.float64, copy=True)
    n = n_components
    spread_hi = max(0.0, float(spread_max))
    spread_lo = 0.0
    eps = 1e-12

    def _softplus_np(x: float) -> float:
        return float(np.log1p(np.exp(-abs(float(x)))) + max(float(x), 0.0) + 1e-8)

    def _sigmoid_np(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _effective_qrange(side_off: int) -> float:
        q = 0.0
        for i in range(n):
            L_eff = _softplus_np(out[side_off + 3 * i])
            k_eff = _softplus_np(out[side_off + 3 * i + 1])
            x0 = float(out[side_off + 3 * i + 2])
            s_hi = _sigmoid_np(k_eff * (spread_hi - x0))
            s_lo = _sigmoid_np(k_eff * (spread_lo - x0))
            q += L_eff * (s_hi - s_lo)
        return float(q)

    def _project_side(side_off: int) -> None:
        cur_q = _effective_qrange(side_off)
        desired = _desired_quote_size_per_side(
            cur_q,
            quote_size_target_per_side=quote_size_target_per_side,
            hard_project_quote_size_target=hard_project_quote_size_target,
            quote_size_min_per_side=quote_size_min_per_side,
            quote_size_max_per_side=quote_size_max_per_side,
            hard_project_quote_size_range=hard_project_quote_size_range,
        )
        if desired is None or cur_q <= eps:
            return
        scale = float(desired) / max(cur_q, eps)
        for i in range(n):
            out[side_off + 3 * i] *= scale

    _project_side(0)
    _project_side(3 * n)
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
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
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
    our_bid_level = marg_bid.unsqueeze(0).expand_as(book_bid_level)
    our_ask_level = marg_ask.unsqueeze(0).expand_as(book_ask_level)
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
    consumed_bid_level = comb_bid_level * torch.sigmoid(reach_bid / soft_temp)
    consumed_ask_level = comb_ask_level * torch.sigmoid(reach_ask / soft_temp)

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


def _action_uniformity_l2_from_marginal_depth(
    params: torch.Tensor,
    spread_ticks: torch.Tensor,
    n_components: int,
    quote_size_target_per_side: float | None,
    spread_max: float,
    *,
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
) -> torch.Tensor:
    """
    L2-to-uniform regularizer from discrete marginal depth implied by cumulative depth.

    To match action discretization semantics, marginal levels are built from cumulative
    depth on integer spreads with spread-0 baseline:
      marg[1] = D(1) - D(0)
      marg[s] = D(s) - D(s-1), s >= 2

    Then each side is penalized in raw units against the uniform target level:
      q_uniform = quote_size_target_per_side / spread_max
    (no probability normalization).
    Returns average penalty across bid and ask.
    """
    if n_components <= 0 or spread_ticks.numel() <= 1:
        return params.sum() * 0.0

    torch_mod = _get_torch()
    bid_cum = effective_depth_bid_tensor(
        params, spread_ticks, n_components, use_softplus=use_softplus, l_param=l_param
    )
    ask_cum = effective_depth_ask_tensor(
        params, spread_ticks, n_components, use_softplus=use_softplus, l_param=l_param
    )
    bid_prev = torch_mod.cat([torch_mod.zeros_like(bid_cum[:1]), bid_cum[:-1]], dim=0)
    ask_prev = torch_mod.cat([torch_mod.zeros_like(ask_cum[:1]), ask_cum[:-1]], dim=0)
    bid_marg = torch_mod.clamp(
        bid_cum - bid_prev,
        min=0.0,
    )
    ask_marg = torch_mod.clamp(
        ask_cum - ask_prev,
        min=0.0,
    )
    q_target = max(0.0, float(quote_size_target_per_side or 0.0))
    denom = max(1.0, float(spread_max))
    q_uniform = q_target / denom
    uni = torch_mod.full_like(bid_marg, q_uniform)
    bid_l2 = torch.mean((bid_marg - uni) ** 2)
    ask_l2 = torch.mean((ask_marg - uni) ** 2)
    l2_mean = 0.5 * (bid_l2 + ask_l2)
    return torch_mod.sqrt(torch_mod.clamp(l2_mean, min=1e-20))


def _action_size_penalty_qrange(
    params: torch.Tensor,
    n_components: int,
    spread_max: float,
    quote_size_target_per_side: float | None,
    *,
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
) -> torch.Tensor:
    """
    Squared error penalty to target quoted size over discretizer range s=1..N
    (N=spread_max), using baseline-shift convention:
      (Q_range_bid - Q_target)^2 + (Q_range_ask - Q_target)^2
    where Q_range_side = D_side(spread_max) - D_side(0).
    """
    if quote_size_target_per_side is None or float(quote_size_target_per_side) <= 0.0 or n_components <= 0:
        return params.sum() * 0.0
    torch_mod = _get_torch()
    s = torch_mod.tensor(
        [0.0, max(0.0, float(spread_max))],
        dtype=params.dtype,
        device=params.device,
    )
    bid_depth = effective_depth_bid_tensor(
        params, s, n_components, use_softplus=use_softplus, l_param=l_param
    )
    ask_depth = effective_depth_ask_tensor(
        params, s, n_components, use_softplus=use_softplus, l_param=l_param
    )
    q_bid = torch_mod.clamp(bid_depth[1] - bid_depth[0], min=0.0)
    q_ask = torch_mod.clamp(ask_depth[1] - ask_depth[0], min=0.0)
    tgt_q = (
        l_param.effective_quote_size_target()
        if l_param is not None and l_param.use_softmax
        else float(quote_size_target_per_side)
    )
    tgt = torch_mod.tensor(float(tgt_q), dtype=params.dtype, device=params.device)
    return (q_bid - tgt) ** 2 + (q_ask - tgt) ** 2


def _action_spread_at_depth(
    params: torch.Tensor,
    n_components: int,
    spread_max: float,
    target_cumulative_qty: float,
    *,
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
    grid_points: int = 100,
    softmin_temp: float = 40.0,
) -> torch.Tensor:
    """
    Differentiable proxy for spread at which cumulative bid/ask depth reaches target_cumulative_qty.

    For each side, evaluate effective quoted depth D(δ) - D(0) on a dense spread grid
    in [0, spread_max], then use a soft-argmin of |depth(δ) - target| to get an effective spread. Returns:
      spread_bid(target) + spread_ask(target)
    """
    if n_components <= 0 or float(target_cumulative_qty) <= 0.0:
        return params.sum() * 0.0
    torch_mod = _get_torch()
    s = torch_mod.linspace(
        0.0,
        max(0.0, float(spread_max)),
        steps=max(8, int(grid_points)),
        dtype=params.dtype,
        device=params.device,
    )
    bid_depth = effective_depth_bid_tensor(
        params, s, n_components, use_softplus=use_softplus, l_param=l_param
    )
    ask_depth = effective_depth_ask_tensor(
        params, s, n_components, use_softplus=use_softplus, l_param=l_param
    )
    tgt = torch_mod.tensor(float(target_cumulative_qty), dtype=params.dtype, device=params.device)
    bid_err = torch_mod.abs(bid_depth - tgt)
    ask_err = torch_mod.abs(ask_depth - tgt)
    temp = max(1e-6, float(softmin_temp))
    bid_w = torch_mod.softmax(-temp * bid_err, dim=0)
    ask_w = torch_mod.softmax(-temp * ask_err, dim=0)
    bid_spread = torch_mod.sum(bid_w * s)
    ask_spread = torch_mod.sum(ask_w * s)
    return bid_spread + ask_spread


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
        hedge_price_model: Any = None,
        n_components: int = 2,
        n_samples: int = 32,
        horizon: int = 4,
        spread_max: float = 100.0,
        spread_step: float = 1.0,
        soft_fill_temp: float = 1.0,
        adam_lr: float = 0.05,
        adam_epochs: int = 80,
        adam_grad_clip: float = 10.0,
        device: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        warm_start: bool = False,
        quote_size_min_per_side: float | int | None = None,
        quote_size_max_per_side: float | int | None = None,
        fixed_mc_samples: bool = False,
        verbose: bool = False,
        inv_coef: float | None = None,
        target_inv: float | None = None,
        risk_coef: float | None = None,
        uniformity_coef: float | None = None,
        reward_coef: float | None = None,
        size_coef: float | None = None,
        spread_coef: float | None = None,
        spread_depth: float | None = None,
        hedge_enabled: bool | None = None,
        hedge_coef: float = 0.0,
        hedge_fee: float = 0.0,
        hedge_reward_weight: float = 1.0,
        expected_pnl_weight: float | None = None,
        pnl_volatility_penalty_weight: float | None = None,
        inventory_target_penalty_weight: float | None = None,
        target_inventory: float | None = None,
        quote_shape_uniformity_penalty_weight: float | None = None,
        quote_size_target_penalty_weight: float | None = None,
        quote_spread_penalty_weight: float | None = None,
        quote_depth_target_for_spread: float | None = None,
        hedge_rollout_enabled: bool | None = None,
        quote_size_target_per_side: float | int | None = None,
        hard_project_quote_size_target: bool = False,
        hard_project_quote_size_range: bool = False,
        quote_size_softmax_L: bool = False,
    ):
        self.price_model = price_model
        self.market_flow_model = market_flow_model
        self.execution_model = execution_model
        self.hedge_price_model = hedge_price_model
        self.n_components = max(1, n_components)
        self.n_samples = max(1, n_samples)
        self.horizon = max(1, horizon)
        self.spread_max = float(spread_max)
        self.spread_step = max(1e-12, float(spread_step))
        self.grid_levels = _grid_levels(self.spread_max, self.spread_step)
        self.soft_fill_temp = soft_fill_temp
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
        self.fixed_mc_samples = fixed_mc_samples
        self.verbose = verbose
        self.expected_pnl_weight = float(_coalesce_param(expected_pnl_weight, reward_coef, default=1.0))
        self.pnl_volatility_penalty_weight = max(
            0.0, float(_coalesce_param(pnl_volatility_penalty_weight, risk_coef, default=0.0))
        )
        self.inventory_target_penalty_weight = max(
            0.0, float(_coalesce_param(inventory_target_penalty_weight, inv_coef, default=0.0))
        )
        self.target_inventory = float(_coalesce_param(target_inventory, target_inv, default=0.0))
        self.quote_shape_uniformity_penalty_weight = max(
            0.0, float(_coalesce_param(quote_shape_uniformity_penalty_weight, uniformity_coef, default=0.0))
        )
        self.quote_size_target_penalty_weight = max(
            0.0, float(_coalesce_param(quote_size_target_penalty_weight, size_coef, default=0.0))
        )
        self.quote_spread_penalty_weight = max(
            0.0, float(_coalesce_param(quote_spread_penalty_weight, spread_coef, default=0.0))
        )
        self.quote_depth_target_for_spread = max(
            0.0, float(_coalesce_param(quote_depth_target_for_spread, spread_depth, default=1.0))
        )
        hedge_flag = _coalesce_param(hedge_rollout_enabled, hedge_enabled, default=None)
        if hedge_flag is not None:
            self.hedge_enabled = bool(hedge_flag)
        else:
            # Legacy: non-zero hedge_coef enables hedge dynamics in rollouts.
            self.hedge_enabled = bool(float(hedge_coef) != 0.0)
        # Rollout hedge fee: prefer ``execution_model.hedge_fee`` inside ``optimize()`` when set;
        # otherwise this constructor value is used (backward compatible).
        self.hedge_fee = max(0.0, min(1.0 - 1e-9, float(hedge_fee)))
        self.hedge_reward_weight = max(0.0, float(hedge_reward_weight))
        self.quote_size_target_per_side = (
            float(quote_size_target_per_side) if quote_size_target_per_side is not None else None
        )
        self.hard_project_quote_size_target = bool(hard_project_quote_size_target)
        self.hard_project_quote_size_range = bool(hard_project_quote_size_range)
        self.quote_size_softmax_L = bool(quote_size_softmax_L)
        if self.quote_size_softmax_L and (
            self.quote_size_target_per_side is None or float(self.quote_size_target_per_side) <= 0.0
        ):
            raise ValueError(
                "quote_size_softmax_L requires positive quote_size_target_per_side"
            )
        # Backward-compatible attribute names used internally and by existing tests/tools.
        self.reward_coef = self.expected_pnl_weight
        self.risk_coef = self.pnl_volatility_penalty_weight
        self.inv_coef = self.inventory_target_penalty_weight
        self.uniformity_coef = self.quote_shape_uniformity_penalty_weight
        self.size_coef = self.quote_size_target_penalty_weight
        self.spread_coef = self.quote_spread_penalty_weight
        self.spread_depth = self.quote_depth_target_for_spread
        self.target_inv = self.target_inventory
        self._last_action: SigmoidMixtureAction | None = None
        self._last_raw_flat: np.ndarray | None = None
        self._last_J_components: dict | None = None
        self._last_J_components_initial: dict | None = None
        self._last_J_mean_raw: float = 0.0
        self._last_E_dW_next: float = 0.0
        self._last_main_reward_mean: float = 0.0
        self._last_hedge_reward_mean: float = 0.0
        self._last_main_reward_mean_initial: float = 0.0
        self._last_hedge_reward_mean_initial: float = 0.0
        self._last_sigmoid_action: SigmoidMixtureAction | None = None
        self._last_diagnostics: OptimizerRunDiagnostics | None = None
        self._x_torch: Any = None
        self._optimizer: Any = None

        if bounds is None:
            if self.quote_size_softmax_L:
                n = self.n_components
                logit_bound = (-30.0, 30.0)
                k_bound = (1e-6, 100.0)
                x0_bound = (-self.spread_max, self.spread_max)
                self.bounds = [logit_bound, k_bound, x0_bound] * (2 * n)
            else:
                L_max = 1e5
                if self.quote_size_max_per_side is not None:
                    L_max = max(L_max, float(self.quote_size_max_per_side))
                if self.quote_size_target_per_side is not None:
                    L_max = max(L_max, float(self.quote_size_target_per_side))
                self.bounds = sigmoid_mixture_action_bounds(
                    n_components=self.n_components,
                    L_max=L_max,
                    k_max=100.0,
                    x0_max=self.spread_max,
                )
        else:
            self.bounds = bounds

    @property
    def last_diagnostics(self) -> OptimizerRunDiagnostics | None:
        return self._last_diagnostics

    def _l_param_config(self) -> LParamConfig | None:
        if not self.quote_size_softmax_L:
            return None
        return LParamConfig(
            use_softmax=True,
            quote_size_target=self.quote_size_target_per_side,
            spread_max=self.spread_max,
            quote_size_min=self.quote_size_min_per_side,
            quote_size_max=self.quote_size_max_per_side,
            apply_quote_size_range=self.hard_project_quote_size_range,
        )

    def _depth_call_kwargs(self) -> dict[str, Any]:
        return {"use_softplus": True, "l_param": self._l_param_config()}

    def _flat_to_effective_action(self, flat: np.ndarray) -> SigmoidMixtureAction:
        lp = self._l_param_config()
        if lp is not None:
            return sigmoid_action_from_flat_raw(flat, self.n_components, l_param=lp)
        return SigmoidMixtureAction.from_flat_positive(flat, n_components=self.n_components).as_effective()

    def _quote_size_projection_kwargs(self) -> dict[str, Any]:
        return {
            "quote_size_target_per_side": self.quote_size_target_per_side,
            "hard_project_quote_size_target": self.hard_project_quote_size_target,
            "quote_size_min_per_side": self.quote_size_min_per_side,
            "quote_size_max_per_side": self.quote_size_max_per_side,
            "hard_project_quote_size_range": self.hard_project_quote_size_range,
        }

    def optimize(self, state: State) -> SigmoidMixtureAction:
        action, _ = self.optimize_with_diagnostics(state)
        return action

    def optimize_with_diagnostics(self, state: State) -> tuple[SigmoidMixtureAction, OptimizerRunDiagnostics]:
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
        single_len = 6 * n
        param_len = self.horizon * single_len

        if self.warm_start and self._last_raw_flat is not None:
            one_action = np.array(self._last_raw_flat[:single_len], dtype=np.float64)
        elif self.warm_start and self._last_action is not None:
            one_action = np.array(self._last_action.to_flat(), dtype=np.float64)
        else:
            one_action = _initial_action_uniform_marginal_qrange(
                n_components=n,
                spread_max=self.spread_max,
                quote_size_target_per_side=self.quote_size_target_per_side,
                use_softmax_L=self.quote_size_softmax_L,
            )
            if one_action is None:
                sm = self.spread_max
                bid_part = []
                ask_part = []
                for i in range(n):
                    x0_i = 1.0 + (sm - 1.0) * (i + 1) / (n + 1)
                    bid_part.extend([10.0 + 10.0 * i, 0.5 - 0.1 * i, x0_i])
                    ask_part.extend([10.0 + 10.0 * i, 0.5 - 0.1 * i, x0_i])
                one_action = np.array(bid_part + ask_part, dtype=np.float64)
        x0_arr = np.tile(one_action, self.horizon)

        # Bounds are specified in effective model space. Convert L/k bounds to raw
        # parameter space because optimization variable x_torch is raw.
        bounds_raw = _bounds_effective_to_raw(self.bounds, l_softmax=self.quote_size_softmax_L)
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
        hedge_fee_rollout = getattr(self.execution_model, "hedge_fee", self.hedge_fee)
        hedge_fee_rollout = max(0.0, min(1.0 - 1e-9, float(hedge_fee_rollout)))
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
            if self.hedge_price_model is not None and hasattr(self.hedge_price_model, "sample_deltas"):
                d_hedge_buy, d_hedge_sell = self.hedge_price_model.sample_deltas(
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
        if not self.quote_size_softmax_L:
            initial_flat = _project_quote_size_flat(
                initial_flat,
                n,
                spread_max=self.spread_max,
                **self._quote_size_projection_kwargs(),
            )
        initial_action = self._flat_to_effective_action(initial_flat)
        action_initial_snap = build_action_snapshot(
            initial_action, mid=mid_0, spread_levels=spread_levels
        )

        best_loss = float("inf")
        best_x = x_torch.detach().clone()
        best_J_mean: float | None = None
        best_J_mean_next: float | None = None
        best_main_reward_mean: float | None = None
        best_hedge_reward_mean: float | None = None
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
                if not self.quote_size_softmax_L:
                    cur_flat = _project_quote_size_flat(
                        cur_flat,
                        n,
                        spread_max=self.spread_max,
                        **self._quote_size_projection_kwargs(),
                    )
                print(f"  Epoch {epoch}: action  {_format_action_verbose(cur_flat, n)}")

            def closure():
                nonlocal batch_time_sec, captured_initial_components, initial_metrics, final_metrics
                nonlocal nonfinite_loss_count, nonfinite_grad_count
                optimizer.zero_grad()
                with torch_mod.no_grad():
                    x_torch.data = torch_mod.clamp(x_torch.data, lower, upper)
                    if not self.quote_size_softmax_L:
                        _project_quote_size_torch(
                            x_torch,
                            single_len,
                            n,
                            spread_max=self.spread_max,
                            torch_mod=torch_mod,
                            **self._quote_size_projection_kwargs(),
                        )

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
                hedge_buy_vwap_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(state.hedge_buy_vwap if state.hedge_buy_vwap is not None else state.buy_vwap),
                    dtype=dtype,
                    device=_device,
                )
                hedge_sell_vwap_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(state.hedge_sell_vwap if state.hedge_sell_vwap is not None else state.sell_vwap),
                    dtype=dtype,
                    device=_device,
                )
                hedge_inv_t = torch_mod.full(
                    (self.n_samples, 1),
                    float(getattr(state, "hedge_inventory", 0)),
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
                action_uniformity_sum = torch_mod.zeros((), dtype=dtype, device=_device)
                action_size_sum = torch_mod.zeros((), dtype=dtype, device=_device)
                action_spread_sum = torch_mod.zeros((), dtype=dtype, device=_device)

                for h in range(self.horizon):
                    if h == 0:
                        book_cum_bid_h = book_cum_bid_0
                        book_cum_ask_h = book_cum_ask_0
                    else:
                        book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks)

                    action_h = x_torch[h * single_len : (h + 1) * single_len]
                    depth_kw = self._depth_call_kwargs()
                    action_uniformity_sum = action_uniformity_sum + _action_uniformity_l2_from_marginal_depth(
                        action_h,
                        spread_grid,
                        n,
                        self.quote_size_target_per_side,
                        self.spread_max,
                        **depth_kw,
                    )
                    action_size_sum = action_size_sum + _action_size_penalty_qrange(
                        action_h,
                        n,
                        self.spread_max,
                        self.quote_size_target_per_side,
                        **depth_kw,
                    )
                    action_spread_sum = action_spread_sum + _action_spread_at_depth(
                        action_h,
                        n,
                        self.spread_max,
                        self.spread_depth,
                        **depth_kw,
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
                        **depth_kw,
                    )
                    inv_next = inv_t + inv_delta.unsqueeze(1)
                    cash_next = cash_t + cash_delta.unsqueeze(1)
                    if self.hedge_enabled:
                        d_buy = d_hedge_buy[:, h : h + 1]
                        d_sell = d_hedge_sell[:, h : h + 1]
                        hedge_buy_vwap_next = hedge_buy_vwap_t + d_buy
                        hedge_sell_vwap_next = hedge_sell_vwap_t + d_sell
                        d_inv = inv_delta.unsqueeze(1)  # fee-adjusted executed inventory change
                        # Hedge trade at t+1 on hedge venue:
                        # - d_inv>0 (longer main) -> sell hedge inventory at hedge bid
                        # - d_inv<0 (shorter main) -> buy hedge inventory at hedge ask
                        trade_cash_hedge = torch_mod.where(
                            d_inv >= 0.0,
                            d_inv * hedge_sell_vwap_next,
                            d_inv * hedge_buy_vwap_next,
                        )
                        one_minus_hedge_fee = float(1.0 - hedge_fee_rollout)
                        hedge_cash_next = hedge_cash_t + trade_cash_hedge * one_minus_hedge_fee
                        hedge_inv_next = hedge_inv_t - d_inv
                    else:
                        hedge_buy_vwap_next = hedge_buy_vwap_t
                        hedge_sell_vwap_next = hedge_sell_vwap_t
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
                    if self.hedge_enabled:
                        hedge_wealth_t = hedge_cash_t + hedge_inv_t * hedge_sell_vwap_t
                        hedge_wealth_next = hedge_cash_next + hedge_inv_next * hedge_sell_vwap_next
                        hedge_reward = (hedge_wealth_next - hedge_wealth_t).squeeze(1)
                    step_reward = step_reward_main + float(self.hedge_reward_weight) * hedge_reward
                    if h == 0:
                        step_reward_0 = step_reward
                    total_reward = total_reward + step_reward
                    total_main_reward = total_main_reward + step_reward_main
                    total_hedge_reward = total_hedge_reward + hedge_reward

                    mid_fallback_t = mid_next
                    inv_t = inv_next
                    cash_t = cash_next
                    hedge_buy_vwap_t = hedge_buy_vwap_next
                    hedge_sell_vwap_t = hedge_sell_vwap_next
                    hedge_cash_t = hedge_cash_next
                    hedge_inv_t = hedge_inv_next

                # total_reward = ΔW per trajectory (same as W(s_{t+H}) - W(s_t)); inv_t = terminal inventory
                action_uniformity_mean = action_uniformity_sum / float(self.horizon)
                action_size_mean = action_size_sum / float(self.horizon)
                action_spread_mean = action_spread_sum / float(self.horizon)
                J_mean_raw = torch_mod.nan_to_num(
                    total_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J_mean_next_raw = torch_mod.nan_to_num(
                    step_reward_0.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                main_reward_mean_raw = torch_mod.nan_to_num(
                    total_main_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                hedge_reward_mean_raw = torch_mod.nan_to_num(
                    total_hedge_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J, components = terminal_objective(
                    total_reward,
                    inv_t,
                    expected_pnl_weight=self.expected_pnl_weight,
                    inventory_target_penalty_weight=self.inventory_target_penalty_weight,
                    target_inventory=self.target_inventory,
                    pnl_volatility_penalty_weight=self.pnl_volatility_penalty_weight,
                    uniformity=action_uniformity_mean,
                    quote_shape_uniformity_penalty_weight=self.quote_shape_uniformity_penalty_weight,
                    size=action_size_mean,
                    quote_size_target_penalty_weight=self.quote_size_target_penalty_weight,
                    spread=action_spread_mean,
                    quote_spread_penalty_weight=self.quote_spread_penalty_weight,
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
                    expected_next_reward=_finite_item(J_mean_next_raw),
                    main_reward_mean=_finite_item(main_reward_mean_raw),
                    hedge_reward_mean=_finite_item(hedge_reward_mean_raw),
                )
                self._last_J_components = dict(epoch_metrics.components)
                if not captured_initial_components:
                    self._last_J_components_initial = dict(epoch_metrics.components)
                    self._last_main_reward_mean_initial = epoch_metrics.main_reward_mean
                    self._last_hedge_reward_mean_initial = epoch_metrics.hedge_reward_mean
                    initial_metrics = epoch_metrics
                    captured_initial_components = True
                final_metrics = epoch_metrics
                self._last_J_mean_raw = epoch_metrics.expected_total_reward
                self._last_E_dW_next = epoch_metrics.expected_next_reward
                self._last_main_reward_mean = epoch_metrics.main_reward_mean
                self._last_hedge_reward_mean = epoch_metrics.hedge_reward_mean
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
                best_hedge_reward_mean = final_metrics.hedge_reward_mean if final_metrics else 0.0

        loop_elapsed = time.perf_counter() - t0
        if self.fixed_mc_samples:
            train_time_sec = loop_elapsed
            total_sec = batch_time_sec + loop_elapsed
        else:
            train_time_sec = loop_elapsed - batch_time_sec
            total_sec = loop_elapsed
        best_flat = best_x.cpu().numpy()
        best_flat_slice = best_flat[:single_len]
        if not self.quote_size_softmax_L:
            best_flat_slice = _project_quote_size_flat(
                best_flat_slice,
                n,
                spread_max=self.spread_max,
                **self._quote_size_projection_kwargs(),
            )
        best_action = self._flat_to_effective_action(best_flat_slice)
        final_flat_slice = np.array(x_torch.detach().cpu().numpy()[:single_len], dtype=np.float64)
        if not self.quote_size_softmax_L:
            final_flat_slice = _project_quote_size_flat(
                final_flat_slice,
                n,
                spread_max=self.spread_max,
                **self._quote_size_projection_kwargs(),
            )
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
        if best_hedge_reward_mean is not None and math.isfinite(float(best_hedge_reward_mean)):
            self._last_hedge_reward_mean = float(best_hedge_reward_mean)
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
                "hedge_enabled": self.hedge_enabled,
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
