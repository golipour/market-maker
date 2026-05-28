"""Joint-venue Monte Carlo MPC optimizer rollout (Phase 7)."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mm.mpc.joint_mpc import (
    JointSigmoidMixtureAction,
    joint_action_param_len,
    joint_terminal_objective,
    reference_price_from_state,
    split_venue_action_flat,
    wealth_0_from_state,
    _finite_item,
)
from mm.mpc.l_param import action_param_len, sigmoid_action_from_flat_raw
from mm.mpc.mpc_state import OrderbookRelative, SigmoidMixtureAction, State
from mm.mpc.mpc_tensor_batched import (
    _book_cumulative_from_arrays,
    _book_cumulative_from_ob_rel,
    _bounds_effective_to_raw,
    _format_action_verbose,
    _get_torch,
    _grid_levels,
    _initial_action_uniform_marginal_qrange,
    _mid_from_book_best_levels_batched,
    _quote_capacity_maxima_batched,
    _quote_capacity_maxima_scalar,
    _spread_bucket_index,
    _spread_grid_tensor,
    soft_fill_batched,
)
from mm.mpc.optimizer_diagnostics import (
    EpochObjectiveMetrics,
    OptimizerRunDiagnostics,
    build_action_snapshot,
)
from mm.observability.logs import log_event
from mm.pricing import vwap_prices_batched

if TYPE_CHECKING:
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched


def _ob_levels_from_rel(
    ob_rel: OrderbookRelative | None,
    *,
    grid_levels: int,
    spread_step: float,
    n_samples: int,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    torch_mod = _get_torch()
    ob_bids = torch_mod.zeros(n_samples, grid_levels, dtype=dtype, device=device)
    ob_asks = torch_mod.zeros(n_samples, grid_levels, dtype=dtype, device=device)
    if ob_rel is None:
        return ob_bids, ob_asks
    for spread, qty in ob_rel.bids:
        idx = _spread_bucket_index(float(spread), spread_step, grid_levels)
        if 0 <= idx < grid_levels:
            ob_bids[:, idx] += float(qty)
    for spread, qty in ob_rel.asks:
        idx = _spread_bucket_index(float(spread), spread_step, grid_levels)
        if 0 <= idx < grid_levels:
            ob_asks[:, idx] += float(qty)
    return ob_bids, ob_asks


def optimize_joint_with_diagnostics(
    mpc: MonteCarloMPCTensorBatched,
    state: State,
) -> tuple[SigmoidMixtureAction, OptimizerRunDiagnostics]:
    """Run four-curve joint MPC with consolidated wealth objective."""

    torch_mod = _get_torch()
    if isinstance(mpc.device, str) and mpc.device.lower() == "cpu":
        _device = torch_mod.device("cpu")
    elif mpc.device is None:
        from mm.mpc.torch_device import resolve_torch_device

        _device = resolve_torch_device(None)
    else:
        from mm.mpc.torch_device import resolve_torch_device

        _device = resolve_torch_device(str(mpc.device).strip() or None)
    dtype = torch_mod.float64

    global_active = bool(state.global_active)
    global_mask = 1.0 if global_active else 0.0
    reference_price_source = str(getattr(mpc, "reference_price_source", "global")).strip().lower()
    if reference_price_source not in ("global", "local"):
        reference_price_source = "global"
    use_global_reference = (
        reference_price_source == "global"
        and state.global_sell_vwap is not None
        and float(state.global_sell_vwap) > 0.0
    )
    ref_0 = float(reference_price_from_state(state, reference_price_source))
    wealth_0 = wealth_0_from_state(state, reference_price_source)
    obj_local_target = mpc._resolve_inventory_target("local")
    obj_global_target = mpc._resolve_inventory_target("global")
    obj_local_scale = mpc._resolve_inventory_scale("local")
    obj_global_scale = mpc._resolve_inventory_scale("global")

    book_cum_bid, book_cum_ask = _book_cumulative_from_ob_rel(
        state.ob_rel, mpc.grid_levels, mpc.spread_step, _device, dtype
    )
    global_ob = state.global_ob_rel or OrderbookRelative(bids=(), asks=())
    global_book_cum_bid, global_book_cum_ask = _book_cumulative_from_ob_rel(
        global_ob, mpc.grid_levels, mpc.spread_step, _device, dtype
    )
    spread_grid = _spread_grid_tensor(torch_mod, mpc.grid_levels, mpc.spread_step, _device, dtype)
    mid_0 = float(state.mid)
    global_mid_0 = float(state.global_mid or state.reference_price or state.global_sell_vwap or mid_0)
    inv_0 = float(state.inventory)
    cash_0 = float(state.cash)
    global_inv_0 = float(state.global_inventory)
    global_cash_0 = float(state.global_cash)

    ob_bids_0, ob_asks_0 = _ob_levels_from_rel(
        state.ob_rel,
        grid_levels=mpc.grid_levels,
        spread_step=mpc.spread_step,
        n_samples=mpc.n_samples,
        device=_device,
        dtype=dtype,
    )
    global_ob_bids_0, global_ob_asks_0 = _ob_levels_from_rel(
        global_ob,
        grid_levels=mpc.grid_levels,
        spread_step=mpc.spread_step,
        n_samples=mpc.n_samples,
        device=_device,
        dtype=dtype,
    )

    n = mpc.n_components
    venue_len = action_param_len(n)
    joint_step_len = joint_action_param_len(n)
    param_len = mpc.horizon * joint_step_len
    local_l_param = mpc._l_param_config("local")
    global_l_param = mpc._l_param_config("global")
    local_depth_kw = mpc._depth_call_kwargs("local")
    global_depth_kw = mpc._depth_call_kwargs("global")
    local_bid_quote_size_max_0, local_ask_quote_size_max_0 = _quote_capacity_maxima_scalar(
        cash=cash_0,
        inventory=inv_0,
        mid_price=mid_0,
        optimizer=mpc.local_optimizer,
    )
    global_bid_quote_size_max_0, global_ask_quote_size_max_0 = _quote_capacity_maxima_scalar(
        cash=global_cash_0,
        inventory=global_inv_0,
        mid_price=global_mid_0,
        optimizer=mpc.global_optimizer,
    )

    if mpc.warm_start and mpc._last_raw_flat is not None and mpc._last_raw_flat.size >= joint_step_len:
        one_action = np.array(mpc._last_raw_flat[:joint_step_len], dtype=np.float64)
    else:
        local_action = _initial_action_uniform_marginal_qrange(
            n_components=n,
            spread_max=mpc.spread_max,
            l_param=local_l_param,
        )
        global_action = _initial_action_uniform_marginal_qrange(
            n_components=n,
            spread_max=mpc.spread_max,
            l_param=global_l_param,
        )
        one_action = np.concatenate([local_action, global_action])
    x0_arr = np.tile(one_action, mpc.horizon)

    local_bounds = _bounds_effective_to_raw(mpc.local_bounds)[:venue_len]
    global_bounds = _bounds_effective_to_raw(mpc.global_bounds)[:venue_len]
    joint_bounds = local_bounds + global_bounds
    bounds_full = joint_bounds * mpc.horizon
    lower = torch_mod.tensor([b[0] for b in bounds_full], dtype=dtype, device=_device)
    upper = torch_mod.tensor([b[1] for b in bounds_full], dtype=dtype, device=_device)

    if mpc._x_torch is None or mpc._x_torch.numel() != param_len:
        mpc._x_torch = torch_mod.tensor(x0_arr, dtype=dtype, device=_device, requires_grad=True)
        mpc._optimizer = torch_mod.optim.Adam([mpc._x_torch], lr=mpc.adam_lr)
    else:
        with torch_mod.no_grad():
            mpc._x_torch.data = torch_mod.tensor(x0_arr, dtype=dtype, device=_device)
        if not mpc.warm_start:
            mpc._optimizer = torch_mod.optim.Adam([mpc._x_torch], lr=mpc.adam_lr)
    x_torch = mpc._x_torch
    optimizer = mpc._optimizer
    maker_fee = getattr(mpc.execution_model, "maker_fee", 0.0)
    vwap_depth_0 = max(0.0, float(getattr(state, "vwap_depth", 0.0)))
    global_flow_model = mpc.global_market_flow_model
    global_price_model = mpc.global_price_model
    local_opt = mpc.local_optimizer
    global_opt = mpc.global_optimizer

    def draw_samples():
        sample_shape = (mpc.n_samples, mpc.horizon)
        if hasattr(mpc.local_market_flow_model, "sample"):
            buy_vol, sell_vol = mpc.local_market_flow_model.sample(sample_shape, device=_device, dtype=dtype)
        else:
            buy_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            sell_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
        if hasattr(global_flow_model, "sample"):
            global_buy_vol, global_sell_vol = global_flow_model.sample(
                sample_shape, device=_device, dtype=dtype
            )
        else:
            global_buy_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            global_sell_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
        if hasattr(mpc.local_price_model, "sample_residual"):
            eps_buy, eps_sell = mpc.local_price_model.sample_residual(
                sample_shape, device=_device, dtype=dtype
            )
        else:
            eps_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            eps_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
        if hasattr(global_price_model, "sample_residual"):
            global_eps_buy, global_eps_sell = global_price_model.sample_residual(
                sample_shape, device=_device, dtype=dtype
            )
        else:
            global_eps_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            global_eps_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
        return (
            buy_vol,
            sell_vol,
            global_buy_vol,
            global_sell_vol,
            eps_buy,
            eps_sell,
            global_eps_buy,
            global_eps_sell,
        )

    batch_time_sec = 0.0
    if mpc.fixed_mc_samples:
        t_batch = time.perf_counter()
        fixed_samples = draw_samples()
        batch_time_sec = time.perf_counter() - t_batch

    spread_levels = tuple(round(i * mpc.spread_step, 12) for i in range(mpc.grid_levels + 1))
    initial_flat = np.array(x_torch.detach().cpu().numpy()[:joint_step_len], dtype=np.float64)
    local_initial_flat, _ = split_venue_action_flat(initial_flat, n)
    initial_action = sigmoid_action_from_flat_raw(
        local_initial_flat,
        n,
        l_param=local_l_param,
        bid_quote_size_max=local_bid_quote_size_max_0,
        ask_quote_size_max=local_ask_quote_size_max_0,
    )
    action_initial_snap = build_action_snapshot(initial_action, mid=mid_0, spread_levels=spread_levels)

    best_loss = float("inf")
    best_x = x_torch.detach().clone()
    best_epoch: int | None = None
    initial_metrics: EpochObjectiveMetrics | None = None
    final_metrics: EpochObjectiveMetrics | None = None
    best_metrics: EpochObjectiveMetrics | None = None
    nonfinite_loss_count = 0
    nonfinite_grad_count = 0
    last_eval_x = best_x.detach().clone()
    joint_diag_accum = {
        "local_fill_bid_mean": 0.0,
        "local_fill_ask_mean": 0.0,
        "global_fill_bid_mean": 0.0,
        "global_fill_ask_mean": 0.0,
        "local_wealth_contrib_mean": 0.0,
        "global_wealth_contrib_mean": 0.0,
    }
    t0 = time.perf_counter()

    for epoch in range(mpc.adam_epochs):
        if mpc.verbose:
            cur_flat = x_torch.detach().cpu().numpy()[:joint_step_len]
            local_flat, global_flat = split_venue_action_flat(cur_flat, n)
            print(
                f"  Epoch {epoch}: local {_format_action_verbose(local_flat, n)} | "
                f"global {_format_action_verbose(global_flat, n)}"
            )

        def closure():
            nonlocal batch_time_sec, initial_metrics, final_metrics
            nonlocal nonfinite_loss_count, nonfinite_grad_count, last_eval_x
            optimizer.zero_grad()
            with torch_mod.no_grad():
                x_torch.data = torch_mod.clamp(x_torch.data, lower, upper)
                last_eval_x = x_torch.detach().clone()

            if mpc.fixed_mc_samples:
                (
                    buy_vol,
                    sell_vol,
                    global_buy_vol,
                    global_sell_vol,
                    eps_buy,
                    eps_sell,
                    global_eps_buy,
                    global_eps_sell,
                ) = fixed_samples
            else:
                t_batch = time.perf_counter()
                (
                    buy_vol,
                    sell_vol,
                    global_buy_vol,
                    global_sell_vol,
                    eps_buy,
                    eps_sell,
                    global_eps_buy,
                    global_eps_sell,
                ) = draw_samples()
                nonlocal batch_time_sec
                batch_time_sec += time.perf_counter() - t_batch

            inv_t = torch_mod.full((mpc.n_samples, 1), inv_0, dtype=dtype, device=_device)
            cash_t = torch_mod.full((mpc.n_samples, 1), cash_0, dtype=dtype, device=_device)
            global_inv_t = torch_mod.full((mpc.n_samples, 1), global_inv_0, dtype=dtype, device=_device)
            global_cash_t = torch_mod.full((mpc.n_samples, 1), global_cash_0, dtype=dtype, device=_device)
            ref_t = torch_mod.full((mpc.n_samples, 1), ref_0, dtype=dtype, device=_device)

            ob_bids = ob_bids_0.clone()
            ob_asks = ob_asks_0.clone()
            global_ob_bids = global_ob_bids_0.clone()
            global_ob_asks = global_ob_asks_0.clone()
            mid_fallback_t = torch_mod.full((mpc.n_samples, 1), mid_0, dtype=dtype, device=_device)
            global_mid_fallback_t = torch_mod.full(
                (mpc.n_samples, 1),
                global_mid_0,
                dtype=dtype,
                device=_device,
            )

            book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(mpc.n_samples, -1)
            book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(mpc.n_samples, -1)
            global_book_cum_bid_0 = global_book_cum_bid.unsqueeze(0).expand(mpc.n_samples, -1)
            global_book_cum_ask_0 = global_book_cum_ask.unsqueeze(0).expand(mpc.n_samples, -1)

            total_reward = torch_mod.zeros(mpc.n_samples, dtype=dtype, device=_device)
            total_local_reward = torch_mod.zeros(mpc.n_samples, dtype=dtype, device=_device)
            total_global_reward = torch_mod.zeros(mpc.n_samples, dtype=dtype, device=_device)
            local_inv_path = torch_mod.zeros(mpc.n_samples, mpc.horizon, dtype=dtype, device=_device)
            global_inv_path = torch_mod.zeros(mpc.n_samples, mpc.horizon, dtype=dtype, device=_device)
            step_reward_0 = torch_mod.zeros(mpc.n_samples, dtype=dtype, device=_device)
            local_fill_bid_sum = torch_mod.zeros((), dtype=dtype, device=_device)
            local_fill_ask_sum = torch_mod.zeros((), dtype=dtype, device=_device)
            global_fill_bid_sum = torch_mod.zeros((), dtype=dtype, device=_device)
            global_fill_ask_sum = torch_mod.zeros((), dtype=dtype, device=_device)

            for h in range(mpc.horizon):
                if h == 0:
                    book_cum_bid_h = book_cum_bid_0
                    book_cum_ask_h = book_cum_ask_0
                    global_book_cum_bid_h = global_book_cum_bid_0
                    global_book_cum_ask_h = global_book_cum_ask_0
                else:
                    book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks)
                    global_book_cum_bid_h, global_book_cum_ask_h = _book_cumulative_from_arrays(
                        global_ob_bids, global_ob_asks
                    )

                joint_h = x_torch[h * joint_step_len : (h + 1) * joint_step_len]
                local_h, global_h = torch_mod.split(joint_h, venue_len)

                sell_h = sell_vol[:, h : h + 1]
                buy_h = buy_vol[:, h : h + 1]
                global_buy_h = global_buy_vol[:, h : h + 1]
                global_sell_h = global_sell_vol[:, h : h + 1]
                eps_buy_h = eps_buy[:, h : h + 1]
                eps_sell_h = eps_sell[:, h : h + 1]
                global_eps_buy_h = global_eps_buy[:, h : h + 1]
                global_eps_sell_h = global_eps_sell[:, h : h + 1]
                if h + 1 < mpc.horizon:
                    global_eps_buy_next = global_eps_buy[:, h + 1 : h + 2]
                    global_eps_sell_next = global_eps_sell[:, h + 1 : h + 2]
                else:
                    global_eps_buy_next = global_eps_buy_h
                    global_eps_sell_next = global_eps_sell_h

                mid_t = _mid_from_book_best_levels_batched(
                    ob_bids, ob_asks, mid_fallback_t, spread_grid
                )
                global_mid_t = _mid_from_book_best_levels_batched(
                    global_ob_bids,
                    global_ob_asks,
                    global_mid_fallback_t,
                    spread_grid,
                )
                local_bid_quote_size_max_t, local_ask_quote_size_max_t = _quote_capacity_maxima_batched(
                    cash=cash_t,
                    inventory=inv_t,
                    mid_price=mid_t,
                    optimizer=local_opt,
                )
                global_bid_quote_size_max_t, global_ask_quote_size_max_t = _quote_capacity_maxima_batched(
                    cash=global_cash_t,
                    inventory=global_inv_t,
                    mid_price=global_mid_t,
                    optimizer=global_opt,
                )
                vwap_depth_t = torch_mod.abs(
                    inv_t + global_inv_t if mpc.liquidity_sourcing_enabled else inv_t
                )
                if h == 0:
                    buy_vwap_t = torch_mod.full((mpc.n_samples, 1), float(state.buy_vwap), dtype=dtype, device=_device)
                    sell_vwap_t = torch_mod.full((mpc.n_samples, 1), float(state.sell_vwap), dtype=dtype, device=_device)
                    global_buy_vwap_t = torch_mod.full(
                        (mpc.n_samples, 1), float(state.global_buy_vwap or state.buy_vwap), dtype=dtype, device=_device
                    )
                    global_sell_vwap_t = torch_mod.full(
                        (mpc.n_samples, 1),
                        float(state.global_sell_vwap or state.sell_vwap),
                        dtype=dtype,
                        device=_device,
                    )
                else:
                    buy_vwap_t, sell_vwap_t = vwap_prices_batched(
                        ob_bids, ob_asks, mid_t, spread_grid, target_depth=vwap_depth_t
                    )
                    global_buy_vwap_t, global_sell_vwap_t = vwap_prices_batched(
                        global_ob_bids,
                        global_ob_asks,
                        global_mid_t,
                        spread_grid,
                        target_depth=vwap_depth_t,
                    )
                buy_vwap_t = buy_vwap_t + eps_buy_h
                sell_vwap_t = sell_vwap_t + eps_sell_h
                global_buy_vwap_t = global_buy_vwap_t + global_eps_buy_h
                global_sell_vwap_t = global_sell_vwap_t + global_eps_sell_h

                ob_bids_next, ob_asks_next, local_cash_delta, local_inv_delta = soft_fill_batched(
                    local_h,
                    n,
                    mid_t,
                    book_cum_bid_h,
                    book_cum_ask_h,
                    sell_h,
                    buy_h,
                    spread_grid,
                    maker_fee,
                    soft_temp=local_opt.soft_fill_temp,
                    soft_fill_activation=local_opt.soft_fill_activation,
                    soft_fill_queue_penalty=local_opt.soft_fill_queue_penalty,
                    bid_quote_size_max=local_bid_quote_size_max_t,
                    ask_quote_size_max=local_ask_quote_size_max_t,
                    depth_scale=1.0,
                    **local_depth_kw,
                )
                (
                    global_ob_bids_next,
                    global_ob_asks_next,
                    global_cash_delta,
                    global_inv_delta,
                ) = soft_fill_batched(
                    global_h,
                    n,
                    global_mid_t,
                    global_book_cum_bid_h,
                    global_book_cum_ask_h,
                    global_sell_h,
                    global_buy_h,
                    spread_grid,
                    maker_fee,
                    soft_temp=global_opt.soft_fill_temp,
                    soft_fill_activation=global_opt.soft_fill_activation,
                    soft_fill_queue_penalty=global_opt.soft_fill_queue_penalty,
                    bid_quote_size_max=global_bid_quote_size_max_t,
                    ask_quote_size_max=global_ask_quote_size_max_t,
                    depth_scale=global_mask,
                    **global_depth_kw,
                )

                inv_next = inv_t + local_inv_delta.unsqueeze(1)
                cash_next = cash_t + local_cash_delta.unsqueeze(1)
                global_inv_next = global_inv_t + global_inv_delta.unsqueeze(1)
                global_cash_next = global_cash_t + global_cash_delta.unsqueeze(1)
                local_inv_path[:, h] = inv_next.squeeze(1)
                global_inv_path[:, h] = global_inv_next.squeeze(1)

                ref_next = global_sell_vwap_t if use_global_reference else sell_vwap_t
                local_step_reward = (
                    cash_next + inv_next * ref_next - (cash_t + inv_t * ref_t)
                ).squeeze(1)
                global_step_reward = (
                    global_cash_next
                    + global_inv_next * ref_next
                    - (global_cash_t + global_inv_t * ref_t)
                ).squeeze(1)
                step_reward = local_step_reward + global_step_reward
                if h == 0:
                    step_reward_0 = step_reward
                    local_fill_bid_sum = local_fill_bid_sum + local_inv_delta.mean()
                    local_fill_ask_sum = local_fill_ask_sum + (-local_inv_delta).clamp(min=0.0).mean()
                    global_fill_bid_sum = global_fill_bid_sum + global_inv_delta.mean() * global_mask
                    global_fill_ask_sum = (
                        global_fill_ask_sum + (-global_inv_delta).clamp(min=0.0).mean() * global_mask
                    )
                total_reward = total_reward + step_reward
                total_local_reward = total_local_reward + local_step_reward
                total_global_reward = total_global_reward + global_step_reward

                ob_bids, ob_asks = ob_bids_next, ob_asks_next
                global_ob_bids, global_ob_asks = global_ob_bids_next, global_ob_asks_next
                inv_t = inv_next
                cash_t = cash_next
                global_inv_t = global_inv_next
                global_cash_t = global_cash_next
                ref_t = ref_next
                mid_fallback_t = mid_t
                global_mid_fallback_t = global_mid_t

            returns = total_reward / wealth_0
            local_returns = total_local_reward / wealth_0
            global_returns = total_global_reward / wealth_0
            J, components = joint_terminal_objective(
                local_returns,
                global_returns,
                local_inv_path,
                global_inv_path,
                local_return_risk_aversion_beta=local_opt.return_risk_aversion_beta,
                global_return_risk_aversion_beta=global_opt.return_risk_aversion_beta,
                local_objective_weight=local_opt.objective_weight,
                global_objective_weight=global_opt.objective_weight,
                local_inventory_target_penalty_weight=local_opt.inventory_target_penalty_weight,
                global_inventory_target_penalty_weight=global_opt.inventory_target_penalty_weight,
                local_target_inventory=obj_local_target,
                global_target_inventory=obj_global_target,
                local_inventory_scale=obj_local_scale,
                global_inventory_scale=obj_global_scale,
                global_active=global_mask,
            )
            J_mean_raw = torch_mod.nan_to_num(
                total_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
            )
            J_return_raw = torch_mod.nan_to_num(
                returns.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
            )
            local_reward_raw = torch_mod.nan_to_num(
                total_local_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
            )
            global_reward_raw = torch_mod.nan_to_num(
                total_global_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
            )
            J_mean_next_raw = torch_mod.nan_to_num(
                step_reward_0.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
            )
            epoch_metrics = EpochObjectiveMetrics.from_tensors(
                objective_value=_finite_item(J),
                components={k: _finite_item(v) for k, v in components.items()},
                expected_total_reward=_finite_item(J_mean_raw),
                expected_total_return=_finite_item(J_return_raw),
                expected_next_reward=_finite_item(J_mean_next_raw),
                main_reward_mean=_finite_item(local_reward_raw),
                global_reward_mean=_finite_item(global_reward_raw),
            )
            mpc._last_J_components = dict(epoch_metrics.components)
            if initial_metrics is None:
                initial_metrics = epoch_metrics
            final_metrics = epoch_metrics
            mpc._last_J_mean_raw = epoch_metrics.expected_total_reward
            mpc._last_E_dW_next = epoch_metrics.expected_next_reward
            mpc._last_main_reward_mean = _finite_item(local_reward_raw)
            mpc._last_global_reward_mean = _finite_item(global_reward_raw)
            joint_diag_accum["local_fill_bid_mean"] = float(local_fill_bid_sum.detach().item())
            joint_diag_accum["local_fill_ask_mean"] = float(local_fill_ask_sum.detach().item())
            joint_diag_accum["global_fill_bid_mean"] = float(global_fill_bid_sum.detach().item())
            joint_diag_accum["global_fill_ask_mean"] = float(global_fill_ask_sum.detach().item())
            joint_diag_accum["local_wealth_contrib_mean"] = _finite_item(local_reward_raw)
            joint_diag_accum["global_wealth_contrib_mean"] = _finite_item(global_reward_raw)
            loss = -J
            if not torch_mod.isfinite(loss):
                nonfinite_loss_count += 1
            if torch_mod.isfinite(loss):
                loss.backward()
                if x_torch.grad is not None:
                    if not torch_mod.isfinite(x_torch.grad).all():
                        nonfinite_grad_count += 1
                    torch_mod.nn.utils.clip_grad_norm_([x_torch], mpc.adam_grad_clip)
            return loss

        loss = optimizer.step(closure)
        if loss is not None and torch_mod.isfinite(loss) and loss.item() < best_loss:
            best_loss = loss.item()
            best_x = last_eval_x.detach().clone()
            best_epoch = epoch
            if final_metrics is not None:
                best_metrics = final_metrics

    loop_elapsed = time.perf_counter() - t0
    if mpc.fixed_mc_samples:
        train_time_sec = loop_elapsed
        total_sec = batch_time_sec + loop_elapsed
    else:
        train_time_sec = loop_elapsed - batch_time_sec
        total_sec = loop_elapsed

    best_flat = best_x.cpu().numpy()
    best_joint_flat = best_flat[:joint_step_len]
    local_flat, global_flat = split_venue_action_flat(best_joint_flat, n)
    best_local = sigmoid_action_from_flat_raw(
        local_flat,
        n,
        l_param=local_l_param,
        bid_quote_size_max=local_bid_quote_size_max_0,
        ask_quote_size_max=local_ask_quote_size_max_0,
    )
    best_global = sigmoid_action_from_flat_raw(
        global_flat,
        n,
        l_param=global_l_param,
        bid_quote_size_max=global_bid_quote_size_max_0,
        ask_quote_size_max=global_ask_quote_size_max_0,
    )
    joint_action = JointSigmoidMixtureAction(
        local=best_local, global_=best_global, global_active=global_active
    )
    mpc._last_joint_action = joint_action
    mpc._last_global_sigmoid_action = best_global if global_active else None
    mpc._last_sigmoid_action = best_local
    if mpc.warm_start:
        mpc._last_action = best_local
        mpc._last_raw_flat = np.array(best_joint_flat, dtype=np.float64)

    final_local_flat, _ = split_venue_action_flat(
        np.array(x_torch.detach().cpu().numpy()[:joint_step_len], dtype=np.float64), n
    )
    final_action = sigmoid_action_from_flat_raw(
        final_local_flat,
        n,
        l_param=local_l_param,
        bid_quote_size_max=local_bid_quote_size_max_0,
        ask_quote_size_max=local_ask_quote_size_max_0,
    )

    diagnostics = OptimizerRunDiagnostics(
        config={
            "kind": "monte_carlo_mpc_tensor_batched",
            "joint_venue": True,
            "global_active": global_active,
            "n_samples": mpc.n_samples,
            "horizon": mpc.horizon,
            "adam_epochs": mpc.adam_epochs,
            "fixed_mc_samples": mpc.fixed_mc_samples,
            "spread_max": mpc.spread_max,
            "spread_step": mpc.spread_step,
            "vwap_depth": vwap_depth_0,
            "reference_price": ref_0,
            "reference_price_source": reference_price_source,
            "wealth_0": wealth_0,
            "local_optimizer": local_opt.as_config(
                spread_max=mpc.spread_max,
                n_components=mpc.n_components,
            ),
            "global_optimizer": global_opt.as_config(
                spread_max=mpc.spread_max,
                n_components=mpc.n_components,
            ),
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
        action_best=build_action_snapshot(best_local, mid=mid_0, spread_levels=spread_levels),
        gradients={
            "nonfinite_loss_count": nonfinite_loss_count,
            "nonfinite_grad_count": nonfinite_grad_count,
        },
        rollout={
            "n_samples": mpc.n_samples,
            "horizon": mpc.horizon,
            "liquidity_sourcing": bool(mpc.liquidity_sourcing_enabled),
            "global_reference": bool(use_global_reference),
            "global_active": global_active,
            "consolidated_wealth": True,
            "global_rollout": False,
            "joint_diagnostics": dict(joint_diag_accum),
            "global_action_flat": list(global_flat),
        },
    )
    mpc._last_diagnostics = diagnostics
    log_event(
        "mpc_joint_optimizer_timing",
        {
            "batch_sec": batch_time_sec,
            "train_sec": train_time_sec,
            "total_sec": total_sec,
            "global_active": global_active,
            "best_epoch": best_epoch,
        },
    )
    return best_local, diagnostics
