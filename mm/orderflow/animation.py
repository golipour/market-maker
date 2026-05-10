"""
Order-flow animation: plotting and matplotlib only.
Layout is dynamic: each pipeline component (estimator, tester) declares its subplots via get_plot_specs();
the figure is built from these specs so changing the pipeline does not require animation code changes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np

from mm_runtime.order_flow_stepper import Step
from mm_runtime.order_flow_runner import OrderFlowRunner
from mm.orderflow.testers import MarketOrderFlowTester

from mm.orderflow.animation_layout import build_layout, section_ranges
from mm.core.orderflow_execution import cumulative_depth_curves, depth_from_sigmoid_params
from mm.core.orderflow_state import ob_to_ob_rel
from mm.mpc.limit_order_flow import limit_order_flow_observed

DELTA_OB_SPREAD_MAX = 100.0
# QQ plots: empirical quantiles use at most this many most recent steps (buy/sell per step).
QQ_OBS_WINDOW = 10


def _finite_series_ylim(
    series,
    *,
    pad_min: float = 0.5,
    pad_frac: float = 0.1,
    default: tuple[float, float] = (-1.0, 1.0),
) -> tuple[float, float]:
    """Y-limits from numeric series; ignores NaN/Inf so matplotlib never sees invalid limits."""
    vals: list[float] = []
    for x in series:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return default
    y_lo, y_hi = min(vals), max(vals)
    span = y_hi - y_lo
    pad = max(span * pad_frac, pad_min) if span > 0 else max(abs(y_lo) * pad_frac, pad_min)
    return (y_lo - pad, y_hi + pad)


def _delta_orderbook_curves(
    prev_ob,
    cur_ob,
    buy_vol: float,
    sell_vol: float,
    spread_max: float = DELTA_OB_SPREAD_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delta orderbook: (current OB aligned to prev mid, minus traded) − (prev OB minus traded).
    Volumes are level (per-level), not cumulative; we compute cumulative at each spread then subtract.
    Returns (x_grid, delta_cumvol) with x in [-spread_max, spread_max], bid side x <= 0, ask x >= 0.
    """
    mid_prev = float(prev_ob.mid)
    mid_cur = float(cur_ob.mid)
    delta_mid = mid_cur - mid_prev
    prev_rel = ob_to_ob_rel(prev_ob, spread_max=spread_max)
    cur_rel = ob_to_ob_rel(cur_ob, spread_max=spread_max)
    # Align current OB to previous mid: bid spread in prev frame = spread_cur - delta_mid, ask = spread_cur + delta_mid
    cur_bid_aligned = [(s - delta_mid, q) for (s, q) in cur_rel.bids]
    cur_ask_aligned = [(s + delta_mid, q) for (s, q) in cur_rel.asks]
    # Prev minus traded: reduce prev asks by buy_vol (from best ask), prev bids by sell_vol (from best bid). Levels are level volume.
    prev_asks_sorted = sorted(prev_rel.asks, key=lambda t: t[0])
    prev_bids_sorted = sorted(prev_rel.bids, key=lambda t: t[0])

    def reduce_by_flow(levels_sorted: list, flow_vol: float) -> list:
        remaining = float(flow_vol)
        out = []
        for (s, q) in levels_sorted:
            take = min(q, max(0, int(round(remaining))))
            new_q = q - take
            remaining -= take
            if new_q > 0:
                out.append((s, new_q))
        return out

    prev_ask_reduced = reduce_by_flow(prev_asks_sorted, buy_vol)
    prev_bid_reduced = reduce_by_flow(prev_bids_sorted, sell_vol)

    def cumvol_at(spread: float, levels: list) -> float:
        """Cumulative volume at spread and better (levels with sp <= spread)."""
        return sum(float(q) for (sp, q) in levels if sp <= spread)

    grid = np.linspace(-spread_max, spread_max, max(201, int(2 * spread_max)))
    delta_combined = np.zeros_like(grid)
    for i, g in enumerate(grid):
        if g <= 0:
            s_bid = -g
            delta_combined[i] = cumvol_at(s_bid, cur_bid_aligned) - cumvol_at(s_bid, prev_bid_reduced)
        else:
            s_ask = g
            delta_combined[i] = cumvol_at(s_ask, cur_ask_aligned) - cumvol_at(s_ask, prev_ask_reduced)
    return grid, delta_combined


def _merged_log_loss_plot_data(
    tester: MarketOrderFlowTester,
    price_tester: object,
    limit_order_flow_model: object = None,
) -> dict:
    """Merge market order flow tester (buy/sell NLL) and optional price_tester (Δmid NLL) into one chart."""
    vol_data = tester.get_plot_data()
    series = list(vol_data.get("series", []))
    n = len(series[0]["y"]) if series else 0
    if price_tester is not None and n > 0 and hasattr(price_tester, "nll_hist"):
        price_hist = price_tester.nll_hist
        pad = n - len(price_hist)
        price_y = [0.0] * pad + price_hist if pad > 0 else price_hist
        series.append({"label": "Δmid NLL", "color": "blue", "y": price_y})
    return {
        "title": "Log loss (buy, sell, Δmid)",
        "ylabel": "EWMA −log p",
        "series": series,
    }


def _log_loss_subplots_data(
    tester: MarketOrderFlowTester,
    price_tester: object,
    limit_order_flow_model: object = None,
    n_dates: int | None = None,
    runner: OrderFlowRunner | None = None,
    market_flow_model: object | None = None,
) -> list[dict]:
    """
    Build one subplot per predictive log loss (Buy NLL, Sell NLL, LOF mean NLL),
    plus a mid-forecast running MSE comparison when runner data is available.
    No price NLL subplot (residual is fixed). Each subplot has a single series.
    If n_dates is set, pad each series' y at the front so len(y) == n_dates (scores start after first step).
    """
    vol_data = tester.get_plot_data()
    series = list(vol_data.get("series", []))
    n = len(series[0]["y"]) if series else 0
    subplots = []
    # Market-order-flow parameter subplots (moved from figure 1 to figure 2).
    if market_flow_model is not None and hasattr(market_flow_model, "get_plot_data"):
        model_plot_data = market_flow_model.get_plot_data()
        model_subplots = model_plot_data.get("subplots")
        if model_subplots is None:
            model_subplots = [{
                "title": model_plot_data.get("title", "Estimator"),
                "ylabel": model_plot_data.get("ylabel", ""),
                "series": model_plot_data.get("series", []),
            }]
        for sp in model_subplots:
            y_max_len = 0
            for ser in sp.get("series", []):
                y_max_len = max(y_max_len, len(list(ser.get("y", []))))
            if n_dates is not None and n_dates > 0 and y_max_len < n_dates:
                pad = n_dates - y_max_len
                sp_series = []
                for ser in sp.get("series", []):
                    y = list(ser.get("y", []))
                    sp_series.append({**ser, "y": ([0.0] * pad + y)})
                subplots.append({**sp, "series": sp_series})
            else:
                subplots.append(sp)
    for s in series:
        y = list(s.get("y", []))
        if n_dates is not None and n_dates > 0 and len(y) < n_dates:
            pad = n_dates - len(y)
            y = [0.0] * pad + y
        subplots.append({
            "title": s.get("label", "NLL"),
            "ylabel": "EWMA −log p",
            "series": [{**s, "y": y}],
        })
    # Limit order flow mean log loss in unit-step (basis weight) space, if available.
    if limit_order_flow_model is not None and hasattr(limit_order_flow_model, "log_loss_hist"):
        lof_hist = list(getattr(limit_order_flow_model, "log_loss_hist", []))
        if n_dates is not None and n_dates > 0 and len(lof_hist) < n_dates:
            pad = n_dates - len(lof_hist)
            lof_hist = [0.0] * pad + lof_hist
        subplots.append({
            "title": "LOF mean NLL (unit-step, EWMA)",
            "ylabel": "EWMA −log p",
            "series": [{"label": "LOF mean NLL", "color": "purple", "y": lof_hist}],
        })
    # Mid forecast quality: running MSE of expected-mid forecast vs a naive previous-mid baseline.
    # Keep this subplot always present so Figure 2 axis count stays stable (avoid close/recreate flicker).
    exp_running_mse: list[float] = []
    prev_running_mse: list[float] = []
    if runner is not None and getattr(runner, "steps", None):
        steps = list(getattr(runner, "steps", []))
        expected_mid = list(getattr(runner, "expected_mid_hist", []))
        if len(steps) >= 2 and expected_mid:
            realized_mid = [float(s[1].mid) for s in steps]
            n_trans = len(realized_mid) - 1
            n_use = min(n_trans, len(expected_mid))
            if n_use > 0:
                exp_sq_err = []
                prev_sq_err = []
                for i in range(n_use):
                    y_true = realized_mid[i + 1]
                    y_prev = realized_mid[i]
                    y_exp = float(expected_mid[i])
                    exp_sq_err.append((y_exp - y_true) ** 2)
                    prev_sq_err.append((y_prev - y_true) ** 2)
                exp_running_mse = (np.cumsum(exp_sq_err) / np.arange(1, len(exp_sq_err) + 1)).tolist()
                prev_running_mse = (np.cumsum(prev_sq_err) / np.arange(1, len(prev_sq_err) + 1)).tolist()
    if n_dates is not None and n_dates > 0:
        exp_running_mse = [0.0] * max(0, n_dates - len(exp_running_mse)) + exp_running_mse
        prev_running_mse = [0.0] * max(0, n_dates - len(prev_running_mse)) + prev_running_mse
    subplots.append({
        "title": "Mid forecast MSE (running)",
        "ylabel": "MSE (ticks^2)",
        "series": [
            {"label": "Expected mid vs realized", "color": "darkorange", "y": exp_running_mse},
            {"label": "Previous mid vs realized", "color": "gray", "y": prev_running_mse},
        ],
    })
    return subplots


def run_animation(
    runner: OrderFlowRunner,
    exchange: str,
    symbol: str,
    T_ms: int,
    *,
    fixed_vol_max: float | None = None,
    orderbook_ticks: int = 50,
    decay: float = 0.99,
    spread_max: float | None = None,
    animation_interval_ms: int | None = None,
    max_steps_per_frame: int = 5,
    enable_loss_figure: bool = True,
    enable_action_figure: bool = True,
    enable_price_figure: bool = True,
    enable_qq_figure: bool = True,
    enable_hedge_figure: bool = True,
) -> None:
    """
    Run the animation loop. Figure layout is built from runner: estimator and tester
    declare subplots via get_plot_specs(); no fixed axis count.
    If spread_max is provided, depth subplots use it for x-axis limits.
    """
    bucket_ms = max(1, T_ms)
    bar_width_days = (bucket_ms / 1000.0) / 86400
    sections, height_ratios = build_layout(runner)
    n_total = sum(n for _, n in sections)
    ranges = section_ranges(sections)

    fig, axes = plt.subplots(
        n_total,
        1,
        figsize=(10, 4 + n_total * 0.8),
        height_ratios=height_ratios,
    )
    if n_total == 1:
        axes = [axes]
    # Layout fig 1 explicitly: more vertical space between subplots so titles/labels don't overlap.
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98), h_pad=1.2)

    # Figure 3: optimizer/components + depth/action diagnostics moved out of main window.
    fig_action = None
    axes_action = None
    if enable_action_figure:
        fig_action, axes_action = plt.subplots(5, 1, figsize=(10, 13.2), sharex=False)
        axes_action = np.atleast_1d(axes_action).ravel()
        fig_action.suptitle("Action & Depth diagnostics", fontsize=12)
        fig_action.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
    ranges_action = {
        "J_components": (0, 1),
        "main_hedge_rewards": (1, 2),
        "depth": (2, 3),
        "action_depth": (3, 5),
    }

    # Figure 4: price-model diagnostics (flow, rep prices, σ, μ under σ, MC pred error).
    fig_mid_price = None
    axes_mid_price = None
    if enable_price_figure:
        fig_mid_price, axes_mid_price = plt.subplots(5, 1, figsize=(10, 13.8), sharex=False)
        axes_mid_price = np.atleast_1d(axes_mid_price).ravel()
        fig_mid_price.suptitle("Price model diagnostics", fontsize=12)
        fig_mid_price.tight_layout()

    # Figure 5: QQ chart of observed trade volumes vs current learned flow model.
    fig_qq = None
    axes_qq = None
    if enable_qq_figure:
        fig_qq, axes_qq = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=False)
        axes_qq = np.atleast_1d(axes_qq).ravel()
        fig_qq.suptitle("Trade volume QQ: observed vs learned model", fontsize=12)
        fig_qq.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    # Figure 6: main/hedge representative buy/sell prices in a single subplot.
    fig_hedge = None
    axes_hedge = None
    if enable_hedge_figure:
        fig_hedge, axes_hedge = plt.subplots(3, 1, figsize=(10, 11.0), sharex=False)
        axes_hedge = np.atleast_1d(axes_hedge).ravel()
        fig_hedge.suptitle("Main vs hedge representative prices", fontsize=12)
        fig_hedge.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))

    # Figure 2 — NLL / predictive log loss. Must be created before plt.show(); figures made only
    # inside FuncAnimation callbacks often never get a window on Qt/GTk backends.
    loss_state: dict = {"fig": None, "axes": None, "n": 0}

    def _ensure_loss_figure(n_axes: int) -> None:
        n_axes = max(1, int(n_axes))
        if loss_state["fig"] is not None:
            return
        loss_state["n"] = n_axes
        fig_l, axarr = plt.subplots(n_axes, 1, figsize=(10, 2.0 + n_axes * 0.72), sharex=False)
        loss_state["fig"] = fig_l
        loss_state["axes"] = [axarr] if n_axes == 1 else np.atleast_1d(axarr).ravel().tolist()
        loss_state["fig"].suptitle("Figure 2 — NLL & predictive log loss", fontsize=12)
        loss_state["fig"].tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    if enable_loss_figure:
        subplots_loss_init = _log_loss_subplots_data(
            runner.tester,
            getattr(runner, "price_tester", None),
            limit_order_flow_model=runner.limit_order_flow_model,
            n_dates=len(runner.steps),
            runner=runner,
            market_flow_model=getattr(runner, "mof_model", None),
        )
        if not subplots_loss_init:
            subplots_loss_init = [{"title": "Log loss (NLL)", "ylabel": "EWMA −log p", "series": []}]
        _ensure_loss_figure(len(subplots_loss_init))
        dates_init = [
            datetime.fromtimestamp((s[0] - bucket_ms / 2) / 1000.0, tz=timezone.utc) for s in runner.steps
        ]
        draw_log_loss_window(loss_state["axes"], dates_init, subplots_loss_init, loss_state["fig"])

    def init():
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        if axes_action is not None:
            for ax_ad in axes_action:
                ax_ad.set_xlim(0, 1)
                ax_ad.set_ylim(-1, 1)
        if axes_mid_price is not None:
            for ax_mp in axes_mid_price:
                ax_mp.set_xlim(0, 1)
                ax_mp.set_ylim(-1, 1)
        for ax_l in loss_state["axes"] or []:
            ax_l.set_xlim(0, 1)
            ax_l.set_ylim(0, 1)
        return ()

    if animation_interval_ms is None:
        # Keep UI responsive while stepping is controlled by runner's scheduler.
        animation_interval_ms = min(250, max(20, T_ms // 10))

    def update(_frame):
        runner.advance_scheduled(max_steps_per_call=max_steps_per_frame)
        # Always redraw so depth and all panels stay in sync with current runner state.
        mof_model = getattr(runner, "mof_model", None)
        param_plot_data = mof_model.get_plot_data() if mof_model is not None else {}
        dates = [datetime.fromtimestamp((s[0] - bucket_ms / 2) / 1000.0, tz=timezone.utc) for s in runner.steps]
        draw_dynamic(
            runner.steps,
            bucket_ms,
            bar_width_days,
            fixed_vol_max,
            exchange,
            symbol,
            axes,
            ranges,
            param_plot_data,
            {},  # score no longer in main window
            fig,
            decay,
            runner.action_times,
            runner.delta_b_hist,
            runner.delta_a_hist,
            runner.q_b_hist,
            runner.q_a_hist,
            runner.inv_times,
            runner.inv_hist,
            runner.wealth_times,
            runner.cash_hist,
            runner.inv_value_hist,
            runner.wealth_hist,
            runner=runner,
            spread_max=spread_max,
        )
        if axes_action is not None and fig_action is not None:
            draw_dynamic(
                runner.steps,
                bucket_ms,
                bar_width_days,
                fixed_vol_max,
                exchange,
                symbol,
                axes_action,
                ranges_action,
                param_plot_data,
                {},
                fig_action,
                decay,
                runner.action_times,
                runner.delta_b_hist,
                runner.delta_a_hist,
                runner.q_b_hist,
                runner.q_a_hist,
                runner.inv_times,
                runner.inv_hist,
                runner.wealth_times,
                runner.cash_hist,
                runner.inv_value_hist,
                runner.wealth_hist,
                runner=runner,
                spread_max=spread_max,
            )
            fig_action.canvas.draw_idle()
        if axes_mid_price is not None and fig_mid_price is not None:
            draw_mid_price_window(axes_mid_price, dates, runner, fig_mid_price)
            fig_mid_price.canvas.draw_idle()
        if axes_qq is not None and fig_qq is not None:
            draw_volume_qq_window(axes_qq, runner, fig_qq)
            fig_qq.canvas.draw_idle()
        if axes_hedge is not None and fig_hedge is not None:
            draw_hedge_reference_window(axes_hedge, runner, fig_hedge)
            fig_hedge.canvas.draw_idle()
        if enable_loss_figure:
            subplots_loss = _log_loss_subplots_data(
                runner.tester,
                getattr(runner, "price_tester", None),
                limit_order_flow_model=runner.limit_order_flow_model,
                n_dates=len(dates),
                runner=runner,
                market_flow_model=getattr(runner, "mof_model", None),
            )
            if not subplots_loss:
                subplots_loss = [{"title": "Log loss", "ylabel": "", "series": []}]
            _ensure_loss_figure(len(subplots_loss))
            draw_log_loss_window(loss_state["axes"], dates, subplots_loss, loss_state["fig"])
            if loss_state["fig"] is not None:
                loss_state["fig"].canvas.draw_idle()
        return ()

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=animation_interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    # Re-apply layout to fig (main window) so it stays correct; avoid touching other figures.
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98), h_pad=1.2)
    plt.show()


def _to_scalar(v) -> float:
    try:
        if hasattr(v, "detach"):
            import torch
            t = v.detach()
            if isinstance(t, torch.Tensor):
                return float(t.cpu().reshape(-1)[0].item())
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        return float(arr[0]) if arr.size > 0 else 0.0
    except Exception:
        return float(v)


def _sample_model_volumes(model: object, n_samples: int) -> tuple[list[float], list[float]]:
    buy: list[float] = []
    sell: list[float] = []
    if model is None or not hasattr(model, "sample"):
        return buy, sell
    for _ in range(max(1, int(n_samples))):
        try:
            bv, sv = model.sample(())
            b = max(0.0, _to_scalar(bv))
            s = max(0.0, _to_scalar(sv))
            if np.isfinite(b) and np.isfinite(s):
                buy.append(float(b))
                sell.append(float(s))
        except Exception:
            continue
    return buy, sell


def draw_volume_qq_window(axes_qq, runner: OrderFlowRunner, fig) -> None:
    """Figure 5: QQ charts (buy/sell) of observed volumes vs current learned model.

    Observed quantiles use only the last ``QQ_OBS_WINDOW`` steps (not full history).
    """
    ax_buy = axes_qq[0]
    ax_sell = axes_qq[1]
    for ax in (ax_buy, ax_sell):
        ax.clear()
        ax.set_xlabel("Model quantile")
        ax.set_ylabel("Observed quantile")
        ax.grid(alpha=0.25, linestyle=":")
    ax_buy.set_title("Buy volume QQ")
    ax_sell.set_title("Sell volume QQ")

    steps = list(getattr(runner, "steps", []) or [])
    if QQ_OBS_WINDOW > 0:
        steps = steps[-QQ_OBS_WINDOW:]
    obs_buy = [max(0.0, float(s[2])) for s in steps]
    obs_sell = [max(0.0, float(s[3])) for s in steps]
    n_obs = min(len(obs_buy), len(obs_sell))
    if n_obs < 8:
        for ax in (ax_buy, ax_sell):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "Waiting for more data...", ha="center", va="center", transform=ax.transAxes)
        if fig:
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
        return

    n_sim = min(1500, max(300, n_obs))
    sim_buy, sim_sell = _sample_model_volumes(getattr(runner, "mof_model", None), n_sim)
    if len(sim_buy) < 8 or len(sim_sell) < 8:
        for ax in (ax_buy, ax_sell):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "Model samples unavailable", ha="center", va="center", transform=ax.transAxes)
        if fig:
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
        return

    q = np.linspace(0.02, 0.98, 49)
    obs_buy_q = np.quantile(np.asarray(obs_buy, dtype=np.float64), q)
    obs_sell_q = np.quantile(np.asarray(obs_sell, dtype=np.float64), q)
    sim_buy_q = np.quantile(np.asarray(sim_buy, dtype=np.float64), q)
    sim_sell_q = np.quantile(np.asarray(sim_sell, dtype=np.float64), q)

    def _draw(ax, x, y, side: str):
        ax.scatter(x, y, s=16, alpha=0.75, color="tab:blue", label="QQ points")
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = max(1e-6, 0.05 * (hi - lo if hi > lo else 1.0))
        lo_p, hi_p = lo - pad, hi + pad
        ax.plot([lo_p, hi_p], [lo_p, hi_p], color="black", linestyle="--", linewidth=1.0, label="y=x")
        ax.set_xlim(lo_p, hi_p)
        ax.set_ylim(lo_p, hi_p)
        ax.legend(loc="upper left")
        ax.set_title(f"{side} volume QQ (n_obs={n_obs}, n_model={len(sim_buy) if side=='Buy' else len(sim_sell)})")

    _draw(ax_buy, sim_buy_q, obs_buy_q, "Buy")
    _draw(ax_sell, sim_sell_q, obs_sell_q, "Sell")
    if fig:
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))


def draw_mid_price_window(axes_mid_price, dates: list, runner: OrderFlowRunner, fig) -> None:
    """Figure 4: buy/sell NLL, representative prices, residual σ, residual μ, MC prediction errors."""
    ax_flow = axes_mid_price[0]
    ax_price = axes_mid_price[1]
    ax_sigma = axes_mid_price[2]
    ax_mu = axes_mid_price[3]
    ax_pred_err = axes_mid_price[4]
    ax_flow.clear()
    ax_price.clear()
    ax_sigma.clear()
    ax_mu.clear()
    ax_pred_err.clear()
    ax_flow.set_ylabel("EWMA −log p")
    ax_flow.set_title("Market order flow NLL (buy / sell)")
    ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_flow.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price (ticks)")
    ax_price.set_title("Representative buy/sell price (EWMA VWAP to shared depth)")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_sigma.set_xlabel("Time")
    ax_sigma.set_ylabel("σ (ticks)")
    ax_sigma.set_title("Residual price model: σ (buy / sell)")
    ax_sigma.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_sigma.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_sigma.set_ylim(bottom=0)
    ax_mu.set_xlabel("Time")
    ax_mu.set_ylabel("μ (ticks)")
    ax_mu.set_title("Residual price model: μ (buy / sell)")
    ax_mu.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_mu.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_mu.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax_pred_err.set_xlabel("Time")
    ax_pred_err.set_ylabel("Error (ticks)")
    ax_pred_err.set_title("MC representative-price prediction error (mean prediction - realized)")
    ax_pred_err.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_pred_err.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pred_err.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    pm = getattr(runner, "price_model", None)
    if pm is None or not dates:
        ax_flow.set_xlim(0, 1)
        ax_flow.set_ylim(-1, 1)
        ax_price.set_xlim(0, 1)
        ax_price.set_ylim(0, 1)
        ax_sigma.set_xlim(0, 1)
        ax_sigma.set_ylim(0, 1)
        ax_mu.set_xlim(0, 1)
        ax_mu.set_ylim(-1, 1)
        ax_pred_err.set_xlim(0, 1)
        ax_pred_err.set_ylim(-1, 1)
        return
    steps = list(getattr(runner, "steps", []) or [])
    buy_raw = [max(0.0, float(s[2])) for s in steps]
    sell_raw = [max(0.0, float(s[3])) for s in steps]
    n = min(len(dates), len(buy_raw), len(sell_raw))
    if n <= 0:
        ax_flow.set_xlim(0, 1)
        ax_flow.set_ylim(-1, 1)
        ax_price.set_xlim(0, 1)
        ax_price.set_ylim(0, 1)
        ax_sigma.set_xlim(0, 1)
        ax_sigma.set_ylim(0, 1)
        ax_mu.set_xlim(0, 1)
        ax_mu.set_ylim(-1, 1)
        ax_pred_err.set_xlim(0, 1)
        ax_pred_err.set_ylim(-1, 1)
        return
    # Draw buy/sell NLL in one subplot for cleaner Figure 4.
    td = dates[:n]
    tester = getattr(runner, "tester", None)
    nll_series = []
    if tester is not None and hasattr(tester, "get_plot_data"):
        try:
            nll_series = list((tester.get_plot_data() or {}).get("series", []))
        except Exception:
            nll_series = []
    y_buy_nll = []
    y_sell_nll = []
    for ser in nll_series:
        label = str(ser.get("label", "")).lower()
        y = list(ser.get("y", []))
        if len(y) < n:
            y = [0.0] * (n - len(y)) + y
        y = y[:n]
        if "buy" in label and "nll" in label:
            y_buy_nll = [float(v) for v in y]
        elif "sell" in label and "nll" in label:
            y_sell_nll = [float(v) for v in y]
    if y_buy_nll or y_sell_nll:
        if y_buy_nll:
            ax_flow.plot(td, y_buy_nll, color="green", linewidth=1.2, label="Buy NLL")
        if y_sell_nll:
            ax_flow.plot(td, y_sell_nll, color="red", linewidth=1.2, label="Sell NLL")
        ax_flow.legend(loc="upper right")
        lo, hi = _finite_series_ylim(y_buy_nll + y_sell_nll, pad_min=0.2, default=(0.0, 1.0))
        ax_flow.set_ylim(lo, hi)
        if n > 1:
            ax_flow.set_xlim(td[0], td[-1])
    else:
        ax_flow.set_xlim(0, 1)
        ax_flow.set_ylim(-1, 1)

    buy_real_px = list(getattr(pm, "realized_buy_price_hist", []) or [])
    buy_cmp_px = list(getattr(pm, "computed_buy_price_hist", []) or [])
    sell_real_px = list(getattr(pm, "realized_sell_price_hist", []) or [])
    sell_cmp_px = list(getattr(pm, "computed_sell_price_hist", []) or [])
    sigma_buy = list(getattr(pm, "sigma_buy_hist", []) or [])
    sigma_sell = list(getattr(pm, "sigma_sell_hist", []) or [])
    mu_buy = list(getattr(pm, "mu_buy_hist", []) or [])
    mu_sell = list(getattr(pm, "mu_sell_hist", []) or [])

    n_p = min(len(dates) - 1, len(buy_real_px), len(buy_cmp_px), len(sell_real_px), len(sell_cmp_px))
    if n_p > 0:
        td_p = list(dates[1 : 1 + n_p])
        y_br = buy_real_px[:n_p]
        y_bc = buy_cmp_px[:n_p]
        y_sr = sell_real_px[:n_p]
        y_sc = sell_cmp_px[:n_p]
        ax_price.plot(td_p, y_br, color="navy", linewidth=1.2, label="Buy realized")
        ax_price.plot(td_p, y_bc, color="darkorange", linewidth=1.2, label="Buy computed (MOF-only)")
        ax_price.plot(td_p, y_sr, color="green", linewidth=1.2, label="Sell realized")
        ax_price.plot(td_p, y_sc, color="red", linewidth=1.2, label="Sell computed (MOF-only)")
        ax_price.legend(loc="upper right")
        lo_p, hi_p = _finite_series_ylim(y_br + y_bc + y_sr + y_sc, pad_min=1.0, default=(0.0, 1.0))
        ax_price.set_ylim(lo_p, hi_p)
        if n_p > 1:
            ax_price.set_xlim(td_p[0], td_p[-1])
    else:
        ax_price.set_xlim(0, 1)
        ax_price.set_ylim(0, 1)

    n_s_sig = min(len(dates) - 1, len(sigma_buy), len(sigma_sell))
    if n_s_sig > 0:
        td_s = list(dates[1 : 1 + n_s_sig])
        y_sb = sigma_buy[:n_s_sig]
        y_ss = sigma_sell[:n_s_sig]
        ax_sigma.plot(td_s, y_sb, color="blue", linewidth=1.2, label="σ buy")
        ax_sigma.plot(td_s, y_ss, color="red", linewidth=1.2, label="σ sell")
        ax_sigma.legend(loc="upper right")
        lo_s, hi_s = _finite_series_ylim(y_sb + y_ss, pad_min=0.1, default=(0.0, 1.0))
        ax_sigma.set_ylim(max(0.0, lo_s), hi_s)
        if n_s_sig > 1:
            ax_sigma.set_xlim(td_s[0], td_s[-1])
    else:
        ax_sigma.set_xlim(0, 1)
        ax_sigma.set_ylim(0, 1)

    n_s_mu = min(n_s_sig, len(mu_buy), len(mu_sell))
    if n_s_mu > 0:
        td_m = list(dates[1 : 1 + n_s_mu])
        y_mb = mu_buy[:n_s_mu]
        y_ms = mu_sell[:n_s_mu]
        ax_mu.plot(td_m, y_mb, color="cyan", linewidth=1.2, label="μ buy")
        ax_mu.plot(td_m, y_ms, color="magenta", linewidth=1.2, label="μ sell")
        ax_mu.legend(loc="upper right")
        lo_m, hi_m = _finite_series_ylim(y_mb + y_ms, pad_min=0.5, default=(-1.0, 1.0))
        if lo_m == hi_m:
            lo_m, hi_m = lo_m - 0.5, hi_m + 0.5
        ax_mu.set_ylim(lo_m, hi_m)
        if n_s_mu > 1:
            ax_mu.set_xlim(td_m[0], td_m[-1])
    else:
        ax_mu.set_xlim(0, 1)
        ax_mu.set_ylim(-1, 1)

    pred_buy_err = list(getattr(runner, "pred_buy_rep_error_hist", []) or [])
    pred_sell_err = list(getattr(runner, "pred_sell_rep_error_hist", []) or [])
    pred_buy_prev_err = list(getattr(runner, "pred_buy_rep_prev_error_hist", []) or [])
    pred_sell_prev_err = list(getattr(runner, "pred_sell_rep_prev_error_hist", []) or [])
    n_e = min(
        max(0, len(dates) - 1),
        len(pred_buy_err),
        len(pred_sell_err),
        len(pred_buy_prev_err),
        len(pred_sell_prev_err),
    )
    if n_e > 0:
        n_ec = min(n_e, n_p)
        td_e = list(dates[1 : 1 + n_ec])
        y_be = pred_buy_err[:n_ec]
        y_se = pred_sell_err[:n_ec]
        y_be_prev = pred_buy_prev_err[:n_ec]
        y_se_prev = pred_sell_prev_err[:n_ec]
        y_be_cmp = [float(bc - br) for br, bc in zip(y_br[:n_ec], y_bc[:n_ec])]
        y_se_cmp = [float(sc - sr) for sr, sc in zip(y_sr[:n_ec], y_sc[:n_ec])]
        ax_pred_err.plot(
            td_e,
            y_be,
            color="royalblue",
            linewidth=1.2,
            label="Buy MC pred error (MC pred − realized)",
        )
        ax_pred_err.plot(
            td_e,
            y_se,
            color="tomato",
            linewidth=1.2,
            label="Sell MC pred error (MC pred − realized)",
        )
        ax_pred_err.plot(
            td_e,
            y_be_cmp,
            color="deepskyblue",
            linewidth=1.0,
            linestyle=":",
            label="Buy computed error (realized MOF)",
        )
        ax_pred_err.plot(
            td_e,
            y_se_cmp,
            color="orange",
            linewidth=1.0,
            linestyle=":",
            label="Sell computed error (realized MOF)",
        )
        ax_pred_err.plot(
            td_e,
            y_be_prev,
            color="navy",
            linewidth=1.0,
            linestyle="--",
            label="Buy prev-price error",
        )
        ax_pred_err.plot(
            td_e,
            y_se_prev,
            color="darkred",
            linewidth=1.0,
            linestyle="--",
            label="Sell prev-price error",
        )
        ax_pred_err.legend(loc="upper right")
        lo_e, hi_e = _finite_series_ylim(
            y_be + y_se + y_be_cmp + y_se_cmp + y_be_prev + y_se_prev,
            pad_min=0.5,
            default=(-1.0, 1.0),
        )
        ax_pred_err.set_ylim(lo_e, hi_e)
        if n_e > 1:
            ax_pred_err.set_xlim(td_e[0], td_e[-1])
    else:
        ax_pred_err.set_xlim(0, 1)
        ax_pred_err.set_ylim(-1, 1)
    if fig:
        fig.autofmt_xdate()


def draw_hedge_reference_window(axes_hedge, runner: OrderFlowRunner, fig) -> None:
    """Figure 6: representative prices, hedge delta params, hedge inv/cash."""
    axes = np.atleast_1d(axes_hedge).ravel()
    ax_hedge = axes[0]
    ax_params = axes[1] if len(axes) > 1 else None
    ax_portfolio = axes[2] if len(axes) > 2 else None
    ax_hedge.clear()
    ax_hedge.set_xlabel("Time")
    ax_hedge.set_ylabel("Representative price (ticks)")
    ax_hedge.set_title("Main + hedge representative buy/sell")
    ax_hedge.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_hedge.xaxis.set_major_locator(mdates.AutoDateLocator())

    t_main = list(getattr(runner, "main_reference_times", []) or [])
    y_main_buy = list(getattr(runner, "main_buy_reference_hist", []) or [])
    y_main_sell = list(getattr(runner, "main_sell_reference_hist", []) or [])
    t_hedge = list(getattr(runner, "hedge_reference_times", []) or [])
    y_hedge_buy = list(getattr(runner, "hedge_buy_reference_hist", []) or [])
    y_hedge_sell = list(getattr(runner, "hedge_sell_reference_hist", []) or [])

    plotted_any = False
    all_y: list[float] = []
    if t_main and len(t_main) == len(y_main_buy) == len(y_main_sell):
        ax_hedge.plot(
            t_main,
            y_main_buy,
            color="navy",
            linewidth=1.4,
            label="Main buy rep",
        )
        ax_hedge.plot(
            t_main,
            y_main_sell,
            color="green",
            linewidth=1.4,
            label="Main sell rep",
        )
        all_y.extend(float(v) for v in y_main_buy)
        all_y.extend(float(v) for v in y_main_sell)
        plotted_any = True
    if t_hedge and len(t_hedge) == len(y_hedge_buy) == len(y_hedge_sell):
        ax_hedge.plot(
            t_hedge,
            y_hedge_buy,
            color="darkorange",
            linewidth=1.2,
            linestyle="--",
            label="Hedge buy rep",
        )
        ax_hedge.plot(
            t_hedge,
            y_hedge_sell,
            color="red",
            linewidth=1.2,
            linestyle="--",
            label="Hedge sell rep",
        )
        all_y.extend(float(v) for v in y_hedge_buy)
        all_y.extend(float(v) for v in y_hedge_sell)
        plotted_any = True

    if plotted_any:
        ax_hedge.legend(loc="upper right")
        t_all = t_main + t_hedge
        t_all.sort()
        if len(t_all) > 1:
            ax_hedge.set_xlim(t_all[0], t_all[-1])
        y_lo, y_hi = _finite_series_ylim(all_y, pad_min=1.0, default=(0.0, 1.0))
        ax_hedge.set_ylim(y_lo, y_hi)
    else:
        ax_hedge.set_xlim(0, 1)
        ax_hedge.set_ylim(0, 1)
        ax_hedge.text(
            0.5,
            0.5,
            "Waiting for representative prices...",
            ha="center",
            va="center",
            transform=ax_hedge.transAxes,
        )

    if ax_params is not None:
        ax_params.clear()
        ax_params.set_xlabel("Time")
        ax_params.set_ylabel("Delta ticks")
        ax_params.set_title("Hedge Gaussian learned delta params (EWMA MLE)")
        ax_params.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_params.xaxis.set_major_locator(mdates.AutoDateLocator())
        t_p = list(getattr(runner, "hedge_price_param_times", []) or [])
        mu_b = list(getattr(runner, "hedge_mu_buy_hist", []) or [])
        mu_s = list(getattr(runner, "hedge_mu_sell_hist", []) or [])
        sg_b = list(getattr(runner, "hedge_sigma_buy_hist", []) or [])
        sg_s = list(getattr(runner, "hedge_sigma_sell_hist", []) or [])
        n_p = min(len(t_p), len(mu_b), len(mu_s), len(sg_b), len(sg_s))
        if n_p > 0:
            t = t_p[:n_p]
            y_mu_b = [float(v) for v in mu_b[:n_p]]
            y_mu_s = [float(v) for v in mu_s[:n_p]]
            y_sg_b = [float(v) for v in sg_b[:n_p]]
            y_sg_s = [float(v) for v in sg_s[:n_p]]
            ax_params.plot(t, y_mu_b, color="darkorange", linewidth=1.2, label="mu_delta_buy")
            ax_params.plot(t, y_mu_s, color="red", linewidth=1.2, label="mu_delta_sell")
            ax_params.plot(t, y_sg_b, color="teal", linewidth=1.1, linestyle="--", label="sigma_delta_buy")
            ax_params.plot(t, y_sg_s, color="purple", linewidth=1.1, linestyle="--", label="sigma_delta_sell")
            ax_params.legend(loc="upper right", fontsize=8)
            if n_p > 1:
                ax_params.set_xlim(t[0], t[-1])
            lo, hi = _finite_series_ylim(y_mu_b + y_mu_s + y_sg_b + y_sg_s, pad_min=1.0, default=(0.0, 1.0))
            ax_params.set_ylim(lo, hi)
        else:
            ax_params.set_xlim(0, 1)
            ax_params.set_ylim(0, 1)
            ax_params.text(
                0.5,
                0.5,
                "Waiting for learned hedge params...",
                ha="center",
                va="center",
                transform=ax_params.transAxes,
            )

    if ax_portfolio is not None:
        ax_portfolio.clear()
        ax_portfolio.set_xlabel("Time")
        ax_portfolio.set_ylabel("Main units")
        ax_portfolio.set_title("Hedge inventory/cash (converted to main exchange units)")
        ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_portfolio.xaxis.set_major_locator(mdates.AutoDateLocator())
        t_a = list(getattr(runner, "hedge_portfolio_times", []) or [])
        inv_a = list(getattr(runner, "hedge_inventory_hist", []) or [])
        cash_a = list(getattr(runner, "hedge_cash_hist", []) or [])
        n_a = min(len(t_a), len(inv_a), len(cash_a))
        if n_a > 0:
            t = t_a[:n_a]
            y_inv = [float(v) for v in inv_a[:n_a]]
            y_cash = [float(v) for v in cash_a[:n_a]]
            ax_portfolio.plot(t, y_inv, color="navy", linewidth=1.2, label="Hedge inventory (lots)")
            ax_portfolio.plot(t, y_cash, color="darkgreen", linewidth=1.2, label="Hedge cash (tick*lot)")
            ax_portfolio.legend(loc="upper right", fontsize=8)
            if n_a > 1:
                ax_portfolio.set_xlim(t[0], t[-1])
            lo, hi = _finite_series_ylim(y_inv + y_cash, pad_min=1.0, default=(0.0, 1.0))
            ax_portfolio.set_ylim(lo, hi)
        else:
            ax_portfolio.set_xlim(0, 1)
            ax_portfolio.set_ylim(0, 1)
            ax_portfolio.text(
                0.5,
                0.5,
                "Waiting for hedge inventory/cash...",
                ha="center",
                va="center",
                transform=ax_portfolio.transAxes,
            )
    if fig:
        fig.autofmt_xdate()


def draw_log_loss_window(
    axes_loss: list,
    dates: list,
    subplots_data: list[dict],
    fig_loss,
) -> None:
    """Draw the predictive log loss window: one subplot per loss (Buy NLL, Sell NLL, Δmid NLL)."""
    for i, ax in enumerate(axes_loss):
        ax.clear()
        if i < len(subplots_data):
            _draw_line_subplot(ax, dates, subplots_data[i], fig_loss)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)


def _draw_line_subplot(ax, dates, subplot_data: dict, fig, y_min_bottom: bool = True) -> None:
    """Draw one subplot from subplot_data (title, ylabel, series). Use subplot_data.get('y_min_bottom', True) to allow negative y."""
    ax.clear()
    ax.set_xlabel("Time")
    ax.set_ylabel(subplot_data.get("ylabel", ""))
    ax.set_title(subplot_data.get("title", ""))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    y_min_bottom = subplot_data.get("y_min_bottom", y_min_bottom)
    if y_min_bottom:
        ax.set_ylim(bottom=0)
    series = subplot_data.get("series", [])
    if dates and series and all(len(s.get("y", [])) == len(dates) for s in series):
        for s in series:
            ax.plot(dates, s["y"], color=s.get("color", "blue"), linewidth=1.5, label=s.get("label", ""))
        ax.legend(loc="upper right")
        x_min, x_max = dates[0], dates[-1]
        if x_max <= x_min:
            x_min = x_min - timedelta(seconds=1)
            x_max = x_max + timedelta(seconds=1)
        ax.set_xlim(x_min, x_max)
        ylim_fixed = subplot_data.get("ylim")
        if ylim_fixed is not None:
            ax.set_ylim(ylim_fixed[0], ylim_fixed[1])
        else:
            all_y = [v for s in series for v in s.get("y", [])]
            finite_y = [v for v in all_y if np.isfinite(v)]
            y_max = max(finite_y) if finite_y else 1.0
            top = max(y_max * 1.1, 1e-9)
            if np.isfinite(top):
                ax.set_ylim(0, top) if y_min_bottom else ax.autoscale(axis="y")
            else:
                ax.autoscale(axis="y")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    if fig:
        fig.autofmt_xdate()


def draw_dynamic(
    steps_data: list[Step],
    bucket_ms: int,
    bar_width: float,
    v_scale: float | None,
    exchange: str,
    symbol: str,
    axes: list,
    ranges: dict[str, tuple[int, int]],
    param_plot_data: dict,
    score_plot_data: dict,
    fig,
    decay: float,
    action_times: list | None,
    delta_b_hist: list | None,
    delta_a_hist: list | None,
    q_b_hist: list | None,
    q_a_hist: list | None,
    inv_times: list | None,
    inv_hist: list | None,
    wealth_times: list | None,
    cash_hist: list | None,
    inv_value_hist: list | None,
    wealth_hist: list | None,
    runner: OrderFlowRunner | None = None,
    spread_max: float | None = None,
) -> None:
    if not steps_data:
        return
    dates = [datetime.fromtimestamp((s[0] - bucket_ms / 2) / 1000.0, tz=timezone.utc) for s in steps_data]
    buy_volumes = [s[2] for s in steps_data]
    sell_volumes = [s[3] for s in steps_data]
    buy_vol_arr = np.maximum(np.asarray(buy_volumes, dtype=np.float64), 0.0)
    sell_vol_arr = np.maximum(np.asarray(sell_volumes, dtype=np.float64), 0.0)
    buy_vol_log = np.log1p(buy_vol_arr)
    sell_vol_log = np.log1p(sell_vol_arr)
    action_times = action_times or []
    delta_b_hist = delta_b_hist or []
    delta_a_hist = delta_a_hist or []
    q_b_hist = q_b_hist or []
    q_a_hist = q_a_hist or []
    inv_times = inv_times or []
    inv_hist = inv_hist or []
    wealth_times = wealth_times or []
    cash_hist = cash_hist or []
    wealth_hist = wealth_hist or []

    # --- Market order flow (buy/sell volumes) ---
    start, end = ranges.get("market_order_flow", (0, 1))
    if end > start:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("ln(1 + volume)")
        ax.set_title(
            f"Market order flow (ln(1+vol)) — {exchange} {symbol} "
            f"(bars=realized; dashed=MC mean of model.sample())"
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        ax.bar(dates, buy_vol_log, width=bar_width, color="green", alpha=0.8, label="Market buy")
        ax.bar(dates, -sell_vol_log, width=bar_width, color="red", alpha=0.8, label="Market sell")
        if runner is not None:
            exp_b = list(getattr(runner, "expected_buy_flow_hist", []) or [])
            exp_s = list(getattr(runner, "expected_sell_flow_hist", []) or [])
            n_e = min(len(dates), len(exp_b), len(exp_s))
            if n_e > 0:
                td_e = dates[:n_e]
                e_buy_log = [float(np.log1p(max(0.0, float(x)))) for x in exp_b[:n_e]]
                e_sell_log = [float(-np.log1p(max(0.0, float(x)))) for x in exp_s[:n_e]]
                ax.plot(
                    td_e,
                    e_buy_log,
                    color="darkgreen",
                    linewidth=1.2,
                    linestyle="--",
                    label="E[Buy MOF] (MC)",
                )
                ax.plot(
                    td_e,
                    e_sell_log,
                    color="darkred",
                    linewidth=1.2,
                    linestyle="--",
                    label="E[Sell MOF] (MC)",
                )
        ax.legend(loc="upper right")
        max_log = 0.0
        if buy_vol_log.size or sell_vol_log.size:
            max_log = float(max(buy_vol_log.max(initial=0.0), sell_vol_log.max(initial=0.0)))
        if runner is not None:
            exp_b = list(getattr(runner, "expected_buy_flow_hist", []) or [])
            exp_s = list(getattr(runner, "expected_sell_flow_hist", []) or [])
            n_e = min(len(dates), len(exp_b), len(exp_s))
            if n_e > 0:
                for x in exp_b[:n_e]:
                    max_log = max(max_log, float(np.log1p(max(0.0, float(x)))))
                for x in exp_s[:n_e]:
                    max_log = max(max_log, float(np.log1p(max(0.0, float(x)))))
        if v_scale is not None and v_scale > 0:
            scale = float(np.log1p(v_scale)) * 1.15
        else:
            scale = max(max_log, 1e-9) * 1.15
        ax.set_ylim(-scale, scale)
        if dates:
            t0 = (steps_data[0][0] - bucket_ms) / 1000.0
            t1 = steps_data[-1][0] / 1000.0
            ax.set_xlim(datetime.fromtimestamp(t0, tz=timezone.utc), datetime.fromtimestamp(t1, tz=timezone.utc))
        if fig:
            fig.autofmt_xdate()

    # --- Estimator (dynamic subplots from param_plot_data) ---
    start, end = ranges.get("estimator", (0, 0))
    subplots = param_plot_data.get("subplots")
    if subplots is None:
        subplots = [{"title": param_plot_data.get("title", "Estimator"), "ylabel": param_plot_data.get("ylabel", ""), "series": param_plot_data.get("series", [])}]
    for i in range(end - start):
        if i < len(subplots):
            _draw_line_subplot(axes[start + i], dates, subplots[i], fig)
        else:
            axes[start + i].clear()
            axes[start + i].set_xlim(0, 1)
            axes[start + i].set_ylim(0, 1)

    # --- Price estimator (μ_mid, σ_mid for Δmid ~ N(μ, σ²)) ---
    start, end = ranges.get("price_model", (0, 0))
    if end > start and runner is not None:
        pe = getattr(runner, "price_model", None)
        if pe is not None and hasattr(pe, "get_plot_data"):
            price_plot_data = pe.get_plot_data()
            price_subplots = price_plot_data.get("subplots")
            if price_subplots is None:
                price_subplots = [{"title": price_plot_data.get("title", "Price"), "ylabel": price_plot_data.get("ylabel", ""), "series": price_plot_data.get("series", [])}]
            for i in range(end - start):
                if i < len(price_subplots):
                    _draw_line_subplot(axes[start + i], dates, price_subplots[i], fig)
                else:
                    axes[start + i].clear()
                    axes[start + i].set_xlim(0, 1)
                    axes[start + i].set_ylim(0, 1)
        else:
            for i in range(end - start):
                axes[start + i].clear()
                axes[start + i].set_xlim(0, 1)
                axes[start + i].set_ylim(0, 1)

    # --- LOF observation (basis weights: pinv(B) @ obs) ---
    start, end = ranges.get("limit_order_flow_model", (0, 0))
    if end > start and runner is not None:
        lof = getattr(runner, "limit_order_flow_model", None)
        if lof is not None and hasattr(lof, "get_plot_data"):
            lof_plot_data = lof.get_plot_data()
            lof_subplots = lof_plot_data.get("subplots") or []
            for i in range(end - start):
                ax = axes[start + i]
                ax.clear()
                if i < len(lof_subplots):
                    subplot_dict = lof_subplots[i]
                    if subplot_dict.get("kind") == "weights":
                        n_basis = subplot_dict.get("n_basis", 0)
                        w_bid = subplot_dict.get("w_bid", [])
                        w_ask = subplot_dict.get("w_ask", [])
                        if n_basis > 0 and len(w_bid) == n_basis and len(w_ask) == n_basis:
                            x = np.arange(n_basis, dtype=np.float64)
                            ax.plot(x, w_bid, color="green", linewidth=1.2, label="Bid (obs)")
                            ax.plot(x, w_ask, color="red", linewidth=1.2, label="Ask (obs)")
                            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
                            ax.set_xlabel("Basis index")
                            ax.set_ylabel(subplot_dict.get("ylabel", "Weight"))
                            ax.set_title(subplot_dict.get("title", "Cumulative order flow from mid — basis weights"))
                            ax.legend(loc="upper right", fontsize=8)
                            ax.autoscale(axis="y")
                        else:
                            ax.set_title(subplot_dict.get("title", "Cumulative from mid — basis weights"))
                            ax.set_xlim(0, 1)
                            ax.set_ylim(-1, 1)
                    else:
                        if subplot_dict.get("series") is not None:
                            # LOF line series (e.g. ||Beta|| history) may start after first step.
                            # Pad at front so len(y) matches len(dates) and plotting is not dropped.
                            series = list(subplot_dict.get("series", []))
                            if dates and series:
                                series_padded = []
                                for s in series:
                                    y = list(s.get("y", []))
                                    if len(y) < len(dates):
                                        y = [0.0] * (len(dates) - len(y)) + y
                                    elif len(y) > len(dates):
                                        y = y[-len(dates):]
                                    series_padded.append({**s, "y": y})
                                subplot_padded = {**subplot_dict, "series": series_padded}
                                _draw_line_subplot(ax, dates, subplot_padded, fig)
                            else:
                                _draw_line_subplot(ax, dates, subplot_dict, fig)
                        else:
                            ax.set_title(subplot_dict.get("title", "LOF"))
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                else:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
        else:
            for i in range(end - start):
                axes[start + i].clear()
                axes[start + i].set_xlim(0, 1)
                axes[start + i].set_ylim(0, 1)

    # --- LOF estimated μ and σ per weight (aligned by basis index) ---
    start, end = ranges.get("lof_mu_sigma", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        lof = getattr(runner, "limit_order_flow_model", None)
        if lof is not None and hasattr(lof, "get_params"):
            params = lof.get_params()
            mu_bid = np.asarray(params.get("mu_bid", []), dtype=np.float64)
            mu_ask = np.asarray(params.get("mu_ask", []), dtype=np.float64)
            sigma_bid = np.asarray(params.get("sigma_bid", []), dtype=np.float64)
            sigma_ask = np.asarray(params.get("sigma_ask", []), dtype=np.float64)
            n_basis = len(mu_bid)
            if n_basis > 0 and len(mu_ask) == n_basis and len(sigma_bid) == n_basis and len(sigma_ask) == n_basis:
                x = np.arange(n_basis, dtype=np.float64)
                ax.plot(x, mu_bid, color="green", linewidth=1.2, label="μ bid")
                ax.plot(x, mu_ask, color="red", linewidth=1.2, label="μ ask")
                ax.plot(x, sigma_bid, color="green", linewidth=1.0, linestyle="--", label="σ bid")
                ax.plot(x, sigma_ask, color="red", linewidth=1.0, linestyle="--", label="σ ask")
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
                ax.set_xlabel("Basis index")
                ax.set_ylabel("μ, σ")
                ax.set_title("LOF weight estimates (μ and σ per basis index)")
                ax.legend(loc="upper right", fontsize=8)
                ax.autoscale(axis="y")
            else:
                ax.set_title("LOF weight estimates (μ, σ)")
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
        else:
            ax.set_title("LOF weight estimates (no estimator)")
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)

    # --- Score (tester) ---
    start, end = ranges.get("score", (0, 0))
    if end > start:
        score_subplots = score_plot_data.get("subplots")
        if score_subplots is None:
            score_subplots = [{
                "title": score_plot_data.get("title", "Score"),
                "ylabel": score_plot_data.get("ylabel", ""),
                "series": score_plot_data.get("series", []),
                "ylim": score_plot_data.get("ylim"),
            }]
        _draw_line_subplot(axes[start], dates, score_subplots[0], fig)

    # --- Inv ---
    start, end = ranges.get("inv", (0, 0))
    if end > start:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Inventory (lots)")
        ax.set_title("Inventory")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        if inv_times and inv_hist:
            ax.plot(inv_times, inv_hist, color="blue", linewidth=1.5, label="Inventory")
            ax.legend(loc="upper left")
        if inv_times or wealth_times:
            t_min = (inv_times or wealth_times)[0]
            t_max = (inv_times or wealth_times)[-1]
            if t_min != t_max:
                ax.set_xlim(t_min, t_max)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Cash ---
    start, end = ranges.get("cash", (0, 0))
    if end > start:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Cash (tick×lot)")
        ax.set_title("Cash")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        if inv_times and cash_hist and len(cash_hist) == len(inv_times):
            ax.plot(inv_times, cash_hist, color="green", linewidth=1.5, label="Cash")
            ax.legend(loc="upper left")
            if cash_hist:
                low, high = _finite_series_ylim(cash_hist, pad_min=1.0)
                if high <= low:
                    high = low + max(abs(low) * 0.1, 1.0)
                ax.set_ylim(low, high)
        if inv_times or wealth_times:
            t_min = (inv_times or wealth_times)[0]
            t_max = (inv_times or wealth_times)[-1]
            if t_min != t_max:
                ax.set_xlim(t_min, t_max)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Wealth ---
    start, end = ranges.get("wealth", (0, 0))
    if end > start:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("(tick_size × lot_size)")
        ax.set_title("Wealth")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        wealth_from_dw_times = getattr(runner, "wealth_from_dw_times", None) or []
        wealth_from_dw_hist = getattr(runner, "wealth_from_dw_hist", None) or []
        plotted = False
        if wealth_times and wealth_hist and len(wealth_hist) == len(wealth_times):
            ax.plot(wealth_times, wealth_hist, color="orange", linewidth=1.5, label="Wealth")
            plotted = True
        if wealth_from_dw_times and wealth_from_dw_hist and len(wealth_from_dw_hist) == len(wealth_from_dw_times):
            ax.plot(
                wealth_from_dw_times,
                wealth_from_dw_hist,
                color="blue",
                linewidth=1.2,
                linestyle="--",
                label="Wealth from cum(ΔW)",
            )
            plotted = True
        if plotted:
            ax.legend(loc="upper right")
            x_ref = wealth_times if wealth_times else wealth_from_dw_times
            t_min, t_max = x_ref[0], x_ref[-1]
            if t_min != t_max:
                ax.set_xlim(t_min, t_max)
            y_for_ylim = list(wealth_hist)
            if wealth_from_dw_hist:
                y_for_ylim.extend(float(v) for v in wealth_from_dw_hist)
            if y_for_ylim:
                lo, hi = _finite_series_ylim(y_for_ylim, pad_min=1.0)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Δ spread PnL (exact decomposition) ---
    start, end = ranges.get("delta_spread_pnl", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Δ spread PnL")
        ax.set_title("Delta spread PnL")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        pnl_times = getattr(runner, "pnl_times", None) or []
        d_spread = getattr(runner, "delta_spread_pnl_hist", None) or []
        if pnl_times and d_spread and len(pnl_times) == len(d_spread):
            ax.step(pnl_times, d_spread, where="mid", color="green", linewidth=1.5, label="Δ spread PnL")
            ax.legend(loc="upper right")
            if len(pnl_times) > 1:
                ax.set_xlim(pnl_times[0], pnl_times[-1])
            if d_spread:
                lo, hi = _finite_series_ylim(d_spread)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Δ inventory PnL ---
    start, end = ranges.get("delta_inventory_pnl", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Δ inventory PnL")
        ax.set_title("Delta inventory PnL")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        pnl_times = getattr(runner, "pnl_times", None) or []
        d_inv = getattr(runner, "delta_inventory_pnl_hist", None) or []
        if pnl_times and d_inv and len(pnl_times) == len(d_inv):
            ax.step(pnl_times, d_inv, where="mid", color="blue", linewidth=1.5, label="Δ inventory PnL")
            ax.legend(loc="upper right")
            if len(pnl_times) > 1:
                ax.set_xlim(pnl_times[0], pnl_times[-1])
            if d_inv:
                lo, hi = _finite_series_ylim(d_inv)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Δ adverse PnL (cross term) ---
    start, end = ranges.get("delta_adverse_pnl", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Δ adverse PnL")
        ax.set_title("Delta adverse PnL (cross term)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        pnl_times = getattr(runner, "pnl_times", None) or []
        d_cross = getattr(runner, "delta_cross_pnl_hist", None) or []
        if pnl_times and d_cross and len(pnl_times) == len(d_cross):
            ax.step(pnl_times, d_cross, where="mid", color="red", linewidth=1.5, label="Δ adverse PnL")
            ax.legend(loc="upper right")
            if len(pnl_times) > 1:
                ax.set_xlim(pnl_times[0], pnl_times[-1])
            if d_cross:
                lo, hi = _finite_series_ylim(d_cross)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()


    # --- Δ fee PnL ---
    start, end = ranges.get("delta_fee_pnl", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Δ fee PnL")
        ax.set_title("Delta fee PnL")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        pnl_times = getattr(runner, "pnl_times", None) or []
        d_fee = getattr(runner, "delta_fee_pnl_hist", None) or []
        if pnl_times and d_fee and len(pnl_times) == len(d_fee):
            ax.step(pnl_times, d_fee, where="mid", color="purple", linewidth=1.5, label="Δ fee PnL")
            ax.legend(loc="upper right")
            if len(pnl_times) > 1:
                ax.set_xlim(pnl_times[0], pnl_times[-1])
            if d_fee:
                lo, hi = _finite_series_ylim(d_fee)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Realized ΔW from state_after_step ---
    start, end = ranges.get("J_components", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Objective components")
        ax.set_title("Optimizer J components")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        t_comp = getattr(runner, "J_comp_times", None) or []
        j_mean = getattr(runner, "J_mean_hist", None) or []
        j_inv = getattr(runner, "J_inv_hist", None) or []
        j_std = getattr(runner, "J_std_hist", None) or []
        j_uni = getattr(runner, "J_uniformity_hist", None) or []
        j_spread = getattr(runner, "J_spread_hist", None) or []
        j_mean_init = getattr(runner, "J_mean_init_hist", None) or []
        j_inv_init = getattr(runner, "J_inv_init_hist", None) or []
        j_std_init = getattr(runner, "J_std_init_hist", None) or []
        j_uni_init = getattr(runner, "J_uniformity_init_hist", None) or []
        j_spread_init = getattr(runner, "J_spread_init_hist", None) or []
        n_comp = min(len(t_comp), len(j_mean), len(j_inv), len(j_std), len(j_uni), len(j_spread))
        if n_comp > 0:
            t = t_comp[:n_comp]
            y_mean = [float(v) for v in j_mean[:n_comp]]
            y_inv = [float(v) for v in j_inv[:n_comp]]
            y_std = [float(v) for v in j_std[:n_comp]]
            y_uni = [float(v) for v in j_uni[:n_comp]]
            y_spread = [float(v) for v in j_spread[:n_comp]]
            ax.plot(t, y_mean, color="blue", linewidth=1.2, label="reward_coef * J_mean (opt)")
            ax.plot(t, y_inv, color="red", linewidth=1.2, label="-inv_coef * J_inv_penalty (opt)")
            ax.plot(t, y_std, color="purple", linewidth=1.2, label="-risk_coef * J_std (opt)")
            ax.plot(t, y_uni, color="green", linewidth=1.2, label="-uniformity_coef * J_uniformity (opt)")
            ax.plot(t, y_spread, color="black", linewidth=1.2, label="-spread_coef * J_spread (opt)")
            n_init = min(len(t_comp), len(j_mean_init), len(j_inv_init), len(j_std_init), len(j_uni_init), len(j_spread_init))
            if n_init > 0:
                t_i = t_comp[:n_init]
                y_mean_i = [float(v) for v in j_mean_init[:n_init]]
                y_inv_i = [float(v) for v in j_inv_init[:n_init]]
                y_std_i = [float(v) for v in j_std_init[:n_init]]
                y_uni_i = [float(v) for v in j_uni_init[:n_init]]
                y_spread_i = [float(v) for v in j_spread_init[:n_init]]
                ax.plot(t_i, y_mean_i, color="blue", linewidth=1.1, linestyle=":", label="reward_coef * J_mean (init)")
                ax.plot(t_i, y_inv_i, color="red", linewidth=1.1, linestyle=":", label="-inv_coef * J_inv_penalty (init)")
                ax.plot(t_i, y_std_i, color="purple", linewidth=1.1, linestyle=":", label="-risk_coef * J_std (init)")
                ax.plot(t_i, y_uni_i, color="green", linewidth=1.1, linestyle=":", label="-uniformity_coef * J_uniformity (init)")
                ax.plot(t_i, y_spread_i, color="black", linewidth=1.1, linestyle=":", label="-spread_coef * J_spread (init)")
            ax.legend(loc="upper right", fontsize=7)
            if n_comp > 1:
                ax.set_xlim(t[0], t[-1])
            all_y = y_mean + y_inv + y_std + y_uni + y_spread
            if n_init > 0:
                all_y += y_mean_i + y_inv_i + y_std_i + y_uni_i + y_spread_i
            lo, hi = _finite_series_ylim(all_y, pad_min=1.0)
            ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Main/hedge reward (init vs opt) ---
    start, end = ranges.get("main_hedge_rewards", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Reward (tick×lot)")
        ax.set_title("Main and hedge rewards (initial vs optimal)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        t_comp = getattr(runner, "J_comp_times", None) or []
        main_opt = getattr(runner, "main_reward_opt_hist", None) or []
        hedge_opt = getattr(runner, "hedge_reward_opt_hist", None) or []
        main_init = getattr(runner, "main_reward_init_hist", None) or []
        hedge_init = getattr(runner, "hedge_reward_init_hist", None) or []
        n_opt = min(len(t_comp), len(main_opt), len(hedge_opt))
        if n_opt > 0:
            t = t_comp[:n_opt]
            y_main_opt = [float(v) for v in main_opt[:n_opt]]
            y_hedge_opt = [float(v) for v in hedge_opt[:n_opt]]
            ax.plot(t, y_main_opt, color="blue", linewidth=1.2, label="Main reward (opt)")
            ax.plot(t, y_hedge_opt, color="orange", linewidth=1.2, label="Hedge reward (opt)")
            n_init = min(len(t_comp), len(main_init), len(hedge_init))
            if n_init > 0:
                t_i = t_comp[:n_init]
                y_main_init = [float(v) for v in main_init[:n_init]]
                y_hedge_init = [float(v) for v in hedge_init[:n_init]]
                ax.plot(t_i, y_main_init, color="blue", linewidth=1.1, linestyle=":", label="Main reward (init)")
                ax.plot(t_i, y_hedge_init, color="orange", linewidth=1.1, linestyle=":", label="Hedge reward (init)")
            ax.legend(loc="upper right", fontsize=7)
            if n_opt > 1:
                ax.set_xlim(t[0], t[-1])
            all_y = y_main_opt + y_hedge_opt
            if n_init > 0:
                all_y += y_main_init + y_hedge_init
            lo, hi = _finite_series_ylim(all_y, pad_min=1.0)
            ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Realized ΔW from state_after_step ---
    start, end = ranges.get("realized_dW", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("ΔW (tick×lot)")
        ax.set_title("Realized ΔW (step; includes inventory reval when mid moves)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        dW_times = getattr(runner, "dW_times", None) or []
        dW_hist = getattr(runner, "dW_hist", None) or []
        e_dw_times = getattr(runner, "E_dW_times", None) or []
        e_dw_hist = getattr(runner, "E_dW_hist", None) or []
        has_realized = bool(dW_times and dW_hist and len(dW_times) == len(dW_hist))
        has_expected = bool(e_dw_times and e_dw_hist and len(e_dw_times) == len(e_dw_hist))
        if has_realized or has_expected:
            all_y: list[float] = []
            t_min = None
            t_max = None
            if has_realized:
                all_y.extend(float(v) for v in dW_hist)
                t_min = dW_times[0] if t_min is None else min(t_min, dW_times[0])
                t_max = dW_times[-1] if t_max is None else max(t_max, dW_times[-1])
                ax.step(dW_times, dW_hist, where="mid", color="green", linewidth=1.5, label="Realized ΔW")
            if has_expected:
                all_y.extend(float(v) for v in e_dw_hist)
                t_min = e_dw_times[0] if t_min is None else min(t_min, e_dw_times[0])
                t_max = e_dw_times[-1] if t_max is None else max(t_max, e_dw_times[-1])
                ax.plot(
                    e_dw_times,
                    e_dw_hist,
                    color="darkorange",
                    linewidth=1.3,
                    linestyle="--",
                    label="Expected next-step ΔW (J_mean, h=0)",
                )
            ax.legend(loc="upper right")
            if t_min is not None and t_max is not None and t_min != t_max:
                ax.set_xlim(t_min, t_max)
            if all_y:
                lo, hi = _finite_series_ylim(all_y)
                ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Mid ---
    start, end = ranges.get("mid", (0, 0))
    if end > start and steps_data and dates:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Mid (ticks)")
        ax.set_title("Mid price")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        mid_hist = [float(s[1].mid) for s in steps_data]
        ax.plot(dates, mid_hist, color="navy", linewidth=1.5, label="Realized mid")
        # Expected mid (flow-implied, without residual) when available: one value per transition, align to dates[1:]
        if runner is not None:
            expected_mid = getattr(runner, "expected_mid_hist", None)
            if expected_mid:
                n_expected = min(len(expected_mid), max(0, len(dates) - 1))
                if n_expected > 0:
                    dates_expected = list(dates[1 : 1 + n_expected])
                    y_expected = list(expected_mid[:n_expected])
                    ax.plot(dates_expected, y_expected, color="darkorange", linewidth=1.2, linestyle="--", label="Expected mid (flow-implied)")
        ax.legend(loc="upper right")
        t0 = (steps_data[0][0] - bucket_ms) / 1000.0
        t1 = steps_data[-1][0] / 1000.0
        ax.set_xlim(datetime.fromtimestamp(t0, tz=timezone.utc), datetime.fromtimestamp(t1, tz=timezone.utc))
        if mid_hist:
            all_y = list(mid_hist)
            if runner and getattr(runner, "expected_mid_hist", None):
                all_y = all_y + list(runner.expected_mid_hist)
            lo, hi = _finite_series_ylim(all_y, pad_min=1.0)
            ax.set_ylim(lo, hi)
        else:
            ax.set_ylim(0, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Expected mid Δ decomposition (MC counterfactuals vs prev mid) ---
    start, end = ranges.get("expected_mid_decomp", (0, 0))
    if end > start and steps_data and dates and runner is not None:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Time")
        ax.set_ylabel("Δ mid (ticks)")
        ax.set_title(
            "Expected Δmid vs prev (MC): MOF-only · LOF-only · full flow · residual mean "
            "(MOF/LOF not additive; interaction in full flow)"
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        dm = getattr(runner, "expected_mid_delta_mof_hist", None) or []
        dl = getattr(runner, "expected_mid_delta_lof_hist", None) or []
        df = getattr(runner, "expected_mid_delta_full_flow_hist", None) or []
        dr = getattr(runner, "expected_mid_delta_residual_hist", None) or []
        n = min(len(dm), len(dl), len(df), len(dr), max(0, len(dates) - 1))
        if n > 0:
            td = list(dates[1 : 1 + n])
            ax.plot(td, dm[:n], color="teal", linewidth=1.3, label="MOF-only (sample MOF, mean LOF)")
            ax.plot(td, dl[:n], color="purple", linewidth=1.3, label="LOF-only (E[MOF], sample LOF)")
            ax.plot(td, df[:n], color="darkorange", linewidth=1.0, linestyle="--", label="Full flow (both sampled)")
            ax.plot(td, dr[:n], color="brown", linewidth=1.0, linestyle=":", label="Residual mean")
            ax.legend(loc="upper right", fontsize=7)
            t0 = (steps_data[0][0] - bucket_ms) / 1000.0
            t1 = steps_data[-1][0] / 1000.0
            ax.set_xlim(datetime.fromtimestamp(t0, tz=timezone.utc), datetime.fromtimestamp(t1, tz=timezone.utc))
            all_y = dm[:n] + dl[:n] + df[:n] + dr[:n]
            lo, hi = _finite_series_ylim(all_y, pad_min=1.0, pad_frac=0.12)
            ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
        if fig:
            fig.autofmt_xdate()

    # --- Depth (exchange-style: mid at center, bid left/negative spread, ask right/positive) ---
    # Plot raw depth from current OB + already-fitted curves from last MPC step (no redundant fitting).
    # Use shared spread range so depth and action_depth subplots stay aligned (zero = mid, same x range).
    # If spread_max is provided, use it for x-axis limits and filter orderbook data.
    _depth_x_max = 1.0  # default if no OB data; overwritten below
    start, end = ranges.get("depth", (0, 0))
    if end > start and steps_data:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax.set_ylabel("Cumulative volume")
        ax.set_title("Order book depth (bid / ask) and fitted curves")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        ob = steps_data[-1][1]
        ob_rel = ob_to_ob_rel(ob, spread_max=spread_max)
        (bid_spreads, bid_cumvol), (ask_spreads, ask_cumvol) = cumulative_depth_curves(ob_rel)
        max_bid = float(np.max(bid_spreads)) if bid_spreads.size > 0 else 0.0
        max_ask = float(np.max(ask_spreads)) if ask_spreads.size > 0 else 0.0
        # Use spread_max if provided, otherwise use actual max from data
        if spread_max is not None:
            _depth_x_max = float(spread_max)
        else:
            _depth_x_max = max(max_bid, max_ask, 1.0)
        # Raw depth: bid on left (negative x, mirrored), ask on right (positive x)
        if bid_spreads.size > 0:
            x_bid = np.concatenate([[0], -bid_spreads])
            y_bid = np.concatenate([[0], bid_cumvol])
            ax.step(x_bid, y_bid, where="post", color="green", linewidth=1.5, label="Bid depth")
        if ask_spreads.size > 0:
            x_ask = np.concatenate([[0], ask_spreads])
            y_ask = np.concatenate([[0], ask_cumvol])
            ax.step(x_ask, y_ask, where="post", color="red", linewidth=1.5, label="Ask depth")
        # Fitted curves: use last MPC result only (no fitting here)
        book_params = runner.last_book_params if runner is not None else None
        n_pts = 200
        if book_params and _depth_x_max > 0 and "bid" in book_params:
            x_bid_fit = np.linspace(-_depth_x_max, 0, n_pts)
            fit_bid = np.array([depth_from_sigmoid_params(book_params["bid"], -x) for x in x_bid_fit])
            ax.plot(x_bid_fit, fit_bid, color="green", linestyle="--", linewidth=1.2, label="Bid fit")
        if book_params and _depth_x_max > 0 and "ask" in book_params:
            x_ask_fit = np.linspace(0, _depth_x_max, n_pts)
            fit_ask = np.array([depth_from_sigmoid_params(book_params["ask"], x) for x in x_ask_fit])
            ax.plot(x_ask_fit, fit_ask, color="red", linestyle="--", linewidth=1.2, label="Ask fit")
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=0)
        if bid_cumvol.size > 0 or ask_cumvol.size > 0:
            y_max = max(
                float(np.max(bid_cumvol)) if bid_cumvol.size > 0 else 0,
                float(np.max(ask_cumvol)) if ask_cumvol.size > 0 else 0,
            )
            ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1.0)
        else:
            ax.set_ylim(0, 1)
        pad = _depth_x_max * 0.05
        ax.set_xlim(-_depth_x_max - pad, _depth_x_max + pad)

    # --- Cumulative order flow from mid (observed; computed and plotted up to spread_max) ---
    start, end = ranges.get("lof_cumulative", (0, 0))
    spread_max_lof = float(spread_max) if spread_max is not None else DELTA_OB_SPREAD_MAX
    if end > start and steps_data and len(steps_data) >= 2:
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax.set_ylabel("Cumulative order change")
        ax.set_title("Cumulative order flow from mid (bid / ask)")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        prev_step = steps_data[-2]
        cur_step = steps_data[-1]
        prev_ob = prev_step[1]
        cur_ob = cur_step[1]
        buy_vol = float(cur_step[2])
        sell_vol = float(cur_step[3])
        x_centers, cum_bid, cum_ask = limit_order_flow_observed(
            prev_ob, cur_ob, buy_vol, sell_vol, spread_max=spread_max_lof
        )
        mask_bid = x_centers <= 0
        mask_ask = x_centers >= 0
        if np.any(mask_bid):
            x_bid = x_centers[mask_bid][::-1]
            y_bid = cum_bid[mask_bid][::-1]
            ax.step(x_bid, y_bid, where="post", color="green", linewidth=1.5, label="Bid")
        if np.any(mask_ask):
            x_ask = x_centers[mask_ask]
            y_ask = cum_ask[mask_ask]
            ax.step(x_ask, y_ask, where="post", color="red", linewidth=1.5, label="Ask")
        ax.legend(loc="upper right")
        ax.set_xlim(-spread_max_lof, spread_max_lof)
        if cum_bid.size > 0 or cum_ask.size > 0:
            all_cum = np.concatenate([cum_bid[mask_bid] if np.any(mask_bid) else [0], cum_ask[mask_ask] if np.any(mask_ask) else [0]])
            y_min, y_max = float(np.nanmin(all_cum)), float(np.nanmax(all_cum))
            pad_y = max((y_max - y_min) * 0.1, 1e-9) if y_max != y_min else 1.0
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
        else:
            ax.set_ylim(-1, 1)
    elif end > start:
        ax = axes[start]
        ax.clear()
        ax.set_title("Cumulative order flow from mid (need ≥2 steps)")
        ax.set_xlim(-spread_max_lof, spread_max_lof)
        ax.set_ylim(-1, 1)

    # --- LOF model sample (original space: cumulative path from model) ---
    start, end = ranges.get("lof_sample", (0, 0))
    if end > start and runner is not None:
        ax = axes[start]
        ax.clear()
        lof = getattr(runner, "limit_order_flow_model", None)
        if lof is not None and hasattr(lof, "sample_cumulative_path"):
            spread_max_ob = int(getattr(lof, "_spread_max", DELTA_OB_SPREAD_MAX))
            sample_bid = lof.sample_cumulative_path("bid", spread_max_ob)
            sample_ask = lof.sample_cumulative_path("ask", spread_max_ob)
            x_bid = np.arange(0, -spread_max_ob - 1, -1, dtype=np.float64)
            x_ask = np.arange(0, spread_max_ob + 1, dtype=np.float64)
            ax.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
            ax.set_ylabel("Cumulative (sample)")
            ax.set_title("LOF model sample (cumulative path)")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
            ax.step(x_bid, sample_bid, where="post", color="green", linewidth=1.5, label="Bid (sample)")
            ax.step(x_ask, sample_ask, where="post", color="red", linewidth=1.5, label="Ask (sample)")
            ax.legend(loc="upper right")
            ax.set_xlim(-spread_max_ob, spread_max_ob)
            y_vals = np.concatenate([sample_bid, sample_ask])
            y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=0.0, neginf=0.0)
            y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                y_min, y_max = -1.0, 1.0
            pad = max((y_max - y_min) * 0.1, 1e-9) if (np.isfinite(y_max - y_min) and y_max != y_min) else 1.0
            if not np.isfinite(pad):
                pad = 1.0
            ax.set_ylim(y_min - pad, y_max + pad)
        else:
            ax.set_title("LOF model sample (no estimator)")
            ax.set_xlim(-DELTA_OB_SPREAD_MAX, DELTA_OB_SPREAD_MAX)
            ax.set_ylim(-1, 1)

    # --- DiscretizedOrderAction: marginal and cumulative (output of controller.act) ---
    # Same x range and zero alignment as depth subplot (use _depth_x_max).
    start, end = ranges.get("action_depth", (0, 0))
    if end > start and steps_data and runner is not None:
        action_disc = getattr(runner, "last_discretized_action", None)
        pad = _depth_x_max * 0.05
        xlim = (-_depth_x_max - pad, _depth_x_max + pad)

        def _action_arrays(act):
            if act is None or (not act.bids and not act.asks):
                return (None, None, None, None, None, None)
            bid_levels = sorted(act.bids, key=lambda x: x[0])
            ask_levels = sorted(act.asks, key=lambda x: x[0])
            x_b = np.array([-float(s) for s, _ in bid_levels])
            q_b = np.array([float(q) for _, q in bid_levels])
            x_a = np.array([float(s) for s, _ in ask_levels])
            q_a = np.array([float(q) for _, q in ask_levels])
            cum_b = np.cumsum(q_b) if q_b.size else np.array([])
            cum_a = np.cumsum(q_a) if q_a.size else np.array([])
            return (x_b, q_b, x_a, q_a, cum_b, cum_a)

        arrs = _action_arrays(action_disc) if action_disc else (None, None, None, None, None, None)

        has_data = arrs[0] is not None and (arrs[1].size > 0 or arrs[3].size > 0)

        # Subplot 0: marginal (qty at each spread level)
        ax = axes[start]
        ax.clear()
        ax.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax.set_ylabel("Marginal volume")
        ax.set_title("DiscretizedOrderAction — marginal")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        if has_data:
            x_b, q_b, x_a, q_a, _, _ = arrs
            if x_b.size > 0:
                ax.step(np.concatenate([[x_b[0] - 0.5], x_b]), np.concatenate([[0], q_b]), where="post", color="green", linewidth=1.5, label="Bid")
            if x_a.size > 0:
                ax.step(np.concatenate([[x_a[0] - 0.5], x_a]), np.concatenate([[0], q_a]), where="post", color="red", linewidth=1.5, label="Ask")
            ax.legend(loc="upper right")
            y_max = max(float(np.max(q_b)) if q_b.size else 0.0, float(np.max(q_a)) if q_a.size else 0.0)
            ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1.0)
        else:
            ax.set_ylim(0, 1)
        ax.set_xlim(xlim[0], xlim[1])

        # Subplot 1: cumulative (sum of qty up to each spread)
        ax = axes[start + 1]
        ax.clear()
        ax.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax.set_ylabel("Cumulative volume")
        ax.set_title("DiscretizedOrderAction — cumulative")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        if has_data:
            x_b, q_b, x_a, q_a, cum_b, cum_a = arrs
            if x_b.size > 0 and cum_b.size > 0:
                ax.step(np.concatenate([[x_b[0] - 0.5], x_b]), np.concatenate([[0], cum_b]), where="post", color="green", linewidth=1.5, label="Bid")
            if x_a.size > 0 and cum_a.size > 0:
                ax.step(np.concatenate([[x_a[0] - 0.5], x_a]), np.concatenate([[0], cum_a]), where="post", color="red", linewidth=1.5, label="Ask")
            ax.legend(loc="upper right")
            y_max = max(float(np.max(cum_b)) if cum_b.size else 0.0, float(np.max(cum_a)) if cum_a.size else 0.0)
            ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1.0)
        else:
            ax.set_ylim(0, 1)
        ax.set_xlim(xlim[0], xlim[1])

    # # --- Toxicity (commented out) ---
    # start, end = ranges.get("toxicity", (0, 0))
    # if end > start:
    #     ax = axes[start]
    #     ax.clear()
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Toxicity")
    #     ax.set_title("Flow toxicity (weighted mean, γ=decay): Δmid × (buy_vol − sell_vol)")
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    #     ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    #     if steps_data and dates:
    #         toxicity = flow_toxicity_weighted(steps_data, decay=decay)
    #         ax.plot(dates, toxicity, color="purple", linewidth=1.0, label="Toxicity")
    #         ax.legend(loc="upper right")
    #         t0 = (steps_data[0][0] - bucket_ms) / 1000.0
    #         t1 = steps_data[-1][0] / 1000.0
    #         ax.set_xlim(datetime.fromtimestamp(t0, tz=timezone.utc), datetime.fromtimestamp(t1, tz=timezone.utc))
    #         if toxicity:
    #             y_lo, y_hi = min(toxicity), max(toxicity)
    #             pad = max((y_hi - y_lo) * 0.1, 1e-9) if y_hi != y_lo else 1.0
    #             ax.set_ylim(y_lo - pad, y_hi + pad)
    #     else:
    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #     if fig:
    #         fig.autofmt_xdate()




if __name__ == "__main__":
    from mm_runtime.pipeline import main
    main()

