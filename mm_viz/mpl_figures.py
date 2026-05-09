"""
Matplotlib figures that mirror ``order_flow_animation.draw_dynamic`` / ``fig_action``.

``st.line_chart`` plots each series against row index, so spread-space step plots (depth,
ladder) looked wrong. These functions use the same axes, step/post, colors, and limits
as the animation code path.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite_series_ylim(
    series,
    *,
    pad_min: float = 0.5,
    pad_frac: float = 0.1,
    default: tuple[float, float] = (-1.0, 1.0),
) -> tuple[float, float]:
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


def figure_action_depth_window(frame: dict[str, Any]) -> plt.Figure:
    """
    Match ``fig_action`` in ``order_flow_animation.run_animation``:
    5 rows — J components, main/hedge rewards, depth, action marginal, action cumulative.
    X-axis for rows 0–1 is decision index (same ordering as the runner histories).
    """
    rs = (frame.get("runner_series") or {}) if isinstance(frame, dict) else {}
    jc = (rs.get("J_components") or {}) if isinstance(rs, dict) else {}
    rw = (rs.get("rewards") or {}) if isinstance(rs, dict) else {}

    fig, axes = plt.subplots(5, 1, figsize=(10, 13.2), sharex=False)
    ax_j, ax_rw, ax_depth, ax_marg, ax_cum = axes

    # --- Row 0: Optimizer J components (same labels/colors as animation) ---
    ax_j.set_xlabel("Decision index")
    ax_j.set_ylabel("Objective components")
    ax_j.set_title("Optimizer J components")
    ax_j.axhline(0, color="gray", linewidth=0.8, linestyle="-")

    y_mean = list(jc.get("mean") or [])
    y_inv = list(jc.get("inv") or [])
    y_std = list(jc.get("std") or [])
    y_uni = list(jc.get("uniformity") or [])
    y_spr = list(jc.get("spread") or [])
    y_mi = list(jc.get("mean_init") or [])
    y_ii = list(jc.get("inv_init") or [])
    y_si = list(jc.get("std_init") or [])
    y_ui = list(jc.get("uniformity_init") or [])
    y_spi = list(jc.get("spread_init") or [])

    n_comp = min(len(y_mean), len(y_inv), len(y_std), len(y_uni), len(y_spr))
    if n_comp > 0:
        t = np.arange(n_comp, dtype=np.float64)
        ax_j.plot(t, y_mean[:n_comp], color="blue", linewidth=1.2, label="reward_coef * J_mean (opt)")
        ax_j.plot(t, y_inv[:n_comp], color="red", linewidth=1.2, label="-inv_coef * J_inv_penalty (opt)")
        ax_j.plot(t, y_std[:n_comp], color="purple", linewidth=1.2, label="-risk_coef * J_std (opt)")
        ax_j.plot(t, y_uni[:n_comp], color="green", linewidth=1.2, label="-uniformity_coef * J_uniformity (opt)")
        ax_j.plot(t, y_spr[:n_comp], color="black", linewidth=1.2, label="-spread_coef * J_spread (opt)")
        n_init = min(n_comp, len(y_mi), len(y_ii), len(y_si), len(y_ui), len(y_spi))
        if n_init > 0:
            ti = np.arange(n_init, dtype=np.float64)
            ax_j.plot(ti, y_mi[:n_init], color="blue", linewidth=1.1, linestyle=":", label="reward_coef * J_mean (init)")
            ax_j.plot(ti, y_ii[:n_init], color="red", linewidth=1.1, linestyle=":", label="-inv_coef * J_inv_penalty (init)")
            ax_j.plot(ti, y_si[:n_init], color="purple", linewidth=1.1, linestyle=":", label="-risk_coef * J_std (init)")
            ax_j.plot(ti, y_ui[:n_init], color="green", linewidth=1.1, linestyle=":", label="-uniformity_coef * J_uniformity (init)")
            ax_j.plot(ti, y_spi[:n_init], color="black", linewidth=1.1, linestyle=":", label="-spread_coef * J_spread (init)")
        ax_j.legend(loc="upper right", fontsize=7)
        if n_comp > 1:
            ax_j.set_xlim(t[0], t[-1])
        all_y = y_mean[:n_comp] + y_inv[:n_comp] + y_std[:n_comp] + y_uni[:n_comp] + y_spr[:n_comp]
        if n_init > 0:
            all_y = all_y + y_mi[:n_init] + y_ii[:n_init] + y_si[:n_init] + y_ui[:n_init] + y_spi[:n_init]
        lo, hi = _finite_series_ylim(all_y, pad_min=1.0)
        ax_j.set_ylim(lo, hi)
    else:
        ax_j.set_xlim(0, 1)
        ax_j.set_ylim(-1, 1)

    # --- Row 1: Main / hedge rewards ---
    ax_rw.set_xlabel("Decision index")
    ax_rw.set_ylabel("Reward (tick×lot)")
    ax_rw.set_title("Main and hedge rewards (initial vs optimal)")
    ax_rw.axhline(0, color="gray", linewidth=0.8, linestyle="-")

    main_opt = list(rw.get("main_opt") or [])
    hedge_opt = list(rw.get("hedge_opt") or [])
    main_init = list(rw.get("main_init") or [])
    hedge_init = list(rw.get("hedge_init") or [])
    n_opt = min(len(main_opt), len(hedge_opt))
    if n_opt > 0:
        t = np.arange(n_opt, dtype=np.float64)
        ax_rw.plot(t, main_opt[:n_opt], color="blue", linewidth=1.2, label="Main reward (opt)")
        ax_rw.plot(t, hedge_opt[:n_opt], color="orange", linewidth=1.2, label="Hedge reward (opt)")
        n_init = min(len(main_init), len(hedge_init))
        if n_init > 0:
            ti = np.arange(n_init, dtype=np.float64)
            ax_rw.plot(ti, main_init[:n_init], color="blue", linewidth=1.1, linestyle=":", label="Main reward (init)")
            ax_rw.plot(ti, hedge_init[:n_init], color="orange", linewidth=1.1, linestyle=":", label="Hedge reward (init)")
        ax_rw.legend(loc="upper right", fontsize=7)
        if n_opt > 1:
            ax_rw.set_xlim(t[0], t[-1])
        all_y = main_opt[:n_opt] + hedge_opt[:n_opt]
        if n_init > 0:
            all_y = all_y + main_init[:n_init] + hedge_init[:n_init]
        lo, hi = _finite_series_ylim(all_y, pad_min=1.0)
        ax_rw.set_ylim(lo, hi)
    else:
        ax_rw.set_xlim(0, 1)
        ax_rw.set_ylim(-1, 1)

    # --- Depth + fit (spread space), shared x_max with action rows ---
    depth = frame.get("depth") or {}
    x_max = float(depth.get("x_max", 1.0))
    pad = x_max * 0.05
    xlim = (-x_max - pad, x_max + pad)

    ax_depth.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
    ax_depth.set_ylabel("Cumulative volume")
    ax_depth.set_title("Order book depth (bid / ask) and fitted curves")
    ax_depth.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)

    b = depth.get("bid") or {}
    a = depth.get("ask") or {}
    xb = np.asarray(b.get("x") or [0.0], dtype=np.float64)
    yb = np.asarray(b.get("y") or [0.0], dtype=np.float64)
    xa = np.asarray(a.get("x") or [0.0], dtype=np.float64)
    ya = np.asarray(a.get("y") or [0.0], dtype=np.float64)
    # Match animation: only draw a side if there is at least one non-mid level (len>1 step path).
    if xb.size >= 2:
        ax_depth.step(xb, yb, where="post", color="green", linewidth=1.5, label="Bid depth")
    if xa.size >= 2:
        ax_depth.step(xa, ya, where="post", color="red", linewidth=1.5, label="Ask depth")

    fit = depth.get("fit") or {}
    if fit.get("present"):
        if isinstance(fit.get("bid"), dict):
            fxb = np.asarray(fit["bid"].get("x") or [], dtype=np.float64)
            fyb = np.asarray(fit["bid"].get("y") or [], dtype=np.float64)
            if fxb.size and fyb.size:
                ax_depth.plot(fxb, fyb, color="green", linestyle="--", linewidth=1.2, label="Bid fit")
        if isinstance(fit.get("ask"), dict):
            fxa = np.asarray(fit["ask"].get("x") or [], dtype=np.float64)
            fya = np.asarray(fit["ask"].get("y") or [], dtype=np.float64)
            if fxa.size and fya.size:
                ax_depth.plot(fxa, fya, color="red", linestyle="--", linewidth=1.2, label="Ask fit")

    ax_depth.legend(loc="upper right")
    ax_depth.set_ylim(bottom=0)
    y_max_d = 0.0
    if yb.size:
        y_max_d = max(y_max_d, float(np.max(yb)))
    if ya.size:
        y_max_d = max(y_max_d, float(np.max(ya)))
    ax_depth.set_ylim(0, y_max_d * 1.1 if y_max_d > 0 else 1.0)
    ax_depth.set_xlim(xlim[0], xlim[1])

    # --- Action marginal / cumulative (same as draw_dynamic action_depth) ---
    ad = frame.get("action_depth") or {}
    ax_marg.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
    ax_marg.set_ylabel("Marginal volume")
    ax_marg.set_title("DiscretizedOrderAction — marginal")
    ax_marg.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)

    ax_cum.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
    ax_cum.set_ylabel("Cumulative volume")
    ax_cum.set_title("DiscretizedOrderAction — cumulative")
    ax_cum.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)

    if ad.get("present"):
        x_b = np.asarray(ad.get("x_bid") or [], dtype=np.float64)
        q_b = np.asarray(ad.get("q_bid") or [], dtype=np.float64)
        x_a = np.asarray(ad.get("x_ask") or [], dtype=np.float64)
        q_a = np.asarray(ad.get("q_ask") or [], dtype=np.float64)
        cum_b = np.asarray(ad.get("cum_bid") or [], dtype=np.float64)
        cum_a = np.asarray(ad.get("cum_ask") or [], dtype=np.float64)

        has_data = x_b.size > 0 or q_a.size > 0

        if has_data:
            if x_b.size > 0:
                ax_marg.step(
                    np.concatenate([[x_b[0] - 0.5], x_b]),
                    np.concatenate([[0], q_b]),
                    where="post",
                    color="green",
                    linewidth=1.5,
                    label="Bid",
                )
            if x_a.size > 0:
                ax_marg.step(
                    np.concatenate([[x_a[0] - 0.5], x_a]),
                    np.concatenate([[0], q_a]),
                    where="post",
                    color="red",
                    linewidth=1.5,
                    label="Ask",
                )
            ax_marg.legend(loc="upper right")
            y_max_m = max(float(np.max(q_b)) if q_b.size else 0.0, float(np.max(q_a)) if q_a.size else 0.0)
            ax_marg.set_ylim(0, y_max_m * 1.1 if y_max_m > 0 else 1.0)
        else:
            ax_marg.set_ylim(0, 1)
        ax_marg.set_xlim(xlim[0], xlim[1])

        if has_data:
            if x_b.size > 0 and cum_b.size > 0:
                ax_cum.step(
                    np.concatenate([[x_b[0] - 0.5], x_b]),
                    np.concatenate([[0], cum_b]),
                    where="post",
                    color="green",
                    linewidth=1.5,
                    label="Bid",
                )
            if x_a.size > 0 and cum_a.size > 0:
                ax_cum.step(
                    np.concatenate([[x_a[0] - 0.5], x_a]),
                    np.concatenate([[0], cum_a]),
                    where="post",
                    color="red",
                    linewidth=1.5,
                    label="Ask",
                )
            ax_cum.legend(loc="upper right")
            y_max_c = max(float(np.max(cum_b)) if cum_b.size else 0.0, float(np.max(cum_a)) if cum_a.size else 0.0)
            ax_cum.set_ylim(0, y_max_c * 1.1 if y_max_c > 0 else 1.0)
        else:
            ax_cum.set_ylim(0, 1)
        ax_cum.set_xlim(xlim[0], xlim[1])
    else:
        ax_marg.set_ylim(0, 1)
        ax_cum.set_ylim(0, 1)
        ax_marg.set_xlim(xlim[0], xlim[1])
        ax_cum.set_xlim(xlim[0], xlim[1])

    fig.suptitle("Action & Depth diagnostics", fontsize=12, y=1.01)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98), h_pad=1.2)
    return fig


def figure_lof_observed_and_sample(frame: dict[str, Any]) -> plt.Figure | None:
    """LOF cumulative (observed) + model sample, matching animation styling."""
    lof_obs = frame.get("lof_observed") or {}
    lof_s = frame.get("lof_sample") or {}
    if not lof_obs.get("present") and not lof_s.get("present"):
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.2), sharex=False)
    ax0, ax1 = axes

    if lof_obs.get("present"):
        spread_max_lof = float(lof_obs.get("spread_max") or 100.0)
        x_centers = np.asarray(lof_obs.get("x") or [], dtype=np.float64)
        cum_bid = np.asarray(lof_obs.get("cum_bid") or [], dtype=np.float64)
        cum_ask = np.asarray(lof_obs.get("cum_ask") or [], dtype=np.float64)
        ax0.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax0.set_ylabel("Cumulative order change")
        ax0.set_title("Cumulative order flow from mid (bid / ask)")
        ax0.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        ax0.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        mask_bid = x_centers <= 0
        mask_ask = x_centers >= 0
        if np.any(mask_bid):
            x_bid = x_centers[mask_bid][::-1]
            y_bid = cum_bid[mask_bid][::-1]
            ax0.step(x_bid, y_bid, where="post", color="green", linewidth=1.5, label="Bid")
        if np.any(mask_ask):
            x_ask = x_centers[mask_ask]
            y_ask = cum_ask[mask_ask]
            ax0.step(x_ask, y_ask, where="post", color="red", linewidth=1.5, label="Ask")
        ax0.legend(loc="upper right")
        ax0.set_xlim(-spread_max_lof, spread_max_lof)
        if cum_bid.size > 0 or cum_ask.size > 0:
            parts = []
            if np.any(mask_bid):
                parts.append(cum_bid[mask_bid])
            if np.any(mask_ask):
                parts.append(cum_ask[mask_ask])
            all_cum = np.concatenate(parts) if parts else np.array([0.0])
            y_min, y_max = float(np.nanmin(all_cum)), float(np.nanmax(all_cum))
            pad_y = max((y_max - y_min) * 0.1, 1e-9) if y_max != y_min else 1.0
            ax0.set_ylim(y_min - pad_y, y_max + pad_y)
        else:
            ax0.set_ylim(-1, 1)
    else:
        ax0.set_title("Cumulative order flow from mid (need ≥2 steps)")
        ax0.set_xlim(-100.0, 100.0)
        ax0.set_ylim(-1, 1)

    if lof_s.get("present"):
        spread_max_ob = int(lof_s.get("spread_max") or 100)
        sample_bid = np.asarray(lof_s.get("y_bid") or [], dtype=np.float64)
        sample_ask = np.asarray(lof_s.get("y_ask") or [], dtype=np.float64)
        x_bid = np.arange(0, -spread_max_ob - 1, -1, dtype=np.float64)
        x_ask = np.arange(0, spread_max_ob + 1, dtype=np.float64)
        ax1.set_xlabel("Spread from mid (ticks) — bid ← | mid | → ask")
        ax1.set_ylabel("Cumulative (sample)")
        ax1.set_title("LOF model sample (cumulative path)")
        ax1.axvline(0, color="gray", linewidth=0.8, linestyle="-", zorder=0)
        ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        if x_bid.size and sample_bid.size:
            ax1.step(x_bid, sample_bid, where="post", color="green", linewidth=1.5, label="Bid (sample)")
        if x_ask.size and sample_ask.size:
            ax1.step(x_ask, sample_ask, where="post", color="red", linewidth=1.5, label="Ask (sample)")
        ax1.legend(loc="upper right")
        ax1.set_xlim(-spread_max_ob, spread_max_ob)
        y_vals = np.concatenate([sample_bid, sample_ask]) if sample_bid.size or sample_ask.size else np.array([0.0])
        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=0.0, neginf=0.0)
        y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = -1.0, 1.0
        pad = max((y_max - y_min) * 0.1, 1e-9) if (np.isfinite(y_max - y_min) and y_max != y_min) else 1.0
        if not np.isfinite(pad):
            pad = 1.0
        ax1.set_ylim(y_min - pad, y_max + pad)
    else:
        ax1.set_title("LOF model sample (no estimator)")
        ax1.set_xlim(-100.0, 100.0)
        ax1.set_ylim(-1, 1)

    fig.suptitle("Limit order flow diagnostics", fontsize=12, y=1.01)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96), h_pad=1.2)
    return fig
