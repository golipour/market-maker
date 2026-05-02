from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from market_making_controller.models.flow_implied import Step
from market_making_controller.models.execution import cumulative_depth_curves, depth_from_sigmoid_params
from market_making_controller.state import DiscretizedOrderAction, ob_to_ob_rel

from market_making_controller.models.flow_implied import limit_order_flow_observed


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _np_list(a: Any) -> list[float]:
    if a is None:
        return []
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(v) for v in arr.reshape(-1)]


def _finite(v: float) -> bool:
    return bool(np.isfinite(v))


def _sample_model_volumes(model: Any, n_samples: int) -> tuple[list[float], list[float]]:
    buy: list[float] = []
    sell: list[float] = []
    if model is None or not hasattr(model, "sample"):
        return buy, sell
    for _ in range(max(1, int(n_samples))):
        try:
            bv, sv = model.sample(())
            b = max(0.0, _to_float(bv))
            s = max(0.0, _to_float(sv))
            if _finite(b) and _finite(s):
                buy.append(float(b))
                sell.append(float(s))
        except Exception:
            continue
    return buy, sell


def _log_loss_subplots_data(
    tester: Any,
    price_tester: Any,
    *,
    limit_order_flow_model: Any,
    n_dates: int,
    runner: Any,
    market_flow_model: Any,
) -> list[dict[str, Any]]:
    """
    Mirror of order_flow_animation._log_loss_subplots_data, but returns pure data.
    """
    subplots: list[dict[str, Any]] = []
    if tester is not None and hasattr(tester, "get_plot_data"):
        try:
            tpd = tester.get_plot_data() or {}
            # Some testers expose "subplots", some only "series".
            if isinstance(tpd.get("subplots"), list):
                subplots.extend([dict(x) for x in (tpd.get("subplots") or [])])
            elif isinstance(tpd.get("series"), list):
                subplots.append(
                    {
                        "title": str(tpd.get("title", "Log loss (NLL)")),
                        "ylabel": str(tpd.get("ylabel", "EWMA −log p")),
                        "series": list(tpd.get("series") or []),
                        "ylim": tpd.get("ylim"),
                    }
                )
        except Exception:
            pass

    # Price tester plot data is typically one subplot for Δmid NLL / residual.
    if price_tester is not None and hasattr(price_tester, "get_plot_data"):
        try:
            ppd = price_tester.get_plot_data() or {}
            if isinstance(ppd.get("subplots"), list):
                subplots.extend([dict(x) for x in (ppd.get("subplots") or [])])
            elif isinstance(ppd.get("series"), list):
                subplots.append(
                    {
                        "title": str(ppd.get("title", "Price log loss")),
                        "ylabel": str(ppd.get("ylabel", "EWMA −log p")),
                        "series": list(ppd.get("series") or []),
                        "ylim": ppd.get("ylim"),
                    }
                )
        except Exception:
            pass

    # LOF might have its own tester-like plot data; if present, include it (no assumptions).
    lof = limit_order_flow_model
    if lof is not None and hasattr(lof, "get_plot_data"):
        try:
            lpd = lof.get_plot_data() or {}
            if isinstance(lpd.get("subplots"), list):
                for sp in lpd.get("subplots") or []:
                    if isinstance(sp, dict) and sp.get("title") and sp.get("series") is not None:
                        subplots.append(dict(sp))
        except Exception:
            pass

    # Ensure series lengths align with dates in the UI (same padding behavior as animation).
    for sp in subplots:
        series = list(sp.get("series") or [])
        padded = []
        for ser in series:
            y = list(ser.get("y") or [])
            if n_dates > 0 and len(y) < n_dates:
                y = [0.0] * (n_dates - len(y)) + y
            elif n_dates > 0 and len(y) > n_dates:
                y = y[-n_dates:]
            padded.append({**ser, "y": [float(v) for v in y]})
        sp["series"] = padded

    if not subplots:
        subplots = [{"title": "Log loss (NLL)", "ylabel": "EWMA −log p", "series": []}]
    return subplots


def _action_arrays(action_disc: Any) -> dict[str, Any]:
    act = action_disc if isinstance(action_disc, DiscretizedOrderAction) else None
    if act is None or (not act.bids and not act.asks):
        return {"present": False}
    bid_levels = sorted(act.bids, key=lambda x: x[0])
    ask_levels = sorted(act.asks, key=lambda x: x[0])
    x_b = np.array([-float(s) for s, _ in bid_levels], dtype=np.float64)
    q_b = np.array([float(q) for _, q in bid_levels], dtype=np.float64)
    x_a = np.array([float(s) for s, _ in ask_levels], dtype=np.float64)
    q_a = np.array([float(q) for _, q in ask_levels], dtype=np.float64)
    return {
        "present": True,
        "x_bid": _np_list(x_b),
        "q_bid": _np_list(q_b),
        "x_ask": _np_list(x_a),
        "q_ask": _np_list(q_a),
        "cum_bid": _np_list(np.cumsum(q_b)) if q_b.size else [],
        "cum_ask": _np_list(np.cumsum(q_a)) if q_a.size else [],
    }


def _depth_arrays(ob: Any, *, spread_max: float | None, book_params: dict[str, Any] | None) -> dict[str, Any]:
    # Raw depth from OB at current step in "spread space" (mid at 0).
    try:
        ob_rel = ob_to_ob_rel(ob, spread_max=spread_max)
        (bid_spreads, bid_cumvol), (ask_spreads, ask_cumvol) = cumulative_depth_curves(ob_rel)
    except Exception:
        bid_spreads = np.asarray([], dtype=np.float64)
        bid_cumvol = np.asarray([], dtype=np.float64)
        ask_spreads = np.asarray([], dtype=np.float64)
        ask_cumvol = np.asarray([], dtype=np.float64)

    x_bid = np.concatenate([[0.0], -np.asarray(bid_spreads, dtype=np.float64)]) if bid_spreads.size else np.asarray([0.0])
    y_bid = np.concatenate([[0.0], np.asarray(bid_cumvol, dtype=np.float64)]) if bid_cumvol.size else np.asarray([0.0])
    x_ask = np.concatenate([[0.0], np.asarray(ask_spreads, dtype=np.float64)]) if ask_spreads.size else np.asarray([0.0])
    y_ask = np.concatenate([[0.0], np.asarray(ask_cumvol, dtype=np.float64)]) if ask_cumvol.size else np.asarray([0.0])

    max_bid = float(np.max(bid_spreads)) if bid_spreads.size > 0 else 0.0
    max_ask = float(np.max(ask_spreads)) if ask_spreads.size > 0 else 0.0
    x_max = float(spread_max) if spread_max is not None else max(max_bid, max_ask, 1.0)
    fit: dict[str, Any] = {"present": False}
    try:
        if book_params and x_max > 0 and ("bid" in book_params or "ask" in book_params):
            n_pts = 200
            out: dict[str, Any] = {"present": True}
            if "bid" in book_params:
                x_bid_fit = np.linspace(-x_max, 0.0, n_pts, dtype=np.float64)
                y_bid_fit = np.asarray([depth_from_sigmoid_params(book_params["bid"], -float(x)) for x in x_bid_fit], dtype=np.float64)
                out["bid"] = {"x": _np_list(x_bid_fit), "y": _np_list(y_bid_fit)}
            if "ask" in book_params:
                x_ask_fit = np.linspace(0.0, x_max, n_pts, dtype=np.float64)
                y_ask_fit = np.asarray([depth_from_sigmoid_params(book_params["ask"], float(x)) for x in x_ask_fit], dtype=np.float64)
                out["ask"] = {"x": _np_list(x_ask_fit), "y": _np_list(y_ask_fit)}
            fit = out
    except Exception:
        fit = {"present": False}

    return {
        "x_max": x_max,
        "bid": {"x": _np_list(x_bid), "y": _np_list(y_bid)},
        "ask": {"x": _np_list(x_ask), "y": _np_list(y_ask)},
        "fit": fit,
    }


def _qq_data(runner: Any) -> dict[str, Any]:
    steps = list(getattr(runner, "steps", []) or [])
    qq_obs_window = int(getattr(runner, "QQ_OBS_WINDOW", 120) or 120)
    if qq_obs_window > 0:
        steps = steps[-qq_obs_window:]
    obs_buy = [max(0.0, float(s[2])) for s in steps]
    obs_sell = [max(0.0, float(s[3])) for s in steps]
    n_obs = min(len(obs_buy), len(obs_sell))
    if n_obs < 8:
        return {"present": False, "reason": "waiting_for_data", "n_obs": n_obs}
    n_sim = min(1500, max(300, n_obs))
    sim_buy, sim_sell = _sample_model_volumes(getattr(runner, "mof_model", None), n_sim)
    if len(sim_buy) < 8 or len(sim_sell) < 8:
        return {"present": False, "reason": "model_samples_unavailable", "n_obs": n_obs, "n_sim": len(sim_buy)}
    q = np.linspace(0.02, 0.98, 49)
    obs_buy_q = np.quantile(np.asarray(obs_buy, dtype=np.float64), q)
    obs_sell_q = np.quantile(np.asarray(obs_sell, dtype=np.float64), q)
    sim_buy_q = np.quantile(np.asarray(sim_buy, dtype=np.float64), q)
    sim_sell_q = np.quantile(np.asarray(sim_sell, dtype=np.float64), q)
    return {
        "present": True,
        "n_obs": n_obs,
        "n_sim_buy": len(sim_buy),
        "n_sim_sell": len(sim_sell),
        "buy": {"x": _np_list(sim_buy_q), "y": _np_list(obs_buy_q)},
        "sell": {"x": _np_list(sim_sell_q), "y": _np_list(obs_sell_q)},
    }


def build_diagnostics_frame(
    *,
    runner: Any,
    exchange: str,
    symbol: str,
    spread_max: float | None,
) -> dict[str, Any]:
    """
    Build a single JSON-serializable frame that contains ALL datasets needed to recreate:
    - Figure 1 (main window) subplots
    - Figure 2 (loss/NLL)
    - Figure 3 (action/depth diagnostics)
    - Figure 4 (price diagnostics)
    - Figure 5 (volume QQ)
    - Figure 6 (anchor vs main reps + anchor params/portfolio)

    The realtime UI (mm_viz) consumes this frame stream and renders the equivalent subplots.
    """
    steps_data: list[Step] = list(getattr(runner, "steps", []) or [])
    last_step = steps_data[-1] if steps_data else None
    prev_step = steps_data[-2] if len(steps_data) >= 2 else None

    # Snapshot-friendly "latest" point values.
    latest = {
        "t_end_ms": _to_int(getattr(last_step, "__getitem__", lambda i: None)(0)) if last_step else None,
        "buy_vol": _to_float(last_step[2]) if last_step else None,
        "sell_vol": _to_float(last_step[3]) if last_step else None,
        "mid_ticks": _to_float(getattr(last_step[1], "mid", float("nan"))) if last_step else None,
    }

    # Main plot data from models/testers (already JSON-ish).
    mof_model = getattr(runner, "mof_model", None)
    param_plot_data = {}
    if mof_model is not None and hasattr(mof_model, "get_plot_data"):
        try:
            param_plot_data = mof_model.get_plot_data() or {}
        except Exception:
            param_plot_data = {}

    tester = getattr(runner, "tester", None)
    score_plot_data = {}
    if tester is not None and hasattr(tester, "get_score_plot_data"):
        try:
            score_plot_data = tester.get_score_plot_data() or {}
        except Exception:
            score_plot_data = {}

    # Depth + action arrays for the most recent step/action.
    depth = (
        _depth_arrays(
            last_step[1],
            spread_max=spread_max,
            book_params=getattr(runner, "last_book_params", None),
        )
        if last_step
        else {"x_max": 1.0, "bid": {"x": [0.0], "y": [0.0]}, "ask": {"x": [0.0], "y": [0.0]}, "fit": {"present": False}}
    )
    action_depth = _action_arrays(getattr(runner, "last_discretized_action", None))

    # LOF observed cumulative from last transition (if we have ≥2 steps).
    lof_obs = {"present": False}
    if prev_step is not None and last_step is not None:
        try:
            spread_max_lof = float(spread_max) if spread_max is not None else 100.0
            x_centers, cum_bid, cum_ask = limit_order_flow_observed(
                prev_step[1],
                last_step[1],
                float(last_step[2]),
                float(last_step[3]),
                spread_max=spread_max_lof,
            )
            lof_obs = {
                "present": True,
                "spread_max": float(spread_max_lof),
                "x": _np_list(x_centers),
                "cum_bid": _np_list(cum_bid),
                "cum_ask": _np_list(cum_ask),
            }
        except Exception:
            lof_obs = {"present": False}

    lof_model = getattr(runner, "limit_order_flow_model", None)
    lof_plot_data = {}
    lof_params = {}
    lof_sample = {"present": False}
    if lof_model is not None:
        if hasattr(lof_model, "get_plot_data"):
            try:
                lof_plot_data = lof_model.get_plot_data() or {}
            except Exception:
                lof_plot_data = {}
        if hasattr(lof_model, "get_params"):
            try:
                lof_params = lof_model.get_params() or {}
            except Exception:
                lof_params = {}
        if hasattr(lof_model, "sample_cumulative_path"):
            try:
                spread_max_ob = int(getattr(lof_model, "_spread_max", 100) or 100)
                sb = lof_model.sample_cumulative_path("bid", spread_max_ob)
                sa = lof_model.sample_cumulative_path("ask", spread_max_ob)
                lof_sample = {
                    "present": True,
                    "spread_max": spread_max_ob,
                    "x_bid": _np_list(np.arange(0, -spread_max_ob - 1, -1, dtype=np.float64)),
                    "x_ask": _np_list(np.arange(0, spread_max_ob + 1, dtype=np.float64)),
                    "y_bid": _np_list(sb),
                    "y_ask": _np_list(sa),
                }
            except Exception:
                lof_sample = {"present": False}

    # Price diagnostics payload (uses runner.price_model histories + tester NLL).
    pm = getattr(runner, "price_model", None)
    price_diag: dict[str, Any] = {"present": bool(pm is not None)}
    if pm is not None:
        price_diag.update(
            {
                "rep_prices": {
                    "buy_realized": list(getattr(pm, "realized_buy_price_hist", []) or []),
                    "buy_computed": list(getattr(pm, "computed_buy_price_hist", []) or []),
                    "sell_realized": list(getattr(pm, "realized_sell_price_hist", []) or []),
                    "sell_computed": list(getattr(pm, "computed_sell_price_hist", []) or []),
                },
                "residual_sigma": {
                    "buy": list(getattr(pm, "sigma_buy_hist", []) or []),
                    "sell": list(getattr(pm, "sigma_sell_hist", []) or []),
                },
                "residual_mu": {
                    "buy": list(getattr(pm, "mu_buy_hist", []) or []),
                    "sell": list(getattr(pm, "mu_sell_hist", []) or []),
                },
                "pred_rep_errors": {
                    "buy": list(getattr(runner, "pred_buy_rep_error_hist", []) or []),
                    "sell": list(getattr(runner, "pred_sell_rep_error_hist", []) or []),
                    "buy_prev": list(getattr(runner, "pred_buy_rep_prev_error_hist", []) or []),
                    "sell_prev": list(getattr(runner, "pred_sell_rep_prev_error_hist", []) or []),
                },
            }
        )
        # NLL series is embedded in tester.get_plot_data(); UI can extract buy/sell NLL as animation does.
        try:
            price_diag["flow_nll_series"] = list((tester.get_plot_data() or {}).get("series", [])) if tester is not None and hasattr(tester, "get_plot_data") else []
        except Exception:
            price_diag["flow_nll_series"] = []

    # Anchor diagnostics payload (series already on runner).
    anchor_diag = {
        "main_ref": {
            "t": [getattr(t, "timestamp", lambda: None)() for t in (getattr(runner, "main_reference_times", []) or [])],  # not used directly
            "buy": list(getattr(runner, "main_buy_reference_hist", []) or []),
            "sell": list(getattr(runner, "main_sell_reference_hist", []) or []),
        },
        "anchor_ref": {
            "buy": list(getattr(runner, "anchor_buy_reference_hist", []) or []),
            "sell": list(getattr(runner, "anchor_sell_reference_hist", []) or []),
        },
        "anchor_params": {
            "mu_buy": list(getattr(runner, "anchor_mu_buy_hist", []) or []),
            "mu_sell": list(getattr(runner, "anchor_mu_sell_hist", []) or []),
            "sigma_buy": list(getattr(runner, "anchor_sigma_buy_hist", []) or []),
            "sigma_sell": list(getattr(runner, "anchor_sigma_sell_hist", []) or []),
        },
        "anchor_portfolio": {
            "inventory": list(getattr(runner, "anchor_inventory_hist", []) or []),
            "cash": list(getattr(runner, "anchor_cash_hist", []) or []),
        },
    }

    # Loss (Figure 2) data (subplots dicts with series).
    loss = _log_loss_subplots_data(
        tester=getattr(runner, "tester", None),
        price_tester=getattr(runner, "price_tester", None),
        limit_order_flow_model=getattr(runner, "limit_order_flow_model", None),
        n_dates=len(steps_data),
        runner=runner,
        market_flow_model=mof_model,
    )

    # QoL: include a compact "timeseries" point (used for streaming UI without recomputing).
    point = {
        "step_counter": _to_int(getattr(runner, "_step_counter", 0)),
        "inventory_lots": _to_float(getattr(runner, "inventory", float("nan"))),
        "cash_tick_lot": _to_float(getattr(runner, "cash", float("nan"))),
        "wealth_tick_lot": _to_float(getattr(runner, "wealth_hist", [float("nan")])[-1] if getattr(runner, "wealth_hist", None) else float("nan")),
        "wealth_quote": _to_float(getattr(runner, "wealth_quote_hist", [float("nan")])[-1] if getattr(runner, "wealth_quote_hist", None) else float("nan")),
        "mid_ticks": latest.get("mid_ticks"),
    }

    return {
        "schema": "DiagnosticsFrame",
        "exchange": str(exchange),
        "symbol": str(symbol),
        "latest": latest,
        "point": point,
        "plot_data": {
            "mof_params": param_plot_data,
            "score": score_plot_data,
            "lof_plot": lof_plot_data,
        },
        "depth": depth,
        "action_depth": action_depth,
        "lof_observed": lof_obs,
        "lof_params": lof_params,
        "lof_sample": lof_sample,
        "price_diag": price_diag,
        "volume_qq": _qq_data(runner),
        "anchor_diag": anchor_diag,
        "loss_subplots": loss,
        "book_params": getattr(runner, "last_book_params", None),
        "sigmoid_action": getattr(runner, "last_sigmoid_action", None),
        "runner_series": {
            # Core histories used across subplots (UI can reproduce exact Matplotlib series).
            "dW": list(getattr(runner, "dW_hist", []) or []),
            "delta_spread_pnl": list(getattr(runner, "delta_spread_pnl_hist", []) or []),
            "delta_inventory_pnl": list(getattr(runner, "delta_inventory_pnl_hist", []) or []),
            "delta_cross_pnl": list(getattr(runner, "delta_cross_pnl_hist", []) or []),
            "delta_fee_pnl": list(getattr(runner, "delta_fee_pnl_hist", []) or []),
            "expected_mid": list(getattr(runner, "expected_mid_hist", []) or []),
            "expected_mid_decomp": {
                "mof": list(getattr(runner, "expected_mid_delta_mof_hist", []) or []),
                "lof": list(getattr(runner, "expected_mid_delta_lof_hist", []) or []),
                "full": list(getattr(runner, "expected_mid_delta_full_flow_hist", []) or []),
                "residual": list(getattr(runner, "expected_mid_delta_residual_hist", []) or []),
            },
            "E_dW": list(getattr(runner, "E_dW_hist", []) or []),
            "J_components": {
                "mean": list(getattr(runner, "J_mean_hist", []) or []),
                "inv": list(getattr(runner, "J_inv_hist", []) or []),
                "std": list(getattr(runner, "J_std_hist", []) or []),
                "uniformity": list(getattr(runner, "J_uniformity_hist", []) or []),
                "spread": list(getattr(runner, "J_spread_hist", []) or []),
                "mean_init": list(getattr(runner, "J_mean_init_hist", []) or []),
                "inv_init": list(getattr(runner, "J_inv_init_hist", []) or []),
                "std_init": list(getattr(runner, "J_std_init_hist", []) or []),
                "uniformity_init": list(getattr(runner, "J_uniformity_init_hist", []) or []),
                "spread_init": list(getattr(runner, "J_spread_init_hist", []) or []),
            },
            "rewards": {
                "main_opt": list(getattr(runner, "main_reward_opt_hist", []) or []),
                "anchor_opt": list(getattr(runner, "anchor_reward_opt_hist", []) or []),
                "main_init": list(getattr(runner, "main_reward_init_hist", []) or []),
                "anchor_init": list(getattr(runner, "anchor_reward_init_hist", []) or []),
            },
            "wealth_tick_lot": list(getattr(runner, "wealth_hist", []) or []),
            "wealth_from_dw_tick_lot": list(getattr(runner, "wealth_from_dw_hist", []) or []),
            "wealth_quote": list(getattr(runner, "wealth_quote_hist", []) or []),
            "inventory_lots": list(getattr(runner, "inv_hist", []) or []),
            "cash_tick_lot": list(getattr(runner, "cash_hist", []) or []),
            "action_summary": {
                "delta_b": list(getattr(runner, "delta_b_hist", []) or []),
                "delta_a": list(getattr(runner, "delta_a_hist", []) or []),
                "q_b": list(getattr(runner, "q_b_hist", []) or []),
                "q_a": list(getattr(runner, "q_a_hist", []) or []),
            },
        },
    }

