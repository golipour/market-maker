from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from market_making_controller.models.flow_implied import Step
from market_making_controller.state import DiscretizedOrderAction, ZERO_DISCRETIZED_ACTION

from mm_core.app.parsed_step import parse_step
from mm_core.app.state_assembler import StateAssembler
from mm_core.app.observability_defaults import (
    resolve_diagnostics_emit,
    resolve_metric_labels,
    resolve_metrics_registry,
    resolve_structured_log,
)
from mm_core.domain.types import InstrumentId
from mm_core.units.converter import instrument_spec_from_trade_stats
from mm_core.app.model_updates import update_flow_models_after_step
from mm_core.app.step_economics_pg import keep_runner_pnl_series_memory, persist_step_economics_row
from mm_core.diagnostics.diagnostics_frame import build_diagnostics_frame
from mm_core.pricing import main_inventory_value_ticks, portfolio_wealth_quote, portfolio_wealth_ticks


def advance_order_flow_runner(runner: Any, t_end_ms: int | None = None) -> bool:
    """
    Extracted orchestration for one OrderFlowRunner step.

    This is an incremental refactor: it keeps behavior identical while moving the
    main loop logic under mm_core. The runner object is expected to be the current
    `order_flow_runner.OrderFlowRunner` instance (duck-typed).
    """
    step = runner.stepper.advance(t_end_ms=t_end_ms)
    if step is None:
        return False
    log_emit = resolve_structured_log(runner)
    METRICS = resolve_metrics_registry(runner)
    metric_labels = resolve_metric_labels(runner)
    emit_diagnostics = resolve_diagnostics_emit(runner)
    if not runner.anchor_hedge_enabled and len(step) >= 15:
        seq = list(step)
        step = Step(*(seq[:-2] + [0, 0]))
    runner._step_counter += 1
    s = parse_step(step)
    jp = os.environ.get("MM_STEP_JOURNAL_PATH", "").strip()
    if jp:
        try:
            from mm_core.app.step_journal import append_step_journal

            append_step_journal(jp, s)
        except Exception:
            pass

    # Migration aid: compute a typed snapshot in canonical float units.
    # This does not affect trading behavior; it is for observability/debugging and future refactors.
    try:
        inst = None
        spec = None
        if hasattr(runner, "_get_instrument_id_and_spec"):
            inst, spec = runner._get_instrument_id_and_spec()
        if inst is None or spec is None:
            ex = str(getattr(runner.stepper, "exchange", "") or "").strip()
            sym = str(getattr(runner.stepper, "symbol", "") or "").strip()
            if ex and sym:
                spec = instrument_spec_from_trade_stats(exchange=ex, symbol=sym)
                inst = InstrumentId(exchange=ex, symbol=sym)
        if inst is not None and spec is not None:
            runner._last_step_snapshot = StateAssembler.step_snapshot(s, instrument=inst, spec=spec)
    except Exception:
        runner._last_step_snapshot = None

    # Optional structured log (off by default; enable with MM_JSON_LOG=1).
    snap = getattr(runner, "_last_step_snapshot", None)
    if snap is not None and isinstance(getattr(snap, "data", None), dict):
        try:
            mi = getattr(snap, "main_instrument", None)
            if mi is not None:
                mid_q = float(mi.mid)
                buy_ref_q = float(mi.buy_ref)
                sell_ref_q = float(mi.sell_ref)
                inv_b = float(mi.inventory)
                cash_q = float(mi.cash)
            else:
                main = snap.data.get("main") or {}
                mid_q = float(main.get("mid", 0.0))
                buy_ref_q = float(main.get("buy_ref", 0.0))
                sell_ref_q = float(main.get("sell_ref", 0.0))
                inv_b = float(main.get("inventory", 0.0))
                cash_q = float(main.get("cash", 0.0))
            anchor = snap.data.get("anchor") or {}
            log_emit(
                "mm.step",
                {
                    "exchange": getattr(getattr(snap, "instrument", None), "exchange", None),
                    "symbol": getattr(getattr(snap, "instrument", None), "symbol", None),
                    "t_end_ms": int(s.t_end_ms),
                    "step_counter": int(getattr(runner, "_step_counter", 0)),
                    # Canonical floats (quote/base):
                    "mid_quote": mid_q,
                    "buy_ref_quote": buy_ref_q,
                    "sell_ref_quote": sell_ref_q,
                    "inventory_base": inv_b,
                    "cash_quote": cash_q,
                    "anchor_sell_ref_quote": anchor.get("sell_ref_quote"),
                    "anchor_inventory_base": anchor.get("inventory_base"),
                    "anchor_cash_quote": anchor.get("cash_quote"),
                },
            )
        except Exception:
            pass

        # Optional metrics (enable by setting MM_METRICS_PORT).
        try:
            labels = metric_labels(
                extra={
                    "exchange": getattr(getattr(snap, "instrument", None), "exchange", ""),
                    "symbol": getattr(getattr(snap, "instrument", None), "symbol", ""),
                }
            )
            mi = getattr(snap, "main_instrument", None)
            if mi is not None:
                mid_q = float(mi.mid)
                sell_ref_q = float(mi.sell_ref)
                inv_b = float(mi.inventory)
                cash_q = float(mi.cash)
            else:
                main = snap.data.get("main") or {}
                mid_q = float(main.get("mid", 0.0))
                sell_ref_q = float(main.get("sell_ref", 0.0))
                inv_b = float(main.get("inventory", 0.0))
                cash_q = float(main.get("cash", 0.0))
            METRICS.gauge("mm_step_counter", float(getattr(runner, "_step_counter", 0)), labels=labels, help="Step counter")
            METRICS.gauge("mm_mid_quote", mid_q, labels=labels, help="Mid price in quote/base")
            METRICS.gauge("mm_sell_ref_quote", sell_ref_q, labels=labels, help="Sell-side representative price in quote/base")
            METRICS.gauge("mm_inventory_base", inv_b, labels=labels, help="Inventory in base units")
            METRICS.gauge("mm_cash_quote", cash_q, labels=labels, help="Cash in quote units")
            if runner.wealth_quote_hist:
                METRICS.gauge("mm_wealth_quote", float(runner.wealth_quote_hist[-1]), labels=labels, help="Total wealth in quote units")
        except Exception:
            pass
    bucket_ms = runner.stepper.T_ms
    mid_now_ticks = float(s.ob.mid)
    anchor_inventory = int(s.anchor_inventory)
    anchor_cash = float(s.anchor_cash)
    if not runner.anchor_hedge_enabled:
        anchor_inventory = 0
        anchor_cash = 0.0
        anchor_buy_ref = None
        anchor_sell_ref = None
    else:
        anchor_buy_ref = s.anchor_buy_ref
        anchor_sell_ref = s.anchor_sell_ref

    t_anchor_port = datetime.fromtimestamp(s.t_end_ms / 1000.0, tz=timezone.utc)
    runner.anchor_portfolio_times.append(t_anchor_port)
    runner.anchor_inventory_hist.append(int(anchor_inventory))
    runner.anchor_cash_hist.append(float(anchor_cash))
    t_main = datetime.fromtimestamp(s.t_end_ms / 1000.0, tz=timezone.utc)
    runner.main_reference_times.append(t_main)
    runner.main_buy_reference_hist.append(float(s.main_buy_ref))
    runner.main_sell_reference_hist.append(float(s.main_sell_ref))

    if anchor_buy_ref is not None and anchor_sell_ref is not None:
        if runner.anchor_price_model is not None:
            runner.anchor_price_model.update(float(anchor_buy_ref), float(anchor_sell_ref))
            runner.last_anchor_price_model_params = runner.anchor_price_model.get_params()
        t_anchor = datetime.fromtimestamp(s.t_end_ms / 1000.0, tz=timezone.utc)
        runner.anchor_reference_times.append(t_anchor)
        runner.anchor_buy_reference_hist.append(float(anchor_buy_ref))
        runner.anchor_sell_reference_hist.append(float(anchor_sell_ref))
        p = runner.last_anchor_price_model_params or {}
        runner.anchor_price_param_times.append(t_anchor)
        runner.anchor_mu_buy_hist.append(float(p.get("mu_delta_buy", float("nan"))))
        runner.anchor_mu_sell_hist.append(float(p.get("mu_delta_sell", float("nan"))))
        runner.anchor_sigma_buy_hist.append(float(p.get("sigma_delta_buy", float("nan"))))
        runner.anchor_sigma_sell_hist.append(float(p.get("sigma_delta_sell", float("nan"))))

    # Prior controller state vs current step → realized PnL (inventory/cash from step).
    current_state = None
    _exec = runner.runner_execution_model or runner.execution_model
    dW_step: float | None = None
    if runner._pending_state is not None and _exec is not None:
        current_state, delta_pnl = _exec.state_after_step(
            runner._pending_state,
            step,
            spread_max=runner.spread_max,
        )
        runner.inventory = current_state.inventory
        runner.cash = current_state.cash
        t_fill = datetime.fromtimestamp(s.t_end_ms / 1000.0, tz=timezone.utc)
        dW_step = float(delta_pnl.wealth_pnl)
        try:
            ex_pg = str(getattr(runner.stepper, "exchange", "") or "").strip()
            sym_pg = str(getattr(runner.stepper, "symbol", "") or "").strip()
            persist_step_economics_row(
                t_fill=t_fill,
                t_end_ms=int(s.t_end_ms),
                step_counter=int(getattr(runner, "_step_counter", 0)),
                exchange=ex_pg,
                symbol=sym_pg,
                d_w=float(dW_step),
                spread_pnl=float(delta_pnl.spread_pnl),
                inventory_pnl=float(delta_pnl.inventory_pnl),
                cross_pnl=float(delta_pnl.cross_pnl),
                fee_pnl=float(delta_pnl.fee_pnl),
                wealth_pnl=float(delta_pnl.wealth_pnl),
            )
        except Exception:
            pass
        if keep_runner_pnl_series_memory():
            runner.dW_times.append(t_fill)
            runner.dW_hist.append(dW_step)
            runner.pnl_times.append(t_fill)
            runner.delta_spread_pnl_hist.append(float(delta_pnl.spread_pnl))
            runner.delta_inventory_pnl_hist.append(float(delta_pnl.inventory_pnl))
            runner.delta_cross_pnl_hist.append(float(delta_pnl.cross_pnl))
            runner.delta_fee_pnl_hist.append(float(delta_pnl.fee_pnl))

        # Wealth: single valuation policy (default sell_ref / mark-to-bid).
        main_sell_ref_now = float(getattr(current_state, "sell_ref", mid_now_ticks))
        anchor_inv = int(getattr(current_state, "anchor_inventory", 0))
        anchor_cash_now = float(getattr(current_state, "anchor_cash", 0.0))
        anchor_sell_ref_now = getattr(current_state, "anchor_sell_ref", None)
        vp = runner.valuation_policy
        wealth = portfolio_wealth_ticks(
            cash_tick_lot=float(runner.cash),
            inventory_lots=int(runner.inventory),
            main_sell_ref_ticks=main_sell_ref_now,
            main_mid_ticks=float(mid_now_ticks),
            anchor_inventory_lots=anchor_inv,
            anchor_cash_tick_lot=anchor_cash_now,
            anchor_sell_ref_ticks=float(anchor_sell_ref_now) if anchor_sell_ref_now is not None else None,
            policy=vp,
        )
        inv_value = main_inventory_value_ticks(
            inventory_lots=int(runner.inventory),
            main_sell_ref_ticks=main_sell_ref_now,
            main_mid_ticks=float(mid_now_ticks),
            policy=vp,
        )
        t_fill = datetime.fromtimestamp(s.t_end_ms / 1000.0, tz=timezone.utc)
        runner.inv_times.append(t_fill)
        runner.inv_hist.append(runner.inventory)
        runner.cash_hist.append(runner.cash)
        runner.inv_value_hist.append(inv_value)
        runner.wealth_times.append(t_fill)
        runner.wealth_hist.append(wealth)
        # Parallel wealth series in canonical quote units from StepSnapshot (if available).
        snap = getattr(runner, "_last_step_snapshot", None)
        if snap is not None and isinstance(getattr(snap, "data", None), dict):
            try:
                miw = getattr(snap, "main_instrument", None)
                if miw is not None:
                    cash_quote = float(miw.cash)
                    inv_base = float(miw.inventory)
                    sell_ref_quote = float(miw.sell_ref)
                else:
                    main = (snap.data.get("main") or {})
                    cash_quote = float(main.get("cash", 0.0))
                    inv_base = float(main.get("inventory", 0.0))
                    sell_ref_quote = float(main.get("sell_ref", 0.0))
                anchor = (snap.data.get("anchor") or {})
                a_cash_quote = float(anchor.get("cash_quote", 0.0) or 0.0)
                a_inv_base = float(anchor.get("inventory_base", 0.0) or 0.0)
                a_sell_ref_quote = anchor.get("sell_ref_quote")
                a_sell_ref_opt = float(a_sell_ref_quote) if a_sell_ref_quote is not None else None
                mid_q = float(miw.mid) if miw is not None else float((snap.data.get("main") or {}).get("mid", 0.0))
                wealth_quote = portfolio_wealth_quote(
                    cash_quote=cash_quote,
                    inventory_base=inv_base,
                    main_sell_ref_quote=sell_ref_quote,
                    main_mid_quote=mid_q,
                    anchor_cash_quote=a_cash_quote,
                    anchor_inventory_base=a_inv_base,
                    anchor_sell_ref_quote=a_sell_ref_opt,
                    policy=vp,
                )
                runner.wealth_quote_times.append(t_fill)
                runner.wealth_quote_hist.append(float(wealth_quote))
            except Exception:
                pass
        if dW_step is not None:
            if runner._wealth_from_dw_running is None:
                runner._wealth_from_dw_baseline = float(wealth - dW_step)
                runner._wealth_from_dw_running = float(runner._wealth_from_dw_baseline + dW_step)
            else:
                runner._wealth_from_dw_running = float(runner._wealth_from_dw_running + dW_step)
            runner.wealth_from_dw_times.append(t_fill)
            runner.wealth_from_dw_hist.append(float(runner._wealth_from_dw_running))

    # Score with previous params, then update estimators/models
    prev_step = runner.steps[-2] if len(runner.steps) >= 2 else None
    curr_step = step
    parsed_prev = parse_step(prev_step) if prev_step is not None else None
    parsed_curr = s

    rep_pred: tuple[float, float] | None = None
    if prev_step is not None and runner.price_model is not None:
        mid_exp = runner._estimate_expected_mid_mc(
            prev_step, runner.expected_mid_n_samples, curr_step=curr_step
        )
        if mid_exp is not None:
            runner.expected_mid_hist.append(float(mid_exp))
        rep_pred = runner._estimate_expected_representative_prices_mc(
            prev_step, runner.expected_mid_n_samples, curr_step
        )

    params = getattr(runner.mof_model, "get_params", lambda: {})()
    update_flow_models_after_step(runner, prev_step, curr_step, params)

    if runner.price_model is not None:
        if runner.price_tester is not None:
            price_params = runner.price_model.get_params()
            runner.price_tester.update(
                prev_step,
                curr_step,
                price_params,
                parsed_prev=parsed_prev,
                parsed_curr=parsed_curr,
            )
        runner.price_model.update(
            prev_step, curr_step, parsed_prev=parsed_prev, parsed_curr=parsed_curr
        )
        if (
            prev_step is not None
            and rep_pred is not None
            and runner.price_model.realized_buy_price_hist
            and runner.price_model.realized_sell_price_hist
        ):
            buy_mc, sell_mc = rep_pred
            rb = float(runner.price_model.realized_buy_price_hist[-1])
            rs = float(runner.price_model.realized_sell_price_hist[-1])
            runner.pred_buy_rep_error_hist.append(float(buy_mc - rb))
            runner.pred_sell_rep_error_hist.append(float(sell_mc - rs))
            if len(runner.price_model.realized_buy_price_hist) >= 2:
                prev_b = float(runner.price_model.realized_buy_price_hist[-2])
                prev_s = float(runner.price_model.realized_sell_price_hist[-2])
                runner.pred_buy_rep_prev_error_hist.append(float(prev_b - rb))
                runner.pred_sell_rep_prev_error_hist.append(float(prev_s - rs))
            else:
                runner.pred_buy_rep_prev_error_hist.append(float("nan"))
                runner.pred_sell_rep_prev_error_hist.append(float("nan"))

    # Decide new action
    if runner.controller is not None:
        if current_state is None:
            if runner.runner_execution_model is not None:
                current_state = StateAssembler.controller_state(
                    step,
                    spread_max=runner.spread_max,
                )
                runner.inventory = current_state.inventory
                runner.cash = current_state.cash

        opt = getattr(runner.controller, "optimizer", None)
        if runner.use_zero_action:
            action = ZERO_DISCRETIZED_ACTION
            runner.last_book_params = None
            runner.last_sigmoid_action = None
            runner.last_discretized_action = None
        else:
            action = runner.controller.act(current_state)
            runner.last_book_params = getattr(opt, "_last_book_params", None)
            runner.last_sigmoid_action = getattr(opt, "_last_sigmoid_action", None)

        if isinstance(action, DiscretizedOrderAction):
            runner.last_discretized_action = action
            if runner.live_order_execution:
                if runner._step_counter > runner.action_start_after_steps:
                    runner._execute_live_orders(action, current_state)
            E_dW = None if runner.use_zero_action else getattr(opt, "_last_E_dW_next", None)
            if E_dW is None and not runner.use_zero_action:
                E_dW = getattr(opt, "_last_E_dW", None)
            t_act = datetime.fromtimestamp((s.t_end_ms - bucket_ms / 2) / 1000.0, tz=timezone.utc)
            if E_dW is not None:
                runner.E_dW_times.append(t_act)
                runner.E_dW_hist.append(float(E_dW))
            comp = None if runner.use_zero_action else getattr(opt, "_last_J_components", None)
            if isinstance(comp, dict):
                runner.J_comp_times.append(t_act)
                runner.J_mean_hist.append(float(comp.get("term_reward", 0.0)))
                runner.J_inv_hist.append(float(comp.get("term_inv", 0.0)))
                runner.J_std_hist.append(float(comp.get("term_risk", 0.0)))
                runner.J_uniformity_hist.append(float(comp.get("term_uniformity", 0.0)))
                runner.J_spread_hist.append(float(comp.get("term_spread", 0.0)))
                runner.main_reward_opt_hist.append(float(getattr(opt, "_last_main_reward_mean", float("nan"))))
                runner.anchor_reward_opt_hist.append(float(getattr(opt, "_last_anchor_reward_mean", float("nan"))))
                comp_init = getattr(opt, "_last_J_components_initial", None)
                if isinstance(comp_init, dict):
                    runner.J_mean_init_hist.append(float(comp_init.get("term_reward", 0.0)))
                    runner.J_inv_init_hist.append(float(comp_init.get("term_inv", 0.0)))
                    runner.J_std_init_hist.append(float(comp_init.get("term_risk", 0.0)))
                    runner.J_uniformity_init_hist.append(float(comp_init.get("term_uniformity", 0.0)))
                    runner.J_spread_init_hist.append(float(comp_init.get("term_spread", 0.0)))
                    runner.main_reward_init_hist.append(float(getattr(opt, "_last_main_reward_mean_initial", float("nan"))))
                    runner.anchor_reward_init_hist.append(float(getattr(opt, "_last_anchor_reward_mean_initial", float("nan"))))
                else:
                    runner.J_mean_init_hist.append(float("nan"))
                    runner.J_inv_init_hist.append(float("nan"))
                    runner.J_std_init_hist.append(float("nan"))
                    runner.J_uniformity_init_hist.append(float("nan"))
                    runner.J_spread_init_hist.append(float("nan"))
                    runner.main_reward_init_hist.append(float("nan"))
                    runner.anchor_reward_init_hist.append(float("nan"))

        runner._pending_action = action
        runner._pending_state = current_state
        t_act = datetime.fromtimestamp((s.t_end_ms - bucket_ms / 2) / 1000.0, tz=timezone.utc)
        runner.action_times.append(t_act)
        if isinstance(action, DiscretizedOrderAction):
            delta_b = float(action.bids[0][0]) if action.bids else 0.0
            delta_a = float(action.asks[0][0]) if action.asks else 0.0
            q_b = sum(q for (_, q) in action.bids)
            q_a = sum(q for (_, q) in action.asks)
        else:
            delta_b = action.delta_b_ticks
            delta_a = action.delta_a_ticks
            q_b = action.q_b
            q_a = action.q_a
        runner.delta_b_hist.append(delta_b)
        runner.delta_a_hist.append(delta_a)
        runner.q_b_hist.append(q_b)
        runner.q_a_hist.append(q_a)

    # Emit a single diagnostics frame for mm_viz / Redis consumers.
    try:
        ex = str(getattr(runner.stepper, "exchange", "") or "").strip()
    except Exception:
        ex = ""
    try:
        sym = str(getattr(runner.stepper, "symbol", "") or "").strip()
    except Exception:
        sym = ""
    try:
        frame = build_diagnostics_frame(
            runner=runner,
            exchange=ex,
            symbol=sym,
            spread_max=float(getattr(runner, "spread_max", 0.0) or 0.0) or None,
        )
        emit_diagnostics(frame)
    except Exception:
        pass

    return True

