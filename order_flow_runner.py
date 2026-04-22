"""
Order-flow iteration: one place for the main simulation loop.
Advances stepper, applies fills via ExecutionModel (or ExecutionRule), updates estimators/testers, computes actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING

import numpy as np

from order_flow_stepper import OrderFlowStepper, Step
from trade_stats import (
    cancel_all_orders,
    get_lot_size,
    get_min_order_amount_tick_lot,
    get_min_order_qty_lots,
    get_tick_size,
    place_limit_order_lots,
)
from order_flow_testers import MarketOrderFlowTester

if TYPE_CHECKING:
    from market_making_controller import Controller
    from market_making_controller.models import ExecutionRule, ExecutionModel

from market_making_controller.state import DiscretizedOrderAction, State, ZERO_DISCRETIZED_ACTION, state_from_orderbook, state_from_step
from market_making_controller.ob_reference_prices import mid_ticks_after_market_order_flow

class OrderFlowRunner:
    """
    Runs the order-flow simulation step by step.
    Each advance(): fetch step from stepper, apply transition via
    ``runner_execution_model.state_after_step`` (falls back to ``execution_model``),
    update estimators/testers, optionally get new action.

    Diagnostic Monte Carlo uses ``expected_mid_n_samples`` for: E[mid] (MOF-only book + residual),
    rep-price prediction batches, and E[MOF] as the mean of repeated ``mof_model.sample()`` calls.
    MPC uses its own ``n_samples`` on the optimizer (typically larger).
    """

    def __init__(
        self,
        stepper: OrderFlowStepper,
        model: object,
        tester: MarketOrderFlowTester,
        *,
        controller: Controller | None = None,
        initial_inventory: int = 0,
        initial_cash: float = 0.0,
        price_model: object = None,
        execution_rule: ExecutionRule | None = None,
        execution_model: ExecutionModel | None = None,
        runner_execution_model: ExecutionModel | None = None,
        price_tester: object = None,
        spread_max: float | None = None,
        use_zero_action: bool = False,
        limit_order_flow_model: object = None,
        expected_mid_n_samples: int = 10,
        expected_mid_debug: bool = False,
        expected_mid_debug_every: int = 1,
        live_order_execution: bool = False,
        action_start_after_steps: int = 0,
    ):
        self.stepper = stepper
        # Preferred name for market-order-flow estimator.
        self.mof_model = model
        # Backward-compatible alias (legacy code still reads runner.model).
        self.model = self.mof_model
        self.tester = tester
        self.controller = controller
        self.use_zero_action = use_zero_action
        self.price_model = price_model
        self.price_tester = price_tester
        self.limit_order_flow_model = limit_order_flow_model
        self.execution_model = execution_model
        self.runner_execution_model = runner_execution_model
        self.spread_max = spread_max
        self.expected_mid_n_samples = max(1, int(expected_mid_n_samples))
        self.expected_mid_debug = bool(expected_mid_debug)
        self.expected_mid_debug_every = max(1, int(expected_mid_debug_every))
        self.live_order_execution = bool(live_order_execution)
        self.action_start_after_steps = max(0, int(action_start_after_steps))
        self._live_exec_warned_no_token = False
        self._step_counter = 0
        if execution_model is None and controller is not None and execution_rule is None:
            from market_making_controller.models import DefaultExecutionRule
            execution_rule = DefaultExecutionRule()
        self.execution_rule = execution_rule

        self.inventory = initial_inventory
        self.cash = initial_cash
        self._pending_action = None
        self._pending_state = None

        self.action_times: list[datetime] = []
        self.delta_b_hist: list[float] = []
        self.delta_a_hist: list[float] = []
        self.q_b_hist: list[float] = []
        self.q_a_hist: list[float] = []
        self.expected_buy_flow_hist: list[float] = []
        self.expected_sell_flow_hist: list[float] = []
        self.expected_mid_hist: list[float] = []  # MC mean mid incl. residual
        # Representative-price prediction diagnostics for price window subplot.
        self.pred_buy_rep_error_hist: list[float] = []
        self.pred_sell_rep_error_hist: list[float] = []
        self.pred_buy_rep_prev_error_hist: list[float] = []
        self.pred_sell_rep_prev_error_hist: list[float] = []
        self.market_implied_mid_hist: list[float] = []  # mid from prev OB minus realized market flow only (for plot)
        self.last_book_params: dict | None = None  # fitted depth params from last MPC step (for animation)
        # E[ΔW]: optimizer’s expected total ΔW over horizon (sum over H steps, mean over MC samples). Can be large (e.g. 1e9) if horizon or notional is large.
        self.E_dW_times: list[datetime] = []  # when action was taken
        self.E_dW_hist: list[float] = []
        self.J_comp_times: list[datetime] = []  # when optimizer components were recorded
        self.J_mean_hist: list[float] = []
        self.J_inv_hist: list[float] = []
        self.J_std_hist: list[float] = []
        self.J_uniformity_hist: list[float] = []
        # Realized ΔW: actual wealth change in one step when fills are applied. Often 0 or small if no fills.
        self.dW_times: list[datetime] = []  # when fills were applied
        self.dW_hist: list[float] = []
        self.last_sigmoid_action = None  # last SigmoidMixtureAction from optimizer (for plotting action curves)
        self.last_discretized_action = None  # last DiscretizedOrderAction from controller.act() (for marginal/cumulative plots)
        # PnL decomposition (spread, inventory, cross, fee) for plotting; same time index as dW
        self.pnl_times: list[datetime] = []
        self.cum_spread_pnl_hist: list[float] = []
        self.cum_inventory_pnl_hist: list[float] = []
        self.cum_cross_pnl_hist: list[float] = []
        self.cum_fee_pnl_hist: list[float] = []
        self._cum_spread_pnl = 0.0
        self._cum_inventory_pnl = 0.0
        self._cum_cross_pnl = 0.0
        self._cum_fee_pnl = 0.0
        self.inv_times: list[datetime] = []
        self.inv_hist: list[int] = []
        self.cash_hist: list[float] = []
        self.inv_value_hist: list[float] = []
        self.wealth_times: list[datetime] = []
        self.wealth_hist: list[float] = []
        # Wealth reconstructed as W0 + cumulative sum of realized dW.
        self.wealth_from_dw_times: list[datetime] = []
        self.wealth_from_dw_hist: list[float] = []
        self._wealth_from_dw_baseline: float | None = None
        self._wealth_from_dw_running: float | None = None

    @property
    def steps(self) -> list[Step]:
        return self.stepper.steps

    @staticmethod
    def _to_scalar(v) -> float:
        try:
            if hasattr(v, "detach"):
                import torch
                t = v.detach()
                if isinstance(t, torch.Tensor):
                    return float(t.cpu().reshape(-1)[0].item())
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                return 0.0
            return float(arr[0])
        except Exception:
            return float(v)


    @staticmethod
    def _symbol_assets(symbol: str) -> tuple[str, str]:
        s = str(symbol or "").strip().upper()
        if "_" in s:
            base, quote = s.split("_", 1)
        else:
            base, quote = s[:-3], s[-3:]
        if quote == "IRT":
            quote = "RLS"
        return base, quote

    @staticmethod
    def _depth_by_price(ob, side: str) -> dict[int, float]:
        levels = ob.bids if side == "bid" else ob.asks
        out: dict[int, float] = {}
        for lev in levels:
            p = int(lev.price)
            out[p] = out.get(p, 0.0) + float(lev.qty)
        return out

    @staticmethod
    def _consume_depth(depth: dict[int, float], vol: float, best_first_asc: bool) -> dict[int, float]:
        out = dict(depth)
        remaining = max(0.0, float(vol))
        for p in sorted(list(out.keys()), reverse=not best_first_asc):
            if remaining <= 0:
                break
            q = float(out.get(p, 0.0))
            take = min(q, remaining)
            q_new = q - take
            remaining -= take
            if q_new > 0:
                out[p] = q_new
            elif p in out:
                del out[p]
        return out

    @staticmethod
    def _total_depth(depth: dict[int, float]) -> float:
        return float(sum(max(0.0, q) for q in depth.values()))

    @staticmethod
    def _vwap_to_depth(depth: dict[int, float], depth_lots: float, best_first_asc: bool) -> float:
        if depth_lots <= 0:
            return float("nan")
        remaining = float(depth_lots)
        px_qty = 0.0
        for p in sorted(depth.keys(), reverse=not best_first_asc):
            if remaining <= 0:
                break
            q = max(0.0, float(depth.get(p, 0.0)))
            if q <= 0:
                continue
            take = min(q, remaining)
            px_qty += float(p) * take
            remaining -= take
        if remaining > 1e-9:
            return float("nan")
        return float(px_qty / depth_lots)

    def _estimate_expected_representative_prices_mc(
        self, prev_step: Step, n_samples: int, curr_step: Step | None = None
    ) -> tuple[float, float] | None:
        if self.mof_model is None or self.price_model is None:
            return None
        _tp, prev_ob, _bp, _sp, *_ = prev_step
        bid_prev = self._depth_by_price(prev_ob, "bid")
        ask_prev = self._depth_by_price(prev_ob, "ask")
        target_depth = float(getattr(self.price_model, "_target_depth_lots", 1000.0))
        depth_cur: float | None = None
        if curr_step is not None:
            _tc, cur_ob, _bw, _sw, *_ = curr_step
            bid_cur = self._depth_by_price(cur_ob, "bid")
            ask_cur = self._depth_by_price(cur_ob, "ask")
            depth_cur = float(min(self._total_depth(bid_cur), self._total_depth(ask_cur)))
        buy_preds: list[float] = []
        sell_preds: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.mof_model.sample(())
            buy_vol = self._to_scalar(bv)
            sell_vol = self._to_scalar(sv)
            bid_after = self._consume_depth(bid_prev, float(sell_vol), best_first_asc=False)
            ask_after = self._consume_depth(ask_prev, float(buy_vol), best_first_asc=True)
            depth_cmp = float(min(self._total_depth(bid_after), self._total_depth(ask_after)))
            if depth_cur is not None:
                depth_used = float(min(target_depth, depth_cur, depth_cmp))
            else:
                depth_used = float(min(target_depth, depth_cmp))
            buy_cmp = self._vwap_to_depth(ask_after, depth_used, best_first_asc=True)
            sell_cmp = self._vwap_to_depth(bid_after, depth_used, best_first_asc=False)
            eps_b, eps_s = self.price_model.sample_residual(())
            buy_pred = float(buy_cmp + self._to_scalar(eps_b))
            sell_pred = float(sell_cmp + self._to_scalar(eps_s))
            if np.isfinite(buy_pred) and np.isfinite(sell_pred):
                buy_preds.append(buy_pred)
                sell_preds.append(sell_pred)
        if not buy_preds or not sell_preds:
            return None
        return float(np.mean(buy_preds)), float(np.mean(sell_preds))

    def _expected_mof_volume_mc(self, n_samples: int) -> tuple[float, float]:
        """
        Monte Carlo mean of market buy/sell volume by repeatedly calling mof_model.sample().
        Matches the generative graph (e.g. hurdle then LogNormal); converges to the same
        expectation as the closed-form (1-p0)*E[LogNormal] for the continuous sampler.
        """
        if self.mof_model is None or not hasattr(self.mof_model, "sample"):
            return (0.0, 0.0)
        buy_acc: list[float] = []
        sell_acc: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.mof_model.sample(())
            b = self._to_scalar(bv)
            s = self._to_scalar(sv)
            if np.isfinite(b) and np.isfinite(s):
                buy_acc.append(float(max(0.0, b)))
                sell_acc.append(float(max(0.0, s)))
        if not buy_acc:
            return (0.0, 0.0)
        return (float(np.mean(buy_acc)), float(np.mean(sell_acc)))

    def _estimate_expected_mid_mc(
        self, prev_step: Step, n_samples: int, curr_step: Step | None = None
    ) -> float | None:
        if self.mof_model is None or self.price_model is None:
            return None
        prev_ob = prev_step[1]
        mids: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.mof_model.sample(())
            buy_vol = self._to_scalar(bv)
            sell_vol = self._to_scalar(sv)
            mid_flow = float(
                mid_ticks_after_market_order_flow(
                    prev_ob, buy_vol, sell_vol, spread_max=float(self.spread_max)
                )
            )
            residual = self._to_scalar(self.price_model.sample(()))
            mids.append(float(mid_flow + residual))
        if not mids:
            return None
        return float(np.mean(mids))


    def _execute_live_orders(
        self,
        action: DiscretizedOrderAction,
        state: State,
    ) -> None:
        """For real execution mode: cancel existing active orders, then place current ladder."""
        ex = str(getattr(self.stepper, "exchange", "") or "").strip()
        sym = str(getattr(self.stepper, "symbol", "") or "").strip()
        api_token = getattr(self.stepper, "private_api_token", None)
        if not (ex and sym):
            return
        if not api_token:
            if not self._live_exec_warned_no_token:
                print("[live_execution] skipped: missing private_api_token")
                self._live_exec_warned_no_token = True
            return
        print(
            f"[live_execution] cancel_all request exchange={ex} symbol={sym} execution=limit"
        )
        t0_cancel = time.perf_counter()
        cancel_resp = cancel_all_orders(sym, ex, api_token=api_token, execution="limit")
        cancel_elapsed_ms = (time.perf_counter() - t0_cancel) * 1000.0
        print(
            f"[live_execution] cancel_all response elapsed_ms={cancel_elapsed_ms:.1f} "
            f"response={cancel_resp}"
        )
        if str(cancel_resp.get("status", "")).lower() != "ok":
            print("[live_execution] skipped placements: cancel_all not ok")
            return

        base_asset, quote_asset = self._symbol_assets(sym)
        tick_size = float(get_tick_size(sym, ex))
        lot_size = float(get_lot_size(sym, ex))
        min_qty_lots = max(0.0, float(get_min_order_qty_lots(sym, ex)))
        min_notional_tick_lot = max(0.0, float(get_min_order_amount_tick_lot(sym, ex)))
        # Use current state totals (already synchronized by stepper in real mode) for
        # feasibility checks to avoid an extra balances REST request per step.
        avail_base = max(0.0, float(state.inventory) * lot_size)
        avail_quote = max(0.0, float(state.cash) * tick_size * lot_size)
        print(
            f"[live_execution] balances derived_from_state base_asset={base_asset} avail_base={avail_base:.12f} "
            f"quote_asset={quote_asset} avail_quote={avail_quote:.12f} "
            f"lot_size={lot_size:.12f} tick_size={tick_size:.12f} "
            f"min_qty_lots={min_qty_lots:.6f} min_notional_tick_lot={min_notional_tick_lot:.6f}"
        )

        mid_ticks = float(state.mid)
        for spread, qty in action.bids:
            q = int(qty)
            if q <= 0:
                continue
            px = int(round(mid_ticks - float(spread)))
            if px <= 0:
                continue
            amount_base = float(q) * lot_size
            price_quote = float(px) * tick_size
            notional_quote = amount_base * price_quote
            if min_qty_lots > 0.0 and float(q) < min_qty_lots:
                print(
                    f"[live_execution] skip buy below min qty symbol={sym} "
                    f"price_ticks={px} qty_lots={q} min_qty_lots={min_qty_lots:.6f}"
                )
                continue
            if min_notional_tick_lot > 0.0 and float(q) * float(px) < min_notional_tick_lot:
                print(
                    f"[live_execution] skip buy below min notional symbol={sym} "
                    f"price_ticks={px} qty_lots={q} notional_tick_lot={float(q) * float(px):.6f} "
                    f"min_notional_tick_lot={min_notional_tick_lot:.6f}"
                )
                continue
            if notional_quote > avail_quote + 1e-12:
                print(
                    f"[live_execution] skip buy insufficient quote symbol={sym} "
                    f"price_ticks={px} qty_lots={q} amount_base={amount_base:.12f} "
                    f"price_quote={price_quote:.12f} need_quote={notional_quote:.12f} avail_quote={avail_quote:.12f}"
                )
                continue
            print(
                f"[live_execution] place request side=buy symbol={sym} "
                f"mid_ticks={mid_ticks:.6f} spread_ticks={float(spread):.6f} "
                f"price_ticks={px} qty_lots={q} amount_base={amount_base:.12f} "
                f"price_quote={price_quote:.12f} notional_quote={notional_quote:.12f}"
            )
            t0_place = time.perf_counter()
            resp = place_limit_order_lots(
                symbol=sym,
                exchange=ex,
                side="buy",
                price_ticks=px,
                qty_lots=q,
                api_token=api_token,
            )
            place_elapsed_ms = (time.perf_counter() - t0_place) * 1000.0
            print(
                f"[live_execution] place response side=buy elapsed_ms={place_elapsed_ms:.1f} "
                f"response={resp}"
            )
            if str(resp.get("status", "")).lower() == "ok":
                avail_quote = max(0.0, avail_quote - notional_quote)

        for spread, qty in action.asks:
            q = int(qty)
            if q <= 0:
                continue
            px = int(round(mid_ticks + float(spread)))
            if px <= 0:
                continue
            amount_base = float(q) * lot_size
            price_quote = float(px) * tick_size
            notional_quote = amount_base * price_quote
            if min_qty_lots > 0.0 and float(q) < min_qty_lots:
                print(
                    f"[live_execution] skip sell below min qty symbol={sym} "
                    f"price_ticks={px} qty_lots={q} min_qty_lots={min_qty_lots:.6f}"
                )
                continue
            if min_notional_tick_lot > 0.0 and float(q) * float(px) < min_notional_tick_lot:
                print(
                    f"[live_execution] skip sell below min notional symbol={sym} "
                    f"price_ticks={px} qty_lots={q} notional_tick_lot={float(q) * float(px):.6f} "
                    f"min_notional_tick_lot={min_notional_tick_lot:.6f}"
                )
                continue
            if amount_base > avail_base + 1e-12:
                print(
                    f"[live_execution] skip sell insufficient base symbol={sym} "
                    f"price_ticks={px} qty_lots={q} amount_base={amount_base:.12f} avail_base={avail_base:.12f}"
                )
                continue
            print(
                f"[live_execution] place request side=sell symbol={sym} "
                f"mid_ticks={mid_ticks:.6f} spread_ticks={float(spread):.6f} "
                f"price_ticks={px} qty_lots={q} amount_base={amount_base:.12f} "
                f"price_quote={price_quote:.12f} notional_quote={notional_quote:.12f}"
            )
            t0_place = time.perf_counter()
            resp = place_limit_order_lots(
                symbol=sym,
                exchange=ex,
                side="sell",
                price_ticks=px,
                qty_lots=q,
                api_token=api_token,
            )
            place_elapsed_ms = (time.perf_counter() - t0_place) * 1000.0
            print(
                f"[live_execution] place response side=sell elapsed_ms={place_elapsed_ms:.1f} "
                f"response={resp}"
            )
            if str(resp.get("status", "")).lower() == "ok":
                avail_base = max(0.0, avail_base - amount_base)

    def advance(self, t_end_ms: int | None = None) -> bool:
        """
        One iteration: get current step from stepper; apply previous action with current
        step's volumes via execution model (→ inventory, cash, dW, pnl); update
        estimators/testers; optionally compute new action from controller.
        """
        step = self.stepper.advance(t_end_ms=t_end_ms)
        if step is None:
            return False
        self._step_counter += 1
        t_end, ob, buy_vol, sell_vol, _step_inventory, _step_cash, *_ = step
        bucket_ms = self.stepper.T_ms
        mid_now_ticks = float(ob.mid)

        # Apply previous action with current step's volumes → current state (execution model returns state with current ob)
        current_state = None
        _exec = self.runner_execution_model or self.execution_model
        dW_step: float | None = None
        if self._pending_state is not None and self._pending_action is not None and _exec is not None:
            current_state, delta_pnl = _exec.state_after_step(
                self._pending_state,
                self._pending_action,
                step,
                spread_max=self.spread_max,
            )
            self.inventory = current_state.inventory
            self.cash = current_state.cash
            t_fill = datetime.fromtimestamp(t_end / 1000.0, tz=timezone.utc)
            dW_step = float(delta_pnl.wealth_pnl)
            self.dW_times.append(t_fill)
            self.dW_hist.append(dW_step)
            self.pnl_times.append(t_fill)
            self._cum_spread_pnl += float(delta_pnl.spread_pnl)
            self._cum_inventory_pnl += float(delta_pnl.inventory_pnl)
            self._cum_cross_pnl += float(delta_pnl.cross_pnl)
            self._cum_fee_pnl += float(delta_pnl.fee_pnl)
            self.cum_spread_pnl_hist.append(self._cum_spread_pnl)
            self.cum_inventory_pnl_hist.append(self._cum_inventory_pnl)
            self.cum_cross_pnl_hist.append(self._cum_cross_pnl)
            self.cum_fee_pnl_hist.append(self._cum_fee_pnl)

            inv_value = self.inventory * mid_now_ticks
            wealth = self.cash + inv_value
            t_fill = datetime.fromtimestamp(t_end / 1000.0, tz=timezone.utc)
            self.inv_times.append(t_fill)
            self.inv_hist.append(self.inventory)
            self.cash_hist.append(self.cash)
            self.inv_value_hist.append(inv_value)
            self.wealth_times.append(t_fill)
            self.wealth_hist.append(wealth)
            if dW_step is not None:
                if self._wealth_from_dw_running is None:
                    # First observed transition: infer pre-transition baseline W0.
                    self._wealth_from_dw_baseline = float(wealth - dW_step)
                    self._wealth_from_dw_running = float(self._wealth_from_dw_baseline + dW_step)
                else:
                    self._wealth_from_dw_running = float(self._wealth_from_dw_running + dW_step)
                self.wealth_from_dw_times.append(t_fill)
                self.wealth_from_dw_hist.append(float(self._wealth_from_dw_running))

        # Score with previous params, then update all estimators (same signature: prev_step, curr_step)
        prev_step = self.steps[-2] if len(self.steps) >= 2 else None
        curr_step = step

        # Predictive expected next mid (for transition prev->curr) by MC:
        # sample market-flow and residual; mid from MOF-only book evolution on prev OB.
        rep_pred: tuple[float, float] | None = None
        if prev_step is not None and self.price_model is not None:
            mid_exp = self._estimate_expected_mid_mc(
                prev_step, self.expected_mid_n_samples, curr_step=curr_step
            )
            if mid_exp is not None:
                self.expected_mid_hist.append(float(mid_exp))
            rep_pred = self._estimate_expected_representative_prices_mc(
                prev_step, self.expected_mid_n_samples, curr_step
            )

        params = getattr(self.mof_model, "get_params", lambda: {})()
        self.tester.update(prev_step, curr_step, params)
        self.mof_model.update(prev_step, curr_step)

        # E[MOF] from the same PGM as MPC sampling: MC mean of model.sample() (not closed-form params).
        e_buy_post, e_sell_post = self._expected_mof_volume_mc(self.expected_mid_n_samples)
        self.expected_buy_flow_hist.append(float(e_buy_post))
        self.expected_sell_flow_hist.append(float(e_sell_post))

        if self.price_model is not None:
            if self.price_tester is not None:
                price_params = self.price_model.get_params()
                self.price_tester.update(prev_step, curr_step, price_params)
            self.price_model.update(prev_step, curr_step)
            if (
                prev_step is not None
                and rep_pred is not None
                and self.price_model.realized_buy_price_hist
                and self.price_model.realized_sell_price_hist
            ):
                buy_mc, sell_mc = rep_pred
                rb = float(self.price_model.realized_buy_price_hist[-1])
                rs = float(self.price_model.realized_sell_price_hist[-1])
                self.pred_buy_rep_error_hist.append(float(buy_mc - rb))
                self.pred_sell_rep_error_hist.append(float(sell_mc - rs))
                if len(self.price_model.realized_buy_price_hist) >= 2:
                    prev_b = float(self.price_model.realized_buy_price_hist[-2])
                    prev_s = float(self.price_model.realized_sell_price_hist[-2])
                    self.pred_buy_rep_prev_error_hist.append(float(prev_b - rb))
                    self.pred_sell_rep_prev_error_hist.append(float(prev_s - rs))
                else:
                    self.pred_buy_rep_prev_error_hist.append(float("nan"))
                    self.pred_sell_rep_prev_error_hist.append(float("nan"))

        if self.limit_order_flow_model is not None:
            self.limit_order_flow_model.update(prev_step, curr_step)

        # Decide new action (cancels previous, places new)
        if self.controller is not None:
            if current_state is None:
                # Initial state source depends on execution path:
                # - observed/real: use Step inventory/cash from balances/private events
                # - simulated: use runner inventory/cash (simulated initial + transitions)
                _ref_depth = (
                    float(getattr(self.price_model, "_target_depth_lots", 1000.0))
                    if self.price_model is not None
                    else 1000.0
                )
                if self.runner_execution_model is not None and self.runner_execution_model.__class__.__name__ == "ObservedStepExecutionModel":
                    current_state = state_from_step(
                        step,
                        spread_max=self.spread_max,
                        reference_depth_lots=_ref_depth,
                    )
                    self.inventory = current_state.inventory
                    self.cash = current_state.cash
                else:
                    current_state = state_from_orderbook(
                        ob,
                        buy_vol=float(buy_vol),
                        sell_vol=float(sell_vol),
                        inventory=self.inventory,
                        cash=self.cash,
                        spread_max=self.spread_max,
                        reference_depth_lots=_ref_depth,
                    )
            opt = getattr(self.controller, "optimizer", None)
            if self.use_zero_action:
                action = ZERO_DISCRETIZED_ACTION
                self.last_book_params = None
                self.last_sigmoid_action = None
                self.last_discretized_action = None
            else:
                action = self.controller.act(current_state)
                self.last_book_params = getattr(opt, "_last_book_params", None)
                self.last_sigmoid_action = getattr(opt, "_last_sigmoid_action", None)
            if isinstance(action, DiscretizedOrderAction):
                self.last_discretized_action = action
                if self.live_order_execution:
                    # Warm-up gate applies to live order placement only; controller
                    # still runs and action/diagnostics are recorded from step 1.
                    if self._step_counter > self.action_start_after_steps:
                        self._execute_live_orders(action, current_state)
                E_dW = None if self.use_zero_action else getattr(opt, "_last_E_dW", None)
                t_act = datetime.fromtimestamp((t_end - bucket_ms / 2) / 1000.0, tz=timezone.utc)
                if E_dW is not None:
                    self.E_dW_times.append(t_act)
                    self.E_dW_hist.append(float(E_dW))
                comp = None if self.use_zero_action else getattr(opt, "_last_J_components", None)
                if isinstance(comp, dict):
                    self.J_comp_times.append(t_act)
                    # Weighted objective terms as returned by terminal_objective.
                    self.J_mean_hist.append(float(comp.get("term_reward", 0.0)))
                    self.J_inv_hist.append(float(comp.get("term_inv", 0.0)))
                    self.J_std_hist.append(float(comp.get("term_risk", 0.0)))
                    self.J_uniformity_hist.append(float(comp.get("term_uniformity", 0.0)))
            self._pending_action = action
            self._pending_state = current_state
            t_act = datetime.fromtimestamp((t_end - bucket_ms / 2) / 1000.0, tz=timezone.utc)
            self.action_times.append(t_act)
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
            self.delta_b_hist.append(delta_b)
            self.delta_a_hist.append(delta_a)
            self.q_b_hist.append(q_b)
            self.q_a_hist.append(q_a)

        return True

    def advance_scheduled(self, max_steps_per_call: int = 5) -> int:
        """
        Run steps at exact wall-clock cadence (self.stepper.T_ms), independent of animation refresh rate.
        Returns number of successful steps executed in this call.
        """
        period_s = self.stepper.T_ms / 1000.0
        now_perf = time.perf_counter()
        now_wall_ms = int(time.time() * 1000)

        if not hasattr(self, "_next_due_perf"):
            self._next_due_perf = now_perf
            self._next_step_end_ms = now_wall_ms

        steps_done = 0
        while now_perf >= self._next_due_perf and steps_done < max_steps_per_call:
            ok = self.advance(t_end_ms=int(self._next_step_end_ms))
            self._next_due_perf += period_s
            self._next_step_end_ms += self.stepper.T_ms
            if not ok:
                break
            steps_done += 1
            now_perf = time.perf_counter()

        return steps_done
