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
from trade_stats import cancel_all_orders, get_lot_size, get_tick_size, place_limit_order_lots
from order_flow_testers import MarketOrderFlowTester

if TYPE_CHECKING:
    from market_making_controller import Controller
    from market_making_controller.models import ExecutionRule, ExecutionModel

from market_making_controller.state import DiscretizedOrderAction, State, ZERO_DISCRETIZED_ACTION, state_from_orderbook, state_from_step
from market_making_controller.ob_reference_prices import mid_ticks_after_market_order_flow
from market_making_controller.models.price_torch import _expected_buy_sell_vol_from_params

class OrderFlowRunner:
    """
    Runs the order-flow simulation step by step.
    Each advance(): fetch step from stepper, apply transition via
    ``runner_execution_model.state_after_step`` (falls back to ``execution_model``),
    update estimators/testers, optionally get new action.

    Diagnostic Monte Carlo uses ``expected_mid_n_samples`` for: E[mid] (MOF-only book + residual),
    rep-price prediction batches, and E[MOF] as the mean of repeated ``model.sample()`` calls.
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
        self.model = model
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
        self.last_expected_mid_diag: dict | None = None
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
        # Counterfactual expected Δmid vs prev_mid (same MC batch): MOF-only vs LOF-only vs full flow vs residual.
        self.expected_mid_delta_mof_hist: list[float] = []
        self.expected_mid_delta_lof_hist: list[float] = []
        self.expected_mid_delta_full_flow_hist: list[float] = []
        self.expected_mid_delta_residual_hist: list[float] = []
        self.pred_buy_rep_error_hist: list[float] = []
        self.pred_sell_rep_error_hist: list[float] = []
        self.pred_buy_rep_prev_error_hist: list[float] = []
        self.pred_sell_rep_prev_error_hist: list[float] = []
        self._pred_buy_rep_ewma: float | None = None
        self._pred_sell_rep_ewma: float | None = None
        self._pending_rep_mc: tuple[float, float] | None = None
        self.market_implied_mid_hist: list[float] = []  # mid from prev OB minus realized market flow only (for plot)
        self.last_book_params: dict | None = None  # fitted depth params from last MPC step (for animation)
        # E[ΔW]: optimizer’s expected total ΔW over horizon (sum over H steps, mean over MC samples). Can be large (e.g. 1e9) if horizon or notional is large.
        self.E_dW_times: list[datetime] = []  # when action was taken
        self.E_dW_hist: list[float] = []
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

    @property
    def steps(self) -> list[Step]:
        return self.stepper.steps

    @staticmethod
    def _best_touch_lots(ob) -> tuple[int, int]:
        """Best bid / best ask displayed size in integer lots (same unit as Step buy_vol/sell_vol)."""
        if not getattr(ob, "bids", None):
            bb_lots = 0
        else:
            bb_lev = max(ob.bids, key=lambda L: L.price)
            bb_lots = int(bb_lev.qty)
        if not getattr(ob, "asks", None):
            ba_lots = 0
        else:
            ba_lev = min(ob.asks, key=lambda L: L.price)
            ba_lots = int(ba_lev.qty)
        return bb_lots, ba_lots

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

    def _flow_implied_spread_max(self) -> int:
        """Ticks from mid for MOF-only mid diagnostics; should match ``spread_max`` passed to MPC."""
        return max(1, int(self.spread_max)) if self.spread_max is not None else 100

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

    def _representative_prices_from_transition(self, prev_step: Step, curr_step: Step) -> tuple[float, float]:
        _tp, prev_ob, _bp, _sp, *_ = prev_step
        _tc, cur_ob, buy_vol, sell_vol, *_ = curr_step
        bid_prev = self._depth_by_price(prev_ob, "bid")
        ask_prev = self._depth_by_price(prev_ob, "ask")
        bid_after = self._consume_depth(bid_prev, float(sell_vol), best_first_asc=False)
        ask_after = self._consume_depth(ask_prev, float(buy_vol), best_first_asc=True)
        bid_cur = self._depth_by_price(cur_ob, "bid")
        ask_cur = self._depth_by_price(cur_ob, "ask")
        target_depth = float(getattr(self.price_model, "_target_depth_lots", 1000.0))
        depth_cur = min(self._total_depth(bid_cur), self._total_depth(ask_cur))
        depth_cmp = min(self._total_depth(bid_after), self._total_depth(ask_after))
        depth_used = float(min(target_depth, depth_cur, depth_cmp))
        buy_real = self._vwap_to_depth(ask_cur, depth_used, best_first_asc=True)
        sell_real = self._vwap_to_depth(bid_cur, depth_used, best_first_asc=False)
        return float(buy_real), float(sell_real)

    def _estimate_expected_representative_prices_mc(
        self, prev_step: Step, n_samples: int, curr_step: Step | None = None
    ) -> tuple[float, float] | None:
        if self.model is None or self.price_model is None:
            return None
        _tp, prev_ob, _bp, _sp, *_ = prev_step
        bid_prev = self._depth_by_price(prev_ob, "bid")
        ask_prev = self._depth_by_price(prev_ob, "ask")
        target_depth = float(getattr(self.price_model, "_target_depth_lots", 1000.0))
        # Match TorchResidualPriceModel.update / _representative_prices_from_transition: VWAP depth is
        # min(target, depth_cur, depth_cmp). Omitting depth_cur biased MC VWAP vs realized EWMA (often
        # asymmetric on bid vs ask when the binding constraint differs per side).
        depth_cur: float | None = None
        if curr_step is not None:
            _tc, cur_ob, _bw, _sw, *_ = curr_step
            bid_cur = self._depth_by_price(cur_ob, "bid")
            ask_cur = self._depth_by_price(cur_ob, "ask")
            depth_cur = float(min(self._total_depth(bid_cur), self._total_depth(ask_cur)))
        buy_preds: list[float] = []
        sell_preds: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.model.sample(())
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
        Monte Carlo mean of market buy/sell volume by repeatedly calling model.sample().
        Matches the generative graph (e.g. hurdle then LogNormal); converges to the same
        expectation as the closed-form (1-p0)*E[LogNormal] for the continuous sampler.
        """
        if self.model is None or not hasattr(self.model, "sample"):
            return (0.0, 0.0)
        buy_acc: list[float] = []
        sell_acc: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.model.sample(())
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
        if self.model is None or self.price_model is None:
            return None
        prev_ob = prev_step[1]
        prev_mid = float(prev_ob.mid)
        touch_bb_lots, touch_ba_lots = self._best_touch_lots(prev_ob)
        spread = self._flow_implied_spread_max()
        mf_params = getattr(self.model, "get_params", lambda: {})()
        e_buy, e_sell = _expected_buy_sell_vol_from_params(mf_params)
        window_buy_lots: int | None = None
        window_sell_lots: int | None = None
        if curr_step is not None:
            _te, _ob_c, bw, sw, *_ = curr_step
            window_buy_lots = int(round(float(bw)))
            window_sell_lots = int(round(float(sw)))

        mids: list[float] = []
        flow_mids: list[float] = []
        mid_mof_only: list[float] = []
        mid_lof_only: list[float] = []
        residuals: list[float] = []
        buy_samples: list[float] = []
        sell_samples: list[float] = []
        for _ in range(max(1, int(n_samples))):
            bv, sv = self.model.sample(())
            buy_vol = self._to_scalar(bv)
            sell_vol = self._to_scalar(sv)
            buy_samples.append(float(buy_vol))
            sell_samples.append(float(sell_vol))
            mid_flow = float(
                mid_ticks_after_market_order_flow(
                    prev_ob, buy_vol, sell_vol, spread_max=float(spread)
                )
            )
            flow_mids.append(mid_flow)
            mid_mof_only.append(mid_flow)
            mid_lof_only.append(
                float(
                    mid_ticks_after_market_order_flow(
                        prev_ob, e_buy, e_sell, spread_max=float(spread)
                    )
                )
            )
            residual = self._to_scalar(self.price_model.sample(()))
            residuals.append(float(residual))
            mids.append(float(mid_flow + residual))
        if not mids:
            return None
        flow_arr = np.asarray(flow_mids, dtype=np.float64)
        res_arr = np.asarray(residuals, dtype=np.float64)
        mid_arr = np.asarray(mids, dtype=np.float64)
        mof_arr = np.asarray(mid_mof_only, dtype=np.float64)
        lof_arr = np.asarray(mid_lof_only, dtype=np.float64)
        buy_arr = np.asarray(buy_samples, dtype=np.float64)
        sell_arr = np.asarray(sell_samples, dtype=np.float64)
        delta_mof = float(np.mean(mof_arr - prev_mid))
        delta_lof = float(np.mean(lof_arr - prev_mid))
        delta_full_flow = float(np.mean(flow_arr - prev_mid))
        delta_res = float(np.mean(res_arr))
        r_mu, r_sig = 0.0, 0.5
        if self.price_model is not None and hasattr(self.price_model, "get_params"):
            pp = self.price_model.get_params() or {}
            r_mu = float(pp.get("mu_mid", 0.0))
            r_sig = float(pp.get("sigma_mid", 0.5))
        self.last_expected_mid_diag = {
            "prev_mid": prev_mid,
            "n_samples": int(len(mids)),
            "expected_mid_mean": float(np.mean(mid_arr)),
            "expected_mid_std": float(np.std(mid_arr)),
            "flow_mid_mean": float(np.mean(flow_arr)),
            "flow_mid_std": float(np.std(flow_arr)),
            "flow_delta_mean": float(np.mean(flow_arr - prev_mid)),
            "flow_delta_std": float(np.std(flow_arr - prev_mid)),
            "flow_mid_close_to_prev_ratio": float(np.mean(np.abs(flow_arr - prev_mid) <= 0.5)),
            "residual_mean": float(np.mean(res_arr)),
            "residual_std": float(np.std(res_arr)),
            "buy_mean": float(np.mean(buy_arr)),
            "buy_std": float(np.std(buy_arr)),
            "sell_mean": float(np.mean(sell_arr)),
            "sell_std": float(np.std(sell_arr)),
            "delta_mof_only_mean": delta_mof,
            "delta_lof_only_mean": delta_lof,
            "delta_full_flow_mean": delta_full_flow,
            "delta_residual_mean": delta_res,
            "e_buy": float(e_buy),
            "e_sell": float(e_sell),
            "touch_best_bid_lots": touch_bb_lots,
            "touch_best_ask_lots": touch_ba_lots,
            "window_buy_lots": window_buy_lots,
            "window_sell_lots": window_sell_lots,
            # Expected window flow vs one touch side (dimensionless): buy hits ask, sell hits bid.
            "e_buy_over_touch_ask": float(e_buy) / max(touch_ba_lots, 1),
            "e_sell_over_touch_bid": float(e_sell) / max(touch_bb_lots, 1),
            # Price residual model N(μ,σ²) used in MC (EWMA of VWAP residuals when learn_residual).
            "residual_model_mu": r_mu,
            "residual_model_sigma": r_sig,
        }
        self.expected_mid_delta_mof_hist.append(delta_mof)
        self.expected_mid_delta_lof_hist.append(delta_lof)
        self.expected_mid_delta_full_flow_hist.append(delta_full_flow)
        self.expected_mid_delta_residual_hist.append(delta_res)
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
        # Use current state totals (already synchronized by stepper in real mode) for
        # feasibility checks to avoid an extra balances REST request per step.
        avail_base = max(0.0, float(state.inventory) * lot_size)
        avail_quote = max(0.0, float(state.cash) * tick_size * lot_size)
        print(
            f"[live_execution] balances derived_from_state base_asset={base_asset} avail_base={avail_base:.12f} "
            f"quote_asset={quote_asset} avail_quote={avail_quote:.12f} "
            f"lot_size={lot_size:.12f} tick_size={tick_size:.12f}"
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
            dW = float(delta_pnl.wealth_pnl)
            self.dW_times.append(t_fill)
            self.dW_hist.append(dW)
            self.pnl_times.append(t_fill)
            self._cum_spread_pnl += float(delta_pnl.spread_pnl)
            self._cum_inventory_pnl += float(delta_pnl.inventory_pnl)
            self._cum_cross_pnl += float(delta_pnl.cross_pnl)
            self._cum_fee_pnl += float(delta_pnl.fee_pnl)
            self.cum_spread_pnl_hist.append(self._cum_spread_pnl)
            self.cum_inventory_pnl_hist.append(self._cum_inventory_pnl)
            self.cum_cross_pnl_hist.append(self._cum_cross_pnl)
            self.cum_fee_pnl_hist.append(self._cum_fee_pnl)

        if self._pending_state is not None and self._pending_action is not None:
            inv_value = self.inventory * mid_now_ticks
            wealth = self.cash + inv_value
            t_fill = datetime.fromtimestamp(t_end / 1000.0, tz=timezone.utc)
            self.inv_times.append(t_fill)
            self.inv_hist.append(self.inventory)
            self.cash_hist.append(self.cash)
            self.inv_value_hist.append(inv_value)
            self.wealth_times.append(t_fill)
            self.wealth_hist.append(wealth)

        # Score with previous params, then update all estimators (same signature: prev_step, curr_step)
        prev_step = self.steps[-2] if len(self.steps) >= 2 else None
        curr_step = step

        # Predictive expected next mid (for transition prev->curr) by MC:
        # sample market-flow and residual; mid from MOF-only book evolution on prev OB.
        if prev_step is not None and self.price_model is not None:
            mid_exp = self._estimate_expected_mid_mc(
                prev_step, self.expected_mid_n_samples, curr_step=curr_step
            )
            if mid_exp is not None:
                self.expected_mid_hist.append(float(mid_exp))
            rep_pred = self._estimate_expected_representative_prices_mc(
                prev_step, self.expected_mid_n_samples, curr_step
            )
            self._pending_rep_mc = rep_pred
            if (
                mid_exp is not None
                and self.expected_mid_debug
                and (self._step_counter % self.expected_mid_debug_every == 0)
            ):
                d = self.last_expected_mid_diag or {}
                wb = d.get("window_buy_lots")
                ws = d.get("window_sell_lots")
                win_s = (
                    f" win_buy_lots={wb} win_sell_lots={ws}"
                    if wb is not None and ws is not None
                    else ""
                )
                print(
                    "[expected_mid_debug]"
                    f" n={d.get('n_samples', 0)}"
                    f" prev_mid={d.get('prev_mid', 0.0):.4f}"
                    f" E[mid]={d.get('expected_mid_mean', 0.0):.4f}"
                    f" std(mid)={d.get('expected_mid_std', 0.0):.6f}"
                    f" E[mid_flow]={d.get('flow_mid_mean', 0.0):.4f}"
                    f" std(mid_flow)={d.get('flow_mid_std', 0.0):.6f}"
                    f" E[d_mid_flow]={d.get('flow_delta_mean', 0.0):.6f}"
                    f" std(d_mid_flow)={d.get('flow_delta_std', 0.0):.6f}"
                    f" P(|d_mid_flow|<=0.5)={d.get('flow_mid_close_to_prev_ratio', 0.0):.3f}"
                    f" E[res]={d.get('residual_mean', 0.0):.6f}"
                    f" std(res)={d.get('residual_std', 0.0):.6f}"
                    f" res_model_μ={d.get('residual_model_mu', 0.0):.6f}"
                    f" res_model_σ={d.get('residual_model_sigma', 0.0):.6f}"
                    f" E[buy]={d.get('buy_mean', 0.0):.3f}"
                    f" std(buy)={d.get('buy_std', 0.0):.3f}"
                    f" E[sell]={d.get('sell_mean', 0.0):.3f}"
                    f" std(sell)={d.get('sell_std', 0.0):.3f}"
                    "\n  [vol]"
                    f" touch_BB_lots={d.get('touch_best_bid_lots', 0)}"
                    f" touch_BA_lots={d.get('touch_best_ask_lots', 0)}"
                    f" MOF_E_buy_lots={d.get('e_buy', 0.0):.4f}"
                    f" MOF_E_sell_lots={d.get('e_sell', 0.0):.4f}"
                    f"{win_s}"
                    f" ratio_Ebuy_per_touchAsk={d.get('e_buy_over_touch_ask', 0.0):.6f}"
                    f" ratio_Esell_per_touchBid={d.get('e_sell_over_touch_bid', 0.0):.6f}"
                )

        params = getattr(self.model, "get_params", lambda: {})()
        self.tester.update(prev_step, curr_step, params)
        self.model.update(prev_step, curr_step)
        # E[MOF] from the same PGM as MPC sampling: MC mean of model.sample() (not closed-form params).
        e_buy_post, e_sell_post = self._expected_mof_volume_mc(self.expected_mid_n_samples)
        self.expected_buy_flow_hist.append(float(e_buy_post))
        self.expected_sell_flow_hist.append(float(e_sell_post))

        if self.price_model is not None:
            if self.price_tester is not None:
                price_params = self.price_model.get_params()
                self.price_tester.update(prev_step, curr_step, price_params)
            self.price_model.update(prev_step, curr_step)
            # EWMA-smoothed MC rep price vs EWMA smoothed realized (same decay as price model).
            rep_mc = self._pending_rep_mc
            self._pending_rep_mc = None
            if (
                prev_step is not None
                and rep_mc is not None
                and self.price_model.realized_buy_price_hist
                and self.price_model.realized_sell_price_hist
            ):
                buy_mc, sell_mc = rep_mc
                if np.isfinite(buy_mc) and np.isfinite(sell_mc):
                    g = float(getattr(self.price_model, "_gamma", 0.9))
                    a = 1.0 - g
                    rb = float(self.price_model.realized_buy_price_hist[-1])
                    rs = float(self.price_model.realized_sell_price_hist[-1])
                    if self._pred_buy_rep_ewma is None:
                        # Align with price-model cold start: first realized EWMA equals raw VWAP, while
                        # buy_mc can be far off when MOF params are still near init. Anchor the pred EWMA
                        # to the realized EWMA once; MC enters from the *next* step (g*pred + a*buy_mc).
                        self._pred_buy_rep_ewma = rb
                        self._pred_sell_rep_ewma = rs
                    else:
                        self._pred_buy_rep_ewma = g * self._pred_buy_rep_ewma + a * float(buy_mc)
                        self._pred_sell_rep_ewma = g * self._pred_sell_rep_ewma + a * float(sell_mc)
                    self.pred_buy_rep_error_hist.append(float(self._pred_buy_rep_ewma - rb))
                    self.pred_sell_rep_error_hist.append(float(self._pred_sell_rep_ewma - rs))
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
            in_warmup = self._step_counter <= self.action_start_after_steps
            if self.use_zero_action or in_warmup:
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
                    self._execute_live_orders(action, current_state)
                E_dW = None if self.use_zero_action else getattr(opt, "_last_E_dW", None)
                if E_dW is not None:
                    t_act = datetime.fromtimestamp((t_end - bucket_ms / 2) / 1000.0, tz=timezone.utc)
                    self.E_dW_times.append(t_act)
                    self.E_dW_hist.append(float(E_dW))
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
