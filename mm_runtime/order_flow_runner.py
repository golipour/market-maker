"""
Order-flow iteration: one place for the main simulation loop.
Advances stepper, applies fills via ExecutionModel (or ExecutionRule), updates estimators/testers, computes actions.

Canonical location for ``OrderFlowRunner`` (root ``order_flow_runner.py`` re-exports for compatibility).
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING

import numpy as np

from mm_runtime.order_flow_stepper import OrderFlowStepper, Step
from trade_stats import cancel_all_orders
from order_flow_testers import MarketOrderFlowTester

if TYPE_CHECKING:
    from market_making_controller import Controller
    from market_making_controller.models import ExecutionRule, ExecutionModel
    from mm_core.domain.types import ValuationPolicy

from market_making_controller.state import DiscretizedOrderAction, State, ZERO_DISCRETIZED_ACTION, state_from_step
from mm_core.pricing import (
    mid_after_market_order_flow_ticks as mid_ticks_after_market_order_flow,
    resolve_valuation_policy,
    target_depth_lots_from_price_model,
)

from mm_core.app.engine import advance_order_flow_runner
from mm_core.domain.orders import ExecutionReport
from mm_core.domain.types import InstrumentId
from mm_core.units.converter import (
    UnitConverter,
    instrument_spec_from_trade_stats,
)
from mm_adapters.exchanges.execution import execution_report_for_json, place_limit_order_lots_with_report
from mm_adapters.observability.metrics import REGISTRY as METRICS
from mm_adapters.observability.context import metric_labels

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
        expected_mid_n_samples: int = 10,
        expected_mid_debug: bool = False,
        expected_mid_debug_every: int = 1,
        live_order_execution: bool = False,
        action_start_after_steps: int = 0,
        anchor_hedge_enabled: bool = False,
        anchor_price_model: object | None = None,
        valuation_policy: ValuationPolicy | str | None = None,
    ):
        self.stepper = stepper
        self.mof_model = model
        self.tester = tester
        self.controller = controller
        self.use_zero_action = use_zero_action
        self.price_model = price_model
        self.price_tester = price_tester
        self.execution_model = execution_model
        self.runner_execution_model = runner_execution_model
        self.spread_max = spread_max
        self.expected_mid_n_samples = max(1, int(expected_mid_n_samples))
        self.expected_mid_debug = bool(expected_mid_debug)
        self.expected_mid_debug_every = max(1, int(expected_mid_debug_every))
        self.live_order_execution = bool(live_order_execution)
        self.action_start_after_steps = max(0, int(action_start_after_steps))
        self.anchor_hedge_enabled = bool(anchor_hedge_enabled)
        # Anchor rep-price delta model (updated in the runner alongside other models).
        # MPC reads its (mu, sigma) params from the model object (not from State).
        self.anchor_price_model = anchor_price_model
        self.valuation_policy = resolve_valuation_policy(valuation_policy)
        # Cached instrument identity/spec (lazy-filled).
        self._instrument_id = None
        self._instrument_spec = None
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
        self.expected_mid_hist: list[float] = []  # MC mean mid incl. residual
        # Representative-price prediction diagnostics for price window subplot.
        self.pred_buy_rep_error_hist: list[float] = []
        self.pred_sell_rep_error_hist: list[float] = []
        self.pred_buy_rep_prev_error_hist: list[float] = []
        self.pred_sell_rep_prev_error_hist: list[float] = []
        self.market_implied_mid_hist: list[float] = []  # mid from prev OB minus realized market flow only (for plot)
        self.anchor_buy_reference_hist: list[float] = []
        self.anchor_sell_reference_hist: list[float] = []
        self.anchor_reference_times: list[datetime] = []
        self.anchor_price_param_times: list[datetime] = []
        self.anchor_mu_buy_hist: list[float] = []
        self.anchor_mu_sell_hist: list[float] = []
        self.anchor_sigma_buy_hist: list[float] = []
        self.anchor_sigma_sell_hist: list[float] = []
        self.anchor_portfolio_times: list[datetime] = []
        self.anchor_inventory_hist: list[int] = []
        self.anchor_cash_hist: list[float] = []
        self.main_buy_reference_hist: list[float] = []
        self.main_sell_reference_hist: list[float] = []
        self.main_reference_times: list[datetime] = []
        self.last_anchor_orderbook = None
        self.last_anchor_price_model_params: dict | None = None
        self.last_book_params: dict | None = None  # fitted depth params from last MPC step (for animation)
        # E[ΔW_next]: optimizer expected one-step wealth change (t -> t+1), mean over MC samples.
        self.E_dW_times: list[datetime] = []  # when action was taken
        self.E_dW_hist: list[float] = []
        self.J_comp_times: list[datetime] = []  # when optimizer components were recorded
        self.J_mean_hist: list[float] = []
        self.J_inv_hist: list[float] = []
        self.J_std_hist: list[float] = []
        self.J_uniformity_hist: list[float] = []
        self.J_spread_hist: list[float] = []
        self.J_mean_init_hist: list[float] = []
        self.J_inv_init_hist: list[float] = []
        self.J_std_init_hist: list[float] = []
        self.J_uniformity_init_hist: list[float] = []
        self.J_spread_init_hist: list[float] = []
        self.main_reward_opt_hist: list[float] = []
        self.anchor_reward_opt_hist: list[float] = []
        self.main_reward_init_hist: list[float] = []
        self.anchor_reward_init_hist: list[float] = []
        # Realized ΔW: actual wealth change in one step when fills are applied. Often 0 or small if no fills.
        self.dW_times: list[datetime] = []  # when fills were applied
        self.dW_hist: list[float] = []
        self.last_sigmoid_action = None  # last SigmoidMixtureAction from optimizer (for plotting action curves)
        self.last_discretized_action = None  # last DiscretizedOrderAction from controller.act() (for marginal/cumulative plots)
        # PnL decomposition (spread, inventory, cross, fee) for plotting; same time index as dW
        self.pnl_times: list[datetime] = []
        self.delta_spread_pnl_hist: list[float] = []
        self.delta_inventory_pnl_hist: list[float] = []
        self.delta_cross_pnl_hist: list[float] = []
        self.delta_fee_pnl_hist: list[float] = []
        self.inv_times: list[datetime] = []
        self.inv_hist: list[int] = []
        self.cash_hist: list[float] = []
        self.inv_value_hist: list[float] = []
        self.wealth_times: list[datetime] = []
        self.wealth_hist: list[float] = []
        # Canonical-unit wealth series (quote units) computed from mm_core StepSnapshot.
        # This is a migration aid; it does not affect execution or controller behavior.
        self.wealth_quote_times: list[datetime] = []
        self.wealth_quote_hist: list[float] = []
        # Wealth reconstructed as W0 + cumulative sum of realized dW.
        self.wealth_from_dw_times: list[datetime] = []
        self.wealth_from_dw_hist: list[float] = []
        self._wealth_from_dw_baseline: float | None = None
        self._wealth_from_dw_running: float | None = None

    def _get_instrument_id_and_spec(self) -> tuple[object | None, object | None]:
        """
        Lazily build (InstrumentId, InstrumentSpec) for this runner.

        Kept as `object` types to avoid importing mm_core domain types here;
        engine/assemblers supply concrete types when present.
        """
        if self._instrument_id is not None and self._instrument_spec is not None:
            return self._instrument_id, self._instrument_spec
        ex = str(getattr(self.stepper, "exchange", "") or "").strip()
        sym = str(getattr(self.stepper, "symbol", "") or "").strip()
        if not (ex and sym):
            return None, None
        try:
            from mm_core.domain.types import InstrumentId
            from mm_core.units.converter import instrument_spec_from_trade_stats

            self._instrument_id = InstrumentId(exchange=ex, symbol=sym)
            self._instrument_spec = instrument_spec_from_trade_stats(exchange=ex, symbol=sym)
            return self._instrument_id, self._instrument_spec
        except Exception:
            return None, None

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
        target_depth = target_depth_lots_from_price_model(self.price_model)
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
            try:
                METRICS.counter_inc(
                    "mm_live_execution_skipped_total",
                    labels=metric_labels(
                        extra={"exchange": ex, "symbol": sym, "reason": "missing_private_api_token"}
                    ),
                    help="Live execution skips by reason",
                )
            except Exception:
                pass
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
            try:
                METRICS.counter_inc(
                    "mm_live_execution_skipped_total",
                    labels=metric_labels(
                        extra={"exchange": ex, "symbol": sym, "reason": "cancel_all_not_ok"}
                    ),
                    help="Live execution skips by reason",
                )
            except Exception:
                pass
            return

        base_asset, quote_asset = self._symbol_assets(sym)
        _inst, spec = self._get_instrument_id_and_spec()
        if spec is None:
            spec = instrument_spec_from_trade_stats(exchange=ex, symbol=sym)
        place_instrument = _inst if _inst is not None else InstrumentId(exchange=ex, symbol=sym)
        uc = UnitConverter(spec)
        tick_size = float(spec.tick_size)
        lot_size = float(spec.lot_size)
        min_qty_lots = float(spec.min_qty_lots)
        min_notional_tick_lot = float(spec.min_notional_tick_lot)
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
        print(
            f"[live_execution] units base_qty={base_asset} quote={quote_asset} "
            f"min_qty_base={uc.qty_from_lots(int(round(min_qty_lots))):.12f} "
            f"min_notional_quote={uc.notional_from_tick_lot(min_notional_tick_lot):.12f}"
        )

        mid_ticks = float(state.mid)
        mid_quote = uc.price_from_ticks(int(round(mid_ticks)))
        for spread, qty in action.bids:
            o = uc.ladder_to_venue_order(
                side="buy",
                mid_ticks=mid_ticks,
                spread_ticks=float(spread),
                qty_lots=int(qty),
            )
            if o is None:
                continue
            q = int(o.qty_lots)
            px = int(o.price_ticks)
            amount_base = uc.qty_from_lots(q)
            price_quote = uc.price_from_ticks(px)
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
            print(
                f"[live_execution] place (human) side=buy symbol={sym} "
                f"mid_quote={mid_quote:.12f} price_quote={price_quote:.12f} qty_base={amount_base:.12f}"
            )
            t0_place = time.perf_counter()
            resp, report = place_limit_order_lots_with_report(
                symbol=sym,
                exchange=ex,
                side="buy",
                price_ticks=px,
                qty_lots=q,
                api_token=api_token,
                instrument=place_instrument,
            )
            self._emit_execution_log(
                instrument=place_instrument,
                report=report,
                resp=resp,
                intent_side="buy",
                price_quote=price_quote,
                qty_base=amount_base,
            )
            place_elapsed_ms = (time.perf_counter() - t0_place) * 1000.0
            print(
                f"[live_execution] place response side=buy elapsed_ms={place_elapsed_ms:.1f} "
                f"response={resp}"
            )
            if str(resp.get("status", "")).lower() == "ok":
                avail_quote = max(0.0, avail_quote - notional_quote)
                try:
                    METRICS.counter_inc(
                        "mm_orders_placed_total",
                        labels=metric_labels(extra={"exchange": ex, "symbol": sym, "side": "buy"}),
                        help="Orders placed",
                    )
                except Exception:
                    pass

        for spread, qty in action.asks:
            o = uc.ladder_to_venue_order(
                side="sell",
                mid_ticks=mid_ticks,
                spread_ticks=float(spread),
                qty_lots=int(qty),
            )
            if o is None:
                continue
            q = int(o.qty_lots)
            px = int(o.price_ticks)
            amount_base = uc.qty_from_lots(q)
            price_quote = uc.price_from_ticks(px)
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
            print(
                f"[live_execution] place (human) side=sell symbol={sym} "
                f"mid_quote={mid_quote:.12f} price_quote={price_quote:.12f} qty_base={amount_base:.12f}"
            )
            t0_place = time.perf_counter()
            resp, report = place_limit_order_lots_with_report(
                symbol=sym,
                exchange=ex,
                side="sell",
                price_ticks=px,
                qty_lots=q,
                api_token=api_token,
                instrument=place_instrument,
            )
            self._emit_execution_log(
                instrument=place_instrument,
                report=report,
                resp=resp,
                intent_side="sell",
                price_quote=price_quote,
                qty_base=amount_base,
            )
            place_elapsed_ms = (time.perf_counter() - t0_place) * 1000.0
            print(
                f"[live_execution] place response side=sell elapsed_ms={place_elapsed_ms:.1f} "
                f"response={resp}"
            )
            if str(resp.get("status", "")).lower() == "ok":
                avail_base = max(0.0, avail_base - amount_base)
                try:
                    METRICS.counter_inc(
                        "mm_orders_placed_total",
                        labels=metric_labels(extra={"exchange": ex, "symbol": sym, "side": "sell"}),
                        help="Orders placed",
                    )
                except Exception:
                    pass

    def _emit_execution_log(
        self,
        *,
        instrument: InstrumentId,
        report: ExecutionReport,
        resp: dict,
        intent_side: str,
        price_quote: float,
        qty_base: float,
    ) -> None:
        """Structured JSON log for Grafana/Loki when ``MM_JSON_LOG=1`` (best-effort)."""
        try:
            from mm_adapters.observability.jsonlog import emit as log_emit

            payload = execution_report_for_json(report)
            payload["intent_side"] = intent_side
            payload["intent_price_quote"] = float(price_quote)
            payload["intent_qty_base"] = float(qty_base)
            payload["raw_status"] = resp.get("status")
            log_emit("mm.execution", payload)
        except Exception:
            pass

    def advance(self, t_end_ms: int | None = None) -> bool:
        """
        One iteration: get current step from stepper; apply previous action with current
        step's volumes via execution model (→ inventory, cash, dW, pnl); update
        estimators/testers; optionally compute new action from controller.
        """
        return advance_order_flow_runner(self, t_end_ms=t_end_ms)

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
