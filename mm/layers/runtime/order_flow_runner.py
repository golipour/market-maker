"""
Order-flow iteration: one place for the main simulation loop.
Advances stepper, applies fills via ExecutionModel (or ExecutionRule), updates estimators/testers, computes actions.

Canonical location for ``OrderFlowRunner`` in this repo.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING

import numpy as np

from mm.mpc.step import Step
from mm.layers.runtime.order_flow_stepper import OrderFlowStepper
from mm.orderflow.testers import MarketOrderFlowTester

if TYPE_CHECKING:
    from mm.core.orderflow_execution import ExecutionModel, ExecutionRule
    from mm.strategies.sigmoid_controller import Controller
    from mm.layers.core.domain.types import ValuationPolicy

from mm.core.orderflow_state import DiscretizedOrderAction, State, ZERO_DISCRETIZED_ACTION, state_from_step
from mm.layers.core.pricing import (
    mid_after_market_order_flow_ticks as mid_ticks_after_market_order_flow,
    resolve_valuation_policy,
    target_depth_lots_from_price_model,
)

from mm.layers.core.app.engine import advance_order_flow_runner
from mm.layers.core.domain.types import InstrumentId
from mm.layers.adapters.diagnostics.emitter import emit as _default_diagnostics_emit
from mm.layers.adapters.observability.context import metric_labels
from mm.layers.adapters.observability.jsonlog import emit as _default_structured_log_emit
from mm.layers.adapters.observability.metrics import REGISTRY as METRICS

class OrderFlowRunner:
    """
    Runs the order-flow loop step by step.
    Each advance(): fetch step from stepper, apply
    ``runner_execution_model.state_after_step(prev_state, step)`` (falls back to ``execution_model``)
    when ``_pending_state`` is set, update estimators/testers, optionally compute a new action.

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
        hedge_enabled: bool = False,
        hedge_price_model: object | None = None,
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
        self.hedge_enabled = bool(hedge_enabled)
        # Anchor rep-price delta model (updated in the runner alongside other models).
        # MPC reads its (mu, sigma) params from the model object (not from State).
        self.hedge_price_model = hedge_price_model
        self.valuation_policy = resolve_valuation_policy(valuation_policy)
        # Cached instrument identity/spec (lazy-filled).
        self._instrument_id = None
        self._instrument_spec = None
        self._live_exec_warned_no_token = False
        self._live_exec_deprecated_warned = False
        self._step_counter = 0
        if execution_model is None and controller is not None and execution_rule is None:
            from mm.core.orderflow_execution import DefaultExecutionRule
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
        self.hedge_buy_reference_hist: list[float] = []
        self.hedge_sell_reference_hist: list[float] = []
        self.hedge_reference_times: list[datetime] = []
        self.hedge_price_param_times: list[datetime] = []
        self.hedge_mu_buy_hist: list[float] = []
        self.hedge_mu_sell_hist: list[float] = []
        self.hedge_sigma_buy_hist: list[float] = []
        self.hedge_sigma_sell_hist: list[float] = []
        self.hedge_portfolio_times: list[datetime] = []
        self.hedge_inventory_hist: list[int] = []
        self.hedge_cash_hist: list[float] = []
        self.main_buy_reference_hist: list[float] = []
        self.main_sell_reference_hist: list[float] = []
        self.main_reference_times: list[datetime] = []
        self.last_hedge_orderbook = None
        self.last_hedge_price_model_params: dict | None = None
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
        self.hedge_reward_opt_hist: list[float] = []
        self.main_reward_init_hist: list[float] = []
        self.hedge_reward_init_hist: list[float] = []
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
        # Canonical-unit wealth series (quote units) computed from mm.layers.core StepSnapshot.
        # This is a migration aid; it does not affect execution or controller behavior.
        self.wealth_quote_times: list[datetime] = []
        self.wealth_quote_hist: list[float] = []
        # Wealth reconstructed as W0 + cumulative sum of realized dW.
        self.wealth_from_dw_times: list[datetime] = []
        self.wealth_from_dw_hist: list[float] = []
        self._wealth_from_dw_baseline: float | None = None
        self._wealth_from_dw_running: float | None = None

        # Observability ports: defaults use mm.layers.adapters; core reads these via duck-typing (no mm.layers.core→adapter import).
        self.structured_log_emit = _default_structured_log_emit
        self.metrics_registry = METRICS
        self.metric_labels = metric_labels
        self.diagnostics_emit = _default_diagnostics_emit

    def _get_instrument_id_and_spec(self) -> tuple[object | None, object | None]:
        """
        Lazily build (InstrumentId, InstrumentSpec) for this runner.

        Kept as `object` types to avoid importing mm.layers.core domain types here;
        engine/assemblers supply concrete types when present.
        """
        if self._instrument_id is not None and self._instrument_spec is not None:
            return self._instrument_id, self._instrument_spec
        ex = str(getattr(self.stepper, "exchange", "") or "").strip()
        sym = str(getattr(self.stepper, "symbol", "") or "").strip()
        if not (ex and sym):
            return None, None
        try:
            from mm.layers.core.domain.types import InstrumentId
            from mm.layers.core.units.converter import instrument_spec_from_trade_stats

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
        """Deprecated: legacy runner no longer performs REST execution.

        Production paths use ``mm.runtime`` with ``OrderManager`` / exchange adapters.
        Simulation and inventory updates still run via the execution model in
        ``advance_order_flow_runner``.
        """
        _ = (action, state)
        if self._live_exec_deprecated_warned:
            return
        self._live_exec_deprecated_warned = True
        ex = str(getattr(self.stepper, "exchange", "") or "").strip() or "unknown"
        sym = str(getattr(self.stepper, "symbol", "") or "").strip() or "unknown"
        try:
            METRICS.counter_inc(
                "mm_live_execution_skipped_total",
                labels=metric_labels(
                    extra={"exchange": ex, "symbol": sym, "reason": "legacy_runner_disabled"}
                ),
                help="Live execution skips by reason",
            )
        except Exception:
            pass

    def advance(self, t_end_ms: int | None = None) -> bool:
        """
        One iteration: get current step from stepper; apply previous action with current
        step's volumes via execution model (→ inventory, cash, dW, pnl); update
        estimators/testers; optionally compute new action from controller.
        """
        return advance_order_flow_runner(self, t_end_ms=t_end_ms)

    def advance_scheduled(self, max_steps_per_call: int = 1) -> int:
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
