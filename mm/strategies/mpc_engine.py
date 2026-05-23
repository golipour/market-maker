"""Synchronous MPC strategy engine; used by ``StrategyRunner`` and unit tests."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from mm.core.order_constraints import (
    OrderConstraints,
    build_orders_from_sigmoid_with_report,
    mid_from_orderbook,
)
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import DecisionDiagnostics, ModelUpdateReport, StrategyDecision, StrategyInputFrame
from mm.core.wallet import inventory_cash_from_snapshot
from mm.models.scoring import refresh_score_params_after, score_frame_models
from mm.mpc.mpc_state import State
from mm.pricing import DEFAULT_VWAP_DEPTH_LOTS, vwap_from_raw_orderbook
from mm.runtime.config import ControllerRuntimeConfig


def _vwap_target_depth(price_model: Any | None) -> float:
    """Depth cap (base units) aligned with ``TorchResidualPriceModel._target_depth_lots``."""

    return float(getattr(price_model, "_target_depth_lots", DEFAULT_VWAP_DEPTH_LOTS))


def _market_flow_step_tuple(frame: StrategyInputFrame) -> tuple[Any, ...]:
    """Tuple shape expected by ``mm.mpc`` market-flow ``update`` (stats-only path)."""

    return (
        frame.primary.t_end_ms,
        None,
        frame.primary.buy_volume,
        frame.primary.sell_volume,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
    )


@dataclass(slots=True)
class MpcStrategyEngine:
    """Synchronous strategy decision engine for legacy MPC internals.

    The bridge keeps the new runtime float-first: mid/prices are quote-per-base,
    quantities are base units, and order commands stay float until execution.
    """

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    _controller: Any = field(init=False, repr=False)
    _market_flow_model: Any = field(init=False, repr=False)
    _price_model: Any = field(init=False, repr=False)
    _hedge_price_model: Any = field(init=False, repr=False)
    _previous_frame: StrategyInputFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = _build_controller(self.config, self.binding)
        self._market_flow_model = getattr(self._controller, "market_flow_model", None)
        self._price_model = getattr(self._controller, "price_model", None)
        self._hedge_price_model = getattr(self._controller, "hedge_price_model", None)

    def decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        t0 = time.monotonic_ns()
        state = _state_from_frame(
            frame,
            self.binding,
            self.config.orderbook_grid.spread_max,
            price_model=self._price_model,
        )
        flow_kind = str((self.config.market_flow_model or {}).get("kind", "hurdle_lognormal"))
        price_kind = str((self.config.price_model or {}).get("kind", "residual"))
        hedge_kind = str((self.config.hedge_price_model or {}).get("kind", "legacy"))
        model_scores = score_frame_models(
            frame=frame,
            previous_frame=self._previous_frame,
            state=state,
            market_flow_model=self._market_flow_model,
            price_model=self._price_model,
            hedge_price_model=self._hedge_price_model,
            market_flow_kind=flow_kind,
            price_kind=price_kind,
            hedge_price_kind=hedge_kind,
        )
        _update_hedge_vwap_price_model(self._hedge_price_model, state)
        if (
            self._hedge_price_model is not None
            and "hedge_price" in model_scores
            and hasattr(self._hedge_price_model, "get_params")
        ):
            refresh_score_params_after(
                model_scores,
                key="hedge_price",
                params_after=self._hedge_price_model.get_params(),
                updated=True,
            )
        _update_market_flow_model(self._market_flow_model, frame)
        if self._market_flow_model is not None and "market_flow" in model_scores and hasattr(
            self._market_flow_model, "get_params"
        ):
            refresh_score_params_after(
                model_scores,
                key="market_flow",
                params_after=self._market_flow_model.get_params(),
                updated=True,
            )
        _update_residual_mid_price_model(self._price_model, self._previous_frame, frame)
        if self._price_model is not None and "price" in model_scores and hasattr(self._price_model, "get_params"):
            refresh_score_params_after(
                model_scores,
                key="price",
                params_after=self._price_model.get_params(),
                updated=self._previous_frame is not None,
            )
        optimizer_stdout = io.StringIO()
        with contextlib.redirect_stdout(optimizer_stdout):
            sigmoid_action = self._controller.optimize(state)
        optimizer_run = getattr(self._controller, "last_diagnostics", None)
        order_constraints = OrderConstraints.from_spec(
            self.binding.primary,
            max_orders_per_side=self.config.max_orders_per_side,
        )
        continuous_depth = _continuous_action_depth_series(
            sigmoid_action,
            mid=float(state.mid),
            orderbook_grid=self.config.orderbook_grid,
        )
        orders, action_pipeline, ladder = build_orders_from_sigmoid_with_report(
            actor_id=self.config.actor_id,
            event=frame.primary,
            sigmoid_action=sigmoid_action,
            orderbook_grid=self.config.orderbook_grid,
            constraints=order_constraints,
            continuous=continuous_depth,
        )
        action = Action(actor_id=self.config.actor_id, orders=tuple(orders))
        order_summaries = tuple(_order_summary(order) for order in action.orders)
        decision_hash = hashlib.sha256(
            json.dumps(order_summaries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        latency_ms = (time.monotonic_ns() - t0) / 1_000_000
        optimizer_log = optimizer_stdout.getvalue().strip()
        if optimizer_run is not None:
            optimizer = optimizer_run.to_payload()
            optimizer["rollout"] = {**dict(optimizer.get("rollout", {})), "log": optimizer_log}
        else:
            optimizer = {
                "kind": (self.config.optimizer or {}).get("kind", "monte_carlo_mpc_tensor_batched"),
                "last_expected_delta_wealth": getattr(self._controller, "_last_E_dW", None),
                "last_components": getattr(self._controller, "_last_J_components", None),
                "log": optimizer_log,
            }
        decision = StrategyDecision(
            actor_id=self.config.actor_id,
            exchange=frame.primary.exchange,
            symbol=frame.primary.symbol,
            t_end_ms=frame.primary.t_end_ms,
            frame_step=frame_step,
            action=action,
            observe_only=self.config.observe_only,
            action_start_after_steps=self.config.action_start_after_steps,
            live_execution_skipped_warmup=(
                not self.config.observe_only and frame_step <= self.config.action_start_after_steps
            ),
            hedge_status=frame.hedge_status,
            has_hedge=frame.has_hedge,
            mid=float(state.mid),
            primary_buy_vwap=float(state.buy_vwap),
            primary_sell_vwap=float(state.sell_vwap),
            hedge_buy_vwap=state.hedge_buy_vwap,
            hedge_sell_vwap=state.hedge_sell_vwap,
            inventory=float(state.inventory),
            cash=float(state.cash),
            buy_volume=float(state.buy_vol),
            sell_volume=float(state.sell_vol),
            latency_ms=float(latency_ms),
            decision_hash=decision_hash,
            orders=order_summaries,
            ladder_orders=tuple(_order_summary(order) for order in ladder),
            action_pipeline_report=action_pipeline,
            model_updates=self._model_update_report(),
            model_scores=model_scores,
            diagnostics=DecisionDiagnostics(
                optimizer=optimizer,
                continuous_action_depth=continuous_depth,
                orderbook_grid=self.config.orderbook_grid.as_contract(),
            ),
        )
        self._previous_frame = frame
        return decision

    def _model_update_report(self) -> ModelUpdateReport:
        return ModelUpdateReport(
            market_flow={
                "kind": (self.config.market_flow_model or {}).get("kind", "hurdle_lognormal"),
                "params": self._market_flow_model.get_params() if hasattr(self._market_flow_model, "get_params") else {},
            },
            price={
                "kind": (self.config.price_model or {}).get("kind", "residual"),
                "params": self._price_model.get_params() if hasattr(self._price_model, "get_params") else {},
            },
            hedge_price={
                "kind": (self.config.hedge_price_model or {}).get("kind", "legacy"),
                "params": (
                    self._hedge_price_model.get_params()
                    if self._hedge_price_model is not None and hasattr(self._hedge_price_model, "get_params")
                    else {}
                ),
                "active_in_mpc": self._hedge_price_model is not None
                and bool(getattr(self._controller, "hedge_enabled", False)),
            },
        )


def _build_controller(config: ControllerRuntimeConfig, binding: SymbolBinding) -> Any:
    from mm.core.models.real_execution import RealExecutionModel
    from mm.mpc.market_order_flow_torch import (
        TorchDiscreteGammaModel,
        TorchHurdleGammaModel,
        TorchHurdleLogNormalModel,
        TorchHurdleWeibullModel,
        TorchWeibullModel,
    )
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
    from mm.mpc.price_torch import TorchResidualPriceModel

    flow_cfg = dict(config.market_flow_model or {})
    flow_kind = str(flow_cfg.pop("kind", "hurdle_lognormal")).lower()
    if flow_kind == "gamma":
        market_flow_model = TorchDiscreteGammaModel(**flow_cfg)
    elif flow_kind == "weibull":
        market_flow_model = TorchWeibullModel(**flow_cfg)
    elif flow_kind == "hurdle_gamma":
        market_flow_model = TorchHurdleGammaModel(**flow_cfg)
    elif flow_kind == "hurdle_weibull":
        market_flow_model = TorchHurdleWeibullModel(**flow_cfg)
    elif flow_kind == "hurdle_lognormal":
        market_flow_model = TorchHurdleLogNormalModel(**flow_cfg)
    else:
        raise ValueError(f"unsupported market_flow_model kind: {flow_kind}")

    price_cfg = dict(config.price_model or {})
    price_kind = str(price_cfg.pop("kind", "residual")).lower()
    if price_kind != "residual":
        raise ValueError(f"unsupported price_model kind: {price_kind}")
    nested_vwap = price_cfg.pop("vwap_depth", None)
    vwap_depth = config.vwap_depth if config.vwap_depth is not None else nested_vwap
    if vwap_depth is not None:
        price_cfg["vwap_depth"] = float(vwap_depth)
    price_model = TorchResidualPriceModel(market_flow_model=market_flow_model, **price_cfg)

    execution_cfg = dict(config.execution_model or {})
    execution_kind = str(execution_cfg.pop("kind", "real")).lower()
    if execution_kind == "observed_step":
        execution_kind = "real"
    if execution_kind != "real":
        raise ValueError(f"unsupported execution_model kind: {execution_kind}")
    execution_model = RealExecutionModel(**execution_cfg)

    optimizer_cfg = dict(config.optimizer or {})
    optimizer_kind = str(optimizer_cfg.pop("kind", "monte_carlo_mpc_tensor_batched")).lower()
    if optimizer_kind != "monte_carlo_mpc_tensor_batched":
        raise ValueError(f"unsupported optimizer kind: {optimizer_kind}")
    grid = config.orderbook_grid
    optimizer_cfg.pop("spread_max", None)
    optimizer_cfg.pop("spread_step", None)
    order_constraints = OrderConstraints.from_spec(
        binding.primary,
        max_orders_per_side=config.max_orders_per_side,
    )
    optimizer_defaults = {
        "n_samples": 8,
        "horizon": 2,
        "adam_epochs": 10,
        "spread_max": grid.spread_max,
        "spread_step": grid.spread_step,
        "quote_size_min_per_side": order_constraints.min_qty,
        "device": "cpu",
    }
    optimizer_defaults.update(optimizer_cfg)
    hedge_price_model = _build_hedge_price_model(config.hedge_price_model)
    return MonteCarloMPCTensorBatched(
        price_model=price_model,
        market_flow_model=market_flow_model,
        execution_model=execution_model,
        hedge_price_model=hedge_price_model,
        **optimizer_defaults,
    )


def _build_hedge_price_model(cfg: dict[str, Any] | None) -> Any:
    from mm.mpc.hedge_price_torch import GaussianHedgeDeltaModel, HedgeGaussianPriceModel

    if not cfg:
        return None
    kind = str(cfg.get("kind", "legacy")).lower().strip()
    if kind in ("none", "null", "disabled"):
        return None
    if kind in ("legacy", "ewma_gaussian", "hedge_gaussian_rep"):
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return HedgeGaussianPriceModel(**payload)
    if kind == "gaussian":
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return GaussianHedgeDeltaModel(**payload)
    raise ValueError(f"unsupported hedge_price_model kind: {kind}")


def _update_hedge_vwap_price_model(model: Any, state: State) -> None:
    """Feed hedge Gaussian / delta models with VWAP refs already on ``State``."""

    if model is None or not hasattr(model, "update"):
        return
    if state.hedge_buy_vwap is None or state.hedge_sell_vwap is None:
        return
    model.update(float(state.hedge_buy_vwap), float(state.hedge_sell_vwap))


def _state_from_frame(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    spread_max: float,
    *,
    price_model: Any | None,
) -> State:
    from mm.mpc.mpc_state import OrderbookRelative

    vwap_depth = _vwap_target_depth(price_model)
    primary_mid = mid_from_orderbook(frame.primary.orderbook)
    bids = tuple(
        (spread, level.qty)
        for level in frame.primary.orderbook.bids
        if (spread := primary_mid - level.price) >= 0 and spread <= spread_max and level.qty > 0
    )
    asks = tuple(
        (spread, level.qty)
        for level in frame.primary.orderbook.asks
        if (spread := level.price - primary_mid) >= 0 and spread <= spread_max and level.qty > 0
    )
    primary_inventory, primary_cash = inventory_cash_from_snapshot(
        frame.primary_private,
        binding.primary.base_asset,
        binding.primary.quote_asset,
        trading_symbol=binding.primary.symbol,
    )
    buy_vwap, sell_vwap = vwap_from_raw_orderbook(frame.primary.orderbook, target_depth=vwap_depth)
    hedge_buy_vwap = hedge_sell_vwap = None
    hedge_inventory = 0.0
    hedge_cash = 0.0
    if frame.hedge is not None:
        try:
            hedge_buy_vwap, hedge_sell_vwap = vwap_from_raw_orderbook(
                frame.hedge.orderbook, target_depth=vwap_depth
            )
        except ValueError:
            hedge_buy_vwap = hedge_sell_vwap = None
    if frame.hedge_private is not None and binding.hedge is not None:
        hedge_inventory, hedge_cash = inventory_cash_from_snapshot(
            frame.hedge_private,
            binding.hedge.base_asset,
            binding.hedge.quote_asset,
            trading_symbol=binding.hedge.symbol,
        )
    return State(
        mid=primary_mid,
        ob_rel=OrderbookRelative(bids=bids, asks=asks),
        buy_vol=frame.primary.buy_volume,
        sell_vol=frame.primary.sell_volume,
        inventory=primary_inventory,
        cash=primary_cash,
        buy_vwap=buy_vwap,
        sell_vwap=sell_vwap,
        hedge_buy_vwap=hedge_buy_vwap,
        hedge_sell_vwap=hedge_sell_vwap,
        hedge_inventory=hedge_inventory,
        hedge_cash=hedge_cash,
    )


def _update_market_flow_model(model: Any, frame: StrategyInputFrame) -> None:
    if not hasattr(model, "update"):
        return
    model.update(None, _market_flow_step_tuple(frame))


def _update_residual_mid_price_model(
    model: Any, previous: StrategyInputFrame | None, current: StrategyInputFrame
) -> None:
    """EWMA on primary **mid** changes (``TorchResidualPriceModel`` side state); not orderbook VWAP."""

    if model is None or previous is None:
        return
    try:
        prev_mid = mid_from_orderbook(previous.primary.orderbook)
        curr_mid = mid_from_orderbook(current.primary.orderbook)
    except ValueError:
        return
    delta = float(curr_mid - prev_mid)
    gamma = float(getattr(model, "_gamma", 0.95))
    gamma = max(1e-6, min(1.0, gamma))
    alpha = 1.0 - gamma
    sigma_min = float(getattr(model, "_sigma_min", 1e-6))
    mu_buy = gamma * float(getattr(model, "_mu_buy", 0.0)) + alpha * delta
    mu_sell = gamma * float(getattr(model, "_mu_sell", 0.0)) + alpha * delta
    sigma2_buy = gamma * float(getattr(model, "_sigma2_buy", 1.0)) + alpha * (delta - mu_buy) ** 2
    sigma2_sell = gamma * float(getattr(model, "_sigma2_sell", 1.0)) + alpha * (delta - mu_sell) ** 2
    model._mu_buy = float(mu_buy)
    model._mu_sell = float(mu_sell)
    model._sigma2_buy = float(max(sigma_min * sigma_min, sigma2_buy))
    model._sigma2_sell = float(max(sigma_min * sigma_min, sigma2_sell))
    model._last_residual = delta
    sigma_buy = math.sqrt(model._sigma2_buy)
    sigma_sell = math.sqrt(model._sigma2_sell)
    if hasattr(model, "mu_buy_hist"):
        model.mu_buy_hist.append(float(mu_buy))
    if hasattr(model, "mu_sell_hist"):
        model.mu_sell_hist.append(float(mu_sell))
    if hasattr(model, "sigma_buy_hist"):
        model.sigma_buy_hist.append(float(sigma_buy))
    if hasattr(model, "sigma_sell_hist"):
        model.sigma_sell_hist.append(float(sigma_sell))
    if hasattr(model, "residual_mu_hist"):
        model.residual_mu_hist.append(float(0.5 * (mu_buy + mu_sell)))
    if hasattr(model, "residual_sigma_hist"):
        model.residual_sigma_hist.append(float(math.sqrt(0.5 * (sigma_buy * sigma_buy + sigma_sell * sigma_sell))))


def _continuous_action_depth_series(
    sigmoid_action: Any,
    *,
    mid: float,
    orderbook_grid: Any,
) -> dict[str, Any]:
    """Effective cumulative depth ``D(spread) - D(0)`` on the same grid as ladder orders."""

    depth_bid = _effective_depth_callable(sigmoid_action, "bid")
    depth_ask = _effective_depth_callable(sigmoid_action, "ask")
    if not callable(depth_bid) or not callable(depth_ask):
        return {"kind": "unknown", "mid": float(mid), "bid": [], "ask": []}
    levels = list(orderbook_grid.spread_ladder())
    mid_f = float(mid)
    bid: list[dict[str, float]] = []
    ask: list[dict[str, float]] = []
    bid_base = _raw_depth_at_zero(sigmoid_action, "bid")
    ask_base = _raw_depth_at_zero(sigmoid_action, "ask")
    for spread in levels:
        s = float(spread)
        try:
            db = float(depth_bid(s))
        except Exception:
            db = float("nan")
        try:
            da = float(depth_ask(s))
        except Exception:
            da = float("nan")
        if math.isfinite(db):
            bid.append({"spread_quote": s, "price": mid_f - s, "cumulative_qty": db})
        if math.isfinite(da):
            ask.append({"spread_quote": s, "price": mid_f + s, "cumulative_qty": da})
    return {
        "kind": "sigmoid_mixture",
        "mid": mid_f,
        "baseline_policy": "cumulative_qty = raw_depth(spread) - raw_depth(0)",
        "bid_raw_depth_at_zero": bid_base,
        "ask_raw_depth_at_zero": ask_base,
        "bid": bid,
        "ask": ask,
    }


def _effective_depth_callable(sigmoid_action: Any, side: str) -> Any:
    name = "effective_depth_bid" if side == "bid" else "effective_depth_ask"
    fn = getattr(sigmoid_action, name, None)
    if callable(fn):
        return fn
    raw_name = "depth_bid" if side == "bid" else "depth_ask"
    raw = getattr(sigmoid_action, raw_name, None)
    if not callable(raw):
        return None
    try:
        base = float(raw(0.0))
    except Exception:
        base = 0.0

    def _depth(spread: float) -> float:
        return max(0.0, float(raw(spread)) - base)

    return _depth


def _raw_depth_at_zero(sigmoid_action: Any, side: str) -> float:
    raw_name = "raw_depth_bid" if side == "bid" else "raw_depth_ask"
    raw = getattr(sigmoid_action, raw_name, None)
    if not callable(raw):
        raw_name = "depth_bid" if side == "bid" else "depth_ask"
        raw = getattr(sigmoid_action, raw_name, None)
    if not callable(raw):
        return 0.0
    try:
        return float(raw(0.0))
    except Exception:
        return 0.0


def order_summary_dict(order: Order) -> dict[str, Any]:
    return {
        "exchange": order.exchange,
        "symbol": order.symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "notional": order.price * order.qty,
        "client_order_id": order.client_order_id,
    }


def _order_summary(order: Order) -> dict[str, Any]:
    return order_summary_dict(order)
