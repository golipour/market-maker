"""Synchronous MPC strategy engine; used by ``StrategyRunner`` and unit tests."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from mm.core.order_constraints import (
    OrderConstraints,
    mid_from_orderbook,
)
from mm.core.models.target_book import JointTargetBook
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import DecisionDiagnostics, ModelUpdateReport, StrategyDecision, StrategyInputFrame
from mm.core.wallet import inventory_cash_from_snapshot
from mm.models.scoring import refresh_score_params_after, score_frame_models
from mm.mpc.joint_mpc import enrich_state_for_joint, ob_rel_from_market_event
from mm.mpc.mpc_state import State
from mm.pricing import DEFAULT_VWAP_DEPTH_LOTS, vwap_from_raw_orderbook
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
)
from mm.strategies.target_book_builder import (
    build_joint_target_book_from_sigmoid,
    joint_target_book_to_legacy_action,
    local_constraints_from_spec,
)


def _vwap_target_depth(price_model: Any | None) -> float:
    """Depth cap (base units) aligned with ``TorchResidualPriceModel._target_depth_lots``."""

    return float(getattr(price_model, "_target_depth_lots", DEFAULT_VWAP_DEPTH_LOTS))


def _market_flow_step_tuple(frame: StrategyInputFrame) -> tuple[Any, ...]:
    """Tuple shape expected by ``mm.mpc`` market-flow ``update`` (stats-only path)."""

    return (
        frame.local.t_end_ms,
        None,
        frame.local.buy_volume,
        frame.local.sell_volume,
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
    liquidity_sourcing: LiquiditySourcingRuntimeConfig = field(
        default_factory=LiquiditySourcingRuntimeConfig
    )
    execution: ExecutionRuntimeConfig = field(default_factory=ExecutionRuntimeConfig)
    _controller: Any = field(init=False, repr=False)
    _market_flow_model: Any = field(init=False, repr=False)
    _price_model: Any = field(init=False, repr=False)
    _global_price_model: Any = field(init=False, repr=False)
    _previous_frame: StrategyInputFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = _build_controller(
            self.config,
            self.binding,
            liquidity_sourcing=self.liquidity_sourcing,
        )
        self._market_flow_model = getattr(self._controller, "market_flow_model", None)
        self._price_model = getattr(self._controller, "price_model", None)
        self._global_price_model = getattr(self._controller, "global_price_model", None)

    def decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        t0 = time.monotonic_ns()
        state = _state_from_frame(
            frame,
            self.binding,
            self.config.orderbook_grid.spread_max,
            price_model=self._price_model,
            liquidity_sourcing=self.liquidity_sourcing,
        )
        if self.liquidity_sourcing.enabled:
            state = enrich_state_for_joint(
                state,
                global_active=_global_active_for_frame(frame, self.liquidity_sourcing),
            )
        flow_kind = str((self.config.market_flow_model or {}).get("kind", "hurdle_lognormal"))
        price_kind = str((self.config.price_model or {}).get("kind", "residual"))
        hedge_kind = str((self.config.global_price_model or {}).get("kind", "legacy"))
        model_scores = score_frame_models(
            frame=frame,
            previous_frame=self._previous_frame,
            state=state,
            market_flow_model=self._market_flow_model,
            price_model=self._price_model,
            global_price_model=self._global_price_model,
            market_flow_kind=flow_kind,
            price_kind=price_kind,
            global_price_kind=hedge_kind,
        )
        _update_hedge_vwap_price_model(
            self._global_price_model,
            state,
            previous=self._previous_frame,
            current=frame,
        )
        if (
            self._global_price_model is not None
            and "global_price" in model_scores
            and hasattr(self._global_price_model, "get_params")
        ):
            refresh_score_params_after(
                model_scores,
                key="global_price",
                params_after=self._global_price_model.get_params(),
                updated=True,
            )
        _update_market_flow_model(self._market_flow_model, frame)
        global_flow_model = getattr(self._controller, "global_market_flow_model", None)
        if global_flow_model is not None:
            _update_global_market_flow_model(global_flow_model, frame)
        if self._market_flow_model is not None and "market_flow" in model_scores and hasattr(
            self._market_flow_model, "get_params"
        ):
            refresh_score_params_after(
                model_scores,
                key="market_flow",
                params_after=self._market_flow_model.get_params(),
                updated=True,
            )
        _update_residual_price_model(self._price_model, self._previous_frame, frame)
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
        order_constraints = local_constraints_from_spec(
            self.binding.local,
            max_orders_per_side=self.config.max_orders_per_side,
        )
        continuous_depth = _continuous_action_depth_series(
            sigmoid_action,
            mid=float(state.mid),
            orderbook_grid=self.config.orderbook_grid,
        )
        sourcing_mode = (
            "liquidity_sourcing"
            if self.liquidity_sourcing.enabled and self.liquidity_sourcing.mode == "liquidity_sourcing"
            else "single_venue"
        )
        global_sigmoid = getattr(self._controller, "_last_global_sigmoid_action", None)
        global_event = frame.global_ if sourcing_mode == "liquidity_sourcing" else None
        global_constraints = None
        if global_event is not None and self.binding.global_ is not None:
            global_constraints = OrderConstraints.from_spec(
                self.binding.global_,
                max_orders_per_side=self.config.max_orders_per_side,
            )
        joint_target_book, action_pipeline, ladder, _local_orders = build_joint_target_book_from_sigmoid(
            actor_id=self.config.actor_id,
            local_event=frame.local,
            sigmoid_action=sigmoid_action,
            orderbook_grid=self.config.orderbook_grid,
            local_constraints=order_constraints,
            continuous=continuous_depth,
            mode=sourcing_mode,
            local_policy=self.execution.local.planner,
            global_policy=self.execution.global_.planner,
            global_event=global_event,
            global_constraints=global_constraints,
            global_sigmoid_action=global_sigmoid,
        )
        action = joint_target_book_to_legacy_action(joint_target_book)
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
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            action=action,
            joint_target_book=joint_target_book,
            observe_only=self.config.observe_only,
            action_start_after_steps=self.config.action_start_after_steps,
            live_execution_skipped_warmup=(
                not self.config.observe_only and frame_step <= self.config.action_start_after_steps
            ),
            global_status=frame.global_status,
            has_global=frame.has_global,
            mid=float(state.mid),
            local_buy_vwap=float(state.buy_vwap),
            local_sell_vwap=float(state.sell_vwap),
            global_buy_vwap=state.global_buy_vwap,
            global_sell_vwap=state.global_sell_vwap,
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
            global_price={
                "kind": (self.config.global_price_model or {}).get("kind", "legacy"),
                "params": (
                    self._global_price_model.get_params()
                    if self._global_price_model is not None and hasattr(self._global_price_model, "get_params")
                    else {}
                ),
                "active_in_mpc": self._global_price_model is not None
                and bool(getattr(self._controller, "global_enabled", False)),
            },
        )


def _build_controller(
    config: ControllerRuntimeConfig,
    binding: SymbolBinding,
    *,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig | None = None,
) -> Any:
    from mm.core.models.real_execution import RealExecutionModel
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched

    market_flow_model = _build_market_flow_model(config.market_flow_model)
    price_model = _build_price_model(
        config.price_model,
        market_flow_model=market_flow_model,
        vwap_depth=config.vwap_depth,
    )

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
        binding.local,
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
    reference_global_price_model = _build_global_price_model(config.global_price_model)
    sourcing = liquidity_sourcing or LiquiditySourcingRuntimeConfig()
    global_flow_cfg = (config.optimizer or {}).get("global_market_flow_model") or config.market_flow_model
    global_price_cfg = (config.optimizer or {}).get("global_price_model") or config.global_price_model
    global_flow_model = _build_market_flow_model(global_flow_cfg)
    sourcing_global_price_model = _build_sourcing_global_price_model(
        global_price_cfg,
        market_flow_model=global_flow_model,
        vwap_depth=config.vwap_depth,
    )
    return MonteCarloMPCTensorBatched(
        price_model=price_model,
        market_flow_model=market_flow_model,
        execution_model=execution_model,
        reference_global_price_model=reference_global_price_model,
        liquidity_sourcing_enabled=sourcing.enabled,
        global_market_flow_model=global_flow_model if sourcing.enabled else None,
        global_price_model=sourcing_global_price_model if sourcing.enabled else None,
        **optimizer_defaults,
    )


def _build_market_flow_model(cfg: dict[str, Any] | None) -> Any:
    from mm.mpc.market_order_flow_torch import (
        TorchDiscreteGammaModel,
        TorchHurdleGammaModel,
        TorchHurdleLogNormalModel,
        TorchHurdleWeibullModel,
        TorchWeibullModel,
    )

    flow_cfg = dict(cfg or {})
    flow_kind = str(flow_cfg.pop("kind", "hurdle_lognormal")).lower()
    if flow_kind == "gamma":
        return TorchDiscreteGammaModel(**flow_cfg)
    if flow_kind == "weibull":
        return TorchWeibullModel(**flow_cfg)
    if flow_kind == "hurdle_gamma":
        return TorchHurdleGammaModel(**flow_cfg)
    if flow_kind == "hurdle_weibull":
        return TorchHurdleWeibullModel(**flow_cfg)
    if flow_kind == "hurdle_lognormal":
        return TorchHurdleLogNormalModel(**flow_cfg)
    raise ValueError(f"unsupported market_flow_model kind: {flow_kind}")


def _build_price_model(
    cfg: dict[str, Any] | None,
    *,
    market_flow_model: Any,
    vwap_depth: float | None,
) -> Any:
    from mm.mpc.price_torch import TorchResidualPriceModel

    price_cfg = dict(cfg or {})
    price_kind = str(price_cfg.pop("kind", "residual")).lower()
    if price_kind != "residual":
        raise ValueError(f"unsupported price_model kind: {price_kind}")
    nested_vwap = price_cfg.pop("vwap_depth", None)
    resolved_vwap = vwap_depth if vwap_depth is not None else nested_vwap
    if resolved_vwap is not None:
        price_cfg["vwap_depth"] = float(resolved_vwap)
    return TorchResidualPriceModel(market_flow_model=market_flow_model, **price_cfg)


def _build_sourcing_global_price_model(
    cfg: dict[str, Any] | None,
    *,
    market_flow_model: Any,
    vwap_depth: float | None,
) -> Any | None:
    """Build the global price model used for liquidity-sourcing MPC and diagnostics."""

    if not cfg:
        return None
    kind = str(cfg.get("kind", "legacy")).lower().strip()
    if kind in ("none", "null", "disabled"):
        return None
    if kind == "residual":
        return _build_price_model(cfg, market_flow_model=market_flow_model, vwap_depth=vwap_depth)
    return _build_global_price_model(cfg)


def _global_active_for_frame(
    frame: StrategyInputFrame,
    sourcing: LiquiditySourcingRuntimeConfig,
) -> bool:
    if not sourcing.enabled:
        return False
    if sourcing.mode != "liquidity_sourcing":
        return False
    return frame.global_ is not None and frame.global_status in (None, "ready", "stale")


def _update_global_market_flow_model(model: Any, frame: StrategyInputFrame) -> None:
    if frame.global_ is None or not hasattr(model, "update"):
        return
    model.update(None, _global_market_flow_step_tuple(frame))


def _global_market_flow_step_tuple(frame: StrategyInputFrame) -> tuple[Any, ...]:
    hedge = frame.global_
    assert hedge is not None
    return (
        hedge.t_end_ms,
        None,
        hedge.buy_volume,
        hedge.sell_volume,
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


def _build_global_price_model(cfg: dict[str, Any] | None) -> Any:
    from mm.mpc.global_price_torch import GaussianGlobalDeltaModel, GlobalGaussianPriceModel

    if not cfg:
        return None
    kind = str(cfg.get("kind", "legacy")).lower().strip()
    if kind in ("none", "null", "disabled"):
        return None
    if kind in ("legacy", "ewma_gaussian", "hedge_gaussian_rep"):
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return GlobalGaussianPriceModel(**payload)
    if kind == "gaussian":
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return GaussianGlobalDeltaModel(**payload)
    raise ValueError(f"unsupported global_price_model kind: {kind}")


def _update_hedge_vwap_price_model(
    model: Any,
    state: State,
    *,
    previous: StrategyInputFrame | None = None,
    current: StrategyInputFrame | None = None,
) -> None:
    """Update the configured global price model from frame VWAP observations."""

    if model is None or not hasattr(model, "update"):
        return
    if hasattr(model, "sample_deltas"):
        if state.global_buy_vwap is None or state.global_sell_vwap is None:
            return
        model.update(float(state.global_buy_vwap), float(state.global_sell_vwap))
        return
    if previous is not None and current is not None:
        _update_global_residual_price_model(model, previous, current)


def _update_global_residual_price_model(
    model: Any,
    previous: StrategyInputFrame,
    current: StrategyInputFrame,
) -> None:
    """EWMA on global VWAP residuals when the global model is ``TorchResidualPriceModel``."""

    if previous.global_ is None or current.global_ is None:
        return
    if not hasattr(model, "update"):
        return
    prev_step = _price_model_step(previous.global_)
    curr_step = _price_model_step(current.global_)
    try:
        model.update(prev_step, curr_step)
    except (ValueError, ZeroDivisionError, TypeError):
        return


def _state_from_frame(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    spread_max: float,
    *,
    price_model: Any | None,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig | None = None,
) -> State:
    from mm.mpc.mpc_state import OrderbookRelative

    sourcing = liquidity_sourcing or LiquiditySourcingRuntimeConfig()
    vwap_depth = _vwap_target_depth(price_model)
    primary_mid = mid_from_orderbook(frame.local.orderbook)
    bids = tuple(
        (spread, level.qty)
        for level in frame.local.orderbook.bids
        if (spread := primary_mid - level.price) >= 0 and spread <= spread_max and level.qty > 0
    )
    asks = tuple(
        (spread, level.qty)
        for level in frame.local.orderbook.asks
        if (spread := level.price - primary_mid) >= 0 and spread <= spread_max and level.qty > 0
    )
    local_inventory, primary_cash = inventory_cash_from_snapshot(
        frame.local_private,
        binding.local.base_asset,
        binding.local.quote_asset,
        trading_symbol=binding.local.symbol,
    )
    buy_vwap, sell_vwap = vwap_from_raw_orderbook(frame.local.orderbook, target_depth=vwap_depth)
    global_buy_vwap = global_sell_vwap = None
    global_inventory = 0.0
    hedge_cash = 0.0
    global_ob_rel = None
    global_buy_vol = 0.0
    global_sell_vol = 0.0
    global_inventory = 0.0
    global_cash = 0.0
    global_buy_vwap = 0.0
    global_sell_vwap = 0.0
    if frame.global_ is not None:
        global_ob_rel = ob_rel_from_market_event(frame.global_.orderbook, spread_max=spread_max)
        global_buy_vol = float(frame.global_.buy_volume)
        global_sell_vol = float(frame.global_.sell_volume)
        try:
            global_buy_vwap, global_sell_vwap = vwap_from_raw_orderbook(
                frame.global_.orderbook, target_depth=vwap_depth
            )
            global_buy_vwap = float(global_buy_vwap)
            global_sell_vwap = float(global_sell_vwap)
        except ValueError:
            global_buy_vwap = global_sell_vwap = None
    if frame.global_private is not None and binding.global_ is not None:
        global_inventory, hedge_cash = inventory_cash_from_snapshot(
            frame.global_private,
            binding.global_.base_asset,
            binding.global_.quote_asset,
            trading_symbol=binding.global_.symbol,
        )
        global_inventory = global_inventory
        global_cash = hedge_cash
    reference_price = float(global_sell_vwap or sell_vwap)
    global_active = _global_active_for_frame(frame, sourcing)
    return State(
        mid=primary_mid,
        ob_rel=OrderbookRelative(bids=bids, asks=asks),
        buy_vol=frame.local.buy_volume,
        sell_vol=frame.local.sell_volume,
        inventory=local_inventory,
        cash=primary_cash,
        buy_vwap=buy_vwap,
        sell_vwap=sell_vwap,
        global_buy_vwap=global_buy_vwap,
        global_sell_vwap=global_sell_vwap,
        global_inventory=global_inventory,
        global_cash=global_cash,
        global_ob_rel=global_ob_rel,
        global_buy_vol=global_buy_vol,
        global_sell_vol=global_sell_vol,
        reference_price=reference_price,
        global_active=global_active,
    )


def _update_market_flow_model(model: Any, frame: StrategyInputFrame) -> None:
    if not hasattr(model, "update"):
        return
    model.update(None, _market_flow_step_tuple(frame))


def _update_residual_price_model(
    model: Any, previous: StrategyInputFrame | None, current: StrategyInputFrame
) -> None:
    """EWMA on true primary VWAP residuals after market-flow-only book evolution."""

    if model is None or previous is None:
        return
    if not hasattr(model, "update"):
        return
    prev_step = _price_model_step(previous.local)
    curr_step = _price_model_step(current.local)
    try:
        model.update(prev_step, curr_step)
    except (ValueError, ZeroDivisionError, TypeError):
        return


def _price_model_step(event: MarketEvent) -> tuple[Any, ...]:
    """Tuple shape expected by ``TorchResidualPriceModel.update``."""

    return (
        event.t_end_ms,
        _orderbook_with_mid(event),
        event.buy_volume,
        event.sell_volume,
        0.0,
        0.0,
        0.0,
        (),
        (),
        None,
        None,
        None,
        None,
        0.0,
        0.0,
    )


def _orderbook_with_mid(event: MarketEvent) -> Any:
    return SimpleNamespace(
        bids=event.orderbook.bids,
        asks=event.orderbook.asks,
        mid=mid_from_orderbook(event.orderbook),
    )


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
