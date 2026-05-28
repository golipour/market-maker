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
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import DecisionDiagnostics, ModelUpdateReport, StrategyDecision, StrategyInputFrame
from mm.core.wallet import inventory_cash_from_snapshot, inventory_cash_from_venue_ledger
from mm.core.adapters.joint import joint_target_book_to_action
from mm.models.scoring import refresh_score_params_after, score_frame_models
from mm.mpc.joint_mpc import enrich_state_for_joint, ob_rel_from_market_event
from mm.mpc.mpc_state import State
from mm.pricing import vwap_from_raw_orderbook
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
)
from mm.strategies.target_book_builder import (
    build_joint_target_book_from_sigmoid,
    local_constraints_from_spec,
)


_VENUE_OPTIMIZER_KEYS = {
    "return_risk_aversion_beta",
    "objective_weight",
    "inventory_target_penalty_weight",
    "target_inventory",
    "target_inventory_multiplier",
    "quote_size_min_per_side",
    "quote_size_max_per_side",
    "quote_spread_max",
    "quote_spread_min_depth",
    "quote_spread_k_epsilon",
    "soft_fill_temp",
    "soft_fill_activation",
    "soft_fill_queue_penalty",
}
_REMOVED_VENUE_OPTIMIZER_ALIASES = {
    "inv_coef": "inventory_target_penalty_weight",
    "target_inv": "target_inventory",
}
_FLAT_VENUE_OPTIMIZER_KEYS = _VENUE_OPTIMIZER_KEYS | set(_REMOVED_VENUE_OPTIMIZER_ALIASES)


def _venue_optimizer_config(payload: Any, *, venue: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"controller.optimizer.{venue} must be an object")
    data = dict(payload)
    removed = sorted(key for key in _REMOVED_VENUE_OPTIMIZER_ALIASES if key in data)
    if removed:
        replacements = ", ".join(
            f"{key}->{_REMOVED_VENUE_OPTIMIZER_ALIASES[key]}" for key in removed
        )
        raise ValueError(
            f"removed venue optimizer aliases under controller.optimizer.{venue}: {replacements}"
        )
    unknown = sorted(key for key in data if key not in _VENUE_OPTIMIZER_KEYS)
    if unknown:
        raise ValueError(
            f"unknown controller.optimizer.{venue} keys: {', '.join(unknown)}"
        )
    return data


def _liquidity_sourcing_portfolio_enabled(sourcing: LiquiditySourcingRuntimeConfig) -> bool:
    return bool(sourcing.enabled and sourcing.mode == "liquidity_sourcing")


def _inventory_vwap_depth(
    local_inventory: float,
    global_inventory: float = 0.0,
    *,
    liquidity_sourcing_enabled: bool,
) -> float:
    """VWAP depth is the absolute inventory whose mark-to-market is being optimized."""

    total = float(local_inventory)
    if liquidity_sourcing_enabled:
        total += float(global_inventory)
    if not math.isfinite(total):
        return 0.0
    return max(0.0, abs(total))


def _set_model_vwap_depth(model: Any, vwap_depth: float | None) -> None:
    if model is None or vwap_depth is None or not hasattr(model, "_target_depth_lots"):
        return
    try:
        depth = float(vwap_depth)
    except (TypeError, ValueError):
        return
    if not math.isfinite(depth):
        return
    setattr(model, "_target_depth_lots", max(0.0, depth))


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
    """Synchronous strategy decision engine for MPC controllers.

    The runtime stays float-first: mid/prices are quote-per-base, quantities are
    base units, and order commands stay float until execution.
    """

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    liquidity_sourcing: LiquiditySourcingRuntimeConfig = field(
        default_factory=LiquiditySourcingRuntimeConfig
    )
    execution: ExecutionRuntimeConfig = field(default_factory=ExecutionRuntimeConfig)
    _controller: Any = field(init=False, repr=False)
    _local_market_flow_model: Any = field(init=False, repr=False)
    _local_price_model: Any = field(init=False, repr=False)
    _global_market_flow_model: Any = field(init=False, repr=False)
    _global_price_model: Any = field(init=False, repr=False)
    _previous_frame: StrategyInputFrame | None = field(default=None, init=False, repr=False)
    _ledger_context: JointLedgerSnapshot | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = _build_controller(
            self.config,
            self.binding,
            liquidity_sourcing=self.liquidity_sourcing,
        )
        self._local_market_flow_model = getattr(self._controller, "local_market_flow_model", None)
        self._local_price_model = getattr(self._controller, "local_price_model", None)
        self._global_market_flow_model = getattr(self._controller, "global_market_flow_model", None)
        self._global_price_model = getattr(self._controller, "global_price_model", None)

    def set_ledger_context(self, ledger: JointLedgerSnapshot | None) -> None:
        self._ledger_context = ledger

    def decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        t0 = time.monotonic_ns()
        state = _state_from_frame(
            frame,
            self.binding,
            self.config.orderbook_grid.spread_max,
            reference_price_source=self.config.reference_price_source,
            liquidity_sourcing=self.liquidity_sourcing,
            ledger=self._ledger_context,
        )
        if self.liquidity_sourcing.enabled:
            state = enrich_state_for_joint(
                state,
                global_active=_global_active_for_frame(frame, self.liquidity_sourcing),
            )
        local_flow_kind = str((self.config.local_market_flow_model or {}).get("kind", "hurdle_lognormal"))
        local_price_kind = str((self.config.local_price_model or {}).get("kind", "residual"))
        global_flow_kind = str((self.config.global_market_flow_model or {}).get("kind", "hurdle_lognormal"))
        global_price_kind = str((self.config.global_price_model or {}).get("kind", "ewma_gaussian"))
        model_scores = score_frame_models(
            frame=frame,
            previous_frame=self._previous_frame,
            state=state,
            local_market_flow_model=self._local_market_flow_model,
            local_price_model=self._local_price_model,
            global_market_flow_model=self._global_market_flow_model,
            global_price_model=self._global_price_model,
            local_market_flow_kind=local_flow_kind,
            local_price_kind=local_price_kind,
            global_market_flow_kind=global_flow_kind,
            global_price_kind=global_price_kind,
        )
        _update_global_vwap_price_model(
            self._global_price_model,
            state,
            previous=self._previous_frame,
            current=frame,
            vwap_depth=state.vwap_depth,
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
        _update_market_flow_model(self._local_market_flow_model, frame)
        if self._global_market_flow_model is not None:
            _update_global_market_flow_model(self._global_market_flow_model, frame)
        if (
            self._local_market_flow_model is not None
            and "local_market_flow" in model_scores
            and hasattr(self._local_market_flow_model, "get_params")
        ):
            refresh_score_params_after(
                model_scores,
                key="local_market_flow",
                params_after=self._local_market_flow_model.get_params(),
                updated=True,
            )
        if (
            self._global_market_flow_model is not None
            and "global_market_flow" in model_scores
            and hasattr(self._global_market_flow_model, "get_params")
        ):
            refresh_score_params_after(
                model_scores,
                key="global_market_flow",
                params_after=self._global_market_flow_model.get_params(),
                updated=frame.has_global,
            )
        _update_residual_price_model(
            self._local_price_model,
            self._previous_frame,
            frame,
            state.vwap_depth,
        )
        if (
            self._local_price_model is not None
            and "local_price" in model_scores
            and hasattr(self._local_price_model, "get_params")
        ):
            refresh_score_params_after(
                model_scores,
                key="local_price",
                params_after=self._local_price_model.get_params(),
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
        action = joint_target_book_to_action(joint_target_book)
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
            local_market_flow={
                "kind": (self.config.local_market_flow_model or {}).get("kind", "hurdle_lognormal"),
                "params": (
                    self._local_market_flow_model.get_params()
                    if hasattr(self._local_market_flow_model, "get_params")
                    else {}
                ),
            },
            local_price={
                "kind": (self.config.local_price_model or {}).get("kind", "residual"),
                "params": (
                    self._local_price_model.get_params()
                    if hasattr(self._local_price_model, "get_params")
                    else {}
                ),
            },
            global_market_flow={
                "kind": (self.config.global_market_flow_model or {}).get("kind", "hurdle_lognormal"),
                "params": (
                    self._global_market_flow_model.get_params()
                    if self._global_market_flow_model is not None
                    and hasattr(self._global_market_flow_model, "get_params")
                    else {}
                ),
                "active_in_mpc": bool(
                    self.binding.global_ is not None
                    and (
                        self.liquidity_sourcing.enabled
                        or self.config.reference_price_source == "global"
                    )
                    and self._global_market_flow_model is not None
                ),
            },
            global_price={
                "kind": (self.config.global_price_model or {}).get("kind", "ewma_gaussian"),
                "params": (
                    self._global_price_model.get_params()
                    if self._global_price_model is not None and hasattr(self._global_price_model, "get_params")
                    else {}
                ),
                "active_in_mpc": bool(
                    self.binding.global_ is not None
                    and (
                        self.liquidity_sourcing.enabled
                        or self.config.reference_price_source == "global"
                    )
                    and self._global_price_model is not None
                ),
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

    local_market_flow_model = _build_market_flow_model(
        config.local_market_flow_model,
        config_name="local_market_flow_model",
    )
    local_price_model = _build_price_model(
        config.local_price_model,
        market_flow_model=local_market_flow_model,
    )

    execution_cfg = dict(config.execution_model or {})
    execution_kind = str(execution_cfg.pop("kind", "real")).lower()
    if execution_kind == "observed_step":
        execution_kind = "real"
    if execution_kind != "real":
        raise ValueError(f"unsupported execution_model kind: {execution_kind}")
    execution_model = RealExecutionModel(**execution_cfg)

    optimizer_cfg = dict(config.optimizer or {})
    nested_global_model_keys = [
        key for key in ("global_market_flow_model", "global_price_model") if key in optimizer_cfg
    ]
    if nested_global_model_keys:
        raise ValueError(
            "global model config belongs directly under controller: "
            f"{', '.join(nested_global_model_keys)}"
        )
    flat_venue_optimizer_keys = sorted(key for key in _FLAT_VENUE_OPTIMIZER_KEYS if key in optimizer_cfg)
    if flat_venue_optimizer_keys:
        raise ValueError(
            "venue optimizer config belongs under controller.optimizer.local/global: "
            f"{', '.join(flat_venue_optimizer_keys)}"
        )
    local_optimizer_cfg = _venue_optimizer_config(optimizer_cfg.pop("local", None), venue="local")
    global_optimizer_cfg = _venue_optimizer_config(optimizer_cfg.pop("global", None), venue="global")
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
    global_constraints = (
        OrderConstraints.from_spec(binding.global_, max_orders_per_side=config.max_orders_per_side)
        if binding.global_ is not None
        else order_constraints
    )
    local_min_qty = float(order_constraints.min_qty or 1.0)
    global_min_qty = float(global_constraints.min_qty or local_min_qty)
    local_optimizer_defaults = {
        "quote_size_min_per_side": local_min_qty,
        "quote_size_max_per_side": max(local_min_qty, 10.0),
    }
    global_optimizer_defaults = {
        "quote_size_min_per_side": global_min_qty,
        "quote_size_max_per_side": max(global_min_qty, 10.0),
    }
    local_optimizer_defaults.update(local_optimizer_cfg)
    global_optimizer_defaults.update(global_optimizer_cfg)
    optimizer_defaults = {
        "n_samples": 8,
        "horizon": 2,
        "adam_epochs": 10,
        "spread_max": grid.spread_max,
        "spread_step": grid.spread_step,
        "device": "cpu",
        "reference_price_source": config.reference_price_source,
        "local_optimizer": local_optimizer_defaults,
        "global_optimizer": global_optimizer_defaults,
    }
    optimizer_defaults.update(optimizer_cfg)
    sourcing = liquidity_sourcing or LiquiditySourcingRuntimeConfig()
    global_models_enabled = bool(
        binding.global_ is not None
        and (sourcing.enabled or config.reference_price_source == "global")
    )
    global_market_flow_model = _build_market_flow_model(
        config.global_market_flow_model,
        config_name="global_market_flow_model",
    )
    sourcing_global_price_model = _build_sourcing_global_price_model(
        config.global_price_model,
        market_flow_model=global_market_flow_model,
    )
    return MonteCarloMPCTensorBatched(
        local_price_model=local_price_model,
        local_market_flow_model=local_market_flow_model,
        execution_model=execution_model,
        liquidity_sourcing_enabled=sourcing.enabled,
        global_market_flow_model=global_market_flow_model if global_models_enabled else None,
        global_price_model=sourcing_global_price_model if global_models_enabled else None,
        **optimizer_defaults,
    )


def _build_market_flow_model(cfg: dict[str, Any] | None, *, config_name: str) -> Any:
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
    raise ValueError(f"unsupported {config_name} kind: {flow_kind}")


def _build_price_model(
    cfg: dict[str, Any] | None,
    *,
    market_flow_model: Any,
    config_name: str = "local_price_model",
) -> Any:
    from mm.mpc.price_torch import TorchResidualPriceModel

    price_cfg = dict(cfg or {})
    price_kind = str(price_cfg.pop("kind", "residual")).lower()
    if price_kind != "residual":
        raise ValueError(f"unsupported {config_name} kind: {price_kind}")
    if "vwap_depth" in price_cfg:
        raise ValueError(
            f"{config_name}.vwap_depth was removed; VWAP depth is derived from inventory"
        )
    return TorchResidualPriceModel(market_flow_model=market_flow_model, **price_cfg)


def _build_sourcing_global_price_model(
    cfg: dict[str, Any] | None,
    *,
    market_flow_model: Any,
) -> Any | None:
    """Build the global price model used for liquidity-sourcing MPC and diagnostics."""

    if not cfg:
        return None
    kind = str(cfg.get("kind", "ewma_gaussian")).lower().strip()
    if kind in ("none", "null", "disabled"):
        return None
    if kind == "residual":
        return _build_price_model(
            cfg,
            market_flow_model=market_flow_model,
            config_name="global_price_model",
        )
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
    global_event = frame.global_
    assert global_event is not None
    return (
        global_event.t_end_ms,
        None,
        global_event.buy_volume,
        global_event.sell_volume,
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
    kind = str(cfg.get("kind", "ewma_gaussian")).lower().strip()
    if kind in ("none", "null", "disabled"):
        return None
    if kind == "ewma_gaussian":
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return GlobalGaussianPriceModel(**payload)
    if kind == "gaussian":
        payload = {k: v for k, v in cfg.items() if k != "kind"}
        return GaussianGlobalDeltaModel(**payload)
    raise ValueError(f"unsupported global_price_model kind: {kind}")


def _update_global_vwap_price_model(
    model: Any,
    state: State,
    *,
    previous: StrategyInputFrame | None = None,
    current: StrategyInputFrame | None = None,
    vwap_depth: float | None = None,
) -> None:
    """Update the configured global price model from frame VWAP observations."""

    if model is None or not hasattr(model, "update"):
        return
    _set_model_vwap_depth(model, vwap_depth)
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
    reference_price_source: str = "global",
    liquidity_sourcing: LiquiditySourcingRuntimeConfig | None = None,
    ledger: JointLedgerSnapshot | None = None,
) -> State:
    from mm.mpc.mpc_state import OrderbookRelative

    sourcing = liquidity_sourcing or LiquiditySourcingRuntimeConfig()
    global_portfolio_enabled = _liquidity_sourcing_portfolio_enabled(sourcing)
    local_inventory, local_cash = _venue_inventory_cash(
        ledger.local if ledger is not None else None,
        frame.local_private,
        binding.local,
    )
    global_inventory = 0.0
    global_cash = 0.0
    if global_portfolio_enabled and binding.global_ is not None:
        global_ledger = ledger.global_ if ledger is not None else None
        global_inventory, global_cash = _venue_inventory_cash(
            global_ledger,
            frame.global_private,
            binding.global_,
        )
    vwap_depth = _inventory_vwap_depth(
        local_inventory,
        global_inventory,
        liquidity_sourcing_enabled=global_portfolio_enabled,
    )
    local_mid = mid_from_orderbook(frame.local.orderbook)
    bids = tuple(
        (spread, level.qty)
        for level in frame.local.orderbook.bids
        if (spread := local_mid - level.price) >= 0 and spread <= spread_max and level.qty > 0
    )
    asks = tuple(
        (spread, level.qty)
        for level in frame.local.orderbook.asks
        if (spread := level.price - local_mid) >= 0 and spread <= spread_max and level.qty > 0
    )
    buy_vwap, sell_vwap = vwap_from_raw_orderbook(frame.local.orderbook, target_depth=vwap_depth)
    global_buy_vwap = global_sell_vwap = None
    global_ob_rel = None
    global_buy_vol = 0.0
    global_sell_vol = 0.0
    global_mid = None
    if frame.global_ is not None:
        global_mid = mid_from_orderbook(frame.global_.orderbook)
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
    use_global_reference = (
        str(reference_price_source).strip().lower() == "global"
        and global_sell_vwap is not None
        and float(global_sell_vwap) > 0.0
    )
    reference_price = float(global_sell_vwap if use_global_reference else sell_vwap)
    global_active = _global_active_for_frame(frame, sourcing)
    return State(
        mid=local_mid,
        ob_rel=OrderbookRelative(bids=bids, asks=asks),
        buy_vol=frame.local.buy_volume,
        sell_vol=frame.local.sell_volume,
        inventory=local_inventory,
        cash=local_cash,
        buy_vwap=buy_vwap,
        sell_vwap=sell_vwap,
        vwap_depth=vwap_depth,
        global_buy_vwap=global_buy_vwap,
        global_sell_vwap=global_sell_vwap,
        global_mid=global_mid,
        global_inventory=global_inventory,
        global_cash=global_cash,
        global_ob_rel=global_ob_rel,
        global_buy_vol=global_buy_vol,
        global_sell_vol=global_sell_vol,
        reference_price=reference_price,
        global_active=global_active,
    )


def _venue_inventory_cash(
    ledger: VenueLedger | None,
    snapshot: PrivateStateSnapshot | None,
    spec,
) -> tuple[float, float]:
    if ledger is not None and ledger.last_snapshot_ms > 0:
        return inventory_cash_from_venue_ledger(ledger)
    return inventory_cash_from_snapshot(
        snapshot,
        spec.base_asset,
        spec.quote_asset,
        trading_symbol=spec.symbol,
    )


def _update_market_flow_model(model: Any, frame: StrategyInputFrame) -> None:
    if not hasattr(model, "update"):
        return
    model.update(None, _market_flow_step_tuple(frame))


def _update_residual_price_model(
    model: Any,
    previous: StrategyInputFrame | None,
    current: StrategyInputFrame,
    vwap_depth: float | None = None,
) -> None:
    """EWMA on true local VWAP residuals after market-flow-only book evolution."""

    if model is None or previous is None:
        return
    if not hasattr(model, "update"):
        return
    _set_model_vwap_depth(model, vwap_depth)
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
