"""
Order-flow pipeline: build models, testers, MPC controller, and run headless.

Torch MOF/residual models and MPC live under ``mm.*``; runner/stepper remain in ``mm.layers.runtime``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm.layers.core.domain.types import InstrumentId, ValuationPolicy
from mm.layers.core.hedge.policy import HedgePolicy
from mm.mpc.market_order_flow_torch import TorchHurdleWeibullModel
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.price_torch import TorchResidualPriceModel
from mm.strategies.sigmoid_controller import SigmoidController
from mm.layers.runtime.wiring import create_stepper
from mm.layers.runtime.order_flow_runner import OrderFlowRunner
from mm.orderflow.testers import (
    GeneralOrderFlowLogLossTester,
    MarketOrderFlowTester,
    PriceLogLossTester,
)

if TYPE_CHECKING:
    from mm.core.orderflow_execution import ExecutionModel, ExecutionRule
    from mm.strategies.sigmoid_controller import Controller

PipelineConfig = dict


def _get_init_params(cls: type) -> set:
    import inspect

    sig = inspect.signature(cls.__init__)
    return set(sig.parameters.keys()) - {"self"}


def build_pipeline(
    config: PipelineConfig,
    *,
    decay: float = 0.99,
    lot_size: int = 1,
    exchange: str | None = None,
    symbol: str | None = None,
    hedge_exchange: str | None = None,
    hedge_symbol: str | None = None,
) -> tuple[
    TorchHurdleWeibullModel,
    MarketOrderFlowTester,
    TorchResidualPriceModel,
    object,
    "Controller | None",
    "ExecutionRule | None",
    "ExecutionModel | None",
    "ExecutionModel | None",
    object | None,
    object | None,
]:
    """
    Build pipeline: TorchHurdleWeibullModel, TorchResidualPriceModel (flow-implied + residual),
    testers, MonteCarloMPCTensorBatched + SigmoidController.
    """
    from mm.core.orderflow_execution import DefaultExecutionRule, ObservedStepExecutionModel
    from mm.layers.runtime.order_flow_stepper import HedgeGaussianPriceModel

    mpc_kw = config["mpc_kwargs"]
    spread_max = mpc_kw["spread_max"]
    spread_int = max(1, int(spread_max))

    mof_kw = config["market_order_flow_model_kwargs"]
    market_order_flow_model = TorchHurdleWeibullModel(**{k: v for k, v in mof_kw.items() if k in _get_init_params(TorchHurdleWeibullModel)})
    tester = GeneralOrderFlowLogLossTester(market_order_flow_model, decay=decay)

    p_kw = config["price_model_kwargs"]
    price_model = TorchResidualPriceModel(
        market_flow_model=market_order_flow_model,
        **{k: v for k, v in p_kw.items() if k in _get_init_params(TorchResidualPriceModel)},
    )
    price_tester = PriceLogLossTester(price_model, decay=decay)

    exec_kw = config["execution_model_kwargs"]
    # Simplified: keep only observed execution model (live/replay balances from stepper).
    # maker_fee / hedge_fee are read by MPC from the execution model when present;
    # hedge_fee also remains on MonteCarloMPCTensorBatched ctor for backward compatibility.
    if "hedge_fee" in exec_kw:
        em_hedge_fee = float(exec_kw["hedge_fee"])
    elif "hedge_fee" in mpc_kw:
        em_hedge_fee = float(mpc_kw["hedge_fee"])
    else:
        em_hedge_fee = None
    execution_model = ObservedStepExecutionModel(
        maker_fee=float(exec_kw.get("maker_fee", 0.0)),
        hedge_fee=em_hedge_fee,
    )
    runner_execution_model = execution_model
    hedge_price_model = HedgeGaussianPriceModel(
        decay=float(config.get("hedge_price_model_decay", 0.97) or 0.97),
        sigma_min_ticks=float(config.get("hedge_price_model_sigma_min_ticks", 1.0) or 1.0),
    )
    _ax = (
        str(hedge_exchange).strip()
        if hedge_exchange is not None
        else str(config.get("hedge_exchange") or "").strip()
    )
    _asym = (
        str(hedge_symbol).strip()
        if hedge_symbol is not None
        else str(config.get("hedge_symbol") or "").strip()
    )
    _hedge = HedgePolicy.from_mpc_and_instruments(
        mpc_kw,
        primary=InstrumentId(
            exchange=str(exchange or "").strip(), symbol=str(symbol or "").strip()
        ),
        hedge_exchange=_ax or None,
        hedge_symbol=_asym or None,
    )
    mc_mpc_kw = {
        "n_components": mpc_kw["n_components"],
        "n_samples": mpc_kw["n_samples"],
        "horizon": mpc_kw["horizon"],
        "spread_max": spread_max,
        "n_grid": spread_int,
        "ob_levels": spread_int,
        "adam_lr": mpc_kw["adam_lr"],
        "adam_epochs": mpc_kw["adam_epochs"],
        "adam_grad_clip": mpc_kw["adam_grad_clip"],
        "device": mpc_kw["device"],
        "warm_start": mpc_kw["warm_start"],
        "total_q_min": mpc_kw["total_q_min"],
        "total_q_max": mpc_kw["total_q_max"],
        "fixed_mc_samples": mpc_kw["fixed_mc_samples"],
        "verbose": mpc_kw["verbose"],
        "inv_coef": mpc_kw.get("inv_coef", 0.0),
        "target_inv": mpc_kw.get("target_inv", 0.0),
        "risk_coef": mpc_kw.get("risk_coef", mpc_kw.get("risk_inv", 0.0)),
        "uniformity_coef": mpc_kw.get("uniformity_coef", mpc_kw.get("action_coef", 0.0)),
        "reward_coef": mpc_kw.get("reward_coef", 1.0),
        "size_coef": mpc_kw.get("size_coef", 0.0),
        "spread_coef": mpc_kw.get("spread_coef", 0.0),
        "spread_target_lots": mpc_kw.get("spread_target_lots", 1.0),
        "total_q_target": mpc_kw.get("total_q_target"),
        "project_total_q_target": mpc_kw.get("project_total_q_target", False),
        "soft_fill_temp": mpc_kw.get("soft_fill_temp", 1.0),
    }
    mc_mpc_kw.update(_hedge.mpc_optimizer_kwargs())
    optimizer = MonteCarloMPCTensorBatched(
        price_model,
        market_order_flow_model,
        execution_model,
        hedge_price_model=hedge_price_model,
        **{k: v for k, v in mc_mpc_kw.items() if k in _get_init_params(MonteCarloMPCTensorBatched)},
    )
    # Bounds: bid/ask spread always [1, spread_max]; single source is spread_max.

    ctrl_disc = {
        "s_b_min": 1,
        "s_b_max": spread_max,
        "s_a_min": 1,
        "s_a_max": spread_max,
        "lot_size": lot_size,
        "spread_max": spread_max,
        "zero_action_if_negative_J_mean": mpc_kw.get("zero_action_if_negative_J_mean", False),
    }
    controller = SigmoidController(optimizer, **ctrl_disc)
    execution_rule = DefaultExecutionRule(**config["execution_rule_kwargs"])

    return (
        market_order_flow_model,
        tester,
        price_model,
        price_tester,
        controller,
        execution_rule,
        execution_model,
        runner_execution_model,
        hedge_price_model,
    )


def create_runner(
    exchange: str,
    symbol: str,
    T_ms: int,
    *,
    trades_limit: int,
    depth_limit: int,
    model: TorchHurdleWeibullModel,
    tester: MarketOrderFlowTester,
    controller: "Controller | None",
    initial_inventory: int,
    initial_cash: float,
    price_model: TorchResidualPriceModel | None,
    execution_rule: "ExecutionRule | None",
    execution_model: "ExecutionModel | None",
    price_tester: object,
    decay: float,
    spread_max: float,
    runner_execution_model: "ExecutionModel | None" = None,
    use_buffered_orderbook_provider: bool,
    use_websocket_market_data: bool,
    trades_poll_interval_ms: int = 200,
    trade_buffer_ms: int = 10 * 60 * 1000,
    orderbook_poll_interval_ms: int = 500,
    orderbook_buffer_ms: int = 10 * 60 * 1000,
    websocket_reconnect_ms: int = 2000,
    private_api_token: str | None = None,
    step_window_mode: str = "lookback",
    balance_update_mode: str = "events",
    use_zero_action: bool = False,
    expected_mid_n_samples: int = 10,
    expected_mid_debug: bool = False,
    expected_mid_debug_every: int = 1,
    live_order_execution: bool = False,
    action_start_after_steps: int = 0,
    hedge_exchange: str | None = None,
    hedge_symbol: str | None = None,
    use_hedge_websocket_market_data: bool = True,
    hedge_depth_limit: int | None = None,
    hedge_orderbook_poll_interval_ms: int | None = None,
    hedge_orderbook_buffer_ms: int | None = None,
    hedge_websocket_reconnect_ms: int | None = None,
    main_representative_price_target_depth_lots: float | None = None,
    hedge_representative_price_target_depth_lots: float = 1000.0,
    use_hedge_gaussian_price_model: bool = True,
    hedge_price_model_decay: float = 0.97,
    hedge_price_model_sigma_min_ticks: float = 1.0,
    hedge_private_api_token: str | None = None,
    use_hedge_portfolio_balances: bool = False,
    hedge_enabled: bool = False,
    hedge_price_model: object | None = None,
    valuation_policy: ValuationPolicy | str | None = None,
) -> OrderFlowRunner:
    """
    Build OrderFlowStepper and OrderFlowRunner from pipeline components.

    ``expected_mid_n_samples`` is the shared diagnostic MC size: expected mid (MOF-only
    book + residual), representative-price MC, and E[MOF] from ``model.sample()`` means. The MPC
    optimizer uses ``pipeline['mpc_kwargs']['n_samples']`` (often larger) for batched
    trajectory rollouts — a separate knob.
    """
    # Poll intervals are only relevant for REST/polling providers. In websocket mode,
    # OrderFlowStepper wires WebSocketBufferedMarketDataProvider and ignores them.
    stepper = create_stepper(
        exchange=exchange,
        symbol=symbol,
        T_ms=T_ms,
        trades_limit=trades_limit,
        depth_limit=depth_limit,
        use_buffered_orderbook_provider=use_buffered_orderbook_provider,
        use_websocket_market_data=use_websocket_market_data,
        trades_poll_interval_ms=trades_poll_interval_ms,
        trade_buffer_ms=trade_buffer_ms,
        orderbook_poll_interval_ms=orderbook_poll_interval_ms,
        orderbook_buffer_ms=orderbook_buffer_ms,
        websocket_reconnect_ms=websocket_reconnect_ms,
        private_api_token=private_api_token,
        step_window_mode=step_window_mode,
        balance_update_mode=balance_update_mode,
        hedge_exchange=hedge_exchange,
        hedge_symbol=hedge_symbol,
        use_hedge_websocket_market_data=use_hedge_websocket_market_data,
        hedge_depth_limit=hedge_depth_limit,
        hedge_orderbook_poll_interval_ms=hedge_orderbook_poll_interval_ms,
        hedge_orderbook_buffer_ms=hedge_orderbook_buffer_ms,
        hedge_websocket_reconnect_ms=hedge_websocket_reconnect_ms,
        main_representative_price_target_depth_lots=main_representative_price_target_depth_lots,
        hedge_representative_price_target_depth_lots=hedge_representative_price_target_depth_lots,
        use_hedge_gaussian_price_model=use_hedge_gaussian_price_model,
        hedge_price_model_decay=hedge_price_model_decay,
        hedge_price_model_sigma_min_ticks=hedge_price_model_sigma_min_ticks,
        hedge_private_api_token=hedge_private_api_token,
        use_hedge_portfolio_balances=use_hedge_portfolio_balances,
    )
    return OrderFlowRunner(
        stepper=stepper,
        model=model,
        tester=tester,
        controller=controller,
        initial_inventory=initial_inventory,
        initial_cash=initial_cash,
        price_model=price_model,
        execution_rule=execution_rule,
        execution_model=execution_model,
        price_tester=price_tester,
        runner_execution_model=runner_execution_model,
        spread_max=spread_max,
        use_zero_action=use_zero_action,
        expected_mid_n_samples=expected_mid_n_samples,
        expected_mid_debug=expected_mid_debug,
        expected_mid_debug_every=expected_mid_debug_every,
        live_order_execution=live_order_execution,
        action_start_after_steps=action_start_after_steps,
        hedge_enabled=hedge_enabled,
        hedge_price_model=hedge_price_model,
        valuation_policy=valuation_policy,
    )


def main() -> None:
    """
    Deprecated legacy headless entrypoint.

    Production runtime is ``mm.runtime`` / ``python -m mm.runtime.print_events``.
    """
    raise SystemExit(
        "mm.layers.runtime.pipeline.main is deprecated. "
        "Use `python -m mm.runtime.print_events --config <path>`."
    )


__all__ = ["PipelineConfig", "build_pipeline", "create_runner", "main"]
