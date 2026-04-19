"""
Order-flow pipeline: build models, testers, MPC controller, and run animation.
Uses models from market_making_controller.models: TorchHurdleWeibullModel, TorchResidualPriceModel (MOF + two-sided residual),
MonteCarloMPCTensorBatched, SigmoidController.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from order_flow_stepper import OrderFlowStepper
from trade_stats import get_min_order_amount_tick_lot
from market_making_controller.models import (
    TorchHurdleWeibullModel,
    TorchResidualPriceModel,
    GaussianPriceModel,
    HurdleOrderFlow,
)
from order_flow_testers import (
    MarketOrderFlowTester,
    GeneralOrderFlowLogLossTester,
    PriceLogLossTester,
)
from order_flow_runner import OrderFlowRunner

if TYPE_CHECKING:
    from market_making_controller import Controller
    from market_making_controller.models import ExecutionRule, ExecutionModel

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
    from market_making_controller import (
        SigmoidController,
        MonteCarloMPCTensorBatched,
        DefaultExecutionRule,
        DiscretizedExecutionModel,
        ObservedStepExecutionModel,
    )

    mpc_kw = config["mpc_kwargs"]
    spread_max = mpc_kw["spread_max"]
    spread_int = max(1, int(spread_max))

    mof_kw = config["market_order_flow_model_kwargs"]
    market_order_flow_model = TorchHurdleWeibullModel(**{k: v for k, v in mof_kw.items() if k in _get_init_params(TorchHurdleWeibullModel)})
    tester = GeneralOrderFlowLogLossTester(market_order_flow_model, decay=decay)

    limit_order_flow_model = None

    p_kw = config["price_model_kwargs"]
    price_model = TorchResidualPriceModel(
        market_flow_model=market_order_flow_model,
        **{k: v for k, v in p_kw.items() if k in _get_init_params(TorchResidualPriceModel)},
    )
    price_tester = PriceLogLossTester(price_model, decay=decay)

    order_flow = HurdleOrderFlow(**config["order_flow_model_kwargs"])
    price_model_for_exec = GaussianPriceModel()
    exec_kw = config["execution_model_kwargs"]
    execution_model = DiscretizedExecutionModel(
        order_flow_model=order_flow,
        price_model=price_model_for_exec,
        maker_fee=exec_kw["maker_fee"],
    )
    _exec_mode = config.get("execution_mode", "simulated")
    if _exec_mode not in ("simulated", "real"):
        raise ValueError(
            "execution_mode must be 'simulated' or 'real', "
            f"got {_exec_mode!r}"
        )
    runner_execution_model = (
        ObservedStepExecutionModel()
        if _exec_mode == "real"
        else execution_model
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
        "adam_normalize_inputs": mpc_kw["adam_normalize_inputs"],
        "total_q_min": mpc_kw["total_q_min"],
        "total_q_max": mpc_kw["total_q_max"],
        "fixed_mc_samples": mpc_kw["fixed_mc_samples"],
        "verbose": mpc_kw["verbose"],
        "reward_mode": mpc_kw.get("reward_mode", "step_pnl"),
        "reward_gamma": mpc_kw.get("reward_gamma", 0.0),
        "reward_lambda": mpc_kw.get("reward_lambda", 0.0),
        "reward_eta": mpc_kw.get("reward_eta", 0.0),
        "soft_fill_temp": mpc_kw.get("soft_fill_temp", 1.0),
    }
    optimizer = MonteCarloMPCTensorBatched(
        price_model,
        market_order_flow_model,
        execution_model,
        **{k: v for k, v in mc_mpc_kw.items() if k in _get_init_params(MonteCarloMPCTensorBatched)},
    )
    # Bounds: bid/ask spread always [1, spread_max]; single source is spread_max.

    min_order_notional_tick_lot = config.get("min_order_notional_tick_lot")
    if min_order_notional_tick_lot is None and exchange and symbol:
        try:
            min_order_notional_tick_lot = float(get_min_order_amount_tick_lot(symbol, exchange))
        except Exception:
            min_order_notional_tick_lot = 0.0
    min_order_notional_tick_lot = float(min_order_notional_tick_lot or 0.0)
    min_notional_safety_mult = float(config.get("min_notional_safety_mult", 2.0))
    ctrl_disc = {
        "s_b_min": 1,
        "s_b_max": spread_max,
        "s_a_min": 1,
        "s_a_max": spread_max,
        "lot_size": lot_size,
        "spread_max": spread_max,
        "zero_action_if_negative_J_mean": mpc_kw.get("zero_action_if_negative_J_mean", False),
        "min_order_notional_tick_lot": min_order_notional_tick_lot,
        "min_notional_safety_mult": min_notional_safety_mult,
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
        limit_order_flow_model,
        None,
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
    limit_order_flow_model: object = None,
    expected_mid_n_samples: int = 10,
    expected_mid_debug: bool = False,
    expected_mid_debug_every: int = 1,
    live_order_execution: bool = False,
    action_start_after_steps: int = 0,
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
    stepper = OrderFlowStepper(
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
        limit_order_flow_model=limit_order_flow_model,
        expected_mid_n_samples=expected_mid_n_samples,
        expected_mid_debug=expected_mid_debug,
        expected_mid_debug_every=expected_mid_debug_every,
        live_order_execution=live_order_execution,
        action_start_after_steps=action_start_after_steps,
    )


def main() -> None:
    """Build pipeline and runner from config, then run animation."""
    from order_flow_animation import run_animation

    args = {
        "exchange": "nobitex",
        "symbol": "USDT_IRT",
        "T_ms": 3000,
        "animation_interval_ms": 3000,
        "orderbook_ticks": 50,
        "limit": 1000,
        "depth_limit": 100,
        "lot_size": 1,  # from exchange/symbol API (e.g. trade_stats._lot_size(symbol, exchange))
        "trade_buffer_ms": 1 * 60 * 1000,
        "orderbook_buffer_ms": 1 * 60 * 1000,
        "websocket_reconnect_ms": 2000,
        "private_api_token": "358e4fe544bff82683341eb9ff3e58fbea81cc19",  # API token to enable private ws channels
        "step_window_mode": "previous_end",  # "lookback" or "previous_end"
        "balance_update_mode": "rest",  # "events" (private trades/orders deltas) or "rest" (poll balances each step)
        "decay": 0.90,
        "use_buffered_orderbook_provider": True,
        "use_websocket_market_data": True,
        "use_zero_action": False,  # False to run controller (MonteCarloMPCTensorBatched); True = no quotes
        "action_start_after_steps": 10,  # warm-up: keep zero action for first N steps, then start controller
        "expected_mid_n_samples": 32,
        "expected_mid_debug": False,
        "expected_mid_debug_every": 1,
    }

    pipeline = {
        "market_order_flow_model_kwargs": {
            "decay": 0.90,
            "param_eps": 1e-6,
            "device": None,
            "init_p0": 0.3,
            "init_mean": 10.0,
            "init_k": 0.1,
        },
        "price_model_kwargs": {
            "residual_mu": 0.0,
            "residual_sigma": 1.0,
            "residual_decay": 0.95,
            "learn_residual": True,
            "sigma_min": 1e-6,
            "representative_price_target_depth_lots": 1e12,
        },
        "order_flow_model_kwargs": {},
        "mpc_kwargs": {
            "spread_max": 100,
            "n_components": 1,
            "device": "cpu",
            "total_q_min": 0,
            "total_q_max": 1e2,
            "soft_fill_temp": 1e-6,
            "adam_lr": 1e1,
            "adam_epochs": 21,
            "adam_grad_clip": 1e10,
            "horizon": 3,
            "n_samples": 128,
            "warm_start": False,
            "fixed_mc_samples": True,
            "adam_normalize_inputs": True,
            "verbose": False,
            "reward_mode": "terminal_mean_std",        # "step_pnl", "terminal_pnl_inventory", "terminal_mean_variance", "terminal_cara"
            "reward_gamma": 0.0,                            # for terminal_pnl_inventory
            "reward_lambda": 1.0,                          # for terminal_mean_variance (J = J_mean - λ*J_var);
            "reward_eta": 0.0,                              # for terminal_cara
            "zero_action_if_negative_J_mean": True,   # if True, return no quotes when best action has E[ΔW] < 0
        },
        "execution_model_kwargs": {
            "maker_fee": 0.0001,
            # Used only when execution_mode == "simulated"
            "simulated_initial_inventory": 0,
            "simulated_initial_cash": 0.0,
        },
        "execution_mode": "real",  # "simulated" (OB+flow fills) or "real" (inventory/cash from Step)
        "min_order_notional_tick_lot": None,  # None => auto-load MIN_NOTIONAL from exchange info and convert to tick*lot
        "min_notional_safety_mult": 2.0,      # keep only levels with notional >= mult * MIN_NOTIONAL
        "execution_rule_kwargs": {}
    }

    (
        market_order_flow_model,
        tester,
        price_model,
        price_tester,
        controller,
        execution_rule,
        execution_model,
        runner_execution_model,
        limit_order_flow_model,
        _,
    ) = build_pipeline(
        pipeline,
        decay=args["decay"],
        lot_size=args["lot_size"],
        exchange=args["exchange"],
        symbol=args["symbol"],
    )
    spread_max = pipeline["mpc_kwargs"]["spread_max"]
    execution_mode = str(pipeline.get("execution_mode", "simulated")).lower()
    live_order_execution = execution_mode == "real"
    exec_kw = pipeline.get("execution_model_kwargs", {}) or {}
    sim_initial_inventory = int(exec_kw.get("simulated_initial_inventory", 0))
    sim_initial_cash = float(exec_kw.get("simulated_initial_cash", 0.0))
    runner = create_runner(
        exchange=args["exchange"],
        symbol=args["symbol"],
        T_ms=args["T_ms"],
        trades_limit=args["limit"],
        depth_limit=args["depth_limit"],
        model=market_order_flow_model,
        tester=tester,
        controller=controller,
        initial_inventory=sim_initial_inventory if execution_mode == "simulated" else 0,
        initial_cash=sim_initial_cash if execution_mode == "simulated" else 0.0,
        price_model=price_model,
        execution_rule=execution_rule,
        execution_model=execution_model,
        price_tester=price_tester,
        runner_execution_model=runner_execution_model,
        decay=args["decay"],
        spread_max=spread_max,
        use_buffered_orderbook_provider=args["use_buffered_orderbook_provider"],
        use_websocket_market_data=args["use_websocket_market_data"],
        trade_buffer_ms=args["trade_buffer_ms"],
        orderbook_buffer_ms=args["orderbook_buffer_ms"],
        websocket_reconnect_ms=args["websocket_reconnect_ms"],
        private_api_token=args["private_api_token"],
        step_window_mode=args["step_window_mode"],
        balance_update_mode=args["balance_update_mode"],
        use_zero_action=args["use_zero_action"],
        limit_order_flow_model=limit_order_flow_model,
        expected_mid_n_samples=args["expected_mid_n_samples"],
        expected_mid_debug=args["expected_mid_debug"],
        expected_mid_debug_every=args["expected_mid_debug_every"],
        live_order_execution=live_order_execution,
        action_start_after_steps=args["action_start_after_steps"],
    )
    run_animation(
        runner=runner,
        exchange=args["exchange"],
        symbol=args["symbol"],
        T_ms=args["T_ms"],
        fixed_vol_max=None,
        orderbook_ticks=args["orderbook_ticks"],
        decay=args["decay"],
        spread_max=spread_max,
        animation_interval_ms=args["animation_interval_ms"],
    )


if __name__ == "__main__":
    main()
