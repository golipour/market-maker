"""
Order-flow pipeline: build models, testers, MPC controller, and run animation.
Uses models from market_making_controller.models: TorchHurdleWeibullModel, TorchResidualPriceModel (MOF + two-sided residual),
MonteCarloMPCTensorBatched, SigmoidController.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from order_flow_stepper import OrderFlowStepper
from market_making_controller.models import (
    TorchHurdleWeibullModel,
    TorchResidualPriceModel,
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
        ObservedStepExecutionModel,
    )
    from order_flow_stepper import AnchorGaussianRepPriceModel

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
    # maker_fee is retained for MPC compatibility (it reads execution_model.maker_fee).
    execution_model = ObservedStepExecutionModel(maker_fee=float(exec_kw.get("maker_fee", 0.0)))
    runner_execution_model = execution_model
    anchor_price_model = AnchorGaussianRepPriceModel(
        decay=float(config.get("anchor_price_model_decay", 0.97) or 0.97),
        sigma_min_ticks=float(config.get("anchor_price_model_sigma_min_ticks", 1.0) or 1.0),
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
        "anchor_hedge_coef": 1.0 if bool(mpc_kw.get("anchor_hedge_enabled", mpc_kw.get("anchor_hedge_coef", 0.0))) else 0.0,
        "anchor_hedge_fee": mpc_kw.get("anchor_hedge_fee", 0.0),
        "total_q_target": mpc_kw.get("total_q_target"),
        "project_total_q_target": mpc_kw.get("project_total_q_target", False),
        "soft_fill_temp": mpc_kw.get("soft_fill_temp", 1.0),
    }
    optimizer = MonteCarloMPCTensorBatched(
        price_model,
        market_order_flow_model,
        execution_model,
        anchor_price_model=anchor_price_model,
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
        anchor_price_model,
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
    anchor_exchange: str | None = None,
    anchor_symbol: str | None = None,
    use_anchor_websocket_market_data: bool = True,
    anchor_depth_limit: int | None = None,
    anchor_orderbook_poll_interval_ms: int | None = None,
    anchor_orderbook_buffer_ms: int | None = None,
    anchor_websocket_reconnect_ms: int | None = None,
    main_representative_price_target_depth_lots: float | None = None,
    anchor_representative_price_target_depth_lots: float = 1000.0,
    use_anchor_gaussian_price_model: bool = True,
    anchor_price_model_decay: float = 0.97,
    anchor_price_model_sigma_min_ticks: float = 1.0,
    anchor_private_api_token: str | None = None,
    use_anchor_portfolio_balances: bool = False,
    anchor_hedge_enabled: bool = False,
    anchor_price_model: object | None = None,
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
        anchor_exchange=anchor_exchange,
        anchor_symbol=anchor_symbol,
        use_anchor_websocket_market_data=use_anchor_websocket_market_data,
        anchor_depth_limit=anchor_depth_limit,
        anchor_orderbook_poll_interval_ms=anchor_orderbook_poll_interval_ms,
        anchor_orderbook_buffer_ms=anchor_orderbook_buffer_ms,
        anchor_websocket_reconnect_ms=anchor_websocket_reconnect_ms,
        main_representative_price_target_depth_lots=main_representative_price_target_depth_lots,
        anchor_representative_price_target_depth_lots=anchor_representative_price_target_depth_lots,
        use_anchor_gaussian_price_model=use_anchor_gaussian_price_model,
        anchor_price_model_decay=anchor_price_model_decay,
        anchor_price_model_sigma_min_ticks=anchor_price_model_sigma_min_ticks,
        anchor_private_api_token=anchor_private_api_token,
        use_anchor_portfolio_balances=use_anchor_portfolio_balances,
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
        anchor_hedge_enabled=anchor_hedge_enabled,
        anchor_price_model=anchor_price_model,
    )


def main() -> None:
    """Build pipeline and runner from config, then run animation."""
    from order_flow_animation import run_animation

    args = {
        "exchange": "nobitex",
        "symbol": "USDT_IRT",
        "anchor_exchange": "wallex",
        "T_ms": 10000,
        "animation_interval_ms": 10000,
        "orderbook_ticks": 50,
        "limit": 1000,
        "depth_limit": 100,
        "lot_size": 1,  # from exchange/symbol API (e.g. trade_stats._lot_size(symbol, exchange))
        "trade_buffer_ms": 1 * 60 * 1000,
        "orderbook_buffer_ms": 1 * 60 * 1000,
        "websocket_reconnect_ms": 2000,
        "private_api_token": "c1b236dd68df9686401ca23bce07d2a90e02b96b",  # API token to enable private ws channels
        "step_window_mode": "previous_end",  # "lookback" or "previous_end"
        "balance_update_mode": "rest",  # "events" (private trades/orders deltas) or "rest" (poll balances each step)
        "decay": 0.90,
        "use_buffered_orderbook_provider": True,
        "use_websocket_market_data": True,
        "use_zero_action": False,  # False to run controller (MonteCarloMPCTensorBatched); True = no quotes
        "action_start_after_steps": 10,  # warm-up: delay live order placement for first N steps (controller still runs)
        "expected_mid_n_samples": 32,
        "expected_mid_debug": False,
        "expected_mid_debug_every": 1,
        "anchor_symbol": None,  # defaults to args["symbol"] when anchor_exchange is set
        "use_anchor_websocket_market_data": True,
        "anchor_depth_limit": None,  # defaults to depth_limit
        "anchor_orderbook_poll_interval_ms": None,  # only used when anchor websocket is disabled
        "anchor_orderbook_buffer_ms": None,  # defaults to orderbook_buffer_ms
        "anchor_websocket_reconnect_ms": None,  # defaults to websocket_reconnect_ms
        "main_representative_price_target_depth_lots": None,  # None => same as anchor target depth
        "anchor_representative_price_target_depth_lots": 1e3,
        "use_anchor_gaussian_price_model": True,
        "anchor_price_model_decay": 0.9,
        "anchor_price_model_sigma_min_ticks": 1.0,
        "anchor_private_api_token": "18404|41TYyNTlan4QZhnaBykIkIwbCxZz3roczX7Pvuyr",  # defaults to private_api_token when None
        "use_anchor_portfolio_balances": True,
        # Plot windows toggles (performance): Figure 1 (main) is always enabled.
        "enable_loss_figure": False,    # Figure 2
        "enable_action_figure": True,  # Figure 3
        "enable_price_figure": False,   # Figure 4
        "enable_qq_figure": False,      # Figure 5
        "enable_anchor_figure": True,  # Figure 6
    }

    pipeline = {
        "market_order_flow_model_kwargs": {
            "decay": 0.80,
            "param_eps": 1e-6,
            "device": None,
            "init_p0": 0.3,
            "init_mean": 10.0,
            "init_k": 0.1,
        },
        "price_model_kwargs": {
            "residual_mu": 0.0,
            "residual_sigma": 1.0,
            "residual_decay": 0.80,
            "learn_residual": True,
            "sigma_min": 1e-6,
            "representative_price_target_depth_lots": 1e5,
        },
        "order_flow_model_kwargs": {},
        "mpc_kwargs": {
            "spread_max": 100,
            "n_components": 3,
            "device": "cpu",
            "total_q_min": 0,
            "total_q_max": 1e3,
            "soft_fill_temp": 0.25,
            "adam_lr": 1e0,
            "adam_epochs": 50,
            "adam_grad_clip": 1e10,
            "horizon": 1,
            "n_samples": 128,
            "warm_start": False,
            "fixed_mc_samples": True,
            "verbose": False,
            # Unified objective:
            # J = reward_coef * J_mean - inv_coef * (J_inv - target_inv)^2 - risk_coef * J_std
            #     - uniformity_coef * J_uniformity - size_coef * J_size - spread_coef * J_spread
            "reward_coef": 1.0,  # weight on E[ΔW] term
            "risk_coef": 1.0,
            "inv_coef": 0.0,
            "uniformity_coef": 0.0,
            "size_coef": 0.0,                 # penalize (Q_range_bid-target)^2 + (Q_range_ask-target)^2
            "spread_coef": 100.0,               # penalize bid+ask spread at spread_target_lots cumulative depth
            "spread_target_lots": 50.0,        # depth level used to measure bid/ask spread term
            "anchor_hedge_enabled": True,    # when True, anchor hedge reward is added with fixed weight 1.0
            "anchor_hedge_coef": 0.0,         # deprecated: kept for compatibility as ON/OFF fallback
            "anchor_hedge_fee": 0.0000025,          # proportional fee factor on hedge contribution
            "target_inv": 1e3,
            "total_q_target": 1e3,           # finite-range target on [0, spread_max]
            "project_total_q_target": False,  # if True, rescale L after updates to match total_q_target
            "zero_action_if_negative_J_mean": False,   # if True, return no quotes when best action has E[ΔW] < 0
        },
        "execution_model_kwargs": {
            "maker_fee": 0.00015,
            # Used only when execution_mode == "simulated"
            "simulated_initial_inventory": 0,
            "simulated_initial_cash": 0.0,
        },
        "execution_mode": "real",  # "simulated" (OB+flow fills) or "real" (inventory/cash from Step)
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
        anchor_price_model,
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
    _mpc_cfg = (pipeline.get("mpc_kwargs", {}) or {})
    anchor_hedge_enabled = bool(_mpc_cfg.get("anchor_hedge_enabled", _mpc_cfg.get("anchor_hedge_coef", 0.0)))
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
        expected_mid_n_samples=args["expected_mid_n_samples"],
        expected_mid_debug=args["expected_mid_debug"],
        expected_mid_debug_every=args["expected_mid_debug_every"],
        live_order_execution=live_order_execution,
        action_start_after_steps=args["action_start_after_steps"],
        anchor_exchange=args["anchor_exchange"],
        anchor_symbol=args["anchor_symbol"],
        use_anchor_websocket_market_data=args["use_anchor_websocket_market_data"],
        anchor_depth_limit=args["anchor_depth_limit"],
        anchor_orderbook_poll_interval_ms=args["anchor_orderbook_poll_interval_ms"],
        anchor_orderbook_buffer_ms=args["anchor_orderbook_buffer_ms"],
        anchor_websocket_reconnect_ms=args["anchor_websocket_reconnect_ms"],
        main_representative_price_target_depth_lots=args["main_representative_price_target_depth_lots"],
        anchor_representative_price_target_depth_lots=args["anchor_representative_price_target_depth_lots"],
        use_anchor_gaussian_price_model=args["use_anchor_gaussian_price_model"],
        anchor_price_model_decay=args["anchor_price_model_decay"],
        anchor_price_model_sigma_min_ticks=args["anchor_price_model_sigma_min_ticks"],
        anchor_private_api_token=args["anchor_private_api_token"],
        use_anchor_portfolio_balances=args["use_anchor_portfolio_balances"],
        anchor_hedge_enabled=anchor_hedge_enabled,
        anchor_price_model=anchor_price_model,
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
        enable_loss_figure=args["enable_loss_figure"],
        enable_action_figure=args["enable_action_figure"],
        enable_price_figure=args["enable_price_figure"],
        enable_qq_figure=args["enable_qq_figure"],
        enable_anchor_figure=args["enable_anchor_figure"],
    )


if __name__ == "__main__":
    main()
