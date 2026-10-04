import json
import tempfile
import pytest

from mm.runtime.config import load_runtime_config
from mm.strategies.mpc_engine import MpcStrategyEngine
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.market import MarketEvent, RawOrderbook, PriceLevel
from mm.core.models.private import PrivateStateSnapshot, Balance


def test_k8s_production_config_compatibility() -> None:
    with open("mm/configs/market_maker.runtime.example.json") as f:
        cfg = json.load(f)

    ctrl = cfg["strategy_presets"]["thin_local_liquidity_weighted_mpc"]["controller"]
    ctrl["optimizer"]["adam_lr"] = 1
    ctrl["optimizer"]["horizon"] = 3
    ctrl["optimizer"]["n_samples"] = 16
    ctrl["optimizer"]["adam_epochs"] = 8
    ctrl["optimizer"]["fixed_mc_samples"] = True
    ctrl["optimizer"]["warm_start"] = {
        "alpha": 0.5,
        "enabled": True,
        "alpha_max": 0.95,
        "alpha_mode": "fixed",
        "grad_norm_eps": 1e-12,
        "preserve_optimizer_state": False,
    }
    ctrl["optimizer"]["objective_normalize"] = {
        "enabled": False,
        "ce_ewma_decay": 0.9,
    }
    ctrl["action_topic"] = "strategy:actions"
    ctrl["execution_grid"] = {
        "mode": "log",
        "levels": 16,
        "spread_max": 1000,
        "spread_min": 100,
    }
    ctrl["orderbook_grid"] = {
        "spread_max": 1000,
        "spread_step": 50,
    }
    ctrl["local_price_model"]["min_refresh_weight"] = 0.15
    ctrl["global_price_model"]["min_refresh_weight"] = 0.15

    with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
        json.dump(cfg, tmp)
        tmp.flush()
        rt = load_runtime_config(tmp.name)

    # 1. Verify Grid contracts
    assert rt.controller.orderbook_grid.levels == 20
    assert rt.controller.orderbook_grid.spread_ladder()[0] == 50.0
    assert rt.controller.orderbook_grid.spread_ladder()[-1] == 1000.0
    assert len(rt.controller.orderbook_grid.spread_ladder()) == 20

    exec_ladder = rt.controller.execution_spread_ladder()
    assert len(exec_ladder) == 16
    assert exec_ladder[0] == 100.0
    assert exec_ladder[-1] == 1000.0

    # 2. Verify Optimizer instantiation & warm-up
    engine = MpcStrategyEngine(
        config=rt.controller,
        binding=rt.binding,
        liquidity_sourcing=rt.liquidity_sourcing,
        execution=rt.execution,
    )
    opt = engine._optimizer_backend.controller
    assert opt.adam_lr == 1
    assert opt.horizon == 3
    assert opt.n_samples == 16
    assert opt.adam_epochs == 8
    assert opt.grid_levels == 20
    assert opt.spread_max == 1000.0
    assert opt.spread_step == 50.0
    assert opt.warm_start.enabled is True
    assert opt.warm_start.alpha == 0.5
    assert opt.warm_start.preserve_optimizer_state is False

    # 3. Verify price models
    assert opt.local_price_model._min_refresh_weight == 0.15
    if opt.global_price_model is not None and hasattr(opt.global_price_model, "_min_refresh_weight"):
        assert opt.global_price_model._min_refresh_weight == 0.15

    # 4. Run consecutive frames
    ob = RawOrderbook(
        exchange="nobitex",
        symbol="USDT_IRT",
        timestamp_ms=2000,
        bids=tuple(PriceLevel(price=99000.0 - i * 50.0, qty=1.0) for i in range(25)),
        asks=tuple(PriceLevel(price=101000.0 + i * 50.0, qty=1.0) for i in range(25)),
        is_snapshot=True,
    )
    event = MarketEvent(
        exchange="nobitex",
        symbol="USDT_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=ob,
        trades=(),
        buy_volume=0.5,
        sell_volume=0.5,
        vwap=100000.0,
        imbalance=0.0,
        microprice=100000.0,
    )
    priv = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2000,
        balances={
            "USDT": Balance("nobitex", "USDT", 1000.0, 0.0, 1000.0, 2000),
            "IRT": Balance("nobitex", "IRT", 100_000_000.0, 0.0, 100_000_000.0, 2000),
        },
        open_orders={},
    )
    frame1 = StrategyInputFrame(local=event, local_private=priv)
    decision1 = engine.decide(frame1, frame_step=1)
    assert decision1 is not None
    assert len(decision1.ladder_orders) > 0
    assert len(decision1.orders) > 0

    frame2 = StrategyInputFrame(local=event, local_private=priv)
    decision2 = engine.decide(frame2, frame_step=2)
    assert decision2 is not None
    assert len(decision2.orders) > 0
