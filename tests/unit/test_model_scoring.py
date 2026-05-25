from __future__ import annotations

import math

import pytest

from mm.models.scoring import (
    score_global_vwap_deltas,
    score_local_mid_delta,
    score_market_flow,
)
from mm.mpc.global_price_torch import GlobalGaussianPriceModel
from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel
from mm.mpc.price_torch import TorchResidualPriceModel


def test_hurdle_lognormal_flow_nll_uses_pre_update_params() -> None:
    params = {
        "p0_buy": 0.2,
        "p0_sell": 0.3,
        "mu_buy": 2.0,
        "sigma_buy": 0.5,
        "mu_sell": 1.5,
        "sigma_sell": 0.4,
    }
    result = score_market_flow(
        buy_volume=0.0,
        sell_volume=5.0,
        params_before=params,
        kind="hurdle_lognormal",
    )
    assert result.skip_reason is None
    assert result.nll is not None
    assert result.log_likelihood is not None
    assert result.nll == pytest.approx(-result.log_likelihood)
    # P(buy=0) = p0_buy = 0.2
    assert result.observation["buy_volume"] == 0
    assert result.observation["prob_buy"] == pytest.approx(0.2, rel=1e-6)
    assert result.observation["prob_sell"] > 0.0


def test_local_mid_delta_nll_deterministic() -> None:
    params = {"mu_mid": 1.0, "sigma_mid": 2.0}
    result = score_local_mid_delta(delta_mid=3.0, params_before=params)
    assert result.skip_reason is None
    assert result.nll is not None
    z = (3.0 - 1.0) / 2.0
    expected_ll = -0.5 * math.log(2 * math.pi) - math.log(2.0) - 0.5 * z * z
    assert result.log_likelihood == pytest.approx(expected_ll)
    assert result.standardized_residual == pytest.approx(z)


def test_local_mid_delta_skips_invalid_sigma() -> None:
    result = score_local_mid_delta(delta_mid=1.0, params_before={"mu_mid": 0.0, "sigma_mid": 0.0})
    assert result.skip_reason == "invalid_sigma"
    assert result.nll is None


def test_strategy_price_model_update_uses_true_vwap_residuals() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
    from mm.core.models.strategy import StrategyInputFrame
    from mm.strategies.mpc_engine import _update_residual_price_model

    prev_book = RawOrderbook(
        exchange="nobitex",
        symbol="USDT_IRT",
        timestamp_ms=1000,
        bids=(PriceLevel(99.0, 10.0),),
        asks=(PriceLevel(101.0, 10.0),),
        is_snapshot=True,
    )
    curr_book = RawOrderbook(
        exchange="nobitex",
        symbol="USDT_IRT",
        timestamp_ms=2000,
        bids=(PriceLevel(99.0, 10.0),),
        asks=(PriceLevel(103.0, 10.0),),
        is_snapshot=True,
    )
    previous = StrategyInputFrame(local=MarketEvent(
            exchange="nobitex",
            symbol="USDT_IRT",
            interval_ms=1000,
            t_start_ms=0,
            t_end_ms=1000,
            orderbook=prev_book,
            trades=(),
            buy_volume=0.0,
            sell_volume=0.0,
            vwap=0.0,
            imbalance=0.0,
            microprice=100.0,
        )
    )
    current = StrategyInputFrame(local=MarketEvent(
            exchange="nobitex",
            symbol="USDT_IRT",
            interval_ms=1000,
            t_start_ms=1000,
            t_end_ms=2000,
            orderbook=curr_book,
            trades=(),
            buy_volume=5.0,
            sell_volume=0.0,
            vwap=0.0,
            imbalance=0.0,
            microprice=101.0,
        )
    )
    model = TorchResidualPriceModel(residual_decay=0.5, residual_sigma=1.0, vwap_depth=1.0)

    _update_residual_price_model(model, previous, current)

    params = model.get_params()
    assert params["mu_buy"] == pytest.approx(1.0)
    assert params["mu_sell"] == pytest.approx(0.0)
    assert model.computed_buy_price_hist[-1] == pytest.approx(101.0)
    assert model.realized_buy_price_hist[-1] == pytest.approx(103.0)


def test_global_vwap_delta_nll_deterministic() -> None:
    params = {
        "mu_delta_buy": 0.5,
        "mu_delta_sell": -0.25,
        "sigma_delta_buy": 1.0,
        "sigma_delta_sell": 2.0,
    }
    result = score_global_vwap_deltas(delta_buy=1.5, delta_sell=0.0, params_before=params)
    assert result.skip_reason is None
    assert result.nll is not None
    assert result.nll > 0.0


def test_torch_model_score_before_update_differs_from_after() -> None:
    flow = TorchHurdleLogNormalModel(decay=0.5, init_p0=0.5, init_mu=1.0, init_sigma=0.5)
    params_before = dict(flow.get_params())
    before_nll = score_market_flow(
        buy_volume=10.0,
        sell_volume=0.0,
        params_before=params_before,
        kind="hurdle_lognormal",
    ).nll
    flow.update(None, (0, None, 10.0, 0.0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0))
    params_after = dict(flow.get_params())
    after_nll = score_market_flow(
        buy_volume=10.0,
        sell_volume=0.0,
        params_before=params_after,
        kind="hurdle_lognormal",
    ).nll
    assert before_nll is not None
    assert after_nll is not None
    assert before_nll != after_nll


def test_global_model_first_observation_skips_via_score_frame() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
    from mm.core.models.strategy import StrategyInputFrame
    from mm.models.scoring import score_frame_models
    from mm.mpc.mpc_state import OrderbookRelative, State

    global_price_model = GlobalGaussianPriceModel(decay=0.9, sigma_min_quote=1.0)
    book = RawOrderbook(
        exchange="wallex",
        symbol="BTC_IRT",
        timestamp_ms=1000,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    global_event = MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )
    local_book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=1000,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    local = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=local_book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )
    frame = StrategyInputFrame(local=local, global_=global_event, global_status="ready")
    state = State(
        mid=100.5,
        ob_rel=OrderbookRelative(bids=(), asks=()),
        buy_vol=0.0,
        sell_vol=0.0,
        inventory=0.0,
        cash=0.0,
        buy_vwap=100.0,
        sell_vwap=101.0,
        global_buy_vwap=100.0,
        global_sell_vwap=101.0,
    )
    scores = score_frame_models(
        frame=frame,
        previous_frame=None,
        state=state,
        local_market_flow_model=None,
        local_price_model=TorchResidualPriceModel(),
        global_market_flow_model=None,
        global_price_model=global_price_model,
        local_market_flow_kind="hurdle_lognormal",
        local_price_kind="residual",
        global_market_flow_kind="hurdle_lognormal",
        global_price_kind="ewma_gaussian",
    )
    assert scores["local_price"].skip_reason == "missing_previous_frame"
    assert scores["global_price"].skip_reason == "missing_previous_frame"
