from __future__ import annotations

import math

import pytest

from mm.models.scoring import (
    score_hedge_vwap_deltas,
    score_market_flow,
    score_primary_mid_delta,
)
from mm.mpc.hedge_price_torch import HedgeGaussianPriceModel
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


def test_primary_mid_delta_nll_deterministic() -> None:
    params = {"mu_mid": 1.0, "sigma_mid": 2.0}
    result = score_primary_mid_delta(delta_mid=3.0, params_before=params)
    assert result.skip_reason is None
    assert result.nll is not None
    z = (3.0 - 1.0) / 2.0
    expected_ll = -0.5 * math.log(2 * math.pi) - math.log(2.0) - 0.5 * z * z
    assert result.log_likelihood == pytest.approx(expected_ll)
    assert result.standardized_residual == pytest.approx(z)


def test_primary_mid_delta_skips_invalid_sigma() -> None:
    result = score_primary_mid_delta(delta_mid=1.0, params_before={"mu_mid": 0.0, "sigma_mid": 0.0})
    assert result.skip_reason == "invalid_sigma"
    assert result.nll is None


def test_hedge_vwap_delta_nll_deterministic() -> None:
    params = {
        "mu_delta_buy": 0.5,
        "mu_delta_sell": -0.25,
        "sigma_delta_buy": 1.0,
        "sigma_delta_sell": 2.0,
    }
    result = score_hedge_vwap_deltas(delta_buy=1.5, delta_sell=0.0, params_before=params)
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


def test_hedge_model_first_observation_skips_via_score_frame() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
    from mm.core.models.strategy import StrategyInputFrame
    from mm.models.scoring import score_frame_models
    from mm.mpc.mpc_state import OrderbookRelative, State

    hedge = HedgeGaussianPriceModel(decay=0.9, sigma_min_quote=1.0)
    book = RawOrderbook(
        exchange="wallex",
        symbol="BTC_IRT",
        timestamp_ms=1000,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    hedge_event = MarketEvent(
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
    primary_book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=1000,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    primary = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=primary_book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )
    frame = StrategyInputFrame(primary=primary, hedge=hedge_event, hedge_status="ready")
    state = State(
        mid=100.5,
        ob_rel=OrderbookRelative(bids=(), asks=()),
        buy_vol=0.0,
        sell_vol=0.0,
        inventory=0.0,
        cash=0.0,
        buy_vwap=100.0,
        sell_vwap=101.0,
        hedge_buy_vwap=100.0,
        hedge_sell_vwap=101.0,
    )
    scores = score_frame_models(
        frame=frame,
        previous_frame=None,
        state=state,
        market_flow_model=None,
        price_model=TorchResidualPriceModel(),
        hedge_price_model=hedge,
        market_flow_kind="hurdle_lognormal",
        price_kind="residual",
        hedge_price_kind="legacy",
    )
    assert scores["price"].skip_reason == "missing_previous_frame"
    assert scores["hedge_price"].skip_reason == "missing_previous_frame"
