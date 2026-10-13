"""Unit tests for Hurdle Price models (Student-t and LogNormal) and regularized flow models."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mm.mpc.market_order_flow_torch import (
    TorchHurdleLogNormalModel,
    TorchHurdleLogStudentTModel,
)
from mm.mpc.price_torch import (
    TorchHurdleLogNormalPriceModel,
    TorchHurdleStudentTPriceModel,
    TorchResidualPriceModel,
)
from mm.models.scoring import score_local_mid_delta, score_market_flow
from mm.strategies.mpc_engine import _build_price_model


def _mock_step(
    mid: float,
    *,
    t_end_ms: int = 1000,
    buy_vol: float = 0.0,
    sell_vol: float = 0.0,
    vwap_buy: float | None = None,
    vwap_sell: float | None = None,
) -> tuple:
    half_spread = 0.5
    bid = mid - half_spread
    ask = mid + half_spread
    book = SimpleNamespace(
        mid=mid,
        bids=[SimpleNamespace(price=bid, qty=10.0)],
        asks=[SimpleNamespace(price=ask, qty=10.0)],
    )
    return (
        t_end_ms,
        book,
        buy_vol,
        sell_vol,
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


class TestTorchHurdleStudentTPriceModel:
    def test_pmf_discrete_lattice_sums_to_one(self) -> None:
        model = TorchHurdleStudentTPriceModel(
            student_df=2.5,
            tick_size=0.5,
            init_p0=0.6,
            init_sigma=1.2,
        )
        mid = 100.0
        k_range = np.arange(-5000, 5001)
        next_mids = mid + k_range * 0.5

        probs = [model.prob_mid(nm, mid) for nm in next_mids]
        total_p = sum(probs)
        assert abs(total_p - 1.0) < 1e-4
        assert model.prob_mid(mid, mid) == pytest.approx(0.6, abs=1e-6)

    def test_update_tracks_zero_frequency_and_scale(self) -> None:
        model = TorchHurdleStudentTPriceModel(
            student_df=2.5,
            decay=0.8,
            tick_size=0.5,
            init_p0=0.5,
            init_sigma=1.0,
        )
        s0 = _mock_step(100.0, t_end_ms=1000)
        s1 = _mock_step(100.0, t_end_ms=2000)
        s2 = _mock_step(100.0, t_end_ms=3000)

        model.update(s0, s1)
        model.update(s1, s2)
        params = model.get_params()
        assert params["kind"] == "hurdle_student_t"
        assert params["p0"] > 0.5  # Zero frequency climbed

        # Jump by 10 ticks
        s3 = _mock_step(105.0, t_end_ms=4000)
        model.update(s2, s3)
        params_after_jump = model.get_params()
        assert params_after_jump["p0"] < params["p0"]
        assert params_after_jump["scale"] > 0.0

    def test_sample_residual_shape_and_device(self) -> None:
        model = TorchHurdleStudentTPriceModel(
            student_df=2.5,
            tick_size=0.5,
            init_p0=0.7,
            init_sigma=1.5,
        )
        eps_buy, eps_sell = model.sample_residual((1000,), dtype=torch.float64)
        assert eps_buy.shape == (1000,)
        assert eps_sell.shape == (1000,)
        assert eps_buy.dtype == torch.float64

        # Monte Carlo zero fraction test
        zeros_buy = float((eps_buy == 0.0).sum()) / 1000.0
        assert 0.60 <= zeros_buy <= 0.80

    def test_sample_next_mid_tick_lattice_alignment(self) -> None:
        model = TorchHurdleStudentTPriceModel(
            student_df=2.5,
            tick_size=0.5,
            init_p0=0.3,
            init_sigma=2.0,
        )
        rng = np.random.default_rng(42)
        mid = 100.0
        for _ in range(100):
            next_m = model.sample_next_mid(mid, rng=rng)
            diff = next_m - mid
            # Must be integer multiple of tick size 0.5
            k = diff / 0.5
            assert abs(k - round(k)) < 1e-6


class TestTorchHurdleLogNormalPriceModel:
    def test_pmf_discrete_lattice_sums_to_one(self) -> None:
        model = TorchHurdleLogNormalPriceModel(
            tick_size=100.0,
            init_p0=0.5,
            init_mu=5.0,
            init_sigma=1.0,
        )
        mid = 50_000.0
        k_range = np.arange(-10_000, 10_001)
        next_mids = mid + k_range * 100.0

        probs = [model.prob_mid(nm, mid) for nm in next_mids]
        total_p = sum(probs)
        assert abs(total_p - 1.0) < 1e-4
        assert model.prob_mid(mid, mid) == pytest.approx(0.5, abs=1e-6)

    def test_sample_residual_tensor(self) -> None:
        model = TorchHurdleLogNormalPriceModel(
            tick_size=100.0,
            init_p0=0.8,
            init_mu=4.0,
            init_sigma=0.5,
        )
        eps_buy, eps_sell = model.sample_residual((500,), dtype=torch.float64)
        assert eps_buy.shape == (500,)
        assert eps_sell.shape == (500,)
        zeros_buy = float((eps_buy == 0.0).sum()) / 500.0
        assert 0.70 <= zeros_buy <= 0.90


class TestFlowVarianceRegularizationFloor:
    def test_repeated_identical_trades_do_not_collapse_variance(self) -> None:
        model = TorchHurdleLogNormalModel(
            decay=0.9,
            init_p0=0.2,
            init_mu=1.0,
            init_sigma=0.8,
            sigma_floor=0.35,
        )
        # Feed 20 identical fills of 5 lots
        step = (1000, None, 5.0, 5.0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0)
        for _ in range(20):
            model.update(None, step)

        params = model.get_params()
        # Even with zero empirical sample variance, sigma is floored at 0.35
        assert params["sigma_buy"] >= 0.35
        assert params["sigma_sell"] >= 0.35

        # Score on a non-identical fill (e.g. 15 lots) remains well-behaved
        score = score_market_flow(
            buy_volume=15.0,
            sell_volume=5.0,
            params_before=params,
            kind="hurdle_lognormal",
        )
        assert score.skip_reason is None
        assert score.nll is not None
        assert math.isfinite(score.nll)
        assert score.nll < 50.0  # Does not explode to astronomical values

    def test_log_student_t_model_variance_floor(self) -> None:
        model = TorchHurdleLogStudentTModel(
            decay=0.9,
            init_p0=0.2,
            init_mu=1.0,
            init_sigma=0.8,
            init_df=4.0,
            sigma_floor=0.35,
        )
        step = (1000, None, 10.0, 10.0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0)
        for _ in range(15):
            model.update(None, step)
        params = model.get_params()
        assert params["sigma_buy"] >= 0.35
        assert params["sigma_sell"] >= 0.35


class TestFactoryAndScoringIntegration:
    def test_mpc_engine_factory_builds_hurdle_models(self) -> None:
        flow = TorchHurdleLogNormalModel()
        price_t = _build_price_model(
            {"kind": "hurdle_student_t", "student_df": 2.5, "tick_size": 0.5},
            market_flow_model=flow,
        )
        assert isinstance(price_t, TorchHurdleStudentTPriceModel)

        price_ln = _build_price_model(
            {"kind": "hurdle_lognormal", "tick_size": 100.0},
            market_flow_model=flow,
        )
        assert isinstance(price_ln, TorchHurdleLogNormalPriceModel)

        price_res = _build_price_model(
            {"kind": "residual"},
            market_flow_model=flow,
        )
        assert isinstance(price_res, TorchResidualPriceModel)

    def test_factory_accepts_legacy_mixture_params_without_error(self) -> None:
        flow = TorchHurdleLogNormalModel()
        legacy_cfg = {
            "kind": "hurdle_student_t",
            "decay": 0.99,
            "point_mode": "sample",
            "student_df": 3,
            "shape_alpha": 0.5,
            "initial_sigma": 50,
            "initial_log_sigma": 0.25,
            "reference_vwap_vol": 1000,
            "tick_size": 0.5,
        }
        model = _build_price_model(legacy_cfg, market_flow_model=flow)
        assert isinstance(model, TorchHurdleStudentTPriceModel)
        params = model.get_params()
        assert params["kind"] == "hurdle_student_t"
        assert params["student_df"] == 3.0

    def test_score_local_mid_delta_evaluates_hurdle_models(self) -> None:
        # Score zero change on hurdle student-t
        p_t = {
            "kind": "hurdle_student_t",
            "p0": 0.7,
            "student_df": 2.5,
            "scale": 1.0,
            "tick_size": 0.5,
        }
        res_zero = score_local_mid_delta(delta_mid=0.0, params_before=p_t)
        assert res_zero.skip_reason is None
        assert res_zero.observation["prob"] == pytest.approx(0.7, abs=1e-6)
        assert res_zero.nll == pytest.approx(-math.log(0.7), abs=1e-6)

        # Score jump of 1 tick
        res_one = score_local_mid_delta(delta_mid=0.5, params_before=p_t)
        assert res_one.skip_reason is None
        assert res_one.observation["prob"] > 0.0
        assert math.isfinite(res_one.nll)
