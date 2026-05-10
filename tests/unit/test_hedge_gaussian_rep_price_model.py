from __future__ import annotations

import numpy as np
import pytest

from mm.mpc.hedge_price_torch import HedgeGaussianPriceModel


def test_hedge_gaussian_rep_ewma_and_sampling_are_stable() -> None:
    m = HedgeGaussianPriceModel(decay=0.9, sigma_min_quote=0.5)
    for buy, sell in [(100.0, 99.0), (101.0, 98.5), (100.5, 99.2)]:
        m.update(buy, sell)
    p = m.get_params()
    assert p["mu_delta_buy"] == pytest.approx(0.85)
    assert p["mu_delta_sell"] == pytest.approx(-0.38)
    assert p["sigma_delta_buy"] == pytest.approx(0.5)
    assert p["sigma_delta_sell"] == pytest.approx(0.5)
    assert p["decay"] == pytest.approx(0.9)

    np.random.seed(0)
    d1_b, d1_s = m.sample_deltas((3, 2))
    np.random.seed(0)
    d2_b, d2_s = m.sample_deltas((3, 2))
    np.testing.assert_array_almost_equal(d1_b, d2_b)
    np.testing.assert_array_almost_equal(d1_s, d2_s)


def test_hedge_gaussian_rep_accepts_legacy_sigma_min_ticks_alias() -> None:
    m = HedgeGaussianPriceModel(decay=0.9, sigma_min_ticks=0.25)
    p = m.get_params()
    assert p["sigma_min_quote"] == 0.25
