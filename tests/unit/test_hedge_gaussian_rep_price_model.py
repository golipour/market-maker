from __future__ import annotations

import numpy as np

from mm.mpc.hedge_price_torch import HedgeGaussianPriceModel
from mm_runtime.order_flow_stepper import HedgeGaussianPriceModel as LegacyHedgeGaussianPriceModel


def test_hedge_gaussian_rep_matches_legacy_ewma_and_sampling() -> None:
    m = HedgeGaussianPriceModel(decay=0.9, sigma_min_quote=0.5)
    leg = LegacyHedgeGaussianPriceModel(decay=0.9, sigma_min_ticks=0.5)
    for buy, sell in [(100.0, 99.0), (101.0, 98.5), (100.5, 99.2)]:
        m.update(buy, sell)
        leg.update(buy, sell)
    legacy_keys = ("mu_delta_buy", "sigma_delta_buy", "mu_delta_sell", "sigma_delta_sell", "decay")
    m_params = m.get_params()
    l_params = leg.get_params()
    assert {k: m_params[k] for k in legacy_keys} == {k: l_params[k] for k in legacy_keys}
    rng = np.random.default_rng(0)
    np.random.seed(0)
    d1_b, d1_s = m.sample_deltas((3, 2))
    np.random.seed(0)
    d2_b, d2_s = leg.sample_deltas((3, 2))
    np.testing.assert_array_almost_equal(d1_b, d2_b)
    np.testing.assert_array_almost_equal(d1_s, d2_s)


def test_hedge_gaussian_rep_accepts_legacy_sigma_min_ticks_alias() -> None:
    m = HedgeGaussianPriceModel(decay=0.9, sigma_min_ticks=0.25)
    p = m.get_params()
    assert p["sigma_min_quote"] == 0.25
