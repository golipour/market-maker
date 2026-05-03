"""
Order-flow log-loss testers: score market order flow and price models online.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from trade_stats import Orderbook

from market_making_controller.models import (
    Step,
    TorchDiscreteGammaModel,
    TorchHurdleLogNormalModel,
)
from market_making_controller.models.market_order_flow_torch import (
    _discrete_gamma_pmf,
    _discrete_weibull_pmf,
    _hurdle_gamma_pmf,
    _hurdle_weibull_pmf,
    _hurdle_lognormal_pmf,
    _hurdle_logt_pmf,
    _hurdle_generalized_pareto_pmf,
)


@runtime_checkable
class MarketOrderFlowModel(Protocol):
    """Market order flow model: update(prev_step, curr_step); get_plot_data()."""

    def update(self, prev_step: Step | None, curr_step: Step, **kwargs: Any) -> None:
        ...

    def get_plot_data(self) -> dict:
        ...


@runtime_checkable
class PriceModel(Protocol):
    """Price model: update(prev_step, curr_step); get_params(); get_plot_data()."""

    def update(self, prev_step: Step | None, curr_step: Step, **kwargs: Any) -> None:
        ...

    def get_params(self) -> dict:
        ...

    def get_plot_data(self) -> dict:
        ...


@runtime_checkable
class MarketOrderFlowTester(Protocol):
    """Tests a market order flow model online given current params."""

    def update(self, prev_step: Step | None, curr_step: Step, params: dict, **kwargs: Any) -> None:
        ...

    def get_plot_data(self) -> dict:
        """Data for the tester subplot, same contract as MarketOrderFlowModel.get_plot_data()."""
        ...


def _nll_market_order_flow_gamma(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under discrete Gamma (shape/scale per side)."""
    shape_buy = float(params.get("shape_buy") or 1.0)
    scale_buy = float(params.get("scale_buy") or 1.0)
    shape_sell = float(params.get("shape_sell") or 1.0)
    scale_sell = float(params.get("scale_sell") or 1.0)
    prob_buy = _discrete_gamma_pmf(x_buy, shape_buy, scale_buy)
    prob_sell = _discrete_gamma_pmf(x_sell, shape_sell, scale_sell)
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_weibull(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under discrete Weibull (wb_k, wb_lambda per side)."""
    k_buy = float(params.get("wb_k_buy") or 2.0)
    lam_buy = float(params.get("wb_lambda_buy") or 1.0)
    k_sell = float(params.get("wb_k_sell") or k_buy)
    lam_sell = float(params.get("wb_lambda_sell") or lam_buy)
    prob_buy = _discrete_weibull_pmf(x_buy, max(1e-9, k_buy), max(1e-9, lam_buy))
    prob_sell = _discrete_weibull_pmf(x_sell, max(1e-9, k_sell), max(1e-9, lam_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_hurdle_weibull(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL under hurdle Weibull (p0, wb_k, wb_lambda per side)."""
    p0_buy = float(params.get("p0_buy") or 0.0)
    k_buy = float(params.get("wb_k_buy") or 2.0)
    lam_buy = float(params.get("wb_lambda_buy") or 1.0)
    p0_sell = float(params.get("p0_sell") or 0.0)
    k_sell = float(params.get("wb_k_sell") or k_buy)
    lam_sell = float(params.get("wb_lambda_sell") or lam_buy)
    prob_buy = _hurdle_weibull_pmf(x_buy, p0_buy, max(1e-9, k_buy), max(1e-9, lam_buy))
    prob_sell = _hurdle_weibull_pmf(x_sell, p0_sell, max(1e-9, k_sell), max(1e-9, lam_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_hurdle_gamma(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under hurdle gamma (p0, shape, scale per side)."""
    p0_buy = float(params.get("p0_buy") or 0.0)
    shape_buy = float(params.get("shape_buy") or 1.0)
    scale_buy = float(params.get("scale_buy") or 1.0)
    p0_sell = float(params.get("p0_sell") or 0.0)
    shape_sell = float(params.get("shape_sell") or 1.0)
    scale_sell = float(params.get("scale_sell") or 1.0)
    prob_buy = _hurdle_gamma_pmf(x_buy, p0_buy, max(1e-9, shape_buy), max(1e-9, scale_buy))
    prob_sell = _hurdle_gamma_pmf(x_sell, p0_sell, max(1e-9, shape_sell), max(1e-9, scale_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_hurdle_lognormal(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under hurdle lognormal (p0, mu, sigma per side)."""
    p0_buy = float(params.get("p0_buy") or 0.0)
    mu_buy = float(params.get("mu_buy") or 2.0)
    sigma_buy = float(params.get("sigma_buy") or 1.0)
    p0_sell = float(params.get("p0_sell") or 0.0)
    mu_sell = float(params.get("mu_sell") or 2.0)
    sigma_sell = float(params.get("sigma_sell") or 1.0)
    prob_buy = _hurdle_lognormal_pmf(x_buy, p0_buy, mu_buy, max(1e-9, sigma_buy))
    prob_sell = _hurdle_lognormal_pmf(x_sell, p0_sell, mu_sell, max(1e-9, sigma_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_hurdle_logt(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under hurdle log-Student-t (p0, df, mu, sigma per side)."""
    df = float(params.get("logt_df") or 5.0)
    p0_buy = float(params.get("p0_buy") or 0.0)
    mu_buy = float(params.get("mu_buy") or 2.0)
    sigma_buy = float(params.get("sigma_buy") or 1.0)
    p0_sell = float(params.get("p0_sell") or 0.0)
    mu_sell = float(params.get("mu_sell") or 2.0)
    sigma_sell = float(params.get("sigma_sell") or 1.0)
    prob_buy = _hurdle_logt_pmf(x_buy, p0_buy, max(1e-6, df), mu_buy, max(1e-9, sigma_buy))
    prob_sell = _hurdle_logt_pmf(x_sell, p0_sell, max(1e-6, df), mu_sell, max(1e-9, sigma_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow_hurdle_gpd(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Per-observation NLL for (x_buy, x_sell) under hurdle generalized Pareto."""
    p0_buy = float(params.get("p0_buy") or 0.0)
    xi_buy = float(params.get("xi_buy") or 0.1)
    beta_buy = float(params.get("beta_buy") or 3.0)
    u_buy = float(params.get("u_buy") or 1.0)
    p0_sell = float(params.get("p0_sell") or 0.0)
    xi_sell = float(params.get("xi_sell") or 0.1)
    beta_sell = float(params.get("beta_sell") or 3.0)
    u_sell = float(params.get("u_sell") or 1.0)
    prob_buy = _hurdle_generalized_pareto_pmf(x_buy, p0_buy, xi_buy, max(1e-9, beta_buy), max(1e-9, u_buy))
    prob_sell = _hurdle_generalized_pareto_pmf(x_sell, p0_sell, xi_sell, max(1e-9, beta_sell), max(1e-9, u_sell))
    prob_buy = max(1e-12, min(1.0, float(prob_buy)))
    prob_sell = max(1e-12, min(1.0, float(prob_sell)))
    return float(-np.log(prob_buy)), float(-np.log(prob_sell))


def _nll_market_order_flow(params: dict, x_buy: int, x_sell: int) -> tuple[float, float]:
    """Dispatch to Gamma / Weibull / hurdle gamma / hurdle lognormal / hurdle log-t / hurdle GPD NLL."""
    if "xi_buy" in params or "beta_buy" in params or "u_buy" in params:
        return _nll_market_order_flow_hurdle_gpd(params, x_buy, x_sell)
    if "p0_buy" in params and ("shape_buy" in params or "scale_buy" in params):
        return _nll_market_order_flow_hurdle_gamma(params, x_buy, x_sell)
    if "p0_buy" in params and ("wb_k_buy" in params or "wb_lambda_buy" in params):
        return _nll_market_order_flow_hurdle_weibull(params, x_buy, x_sell)
    if "wb_k_buy" in params or "wb_lambda_buy" in params:
        return _nll_market_order_flow_weibull(params, x_buy, x_sell)
    if "logt_df" in params:
        return _nll_market_order_flow_hurdle_logt(params, x_buy, x_sell)
    if "p0_buy" in params or "mu_buy" in params:
        return _nll_market_order_flow_hurdle_lognormal(params, x_buy, x_sell)
    return _nll_market_order_flow_gamma(params, x_buy, x_sell)


class GeneralOrderFlowLogLossTester:
    """
    EWMA log-loss tester for market order flow (Gamma / Weibull / hurdle gamma / hurdle Weibull / hurdle lognormal / hurdle log-t / hurdle GPD). Uses params from
    the previous step to score the current (buy_vol, sell_vol). Dispatches on estimator params.
    """

    def __init__(self, model: MarketOrderFlowModel, decay: float = 0.99):
        self.model = model
        gamma = max(1e-6, min(1.0, float(decay)))
        self._gamma = gamma
        self._S_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_sell: float = 0.0
        self._W_sell: float = 0.0
        self.nll_buy_hist: list[float] = []
        self.nll_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, params: dict, **_: Any) -> None:
        _t_end, ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        nll_buy, nll_sell = _nll_market_order_flow(params, x_buy, x_sell)
        self._S_buy = self._gamma * self._S_buy + nll_buy
        self._W_buy = self._gamma * self._W_buy + 1.0
        self._S_sell = self._gamma * self._S_sell + nll_sell
        self._W_sell = self._gamma * self._W_sell + 1.0
        score_buy = self._S_buy / self._W_buy if self._W_buy > 1e-12 else 0.0
        score_sell = self._S_sell / self._W_sell if self._W_sell > 1e-12 else 0.0
        self.nll_buy_hist.append(score_buy)
        self.nll_sell_hist.append(score_sell)

    def get_plot_specs(self) -> list[dict]:
        """One subplot for market order flow NLL (buy/sell series)."""
        return [
            {"title": "EWMA −log p(market order flow | model)", "ylabel": "EWMA −log p(x)"},
        ]

    def get_plot_data(self) -> dict:
        series = [
            {"label": "Buy NLL", "color": "green", "y": self.nll_buy_hist},
            {"label": "Sell NLL", "color": "red", "y": self.nll_sell_hist},
        ]
        return {
            "title": f"EWMA −log p(market order flow | model), decay γ={self._gamma}",
            "ylabel": "EWMA −log p(x)",
            "series": series,
            "subplots": [{"title": "EWMA −log p(market order flow | model)", "ylabel": "EWMA −log p(x)", "series": series}],
        }


# -----------------------------------------------------------------------------
# Price (Δmid) log-loss tester
# -----------------------------------------------------------------------------


def _gaussian_nll(observed_delta: float, mu: float, sigma: float) -> float:
    """Negative log-likelihood of observed_delta under N(mu, sigma^2)."""
    sigma = max(1e-12, float(sigma))
    z = (observed_delta - mu) / sigma
    return 0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)


@runtime_checkable
class PriceTester(Protocol):
    """Tests a price (Δmid) model online given params and observed mid."""

    def update(self, prev_step: Step | None, curr_step: Step, params: dict, **kwargs: Any) -> None:
        ...

    def get_plot_data(self) -> dict:
        ...


class PriceLogLossTester:
    """
    EWMA log-loss tester for price models. Scores the observation under the model's N(μ, σ²).
    - For Δmid models: observation = mid_now - last_mid; scores −log p(Δmid | μ, σ²).
    - For residual (flow-implied) models: observation = get_last_residual(); scores −log p(residual | μ, σ²).
    Uses params from the previous step (mu_mid, sigma_mid).
    """

    def __init__(self, price_model: PriceModel, decay: float = 0.99):
        self.price_model = price_model
        gamma = max(1e-6, min(1.0, float(decay)))
        self._gamma = gamma
        self._last_mid: float | None = None
        self._S: float = 0.0
        self._W: float = 0.0
        self.nll_hist: list[float] = []
        self._use_residual: bool = False  # set from price model when present

    def update(self, prev_step: Step | None, curr_step: Step, params: dict, **_: Any) -> None:
        _t_end, ob, buy_vol, sell_vol, *_ = curr_step
        mid_now = float(ob.mid)
        use_residual = hasattr(self.price_model, "sample_residual") and hasattr(
            self.price_model, "get_last_residual"
        )
        if use_residual:
            residual = getattr(self.price_model, "get_last_residual", lambda: None)()
            if residual is None:
                return
            observed = float(residual)
            self._use_residual = True
        else:
            if self._last_mid is None:
                self._last_mid = mid_now
                return
            observed = mid_now - self._last_mid
            self._last_mid = mid_now
            self._use_residual = False
        mu = float(params.get("mu_mid", 0.0))
        sigma = float(params.get("sigma_mid", 1e-4))
        nll = _gaussian_nll(observed, mu, max(1e-12, sigma))
        self._S = self._gamma * self._S + nll
        self._W = self._gamma * self._W + 1.0
        score = self._S / self._W if self._W > 1e-12 else 0.0
        self.nll_hist.append(score)

    def get_plot_data(self) -> dict:
        if self._use_residual or hasattr(self.price_model, "sample_residual"):
            title = f"EWMA −log p(residual | price model), decay γ={self._gamma}"
            ylabel = "EWMA −log p(residual)"
        else:
            title = f"EWMA −log p(Δmid | price model), decay γ={self._gamma}"
            ylabel = "EWMA −log p(Δmid)"
        return {
            "title": title,
            "ylabel": ylabel,
            "series": [
                {"label": "NLL", "color": "blue", "y": self.nll_hist},
            ],
        }
