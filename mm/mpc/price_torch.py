"""
Torch-based price models: Gaussian Δmid (GaussianPriceModel) and two-sided residual
noise around MOF-only VWAPs (TorchResidualPriceModel).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mm.mpc.torch_device import resolve_torch_device
from mm.mpc.step import Step
from mm.mpc_utils import exp_clamped as _exp_clamped
from mm.mpc_utils import expit as _expit
from mm.mpc_utils import log_clamped as _log_clamped
from mm.mpc_utils import log_normal_pdf as _log_normal_pdf
from mm.mpc_utils import log_student_t_pdf as _log_student_t_pdf
from mm.mpc_utils import logaddexp as _logaddexp
from mm.mpc_utils import logsumexp_pair as _logsumexp_pair

if TYPE_CHECKING:
    pass


@runtime_checkable
class PriceModel(Protocol):
    """Sample next mid given current market state and optional params; optional P(mid_next)."""

    def sample_next_mid(
        self,
        mid: float,
        rng: np.random.Generator | None = None,
        params: dict | None = None,
    ) -> float:
        ...

    def prob_mid(
        self,
        mid_next: float,
        mid: float,
        params: dict | None = None,
    ) -> float:
        """Return P(mid_next | mid, params). Default 0.0 if not implemented."""
        ...


class GaussianPriceModel:
    """
    m_{t+1} = m_t + mu + epsilon, epsilon ~ N(0, sigma²).
    When params has mu_mid, sigma_mid (from price model), use those; else use
    default_sigma with scale.
    """

    def __init__(self, scale: float = 1.0, default_sigma: float = 1e-4):
        self.scale = scale
        self.default_sigma = default_sigma

    def sample_next_mid(
        self,
        mid: float,
        rng: np.random.Generator | None = None,
        params: dict | None = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        if params is not None and ("mu_mid" in params or "sigma_mid" in params):
            mu = float(params.get("mu_mid", 0.0))
            sigma = float(params.get("sigma_mid", self.default_sigma))
            sigma = max(1e-12, sigma)
            eps = rng.normal(0.0, sigma)
            return int(round(mid + mu + eps))
        sigma = self.default_sigma * self.scale
        sigma = max(1e-12, sigma)
        eps = rng.normal(0.0, sigma)
        return int(round(mid + eps))

    def prob_mid(
        self,
        mid_next: float,
        mid: float,
        params: dict | None = None,
    ) -> float:
        """P(mid_next | mid, params) for discrete outcome: P(round(mid+mu+eps) = mid_next)."""
        from scipy.stats import norm
        if params is not None and ("mu_mid" in params or "sigma_mid" in params):
            mu = float(params.get("mu_mid", 0.0))
            sigma = float(params.get("sigma_mid", self.default_sigma))
        else:
            mu = 0.0
            sigma = self.default_sigma * self.scale
        sigma = max(1e-12, sigma)
        mean = mid + mu
        low = (mid_next - 0.5 - mean) / sigma
        high = (mid_next + 0.5 - mean) / sigma
        return float(norm.cdf(high) - norm.cdf(low))


def _expected_buy_sell_vol_from_params(params: dict) -> tuple[float, float]:
    """
    Closed-form or MC E[volume] from fitted params (Gamma, Weibull, hurdle Gamma, hurdle Weibull, hurdle GPD, hurdle log-t, hurdle lognormal).
    For TorchHurdleLogNormalModel.sample(), this is the exact mean of the continuous
    hurdle–LogNormal graph; repeated sample() averaging converges here (see OrderFlowRunner
    for diagnostic MC means).
    """
    if "p0_buy" in params and ("shape_buy" in params or "scale_buy" in params):
        p0_buy = float(params.get("p0_buy", params.get("p_buy", 0.5)))
        p0_sell = float(params.get("p0_sell", params.get("p_sell", 0.5)))
        shape_buy = float(params.get("shape_buy", 1.0))
        scale_buy = float(params.get("scale_buy", 1.0))
        shape_sell = float(params.get("shape_sell", shape_buy))
        scale_sell = float(params.get("scale_sell", scale_buy))
        e_pos_buy = shape_buy * scale_buy
        e_pos_sell = shape_sell * scale_sell
        e_buy = (1.0 - p0_buy) * e_pos_buy
        e_sell = (1.0 - p0_sell) * e_pos_sell
        return (float(e_buy), float(e_sell))
    if "p0_buy" in params and ("wb_k_buy" in params or "wb_lambda_buy" in params):
        from scipy.special import gamma as gamma_fn

        p0_buy = float(params.get("p0_buy", params.get("p_buy", 0.5)))
        p0_sell = float(params.get("p0_sell", params.get("p_sell", 0.5)))
        k_buy = max(1e-9, float(params.get("wb_k_buy", 2.0)))
        lam_buy = max(1e-9, float(params.get("wb_lambda_buy", 1.0)))
        k_sell = max(1e-9, float(params.get("wb_k_sell", k_buy)))
        lam_sell = max(1e-9, float(params.get("wb_lambda_sell", lam_buy)))
        e_pos_buy = lam_buy * gamma_fn(1.0 + 1.0 / k_buy)
        e_pos_sell = lam_sell * gamma_fn(1.0 + 1.0 / k_sell)
        e_buy = (1.0 - p0_buy) * e_pos_buy
        e_sell = (1.0 - p0_sell) * e_pos_sell
        return (float(e_buy), float(e_sell))
    if "wb_k_buy" in params or "wb_lambda_buy" in params:
        from scipy.special import gamma as gamma_fn

        k_buy = max(1e-9, float(params.get("wb_k_buy", 2.0)))
        lam_buy = max(1e-9, float(params.get("wb_lambda_buy", 1.0)))
        k_sell = max(1e-9, float(params.get("wb_k_sell", k_buy)))
        lam_sell = max(1e-9, float(params.get("wb_lambda_sell", lam_buy)))
        e_buy = lam_buy * gamma_fn(1.0 + 1.0 / k_buy)
        e_sell = lam_sell * gamma_fn(1.0 + 1.0 / k_sell)
        return (float(e_buy), float(e_sell))
    if "shape_buy" in params and "scale_buy" in params:
        e_buy = params["shape_buy"] * params["scale_buy"]
        e_sell = params.get("shape_sell", params["shape_buy"]) * params.get("scale_sell", params["scale_buy"])
        return (float(e_buy), float(e_sell))
    if "xi_buy" in params or "beta_buy" in params or "u_buy" in params:
        p0_buy = float(params.get("p0_buy", params.get("p_buy", 0.5)))
        p0_sell = float(params.get("p0_sell", params.get("p_sell", 0.5)))
        xi_buy = float(params.get("xi_buy", 0.1))
        xi_sell = float(params.get("xi_sell", 0.1))
        beta_buy = max(1e-9, float(params.get("beta_buy", 3.0)))
        beta_sell = max(1e-9, float(params.get("beta_sell", 3.0)))
        u_buy = max(1e-9, float(params.get("u_buy", 1.0)))
        u_sell = max(1e-9, float(params.get("u_sell", 1.0)))
        xi_buy = min(0.99, xi_buy)
        xi_sell = min(0.99, xi_sell)
        e_pos_buy = u_buy + beta_buy / max(1e-9, 1.0 - xi_buy)
        e_pos_sell = u_sell + beta_sell / max(1e-9, 1.0 - xi_sell)
        e_buy = (1.0 - p0_buy) * e_pos_buy
        e_sell = (1.0 - p0_sell) * e_pos_sell
        return (float(e_buy), float(e_sell))
    if "logt_df" in params:
        from scipy.stats import t as t_dist

        p0_buy = float(params.get("p0_buy", params.get("p_buy", 0.5)))
        p0_sell = float(params.get("p0_sell", params.get("p_sell", 0.5)))
        mu_buy = float(params.get("mu_buy", 0.0))
        sigma_buy = max(1e-9, float(params.get("sigma_buy", 1.0)))
        mu_sell = float(params.get("mu_sell", 0.0))
        sigma_sell = max(1e-9, float(params.get("sigma_sell", 1.0)))
        df = max(1.1, float(params["logt_df"]))
        rng = np.random.default_rng(0)
        n = 4096
        z_buy = t_dist.rvs(df, loc=mu_buy, scale=sigma_buy, size=n, random_state=rng)
        z_sell = t_dist.rvs(df, loc=mu_sell, scale=sigma_sell, size=n, random_state=rng)
        e_pos_buy = float(np.mean(np.exp(np.clip(z_buy, -30.0, 30.0))))
        e_pos_sell = float(np.mean(np.exp(np.clip(z_sell, -30.0, 30.0))))
        e_buy = (1.0 - p0_buy) * e_pos_buy
        e_sell = (1.0 - p0_sell) * e_pos_sell
        return (float(e_buy), float(e_sell))
    if "p_buy" in params or "p0_buy" in params:
        p0_buy = params.get("p0_buy", params.get("p_buy", 0.5))
        p0_sell = params.get("p0_sell", params.get("p_sell", 0.5))
        mu_buy = params.get("mu_buy", 0.0)
        sigma_buy = params.get("sigma_buy", 1.0)
        mu_sell = params.get("mu_sell", 0.0)
        sigma_sell = params.get("sigma_sell", 1.0)
        e_pos_buy = np.exp(mu_buy + 0.5 * sigma_buy ** 2)
        e_pos_sell = np.exp(mu_sell + 0.5 * sigma_sell ** 2)
        e_buy = (1.0 - p0_buy) * e_pos_buy
        e_sell = (1.0 - p0_sell) * e_pos_sell
        return (float(e_buy), float(e_sell))
    return (0.0, 0.0)


class TorchResidualPriceModel:
    """
    Two-sided residual noise on VWAPs (MOF-only book evolution).

    Each step compares instantaneous realized buy/sell VWAP on the current book to the
    VWAP on the book obtained by consuming market flow on the previous book. Residuals
    e_buy, e_sell feed an EWMA on μ and σ² only (price levels themselves are not EWMA-smoothed).
    """

    def __init__(
        self,
        market_flow_model: object | None = None,
        limit_order_flow_model: object | None = None,
        residual_mu: float = 0.0,
        residual_sigma: float = 1.0,
        *,
        residual_decay: float = 0.9,
        learn_residual: bool = True,
        sigma_min: float = 1e-6,
        device: str | None = None,
        default_vwap_depth_lots: float = 1000.0,
        vwap_depth: float | None = None,
    ):
        self._init_mu = float(residual_mu)
        self._init_sigma = max(1e-12, float(residual_sigma))
        self._gamma = max(1e-6, min(1.0, float(residual_decay)))
        self._learn_residual = bool(learn_residual)
        self._sigma_min = float(sigma_min)
        self._mu_buy = float(residual_mu)
        self._mu_sell = float(residual_mu)
        self._last_residual: float | None = None
        depth = float(vwap_depth) if vwap_depth is not None else float(default_vwap_depth_lots)
        self._target_depth_lots = max(1.0, depth)
        sigma0 = max(self._sigma_min, float(residual_sigma))
        self._sigma2_buy = float(sigma0 * sigma0)
        self._sigma2_sell = float(sigma0 * sigma0)
        self.residual_mu_hist: list[float] = []
        self.residual_sigma_hist: list[float] = []
        self.mu_buy_hist: list[float] = []
        self.mu_sell_hist: list[float] = []
        self.realized_buy_price_hist: list[float] = []
        self.computed_buy_price_hist: list[float] = []
        self.realized_sell_price_hist: list[float] = []
        self.computed_sell_price_hist: list[float] = []
        self.sigma_buy_hist: list[float] = []
        self.sigma_sell_hist: list[float] = []

    def get_last_residual(self) -> float | None:
        """Average residual (e_buy + e_sell) / 2 from the last ``update``, or None."""
        return getattr(self, "_last_residual", None)

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        if prev_step is None:
            return
        from mm.pricing import reference_prices_from_transition_ticks

        _t_prev, prev_ob, _b_prev, _s_prev, *_ = prev_step
        _t_curr, cur_ob, buy_vol, sell_vol, *_ = curr_step
        buy_real, sell_real, buy_vwap_mof, sell_vwap_mof = reference_prices_from_transition_ticks(
            prev_ob,
            cur_ob,
            float(buy_vol),
            float(sell_vol),
            self._target_depth_lots,
        )
        self.realized_buy_price_hist.append(float(buy_real))
        self.computed_buy_price_hist.append(float(buy_vwap_mof))
        self.realized_sell_price_hist.append(float(sell_real))
        self.computed_sell_price_hist.append(float(sell_vwap_mof))

        e_buy = float(buy_real - buy_vwap_mof)
        e_sell = float(sell_real - sell_vwap_mof)
        self._last_residual = 0.5 * (e_buy + e_sell)
        alpha = 1.0 - self._gamma
        if self._learn_residual:
            mu_buy_new = self._gamma * self._mu_buy + alpha * e_buy
            mu_sell_new = self._gamma * self._mu_sell + alpha * e_sell
            self._sigma2_buy = self._gamma * self._sigma2_buy + alpha * (e_buy - mu_buy_new) ** 2
            self._sigma2_sell = self._gamma * self._sigma2_sell + alpha * (e_sell - mu_sell_new) ** 2
            self._mu_buy = float(mu_buy_new)
            self._mu_sell = float(mu_sell_new)
        sigma_buy = max(self._sigma_min, float(np.sqrt(max(0.0, self._sigma2_buy))))
        sigma_sell = max(self._sigma_min, float(np.sqrt(max(0.0, self._sigma2_sell))))
        self.sigma_buy_hist.append(sigma_buy)
        self.sigma_sell_hist.append(sigma_sell)
        mu_mid = 0.5 * (self._mu_buy + self._mu_sell)
        sigma_mid = float(np.sqrt(max(0.0, 0.5 * (sigma_buy * sigma_buy + sigma_sell * sigma_sell))))
        self.residual_mu_hist.append(mu_mid)
        self.residual_sigma_hist.append(sigma_mid)
        self.mu_buy_hist.append(self._mu_buy)
        self.mu_sell_hist.append(self._mu_sell)

    def get_params(self) -> dict:
        sigma_buy = self.sigma_buy_hist[-1] if self.sigma_buy_hist else max(self._sigma_min, self._init_sigma)
        sigma_sell = self.sigma_sell_hist[-1] if self.sigma_sell_hist else max(self._sigma_min, self._init_sigma)
        mu_buy = self.mu_buy_hist[-1] if self.mu_buy_hist else self._mu_buy
        mu_sell = self.mu_sell_hist[-1] if self.mu_sell_hist else self._mu_sell
        sigma_mid = float(np.sqrt(max(0.0, 0.5 * (sigma_buy * sigma_buy + sigma_sell * sigma_sell))))
        mu_mid = 0.5 * (float(mu_buy) + float(mu_sell))
        return {
            "mu_mid": float(mu_mid),
            "sigma_mid": max(self._sigma_min, sigma_mid),
            "mu_buy": float(mu_buy),
            "mu_sell": float(mu_sell),
            "sigma_buy": max(self._sigma_min, float(sigma_buy)),
            "sigma_sell": max(self._sigma_min, float(sigma_sell)),
        }

    def get_plot_specs(self) -> list[dict]:
        # Sigmas are shown on Figure 4 only; keep main figure (figure 1) free of price-model axes.
        return []

    def get_plot_data(self) -> dict:
        return {
            "title": f"Residual (μ/σ EWMA γ={self._gamma})",
            "ylabel": "ticks",
            "series": [
                {"label": "μ_buy", "color": "cyan", "y": self.mu_buy_hist},
                {"label": "μ_sell", "color": "magenta", "y": self.mu_sell_hist},
                {"label": "σ_buy", "color": "blue", "y": self.sigma_buy_hist},
                {"label": "σ_sell", "color": "red", "y": self.sigma_sell_hist},
            ],
            "subplots": [
                {"title": "μ_buy (ticks)", "ylabel": "μ", "series": [{"label": "μ_buy", "color": "cyan", "y": self.mu_buy_hist}]},
                {"title": "μ_sell (ticks)", "ylabel": "μ", "series": [{"label": "μ_sell", "color": "magenta", "y": self.mu_sell_hist}]},
                {"title": "σ_buy (ticks)", "ylabel": "σ", "series": [{"label": "σ_buy", "color": "blue", "y": self.sigma_buy_hist}]},
                {"title": "σ_sell (ticks)", "ylabel": "σ", "series": [{"label": "σ_sell", "color": "red", "y": self.sigma_sell_hist}]},
            ],
        }

    def sample_residual(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample independent Gaussian residuals for buy/sell VWAPs."""
        import torch
        import torch.distributions as dist
        p = self.get_params()
        mu_buy = float(p.get("mu_buy", 0.0))
        mu_sell = float(p.get("mu_sell", 0.0))
        sigma_buy = max(1e-12, float(p.get("sigma_buy", p.get("sigma_mid", 1.0))))
        sigma_sell = max(1e-12, float(p.get("sigma_sell", p.get("sigma_mid", 1.0))))
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        db = dist.Normal(torch.tensor(mu_buy, dtype=_dtype, device=_device), torch.tensor(sigma_buy, dtype=_dtype, device=_device))
        ds = dist.Normal(torch.tensor(mu_sell, dtype=_dtype, device=_device), torch.tensor(sigma_sell, dtype=_dtype, device=_device))
        return db.sample(sample_shape), ds.sample(sample_shape)

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample pooled residual ~ N(μ_mid, σ_mid²) from get_params()."""
        import torch
        import torch.distributions as dist
        p = self.get_params()
        mu = float(p["mu_mid"])
        sigma = max(1e-12, float(p["sigma_mid"]))
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        d = dist.Normal(
            torch.tensor(mu, dtype=_dtype, device=_device),
            torch.tensor(sigma, dtype=_dtype, device=_device),
        )
        return d.sample(sample_shape)


class TorchConditionalMechanicalMixturePriceModel:
    """
    Conditional book-state price model for MPC rollouts.

    The model learns fixed-depth mid/spread/shape deltas from observed transitions,
    using market-order-flow-only mechanical deltas as features. Runtime valuation
    converts the predicted fixed-depth shape to dynamic-depth VWAP analytically.
    """

    _component_names = ("mid", "log_spread", "log_buy_shape", "log_sell_shape")

    def __init__(
        self,
        market_flow_model: object | None = None,
        *,
        reference_vwap_vol: float = 100.0,
        shape_alpha: float = 0.5,
        decay: float = 0.99,
        initial_sigma: float = 50.0,
        initial_log_sigma: float = 0.25,
        sigma_min: float = 1e-6,
        student_df: float = 3.0,
        point_mode: str = "sample",
        rls_p0: float = 1_000_000.0,
        param_eps: float = 1e-9,
        device: str | None = None,
    ):
        self.market_flow_model = market_flow_model
        self._reference_vwap_vol = max(1e-12, float(reference_vwap_vol))
        self._target_depth_lots = self._reference_vwap_vol
        self._shape_alpha = max(0.0, float(shape_alpha))
        self._gamma = max(1e-6, min(0.9999, float(decay)))
        self._alpha = 1.0 - self._gamma
        self._sigma_min = max(1e-12, float(sigma_min))
        self._student_df = max(2.1, float(student_df))
        self._point_mode = str(point_mode or "sample").lower().strip()
        self._param_eps = max(1e-12, float(param_eps))
        self._device = device
        sigma0 = max(self._sigma_min, float(initial_sigma))
        log_sigma0 = max(self._sigma_min, float(initial_log_sigma))
        self._gate_beta = np.zeros(2, dtype=np.float64)
        self._market_beta = {
            name: np.zeros(2, dtype=np.float64) for name in self._component_names
        }
        self._market_p = {
            name: np.eye(2, dtype=np.float64) * float(rls_p0)
            for name in self._component_names
        }
        self._refresh_mean = {name: 0.0 for name in self._component_names}
        self._market_var = {
            name: (sigma0 * sigma0 if name == "mid" else log_sigma0 * log_sigma0)
            for name in self._component_names
        }
        self._refresh_var = {
            name: (sigma0 * sigma0 if name == "mid" else log_sigma0 * log_sigma0)
            for name in self._component_names
        }
        self.market_prob_hist: list[float] = []
        self.market_resp_hist: list[float] = []
        self.refresh_sigma_hist: list[float] = []
        self.market_sigma_hist: list[float] = []

    @property
    def reference_vwap_vol(self) -> float:
        return self._reference_vwap_vol

    @property
    def shape_alpha(self) -> float:
        return self._shape_alpha

    def _spread_from_ob(self, ob: Any) -> float:
        try:
            bid = float(ob.bids[0].price)
            ask = float(ob.asks[0].price)
            return max(self._param_eps, ask - bid)
        except (AttributeError, IndexError, TypeError, ValueError):
            return self._param_eps

    def _transition_mechanical_best(self, prev_ob: Any, buy_vol: float, sell_vol: float) -> tuple[float, float]:
        from mm.pricing.reference_prices_ticks import _consume_depth, _depth_by_price

        bid_prev = _depth_by_price(prev_ob, "bid")
        ask_prev = _depth_by_price(prev_ob, "ask")
        bid_after = _consume_depth(bid_prev, float(sell_vol), best_first_asc=False)
        ask_after = _consume_depth(ask_prev, float(buy_vol), best_first_asc=True)
        mid = float(prev_ob.mid)
        mech_bid = float(max(bid_after.keys())) if bid_after else mid
        mech_ask = float(min(ask_after.keys())) if ask_after else mid
        return mech_bid, mech_ask

    def _features_from_prices(
        self,
        *,
        mid: float,
        spread: float,
        buy_vwap: float,
        sell_vwap: float,
    ) -> dict[str, float]:
        half_spread = 0.5 * max(self._param_eps, float(spread))
        return {
            "mid": float(mid),
            "spread": max(self._param_eps, float(spread)),
            "buy_shape": max(self._param_eps, float(buy_vwap) - float(mid) - half_spread),
            "sell_shape": max(self._param_eps, float(mid) - float(sell_vwap) - half_spread),
        }

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        if prev_step is None:
            return
        from mm.pricing.reference_prices_ticks import (
            reference_prices_from_transition_ticks,
            reference_prices_ticks_from_orderbook,
        )

        _t_prev, prev_ob, _b_prev, _s_prev, *_ = prev_step
        _t_curr, cur_ob, buy_vol, sell_vol, *_ = curr_step
        try:
            prev_buy, prev_sell = reference_prices_ticks_from_orderbook(
                prev_ob, self._reference_vwap_vol
            )
            obs_buy, obs_sell, mech_buy, mech_sell = reference_prices_from_transition_ticks(
                prev_ob,
                cur_ob,
                float(buy_vol),
                float(sell_vol),
                self._reference_vwap_vol,
            )
            prev_mid = float(prev_ob.mid)
            obs_mid = float(cur_ob.mid)
            mech_bid, mech_ask = self._transition_mechanical_best(prev_ob, buy_vol, sell_vol)
        except (ValueError, ZeroDivisionError, TypeError, AttributeError):
            return

        prev_spread = self._spread_from_ob(prev_ob)
        obs_spread = self._spread_from_ob(cur_ob)
        mech_spread = max(self._param_eps, mech_ask - mech_bid)
        mech_mid = 0.5 * (mech_bid + mech_ask)
        prev = self._features_from_prices(
            mid=prev_mid,
            spread=prev_spread,
            buy_vwap=prev_buy,
            sell_vwap=prev_sell,
        )
        obs = self._features_from_prices(
            mid=obs_mid,
            spread=obs_spread,
            buy_vwap=obs_buy,
            sell_vwap=obs_sell,
        )
        mech = self._features_from_prices(
            mid=mech_mid,
            spread=mech_spread,
            buy_vwap=mech_buy,
            sell_vwap=mech_sell,
        )
        obs_components = {
            "mid": obs["mid"] - prev["mid"],
            "log_spread": math.log(obs["spread"]) - math.log(prev["spread"]),
            "log_buy_shape": math.log(obs["buy_shape"]) - math.log(prev["buy_shape"]),
            "log_sell_shape": math.log(obs["sell_shape"]) - math.log(prev["sell_shape"]),
        }
        mech_components = {
            "mid": mech["mid"] - prev["mid"],
            "log_spread": math.log(mech["spread"]) - math.log(prev["spread"]),
            "log_buy_shape": math.log(mech["buy_shape"]) - math.log(prev["buy_shape"]),
            "log_sell_shape": math.log(mech["sell_shape"]) - math.log(prev["sell_shape"]),
        }
        if not all(
            math.isfinite(v) for v in (*obs_components.values(), *mech_components.values())
        ):
            return

        total_flow = max(0.0, float(buy_vol) + float(sell_vol))
        gate_log_flow = min(math.log1p(total_flow), 12.0)
        gate_x = np.array([1.0, gate_log_flow], dtype=np.float64)
        q_market = self._market_probability_np(gate_log_flow)
        q_refresh = 1.0 - q_market
        market_log = _log_clamped(q_market)
        refresh_log = _log_clamped(q_refresh)
        market_err: dict[str, float] = {}
        refresh_err: dict[str, float] = {}
        for name in self._component_names:
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_mean = float(x @ self._market_beta[name])
            refresh_mean = float(self._refresh_mean[name])
            market_err[name] = obs_components[name] - market_mean
            refresh_err[name] = obs_components[name] - refresh_mean
            market_log += self._normal_logpdf(market_err[name], self._market_var[name])
            refresh_log += self._student_t_logpdf(refresh_err[name], self._refresh_var[name])
        log_norm = _logaddexp(market_log, refresh_log)
        market_resp = _exp_clamped(market_log - log_norm)
        refresh_resp = 1.0 - market_resp

        self._gate_beta = self._gate_beta + self._alpha * (market_resp - q_market) * gate_x
        self._gate_beta = np.clip(self._gate_beta, -20.0, 20.0)
        for name in self._component_names:
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            self._market_beta[name], self._market_p[name] = self._weighted_rls_update(
                self._market_beta[name],
                self._market_p[name],
                x,
                obs_components[name],
                weight=market_resp,
            )
            self._refresh_mean[name] = (
                self._refresh_mean[name]
                + self._alpha * refresh_resp * (obs_components[name] - self._refresh_mean[name])
            )
            self._market_var[name] = (
                (1.0 - self._alpha * market_resp) * self._market_var[name]
                + self._alpha * market_resp * market_err[name] * market_err[name]
            )
            self._refresh_var[name] = (
                (1.0 - self._alpha * refresh_resp) * self._refresh_var[name]
                + self._alpha * refresh_resp * refresh_err[name] * refresh_err[name]
            )
            self._market_var[name] = max(self._sigma_min * self._sigma_min, self._market_var[name])
            self._refresh_var[name] = max(self._sigma_min * self._sigma_min, self._refresh_var[name])
        self.market_prob_hist.append(float(q_market))
        self.market_resp_hist.append(float(market_resp))
        self.market_sigma_hist.append(
            float(np.sqrt(np.mean([self._market_var[name] for name in self._component_names])))
        )
        self.refresh_sigma_hist.append(
            float(np.sqrt(np.mean([self._refresh_var[name] for name in self._component_names])))
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "kind": "conditional_mechanical_mixture",
            "reference_vwap_vol": self._reference_vwap_vol,
            "shape_alpha": self._shape_alpha,
            "student_df": self._student_df,
            "gate_beta": self._gate_beta.tolist(),
            "market_beta": {name: self._market_beta[name].tolist() for name in self._component_names},
            "refresh_mean": dict(self._refresh_mean),
            "market_sigma": {
                name: float(np.sqrt(max(self._sigma_min * self._sigma_min, self._market_var[name])))
                for name in self._component_names
            },
            "refresh_sigma": {
                name: float(np.sqrt(max(self._sigma_min * self._sigma_min, self._refresh_var[name])))
                for name in self._component_names
            },
            "market_prob": self.market_prob_hist[-1] if self.market_prob_hist else 0.5,
            "market_resp": self.market_resp_hist[-1] if self.market_resp_hist else None,
        }

    def sample_innovations(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ) -> dict[str, Any]:
        import torch
        import torch.distributions as dist

        _device = resolve_torch_device(device or self._device)
        _dtype = dtype if dtype is not None else torch.float64
        shape = tuple(sample_shape)
        return {
            "u_market": torch.rand(shape, dtype=_dtype, device=_device),
            "z_market": {
                name: torch.randn(shape, dtype=_dtype, device=_device)
                for name in self._component_names
            },
            "t_refresh": {
                name: dist.StudentT(
                    torch.tensor(self._student_df, dtype=_dtype, device=_device)
                ).sample(shape)
                for name in self._component_names
            },
        }

    def sample_dynamic_vwap(
        self,
        *,
        prev_mid: Any,
        prev_spread: Any,
        prev_buy_shape: Any,
        prev_sell_shape: Any,
        mech_mid: Any,
        mech_spread: Any,
        mech_buy_shape: Any,
        mech_sell_shape: Any,
        total_flow: Any,
        buy_depth: Any,
        sell_depth: Any,
        innovations: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        import torch

        dtype = prev_mid.dtype
        device = prev_mid.device
        eps = torch.tensor(self._param_eps, dtype=dtype, device=device)
        prev_spread = torch.clamp(prev_spread, min=self._param_eps)
        prev_buy_shape = torch.clamp(prev_buy_shape, min=self._param_eps)
        prev_sell_shape = torch.clamp(prev_sell_shape, min=self._param_eps)
        mech_spread = torch.clamp(mech_spread, min=self._param_eps)
        mech_buy_shape = torch.clamp(mech_buy_shape, min=self._param_eps)
        mech_sell_shape = torch.clamp(mech_sell_shape, min=self._param_eps)
        mech_components = {
            "mid": mech_mid - prev_mid,
            "log_spread": torch.log(mech_spread) - torch.log(prev_spread),
            "log_buy_shape": torch.log(mech_buy_shape) - torch.log(prev_buy_shape),
            "log_sell_shape": torch.log(mech_sell_shape) - torch.log(prev_sell_shape),
        }
        mech_components = {
            name: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            for name, value in mech_components.items()
        }
        total_flow = torch.nan_to_num(torch.clamp(total_flow, min=0.0), nan=0.0, posinf=1e12, neginf=0.0)
        q_market = self._market_probability_torch(total_flow, dtype=dtype, device=device)
        if innovations is None:
            innovations = self.sample_innovations(tuple(prev_mid.shape), device=str(device), dtype=dtype)
        u_market = innovations["u_market"]
        if u_market.ndim == 2 and u_market.shape != prev_mid.shape:
            u_market = u_market.view_as(prev_mid)
        use_market = u_market < q_market
        pred_components: dict[str, Any] = {}
        for name in self._component_names:
            beta = torch.tensor(
                np.nan_to_num(self._market_beta[name], nan=0.0, posinf=0.0, neginf=0.0),
                dtype=dtype,
                device=device,
            )
            market_mean = beta[0] + beta[1] * mech_components[name]
            refresh_mean_value = float(self._refresh_mean[name])
            if not math.isfinite(refresh_mean_value):
                refresh_mean_value = 0.0
            refresh_mean = torch.tensor(refresh_mean_value, dtype=dtype, device=device)
            sigma_market = torch.tensor(
                max(self._sigma_min, math.sqrt(max(self._market_var[name], 0.0))),
                dtype=dtype,
                device=device,
            )
            sigma_refresh = torch.tensor(
                max(self._sigma_min, math.sqrt(max(self._refresh_var[name], 0.0))),
                dtype=dtype,
                device=device,
            )
            z = innovations["z_market"][name].view_as(prev_mid)
            t = innovations["t_refresh"][name].view_as(prev_mid)
            refresh_scale = sigma_refresh * math.sqrt((self._student_df - 2.0) / self._student_df)
            market_sample = market_mean + z * sigma_market
            refresh_sample = refresh_mean + t * refresh_scale
            if self._point_mode == "map":
                pred_components[name] = torch.where(q_market >= 0.5, market_mean, refresh_mean)
            elif self._point_mode == "mean":
                pred_components[name] = q_market * market_mean + (1.0 - q_market) * refresh_mean
            else:
                pred_components[name] = torch.where(use_market, market_sample, refresh_sample)
        pred_mid = prev_mid + pred_components["mid"]
        pred_spread = prev_spread * torch.exp(torch.clamp(pred_components["log_spread"], min=-5.0, max=5.0))
        pred_buy_shape = prev_buy_shape * torch.exp(
            torch.clamp(pred_components["log_buy_shape"], min=-5.0, max=5.0)
        )
        pred_sell_shape = prev_sell_shape * torch.exp(
            torch.clamp(pred_components["log_sell_shape"], min=-5.0, max=5.0)
        )
        ref_depth = torch.tensor(self._reference_vwap_vol, dtype=dtype, device=device)
        alpha = torch.tensor(self._shape_alpha, dtype=dtype, device=device)
        buy_ratio = self._depth_shape_ratio_torch(buy_depth, ref_depth, alpha)
        sell_ratio = self._depth_shape_ratio_torch(sell_depth, ref_depth, alpha)
        buy_vwap = pred_mid + 0.5 * pred_spread + pred_buy_shape * buy_ratio
        sell_vwap = pred_mid - 0.5 * pred_spread - pred_sell_shape * sell_ratio
        prev_buy_vwap = prev_mid + 0.5 * prev_spread + prev_buy_shape * buy_ratio
        prev_sell_vwap = prev_mid - 0.5 * prev_spread - prev_sell_shape * sell_ratio
        buy_vwap = torch.where(torch.isfinite(buy_vwap), buy_vwap, prev_buy_vwap)
        sell_vwap = torch.where(torch.isfinite(sell_vwap), sell_vwap, prev_sell_vwap)
        return buy_vwap, sell_vwap

    def _depth_shape_ratio_torch(self, depth: Any, ref_depth: Any, alpha: Any) -> Any:
        import torch

        depth = torch.nan_to_num(torch.clamp(depth, min=0.0), nan=0.0, posinf=1e12, neginf=0.0)
        if self._shape_alpha == 0.0:
            return torch.where(depth > 0.0, torch.ones_like(depth), torch.zeros_like(depth))
        # For sub-linear alpha, depth**alpha has an infinite derivative at depth=0.
        # Inventory can be exactly zero, so use a shifted power curve that remains
        # zero at zero depth but has finite slope for autograd.
        floor = torch.clamp(ref_depth * 1e-6, min=1e-9)
        shifted = torch.pow((depth + floor) / ref_depth, alpha)
        base = torch.pow(floor / ref_depth, alpha)
        top = torch.pow((ref_depth + floor) / ref_depth, alpha)
        return torch.clamp((shifted - base) / torch.clamp(top - base, min=1e-12), min=0.0)

    def _normal_logpdf(self, error: float, variance: float) -> float:
        var = max(self._sigma_min * self._sigma_min, float(variance))
        return -0.5 * (math.log(2.0 * math.pi * var) + (error * error) / var)

    def _market_probability_np(self, log_flow: float) -> float:
        x = np.array([1.0, float(log_flow) if math.isfinite(float(log_flow)) else 0.0], dtype=np.float64)
        gate_beta = np.nan_to_num(self._gate_beta, nan=0.0, posinf=20.0, neginf=-20.0)
        score = float(x @ gate_beta)
        return min(0.999, max(0.001, _expit(score)))

    def _market_probability_torch(self, total_flow: Any, *, dtype: Any, device: Any) -> Any:
        import torch

        log_flow = torch.clamp(torch.log1p(torch.clamp(total_flow, min=0.0)), max=12.0)
        gate_beta = torch.tensor(
            np.nan_to_num(self._gate_beta, nan=0.0, posinf=20.0, neginf=-20.0),
            dtype=dtype,
            device=device,
        )
        q_market = torch.sigmoid(gate_beta[0] + gate_beta[1] * log_flow)
        q_market = torch.nan_to_num(q_market, nan=0.5, posinf=0.999, neginf=0.001)
        return torch.clamp(q_market, min=0.001, max=0.999)

    def _student_t_logpdf(self, error: float, variance: float) -> float:
        nu = self._student_df
        var = max(self._sigma_min * self._sigma_min, float(variance))
        scale2 = max(self._sigma_min * self._sigma_min, var * (nu - 2.0) / nu)
        return (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu * math.pi * scale2)
            - ((nu + 1.0) / 2.0) * math.log1p((error * error) / (nu * scale2))
        )

    def _logaddexp(self, a: float, b: float) -> float:
        high = max(a, b)
        low = min(a, b)
        return high + math.log1p(math.exp(low - high))

    def _weighted_rls_update(
        self,
        beta: np.ndarray,
        p: np.ndarray,
        x: np.ndarray,
        target: float,
        *,
        weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        w = max(0.0, min(1.0, float(weight)))
        if w <= 1e-6:
            return beta, p
        scale = math.sqrt(w)
        xw = x * scale
        yw = target * scale
        px = p @ xw
        denom = self._gamma + float(xw @ px)
        gain = px / denom
        beta = beta + gain * (yw - float(xw @ beta))
        p = (p - np.outer(gain, xw) @ p) / self._gamma
        p = 0.5 * (p + p.T)
        return beta, p


def score_mid_delta(delta_mid: float, params: dict) -> "ScoreResult":
    """Pre-update Gaussian NLL for local mid change (see ``mm.models.scoring``)."""

    from mm.models.scoring import score_local_mid_delta

    return score_local_mid_delta(delta_mid=delta_mid, params_before=params)
