"""
Torch-based price models: Gaussian Δmid (GaussianPriceModel) and two-sided residual
noise around MOF-only representative VWAPs (TorchResidualPriceModel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mm.mpc.mpc_state import MarketState
from mm.mpc.torch_device import resolve_torch_device
from mm.mpc.step import Step

if TYPE_CHECKING:
    pass


@runtime_checkable
class PriceModel(Protocol):
    """Sample next mid given current market state and optional params; optional P(mid_next)."""

    def sample_next_mid(
        self,
        x: MarketState,
        rng: np.random.Generator | None = None,
        params: dict | None = None,
    ) -> float:
        ...

    def prob_mid(
        self,
        mid_next: float,
        x: MarketState,
        params: dict | None = None,
    ) -> float:
        """Return P(mid_next | x, params). Default 0.0 if not implemented."""
        ...


class GaussianPriceModel:
    """
    m_{t+1} = m_t + mu + epsilon, epsilon ~ N(0, sigma²).
    When params has mu_mid, sigma_mid (from price model), use those; else use
    x.sigma with scale (or default_sigma if x has no sigma).
    """

    def __init__(self, scale: float = 1.0, default_sigma: float = 1e-4):
        self.scale = scale
        self.default_sigma = default_sigma

    def sample_next_mid(
        self,
        x: MarketState,
        rng: np.random.Generator | None = None,
        params: dict | None = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        if params is not None and ("mu_mid" in params or "sigma_mid" in params):
            mu = float(params.get("mu_mid", 0.0))
            sigma = float(params.get("sigma_mid", self.default_sigma))
            sigma = max(1e-12, sigma)
            eps = rng.normal(0.0, sigma)
            return int(round(x.mid + mu + eps))
        sigma = getattr(x, "sigma", self.default_sigma) * self.scale
        sigma = max(1e-12, sigma)
        eps = rng.normal(0.0, sigma)
        return int(round(x.mid + eps))

    def prob_mid(
        self,
        mid_next: float,
        x: MarketState,
        params: dict | None = None,
    ) -> float:
        """P(mid_next | x, params) for discrete outcome: P(round(mid+mu+eps) = mid_next)."""
        from scipy.stats import norm
        if params is not None and ("mu_mid" in params or "sigma_mid" in params):
            mu = float(params.get("mu_mid", 0.0))
            sigma = float(params.get("sigma_mid", self.default_sigma))
        else:
            mu = 0.0
            sigma = getattr(x, "sigma", self.default_sigma) * self.scale
        sigma = max(1e-12, sigma)
        mean = x.mid + mu
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
    Two-sided residual noise on representative VWAPs (MOF-only book evolution).

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
        representative_price_target_depth_lots: float = 1000.0,
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
        depth = float(vwap_depth) if vwap_depth is not None else float(representative_price_target_depth_lots)
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
        buy_real, sell_real, buy_cmp, sell_cmp = reference_prices_from_transition_ticks(
            prev_ob,
            cur_ob,
            float(buy_vol),
            float(sell_vol),
            self._target_depth_lots,
        )
        self.realized_buy_price_hist.append(float(buy_real))
        self.computed_buy_price_hist.append(float(buy_cmp))
        self.realized_sell_price_hist.append(float(sell_real))
        self.computed_sell_price_hist.append(float(sell_cmp))

        e_buy = float(buy_real - buy_cmp)
        e_sell = float(sell_real - sell_cmp)
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
        """Sample independent Gaussian residuals for buy/sell representative prices."""
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
        """Sample pooled residual ~ N(μ_mid, σ_mid²) from get_params() (for legacy single-sided use)."""
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
