"""
Torch-based market order flow models: EWMA-updated params, .sample() for pipeline/MPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mm.mpc.torch_device import resolve_torch_device
from mm.mpc.step import Step

if TYPE_CHECKING:
    pass


@runtime_checkable
class OrderFlowModel(Protocol):
    """Sample (V_plus, V_minus) for next step from params; optional P(V_plus, V_minus)."""

    def sample_flow(
        self,
        params: dict | None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        """Return (V_plus, V_minus) as integer volumes."""
        ...

    def prob_flow(
        self,
        V_plus: int,
        V_minus: int,
        params: dict | None,
    ) -> float:
        """Return P(V_plus, V_minus | params). Default 0.0 if not implemented."""
        ...


@dataclass(slots=True)
class StratifiedScenarios:
    """Exact, weighted scenarios for hurdle order flow expectations."""

    buy_vol: Any  # torch.Tensor [S, horizon]
    sell_vol: Any  # torch.Tensor [S, horizon]
    weights: Any  # torch.Tensor [S], sum(weights) == 1.0
    p0_buy: float
    p0_sell: float
    mu_buy: float = 0.0
    sigma_buy: float = 0.0
    mu_sell: float = 0.0
    sigma_sell: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_hurdle_quantiles(
    p0: float,
    positive_nodes: int,
    quantile_fn: Any,
    include_tail: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    p0_clamped = max(0.0, min(1.0, float(p0)))
    k = max(1, int(positive_nodes))
    if include_tail and k == 3:
        u = np.array([0.25, 0.50, 0.85], dtype=np.float64)
        cond_w = np.array([0.375, 0.30, 0.325], dtype=np.float64)
    elif include_tail and k == 4:
        u = np.array([0.15, 0.40, 0.70, 0.92], dtype=np.float64)
        cond_w = np.array([0.275, 0.275, 0.25, 0.20], dtype=np.float64)
    elif include_tail and k == 5:
        u = np.array([0.10, 0.30, 0.50, 0.75, 0.95], dtype=np.float64)
        cond_w = np.array([0.20, 0.20, 0.225, 0.225, 0.15], dtype=np.float64)
    else:
        u = np.array([(i + 0.5) / k for i in range(k)], dtype=np.float64)
        cond_w = np.full(k, 1.0 / k, dtype=np.float64)

    cond_w = cond_w / cond_w.sum()
    pos_v = np.array([max(1.0, float(quantile_fn(u_i))) for u_i in u], dtype=np.float64)
    volumes = np.empty(k + 1, dtype=np.float64)
    weights = np.empty(k + 1, dtype=np.float64)
    volumes[0] = 0.0
    weights[0] = p0_clamped
    volumes[1:] = pos_v
    weights[1:] = (1.0 - p0_clamped) * cond_w
    return volumes, weights


def build_hurdle_joint_scenarios(
    p0_buy: float,
    p0_sell: float,
    buy_quantile_fn: Any,
    sell_quantile_fn: Any,
    horizon: int = 4,
    positive_nodes: int = 3,
    device: str | None = None,
    dtype: Any = None,
    include_tail: bool = True,
    meta: dict[str, Any] | None = None,
) -> StratifiedScenarios:
    import torch

    _device = resolve_torch_device(device)
    _dtype = dtype if dtype is not None else torch.float64

    v_b, w_b = _build_hurdle_quantiles(p0_buy, positive_nodes, buy_quantile_fn, include_tail)
    v_s, w_s = _build_hurdle_quantiles(p0_sell, positive_nodes, sell_quantile_fn, include_tail)

    Nb = len(v_b)
    Ns = len(v_s)
    S = Nb * Ns
    H = max(1, int(horizon))
    Kb = Nb - 1
    Ks = Ns - 1

    buy_vol = np.zeros((S, H), dtype=np.float64)
    sell_vol = np.zeros((S, H), dtype=np.float64)
    weights = np.zeros(S, dtype=np.float64)

    for i in range(Nb):
        for j in range(Ns):
            s = i * Ns + j
            weights[s] = w_b[i] * w_s[j]
            for h in range(H):
                if i == 0:
                    buy_vol[s, h] = 0.0
                else:
                    pos_idx = 1 + ((i - 1 + h) % Kb)
                    buy_vol[s, h] = v_b[pos_idx]
                if j == 0:
                    sell_vol[s, h] = 0.0
                else:
                    pos_idx = 1 + ((j - 1 + h) % Ks)
                    sell_vol[s, h] = v_s[pos_idx]

    weights = weights / max(1e-12, weights.sum())
    buy_vol_t = torch.tensor(buy_vol, dtype=_dtype, device=_device)
    sell_vol_t = torch.tensor(sell_vol, dtype=_dtype, device=_device)
    weights_t = torch.tensor(weights, dtype=_dtype, device=_device)

    return StratifiedScenarios(
        buy_vol=buy_vol_t,
        sell_vol=sell_vol_t,
        weights=weights_t,
        p0_buy=float(p0_buy),
        p0_sell=float(p0_sell),
        metadata=dict(meta or {}),
    )


class HurdleOrderFlow:
    """
    Sample V+ and V- from a hurdle log-normal or from a discrete Gamma model.

    Params from estimator.get_params():
      - Log-normal: p0_buy, mu_buy, sigma_buy; p0_sell, mu_sell, sigma_sell.
      - Weibull: wb_k_buy, wb_lambda_buy; optional p0_* for hurdle (Bernoulli @ 0 then round(Weibull)).
      - Gamma: shape_buy, scale_buy; shape_sell, scale_sell (no hurdle; X = round(Gamma)).
    """

    def __init__(
        self,
        default_p: float = 0.5,
        default_mu: float = 0.0,
        default_sigma: float = 1.0,
    ):
        self.default_p = default_p
        self.default_mu = default_mu
        self.default_sigma = default_sigma

    @staticmethod
    def _sample_discrete_gamma(shape: float, scale: float, rng: np.random.Generator) -> int:
        """Sample one integer lot count: round(Gamma(shape, scale))."""
        if shape <= 0 or scale <= 0:
            return 0
        try:
            x = rng.gamma(shape, scale)
            return max(0, int(round(x)))
        except Exception:
            return 0

    @staticmethod
    def _sample_discrete_weibull(c: float, scale: float, rng: np.random.Generator) -> int:
        """Sample one integer lot count: round(Weibull(c, scale))."""
        if c <= 0 or scale <= 0:
            return 0
        try:
            from scipy.stats import weibull_min

            x = weibull_min.rvs(c, scale=scale, random_state=rng)
            return max(0, int(round(float(x))))
        except Exception:
            return 0

    @staticmethod
    def _sample_hurdle_log_normal(
        p: float, mu: float, sigma: float, rng: np.random.Generator
    ) -> int:
        """Sample one integer lot count from hurdle log-normal."""
        if p <= 0.0 or p >= 1.0 or sigma <= 0.0:
            return 0
        u = rng.uniform(0.0, 1.0)
        if u > p:
            return 0
        z = rng.normal(mu, sigma)
        x = int(round(np.exp(z)))
        return max(1, x)

    @staticmethod
    def _pmf_hurdle(k: int, p: float, mu: float, sigma: float) -> float:
        """P(X = k) for hurdle log-normal: (1-p) at 0, p * discrete_ln_pmf(k) for k >= 1."""
        if k < 0:
            return 0.0
        if k == 0:
            return 1.0 - p
        if p <= 0.0 or p >= 1.0 or sigma <= 0.0:
            return 0.0
        try:
            from scipy.stats import norm
            a = (np.log(k - 0.5) - mu) / sigma
            b = (np.log(k + 0.5) - mu) / sigma
            phi_zero = norm.cdf((np.log(0.5) - mu) / sigma)
            norm_const = 1.0 - phi_zero
            if norm_const <= 0:
                return 0.0
            return p * float((norm.cdf(b) - norm.cdf(a)) / norm_const)
        except Exception:
            return 0.0

    def sample_flow(
        self,
        params: dict | None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        """Return (V_plus, V_minus); Weibull, else Gamma when shape_buy/scale_buy, else hurdle log-normal."""
        rng = rng or np.random.default_rng()
        par = params or {}
        if "wb_k_buy" in par and "wb_lambda_buy" in par:
            k_buy = float(par["wb_k_buy"])
            lam_buy = float(par["wb_lambda_buy"])
            k_sell = float(par.get("wb_k_sell") or k_buy)
            lam_sell = float(par.get("wb_lambda_sell") or lam_buy)
            if "p0_buy" in par or "p_buy" in par:
                p_buy = float(par.get("p0_buy", par.get("p_buy", 0.0)))
                p_sell = float(par.get("p0_sell", par.get("p_sell", 0.0)))

                def _one_hurdle_wb(p0: float, k: float, lam: float) -> int:
                    if p0 <= 0.0:
                        return self._sample_discrete_weibull(k, lam, rng)
                    if p0 >= 1.0:
                        return 0
                    if rng.uniform(0.0, 1.0) <= p0:
                        return 0
                    return self._sample_discrete_weibull(k, lam, rng)

                v_plus = _one_hurdle_wb(p_buy, k_buy, lam_buy)
                v_minus = _one_hurdle_wb(p_sell, k_sell, lam_sell)
                return (v_plus, v_minus)
            v_plus = self._sample_discrete_weibull(k_buy, lam_buy, rng)
            v_minus = self._sample_discrete_weibull(k_sell, lam_sell, rng)
            return (v_plus, v_minus)
        if "shape_buy" in par and "scale_buy" in par:
            shape_buy = float(par["shape_buy"])
            scale_buy = float(par["scale_buy"])
            shape_sell = float(par.get("shape_sell") or shape_buy)
            scale_sell = float(par.get("scale_sell") or scale_buy)
            v_plus = self._sample_discrete_gamma(shape_buy, scale_buy, rng)
            v_minus = self._sample_discrete_gamma(shape_sell, scale_sell, rng)
            return (v_plus, v_minus)
        p_buy = float(par.get("p_buy") or self.default_p)
        mu_buy = float(par.get("mu_buy") or self.default_mu)
        sigma_buy = float(par.get("sigma_buy") or self.default_sigma)
        p_sell = float(par.get("p_sell") or self.default_p)
        mu_sell = float(par.get("mu_sell") or self.default_mu)
        sigma_sell = float(par.get("sigma_sell") or self.default_sigma)
        v_plus = self._sample_hurdle_log_normal(p_buy, mu_buy, sigma_buy, rng)
        v_minus = self._sample_hurdle_log_normal(p_sell, mu_sell, sigma_sell, rng)
        return (v_plus, v_minus)

    def prob_flow(
        self,
        V_plus: int,
        V_minus: int,
        params: dict | None,
    ) -> float:
        """P(V_plus, V_minus); Weibull, Gamma, or hurdle log-normal per side."""
        par = params or {}
        if "wb_k_buy" in par and "wb_lambda_buy" in par:
            k_buy = float(par["wb_k_buy"])
            lam_buy = float(par["wb_lambda_buy"])
            k_sell = float(par.get("wb_k_sell") or k_buy)
            lam_sell = float(par.get("wb_lambda_sell") or lam_buy)
            if "p0_buy" in par or "p_buy" in par:
                p_buy = float(par.get("p0_buy", par.get("p_buy", 0.0)))
                p_sell = float(par.get("p0_sell", par.get("p_sell", 0.0)))
                prob_buy = _hurdle_weibull_pmf(V_plus, p_buy, k_buy, lam_buy)
                prob_sell = _hurdle_weibull_pmf(V_minus, p_sell, k_sell, lam_sell)
            else:
                prob_buy = _discrete_weibull_pmf(V_plus, k_buy, lam_buy)
                prob_sell = _discrete_weibull_pmf(V_minus, k_sell, lam_sell)
            return prob_buy * prob_sell
        if "shape_buy" in par and "scale_buy" in par:
            shape_buy = float(par["shape_buy"])
            scale_buy = float(par["scale_buy"])
            shape_sell = float(par.get("shape_sell") or shape_buy)
            scale_sell = float(par.get("scale_sell") or scale_buy)
            prob_buy = _discrete_gamma_pmf(V_plus, shape_buy, scale_buy)
            prob_sell = _discrete_gamma_pmf(V_minus, shape_sell, scale_sell)
            return prob_buy * prob_sell
        p_buy = float(par.get("p_buy") or self.default_p)
        mu_buy = float(par.get("mu_buy") or self.default_mu)
        sigma_buy = float(par.get("sigma_buy") or self.default_sigma)
        p_sell = float(par.get("p_sell") or self.default_p)
        mu_sell = float(par.get("mu_sell") or self.default_mu)
        sigma_sell = float(par.get("sigma_sell") or self.default_sigma)
        prob_buy = self._pmf_hurdle(V_plus, p_buy, mu_buy, sigma_buy)
        prob_sell = self._pmf_hurdle(V_minus, p_sell, mu_sell, sigma_sell)
        return prob_buy * prob_sell


def _discrete_gamma_pmf(k: int, shape: float, scale: float) -> float:
    """P(X = k) for X = round(Y), Y ~ Gamma(shape, scale)."""
    if k < 0 or shape <= 0 or scale <= 0:
        return 0.0
    try:
        from scipy.stats import gamma as gamma_dist
        if k == 0:
            return float(gamma_dist.cdf(0.5, a=shape, scale=scale))
        return float(
            gamma_dist.cdf(k + 0.5, a=shape, scale=scale)
            - gamma_dist.cdf(k - 0.5, a=shape, scale=scale)
        )
    except Exception:
        return 0.0


def _discrete_weibull_pmf(k: int, c: float, scale: float) -> float:
    """P(X = k) for X = round(Y), Y ~ Weibull(shape c, scale) (scipy weibull_min)."""
    if k < 0 or c <= 0 or scale <= 0:
        return 0.0
    try:
        from scipy.stats import weibull_min

        cc = float(c)
        sc = float(scale)
        if k == 0:
            return float(weibull_min.cdf(0.5, c=cc, scale=sc))
        return float(
            weibull_min.cdf(float(k) + 0.5, c=cc, scale=sc)
            - weibull_min.cdf(max(0.5, float(k) - 0.5), c=cc, scale=sc)
        )
    except Exception:
        return 0.0


def _weibull_cv_sq(k: float) -> float:
    """Var/mean^2 for Weibull(c, scale) as a function of shape c only."""
    from scipy.special import gamma as gamma_fn

    kk = max(1e-9, float(k))
    g1 = gamma_fn(1.0 + 1.0 / kk)
    g2 = gamma_fn(1.0 + 2.0 / kk)
    return float((g2 - g1 * g1) / (g1 * g1))


def _weibull_k_from_cv2(cv2: float, k_fallback: float) -> float:
    """Invert Var/mean^2 -> Weibull shape c via monotone bisection."""
    from scipy.optimize import brentq

    cv2 = float(np.clip(cv2, 1e-12, 1e6))
    lo, hi = 0.06, 400.0
    v_lo = _weibull_cv_sq(hi)
    v_hi = _weibull_cv_sq(lo)
    if cv2 <= v_lo:
        return float(hi)
    if cv2 >= v_hi:
        return float(lo)
    try:
        return float(brentq(lambda kk: _weibull_cv_sq(kk) - cv2, lo, hi, xtol=1e-5, rtol=1e-5))
    except ValueError:
        return float(k_fallback)


def _hurdle_gamma_pmf(k: int, p0: float, shape: float, scale: float) -> float:
    """P(X = k) for hurdle gamma: P(0)=p0; for k>=1, (1-p0)*P(round(Y)=k), Y~Gamma(shape, scale)."""
    if k < 0 or shape <= 0 or scale <= 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    if k == 0:
        return float(p0)
    try:
        from scipy.stats import gamma as gamma_dist

        cdf_hi = gamma_dist.cdf(float(k) + 0.5, a=float(shape), scale=float(scale))
        cdf_lo = gamma_dist.cdf(max(0.5, float(k) - 0.5), a=float(shape), scale=float(scale))
        return float((1.0 - p0) * max(0.0, cdf_hi - cdf_lo))
    except Exception:
        return 0.0


def _hurdle_weibull_pmf(k: int, p0: float, c: float, scale: float) -> float:
    """P(X = k) for hurdle Weibull: P(0)=p0; for k>=1, (1-p0)*P(round(Y)=k), Y~Weibull(c, scale)."""
    if k < 0 or c <= 0 or scale <= 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    if k == 0:
        return float(p0)
    try:
        from scipy.stats import weibull_min

        cc = float(c)
        sc = float(scale)
        cdf_hi = weibull_min.cdf(float(k) + 0.5, c=cc, scale=sc)
        cdf_lo = weibull_min.cdf(max(0.5, float(k) - 0.5), c=cc, scale=sc)
        return float((1.0 - p0) * max(0.0, cdf_hi - cdf_lo))
    except Exception:
        return 0.0


def _hurdle_lognormal_pmf(k: int, p0: float, mu: float, sigma: float) -> float:
    """P(X = k) for hurdle lognormal: P(0) = p0; for k >= 1, (1-p0) * P(round(Y)=k), Y ~ LogNormal(μ, σ)."""
    if k < 0 or sigma <= 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    if k == 0:
        return float(p0)
    try:
        from scipy.stats import lognorm
        scale = np.exp(float(mu))
        s = float(sigma)
        if scale <= 0 or s <= 0:
            return 0.0
        cdf_hi = lognorm.cdf(float(k) + 0.5, s=s, scale=scale)
        cdf_lo = lognorm.cdf(max(0.5, float(k) - 0.5), s=s, scale=scale)
        return float((1.0 - p0) * max(0.0, cdf_hi - cdf_lo))
    except Exception:
        return 0.0


def _hurdle_logt_pmf(k: int, p0: float, df: float, mu: float, sigma: float) -> float:
    """P(X = k) for hurdle log-t: P(0) = p0; for k >= 1, (1-p0) * P(round(Y)=k), Y = exp(Z), Z ~ Student-t(df, μ, σ)."""
    if k < 0 or sigma <= 0 or df <= 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    if k == 0:
        return float(p0)
    try:
        from scipy.stats import t as t_dist

        d = float(df)
        m = float(mu)
        s = float(sigma)
        if d <= 0 or s <= 0:
            return 0.0
        cdf_hi = t_dist.cdf(np.log(float(k) + 0.5), d, loc=m, scale=s)
        cdf_lo = t_dist.cdf(np.log(max(0.5, float(k) - 0.5)), d, loc=m, scale=s)
        return float((1.0 - p0) * max(0.0, cdf_hi - cdf_lo))
    except Exception:
        return 0.0


def _hurdle_generalized_pareto_pmf(k: int, p0: float, xi: float, beta: float, u: float) -> float:
    """P(X = k) for hurdle generalized Pareto: P(0)=p0; for k>=1, (1-p0)*P(round(U+Y)=k), Y~GPD(xi,beta)."""
    if k < 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    if k == 0:
        return float(p0)
    xi = float(xi)
    beta = max(1e-9, float(beta))
    u = max(1e-9, float(u))

    def gpd_cdf_y(y: float) -> float:
        y = float(y)
        if y < 0:
            return 0.0
        if abs(xi) < 1e-9:
            return 1.0 - np.exp(-y / beta)
        z = 1.0 + xi * y / beta
        if z <= 0:
            # xi < 0 and y is beyond finite support: CDF saturates at 1.
            return 1.0
        return 1.0 - z ** (-1.0 / xi)

    try:
        lo = max(float(k) - 0.5, u)
        hi = float(k) + 0.5
        if hi <= lo:
            return 0.0
        lo_y = max(0.0, lo - u)
        hi_y = max(0.0, hi - u)
        return float((1.0 - p0) * max(0.0, gpd_cdf_y(hi_y) - gpd_cdf_y(lo_y)))
    except Exception:
        return 0.0


class TorchDiscreteGammaModel:
    """
    Market order flow (discrete Gamma) with params updated by EWMA.
    get_params() returns shape_buy, scale_buy, shape_sell, scale_sell.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_mean: float = 10.0,
        init_shape: float = 2.0,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        init_scale = float(init_mean) / max(1e-6, float(init_shape))
        self._S_buy: float = 0.0
        self._S2_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_sell: float = 0.0
        self._S2_sell: float = 0.0
        self._W_sell: float = 0.0
        self._shape_buy: float = float(init_shape)
        self._scale_buy: float = float(init_scale)
        self._shape_sell: float = float(init_shape)
        self._scale_sell: float = float(init_scale)
        self.shape_buy_hist: list[float] = []
        self.scale_buy_hist: list[float] = []
        self.shape_sell_hist: list[float] = []
        self.scale_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps
        if x_buy > 0:
            v = float(x_buy)
            self._S_buy = g * self._S_buy + v
            self._S2_buy = g * self._S2_buy + v * v
            self._W_buy = g * self._W_buy + 1.0
        if x_sell > 0:
            v = float(x_sell)
            self._S_sell = g * self._S_sell + v
            self._S2_sell = g * self._S2_sell + v * v
            self._W_sell = g * self._W_sell + 1.0
        if self._W_buy > 1e-12:
            mean_buy = self._S_buy / self._W_buy
            var_buy = max(eps * eps, self._S2_buy / self._W_buy - mean_buy * mean_buy)
            self._shape_buy = mean_buy * mean_buy / var_buy
            self._scale_buy = var_buy / mean_buy
        if self._W_sell > 1e-12:
            mean_sell = self._S_sell / self._W_sell
            var_sell = max(eps * eps, self._S2_sell / self._W_sell - mean_sell * mean_sell)
            self._shape_sell = mean_sell * mean_sell / var_sell
            self._scale_sell = var_sell / mean_sell
        self.shape_buy_hist.append(self._shape_buy)
        self.scale_buy_hist.append(self._scale_buy)
        self.shape_sell_hist.append(self._shape_sell)
        self.scale_sell_hist.append(self._scale_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (Gamma): shape (buy, sell)", "ylabel": "shape"},
            {"title": "Market order flow (Gamma): scale (buy, sell)", "ylabel": "scale"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (Gamma, EWMA γ={self._gamma})",
            "ylabel": "shape, scale",
            "series": [
                {"label": "shape_buy", "color": "green", "y": self.shape_buy_hist},
                {"label": "shape_sell", "color": "red", "y": self.shape_sell_hist},
                {"label": "scale_buy", "color": "green", "y": self.scale_buy_hist},
                {"label": "scale_sell", "color": "red", "y": self.scale_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (Gamma): shape", "ylabel": "shape", "series": [{"label": "shape_buy", "color": "green", "y": self.shape_buy_hist}, {"label": "shape_sell", "color": "red", "y": self.shape_sell_hist}]},
                {"title": "Market order flow (Gamma): scale", "ylabel": "scale", "series": [{"label": "scale_buy", "color": "green", "y": self.scale_buy_hist}, {"label": "scale_sell", "color": "red", "y": self.scale_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "shape_buy": self._shape_buy,
            "scale_buy": self._scale_buy,
            "shape_sell": self._shape_sell,
            "scale_sell": self._scale_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol) from Gamma(shape, scale) per side."""
        import torch
        import torch.distributions as dist
        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        shape_buy = max(1.0, p["shape_buy"])
        scale_buy = max(1e-9, p["scale_buy"])
        shape_sell = max(1.0, p["shape_sell"])
        scale_sell = max(1e-9, p["scale_sell"])
        gam_buy = dist.Gamma(
            torch.tensor(shape_buy, dtype=_dtype, device=_device),
            torch.tensor(1.0 / scale_buy, dtype=_dtype, device=_device),
        )
        gam_sell = dist.Gamma(
            torch.tensor(shape_sell, dtype=_dtype, device=_device),
            torch.tensor(1.0 / scale_sell, dtype=_dtype, device=_device),
        )
        buy_vol = gam_buy.sample(sample_shape).clamp(min=1e-6)
        sell_vol = gam_sell.sample(sample_shape).clamp(min=1e-6)
        return buy_vol, sell_vol

    @staticmethod
    def pmf(k: int, shape: float, scale: float) -> float:
        return _discrete_gamma_pmf(k, shape, scale)


def _weibull_lambda_from_mean_k(mean: float, k: float) -> float:
    from scipy.special import gamma as gamma_fn

    m = max(1e-12, float(mean))
    kk = max(1e-9, float(k))
    return float(m / max(1e-18, gamma_fn(1.0 + 1.0 / kk)))


class TorchWeibullModel:
    """
    Market order flow (discrete Weibull): X = round(Y), Y ~ Weibull(shape c, scale λ).
    EWMA on positive volumes per side; (c, λ) from mean/variance via Weibull moments.
    get_params() uses wb_k_*, wb_lambda_* (distinct from Gamma shape/scale keys).
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_mean: float = 10.0,
        init_k: float = 2.0,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        k0 = max(0.05, float(init_k))
        lam0 = _weibull_lambda_from_mean_k(float(init_mean), k0)
        self._k_buy: float = k0
        self._lambda_buy: float = lam0
        self._k_sell: float = k0
        self._lambda_sell: float = lam0
        self._S_buy: float = 0.0
        self._S2_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_sell: float = 0.0
        self._S2_sell: float = 0.0
        self._W_sell: float = 0.0
        self.k_buy_hist: list[float] = []
        self.k_sell_hist: list[float] = []
        self.lambda_buy_hist: list[float] = []
        self.lambda_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, _ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps
        if x_buy > 0:
            v = float(x_buy)
            self._S_buy = g * self._S_buy + v
            self._S2_buy = g * self._S2_buy + v * v
            self._W_buy = g * self._W_buy + 1.0
        if x_sell > 0:
            v = float(x_sell)
            self._S_sell = g * self._S_sell + v
            self._S2_sell = g * self._S2_sell + v * v
            self._W_sell = g * self._W_sell + 1.0
        if self._W_buy > 1e-12:
            mean_buy = self._S_buy / self._W_buy
            var_buy = max(eps * eps, self._S2_buy / self._W_buy - mean_buy * mean_buy)
            cv2_buy = var_buy / max(eps * eps, mean_buy * mean_buy)
            self._k_buy = float(np.clip(_weibull_k_from_cv2(cv2_buy, self._k_buy), 0.05, 150.0))
            self._lambda_buy = _weibull_lambda_from_mean_k(mean_buy, self._k_buy)
        if self._W_sell > 1e-12:
            mean_sell = self._S_sell / self._W_sell
            var_sell = max(eps * eps, self._S2_sell / self._W_sell - mean_sell * mean_sell)
            cv2_sell = var_sell / max(eps * eps, mean_sell * mean_sell)
            self._k_sell = float(np.clip(_weibull_k_from_cv2(cv2_sell, self._k_sell), 0.05, 150.0))
            self._lambda_sell = _weibull_lambda_from_mean_k(mean_sell, self._k_sell)
        self.k_buy_hist.append(self._k_buy)
        self.k_sell_hist.append(self._k_sell)
        self.lambda_buy_hist.append(self._lambda_buy)
        self.lambda_sell_hist.append(self._lambda_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (Weibull): k (buy, sell)", "ylabel": "k"},
            {"title": "Market order flow (Weibull): λ (buy, sell)", "ylabel": "λ"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (Weibull, EWMA γ={self._gamma})",
            "ylabel": "k, λ",
            "series": [
                {"label": "k_buy", "color": "green", "y": self.k_buy_hist},
                {"label": "k_sell", "color": "red", "y": self.k_sell_hist},
                {"label": "lambda_buy", "color": "green", "y": self.lambda_buy_hist},
                {"label": "lambda_sell", "color": "red", "y": self.lambda_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (Weibull): k", "ylabel": "k", "series": [{"label": "k_buy", "color": "green", "y": self.k_buy_hist}, {"label": "k_sell", "color": "red", "y": self.k_sell_hist}]},
                {"title": "Market order flow (Weibull): λ", "ylabel": "λ", "series": [{"label": "lambda_buy", "color": "green", "y": self.lambda_buy_hist}, {"label": "lambda_sell", "color": "red", "y": self.lambda_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "wb_k_buy": self._k_buy,
            "wb_lambda_buy": self._lambda_buy,
            "wb_k_sell": self._k_sell,
            "wb_lambda_sell": self._lambda_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol) from Weibull(k, λ) per side (PyTorch)."""
        import torch
        import torch.distributions as dist

        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        k_buy = max(1e-3, float(p["wb_k_buy"]))
        lam_buy = max(1e-9, float(p["wb_lambda_buy"]))
        k_sell = max(1e-3, float(p["wb_k_sell"]))
        lam_sell = max(1e-9, float(p["wb_lambda_sell"]))
        w_buy = dist.Weibull(
            scale=torch.tensor(lam_buy, dtype=_dtype, device=_device),
            concentration=torch.tensor(k_buy, dtype=_dtype, device=_device),
        )
        w_sell = dist.Weibull(
            scale=torch.tensor(lam_sell, dtype=_dtype, device=_device),
            concentration=torch.tensor(k_sell, dtype=_dtype, device=_device),
        )
        buy_vol = w_buy.sample(sample_shape).clamp(min=1e-6)
        sell_vol = w_sell.sample(sample_shape).clamp(min=1e-6)
        return buy_vol, sell_vol

    @staticmethod
    def pmf(k: int, c: float, scale: float) -> float:
        return _discrete_weibull_pmf(k, c, scale)


class TorchHurdleGammaModel:
    """
    Market order flow: hurdle gamma. P(0)=p0; if >0 then Gamma(shape, scale).
    Params updated by EWMA. get_params() returns p0_*, shape_*, scale_* per side.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_p0: float = 0.3,
        init_mean: float = 10.0,
        init_shape: float = 2.0,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        init_scale = float(init_mean) / max(1e-6, float(init_shape))
        self._p0_buy: float = max(0.0, min(1.0, float(init_p0)))
        self._p0_sell: float = max(0.0, min(1.0, float(init_p0)))
        self._S_buy: float = 0.0
        self._S2_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_sell: float = 0.0
        self._S2_sell: float = 0.0
        self._W_sell: float = 0.0
        self._shape_buy: float = float(init_shape)
        self._scale_buy: float = float(init_scale)
        self._shape_sell: float = float(init_shape)
        self._scale_sell: float = float(init_scale)
        self.p0_buy_hist: list[float] = []
        self.p0_sell_hist: list[float] = []
        self.shape_buy_hist: list[float] = []
        self.scale_buy_hist: list[float] = []
        self.shape_sell_hist: list[float] = []
        self.scale_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, _ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps

        z_buy = 1.0 if x_buy == 0 else 0.0
        self._p0_buy = g * self._p0_buy + (1.0 - g) * z_buy
        self._p0_buy = max(0.0, min(1.0, self._p0_buy))
        if x_buy > 0:
            v = float(x_buy)
            self._S_buy = g * self._S_buy + v
            self._S2_buy = g * self._S2_buy + v * v
            self._W_buy = g * self._W_buy + 1.0
        if self._W_buy > 1e-12:
            mean_buy = self._S_buy / self._W_buy
            var_buy = max(eps * eps, self._S2_buy / self._W_buy - mean_buy * mean_buy)
            self._shape_buy = mean_buy * mean_buy / var_buy
            self._scale_buy = var_buy / max(eps, mean_buy)

        z_sell = 1.0 if x_sell == 0 else 0.0
        self._p0_sell = g * self._p0_sell + (1.0 - g) * z_sell
        self._p0_sell = max(0.0, min(1.0, self._p0_sell))
        if x_sell > 0:
            v = float(x_sell)
            self._S_sell = g * self._S_sell + v
            self._S2_sell = g * self._S2_sell + v * v
            self._W_sell = g * self._W_sell + 1.0
        if self._W_sell > 1e-12:
            mean_sell = self._S_sell / self._W_sell
            var_sell = max(eps * eps, self._S2_sell / self._W_sell - mean_sell * mean_sell)
            self._shape_sell = mean_sell * mean_sell / var_sell
            self._scale_sell = var_sell / max(eps, mean_sell)

        self.p0_buy_hist.append(self._p0_buy)
        self.p0_sell_hist.append(self._p0_sell)
        self.shape_buy_hist.append(self._shape_buy)
        self.shape_sell_hist.append(self._shape_sell)
        self.scale_buy_hist.append(self._scale_buy)
        self.scale_sell_hist.append(self._scale_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (hurdle Gamma): P(0) (buy, sell)", "ylabel": "P(0)"},
            {"title": "Market order flow (hurdle Gamma): shape (buy, sell)", "ylabel": "shape"},
            {"title": "Market order flow (hurdle Gamma): scale (buy, sell)", "ylabel": "scale"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (hurdle Gamma, EWMA γ={self._gamma})",
            "ylabel": "P(0), shape, scale",
            "series": [
                {"label": "p0_buy", "color": "green", "y": self.p0_buy_hist},
                {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist},
                {"label": "shape_buy", "color": "green", "y": self.shape_buy_hist},
                {"label": "shape_sell", "color": "red", "y": self.shape_sell_hist},
                {"label": "scale_buy", "color": "green", "y": self.scale_buy_hist},
                {"label": "scale_sell", "color": "red", "y": self.scale_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (hurdle Gamma): P(0)", "ylabel": "P(0)", "series": [{"label": "p0_buy", "color": "green", "y": self.p0_buy_hist}, {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist}]},
                {"title": "Market order flow (hurdle Gamma): shape", "ylabel": "shape", "series": [{"label": "shape_buy", "color": "green", "y": self.shape_buy_hist}, {"label": "shape_sell", "color": "red", "y": self.shape_sell_hist}]},
                {"title": "Market order flow (hurdle Gamma): scale", "ylabel": "scale", "series": [{"label": "scale_buy", "color": "green", "y": self.scale_buy_hist}, {"label": "scale_sell", "color": "red", "y": self.scale_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "p0_buy": self._p0_buy,
            "p0_sell": self._p0_sell,
            "shape_buy": self._shape_buy,
            "scale_buy": self._scale_buy,
            "shape_sell": self._shape_sell,
            "scale_sell": self._scale_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol): hurdle then Gamma, preserving exact zero mass."""
        import torch
        import torch.distributions as dist

        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        p0_buy = max(0.0, min(1.0, p["p0_buy"]))
        p0_sell = max(0.0, min(1.0, p["p0_sell"]))
        shape_buy = max(1.0, p["shape_buy"])
        scale_buy = max(1e-9, p["scale_buy"])
        shape_sell = max(1.0, p["shape_sell"])
        scale_sell = max(1e-9, p["scale_sell"])
        u_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        u_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        gam_buy = dist.Gamma(
            torch.tensor(shape_buy, dtype=_dtype, device=_device),
            torch.tensor(1.0 / scale_buy, dtype=_dtype, device=_device),
        )
        gam_sell = dist.Gamma(
            torch.tensor(shape_sell, dtype=_dtype, device=_device),
            torch.tensor(1.0 / scale_sell, dtype=_dtype, device=_device),
        )
        raw_buy = gam_buy.sample(sample_shape).clamp(min=1e-6)
        raw_sell = gam_sell.sample(sample_shape).clamp(min=1e-6)
        buy_vol = torch.where(u_buy <= p0_buy, torch.zeros_like(raw_buy), raw_buy)
        sell_vol = torch.where(u_sell <= p0_sell, torch.zeros_like(raw_sell), raw_sell)
        return buy_vol, sell_vol

    def stratified_scenarios(
        self,
        horizon: int = 4,
        positive_nodes: int = 3,
        device: str | None = None,
        dtype: Any = None,
        include_tail: bool = True,
    ) -> StratifiedScenarios:
        """Construct exact, weighted stratified scenarios over hurdle Gamma distribution."""
        from scipy.special import gammaincinv

        p = self.get_params()
        p0_buy = max(0.0, min(1.0, float(p["p0_buy"])))
        p0_sell = max(0.0, min(1.0, float(p["p0_sell"])))
        sh_b, sc_b = max(1e-6, float(p["shape_buy"])), max(1e-6, float(p["scale_buy"]))
        sh_s, sc_s = max(1e-6, float(p["shape_sell"])), max(1e-6, float(p["scale_sell"]))
        return build_hurdle_joint_scenarios(
            p0_buy=p0_buy,
            p0_sell=p0_sell,
            buy_quantile_fn=lambda u: sc_b * gammaincinv(sh_b, u),
            sell_quantile_fn=lambda u: sc_s * gammaincinv(sh_s, u),
            horizon=horizon,
            positive_nodes=positive_nodes,
            device=device,
            dtype=dtype,
            include_tail=include_tail,
            meta={"model": "hurdle_gamma", "shape_buy": sh_b, "scale_buy": sc_b, "shape_sell": sh_s, "scale_sell": sc_s},
        )

    @staticmethod
    def pmf(k: int, p0: float, shape: float, scale: float) -> float:
        return _hurdle_gamma_pmf(k, p0, shape, scale)


class TorchHurdleWeibullModel:
    """
    Market order flow: hurdle Weibull. P(0)=p0; if >0 then Weibull(k, λ) (continuous, like TorchWeibullModel.sample).
    EWMA on zeros for p0 and on positive counts for (k, λ) per side.
    get_params() returns p0_*, wb_k_*, wb_lambda_*.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_p0: float = 0.3,
        init_mean: float = 10.0,
        init_k: float = 2.0,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        k0 = max(0.05, float(init_k))
        lam0 = _weibull_lambda_from_mean_k(float(init_mean), k0)
        self._p0_buy: float = max(0.0, min(1.0, float(init_p0)))
        self._p0_sell: float = max(0.0, min(1.0, float(init_p0)))
        self._k_buy: float = k0
        self._lambda_buy: float = lam0
        self._k_sell: float = k0
        self._lambda_sell: float = lam0
        self._S_buy: float = 0.0
        self._S2_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_sell: float = 0.0
        self._S2_sell: float = 0.0
        self._W_sell: float = 0.0
        self.p0_buy_hist: list[float] = []
        self.p0_sell_hist: list[float] = []
        self.k_buy_hist: list[float] = []
        self.k_sell_hist: list[float] = []
        self.lambda_buy_hist: list[float] = []
        self.lambda_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, _ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps

        z_buy = 1.0 if x_buy == 0 else 0.0
        self._p0_buy = g * self._p0_buy + (1.0 - g) * z_buy
        self._p0_buy = max(0.0, min(1.0, self._p0_buy))
        if x_buy > 0:
            v = float(x_buy)
            self._S_buy = g * self._S_buy + v
            self._S2_buy = g * self._S2_buy + v * v
            self._W_buy = g * self._W_buy + 1.0
        if self._W_buy > 1e-12:
            mean_buy = self._S_buy / self._W_buy
            var_buy = max(eps * eps, self._S2_buy / self._W_buy - mean_buy * mean_buy)
            cv2_buy = var_buy / max(eps * eps, mean_buy * mean_buy)
            self._k_buy = float(np.clip(_weibull_k_from_cv2(cv2_buy, self._k_buy), 0.05, 150.0))
            self._lambda_buy = _weibull_lambda_from_mean_k(mean_buy, self._k_buy)

        z_sell = 1.0 if x_sell == 0 else 0.0
        self._p0_sell = g * self._p0_sell + (1.0 - g) * z_sell
        self._p0_sell = max(0.0, min(1.0, self._p0_sell))
        if x_sell > 0:
            v = float(x_sell)
            self._S_sell = g * self._S_sell + v
            self._S2_sell = g * self._S2_sell + v * v
            self._W_sell = g * self._W_sell + 1.0
        if self._W_sell > 1e-12:
            mean_sell = self._S_sell / self._W_sell
            var_sell = max(eps * eps, self._S2_sell / self._W_sell - mean_sell * mean_sell)
            cv2_sell = var_sell / max(eps * eps, mean_sell * mean_sell)
            self._k_sell = float(np.clip(_weibull_k_from_cv2(cv2_sell, self._k_sell), 0.05, 150.0))
            self._lambda_sell = _weibull_lambda_from_mean_k(mean_sell, self._k_sell)

        self.p0_buy_hist.append(self._p0_buy)
        self.p0_sell_hist.append(self._p0_sell)
        self.k_buy_hist.append(self._k_buy)
        self.k_sell_hist.append(self._k_sell)
        self.lambda_buy_hist.append(self._lambda_buy)
        self.lambda_sell_hist.append(self._lambda_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (hurdle Weibull): P(0) (buy, sell)", "ylabel": "P(0)"},
            {"title": "Market order flow (hurdle Weibull): k (buy, sell)", "ylabel": "k"},
            {"title": "Market order flow (hurdle Weibull): λ (buy, sell)", "ylabel": "λ"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (hurdle Weibull, EWMA γ={self._gamma})",
            "ylabel": "P(0), k, λ",
            "series": [
                {"label": "p0_buy", "color": "green", "y": self.p0_buy_hist},
                {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist},
                {"label": "k_buy", "color": "green", "y": self.k_buy_hist},
                {"label": "k_sell", "color": "red", "y": self.k_sell_hist},
                {"label": "lambda_buy", "color": "green", "y": self.lambda_buy_hist},
                {"label": "lambda_sell", "color": "red", "y": self.lambda_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (hurdle Weibull): P(0)", "ylabel": "P(0)", "series": [{"label": "p0_buy", "color": "green", "y": self.p0_buy_hist}, {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist}]},
                {"title": "Market order flow (hurdle Weibull): k", "ylabel": "k", "series": [{"label": "k_buy", "color": "green", "y": self.k_buy_hist}, {"label": "k_sell", "color": "red", "y": self.k_sell_hist}]},
                {"title": "Market order flow (hurdle Weibull): λ", "ylabel": "λ", "series": [{"label": "lambda_buy", "color": "green", "y": self.lambda_buy_hist}, {"label": "lambda_sell", "color": "red", "y": self.lambda_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "p0_buy": self._p0_buy,
            "p0_sell": self._p0_sell,
            "wb_k_buy": self._k_buy,
            "wb_lambda_buy": self._lambda_buy,
            "wb_k_sell": self._k_sell,
            "wb_lambda_sell": self._lambda_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol): hurdle Bernoulli then Weibull, preserving exact zero mass."""
        import torch
        import torch.distributions as dist

        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        p0_buy = max(0.0, min(1.0, p["p0_buy"]))
        p0_sell = max(0.0, min(1.0, p["p0_sell"]))
        k_buy = max(1e-3, float(p["wb_k_buy"]))
        lam_buy = max(1e-9, float(p["wb_lambda_buy"]))
        k_sell = max(1e-3, float(p["wb_k_sell"]))
        lam_sell = max(1e-9, float(p["wb_lambda_sell"]))
        u_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        u_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        w_buy = dist.Weibull(
            scale=torch.tensor(lam_buy, dtype=_dtype, device=_device),
            concentration=torch.tensor(k_buy, dtype=_dtype, device=_device),
        )
        w_sell = dist.Weibull(
            scale=torch.tensor(lam_sell, dtype=_dtype, device=_device),
            concentration=torch.tensor(k_sell, dtype=_dtype, device=_device),
        )
        raw_buy = w_buy.sample(sample_shape).clamp(min=1e-6)
        raw_sell = w_sell.sample(sample_shape).clamp(min=1e-6)
        buy_vol = torch.where(u_buy <= p0_buy, torch.zeros_like(raw_buy), raw_buy)
        sell_vol = torch.where(u_sell <= p0_sell, torch.zeros_like(raw_sell), raw_sell)
        return buy_vol, sell_vol

    def stratified_scenarios(
        self,
        horizon: int = 4,
        positive_nodes: int = 3,
        device: str | None = None,
        dtype: Any = None,
        include_tail: bool = True,
    ) -> StratifiedScenarios:
        """Construct exact, weighted stratified scenarios over hurdle Weibull distribution."""
        p = self.get_params()
        p0_buy = max(0.0, min(1.0, float(p.get("p0_buy", self._p0_buy))))
        p0_sell = max(0.0, min(1.0, float(p.get("p0_sell", self._p0_sell))))
        k_b = max(1e-3, float(p.get("wb_k_buy", self._k_buy)))
        lam_b = max(1e-9, float(p.get("wb_lambda_buy", self._lambda_buy)))
        k_s = max(1e-3, float(p.get("wb_k_sell", self._k_sell)))
        lam_s = max(1e-9, float(p.get("wb_lambda_sell", self._lambda_sell)))
        return build_hurdle_joint_scenarios(
            p0_buy=p0_buy,
            p0_sell=p0_sell,
            buy_quantile_fn=lambda u: float(lam_b * (-np.log(max(1e-12, 1.0 - u))) ** (1.0 / k_b)),
            sell_quantile_fn=lambda u: float(lam_s * (-np.log(max(1e-12, 1.0 - u))) ** (1.0 / k_s)),
            horizon=horizon,
            positive_nodes=positive_nodes,
            device=device,
            dtype=dtype,
            include_tail=include_tail,
            meta={"model": "hurdle_weibull", "wb_k_buy": k_b, "wb_lambda_buy": lam_b, "wb_k_sell": k_s, "wb_lambda_sell": lam_s},
        )

    @staticmethod
    def pmf(k: int, p0: float, c: float, scale: float) -> float:
        return _hurdle_weibull_pmf(k, p0, c, scale)


class TorchHurdleLogNormalModel:
    """
    Market order flow: hurdle lognormal. P(0) = p0; if > 0 then round(LogNormal(μ, σ)).
    Params updated by EWMA. get_params() returns p0_buy, mu_buy, sigma_buy, p0_sell, mu_sell, sigma_sell.
    Enforces a variance prior floor (sigma_floor) to prevent probability collapse on illiquid books.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_p0: float = 0.3,
        init_mu: float = 2.0,
        init_sigma: float = 0.8,
        sigma_floor: float = 0.35,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        self._sigma_floor = max(float(param_eps), float(sigma_floor))
        self._p0_buy: float = max(0.0, min(1.0, float(init_p0)))
        self._p0_sell: float = max(0.0, min(1.0, float(init_p0)))
        self._S_log_buy: float = 0.0
        self._S2_log_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_log_sell: float = 0.0
        self._S2_log_sell: float = 0.0
        self._W_sell: float = 0.0
        self._mu_buy: float = float(init_mu)
        self._sigma_buy: float = max(self._sigma_floor, float(init_sigma))
        self._mu_sell: float = float(init_mu)
        self._sigma_sell: float = max(self._sigma_floor, float(init_sigma))
        self.p0_buy_hist: list[float] = []
        self.p0_sell_hist: list[float] = []
        self.mu_buy_hist: list[float] = []
        self.mu_sell_hist: list[float] = []
        self.sigma_buy_hist: list[float] = []
        self.sigma_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps
        z_buy = 1.0 if x_buy == 0 else 0.0
        self._p0_buy = g * self._p0_buy + (1.0 - g) * z_buy
        self._p0_buy = max(0.0, min(1.0, self._p0_buy))
        if x_buy > 0:
            log_x = np.log(max(1e-12, float(x_buy)))
            self._S_log_buy = g * self._S_log_buy + log_x
            self._S2_log_buy = g * self._S2_log_buy + log_x * log_x
            self._W_buy = g * self._W_buy + 1.0
        if self._W_buy > 1e-12:
            mean_log = self._S_log_buy / self._W_buy
            var_log = max(self._sigma_floor * self._sigma_floor, self._S2_log_buy / self._W_buy - mean_log * mean_log)
            self._mu_buy = mean_log
            self._sigma_buy = max(self._sigma_floor, np.sqrt(var_log))
        z_sell = 1.0 if x_sell == 0 else 0.0
        self._p0_sell = g * self._p0_sell + (1.0 - g) * z_sell
        self._p0_sell = max(0.0, min(1.0, self._p0_sell))
        if x_sell > 0:
            log_x = np.log(max(1e-12, float(x_sell)))
            self._S_log_sell = g * self._S_log_sell + log_x
            self._S2_log_sell = g * self._S2_log_sell + log_x * log_x
            self._W_sell = g * self._W_sell + 1.0
        if self._W_sell > 1e-12:
            mean_log = self._S_log_sell / self._W_sell
            var_log = max(self._sigma_floor * self._sigma_floor, self._S2_log_sell / self._W_sell - mean_log * mean_log)
            self._mu_sell = mean_log
            self._sigma_sell = max(self._sigma_floor, np.sqrt(var_log))
        self.p0_buy_hist.append(self._p0_buy)
        self.p0_sell_hist.append(self._p0_sell)
        self.mu_buy_hist.append(self._mu_buy)
        self.mu_sell_hist.append(self._mu_sell)
        self.sigma_buy_hist.append(self._sigma_buy)
        self.sigma_sell_hist.append(self._sigma_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (hurdle LN): P(0) (buy, sell)", "ylabel": "P(0)"},
            {"title": "Market order flow (hurdle LN): μ (buy, sell)", "ylabel": "μ"},
            {"title": "Market order flow (hurdle LN): σ (buy, sell)", "ylabel": "σ"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (hurdle LN, EWMA γ={self._gamma})",
            "ylabel": "P(0), μ, σ",
            "series": [
                {"label": "p0_buy", "color": "green", "y": self.p0_buy_hist},
                {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist},
                {"label": "mu_buy", "color": "green", "y": self.mu_buy_hist},
                {"label": "mu_sell", "color": "red", "y": self.mu_sell_hist},
                {"label": "sigma_buy", "color": "green", "y": self.sigma_buy_hist},
                {"label": "sigma_sell", "color": "red", "y": self.sigma_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (hurdle LN): P(0)", "ylabel": "P(0)", "series": [{"label": "p0_buy", "color": "green", "y": self.p0_buy_hist}, {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist}]},
                {"title": "Market order flow (hurdle LN): μ", "ylabel": "μ", "series": [{"label": "mu_buy", "color": "green", "y": self.mu_buy_hist}, {"label": "mu_sell", "color": "red", "y": self.mu_sell_hist}]},
                {"title": "Market order flow (hurdle LN): σ", "ylabel": "σ", "series": [{"label": "sigma_buy", "color": "green", "y": self.sigma_buy_hist}, {"label": "sigma_sell", "color": "red", "y": self.sigma_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "p0_buy": self._p0_buy,
            "p0_sell": self._p0_sell,
            "mu_buy": self._mu_buy,
            "sigma_buy": self._sigma_buy,
            "mu_sell": self._mu_sell,
            "sigma_sell": self._sigma_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol): hurdle then LogNormal, preserving exact zero mass."""
        import torch
        import torch.distributions as dist
        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        p0_buy = max(0.0, min(1.0, p["p0_buy"]))
        p0_sell = max(0.0, min(1.0, p["p0_sell"]))
        mu_buy, sigma_buy = p["mu_buy"], max(1e-6, p["sigma_buy"])
        mu_sell, sigma_sell = p["mu_sell"], max(1e-6, p["sigma_sell"])
        u_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        u_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        ln_buy = dist.LogNormal(torch.tensor(mu_buy, device=_device, dtype=_dtype), torch.tensor(sigma_buy, device=_device, dtype=_dtype))
        ln_sell = dist.LogNormal(torch.tensor(mu_sell, device=_device, dtype=_dtype), torch.tensor(sigma_sell, device=_device, dtype=_dtype))
        raw_buy = ln_buy.sample(sample_shape).clamp(min=1e-6)
        raw_sell = ln_sell.sample(sample_shape).clamp(min=1e-6)
        buy_vol = torch.where(u_buy <= p0_buy, torch.zeros_like(raw_buy), raw_buy)
        sell_vol = torch.where(u_sell <= p0_sell, torch.zeros_like(raw_sell), raw_sell)
        return buy_vol, sell_vol

    def stratified_scenarios(
        self,
        horizon: int = 4,
        positive_nodes: int = 3,
        device: str | None = None,
        dtype: Any = None,
        include_tail: bool = True,
    ) -> StratifiedScenarios:
        """Construct exact, weighted stratified scenarios over the hurdle distribution."""
        from scipy.special import ndtri

        p = self.get_params()
        p0_buy = max(0.0, min(1.0, float(p["p0_buy"])))
        p0_sell = max(0.0, min(1.0, float(p["p0_sell"])))
        mu_buy, sigma_buy = float(p["mu_buy"]), max(1e-6, float(p["sigma_buy"]))
        mu_sell, sigma_sell = float(p["mu_sell"]), max(1e-6, float(p["sigma_sell"]))

        def buy_q(u: float) -> float:
            z = float(ndtri(u))
            return float(np.exp(mu_buy + sigma_buy * z))

        def sell_q(u: float) -> float:
            z = float(ndtri(u))
            return float(np.exp(mu_sell + sigma_sell * z))

        res = build_hurdle_joint_scenarios(
            p0_buy=p0_buy,
            p0_sell=p0_sell,
            buy_quantile_fn=buy_q,
            sell_quantile_fn=sell_q,
            horizon=horizon,
            positive_nodes=positive_nodes,
            device=device,
            dtype=dtype,
            include_tail=include_tail,
            meta={
                "model": "hurdle_lognormal",
                "mu_buy": mu_buy,
                "sigma_buy": sigma_buy,
                "mu_sell": mu_sell,
                "sigma_sell": sigma_sell,
            },
        )
        res.mu_buy = mu_buy
        res.sigma_buy = sigma_buy
        res.mu_sell = mu_sell
        res.sigma_sell = sigma_sell
        return res

    @staticmethod
    def pmf(k: int, p0: float, mu: float, sigma: float) -> float:
        return _hurdle_lognormal_pmf(k, p0, mu, sigma)


class TorchHurdleLogStudentTModel:
    """
    Market order flow: hurdle log-Student-t. P(0) = p0; if > 0 then round(exp(Z)) with Z ~ t(df, μ, σ).
    Same EWMA moments on log(·) for μ, σ as hurdle lognormal; df is fixed (init).
    get_params() includes logt_df for NLL dispatch plus p0_*, mu_*, sigma_* per side.
    Enforces a variance prior floor (sigma_floor) to prevent probability collapse on illiquid books.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_p0: float = 0.3,
        init_mu: float = 2.0,
        init_sigma: float = 0.8,
        init_df: float = 5.0,
        sigma_floor: float = 0.35,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        self._sigma_floor = max(float(param_eps), float(sigma_floor))
        self._df: float = max(1.1, float(init_df))
        self._p0_buy: float = max(0.0, min(1.0, float(init_p0)))
        self._p0_sell: float = max(0.0, min(1.0, float(init_p0)))
        self._S_log_buy: float = 0.0
        self._S2_log_buy: float = 0.0
        self._W_buy: float = 0.0
        self._S_log_sell: float = 0.0
        self._S2_log_sell: float = 0.0
        self._W_sell: float = 0.0
        self._mu_buy: float = float(init_mu)
        self._sigma_buy: float = max(self._sigma_floor, float(init_sigma))
        self._mu_sell: float = float(init_mu)
        self._sigma_sell: float = max(self._sigma_floor, float(init_sigma))
        self.p0_buy_hist: list[float] = []
        self.p0_sell_hist: list[float] = []
        self.mu_buy_hist: list[float] = []
        self.mu_sell_hist: list[float] = []
        self.sigma_buy_hist: list[float] = []
        self.sigma_sell_hist: list[float] = []
        self.df_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g, eps = self._gamma, self._param_eps
        z_buy = 1.0 if x_buy == 0 else 0.0
        self._p0_buy = g * self._p0_buy + (1.0 - g) * z_buy
        self._p0_buy = max(0.0, min(1.0, self._p0_buy))
        if x_buy > 0:
            log_x = np.log(max(1e-12, float(x_buy)))
            self._S_log_buy = g * self._S_log_buy + log_x
            self._S2_log_buy = g * self._S2_log_buy + log_x * log_x
            self._W_buy = g * self._W_buy + 1.0
        if self._W_buy > 1e-12:
            mean_log = self._S_log_buy / self._W_buy
            var_log = max(self._sigma_floor * self._sigma_floor, self._S2_log_buy / self._W_buy - mean_log * mean_log)
            self._mu_buy = mean_log
            self._sigma_buy = max(self._sigma_floor, np.sqrt(var_log))
        z_sell = 1.0 if x_sell == 0 else 0.0
        self._p0_sell = g * self._p0_sell + (1.0 - g) * z_sell
        self._p0_sell = max(0.0, min(1.0, self._p0_sell))
        if x_sell > 0:
            log_x = np.log(max(1e-12, float(x_sell)))
            self._S_log_sell = g * self._S_log_sell + log_x
            self._S2_log_sell = g * self._S2_log_sell + log_x * log_x
            self._W_sell = g * self._W_sell + 1.0
        if self._W_sell > 1e-12:
            mean_log = self._S_log_sell / self._W_sell
            var_log = max(self._sigma_floor * self._sigma_floor, self._S2_log_sell / self._W_sell - mean_log * mean_log)
            self._mu_sell = mean_log
            self._sigma_sell = max(self._sigma_floor, np.sqrt(var_log))
        self.p0_buy_hist.append(self._p0_buy)
        self.p0_sell_hist.append(self._p0_sell)
        self.mu_buy_hist.append(self._mu_buy)
        self.mu_sell_hist.append(self._mu_sell)
        self.sigma_buy_hist.append(self._sigma_buy)
        self.sigma_sell_hist.append(self._sigma_sell)
        self.df_hist.append(self._df)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (hurdle log-t): P(0) (buy, sell)", "ylabel": "P(0)"},
            {"title": "Market order flow (hurdle log-t): μ (buy, sell)", "ylabel": "μ"},
            {"title": "Market order flow (hurdle log-t): σ (buy, sell)", "ylabel": "σ"},
            {"title": "Market order flow (hurdle log-t): df", "ylabel": "df"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (hurdle log-t, EWMA γ={self._gamma}, df={self._df})",
            "ylabel": "P(0), μ, σ, df",
            "series": [
                {"label": "p0_buy", "color": "green", "y": self.p0_buy_hist},
                {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist},
                {"label": "mu_buy", "color": "green", "y": self.mu_buy_hist},
                {"label": "mu_sell", "color": "red", "y": self.mu_sell_hist},
                {"label": "sigma_buy", "color": "green", "y": self.sigma_buy_hist},
                {"label": "sigma_sell", "color": "red", "y": self.sigma_sell_hist},
                {"label": "df", "color": "blue", "y": self.df_hist},
            ],
            "subplots": [
                {"title": "Market order flow (hurdle log-t): P(0)", "ylabel": "P(0)", "series": [{"label": "p0_buy", "color": "green", "y": self.p0_buy_hist}, {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist}]},
                {"title": "Market order flow (hurdle log-t): μ", "ylabel": "μ", "series": [{"label": "mu_buy", "color": "green", "y": self.mu_buy_hist}, {"label": "mu_sell", "color": "red", "y": self.mu_sell_hist}]},
                {"title": "Market order flow (hurdle log-t): σ", "ylabel": "σ", "series": [{"label": "sigma_buy", "color": "green", "y": self.sigma_buy_hist}, {"label": "sigma_sell", "color": "red", "y": self.sigma_sell_hist}]},
                {"title": "Market order flow (hurdle log-t): df", "ylabel": "df", "series": [{"label": "df", "color": "blue", "y": self.df_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "p0_buy": self._p0_buy,
            "p0_sell": self._p0_sell,
            "mu_buy": self._mu_buy,
            "sigma_buy": self._sigma_buy,
            "mu_sell": self._mu_sell,
            "sigma_sell": self._sigma_sell,
            "logt_df": self._df,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol): hurdle then exp(Student-t), preserving exact zero mass."""
        import torch
        import torch.distributions as dist

        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        p0_buy = max(0.0, min(1.0, p["p0_buy"]))
        p0_sell = max(0.0, min(1.0, p["p0_sell"]))
        mu_buy, sigma_buy = p["mu_buy"], max(1e-6, p["sigma_buy"])
        mu_sell, sigma_sell = p["mu_sell"], max(1e-6, p["sigma_sell"])
        df = max(1.1, float(self._df))
        df_t = torch.tensor(df, device=_device, dtype=_dtype)
        u_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        u_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        st_buy = dist.StudentT(
            df_t,
            loc=torch.tensor(mu_buy, device=_device, dtype=_dtype),
            scale=torch.tensor(sigma_buy, device=_device, dtype=_dtype),
        )
        st_sell = dist.StudentT(
            df_t,
            loc=torch.tensor(mu_sell, device=_device, dtype=_dtype),
            scale=torch.tensor(sigma_sell, device=_device, dtype=_dtype),
        )
        z_buy = st_buy.sample(sample_shape)
        z_sell = st_sell.sample(sample_shape)
        raw_buy = torch.exp(z_buy).clamp(min=1e-6)
        raw_sell = torch.exp(z_sell).clamp(min=1e-6)
        buy_vol = torch.where(u_buy <= p0_buy, torch.zeros_like(raw_buy), raw_buy)
        sell_vol = torch.where(u_sell <= p0_sell, torch.zeros_like(raw_sell), raw_sell)
        return buy_vol, sell_vol

    def stratified_scenarios(
        self,
        horizon: int = 4,
        positive_nodes: int = 3,
        device: str | None = None,
        dtype: Any = None,
        include_tail: bool = True,
    ) -> StratifiedScenarios:
        """Construct exact, weighted stratified scenarios over hurdle log-Student-t distribution."""
        from scipy.stats import t as student_t

        p = self.get_params()
        p0_buy = max(0.0, min(1.0, float(p["p0_buy"])))
        p0_sell = max(0.0, min(1.0, float(p["p0_sell"])))
        mu_b, sig_b = float(p["mu_buy"]), max(1e-6, float(p["sigma_buy"]))
        mu_s, sig_s = float(p["mu_sell"]), max(1e-6, float(p["sigma_sell"]))
        df = max(1.1, float(self._df))
        return build_hurdle_joint_scenarios(
            p0_buy=p0_buy,
            p0_sell=p0_sell,
            buy_quantile_fn=lambda u: float(np.exp(mu_b + sig_b * student_t.ppf(u, df=df))),
            sell_quantile_fn=lambda u: float(np.exp(mu_s + sig_s * student_t.ppf(u, df=df))),
            horizon=horizon,
            positive_nodes=positive_nodes,
            device=device,
            dtype=dtype,
            include_tail=include_tail,
            meta={"model": "hurdle_logstudent_t", "df": df, "mu_buy": mu_b, "sigma_buy": sig_b},
        )

    @staticmethod
    def pmf(k: int, p0: float, df: float, mu: float, sigma: float) -> float:
        return _hurdle_logt_pmf(k, p0, df, mu, sigma)


class TorchHurdleGeneralizedParetoModel:
    """
    Market order flow: hurdle generalized Pareto. P(0)=p0; if >0 then round(U + GPD(xi, beta)).
    Params updated by EWMA moments of positive exceedances over threshold u.
    get_params() returns p0_*, xi_*, beta_*, u_* per side.
    """

    def __init__(
        self,
        decay: float = 0.99,
        param_eps: float = 1e-6,
        device: str | None = None,
        init_p0: float = 0.3,
        init_xi: float = 0.1,
        init_beta: float = 3.0,
        init_u: float = 1.0,
    ):
        self._gamma = max(1e-6, min(1.0, float(decay)))
        self._param_eps = float(param_eps)
        self._p0_buy: float = max(0.0, min(1.0, float(init_p0)))
        self._p0_sell: float = max(0.0, min(1.0, float(init_p0)))
        self._xi_buy: float = float(np.clip(float(init_xi), -0.45, 0.45))
        self._xi_sell: float = float(np.clip(float(init_xi), -0.45, 0.45))
        self._beta_buy: float = max(self._param_eps, float(init_beta))
        self._beta_sell: float = max(self._param_eps, float(init_beta))
        self._u_buy: float = max(1.0, float(init_u))
        self._u_sell: float = max(1.0, float(init_u))
        self._S_ex_buy: float = 0.0
        self._S2_ex_buy: float = 0.0
        self._S_ex_sell: float = 0.0
        self._S2_ex_sell: float = 0.0
        self._W_buy: float = 0.0
        self._W_sell: float = 0.0
        self.p0_buy_hist: list[float] = []
        self.p0_sell_hist: list[float] = []
        self.xi_buy_hist: list[float] = []
        self.xi_sell_hist: list[float] = []
        self.beta_buy_hist: list[float] = []
        self.beta_sell_hist: list[float] = []
        self.u_buy_hist: list[float] = []
        self.u_sell_hist: list[float] = []

    def update(self, prev_step: Step | None, curr_step: Step, **_: Any) -> None:
        _t_end, _ob, buy_vol, sell_vol, *_ = curr_step
        x_buy = int(round(buy_vol)) if isinstance(buy_vol, (int, float)) else int(buy_vol)
        x_sell = int(round(sell_vol)) if isinstance(sell_vol, (int, float)) else int(sell_vol)
        g = self._gamma
        eps = self._param_eps

        z_buy = 1.0 if x_buy == 0 else 0.0
        self._p0_buy = g * self._p0_buy + (1.0 - g) * z_buy
        self._p0_buy = max(0.0, min(1.0, self._p0_buy))
        if x_buy > self._u_buy:
            ex = max(eps, float(x_buy) - self._u_buy)
            self._S_ex_buy = g * self._S_ex_buy + ex
            self._S2_ex_buy = g * self._S2_ex_buy + ex * ex
            self._W_buy = g * self._W_buy + 1.0
        if self._W_buy > 1e-12:
            m = max(eps, self._S_ex_buy / self._W_buy)
            v = max(eps * eps, self._S2_ex_buy / self._W_buy - m * m)
            r = max(1e-9, v / (m * m))
            xi_hat = 0.5 * (1.0 - 1.0 / r)
            xi_hat = float(np.clip(xi_hat, -0.45, 0.45))
            beta_hat = max(eps, m * (1.0 - xi_hat))
            self._xi_buy = xi_hat
            self._beta_buy = beta_hat

        z_sell = 1.0 if x_sell == 0 else 0.0
        self._p0_sell = g * self._p0_sell + (1.0 - g) * z_sell
        self._p0_sell = max(0.0, min(1.0, self._p0_sell))
        if x_sell > self._u_sell:
            ex = max(eps, float(x_sell) - self._u_sell)
            self._S_ex_sell = g * self._S_ex_sell + ex
            self._S2_ex_sell = g * self._S2_ex_sell + ex * ex
            self._W_sell = g * self._W_sell + 1.0
        if self._W_sell > 1e-12:
            m = max(eps, self._S_ex_sell / self._W_sell)
            v = max(eps * eps, self._S2_ex_sell / self._W_sell - m * m)
            r = max(1e-9, v / (m * m))
            xi_hat = 0.5 * (1.0 - 1.0 / r)
            xi_hat = float(np.clip(xi_hat, -0.45, 0.45))
            beta_hat = max(eps, m * (1.0 - xi_hat))
            self._xi_sell = xi_hat
            self._beta_sell = beta_hat

        self.p0_buy_hist.append(self._p0_buy)
        self.p0_sell_hist.append(self._p0_sell)
        self.xi_buy_hist.append(self._xi_buy)
        self.xi_sell_hist.append(self._xi_sell)
        self.beta_buy_hist.append(self._beta_buy)
        self.beta_sell_hist.append(self._beta_sell)
        self.u_buy_hist.append(self._u_buy)
        self.u_sell_hist.append(self._u_sell)

    def get_plot_specs(self) -> list[dict]:
        return [
            {"title": "Market order flow (hurdle GPD): P(0) (buy, sell)", "ylabel": "P(0)"},
            {"title": "Market order flow (hurdle GPD): xi (buy, sell)", "ylabel": "xi"},
            {"title": "Market order flow (hurdle GPD): beta (buy, sell)", "ylabel": "beta"},
            {"title": "Market order flow (hurdle GPD): u (buy, sell)", "ylabel": "u"},
        ]

    def get_plot_data(self) -> dict:
        return {
            "title": f"Market order flow (hurdle GPD, EWMA γ={self._gamma})",
            "ylabel": "P(0), xi, beta, u",
            "series": [
                {"label": "p0_buy", "color": "green", "y": self.p0_buy_hist},
                {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist},
                {"label": "xi_buy", "color": "green", "y": self.xi_buy_hist},
                {"label": "xi_sell", "color": "red", "y": self.xi_sell_hist},
                {"label": "beta_buy", "color": "green", "y": self.beta_buy_hist},
                {"label": "beta_sell", "color": "red", "y": self.beta_sell_hist},
                {"label": "u_buy", "color": "green", "y": self.u_buy_hist},
                {"label": "u_sell", "color": "red", "y": self.u_sell_hist},
            ],
            "subplots": [
                {"title": "Market order flow (hurdle GPD): P(0)", "ylabel": "P(0)", "series": [{"label": "p0_buy", "color": "green", "y": self.p0_buy_hist}, {"label": "p0_sell", "color": "red", "y": self.p0_sell_hist}]},
                {"title": "Market order flow (hurdle GPD): xi", "ylabel": "xi", "series": [{"label": "xi_buy", "color": "green", "y": self.xi_buy_hist}, {"label": "xi_sell", "color": "red", "y": self.xi_sell_hist}]},
                {"title": "Market order flow (hurdle GPD): beta", "ylabel": "beta", "series": [{"label": "beta_buy", "color": "green", "y": self.beta_buy_hist}, {"label": "beta_sell", "color": "red", "y": self.beta_sell_hist}]},
                {"title": "Market order flow (hurdle GPD): u", "ylabel": "u", "series": [{"label": "u_buy", "color": "green", "y": self.u_buy_hist}, {"label": "u_sell", "color": "red", "y": self.u_sell_hist}]},
            ],
        }

    def get_params(self) -> dict:
        return {
            "p0_buy": self._p0_buy,
            "p0_sell": self._p0_sell,
            "xi_buy": self._xi_buy,
            "xi_sell": self._xi_sell,
            "beta_buy": self._beta_buy,
            "beta_sell": self._beta_sell,
            "u_buy": self._u_buy,
            "u_sell": self._u_sell,
        }

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        device: str | None = None,
        dtype: "np.dtype | None" = None,
    ):
        """Sample (buy_vol, sell_vol): hurdle then generalized Pareto, preserving exact zero mass."""
        import torch

        p = self.get_params()
        _device = resolve_torch_device(device)
        _dtype = dtype if dtype is not None else torch.float64
        p0_buy = max(0.0, min(1.0, p["p0_buy"]))
        p0_sell = max(0.0, min(1.0, p["p0_sell"]))
        xi_buy = float(np.clip(float(p["xi_buy"]), -0.45, 0.95))
        xi_sell = float(np.clip(float(p["xi_sell"]), -0.45, 0.95))
        beta_buy = max(1e-9, float(p["beta_buy"]))
        beta_sell = max(1e-9, float(p["beta_sell"]))
        u_thr_buy = max(1e-9, float(p["u_buy"]))
        u_thr_sell = max(1e-9, float(p["u_sell"]))
        u_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        u_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        w_buy = torch.rand(sample_shape, device=_device, dtype=_dtype)
        w_sell = torch.rand(sample_shape, device=_device, dtype=_dtype)
        if abs(xi_buy) < 1e-9:
            ex_buy = -beta_buy * torch.log(torch.clamp(1.0 - w_buy, min=1e-12))
        else:
            ex_buy = (beta_buy / xi_buy) * (torch.pow(torch.clamp(1.0 - w_buy, min=1e-12), -xi_buy) - 1.0)
        if abs(xi_sell) < 1e-9:
            ex_sell = -beta_sell * torch.log(torch.clamp(1.0 - w_sell, min=1e-12))
        else:
            ex_sell = (beta_sell / xi_sell) * (torch.pow(torch.clamp(1.0 - w_sell, min=1e-12), -xi_sell) - 1.0)
        raw_buy = (u_thr_buy + ex_buy).clamp(min=1e-6)
        raw_sell = (u_thr_sell + ex_sell).clamp(min=1e-6)
        buy_vol = torch.where(u_buy <= p0_buy, torch.zeros_like(raw_buy), raw_buy)
        sell_vol = torch.where(u_sell <= p0_sell, torch.zeros_like(raw_sell), raw_sell)
        return buy_vol, sell_vol

    def stratified_scenarios(
        self,
        horizon: int = 4,
        positive_nodes: int = 3,
        device: str | None = None,
        dtype: Any = None,
        include_tail: bool = True,
    ) -> StratifiedScenarios:
        """Construct exact, weighted stratified scenarios over hurdle GPD distribution."""
        p = self.get_params()
        p0_buy = max(0.0, min(1.0, float(p["p0_buy"])))
        p0_sell = max(0.0, min(1.0, float(p["p0_sell"])))
        xi_b = float(np.clip(float(p["xi_buy"]), -0.45, 0.95))
        xi_s = float(np.clip(float(p["xi_sell"]), -0.45, 0.95))
        beta_b = max(1e-9, float(p["beta_buy"]))
        beta_s = max(1e-9, float(p["beta_sell"]))
        u_b = max(1e-9, float(p["u_buy"]))
        u_s = max(1e-9, float(p["u_sell"]))

        def gpd_q(u: float, xi: float, beta: float, loc: float) -> float:
            if abs(xi) < 1e-9:
                return float(loc - beta * np.log(max(1e-12, 1.0 - u)))
            return float(loc + (beta / xi) * (max(1e-12, 1.0 - u) ** (-xi) - 1.0))

        return build_hurdle_joint_scenarios(
            p0_buy=p0_buy,
            p0_sell=p0_sell,
            buy_quantile_fn=lambda u: gpd_q(u, xi_b, beta_b, u_b),
            sell_quantile_fn=lambda u: gpd_q(u, xi_s, beta_s, u_s),
            horizon=horizon,
            positive_nodes=positive_nodes,
            device=device,
            dtype=dtype,
            include_tail=include_tail,
            meta={"model": "hurdle_gpd", "xi_buy": xi_b, "beta_buy": beta_b, "u_buy": u_b},
        )

    @staticmethod
    def pmf(k: int, p0: float, xi: float, beta: float, u: float) -> float:
        return _hurdle_generalized_pareto_pmf(k, p0, xi, beta, u)

def score_flow_observation(
    buy_volume: float,
    sell_volume: float,
    params: dict,
    *,
    kind: str,
) -> "ScoreResult":
    """Pre-update market-flow score (see ``mm.models.scoring``)."""

    from mm.models.scoring import ScoreResult, score_market_flow

    return score_market_flow(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        params_before=params,
        kind=kind,
    )
