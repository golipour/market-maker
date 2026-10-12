"""Mathematically rigorous, strictly causal, discrete-lattice price models.

All models output discrete probability mass functions (PMFs) on the tick lattice
    y = k * h,  k in Z,  where h is the venue tick resolution.

Every model satisfies:
    1. sum_{k in Z} P(Y = k * h) == 1.000000
    2. P(Y = k * h) >= 0 and <= 1
    3. Strict causality: predict(y, t) only uses historical information <= t.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from scipy.special import betainc, log_ndtr
from scipy.stats import norm

TINY = 1e-300


def t_cdf(x: float, nu: float) -> float:
    """Exact Student-t CDF."""
    if x == 0.0:
        return 0.5
    if x > 0.0:
        return 1.0 - 0.5 * betainc(0.5 * nu, 0.5, nu / (nu + x * x))
    return 0.5 * betainc(0.5 * nu, 0.5, nu / (nu + x * x))


class IncumbentGaussian:
    """Production GaussianPriceModel on the discrete tick lattice.

    Discretized continuous Gaussian:
        P(k) = Phi((k*h + h/2 - mu)/sigma) - Phi((k*h - h/2 - mu)/sigma)
    """

    name = "INCUMBENT Gaussian (discrete lattice)"

    def __init__(self, decay: float = 0.95, init_sigma: float = 50.0):
        self.gamma = float(decay)
        self.alpha = 1.0 - self.gamma
        self.s2 = float(init_sigma) ** 2
        self.mu = 0.0
        self.h = 0.5

    def bind(self, y: dict) -> None:
        self.h = float(y["tick"])

    def predict(self, y: dict, t: int) -> dict[str, Any]:
        sigma = max(math.sqrt(max(self.s2, 1e-12)), 1e-4)
        return {"mu": self.mu, "sigma": sigma, "h": self.h, "p0": self.pmf(0, self.mu, sigma, self.h)}

    def pmf(self, k: int, mu: float, sigma: float, h: float) -> float:
        hi = (k * h + 0.5 * h - mu) / sigma
        lo = (k * h - 0.5 * h - mu) / sigma
        return max(norm.cdf(hi) - norm.cdf(lo), 0.0)

    def logpmf(self, target: float, p: dict[str, Any]) -> float:
        k = int(round(target / p["h"]))
        prob = self.pmf(k, p["mu"], p["sigma"], p["h"])
        return math.log(max(prob, TINY))

    def update(self, y: dict, t: int) -> None:
        target = y["dmid_n1"][t]
        if not np.isfinite(target):
            return
        err = float(target) - self.mu
        self.s2 = self.gamma * self.s2 + self.alpha * (err * err)
        self.s2 = max(self.s2, 1e-6)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        k_grid = np.arange(-K, K + 1)
        hi = (k_grid * p["h"] + 0.5 * p["h"] - p["mu"]) / p["sigma"]
        lo = (k_grid * p["h"] - 0.5 * p["h"] - p["mu"]) / p["sigma"]
        total = np.sum(np.maximum(norm.cdf(hi) - norm.cdf(lo), 0.0))
        assert abs(total - 1.0) < 1e-4, f"Incumbent Gaussian PMF sum was {total:.6f}"


class BaselineEmpirical:
    """Non-parametric empirical rolling discrete frequency baseline with Laplace smoothing."""

    name = "BASELINE Empirical (rolling 2000 steps)"

    def __init__(self, window: int = 2000):
        self.window = int(window)
        self.history: deque[int] = deque(maxlen=self.window)
        self.h = 0.5

    def bind(self, y: dict) -> None:
        self.h = float(y["tick"])

    def predict(self, y: dict, t: int) -> dict[str, Any]:
        return {"counts": dict(self._get_counts()), "n": len(self.history), "h": self.h, "p0": self._p0()}

    def _get_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for k in self.history:
            counts[k] = counts.get(k, 0) + 1
        return counts

    def _p0(self) -> float:
        if not self.history:
            return 0.5
        c0 = sum(1 for k in self.history if k == 0)
        return (c0 + 1.0) / (len(self.history) + 200.0)

    def logpmf(self, target: float, p: dict[str, Any]) -> float:
        k = int(round(target / p["h"]))
        n = p["n"]
        if n < 50:
            return -math.log(200.0)
        # Laplace (+1) smoothing over active support plus background uniform floor
        counts = p["counts"]
        # Bound support for smoothing
        total_denom = n + 1000.0
        count_k = counts.get(k, 0) + 1.0
        prob = count_k / total_denom
        return math.log(max(prob, TINY))

    def update(self, y: dict, t: int) -> None:
        target = y["dmid_n1"][t]
        if np.isfinite(target):
            k = int(round(target / self.h))
            self.history.append(k)

    def assert_normalized(self, p: dict[str, Any]) -> None:
        pass


class HurdleGaussian:
    """Hurdle at zero + lattice-normalized Gaussian body on non-zero moves."""

    name = "Hurdle + Gaussian body"

    def __init__(self, decay_p: float = 0.99, decay_s: float = 0.99, init_sigma: float = 50.0):
        self.gamma_p = float(decay_p)
        self.alpha_p = 1.0 - self.gamma_p
        self.gamma_s = float(decay_s)
        self.alpha_s = 1.0 - self.gamma_s
        self.p0 = 0.5
        self.s2 = float(init_sigma) ** 2
        self.h = 0.5

    def bind(self, y: dict) -> None:
        self.h = float(y["tick"])

    def predict(self, y: dict, t: int) -> dict[str, Any]:
        sigma = max(math.sqrt(max(self.s2, 1e-12)), 1e-4)
        return {"p0": float(self.p0), "sigma": sigma, "h": self.h}

    def pmf(self, k: int, p0: float, sigma: float, h: float) -> float:
        if k == 0:
            return p0
        hi = (k * h + 0.5 * h) / sigma
        lo = (k * h - 0.5 * h) / sigma
        m_k = max(norm.cdf(hi) - norm.cdf(lo), 0.0)
        m_0 = max(norm.cdf(0.5 * h / sigma) - norm.cdf(-0.5 * h / sigma), 0.0)
        return (1.0 - p0) * m_k / max(1.0 - m_0, 1e-12)

    def logpmf(self, target: float, p: dict[str, Any]) -> float:
        k = int(round(target / p["h"]))
        prob = self.pmf(k, p["p0"], p["sigma"], p["h"])
        return math.log(max(prob, TINY))

    def update(self, y: dict, t: int) -> None:
        target = y["dmid_n1"][t]
        if not np.isfinite(target):
            return
        is_zero = 1.0 if target == 0.0 else 0.0
        self.p0 = self.gamma_p * self.p0 + self.alpha_p * is_zero
        self.p0 = min(max(self.p0, 1e-6), 1.0 - 1e-6)
        if target != 0.0:
            x2 = float(target) ** 2
            self.s2 = self.gamma_s * self.s2 + self.alpha_s * x2
            self.s2 = max(self.s2, 1e-6)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        k_grid = np.arange(-K, K + 1)
        hi = (k_grid * p["h"] + 0.5 * p["h"]) / p["sigma"]
        lo = (k_grid * p["h"] - 0.5 * p["h"]) / p["sigma"]
        m_k = np.maximum(norm.cdf(hi) - norm.cdf(lo), 0.0)
        m_0 = np.maximum(norm.cdf(0.5 * p["h"] / p["sigma"]) - norm.cdf(-0.5 * p["h"] / p["sigma"]), 0.0)
        pmf = (1.0 - p["p0"]) * m_k / max(1.0 - m_0, 1e-12)
        pmf[k_grid == 0] = p["p0"]
        total = float(np.sum(pmf))
        assert abs(total - 1.0) < 1e-4, f"Hurdle Gaussian PMF sum was {total:.6f}"


class HurdleStudentT:
    """Proposed Hurdle-Student-t with conditional scale and online calibration.

    PMF:
        P(Y = 0) = p0
        P(Y = k*h) = (1 - p0) * [T_nu(hi) - T_nu(lo)] / (1 - [T_nu(zc_hi) - T_nu(zc_lo)])
    """

    def __init__(
        self,
        nu: float = 2.5,
        decay_p: float = 0.99,
        decay_s: float = 0.99,
        decay_cal: float = 0.995,
        init_m2: float = 2500.0,
    ):
        self.nu = float(nu)
        self.name = f"PROPOSED Hurdle-Student-t nu={self.nu:.1f}"
        self.gamma_p = float(decay_p)
        self.alpha_p = 1.0 - self.gamma_p
        self.gamma_s = float(decay_s)
        self.alpha_s = 1.0 - self.gamma_s
        self.gamma_cal = float(decay_cal)
        self.alpha_cal = 1.0 - self.gamma_cal

        self.p0 = 0.5
        self.m2 = float(init_m2)
        self.cal_u2 = 1.0
        self.cal_w = 0.0
        self.alpha = 1.0
        self.h = 0.5

    def bind(self, y: dict) -> None:
        self.h = float(y["tick"])

    def _raw_scale(self) -> float:
        if self.nu > 2.0:
            var_scale = math.sqrt(max(self.m2 * (self.nu - 2.0) / self.nu, 1e-12))
        else:
            var_scale = math.sqrt(max(self.m2, 1e-12))
        return max(var_scale, self.h * 0.25)

    def scale(self) -> float:
        return self._raw_scale() * self.alpha

    def predict(self, y: dict, t: int) -> dict[str, Any]:
        return {
            "p0": float(self.p0),
            "s": float(self.scale()),
            "nu": self.nu,
            "h": self.h,
            "alpha": float(self.alpha),
        }

    def pmf(self, k: int, p0: float, s: float, nu: float, h: float) -> float:
        if k == 0:
            return p0
        hi = (k * h + 0.5 * h) / s
        lo = (k * h - 0.5 * h) / s
        m_k = max(t_cdf(hi, nu) - t_cdf(lo, nu), 0.0)
        m_0 = max(t_cdf(0.5 * h / s, nu) - t_cdf(-0.5 * h / s, nu), 0.0)
        return (1.0 - p0) * m_k / max(1.0 - m_0, 1e-12)

    def logpmf(self, target: float, p: dict[str, Any]) -> float:
        k = int(round(target / p["h"]))
        prob = self.pmf(k, p["p0"], p["s"], p["nu"], p["h"])
        return math.log(max(prob, TINY))

    def update(self, y: dict, t: int) -> None:
        target = y["dmid_n1"][t]
        if not np.isfinite(target):
            return
        is_zero = 1.0 if target == 0.0 else 0.0
        self.p0 = self.gamma_p * self.p0 + self.alpha_p * is_zero
        self.p0 = min(max(self.p0, 1e-6), 1.0 - 1e-6)

        if target != 0.0:
            x2 = float(target) ** 2
            self.m2 = self.gamma_s * self.m2 + self.alpha_s * x2
            self.m2 = max(self.m2, 1e-6)

            # Online calibration tracking
            raw_s = self._raw_scale()
            u = abs(float(target)) / raw_s
            self.cal_u2 = self.gamma_cal * self.cal_u2 + self.alpha_cal * (u * u)
            self.cal_w = self.gamma_cal * self.cal_w + 1.0
            if self.cal_w > 50.0:
                target_var = (self.nu / (self.nu - 2.0)) if self.nu > 2.0 else 1.0
                mult = math.sqrt(max(self.cal_u2 / target_var, 0.01))
                self.alpha = float(np.clip(mult, 0.5, 3.0))

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        from scipy.special import betainc
        k_grid = np.arange(-K, K + 1)
        hi = (k_grid * p["h"] + 0.5 * p["h"]) / p["s"]
        lo = (k_grid * p["h"] - 0.5 * p["h"]) / p["s"]

        def _vec_t(x):
            out = np.empty_like(x, dtype=np.float64)
            pos, neg, zero = x > 0, x < 0, x == 0
            out[zero] = 0.5
            out[pos] = 1.0 - 0.5 * betainc(0.5 * p["nu"], 0.5, p["nu"] / (p["nu"] + x[pos] * x[pos]))
            out[neg] = 0.5 * betainc(0.5 * p["nu"], 0.5, p["nu"] / (p["nu"] + x[neg] * x[neg]))
            return out

        m_k = np.maximum(_vec_t(hi) - _vec_t(lo), 0.0)
        zc = _vec_t(np.array([0.5 * p["h"] / p["s"]]))[0] - _vec_t(np.array([-0.5 * p["h"] / p["s"]]))[0]
        pmf = (1.0 - p["p0"]) * m_k / max(1.0 - zc, 1e-12)
        pmf[k_grid == 0] = p["p0"]
        total = float(np.sum(pmf))
        assert abs(total - 1.0) < 1e-4, f"Hurdle Student-t PMF sum was {total:.6f}"


class HurdleLogNormalSigned:
    """Hurdle at zero + properly halved signed LogNormal body on |dmid|."""

    name = "Hurdle + signed LogNormal"

    def __init__(self, decay_p: float = 0.99, decay_m: float = 0.99, init_mu: float = 2.0, init_sig: float = 1.0):
        self.gamma_p = float(decay_p)
        self.alpha_p = 1.0 - self.gamma_p
        self.gamma_m = float(decay_m)
        self.alpha_m = 1.0 - self.gamma_m
        self.p0 = 0.5
        self.mu = float(init_mu)
        self.s2 = float(init_sig) ** 2
        self.h = 0.5

    def bind(self, y: dict) -> None:
        self.h = float(y["tick"])

    def predict(self, y: dict, t: int) -> dict[str, Any]:
        sigma = max(math.sqrt(max(self.s2, 1e-12)), 0.2)
        return {"p0": float(self.p0), "mu": self.mu, "sigma": sigma, "h": self.h}

    def pmf(self, k: int, p0: float, mu: float, sigma: float, h: float) -> float:
        if k == 0:
            return p0
        lo_edge = 0.5 * h
        norm_const = 1.0 - norm.cdf((math.log(lo_edge) - mu) / sigma)
        if norm_const <= 1e-12:
            return 0.0
        val = abs(k) * h
        a = (math.log(max(val - 0.5 * h, 1e-12)) - mu) / sigma
        b = (math.log(val + 0.5 * h) - mu) / sigma
        # Halve by 0.5 because positive and negative signs share the (1 - p0) mass
        return 0.5 * (1.0 - p0) * max(norm.cdf(b) - norm.cdf(a), 0.0) / norm_const

    def logpmf(self, target: float, p: dict[str, Any]) -> float:
        k = int(round(target / p["h"]))
        prob = self.pmf(k, p["p0"], p["mu"], p["sigma"], p["h"])
        return math.log(max(prob, TINY))

    def update(self, y: dict, t: int) -> None:
        target = y["dmid_n1"][t]
        if not np.isfinite(target):
            return
        is_zero = 1.0 if target == 0.0 else 0.0
        self.p0 = self.gamma_p * self.p0 + self.alpha_p * is_zero
        self.p0 = min(max(self.p0, 1e-6), 1.0 - 1e-6)

        if target != 0.0:
            lx = math.log(max(abs(float(target)), 0.5 * self.h))
            self.mu = self.gamma_m * self.mu + self.alpha_m * lx
            diff = lx - self.mu
            self.s2 = self.gamma_m * self.s2 + self.alpha_m * (diff * diff)
            self.s2 = max(self.s2, 0.04)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        k_pos = np.arange(1, K + 1)
        lo_edge = 0.5 * p["h"]
        norm_const = 1.0 - norm.cdf((math.log(lo_edge) - p["mu"]) / p["sigma"])
        val = k_pos * p["h"]
        a = (np.log(np.maximum(val - 0.5 * p["h"], 1e-12)) - p["mu"]) / p["sigma"]
        b = (np.log(val + 0.5 * p["h"]) - p["mu"]) / p["sigma"]
        m_pos = np.maximum(norm.cdf(b) - norm.cdf(a), 0.0)
        sum_one_side = float(np.sum(m_pos)) / max(norm_const, 1e-12)
        total = p["p0"] + 2.0 * (0.5 * (1.0 - p["p0"]) * sum_one_side)
        assert abs(total - 1.0) < 1e-4, f"Hurdle LogNormal PMF sum was {total:.6f}"
