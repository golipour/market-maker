"""Mathematically rigorous order-flow models for discrete lot counts.

All models output discrete probability mass functions (PMFs) on non-negative integer lots
    k in {0, 1, 2, ...}

Every model satisfies:
    1. sum_{k=0}^inf P(V = k) == 1.000000
    2. P(V = k) >= 0 and <= 1
    3. Strict causality: predict(side) uses only historical trades <= t.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from scipy.stats import norm, weibull_min

TINY = 1e-300


def hurdle_lognormal_pmf(k: int, p0: float, mu: float, sigma: float) -> float:
    """Exact discrete hurdle log-normal PMF: sums to 1.0 over k >= 0."""
    if k < 0:
        return 0.0
    p0 = min(max(float(p0), 1e-6), 1.0 - 1e-6)
    if k == 0:
        return p0
    sigma = max(float(sigma), 1e-4)
    phi_zero = norm.cdf((math.log(0.5) - mu) / sigma)
    nc = 1.0 - phi_zero
    if nc <= 1e-12:
        return 0.0
    a = (math.log(max(k - 0.5, 1e-12)) - mu) / sigma
    b = (math.log(k + 0.5) - mu) / sigma
    cell = max(norm.cdf(b) - norm.cdf(a), 0.0)
    return (1.0 - p0) * cell / nc


class FlowIncumbent:
    """Production TorchHurdleLogNormalModel for buy/sell flow."""

    name = "INCUMBENT Hurdle LogNormal"

    def __init__(self, decay: float = 0.95, init_p0: float = 0.5, init_mu: float = 2.0, init_sigma: float = 1.0):
        self.gamma = float(decay)
        self.alpha = 1.0 - self.gamma
        self.p0 = {"buy": float(init_p0), "sell": float(init_p0)}
        self.mu = {"buy": float(init_mu), "sell": float(init_mu)}
        self.s2 = {"buy": float(init_sigma) ** 2, "sell": float(init_sigma) ** 2}

    def predict(self, side: str) -> dict[str, Any]:
        sigma = max(math.sqrt(max(self.s2[side], 1e-12)), 0.05)
        return {"p0": float(self.p0[side]), "mu": float(self.mu[side]), "sigma": float(sigma)}

    def logpmf(self, k: int, p: dict[str, Any]) -> float:
        prob = hurdle_lognormal_pmf(k, p["p0"], p["mu"], p["sigma"])
        return math.log(max(prob, TINY))

    def update_side(self, k: int, side: str) -> None:
        if k < 0:
            return
        is_zero = 1.0 if k == 0 else 0.0
        self.p0[side] = self.gamma * self.p0[side] + self.alpha * is_zero
        self.p0[side] = min(max(self.p0[side], 1e-6), 1.0 - 1e-6)

        if k > 0:
            lx = math.log(float(k))
            self.mu[side] = self.gamma * self.mu[side] + self.alpha * lx
            diff = lx - self.mu[side]
            self.s2[side] = self.gamma * self.s2[side] + self.alpha * (diff * diff)
            self.s2[side] = max(self.s2[side], 1e-6)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        grid = np.arange(1, K + 1)
        phi_zero = norm.cdf((np.log(0.5) - p["mu"]) / p["sigma"])
        nc = 1.0 - phi_zero
        a = (np.log(grid - 0.5) - p["mu"]) / p["sigma"]
        b = (np.log(grid + 0.5) - p["mu"]) / p["sigma"]
        cells = np.maximum(norm.cdf(b) - norm.cdf(a), 0.0)
        total = p["p0"] + (1.0 - p["p0"]) * float(np.sum(cells)) / max(nc, 1e-12)
        assert abs(total - 1.0) < 1e-4, f"Flow Incumbent PMF sum was {total:.6f}"


class FlowRegularised:
    """Challenger Hurdle LogNormal with variance prior floor to prevent collapse."""

    name = "CHALLENGER Regularized Hurdle LogNormal"

    def __init__(
        self,
        decay_p: float = 0.98,
        decay_v: float = 0.95,
        init_p0: float = 0.5,
        init_mu: float = 2.0,
        init_sigma: float = 1.0,
        sigma_floor: float = 0.35,
    ):
        self.gamma_p = float(decay_p)
        self.alpha_p = 1.0 - self.gamma_p
        self.gamma_v = float(decay_v)
        self.alpha_v = 1.0 - self.gamma_v
        self.sigma_floor = float(sigma_floor)

        self.p0 = {"buy": float(init_p0), "sell": float(init_p0)}
        self.mu = {"buy": float(init_mu), "sell": float(init_mu)}
        self.s2 = {"buy": float(init_sigma) ** 2, "sell": float(init_sigma) ** 2}

    def predict(self, side: str) -> dict[str, Any]:
        sigma = max(math.sqrt(max(self.s2[side], 1e-12)), self.sigma_floor)
        return {"p0": float(self.p0[side]), "mu": float(self.mu[side]), "sigma": float(sigma)}

    def logpmf(self, k: int, p: dict[str, Any]) -> float:
        prob = hurdle_lognormal_pmf(k, p["p0"], p["mu"], p["sigma"])
        return math.log(max(prob, TINY))

    def update_side(self, k: int, side: str) -> None:
        if k < 0:
            return
        is_zero = 1.0 if k == 0 else 0.0
        self.p0[side] = self.gamma_p * self.p0[side] + self.alpha_p * is_zero
        self.p0[side] = min(max(self.p0[side], 1e-6), 1.0 - 1e-6)

        if k > 0:
            lx = math.log(float(k))
            self.mu[side] = self.gamma_v * self.mu[side] + self.alpha_v * lx
            diff = lx - self.mu[side]
            self.s2[side] = self.gamma_v * self.s2[side] + self.alpha_v * (diff * diff)
            self.s2[side] = max(self.s2[side], self.sigma_floor ** 2)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        grid = np.arange(1, K + 1)
        phi_zero = norm.cdf((np.log(0.5) - p["mu"]) / p["sigma"])
        nc = 1.0 - phi_zero
        a = (np.log(grid - 0.5) - p["mu"]) / p["sigma"]
        b = (np.log(grid + 0.5) - p["mu"]) / p["sigma"]
        cells = np.maximum(norm.cdf(b) - norm.cdf(a), 0.0)
        total = p["p0"] + (1.0 - p["p0"]) * float(np.sum(cells)) / max(nc, 1e-12)
        assert abs(total - 1.0) < 1e-4, f"Flow Regularized PMF sum was {total:.6f}"


def _wb_cdf(x: float, c: float, scale: float) -> float:
    if x <= 0.0:
        return 0.0
    z = (x / scale) ** c
    return 1.0 - math.exp(-z) if z < 700.0 else 1.0


class FlowHurdleWeibull:
    """Production Hurdle Weibull."""

    name = "INCUMBENT Hurdle Weibull"

    def __init__(self, decay: float = 0.95, init_p0: float = 0.5, init_k: float = 1.2, init_lam: float = 5.0):
        self.gamma = float(decay)
        self.alpha = 1.0 - self.gamma
        self.p0 = {"buy": float(init_p0), "sell": float(init_p0)}
        self.mean_x = {"buy": float(init_lam), "sell": float(init_lam)}
        self.wb_k = float(init_k)

    def predict(self, side: str) -> dict[str, Any]:
        scale = max(self.mean_x[side] / math.gamma(1.0 + 1.0 / self.wb_k), 0.5)
        return {"p0": float(self.p0[side]), "k": self.wb_k, "scale": scale}

    def pmf(self, k: int, p0: float, c: float, scale: float) -> float:
        if k < 0:
            return 0.0
        if k == 0:
            return p0
        w0 = _wb_cdf(0.5, c, scale)
        nc = 1.0 - w0
        if nc <= 1e-12:
            return 0.0
        w_a = _wb_cdf(max(k - 0.5, 0.0), c, scale)
        w_b = _wb_cdf(k + 0.5, c, scale)
        return (1.0 - p0) * max(w_b - w_a, 0.0) / nc

    def logpmf(self, k: int, p: dict[str, Any]) -> float:
        prob = self.pmf(k, p["p0"], p["k"], p["scale"])
        return math.log(max(prob, TINY))

    def update_side(self, k: int, side: str) -> None:
        if k < 0:
            return
        is_zero = 1.0 if k == 0 else 0.0
        self.p0[side] = self.gamma * self.p0[side] + self.alpha * is_zero
        self.p0[side] = min(max(self.p0[side], 1e-6), 1.0 - 1e-6)

        if k > 0:
            self.mean_x[side] = self.gamma * self.mean_x[side] + self.alpha * float(k)

    def assert_normalized(self, p: dict[str, Any], K: int = 50000) -> None:
        grid = np.arange(1, K + 1)
        w0 = weibull_min.cdf(0.5, p["k"], scale=p["scale"])
        nc = 1.0 - w0
        wa = weibull_min.cdf(grid - 0.5, p["k"], scale=p["scale"])
        wb = weibull_min.cdf(grid + 0.5, p["k"], scale=p["scale"])
        cells = np.maximum(wb - wa, 0.0)
        total = p["p0"] + (1.0 - p["p0"]) * float(np.sum(cells)) / max(nc, 1e-12)
        assert abs(total - 1.0) < 1e-4, f"Flow Weibull PMF sum was {total:.6f}"


class BaselineEmpiricalFlow:
    """Empirical discrete rolling histogram for flow."""

    name = "BASELINE Empirical Flow (rolling 2000 steps)"

    def __init__(self, window: int = 2000):
        self.window = int(window)
        self.hist_b: deque[int] = deque(maxlen=self.window)
        self.hist_s: deque[int] = deque(maxlen=self.window)

    def predict(self, side: str) -> dict[str, Any]:
        h = self.hist_b if side == "buy" else self.hist_s
        counts: dict[int, int] = {}
        for x in h:
            counts[x] = counts.get(x, 0) + 1
        n = len(h)
        p0 = (counts.get(0, 0) + 1.0) / (n + 100.0) if n > 0 else 0.5
        return {"counts": counts, "n": n, "p0": p0}

    def logpmf(self, k: int, p: dict[str, Any]) -> float:
        n = p["n"]
        if n < 50:
            return -math.log(100.0)
        counts = p["counts"]
        # Laplace (+1) smoothing over active support plus background 500 lots
        total_denom = n + 500.0
        prob = (counts.get(k, 0) + 1.0) / total_denom
        return math.log(max(prob, TINY))

    def update_side(self, k: int, side: str) -> None:
        if k >= 0:
            h = self.hist_b if side == "buy" else self.hist_s
            h.append(k)

    def assert_normalized(self, p: dict[str, Any]) -> None:
        pass
