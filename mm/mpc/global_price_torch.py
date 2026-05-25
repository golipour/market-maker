"""Global-venue VWAP dynamics for MPC rollouts.

Gaussian EWMA-style global VWAP model:
EWMA on (buy, sell) VWAP *deltas*, then Gaussian ``sample_deltas``
for the optimiser. Observations and deltas use the same numeric units as the
runtime (quote-per-base floats).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class GlobalGaussianPriceModel:
    """
    Gaussian model for global VWAP deltas (buy/sell), updated via
    EWMA sufficient statistics (exponentially weighted moments).
    """

    def __init__(
        self,
        decay: float = 0.97,
        sigma_min_quote: float = 1.0,
    ):
        self.decay = min(0.999999, max(0.0, float(decay)))
        self.sigma_min_quote = max(1e-9, float(sigma_min_quote))
        self._m1_buy: float | None = None
        self._m2_buy: float | None = None
        self._m1_sell: float | None = None
        self._m2_sell: float | None = None
        self._last_buy_obs: float | None = None
        self._last_sell_obs: float | None = None

    def _update_one(self, x: float, m1: float | None, m2: float | None) -> tuple[float, float]:
        if m1 is None or m2 is None:
            return float(x), float(x) * float(x)
        a = self.decay
        b = 1.0 - a
        m1_new = a * float(m1) + b * float(x)
        m2_new = a * float(m2) + b * float(x) * float(x)
        return float(m1_new), float(m2_new)

    @staticmethod
    def _var_from_moments(m1: float | None, m2: float | None) -> float:
        if m1 is None or m2 is None:
            return 0.0
        return max(0.0, float(m2) - float(m1) * float(m1))

    def update(self, buy_obs: float, sell_obs: float) -> None:
        b = float(buy_obs)
        s = float(sell_obs)
        if self._last_buy_obs is not None and self._last_sell_obs is not None:
            d_buy = b - float(self._last_buy_obs)
            d_sell = s - float(self._last_sell_obs)
            self._m1_buy, self._m2_buy = self._update_one(d_buy, self._m1_buy, self._m2_buy)
            self._m1_sell, self._m2_sell = self._update_one(d_sell, self._m1_sell, self._m2_sell)
        self._last_buy_obs = b
        self._last_sell_obs = s

    def forecast_next_prices(self) -> tuple[float | None, float | None]:
        if self._last_buy_obs is None or self._last_sell_obs is None:
            return None, None
        mu_db = float(self._m1_buy) if self._m1_buy is not None else 0.0
        mu_ds = float(self._m1_sell) if self._m1_sell is not None else 0.0
        return float(self._last_buy_obs + mu_db), float(self._last_sell_obs + mu_ds)

    def get_params(self) -> dict[str, float]:
        var_buy = self._var_from_moments(self._m1_buy, self._m2_buy)
        var_sell = self._var_from_moments(self._m1_sell, self._m2_sell)
        sigma_buy = max(self.sigma_min_quote, math.sqrt(var_buy))
        sigma_sell = max(self.sigma_min_quote, math.sqrt(var_sell))
        return {
            "mu_delta_buy": float(self._m1_buy) if self._m1_buy is not None else 0.0,
            "sigma_delta_buy": float(sigma_buy),
            "mu_delta_sell": float(self._m1_sell) if self._m1_sell is not None else 0.0,
            "sigma_delta_sell": float(sigma_sell),
            "decay": float(self.decay),
            "sigma_min_quote": float(self.sigma_min_quote),
        }

    def sample_deltas(
        self,
        shape: tuple[int, int],
        *,
        device: Any = None,
        dtype: Any = None,
        torch_mod: Any = None,
    ) -> tuple[Any, Any]:
        """Sample Gaussian delta shocks for (buy, sell) global VWAPs."""
        n_samples, horizon = int(shape[0]), int(shape[1])
        p = self.get_params()
        mu_db = float(p.get("mu_delta_buy", 0.0) or 0.0)
        mu_ds = float(p.get("mu_delta_sell", 0.0) or 0.0)
        sg_db = max(1e-12, float(p.get("sigma_delta_buy", 1e-6) or 1e-6))
        sg_ds = max(1e-12, float(p.get("sigma_delta_sell", 1e-6) or 1e-6))
        if torch_mod is not None:
            d_buy = torch_mod.randn(n_samples, horizon, device=device, dtype=dtype) * sg_db + mu_db
            d_sell = torch_mod.randn(n_samples, horizon, device=device, dtype=dtype) * sg_ds + mu_ds
            return d_buy, d_sell
        z_buy = np.random.randn(n_samples, horizon)
        z_sell = np.random.randn(n_samples, horizon)
        return z_buy * sg_db + mu_db, z_sell * sg_ds + mu_ds


class GaussianGlobalDeltaModel:
    """Static Gaussian increments (no online learning). Optional ``kind: \"gaussian\"``."""

    def __init__(self, sigma_buy: float = 0.0, sigma_sell: float = 0.0):
        self.sigma_buy = max(0.0, float(sigma_buy))
        self.sigma_sell = max(0.0, float(sigma_sell))

    def get_params(self) -> dict[str, float]:
        return {"sigma_buy": self.sigma_buy, "sigma_sell": self.sigma_sell}

    def sample_deltas(
        self,
        sample_shape: tuple[int, ...],
        *,
        device: Any = None,
        dtype: Any = None,
        torch_mod: Any = None,
    ) -> tuple[Any, Any]:
        import torch

        mod = torch_mod or torch
        if len(sample_shape) != 2:
            raise ValueError("sample_shape must be (n_samples, horizon)")
        n, h = int(sample_shape[0]), int(sample_shape[1])
        zeros = mod.zeros(n, h, dtype=dtype, device=device)
        d_buy = zeros if self.sigma_buy <= 0 else mod.randn(n, h, dtype=dtype, device=device) * self.sigma_buy
        d_sell = zeros if self.sigma_sell <= 0 else mod.randn(n, h, dtype=dtype, device=device) * self.sigma_sell
        return d_buy, d_sell


def score_vwap_deltas(
    delta_buy: float,
    delta_sell: float,
    params: dict,
) -> "ScoreResult":
    """Pre-update global VWAP delta score (see ``mm.models.scoring``)."""

    from mm.models.scoring import score_global_vwap_deltas

    return score_global_vwap_deltas(
        delta_buy=delta_buy,
        delta_sell=delta_sell,
        params_before=params,
    )
