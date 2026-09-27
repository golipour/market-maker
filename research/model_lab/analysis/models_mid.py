"""Online models for the transition law of the observed mid.

All models obey the prequential contract in preq.BaseModel: predict() sees only
information at or before t, update() consumes the realised value afterwards.

Design axes, in increasing complexity:
  scale-only  : learn the conditional SHAPE, zero mean
  mean-only   : learn the conditional MEAN, fixed scale
  both        : learn mean and scale jointly
  hurdle      : explicit point mass at zero
  structural  : spread-normalised, microprice-anchored, FTL-aware

Nothing is fitted in batch; every parameter is a recursive statistic.
"""
from __future__ import annotations

import numpy as np

from .preq import BaseModel


def ewma_update(s, w, x, lam, floor=1e-9):
    """One discounted step. Returns (new_sum, new_weight)."""
    s = s * lam
    w = w * lam
    if np.isfinite(x):
        s = s + x
        w = w + 1.0
    return s, max(w, floor)


class EWMAStatic(BaseModel):
    """Zero-mean Gaussian with an EWMA scale. Honest no-memory baseline."""

    name = "scale-only: EWMA Gaussian (zero mean)"

    def __init__(self, lam=0.99, floor=1.0, start_sd=50.0):
        self.lam, self.floor, self.start_sd = lam, floor, start_sd
        self.s, self.w = 0.0, 0.0

    def predict(self, t, y):
        sd = self.s / self.w if self.w > 0 else self.start_sd
        return 0.0, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s, self.w = ewma_update(self.s, self.w, x * x, self.lam)


class StudentEWMA(BaseModel):
    """Zero-mean Student-t with EWMA scale: fixes the Gaussian tail failure."""

    name = "scale-only: EWMA Student-t (zero mean)"
    sentinel = -1.0

    def __init__(self, lam=0.99, floor=1.0, nu=4.0, start_sd=50.0):
        self.lam, self.floor, self.nu, self.start_sd = lam, floor, nu, start_sd
        self.s, self.w = 0.0, 0.0

    def predict(self, t, y):
        sd = self.s / self.w if self.w > 0 else self.start_sd
        return 0.0, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s, self.w = ewma_update(self.s, self.w, x * x, self.lam)


class MA1EWMA(BaseModel):
    """MA(1) mean by recursive least squares + EWMA residual scale."""

    name = "mean+scale: MA(1) RLS + EWMA scale"

    def __init__(self, lam=0.995, floor=1.0, ridge=1e-2, start_sd=50.0):
        self.lam, self.floor, self.ridge = lam, floor, ridge
        self.P = np.eye(2) * 1e-2
        self.b = np.zeros(2)
        self.rs, self.rw = 0.0, 0.0
        self.start_sd = start_sd

    def _x(self, y, t):
        p = y["dmid"][t - 1] if t >= 1 else np.nan
        return np.array([1.0, p]) if np.isfinite(p) else None

    def predict(self, t, y):
        x = self._x(y, t)
        if x is None:
            return 0.0, self.start_sd, self.target(y, t)
        mu = float(x @ self.b)
        sd = self.rs / self.rw if self.rw > 0 else self.start_sd
        return mu, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x = self._x(y, t)
        tgt = self.target(y, t)
        if x is None or not np.isfinite(tgt):
            return
        r = tgt - float(x @ self.b)
        Px = self.P @ x
        k = np.clip(Px / max(self.lam + float(x @ Px), 1e-12), -1.0, 1.0)
        self.b = np.clip(self.b + k * r, -1e5, 1e5)
        self.P = (self.P - np.outer(k, Px)) / self.lam + self.ridge * np.eye(2)
        self.P = np.clip(self.P, 0.0, 1e6)
        self.rs, self.rw = ewma_update(self.rs, self.rw, min(r * r, 1e7), self.lam)


class SpreadNormalised(BaseModel):
    """Conditional scale proportional to the current half-spread.

    dmid_{t+1} / (spread_t/2) is dimensionless; carrying the spread back
    removes the dominant level heterogeneity.  sd_t = c_t * eta_t with c_t an
    EWMA of the squared normalised move.
    """

    name = "scale-only: EWMA sd = c * half-spread"

    def __init__(self, lam=0.99, floor=0.02, c0=1.0):
        self.lam, self.floor, self.c = lam, floor, c0
        self.s, self.w = 0.0, 0.0

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else np.nan

    def predict(self, t, y):
        eta = self._eta(y, t)
        c = self.s / self.w if self.w > 0 else self.c
        return 0.0, max(c * (eta if np.isfinite(eta) else 50.0), self.floor), self.target(y, t)

    def update(self, t, y):
        eta = self._eta(y, t)
        tgt = self.target(y, t)
        if np.isfinite(eta) and eta > 0 and np.isfinite(tgt):
            r = tgt / eta
            self.s, self.w = ewma_update(self.s, self.w, r * r, self.lam)
