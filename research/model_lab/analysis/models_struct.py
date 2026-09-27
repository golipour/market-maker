"""Structural models: one class per mechanism found in the data audit.

  FTLQuoted        - the Roll/Grossman-Miller resampling term removed
                     analytically; the residual is the efficient-price
                     innovation, and the quoted mid adds the redraw back.
  MicropriceAnchor - the next mid move reverts toward the microprice offset,
                     expressed in half-spread units (queue-pressure reversal).
"""
from __future__ import annotations

import numpy as np

from .models_mid import ewma_update
from .preq import BaseModel


class FTLQuoted(BaseModel):
    """Score the *efficient price* innovation, not the raw quoted mid.

    Quotes are a refreshed uniform draw around the efficient price, so
    Cov(dmid_t, dmid_{t-1}) = -eta^2 with eta the half-spread.  Hence
    dmid^2 - eta^2 estimates the efficient-price innovation variance.
    """

    name = "structural: FTL-corrected innovation variance"

    def __init__(self, lam=0.99, floor=1.0, start_sd=50.0, kappa=0.5):
        self.lam, self.floor, self.start_sd, self.kappa = lam, floor, start_sd, kappa
        self.sig2, self.sw = 0.0, 0.0

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 50.0

    def predict(self, t, y):
        eta = self._eta(y, t)
        var_p = (self.sig2 / self.sw) if self.sw > 0 else self.start_sd ** 2
        sd = float(np.sqrt(max(var_p + (self.kappa * eta) ** 2, 1e-6)))
        return 0.0, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        tgt = self.target(y, t)
        if not np.isfinite(tgt):
            return
        eta = self._eta(y, t)
        self.sig2, self.sw = ewma_update(
            self.sig2, self.sw, max(tgt * tgt - eta * eta, 1.0), self.lam)


class MicropriceAnchor(BaseModel):
    """Next mid move reverts toward the microprice offset / half-spread."""

    name = "structural: microprice-anchored mean + EWMA scale"

    def __init__(self, lam=0.995, floor=1.0, start_sd=50.0):
        self.lam, self.floor, self.start_sd = lam, floor, start_sd
        self.P = np.eye(2) * 1e3
        self.b = np.zeros(2)
        self.rs, self.rw = 0.0, 0.0

    def _x(self, y, t):
        e, sp = y["micro_minus_mid"][t], y["spread"][t]
        if not (np.isfinite(e) and np.isfinite(sp) and sp > 0):
            return None
        return np.array([1.0, e / (0.5 * sp)])

    def predict(self, t, y):
        x = self._x(y, t)
        if x is None:
            return 0.0, self.start_sd, self.target(y, t)
        sd = self.rs / self.rw if self.rw > 0 else self.start_sd
        return float(x @ self.b), max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x, tgt = self._x(y, t), self.target(y, t)
        if x is None or not np.isfinite(tgt):
            return
        r = tgt - float(x @ self.b)
        Px = self.P @ x
        k = Px / max(self.lam + float(x @ Px), 1e-12)
        self.b = self.b + k * r
        self.P = (self.P - np.outer(k, Px)) / self.lam + 1e-3 * np.eye(2)
        self.rs, self.rw = ewma_update(self.rs, self.rw, r * r, self.lam)
