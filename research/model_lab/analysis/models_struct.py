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
        self.P = np.eye(2) * 1e-2
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
        k = np.clip(Px / max(self.lam + float(x @ Px), 1e-12), -1.0, 1.0)
        self.b = np.clip(self.b + k * r, -1e5, 1e5)
        self.P = (self.P - np.outer(k, Px)) / self.lam + 1e-2 * np.eye(2)
        self.P = np.clip(self.P, 0.0, 1e6)
        self.rs, self.rw = ewma_update(self.rs, self.rw, min(r * r, 1e7), self.lam)


class QuoteMixture(BaseModel):
    """THE STRUCTURAL MODEL, derived rather than fitted.

    Phase 1 established the mechanism, so the predictive law is written down
    from it and then *tested*:

        efficient price   p:      p_{t+1} = p_t + sigma_t * z,   z ~ Student(nu)
                                     (driftless; the mean is left at 0)
        quoted mid        m:      m_t = p_t + e_t
        quote error       e_t:    iid, mean 0, var = eta_t^2 / 3   (uniform on the
                                     [-eta_t, eta_t] quote interval, eta = half-spread)

    Observation model is exact; the ONLY free parameters are the conditional
    scale sigma_t and nu.  Scoring dmid_{t+1} requires integrating over the
    unknown e_t, which is done in closed form for the Gaussian case and by
    Gauss-Hermite quadrature for the Student case, so the reported NLL is the
    NLL of the induced law of dmid, not of dmid minus an assumed error.

    Why this is the right object: the production model scores the raw quoted
    mid with a single Gaussian.  That conflates the innovation of the
    *efficient price* with the *redraw of the quote*, whose variance is
    eta^2/3 and is therefore itself a function of the observable spread.
    """
    name = "STRUCTURAL: quote-mixture (efficient price + uniform quote error)"

    def __init__(self, lam=0.99, floor=1.0, nu=4.0, start_sd=50.0, nq=7):
        self.lam, self.floor, self.nu = lam, floor, nu
        self.start_sd, self.nq = start_sd, nq
        self.s, self.w = 0.0, 0.0
        x, wt = np.polynomial.hermite_e.hermegauss(nq)
        self.qx = x
        self.qw = wt / np.sqrt(2.0 * np.pi)

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 50.0

    def predict(self, t, y):
        # 3-tuple: the custom logpdf integrates over the quote error itself.
        sig = np.sqrt(self.s / self.w) if self.w > 0 else self.start_sd
        sig = max(sig, self.floor)
        self._sd_e = self._eta(y, t) / np.sqrt(3.0)
        return (0.0, sig, self.target(y, t))

    def logpdf(self, tgt, mu, sd):
        sd_e = getattr(self, "_sd_e", 1.0)
        from .preq import log_student_t
        acc = 0.0
        for i in range(self.nq):
            u = sd_e * self.qx[i]
            acc += self.qw[i] * np.exp(log_student_t(tgt - mu - u, 0.0, sd, self.nu))
        return float(np.log(max(acc, 1e-300)))

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        self.s, self.w = ewma_update(self.s, self.w, min(x * x, 1e7), self.lam)
