"""Online models that carry SERIAL DEPENDENCE explicitly.

The incumbent (EWMA scale, zero mean, iid predictive draws) is a special case
of every model here.  Each adds exactly one mechanism, so the prequential
ladder attributes every nat to a named mechanism.

Mechanisms under test, in the order they are added:

  D0  iid zero-mean Student-t with EWMA scale        (the incumbent)
  D1  + hurdle for the point mass at zero           (bounce-aware)
  D2  + ARCH-in-mean: mu_t = phi * u_{t-1}          (the measured ACF1(z) < 0)
  D3  + GARCH-style scale: sigma_t^2 from past |u|  (the measured ACF(|z|) > 0)
  D4  D2 + D3 jointly
  D5  D4 with the AR coefficient estimated online rather than fixed

All recursions are causal.  `predict` sees only the state built from
realised values up to t; `update` folds in the value realised at t+1.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

LOG2PI = float(np.log(2.0 * np.pi))


def _lt(y, mu, sd, nu):
    sd = max(sd, 1e-9)
    z = (y - mu) / sd
    lc = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
          - 0.5 * (np.log(nu * np.pi) + np.log(sd)))
    return lc - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)


class DepBase:
    """Shared plumbing: target, hurdle-aware log density, diagnostics."""

    name = "dep"
    h = 1

    def target(self, y, t):
        j = t + self.h
        return y["dmid"][j] if j < y["N"] else np.nan

    # subclasses set these
    def predict(self, t, y):
        raise NotImplementedError

    def update(self, t, y):
        raise NotImplementedError

    # a uniform 3-tuple return is accepted by the Runner as (mu, sd, target)


class IIDStudent(DepBase):
    """D0. Zero-mean Student-t, EWMA scale. The incumbent reference."""

    name = "D0 iid Student-t, EWMA scale"

    def __init__(self, lam=0.99, nu=4.0, floor=1.0, s0=50.0):
        self.lam, self.nu, self.floor, self.s0 = lam, nu, floor, s0
        self.s, self.w = 0.0, 0.0

    def _scale(self):
        return max(np.sqrt(self.s / self.w), self.floor) if self.w > 0 else self.s0

    def predict(self, t, y):
        return 0.0, self._scale(), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s = self.s * self.lam + x * x
            self.w = self.w * self.lam + 1.0

    def logpdf(self, tgt, mu, sd):
        return _lt(tgt, mu, sd, self.nu)


class HurdleIID(IIDStudent):
    """D1. D0 + explicit point mass at zero, estimated online.

    The zero mass is NOT a free constant: it is a decayed running mean of the
    indicator 1{dmid==0}, i.e. a recursive conditional probability.
    """

    name = "D1 + hurdle at zero (online p0)"

    def __init__(self, lam=0.99, lam_p=0.99, nu=4.0, floor=1.0, s0=50.0,
                 p0_0=0.35):
        super().__init__(lam, nu, floor, s0)
        self.lam_p, self.zs, self.zw = lam_p, 0.0, 0.0
        self.p0_0 = p0_0

    def p0(self):
        return self.zs / self.zw if self.zw > 0 else self.p0_0

    def predict(self, t, y):
        return 0.0, self._scale(), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s = self.s * self.lam + x * x
            self.w = self.w * self.lam + 1.0
            self.zs = self.zs * self.lam_p + float(x == 0)
            self.zw = self.zw * self.lam_p + 1.0

    def logpdf(self, tgt, mu, sd):
        p0 = min(max(self.p0(), 1e-6), 1 - 1e-6)
        if tgt == 0:
            return np.log(p0)
        # conditional on non-zero, the move is symmetric: model |move|
        return _lt(abs(tgt), 0.0, sd, self.nu) - np.log1p(-p0)


class ArchMean(HurdleIID):
    """D2. D1 + ARCH in the mean:  mu_{t+1} = phi * u_t.

    This is the ONLY mean structure the data supports, and Q1 predicted its
    sign: ACF1 of the standardised innovation is NEGATIVE (-0.189), so phi
    should be negative.  `u_t` is the last standardised innovation, so the
    recursion is a one-step feedback with no free intercept.
    """

    name = "D2 + ARCH mean (fixed phi)"

    def __init__(self, phi=-0.18, **kw):
        super().__init__(**kw)
        self.phi = phi
        self.u_prev = 0.0

    def predict(self, t, y):
        return self.phi * self.u_prev, self._scale(), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            sd = self._scale()
            self.u_prev = x / sd
            self.s = self.s * self.lam + x * x
            self.w = self.w * self.lam + 1.0
            self.zs = self.zs * self.lam_p + float(x == 0)
            self.zw = self.zw * self.lam_p + 1.0

    def logpdf(self, tgt, mu, sd):
        p0 = min(max(self.p0(), 1e-6), 1 - 1e-6)
        if tgt == 0:
            return np.log(p0)
        return _lt(tgt, mu, sd, self.nu) - np.log1p(-p0)


class GarchScale(HurdleIID):
    """D3. D1 + GARCH-style scale driven by past |innovation|.

        s2_t    = omega + a * (|u_{t-1}| - c)^2   (EWMA of squared shock)
        sigma_t = sqrt(s2_t)

    This replaces the plain EWMA of raw squares with an EWMA of the SHOCK
    only, which is the correct recursion when the shocks are standardised.
    A negative  c  would be a mean-reverting volatility; we scan it.
    """

    name = "D3 + GARCH scale from |shock|"

    def __init__(self, lam=0.97, c=0.0, **kw):
        kw.setdefault("lam", 0.99)
        super().__init__(**kw)
        self.lam_g, self.c = lam, c
        self.gs, self.gw = 0.0, 0.0

    def _gscale(self):
        if self.gw <= 1.0:
            return self.s0
        return max(np.sqrt(self.gs / self.gw), self.floor)

    def predict(self, t, y):
        return 0.0, self._gscale(), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            sd = self._gscale()
            shock = abs(x / sd) - self.c
            self.gs = self.gs * self.lam_g + shock * shock
            self.gw = self.gw * self.lam_g + 1.0
            self.s = self.s * self.lam + x * x
            self.w = self.lam * self.w + 1.0
            self.zs = self.zs * self.lam_p + float(x == 0)
            self.zw = self.zw * self.lam_p + 1.0

    def logpdf(self, tgt, mu, sd):
        p0 = min(max(self.p0(), 1e-6), 1 - 1e-6)
        if tgt == 0:
            return np.log(p0)
        return _lt(abs(tgt), 0.0, sd, self.nu) - np.log1p(-p0)


class ArchGarch(DepBase):
    """D4/D5. ARCH mean + GARCH scale + hurdle, with phi estimated ONLINE.

    The mean recursion is  mu_{t+1} = phi * u_t.  Because Q1 measured
    ACF1(z) = -0.189, phi is *initialised* at -0.18 and then updated by a
    heavily-damped recursive least-squares step.  Estimating it online is the
    honest test of whether the effect is stable or an estimation artifact --
    if it is an artifact, phi decays to 0 and the model collapses onto D3,
    losing nothing.  If it is real, it converges and keeps the gain.

    Scale recursion uses the SHOCK variance so that it is a genuine GARCH, and
    the hurdle mass is a separate decayed mean.
    """

    name = "D4 ARCH mean + GARCH scale + hurdle (phi online)"

    def __init__(self, lam_g=0.97, lam_p=0.99, lam_phi=0.995, c=0.0,
                 nu=4.0, floor=1.0, s0=50.0, phi0=-0.18, phi_cap=0.6,
                 p0_0=0.35, use_mean=True):
        self.lam_g, self.lam_p, self.lam_phi = lam_g, lam_p, lam_phi
        self.c, self.nu, self.floor, self.s0 = c, nu, floor, s0
        self.phi, self.phi_cap = phi0, phi_cap
        self.p0_0, self.use_mean = p0_0, use_mean
        self.gs = self.gw = 0.0
        self.zs = self.zw = 0.0
        self.u_prev = 0.0
        self.pphi = 1e-3

    def p0(self):
        return self.zs / self.zw if self.zw > 0 else self.p0_0

    def _scale(self):
        if self.gw <= 1.0:
            return self.s0
        return max(np.sqrt(self.gs / self.gw), self.floor)

    def predict(self, t, y):
        mu = self.phi * self.u_prev if self.use_mean else 0.0
        return mu, self._scale(), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        sd = self._scale()
        # shock, standardised by the scale known at t (causal)
        shock = abs(x / sd) - self.c
        u = x / sd
        if self.use_mean:
            # RLS step for phi in the regression  u_t = phi * u_{t-1} + e
            up = self.u_prev
            r = u - self.phi * up
            self.pphi = (self.pphi + up * up) / self.lam_phi
            k = up / max(self.pphi, 1e-12)
            self.phi = float(np.clip(self.phi + k * r, -self.phi_cap,
                                    self.phi_cap))
        self.u_prev = u
        self.gs = self.gs * self.lam_g + shock * shock
        self.gw = self.gw * self.lam_g + 1.0
        self.zs = self.zs * self.lam_p + float(x == 0)
        self.zw = self.zw * self.lam_p + 1.0

    def logpdf(self, tgt, mu, sd):
        p0 = min(max(self.p0(), 1e-6), 1 - 1e-6)
        if tgt == 0:
            return np.log(p0)
        if self.use_mean:
            return _lt(tgt, mu, sd, self.nu) - np.log1p(-p0)
        return _lt(abs(tgt), 0.0, sd, self.nu) - np.log1p(-p0)
