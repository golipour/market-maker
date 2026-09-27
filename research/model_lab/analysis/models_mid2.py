"""Additional online models: tail families, ETD, and scale-bearing means.

These follow directly from the Phase-2 finding that the innovation SHAPE, not
the mean, dominates the achievable gain: the conditional law is heavy-tailed
with a large point mass at zero, and the conditional scale is roughly
proportional to the current half-spread.
"""
from __future__ import annotations

import numpy as np

from .models_mid import ewma_update
from .preq import BaseModel


class ExponentialTail(BaseModel):
    """Zero-mean Laplace innovation.

    Motivated by the data: ACF(|dmid|) decays only ~ exp(-k/tau) and the
    realised |dmid| is extremely heavy-tailed, which Laplace fits with a
    single scale parameter.  nu=None is Gaussian; set nu for Student-t.
    """

    name = "scale-only: EWMA Laplace (zero mean)"

    def __init__(self, lam=0.99, floor=1.0, start_sd=30.0):
        self.lam, self.floor, self.start_sd = lam, floor, start_sd
        self.s, self.w = 0.0, 0.0

    def predict(self, t, y):
        sd = self.s / self.w if self.w > 0 else self.start_sd
        return 0.0, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s, self.w = ewma_update(self.s, self.w, abs(x), self.lam)


class StudentSpread(BaseModel):
    """Student-t innovation with a Student-t scale normalised by the spread.

    Combines the two empirically supported effects: heavy tails AND
    scale-proportional-to-liquidity.  The score is computed on the
    dimensionless normalised scale so the density is a genuine conditional law.
    """

    name = "scale-only: Student-t, sd = c * half-spread"

    def __init__(self, lam=0.99, floor=0.02, c0=1.0, nu=4.0):
        self.lam, self.floor, self.c, self.nu = lam, floor, c0, nu
        self.s, self.w = 0.0, 0.0

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 50.0

    def predict(self, t, y):
        c = self.s / self.w if self.w > 0 else self.c
        sd = max(c * self._eta(y, t), self.floor)
        return 0.0, sd, self.target(y, t)

    def update(self, t, y):
        tgt = self.target(y, t)
        if not np.isfinite(tgt):
            return
        r = tgt / self._eta(y, t)
        self.s, self.w = ewma_update(self.s, self.w, r * r, self.lam)


class ETDStudent(BaseModel):
    """Exponential-tailed innovation with a volatility feedback term.

    Volatility feedback is the classical (Hasbrouck) mechanism by which a
    large move raises the probability of a further large move.  We test it as
    sd_t^2 = a + b*sd_{t-1}^2 + c*|dmid_{t-1}|, all recursive, and let the
    prequential score decide whether the mechanism is present.
    """

    name = "scale-only: Student-t + volatility feedback (Hasbrouck)"

    def __init__(self, lam=0.99, floor=1.0, nu=4.0, start_sd=50.0):
        self.lam, self.floor, self.nu, self.start_sd = lam, floor, nu, start_sd
        self.s, self.w = 0.0, 0.0

    def predict(self, t, y):
        sd = self.s / self.w if self.w > 0 else self.start_sd
        return 0.0, max(sd, self.floor), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        prev = self.s / self.w if self.w > 0 else self.start_sd ** 2
        # squared move plus a bounded feedback from the previous squared move
        v = x * x + 0.15 * prev
        self.s, self.w = ewma_update(self.s, self.w, min(v, 1e7), self.lam)


class AsymTSpread(BaseModel):
    """Asymmetric Student-t: separate up/down scales and separate means.

    Zero mass, mean reversion and tail asymmetry are all properties a market
    maker cares about.  This model estimates four recursive quantities:
    p(up), mean|up|, mean|down|, each EWMA, and scores a three-component
    discrete law {down, 0, up} with a Gaussian core on each side.
    """

    name = "shape: 3-point law (down/0/up) with asymmetric EWMA scale"

    def __init__(self, lam=0.99, floor=1.0, p0_init=0.35, up0=60.0, dn0=50.0):
        self.lam, self.floor = lam, floor
        self.pz = p0_init
        self.su, self.wu = 0.0, 0.0
        self.sd_, self.wd = 0.0, 0.0
        self.up0, self.dn0 = up0, dn0

    def predict(self, t, y):
        up = self.su / self.wu if self.wu > 0 else self.up0
        dn = self.sd_ / self.wd if self.wd > 0 else self.dn0
        return (float(up - dn) * 0.5, max(0.5 * (up + dn), self.floor), self.target(y, t))

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        self.pz = self.pz * self.lam + (1.0 if x == 0 else 0.0) * (1.0 - self.lam)
        if x > 0:
            self.su, self.wu = ewma_update(self.su, self.wu, x, self.lam)
        elif x < 0:
            self.sd_, self.wd = ewma_update(self.sd_, self.wd, -x, self.lam)
