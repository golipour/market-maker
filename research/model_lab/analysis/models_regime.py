"""Regime-conditional scale models derived from the Phase-6 diagnosis.

Phase 6 established, non-parametrically, that

  * sd(dmid | half-spread) is a near-STEP function of the half-spread:
      eta ~ 0.5 -> sd ~ 2.0
      eta ~ 12  -> sd ~ 2.7
      eta ~ 42  -> sd ~ 2.7
      eta ~ 83  -> sd ~ 2.7
      eta ~ 126 -> sd ~ 3.4
      eta ~ 188 -> sd ~ 4.1
    log-log slope 0.08, i.e. almost NO elasticity.  It is NOT sd ~ eta.

  * within a fixed eta band, the scale rises with book staleness
    (1.35 -> 2.70 -> 6.07) and with trade occurrence (1.35 -> 4.72).

  * the zero mass falls monotonically with the half-spread
    (0.43 -> 0.20): a wide quote means the price is genuinely moving.

So the predictive law is a hurdle over a (spread-regime x staleness) state,
with a slow-drifting scale inside each cell.  That is a *state* model whose
state is directly observable, not a latent mixture to be discovered.
"""
from __future__ import annotations

import numpy as np

from .models_mid import ewma_update
from .preq import BaseModel


class RegScaleHurdle(BaseModel):
    """Hurdle + Student-t whose parameters are per-(spread-regime, staleness) cell.

    Parameters updated by EWMA inside each cell, so a regime is tracked as it
    drifts and a regime that has not been seen yet falls back to a pooled
    prior.  The zero mass p0 and the two conditional scales are separate
    recursive statistics; nothing is fitted in batch.
    """

    name = "REGIME: hurdle x (spread x staleness) cell, Student-t core"

    def __init__(self, lam=0.995, nu=4.0, n_eta=6, n_stale=3,
                 p0_pool=0.33, floor=0.05, start_sd=2.7):
        self.lam, self.nu = lam, nu
        self.n_eta, self.n_stale = n_eta, n_stale
        self.p0_pool, self.floor, self.start_sd = p0_pool, floor, start_sd
        self.eta_edges = np.quantile(np.arange(1.0, 200.0), np.linspace(0, 1, n_eta + 1))[1:-1]
        self.stale_edges = np.array([1500.0, 2800.0])
        nc = n_eta * n_stale
        self.p0 = np.full(nc, p0_pool)
        self.s2 = np.zeros(nc)
        self.w2 = np.zeros(nc)
        self.n = np.zeros(nc)

    def _cell(self, y, t):
        e = 0.5 * y["spread"][t]
        st = y["stale_ms"][t]
        if not np.isfinite(e) or e <= 0:
            return 0
        ie = int(np.searchsorted(self.eta_edges, e))
        ist = 0
        if np.isfinite(st):
            ist = 1 if st > self.stale_edges[0] else 0
            ist = 2 if st > self.stale_edges[1] else ist
        return ie * self.n_stale + ist

    def predict(self, t, y):
        c = self._cell(y, t)
        sd = np.sqrt(self.s2[c] / self.s2[c] if False else
                     (self.s2[c] / self.w2[c] if self.w2[c] > 0 else self.start_sd ** 2))
        sd = max(float(sd), self.floor)
        p0 = float(self.p0[c])
        x = self.target(y, t)
        if not np.isfinite(x):
            return 0.0, sd, None, x
        if x == 0:
            return 0.0, sd, p0, x
        # signed conditional scale: use the pooled one, shape is symmetric
        return 0.0, sd, min(1.0 - p0, 0.999), x

    def update(self, t, y):
        c = self._cell(y, t)
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        self.n[c] += 1.0
        self.p0[c] = self.p0[c] * self.lam + (1.0 if x == 0 else 0.0) * (1.0 - self.lam)
        if x != 0:
            self.s2[c], self.w2[c] = ewma_update(self.s2[c], self.w2[c],
                                                min(x * x, 1e7), self.lam)


class RegScaleHurdleLog(BaseModel):
    """Same, but the scale is tracked in LOG space with pooled shrinkage.

    Cell scales vary by an order of magnitude; a pooled multiplicative prior
    with an EWMA log-update is the variance-efficient way to track them.
    """

    name = "REGIME: hurdle x (spread x staleness), log-scale + shrinkage"

    def __init__(self, lam=0.995, nu=4.0, n_eta=6, n_stale=3,
                 p0_pool=0.33, floor=0.05, start_sd=2.7, shrink=0.02):
        self.lam, self.nu, self.shrink = lam, nu, shrink
        self.n_eta, self.n_stale = n_eta, n_stale
        self.p0_pool, self.floor, self.start_sd = p0_pool, floor, start_sd
        self.eta_edges = np.quantile(np.arange(1.0, 200.0), np.linspace(0, 1, n_eta + 1))[1:-1]
        self.stale_edges = np.array([1500.0, 2800.0])
        nc = n_eta * n_stale
        self.p0 = np.full(nc, p0_pool)
        self.ls = np.log(self.start_sd ** 2) * np.ones(nc)
        self.lw = np.zeros(nc)
        self.n = np.zeros(nc)

    def _cell(self, y, t):
        e = 0.5 * y["spread"][t]
        st = y["stale_ms"][t]
        if not np.isfinite(e) or e <= 0:
            return 0
        ie = int(np.searchsorted(self.eta_edges, e))
        ist = 0
        if np.isfinite(st):
            ist = 1 if st > self.stale_edges[0] else 0
            ist = 2 if st > self.stale_edges[1] else ist
        return ie * self.n_stale + ist

    def _sd(self, c):
        pooled = self.ls.sum() / max(self.w2_total(), 1e-9) if self.w2_total() else np.log(self.start_sd ** 2)
        if self.lw[c] <= 0:
            return self.start_sd
        # multiplicative shrinkage toward the pooled level
        v = (1.0 - self.shrink) * self.ls[c] + self.shrink * pooled
        return float(np.sqrt(max(np.exp(v), self.floor ** 2)))

    def w2_total(self):
        return float(self.lw.sum())

    def predict(self, t, y):
        c = self._cell(y, t)
        sd = self._sd(c)
        p0 = float(self.p0[c])
        x = self.target(y, t)
        if not np.isfinite(x):
            return 0.0, sd, None, x
        if x == 0:
            return 0.0, sd, p0, x
        return 0.0, sd, min(1.0 - p0, 0.999), x

    def update(self, t, y):
        c = self._cell(y, t)
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        self.n[c] += 1.0
        self.p0[c] = self.p0[c] * self.lam + (1.0 if x == 0 else 0.0) * (1.0 - self.lam)
        self.ls[c] = self.ls[c] * self.lam
        self.lw[c] = self.lw[c] * self.lam
        if x != 0:
            self.ls[c] += np.log(max(x * x, 1e-6))
            self.lw[c] += 1.0
