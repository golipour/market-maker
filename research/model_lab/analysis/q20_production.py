"""Q20: score the ACTUAL PRODUCTION model against the proposed one.

The reference used everywhere in Q8-Q18 was a hand-written stand-in.  The
model that actually runs in production is `GaussianPriceModel` in
`mm/mpc/price_torch.py`, wired via `_build_price_model` with the default kind
"residual" (`mm/strategies/mpc_engine.py:312`).

Its transition law, verbatim from the source:

    m_{t+1} = round( m_t + mu + eps ),   eps ~ N(0, sigma^2)

and its density (`prob_mid`, lines 78-97) is the correct lattice-corrected
Gaussian CDF difference

    q(y) = Phi( (y+0.5-mean)/sigma ) - Phi( (y-0.5-mean)/sigma )

So production is: a pure Gaussian, with NO point mass at zero, a single
variance tracked by EWMA (alpha/gamma in the residual model), and a mean
fitted but, per Q4, not exploitable.

This script scores that exact law prequentially, with the SAME online
recursions production uses, against the proposed hurdle-Student-t.  It is the
decision-relevant comparison, because it is the one that would ship.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 96)
print("Q20 - PRODUCTION GAUSSIAN vs THE PROPOSED HURDLE-STUDENT-T")
print("=" * 96)

print("\n[20.1] What production actually implements")
print("  mm/mpc/price_torch.py:49  GaussianPriceModel")
print("    m_{t+1} = round(m_t + mu + eps),  eps ~ N(0, sigma^2)")
print("    density  q(y) = Phi((y+0.5-m)/s) - Phi((y-0.5-m)/s)   [lattice OK]")
print("    sigma tracked by the residual model's EWMA:")
print("      s2 <- gamma*s2 + alpha*(e - mu)^2     (price_torch.py:273-274)")
print("  Default kind = 'residual' (mpc_engine.py:312)")
print("\n  So production already gets the LATTICE right.  What it lacks is")
print("  the point mass at zero and the tail index.")


class ProductionGaussian:
    """Faithful re-implementation of mm/mpc/price_torch.py:GaussianPriceModel.

    mu is included (production fits it) and sigma follows the production
    residual recursion  s2 <- gamma*s2 + alpha*(e-mu)^2.  The density is the
    production lattice-corrected Gaussian CDF difference.
    """

    name = "PRODUCTION: Gaussian, lattice-corrected, EWMA sigma"

    def __init__(self, alpha=0.10, gamma=0.90, mu=0.0, s0=50.0):
        self.alpha, self.gamma, self.mu = alpha, gamma, mu
        self.s2 = s0 * s0

    def target(self, y, t):
        j = t + 1
        return y["dmid"][j] if j < y["N"] else np.nan

    def predict(self, t, y):
        return self.mu, max(np.sqrt(self.s2), 1e-9), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        e = x - self.mu
        self.s2 = self.gamma * self.s2 + self.alpha * e * e
        if not np.isfinite(self.s2) or self.s2 <= 0:
            self.s2 = 2500.0

    def logpdf(self, tgt, mu, sd):
        sd = max(float(sd), 1e-9)
        hi = norm.cdf((tgt + 0.5 - mu) / sd)
        lo = norm.cdf((tgt - 0.5 - mu) / sd)
        return float(np.log(max(hi - lo, 1e-300)))


def run_prod(model):
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        mu, s_, tgt = model.predict(t, y)
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, mu, s_)
            assert ll <= 1e-9, "invalid density at t=%d" % t
            loss[t] = -ll
            sd[t] = s_
        model.update(t, y)
    return loss, sd


def run_new(model):
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    p0s = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        m_, s_, tgt, p_ = model.predict(t, y)
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, m_, s_, p_)
            assert ll <= 1e-9, "invalid density at t=%d" % t
            loss[t] = -ll
            sd[t], p0s[t] = s_, p_
        model.update(t, y)
    return loss, sd, p0s


CANDS = [
    ("PRODUCTION  Gaussian (mu=0)", lambda: ProductionGaussian(alpha=0.10, gamma=0.90, mu=0.0)),
    ("PRODUCTION  Gaussian (mu=AR1)", lambda: ProductionGaussian(alpha=0.10, gamma=0.90, mu=-0.2)),
    ("PRODUCTION  Gaussian slow (a=.02,g=.98)", lambda: ProductionGaussian(alpha=0.02, gamma=0.98, mu=0.0)),
    ("PROPOSED   hurdle-Student-t nu=2.5",
     lambda: M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98)),
    ("PROPOSED   hurdle-Student-t nu=2.5 no calib",
     lambda: M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98, calibrate=False)),
    ("PROPOSED   hurdle-Student-t nu=4",
     lambda: M.HurdleT(nu=4.0, lam=0.98, lam_p=0.98)),
]

print("\n" + "=" * 96)
res = {}
print("  %-42s %10s %10s %10s" % ("model", "mean", "median", "p99"))
for nm, mk in CANDS:
    L, sd = run_prod(mk()) if "PRODUCTION" in nm else run_new(mk())[:2]
    res[nm] = L
    fin = L[np.isfinite(L)]
    print("  %-42s %10.4f %10.4f %10.4f"
          % (nm, fin.mean(), np.percentile(fin, 50), np.percentile(fin, 99)))

REF = "PRODUCTION  Gaussian (mu=0)"
b = res[REF]
print("\n" + "-" * 96)
print("Paired difference vs PRODUCTION, BY HALF (negative = better)")
print("-" * 96)
print("  %-42s %26s %26s" % ("model", "1st half", "2nd half"))
for nm, L in res.items():
    row = "  %-42s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        dd = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=83)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

FIN = "PROPOSED   hurdle-Student-t nu=2.5"
dd = res[FIN] - b
msk = np.isfinite(dd)
mean, l, h, n = S.moving_block_ci(dd[msk], B=3000, seed=89)
tt, pp = S.dm_test(dd[msk])
print("\n" + "-" * 96)
print("HEADLINE: proposed vs production")
print("-" * 96)
print("  production mean NLL = %.4f" % np.nanmean(b))
print("  proposed   mean NLL = %.4f" % np.nanmean(res[FIN]))
print("  GAIN = %.4f nats/step  (%.2f%% of the production loss)"
      % (-mean, 100 * (-mean) / np.nanmean(b)))
print("  95%% CI on the gain: [%.4f, %.4f]" % (-h, -l))
print("  Diebold-Mariano t = %.1f  p = %.3g  n = %d" % (tt, pp, n))
gain_per_day = -mean * 8640.0
print("  In decision terms: %.0f nats per day per path (8640 buckets)." % gain_per_day)

print("\n" + "-" * 96)
print("CALIBRATION: why the production Gaussian is not just wrong, but")
print("DANGEROUS for a sampler (this is what the MPC consumes)")
print("-" * 96)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
for nm, mk in CANDS[:1] + CANDS[3:4]:
    m = mk()
    if "PRODUCTION" in nm:
        L, sd = run_prod(m)
        pm = np.full(N, np.nan)
    else:
        L, sd, pm = run_new(m)
    msk = (np.arange(N) >= WARM) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    zz = tgt[msk] / sd[msk]
    print("\n  %s" % nm)
    print("    z sd          = %.4f   (calibrated target 1.0)" % zz.std())
    for lev, thr in ((0.50, 0.6745), (0.80, 1.2816), (0.90, 1.6449),
                     (0.99, 2.5758)):
        emp = float(np.mean(np.abs(zz) < thr))
        print("    %.0f%% interval  -> %.4f actual   (under-covers by %.1f%%)"
              % (lev * 100, emp, 100 * (lev - emp) / lev))
    if np.isfinite(pm).any():
        print("    P(y=0) nominal %.4f vs actual %.4f"
              % (np.nanmean(pm[msk]), np.mean(tgt[msk] == 0)))
print("\n" + "=" * 96)
