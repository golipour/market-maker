"""Q23: rescore with the LATTICE-CORRECTED hurdle, and check the PIT properly.

Two defects found in my own Q20/Q21 work:

  1. The model I scored was `HurdleT`, whose logpdf uses `t_logpdf_sym` -- a
     CONTINUOUS point density -- not the lattice PMF.  On a 0.5 IRT grid that
     is the same mis-scaling bug I caught in the 3-component mixture earlier.
     The correct class is `HurdleTLattice`.

  2. My PIT routine in Q22 returned KS D = 1.0000, which is not a plausible
     statistic.  A KS D of exactly 1.0 means the empirical CDF is degenerate,
     i.e. the PIT values collapsed to a point.  Cause: for the hurdle the
     non-zero branch PIT must be offset by the zero mass (F(y) = p0 + (1-p0)
     G(y)), and I omitted that offset, so every body draw was scored in the
     wrong region.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")

from research.model_lab.analysis import harness as H
from research.model_lab.analysis import stats1 as S
from research.model_lab.analysis import models_v3 as M

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 94)
print("Q23 - LATTICE-CORRECTED RESCORE")
print("=" * 94)


def ks(u):
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    if n == 0:
        return 0.0
    i = np.arange(1, n + 1) / float(n)
    return float(np.max(np.maximum(i - u, u - (i - 1.0) / n)))


class Prod:
    nu = None

    def __init__(self, decay=0.9, s0=50.0, mu=0.0, learn=True):
        self.g = decay
        self.a = 1.0 - self.g
        self.mu0 = mu
        self.mu = mu
        self.s2 = s0 * s0
        self.learn = learn

    def target(self, t):
        j = t + 1
        return y["dmid"][j] if j < y["N"] else np.nan

    def predict(self, t, y=None):
        return self.mu, max(np.sqrt(self.s2), 1e-9), self.target(t)

    def update(self, t, y=None):
        x = self.target(t)
        if not np.isfinite(x):
            return
        e = x - self.mu
        mn = self.g * self.mu + self.a * e if self.learn else self.mu0
        self.s2 = self.g * self.s2 + self.a * (e - mn) ** 2
        if self.learn:
            self.mu = mn
        if not np.isfinite(self.s2) or self.s2 <= 0:
            self.s2 = 2500.0

    def logpdf(self, tgt, mu, sd):
        sd = max(float(sd), 1e-9)
        hi = norm.cdf((tgt + 0.5 - mu) / sd)
        lo = norm.cdf((tgt - 0.5 - mu) / sd)
        return float(np.log(max(hi - lo, 1e-300)))


def run(model, hurdle):
    """Prequential. Returns nll, and a CORRECT discrete PIT uniform.

    For the hurdle the full predictive cdf at lattice point y is
        F(y) = p0 + (1 - p0) * G(y)      (G = lattice t CDF)
    so the PIT for a non-zero y is p0 + (1-p0)*(mid-cell position in G).
    Getting this offset wrong is what produced KS = 1.0 in Q22.
    """
    nll = np.full(N, np.nan)
    pit = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        if hurdle:
            mu, sd, tgt, p0 = model.predict(t, y)
        else:
            mu, sd, tgt = model.predict(t, y)
            p0 = 0.0
        if np.isfinite(tgt):
            if hurdle:
                ll = model.logpdf(tgt, mu, sd, p0)
            else:
                ll = model.logpdf(tgt, mu, sd)
            assert ll <= 1e-9, "invalid density t=%d" % t
            nll[t] = -ll
            if hurdle:
                s = max(float(sd), 1e-9)
                g0 = M.t_cdf((tgt - 0.5 - mu) / s, model.nu)
                g1 = M.t_cdf((tgt + 0.5 - mu) / s, model.nu)
                base = p0
                span = (1.0 - p0)
            else:
                g0 = norm.cdf((tgt - 0.5 - mu) / sd)
                g1 = norm.cdf((tgt + 0.5 - mu) / sd)
                base = 0.0
                span = 1.0
            u = rng.random()
            pit[t] = base + span * (g0 + u * (g1 - g0))
        model.update(t, y)
    return nll, pit


rng = np.random.default_rng(4242)


CANDS = [
    ("PRODUCTION  Gaussian decay 0.9", lambda: Prod(decay=0.9), False),
    ("PRODUCTION  Gaussian decay 0.99", lambda: Prod(decay=0.99), False),
    ("Q20/Q21 WRONG hurdle-t (point density)", lambda: M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98), True),
    ("CORRECT    hurdle-t LATTICE nu=2.5", lambda: M.HurdleTLattice(nu=2.5, lam=0.98, lam_p=0.98), True),
    ("CORRECT    hurdle-t LATTICE nu=4", lambda: M.HurdleTLattice(nu=4.0, lam=0.98, lam_p=0.98), True),
]

print("\n[23.1] Prequential NLL and CORRECTED discrete PIT")
print("-" * 94)
print("  %-42s %10s %9s %9s %9s" % ("model", "meanNLL", "PITmean", "KS", "crit"))
res = {}
pits = {}
for nm, mk, hur in CANDS:
    L, P = run(mk(), hur)
    res[nm] = L
    pits[nm] = P
    fin = P[np.isfinite(P)]
    k = ks(fin)
    print("  %-42s %10.4f %9.4f %9.5f %9.5f"
          % (nm, np.nanmean(L), fin.mean(), k, 1.36 / np.sqrt(fin.size)))

REF = "PRODUCTION  Gaussian decay 0.99"
COR = "CORRECT    hurdle-t LATTICE nu=2.5"
OLD = "Q20/Q21 WRONG hurdle-t (point density)"
b = res[REF]
print("\n[23.2] Paired difference vs TUNED Gaussian baseline (negative = better)")
print("-" * 94)
print("  %-42s %25s %25s" % ("model", "1st half", "2nd half"))
for nm in (OLD, COR, "CORRECT    hurdle-t LATTICE nu=4"):
    L = res[nm]
    row = "  %-42s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        dd = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=83)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n[23.3] Did the lattice correction change the verdict?")
print("-" * 94)
for a, c in ((OLD, COR),):
    L1, L2 = res[a], res[c]
    for lo, hi, lab in ((WARM, HALF, "1st half"), (HALF, N - 2, "2nd half")):
        dd = L1[lo:hi] - L2[lo:hi]
        mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=87)
        print("  wrong-minus-correct, %s: %+.4f nats [%+.4f,%+.4f]" % (lab, mean, l, h))
dd = res[COR] - b
mean, l, h, n = S.moving_block_ci(dd, B=3000, seed=89)
tt, pp = S.dm_test(dd)
print()
print("  HEADLINE (corrected): %.4f -> %.4f nats" % (np.nanmean(b), np.nanmean(res[COR])))
print("  gain %.4f nats/step, 95%% CI [%.4f, %.4f], DM t=%.1f p=%.3g n=%d"
      % (-mean, -h, -l, tt, pp, n))
print("  = %.1f%% of tuned baseline, %.0f nats/day/path"
      % (100 * (-mean) / np.nanmean(b), -mean * 8640))
