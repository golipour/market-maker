"""Q27: does the hurdle survive ON TOP of the real production model?

Q26 showed the hurdle-t beats the real incumbent 5.13 -> 3.52.  But that is
a like-for-like comparison against a model that has one thing the hurdle
does NOT: the conditional gate + RLS mean.  The right question is not
"hurdle vs production" but

    production mixture  +  a zero atom   vs   production mixture

because the zero atom is the ONLY genuinely missing structural piece.  The
gate, the RLS mean and the Student-t refresh branch are all already there and
should be kept.

Candidate:  q(y) = p0 * 1{y=0} + (1-p0) * [ w*N(0,mv) + (1-w)*t_nu(0,rv) ]

The lattice handling is exact on both branches (CDF differences), and p0 is
estimated by EWMA on the zero indicator, exactly as production estimates its
other state.
"""
import os
import sys

import numpy as np
from scipy.stats import norm, t as tdist

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
print("=" * 96)
print("Q27 - ADDING ONLY THE ZERO ATOM TO THE REAL PRODUCTION MODEL")
print("=" * 96)


class ProdPlusHurdle:
    """Production's regime mixture, PLUS a zero atom.

    q(y) = p0 * 1{y == 0} + (1 - p0) * [ w * Gauss(0, mv) + (1-w) * t_nu(0, rv) ]

    Everything except p0 is production's own recursion.  p0 is an EWMA on the
    zero indicator with the same decay (0.99) the rest of the model uses.
    """

    def __init__(self, nu=3.0, decay=0.99, s0=50.0, p0_0=0.35, hurdle=True):
        self.nu = nu
        self.a = 1.0 - decay
        self.mv = s0 * s0
        self.rv = s0 * s0
        self.w = 0.5
        self.hurdle = hurdle
        self.p0n = p0_0
        self.p0d = p0_0

    def target(self, t):
        j = t + 1
        return y["dmid"][j] if j < y["N"] else np.nan

    def _body(self, tgt):
        gm = (norm.cdf((tgt + 0.5) / np.sqrt(self.mv))
              - norm.cdf((tgt - 0.5) / np.sqrt(self.mv)))
        s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))
        gr = (tdist.cdf((tgt + 0.5) / s, df=self.nu)
              - tdist.cdf((tgt - 0.5) / s, df=self.nu))
        return self.w * gm + (1.0 - self.w) * gr

    def logpdf(self, tgt):
        if self.hurdle:
            p0 = min(max(self.p0n / self.p0d, 1e-6), 1.0 - 1e-6)
            if tgt == 0:
                return float(np.log(p0))
            return float(np.log(max((1.0 - p0) * self._body(tgt), 1e-300)))
        return float(np.log(max(self._body(tgt), 1e-300)))

    def update(self, t):
        x = self.target(t)
        if not np.isfinite(x):
            return
        w = self.w
        self.mv = max((1 - self.a * w) * self.mv + self.a * w * x * x, 1e-6)
        self.rv = max((1 - self.a * (1 - w)) * self.rv + self.a * (1 - w) * x * x, 1e-6)
        s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))
        llm = np.log(max(w * norm.pdf(x, 0, np.sqrt(self.mv)), 1e-300))
        llr = np.log(max((1 - w) * tdist.pdf(x, df=self.nu, scale=s), 1e-300))
        resp = float(np.exp(llm - np.logaddexp(llm, llr)))
        self.w = float(np.clip(0.98 * self.w + 0.02 * resp, 0.01, 0.99))
        if self.hurdle:
            z = 1.0 if x == 0.0 else 0.0
            self.p0n = (1 - self.a) * self.p0n + self.a * z
            self.p0d = (1 - self.a) * self.p0d + self.a


def run(model):
    L = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t)
            continue
        tgt = model.target(t)
        if np.isfinite(tgt):
            L[t] = -model.logpdf(tgt)
        model.update(t)
    return L


def runh(model):
    L = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        mu, sd, tgt, p0 = model.predict(t, y)
        if np.isfinite(tgt):
            L[t] = -model.logpdf(tgt, mu, sd, p0)
        model.update(t, y)
    return L


CANDS = [
    ("PROD  mixture (no zero atom)", lambda: ProdPlusHurdle(nu=3.0, hurdle=False)),
    ("PROD  mixture + zero atom  nu=3", lambda: ProdPlusHurdle(nu=3.0, hurdle=True)),
    ("PROD  mixture + zero atom  nu=2.5", lambda: ProdPlusHurdle(nu=2.5, hurdle=True)),
    ("hurdle-t only (no gate/mixture)", lambda: M.HurdleTLattice(nu=2.5, lam=0.98, lam_p=0.98)),
]
res = {}
print("\n[27.1] Prequential NLL")
print("-" * 96)
for nm, mk in CANDS:
    L = runh(mk()) if "hurdle-t only" in nm else run(mk())
    res[nm] = L
    print("  %-40s mean NLL = %8.4f" % (nm, np.nanmean(L)))

REF = "PROD  mixture (no zero atom)"
b = res[REF]
print("\n[27.2] Paired difference vs the real incumbent (negative = better)")
print("-" * 96)
print("  %-40s %25s %25s" % ("model", "1st half", "2nd half"))
for nm, L in res.items():
    if nm == REF:
        continue
    row = "  %-40s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        dd = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=151)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

for nm in ("PROD  mixture + zero atom  nu=3", "PROD  mixture + zero atom  nu=2.5"):
    dd = res[nm] - b
    mean, l, h, n = S.moving_block_ci(dd, B=3000, seed=157)
    tt, pp = S.dm_test(dd)
    print("\n  %s" % nm)
    print("    gain %.4f nats  95%% CI [%.4f, %.4f]  DM t=%.1f p=%.3g"
          % (-mean, -h, -l, tt, pp))
print("=" * 96)
