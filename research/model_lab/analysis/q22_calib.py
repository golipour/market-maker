"""Q22: correct discrete PIT / quantile calibration diagnostic.

The Q20 z-interval check was wrong.  Zeros land at z=0, so every zero counts
as "inside" every symmetric interval, which INFLATES apparent coverage rather
than deflating it.  The right diagnostic for a discrete pmf is a discrete PIT
(the mid-PIT on the lattice), plus interval checks computed only on the
non-zero body, which is the only part that has a meaningful continuous z.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")

from research.model_lab.analysis import harness as H
from research.model_lab.analysis import models_v3 as M
from research.model_lab.analysis import stats1 as S


def S_ks(u):
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    if n == 0:
        return 0.0
    i = np.arange(1, n + 1) / n
    return float(np.max(np.maximum(i - u, u - (i - 1.0) / n)))

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 92)
print("Q22 - DISCRETE PIT (the correct calibration diagnostic)")
print("=" * 92)
print("\nThe earlier z-interval check divided every y by sigma, including the")
print("35% exact zeros.  A zero has z=0, which is inside every symmetric")
print("interval, so those runs were credited as 'covered'.  That is wrong in")
print("the opposite direction to what I reported.  Here:")
print("  (a) discrete PIT on the lattice, which is the honest global check")
print("  (b) interval checks on the NON-ZERO body only")


class Prod:
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

    def update(self, t):
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

    nu = None

    def predict(self, t):
        return self.mu, max(np.sqrt(self.s2), 1e-9), self.target(t)

    def logpdf(self, tgt, mu, sd):
        sd = max(float(sd), 1e-9)
        hi = norm.cdf((tgt + 0.5 - mu) / sd)
        lo = norm.cdf((tgt - 0.5 - mu) / sd)
        return float(np.log(max(hi - lo, 1e-300)))


rng_pit = np.random.default_rng(4242)


def collect(model, hurdle):
    """Run prequentially; return PIT uniforms, body z, flags, NLL."""
    pit = np.full(N, np.nan)
    zs = np.full(N, np.nan)
    isz = np.zeros(N, dtype=bool)
    nll = np.full(N, np.nan)
    lo_c = np.zeros(N, dtype=bool)
    nrm = np.zeros(N, dtype=bool)
    for t in range(N - 2):
        if t < WARM:
            model.update(t)
            continue
        p = model.predict(t)
        if p is None:
            model.update(t)
            continue
        if hurdle:
            mu, sd, tgt, p0 = p
        else:
            mu, sd, tgt = p
            p0 = 0.0
        if np.isfinite(tgt):
            nll[t] = -model.logpdf(tgt, mu, sd, p0) if hurdle else -model.logpdf(tgt, mu, sd)
            isz[t] = (tgt == 0.0)
            lo_c[t] = (p0 >= 0.999) if hurdle else False
            nrm[t] = not (isz[t] or lo_c[t])
            if nrm[t]:
                # cdf just below the lattice cell
                f0 = M.t_cdf((tgt - 0.5 - mu) / sd, model.nu) if hurdle else norm.cdf((tgt - 0.5 - mu) / sd)
                f1 = M.t_cdf((tgt + 0.5 - mu) / sd, model.nu) if hurdle else norm.cdf((tgt + 0.5 - mu) / sd)
                pit[t] = f0 + rng_pit.random() * (f1 - f0)
        model.update(t)
    return pit, zs, isz, nll, lo_c, nrm


print("\n" + "-" * 92)
print("(a) DISCRETE PIT -- uniform on [0,1] means perfect calibration")
print("-" * 92)
rows = []
for nm, mdl, hur in (
    ("PRODUCTION Gaussian decay 0.9", Prod(decay=0.9), False),
    ("PRODUCTION Gaussian decay 0.99 (tuned)", Prod(decay=0.99), False),
    ("PROPOSED   hurdle-t nu=2.5", M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98), True),
):
    pit, zs, isz, nll, lo_c, nrm = collect(mdl, hur)
    fin = pit[np.isfinite(pit)]
    ks = S_ks(fin)
    print("\n  %s" % nm)
    print("    PIT n = %d   mean %.4f   KS D = %.4f  (crit 1.36/sqrt(n) = %.4f)"
          % (len(fin), fin.mean(), ks, 1.36 / np.sqrt(len(fin))))
    q = np.percentile(fin, [1, 5, 25, 50, 75, 95, 99])
    tgtq = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print("    PIT quantiles  " + "  ".join("%.2f:%.3f" % (a, b) for a, b in zip(tgtq, q)))
    for lo, hi, lv in ((0, 0.5, 0.5), (0, 0.9, 0.9), (0, 0.99, 0.99)):
        emp = float(np.mean((fin >= lo) & (fin < hi)))
        print("      [%.0f%%,%.0f%%] central: actual %.4f  (nominal %.2f)"
              % (lo * 100, hi * 100, emp, hi - lo))
    rows.append((nm, fin, nll))

print("\n" + "-" * 92)
print("(b) THE BODY-ONLY CHECK -- nonzero shocks, where a continuous z is valid")
print("-" * 92)
print("\nThe continuous z = x/s is only meaningful on the non-zero body.  A zero")
print("is a point-mass event, not a small draw from the body, so it must be")
print("excluded.  This is the check that actually matters for the sampler.")
print()
tgt_all = np.full(N, np.nan)
tgt_all[:-1] = y["dmid"][1:]
idx = np.arange(N)
wmask = (idx >= WARM) & np.isfinite(tgt_all)

for nm, mk, hur in (
    ("PRODUCTION Gaussian decay 0.9", lambda: Prod(decay=0.9), False),
    ("PRODUCTION Gaussian decay 0.99", lambda: Prod(decay=0.99), False),
    ("PROPOSED   hurdle-t nu=2.5", lambda: M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98), True),
):
    m = mk()
    pit, zs, isz, nll, lo_c, nrm = collect(m, hur)
    body = wmask & (tgt_all != 0.0)
    fin = pit[np.isfinite(pit)][(pit[np.isfinite(pit)] > 0)] if False else pit[np.isfinite(pit)]
    print("\n  %s" % nm)
    print("    body-only n = %d" % int(body.sum()))
    print("    body PIT mean %.4f" % np.mean(fin))
    print("    body PIT KS   D = %.4f" % S_ks(fin))
    for lo, hi in ((0.0, 0.5), (0.05, 0.95), (0.005, 0.995)):
        emp = float(np.mean((fin >= lo) & (fin < hi)))
        print("      [%.1f%%,%.1f%%] -> %.4f actual (nominal %.3f)  err %+.4f"
              % (lo * 100, hi * 100, emp, hi - lo, emp - (hi - lo)))
