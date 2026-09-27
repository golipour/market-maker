"""Q18: the final honest comparison, after removing the invalid-density bug.

Recap of the investigation, because the negative result matters as much as
the positive one:

  Q1  the incumbent's iid assumption is false (ACF1 of the standardised
      innovation is -0.189, t = -112) -- the past DOES matter
  Q3  the negative ACF is STRONGER on non-zero moves (-0.258)
  Q4  but the ceiling on any linear mean model is 0.024 nats and the honest
      prequential realisation is 0.0004 -> the mean is not exploitable
  Q7  the negative ACF is MECHANICALLY produced by the zero mass: a hurdle
      with p0 = 0.3507 and an iid body predicts ACF1 = -0.351 alone
  Q8  a properly specified hurdle-Student-t beats the reference by
      -0.065 nats, reproduced on both halves
  Q9  the tail index nu = 2.5 is optimal on both halves independently
  Q13 a three-component mixture appeared to gain -0.818 nats
  Q17 THAT GAIN WAS A BUG: the granular component used a continuous density
      at a lattice point, so q exceeded 1 (an impossible probability mass).
      Lattice-correcting it turned -0.818 into +0.22, i.e. the mixture is
      WORSE than the simple hurdle.

This script (a) measures the size of the invalid-density artifact directly,
and (b) reports the honest final ranking of every candidate.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402
from research.model_lab.analysis import models_dep as MD  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 96)
print("Q18 - THE ARTIFACT, MEASURED; AND THE HONEST FINAL RANKING")
print("=" * 96)

print("\n[18.1] Size of the invalid-density artifact")
print("  A pmf on the .5 lattice obeys  q(y) = F(y+0.25) - F(y-0.25) <= 1.")
print("  A continuous density at a point obeys nothing, and for a narrow")
print("  scale exceeds 1 -- which is illegal as a predictive pmf.")
xs = np.round(np.arange(-60, 61) * 0.5, 1)
for s in (0.25, 0.5, 1.0, 2.0, 10.0):
    from scipy.stats import norm
    cont = np.exp(norm.logpdf(xs, 0.0, s))
    latt = norm.cdf((xs + 0.25) / s) - norm.cdf((xs - 0.25) / s)
    print("    sigma=%5.2f  max continuous density = %8.4f  (must be <= 1)"
          % (s, cont.max()))
print("  So at the learned sigma1 ~ 1.0 the continuous form is legal, but")
print("  the SCALE ITSELF was mis-estimated because the illegal scores fed")
print("  back into the online recursions.  Fixing the pmf changed the")
print("  estimate, and the model got worse -- which is the point: the -0.818")
print("  was never a real gain.")


def run_ref(nu=4.0, lam=0.99):
    loss = np.full(N, np.nan)
    m = MD.HurdleIID(lam, lam, nu)
    for t in range(N - 2):
        if t < WARM:
            m.update(t, y)
            continue
        mu, sd, tgt = m.predict(t, y)
        if np.isfinite(tgt):
            pp = min(max(m.p0(), 1e-6), 1 - 1e-6)
            loss[t] = -(float(np.log(pp)) if tgt == 0
                        else float(M.t_logpmf_lattice(tgt, sd, nu)
                                   - np.log1p(-pp)))
        m.update(t, y)
    return loss


def run_v3(model):
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        m_, s_, tgt, p_ = model.predict(t, y)
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, m_, s_, p_)
            assert ll <= 1e-9, "invalid density at t=%d" % t
            loss[t] = -ll
            sd[t] = s_
        model.update(t, y)
    return loss, sd


print("\n" + "=" * 96)
print("[18.2] HONEST FINAL RANKING (every density lattice-corrected and valid)")
print("=" * 96)
CANDS = [
    ("REF  iid t4 + EWMA (incumbent shape)",
     lambda: run_ref(4.0, 0.99)),
    ("     iid t2.5 + EWMA",
     lambda: run_ref(2.5, 0.99)),
    ("G1   hurdle-t nu=4, global p0",
     lambda: run_v3(M.HurdleT(nu=4.0))),
    ("G2   hurdle-t nu=2.5, global p0  [FINAL]",
     lambda: run_v3(M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98))),
    ("G3   hurdle-t nu=2.5, no calib",
     lambda: run_v3(M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98,
                              calibrate=False))),
    ("G4   hurdle-t nu=2.5, cells(sp,st,oc)",
     lambda: run_v3(M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98,
                              state_fn=M.cell_sp_st_oc))),
    ("G5   hurdle-t nu=3.0, global p0",
     lambda: run_v3(M.HurdleT(nu=3.0, lam=0.98, lam_p=0.98))),
]
res = {}
print("  %-40s %10s %10s %10s" % ("model", "mean", "median", "p99"))
for nm, mk in CANDS:
    L = mk()
    if isinstance(L, tuple):
        L = L[0]
    res[nm] = L
    fin = L[np.isfinite(L)]
    print("  %-40s %10.4f %10.4f %10.4f"
          % (nm, fin.mean(), np.percentile(fin, 50), np.percentile(fin, 99)))

REF = CANDS[0][0]
b = res[REF]
print("\n" + "-" * 96)
print("Paired difference vs the incumbent shape, BY HALF (negative = better)")
print("-" * 96)
print("  %-40s %26s %26s" % ("model", "1st half", "2nd half"))
for nm, L in res.items():
    row = "  %-40s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        d = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(d, B=2000, seed=71)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n" + "-" * 96)
print("FULL-SAMPLE test for the recommended model (G2)")
print("-" * 96)
d = res["G2   hurdle-t nu=2.5, global p0  [FINAL]"] - b
msk = np.isfinite(d)
mean, l, h, n = S.moving_block_ci(d[msk], B=3000, seed=73)
tt, pp = S.dm_test(d[msk])
print("  gain = %.4f nats/step (%.2f%% of the incumbent loss)"
      % (-mean, 100 * (-mean) / np.nanmean(b)))
print("  95%% CI on the gain: [%.4f, %.4f]" % (-h, -l))
print("  Diebold-Mariano t = %.1f   p = %.3g   n = %d" % (tt, pp, n))
print("\n" + "=" * 96)
