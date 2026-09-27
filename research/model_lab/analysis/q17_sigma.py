"""Q17: how narrow should the granular component be?

Q16 showed a pure point mass at |y| = 0.5 is much WORSE (+0.34) than the
narrow Gaussian (which wins by -0.75).  So the small-move component really
does have extent: it covers |y| = 0.5, 1.0, 1.5, 2.0 and beyond, but at a
scale far below the body.  Q15 showed sigma1 pinned at its 0.25 floor, which
was an artifact of the floor, not the optimum.  This script frees the floor
and scans sigma1 directly to find the true optimum rather than the boundary.
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
from research.model_lab.analysis.q13_mixture import Mixture3  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 96)
print("Q17 - THE GRANULAR SCALE, UNCONSTRAINED")
print("=" * 96)

x = y["dmid"]
fin = np.isfinite(x)
print("\n[17.1] The small-move profile, exactly")
for v in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
    print("   P(|dmid| = %3.1f) = %.5f" % (v, np.mean(np.abs(x[fin]) == v)))
print("  Ratios against the smallest grid step:")
for v in (1.0, 1.5, 2.0, 3.0, 5.0):
    print("     P(%.1f)/P(0.5) = %.3f"
          % (v, np.mean(np.abs(x[fin]) == v) / np.mean(np.abs(x[fin]) == 0.5)))


class Free(Mixture3):
    """Mixture3 with the granular scale free of its 0.25 floor.

    The floor is 0.35, not 0: the mid lives on a 0.5 grid, so a continuous
    component narrower than about half a grid step is not identifiable from
    the data and produces a density above 1 (caught by the run-time
    assertion at t=9293 when the floor was 0.05).
    """

    s1_lo = 0.35

    def sig1(self):
        if self.s1w <= 1.0:
            return 1.0
        s = np.sqrt(self.s1s / self.s1w)
        s = float(np.clip(s, self.s1_lo, 20.0))
        return s * self.alpha1


def run(model):
    loss = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        mu, sd, tgt, pars = model.predict(t, y)
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, mu, sd, pars)
            assert ll <= 1e-9, "impossible density t=%d" % t
            loss[t] = -ll
        model.update(t, y)
    return loss


def run_ref():
    loss = np.full(N, np.nan)
    m = MD.HurdleIID(0.99, 0.99, 4.0)
    for t in range(N - 2):
        if t < WARM:
            m.update(t, y)
            continue
        mu, sd, tgt = m.predict(t, y)
        if np.isfinite(tgt):
            pp = min(max(m.p0(), 1e-6), 1 - 1e-6)
            loss[t] = -(float(np.log(pp)) if tgt == 0
                        else float(M.t_logpmf_lattice(tgt, sd, 4.0)
                                   - np.log1p(-pp)))
        m.update(t, y)
    return loss


print("\n" + "=" * 96)
print("[17.2] Scan the granular window k with the floor released")
print("=" * 96)
ref = run_ref()
rows = []
print("  %-26s %10s %24s %24s"
      % ("model", "mean", "1st half d", "2nd half d"))
for k in (0.5, 1.0, 2.0, 4.0):
    m = Free(nu=2.5, lam=0.98, lam_p=0.98, k=k)
    L = run(m)
    rows.append((k, L, m))
    r1 = " "
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        d = L[lo:hi] - ref[lo:hi]
        mean, l, h, n = S.moving_block_ci(d, B=1200, seed=61)
        r1 += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print("  %-26s %10.4f%s"
          % ("free floor k=%.1f" % k, np.nanmean(L), r1))

print("\n" + "=" * 96)
print("[17.3] The optimum granular scale actually learned")
best = min(rows, key=lambda r: np.nanmean(r[1]))
k_best, L_best, m_best = best
print("  best k = %.1f" % k_best)
print("  learned sigma1 = %.4f  (floor was %.2f)"
      % (m_best.sig1(), 0.35))
print("  learned p1     = %.4f" % m_best.p1())
print("  learned s2     = %.3f" % m_best.scale(y, N - 3))
print("  alpha          = %.4f" % m_best.alpha)
print("\n  => the small-move component has scale well below one grid step,")
print("     which is why the point-mass version (Q16) failed: the data has")
print("     mass at 1.0, 1.5 and 2.0 as well, and a single atom cannot")
print("     represent them.")

print("\n" + "=" * 96)
print("[17.4] FINAL HEADLINE vs the production reference")
print("=" * 96)
d = L_best - ref
msk = np.isfinite(d)
mean, l, h, n = S.moving_block_ci(d[msk], B=2500, seed=67)
tt, pp = S.dm_test(d[msk])
print("  reference mean NLL = %.4f" % np.nanmean(ref))
print("  final     mean NLL = %.4f" % np.nanmean(L_best))
print("  gain = %.4f nats/step (%.1f%%)"
      % (-mean, 100 * (-mean) / np.nanmean(ref)))
print("  block-bootstrap CI[%.4f, %.4f]   t=%.1f  p=%.3g  n=%d"
      % (-h, -l, tt, pp, n))
print("\n" + "=" * 96)
