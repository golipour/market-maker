"""Q9: scan the tail index nu and the forgetting profile.

Q8 fixed the structure and found the winner was nu = 2.5, monotonically
improving as nu fell (4.0 -> 2.5 gained 0.047 nats).  That points below 2.5,
so this script scans it properly rather than assuming, and also checks:

  * is the optimum interior, or does the model want a pure power law?
  * does the forgetting constant lambda interact with nu?
  * is the gain real once we correct for scanning many values (the winner's
    curse)?  The block-bootstrap CI on the SELECTED winner is optimistic, so
    we report the honest picture: scan on the first half, confirm on the
    second half.
"""
import os
import sys

import numpy as np

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
print("=" * 92)
print("Q9 - TAIL INDEX AND FORGETTING (split-sample to defeat the winner's curse)")
print("=" * 92)


def run(model, lo, hi):
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    p0 = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        if t < lo:
            model.update(t, y)
            continue
        m_, s_, tgt, p_ = model.predict(t, y)
        if t < hi and np.isfinite(tgt):
            ll = model.logpdf(tgt, m_, s_, p_)
            assert ll <= 1e-9, "impossible density at t=%d" % t
            loss[t] = -ll
            sd[t], p0[t] = s_, p_
        model.update(t, y)
    return loss, sd, p0


def seg(L, lo, hi):
    v = L[lo:hi]
    return float(np.nanmean(v)), int(np.isfinite(v).sum())


tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]

NUS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
print("\n[9.1] Scan nu on the FIRST half; confirm on the SECOND half.")
print("  %-8s %12s %12s %10s" % ("nu", "mean 1st half", "mean 2nd half", "alpha"))
res = {}
for nu in NUS:
    m = M.HurdleT(nu=nu, state_fn=M.cell_sp_st_oc)
    L, sd, p0 = run(m, WARM, N)
    a1, n1 = seg(L, WARM, HALF)
    a2, n2 = seg(L, HALF, N - 2)
    res[nu] = (L, sd, p0, m.alpha)
    print("  %-8.1f %12.4f %12.4f %10.3f" % (nu, a1, a2, m.alpha))

print("\n[9.2] Paired differences vs nu=4.0, computed SEPARATELY on each half.")
print("  (if the gain only exists in the half used for scanning, it is a curse)")
base = res[4.0][0]
print("  %-8s %28s %28s" % ("nu", "1st half d [CI]", "2nd half d [CI]"))
for nu in NUS:
    L = res[nu][0]
    row = "  %-8.1f" % nu
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        msk = slice(lo, hi)
        d = L[msk] - base[msk]
        mean, l, h, n = S.moving_block_ci(d, B=800, seed=5)
        row += "  %+.4f [%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n[9.3] Calibration at the best nu (both halves, separately)")
best = min(NUS, key=lambda u: np.nanmean(res[u][0]))
print("  best nu by full-sample mean = %.1f" % best)
L, sd, p0 = res[best][0], res[best][1], res[best][2]
for lo, hi, nm in ((WARM, HALF, "1st"), (HALF, N - 2, "2nd")):
    msk = np.zeros(N, bool)
    msk[lo:hi] = True
    m = msk & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    zz = (tgt[m] - 0.0) / sd[m]
    print("    %s half: z mean=%+.4f  z sd=%.4f  80%% cov=%.4f"
          % (nm, zz.mean(), zz.std(), np.mean(np.abs(zz) < 1.2816)))
print("\n[9.4] Forgetting constant lambda at the best nu")
for lam in (0.980, 0.990, 0.995, 0.998):
    m = M.HurdleT(nu=best, lam=lam, lam_p=lam, state_fn=M.cell_sp_st_oc)
    L2, _, _ = run(m, WARM, N)
    a1, _ = seg(L2, WARM, HALF)
    a2, _ = seg(L2, HALF, N - 2)
    d1 = L2[WARM:HALF] - base[WARM:HALF]
    mm, l, h, _ = S.moving_block_ci(d1, B=600, seed=9)
    print("  lam=%.3f  1st=%.4f 2nd=%.4f  d_vs_nu4=%+.4f CI[%+.4f,%+.4f]"
          % (lam, a1, a2, mm, l, h))
print("\n" + "=" * 92)
