"""Q2: the serial-dependence ladder, scored prequentially WITH calibration.

Every candidate is scored the same way: emit a density at t, score against the
move realised over [t, t+1], THEN update.  Paired differences against the
incumbent use a moving-block bootstrap, because the loss series is strongly
autocorrelated and IID standard errors would be inadmissible.

Calibration is reported for every candidate, not just the loss, because the
roadmap's central methodological point is that an over-confident density can
win on NLL and still be dangerous for an MPC that samples from it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import preq as P  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import models_dep as M  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 90)
print("Q2 - SERIAL-DEPENDENCE LADDER (prequential, paired, block bootstrap)")
print("=" * 90)
print("N = %d  warmup = %d  target: dmid over [t, t+1], IRT ticks" % (N, WARM))


def run(model, name=None):
    r = P.Runner(y, warmup=WARM).run(model)
    return r, (name or model.name)


cands = [
    M.IIDStudent(lam=0.99, nu=4.0),
    M.HurdleIID(lam=0.99, lam_p=0.99, nu=4.0),
    M.ArchMean(phi=-0.18, lam=0.99, lam_p=0.99, nu=4.0),
    M.GarchScale(lam=0.97, c=0.0, lam_p=0.99, nu=4.0),
    M.ArchGarch(lam_g=0.97, lam_p=0.99, use_mean=True),
    M.ArchGarch(lam_g=0.97, lam_p=0.99, use_mean=False),
]

res, losses = {}, {}
for m in cands:
    r, nm = run(m)
    res[nm] = r
    losses[nm] = r.loss
    print(P.fmt(P.summarize(r.loss, nm)))

REF = "D0 iid Student-t, EWMA scale"
print("\n" + "-" * 90)
print("Paired difference vs %s  (negative = better)" % REF)
print("-" * 90)
for nm, L in losses.items():
    dlt = P.paired(L, losses[REF])
    mean, lo, hi, n = S.moving_block_ci(dlt, B=1500, seed=7)
    tt, pp = S.dm_test(dlt)
    verdict = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-46s d=%+.4f CI[%+.4f,%+.4f]  t=%+8.1f p=%.3g  %s"
          % (nm, mean, lo, hi, tt, pp, verdict))

print("\n" + "-" * 90)
print("CALIBRATION (PIT on each model's own predictive law)")
print("-" * 90)
print("  %-46s %8s %8s %10s" % ("model", "z mean", "z sd", "80% cov"))
for nm, r in res.items():
    mu, sd = r.mu, r.sd
    tgt = np.full(N, np.nan)
    tgt[:-1] = y["dmid"][1:]
    m = np.isfinite(mu) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    if m.sum() < 1000:
        print("  %-46s (insufficient)" % nm)
        continue
    zz = (tgt[m] - mu[m]) / sd[m]
    cov = float(np.mean(np.abs(zz) < 1.2816))
    print("  %-46s %+8.4f %8.4f %10.4f"
          % (nm, zz.mean(), zz.std(), cov))
print("  want z mean ~ 0, z sd ~ 1, 80% cov ~ 0.80")
print("\n" + "=" * 90)
