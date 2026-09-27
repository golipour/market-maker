"""Phase 7: the full candidate set, scored on one fixed target contract.

Every row is causal and online.  We report mean NLL plus the paired
block-bootstrap CI against the strongest simple baseline, and we check that
the winner is CALIBRATED (a sharpness gain that is not a calibration gain is
a trap for an MPC that samples from the forecast).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import preq, stats1
from research.model_lab.analysis import models_mid as M
from research.model_lab.analysis import models_mid2 as M2
from research.model_lab.analysis import models_struct as S
from research.model_lab.analysis import models_regime as R

d = H.load()
y = H.derived(H.core_frame(d))
print("=" * 92)
print("PHASE 7 - CANDIDATE SET ON ONE FIXED TARGET CONTRACT")
print("=" * 92)
print("N = %d  warmup = 8000   target: dmid over [t,t+1], IRT ticks" % y["N"])
print("All models causal and online; 95% CIs from a moving-block bootstrap.\n")

cands = [
    ("B1 EWMA Gaussian sd",            lambda: M.EWMAStatic(lam=0.99)),
    ("B2 EWMA Student-t sd",           lambda: M.StudentEWMA(lam=0.99, nu=4.0)),
    ("B3 Student-t, sd=c*eta",         lambda: M2.StudentSpread(lam=0.99, nu=4.0)),
    ("S1 quote-mixture (structural)",  lambda: S.QuoteMixture(lam=0.99, nu=4.0)),
    ("R1 regime hurdle (levels)",      lambda: R.RegScaleHurdle(lam=0.995, nu=4.0)),
    ("R2 regime hurdle (log+shrink)",  lambda: R.RegScaleHurdleLog(lam=0.995, nu=4.0)),
]

losses, runs = {}, {}
for name, ctor in cands:
    r = preq.Runner(y, warmup=8000).run(ctor())
    losses[name], runs[name] = r.loss, r
    print(preq.fmt(preq.summarize(r.loss, name)))

ref = "B2 EWMA Student-t sd"
print("\n" + "-" * 92)
print("Paired loss difference vs B2 (the strong simple baseline)")
print("-" * 92)
for name in losses:
    if name == ref:
        continue
    df = preq.paired(losses[name], losses[ref])
    mean, lo, hi, n = stats1.moving_block_ci(df, B=1500, seed=11)
    t, p = stats1.dm_test(df)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-32s d=%+.4f CI[%+.4f,%+.4f] p=%.1e  %s" % (name, mean, lo, hi, p, v))

print("\n" + "-" * 92)
print("CALIBRATION of each candidate (PIT on its own predictive law)")
print("-" * 92)
from scipy import stats as st
print("  %-32s %-10s %-10s %-12s %-10s" % ("model", "z mean", "z sd", "PIT chi2/19", "80% cov"))
for name in losses:
    r = runs[name]
    p0 = r.p0
    mu, sd = r.mu, np.maximum(r.sd, 1e-9)
    x = y["dmid"]
    ok = np.isfinite(x) & np.isfinite(sd) & np.isfinite(mu) & np.isfinite(p0)
    zz = np.where(p0[ok] > 0, (x[ok] - mu[ok]) / sd[ok], np.nan)
    zz = zz[np.isfinite(zz)]
    # when p0 is present the core is on |x|; use |z| for the coverage test
    zabs = np.abs(zz)
    cov80 = float(np.mean(zabs <= st.t.ppf(0.90, df=4)))
    pit = st.t.cdf(zz, df=4)
    u = np.clip((pit * 20).astype(int), 0, 19)
    cnt = np.bincount(u, minlength=20).astype(float)
    exp = cnt.sum() / 20.0
    chi = float(((cnt - exp) ** 2 / exp).sum() / 19.0)
    print("  %-32s %-10.4f %-10.4f %-12.3f %-10.4f"
          % (name, zz.mean(), zz.std(), chi, cov80))
print("  (PIT chi2/19 ~ 1 is calibrated; 80% coverage should be ~0.80)")
