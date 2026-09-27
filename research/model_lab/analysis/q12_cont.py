"""Q12: is the CONTINUOUS branch calibrated, and is p0 the remaining gap?

Q11 showed the aggregate PIT is non-uniform.  Two candidate causes:

  (a) the continuous branch (a symmetric Student-t on non-zero moves) is
      itself miscalibrated, or
  (b) the continuous branch is fine and the whole gap is the zero mass p0,
      which the model carries as a single global number while the data
      (Q5.1) shows it ranging from 0.14 to 0.52 across the state cells.

If (b), the fix is the conditional p0 -- but Q10 found the cells made the
NLL WORSE, which would be a paradox: a strictly better-specified density
scoring worse.  That paradox is worth resolving, because it is the
difference between "the model is done" and "the evaluation is wrong".
"""
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 88)
print("Q12 - IS THE CONTINUOUS BRANCH CALIBRATED?")
print("=" * 88)


def collect(state_fn, nu=2.5, lam=0.98):
    m = M.HurdleT(nu=nu, lam=lam, lam_p=lam, state_fn=state_fn)
    rec = np.full((N, 4), np.nan)
    for t in range(N - 2):
        if t < WARM:
            m.update(t, y)
            continue
        mu, sd, tgt, p0 = m.predict(t, y)
        if np.isfinite(tgt):
            rec[t] = (mu, sd, tgt, p0)
        m.update(t, y)
    return rec, m.alpha


r_glob, a_glob = collect(None)
r_cell, a_cell = collect(M.cell_sp_st_oc)
print("\n[12.1] Global vs conditional p0: which predicts the zero better?")
for nm, r in (("global p0", r_glob), ("cell p0", r_cell)):
    ok = np.isfinite(r[:, 3]) & np.isfinite(r[:, 2])
    p0, z = r[ok, 3], (r[ok, 2] == 0)
    # log loss of the hurdle ALONE
    ll = np.where(z, np.log(np.clip(p0, 1e-9, 1)),
                  np.log(np.clip(1 - p0, 1e-9, 1))).mean()
    # calibration: bin by predicted p0 and compare to realised
    qs = np.quantile(p0, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(p0, qs[1:-1]), 0, 9)
    cal = [(round(float(p0[b == i].mean()), 4), round(float(z[b == i].mean()), 4),
            int((b == i).sum())) for i in range(10) if (b == i).sum() > 200]
    print("\n  %s (hurdle-only log loss = %.5f, alpha=%.3f)" % (nm, ll, a_glob))
    print("    %-22s %-22s %s" % ("pred p0", "realised P(y=0)", "n"))
    for pr, re_, nn in cal:
        print("      %-20.4f %-20.4f %d" % (pr, re_, nn))

print("\n[12.2] The CONTINUOUS branch alone: is it calibrated?")
r = r_glob
ok = np.isfinite(r[:, 1]) & (r[:, 1] > 0) & np.isfinite(r[:, 2]) & (r[:, 2] != 0)
sd, obs = r[ok, 1], r[ok, 2]
print("  n = %d non-zero moves" % ok.sum())
print("  sd of the standardised residual = %.4f" % (obs / sd).std())
for lev in (0.50, 0.80, 0.90, 0.95, 0.99):
    a = (1 - lev) / 2
    # central interval of the standard t, NOT via the t CDF (avoids sign bugs)
    q = st.t.ppf(1 - a, df=2.5)
    lo, hi = -q * sd, q * sd
    emp = float(np.mean((obs >= lo) & (obs <= hi)))
    print("    nominal %.2f -> empirical %.4f  (ratio %.3f)"
          % (lev, emp, emp / lev))

print("\n[12.3] Is the non-zero move distribution SYMMETRIC?")
print("  mean/sd = %+.5f" % (obs.mean() / sd.mean()))
mneg = float(np.mean(obs < 0))
print("  P(dmid<0 | dmid!=0) = %.4f   (want 0.5000)" % mneg)
k2 = st.kurtosis(obs / np.std(obs), fisher=False)
print("  kurtosis of the standardised residual = %.3f" % k2)
print("  (t2.5 has excess kurtosis %.3f; Gaussian has 3.0)"
      % (st.kurtosis(st.t.rvs(2.5, size=200000), fisher=False)))

print("\n[12.4] Does the conditional p0 reduce the hurdle log loss?")
lg = []
rc = []
for r_, nm in ((r_glob, "global"), (r_cell, "cell")):
    ok = np.isfinite(r_[:, 3]) & np.isfinite(r_[:, 2])
    p0, z = r_[ok, 3], (r_[ok, 2] == 0)
    lg.append(np.where(z, np.log(np.clip(p0, 1e-9, 1)),
                       np.log(np.clip(1 - p0, 1e-9, 1))))
print("  global p0 hurdle loss = %.6f" % lg[0].mean())
print("  cell   p0 hurdle loss = %.6f" % lg[1].mean())
print("  difference            = %+.6f (negative = conditional is better)"
      % (lg[1].mean() - lg[0].mean()))
print("\n" + "=" * 88)
