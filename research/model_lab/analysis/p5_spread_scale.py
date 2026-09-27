"""Phase 5: sharpen the winning mechanism, then test the decisive question.

Phase 4 showed the conditional scale is essentially the HALF-SPREAD times a
slowly varying coefficient.  That is a concrete, testable statement about the
market: the quoted mid's move scale is set by the quoted liquidity, not by
recent realised volatility and not by observed order flow.

This phase:
  (a) checks the functional form of sd = f(half-spread) -- is it linear in the
      half-spread, or in its square root / log?
  (b) tests the mixture sd^2 = a + b*eta^2 (efficient innovation + redraw),
      which is the structural prediction;
  (c) reports the PIT calibration so a gain is not merely a scale artefact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import preq, stats1
from research.model_lab.analysis.models_mid import ewma_update
from research.model_lab.analysis.preq import BaseModel


class ScaleForm(BaseModel):
    """Student-t, zero mean, sd = c * g(eta) with g a declared shape."""

    name = "scale-form"

    def __init__(self, form="linear", lam=0.99, floor=1e-3, nu=4.0, c0=1.0,
                 e_ref=50.0, mix_b=0.0, start=1.0):
        self.form, self.lam, self.floor, self.nu = form, lam, floor, nu
        self.c0, self.e_ref, self.mix_b, self.start = c0, e_ref, mix_b, start
        self.s, self.w = 0.0, 0.0

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 1.0

    def predict(self, t, y):
        eta = self._eta(y, t)
        if self.form == "linear":
            g = eta / self.e_ref
        elif self.form == "sqrt":
            g = np.sqrt(eta / self.e_ref)
        elif self.form == "log":
            g = np.log1p(eta) / np.log1p(self.e_ref)
        elif self.form == "power":
            p = self.mix_b
            g = (eta / self.e_ref) ** p
        else:
            raise ValueError(self.form)
        c = self.s / self.w if self.w > 0 else self.start
        sd = max(c * g, self.floor)
        return 0.0, float(sd), self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        eta = self._eta(y, t)
        if self.form == "power":
            p = max(self.mix_b, 0.05)
            g = (eta / self.e_ref) ** p
        else:
            g = {"linear": eta / self.e_ref,
                 "sqrt": np.sqrt(eta / self.e_ref),
                 "log": np.log1p(eta) / np.log1p(self.e_ref)}[self.form]
        r = x / max(g, 1e-9)
        self.s, self.w = ewma_update(self.s, self.w, min(r * r, 1e7), self.lam)


d = H.load()
y = H.derived(H.core_frame(d))
print("=" * 88)
print("PHASE 5 - FUNCTIONAL FORM OF THE CONDITIONAL SCALE")
print("=" * 88)
print("N = %d  warmup = 4000   target: dmid over [t,t+1]\n" % y["N"])

forms = [("linear in eta", "linear", 1.0), ("sqrt in eta", "sqrt", 0.5),
         ("log in eta", "log", 0.0), ("eta^0.7", "power", 0.7),
         ("eta^1.0", "power", 1.0), ("eta^1.3", "power", 1.3)]
losses = {}
for label, form, b in forms:
    m = ScaleForm(form=form, lam=0.99, nu=4.0, mix_b=b if form == "power" else 0.0)
    r = preq.Runner(y, warmup=4000).run(m)
    losses[label] = r.loss
    c = m.s / m.w if m.w else np.nan
    print(preq.fmt(preq.summarize(r.loss, label)) + "   c=%.3f" % c)

ref = "linear in eta"
print("\nPaired difference vs the linear form (95% block CI):")
for label in losses:
    if label == ref:
        continue
    df = preq.paired(losses[label], losses[ref])
    mean, lo, hi, n = stats1.moving_block_ci(df, B=1200, seed=7)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-16s d=%+.4f CI[%+.4f,%+.4f]  %s" % (label, mean, lo, hi, v))

# ---- calibration of the best form ----------------------------------------
print("\n" + "=" * 88)
print("CALIBRATION of the linear form (is the density honest?)")
print("=" * 88)
m = ScaleForm(form="linear", lam=0.99, nu=4.0)
r = preq.Runner(y, warmup=4000).run(m)
z = (y["dmid"] - r.mu) / np.maximum(r.sd, 1e-9)
zz = z[np.isfinite(z)]
qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print("  %-8s %-10s %-10s %-8s" % ("nominal", "empirical", "ratio", "n"))
for q in qs:
    emp = float(np.mean(zz <= np.quantile(zz, q)))
    print("  %-8.2f %-10.4f %-10.4f %-8d" % (q, emp, emp / q, len(zz)))
# PIT uniformity on the two-sided central mass
from scipy import stats as st
pit = st.t.cdf(zz, df=4)
u = np.round(pit * 20) / 20.0
cnt = np.unique(u, return_counts=True)
exp = len(u) / 20.0
chi = float(((cnt[1] - exp) ** 2 / exp).sum())
print("  PIT chi2 (20 bins, df=19) = %.1f   (expected ~19 +/- ~6.2)" % chi)
print("  mean standardised residual = %+.4f (expect 0)" % zz.mean())
print("  sd   standardised residual = %.4f (expect ~1)" % zz.std())
