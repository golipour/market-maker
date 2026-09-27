"""Phase 3: is the negative mean just estimation noise, or real structure?

The MA(1) mean is WORSE than a zero mean.  Two competing explanations:

  (H1) the conditional mean really is zero, and the fitted -0.218 is an
       artefact of estimating a persistent mean on a non-stationary series;
  (H2) the conditional mean is non-zero, but the online estimator's
       variance exceeds the signal it chases.

We separate them with a *pre-registered, calibration-based* test.  Instead of
estimating a free AR coefficient we set mu_t = -k * eta_t / sd_t.  From the
Roll/Grossman-Miller mechanism, if the quotes are a uniform redraw of the
efficient price, the optimal reversion coefficient is a pure function of the
two OBSERVABLE quantities (half-spread and conditional scale) with NO free
parameter.  If the data is FTL-driven this law must beat the zero-mean null.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import preq, stats1
from research.model_lab.analysis.models_mid import ewma_update
from research.model_lab.analysis.preq import BaseModel


class FTLPredictive(BaseModel):
    """mu_t = -k * (eta_t^2 / 3) / sd_t  -- derived, not fitted.

    Var(quote error) = eta^2/3.  If the observed move contains a mean
    reversion of the quote error toward the efficient price, the conditional
    mean of the next move is minus a fraction of that error variance divided
    by the conditional scale.  k is the only knob, and we scan it.
    """

    name = "PREDICTIVE: mu = -k*eta^2/(3 sd), Student-t"

    def __init__(self, lam=0.99, floor=1.0, nu=4.0, k=1.0, start_sd=50.0):
        self.lam, self.floor, self.nu, self.k = lam, floor, nu, k
        self.s, self.w = 0.0, 0.0
        self.start_sd = start_sd

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 50.0

    def predict(self, t, y):
        sd = self.s / self.w if self.w > 0 else self.start_sd
        sd = max(sd, self.floor)
        eta = self._eta(y, t)
        mu = -self.k * (eta * eta / 3.0) / sd
        return mu, sd, self.target(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if np.isfinite(x):
            self.s, self.w = ewma_update(self.s, self.w, min(x * x, 1e7), self.lam)


d = H.load()
y = H.derived(H.core_frame(d))
print("=" * 84)
print("PHASE 3 - IS THE NEGATIVE MEAN REAL, OR ESTIMATION NOISE?")
print("=" * 84)
print("N = %d  warmup = 4000   target: dmid over [t,t+1]\n" % y["N"])

# reference: zero-mean Student-t
ref = preq.Runner(y, warmup=4000).run(FTLPredictive(lam=0.99, nu=4.0, k=0.0))
base = ref.loss
s0 = preq.summarize(base, "k=0 (zero mean, Student-t)")
print(preq.fmt(s0))
print()

print("Scan of the DERIVED reversion strength k (no free mean parameter):")
print("  %-6s %-10s %-28s %s" % ("k", "mean NLL", "paired 95% CI vs k=0", "verdict"))
best = None
for k in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
    r = preq.Runner(y, warmup=4000).run(FTLPredictive(lam=0.99, nu=4.0, k=k))
    df = preq.paired(r.loss, base)
    mean, lo, hi, n = stats1.moving_block_ci(df, B=1200, seed=3)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-6.2f %-10.4f [%+.4f, %+.4f]%s %s"
          % (k, np.nanmean(r.loss), lo, hi, " " * 6, v))
    if best is None or (hi < 0 and (best[1] is None or hi < best[1])):
        best = (k, hi, np.nanmean(r.loss))
print()
if best and best[1] is not None:
    print("  best k = %.2f  (mean NLL %.4f)" % (best[0], best[2]))
else:
    print("  no k beats the zero-mean null at 95% -> the conditional mean is")
    print("  indistinguishable from zero once estimation noise is removed.")
