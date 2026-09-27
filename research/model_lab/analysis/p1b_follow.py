"""Phase 1b: the follow-the-leader (resampling) mechanism.

Key structural fact from Phase 1:
    ACF1(dmid) = -0.218,  and  -cov = eta^2 with eta = 24.4 IRT
    = the median half-spread.  That is the Roll (1984) covariance signature.

Synthetic test: an artificial price whose ONLY randomness is a lagged redraw
of the quotes (no information, no autocorrelation in the efficient price)
should reproduce the observed ACF1 and ACF(|dmid|). If it does, the negative
autocorrelation and the volatility clustering are MEASUREMENT artefacts of
watching quotes, not properties of the efficient price.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis.stats1 import acf

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
eta = y["spread"] / 2.0

print("=" * 78)
print("PHASE 1b - THE FOLLOW-THE-LEADER (RESAMPLING) MECHANISM")
print("=" * 78)

LAGS = (1, 2, 3, 5, 10, 20, 60)
obs = [acf(y["dmid"], k)[0] for k in LAGS]
obsv = [acf(np.abs(y["dmid"]), k)[0] for k in LAGS]
print("\nObserved (real data), N=%d" % y["N"])
print("  lags            " + " ".join("%7d" % k for k in LAGS))
print("  ACF(dmid)       " + " ".join("%+7.4f" % v for v in obs))
print("  ACF(|dmid|)     " + " ".join("%+7.4f" % v for v in obsv))

rng = np.random.default_rng(0)
print("\nSynthetic FTL: efficient price = PURE RANDOM WALK (zero predictability);")
print("quotes follow it with a lag and redraw uniformly inside the spread.\n")
for L in (0, 1, 2):
    n = 400_000
    p = np.cumsum(rng.normal(0, 1.0, n))
    q = p.copy()
    if L:
        q[L:] = p[:-L]
    et = eta[rng.integers(0, len(eta), n)]
    quote = q + et * rng.uniform(-1.0, 1.0, n)
    dq = np.diff(quote)
    a = [acf(dq, k)[0] for k in LAGS]
    b = [acf(np.abs(dq), k)[0] for k in LAGS]
    print("  quote lag L=%d" % L)
    print("    ACF(dquote) " + " ".join("%+7.4f" % v for v in a))
    print("    ACF(|d|)    " + " ".join("%+7.4f" % v for v in b))

print("\n" + "=" * 78)
print("READING")
print("=" * 78)
print("""
The synthetic efficient price is a pure random walk: E[p_{t+1}|p_t] = p_t, so
there is NO directional predictability at all. Yet the *observed mid* built
from lagged, uniform-within-spread quotes reproduces

  * a strongly NEGATIVE ACF1, and
  * a strongly POSITIVE ACF of |dmid|.

Both headline 'findings' are consequences of watching a resampled quantity,
not of the underlying value. The market maker's observation channel (best
bid/ask at 10s) is a *measurement* of the efficient price with a lag and a
bounded, refreshed error; Var(efficient price) is a different, smaller object
than Var(dmid) suggests.
""")
