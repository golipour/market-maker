"""Q6: why does the GARCH-hurdle report an IMPOSSIBLE mean loss?

A properly normalised predictive density has q <= 1 everywhere, so -log q >= 0
and the mean loss cannot be negative.  Q5 reported mean = -0.55 with median
+3.03, which is arithmetically impossible for a valid density.  Either

  (a) the density is being evaluated outside its domain, or
  (b) the "loss" recorded is not -log q of a normalised q, or
  (c) a small set of steps produce astronomically large q.

This script isolates which, by recording the argmin of the loss and inspecting
the state there.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import models_final as MF  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 80)
print("Q6 - IMPOSSIBLE MEAN LOSS: DIAGNOSIS")
print("=" * 80)

m = MF.GarchHurdle(a=0.10, b=0.85, nu=4.0)
loss = np.full(N, np.nan)
rec = []
for t in range(N - 2):
    if t < WARM:
        m.update(t, y)
        continue
    mu, sd, p0, tgt = m.predict(t, y)
    if np.isfinite(tgt):
        ll = m.logpdf(tgt, mu, sd, p0)
        loss[t] = -ll
        rec.append((t, tgt, mu, sd, p0, ll))
    m.update(t, y)
rec = np.array(rec)
L = loss[np.isfinite(loss)]
print("\n[6.1] Loss distribution")
print("  mean=%.4f  median=%.4f  min=%.4f  max=%.4f"
      % (L.mean(), np.median(L), L.min(), L.max()))
print("  n negative (loss<0): %d  (%.4f%%)"
      % (int((L < 0).sum()), 100.0 * (L < 0).mean()))
print("  the mean is dragged negative by a few astronomically large q.")

k = int(np.argmin(loss[np.isfinite(loss)]))
idx = np.flatnonzero(np.isfinite(loss))[k]
print("\n[6.2] The single most negative loss is at t=%d" % idx)
print("  loss=%.4f  ->  q = %.4g" % (loss[idx], np.exp(-loss[idx])))
mu, sd, p0 = m.predict(idx, y)
print("  mu=%.4f sd=%.4f p0=%.4f" % (mu, sd, p0))
print("  logpdf=%.4f" % m.logpdf(0.0, mu, sd, p0))
print("  _lt(0,0,sd,4) = %.4f  <-- the continuous branch at y=0"
      % MF._lt(0.0, 0.0, sd, m.nu))

print("\n[6.3] Is the loss negative ONLY on zero moves?")
z0 = (y["dmid"][1:] == 0)
fin = np.isfinite(loss[1:])
lz = loss[1:][fin]
zz = z0[fin]
print("  P(loss<0 | dmid==0)    = %.4f" % np.mean(lz[zz] < 0))
print("  P(loss<0 | dmid!=0)   = %.4f" % np.mean(lz[~zz] < 0))

print("\n[6.4] THE BUG: the hurdle branch and the continuous branch disagree.")
print("  logpdf(tgt=0) returns log(p0).")
print("  But the RUNNER also has a p0 slot; check which branch is taken.")
print("  p0 as predicted: %.4f   empirical P(dmid==0) = %.4f"
      % (p0, np.mean(y['dmid'][1:] == 0)))
print("\n  If p0 is being reported as ~0.14 while the true rate is 0.35,")
print("  the hurdle is UNDER-estimated, so zero moves are charged only")
print("  -log(0.14)=1.96 instead of -log(0.35)=1.05.  That is a LOSS of")
print("  accuracy, not a gain -- so it cannot explain a NEGATIVE mean.")
print("\n  The negative mean must come from the continuous branch at small sd.")
print("  min sd seen: %.6f" % np.nanmin(np.abs(rec[:, 3])))

print("\n[6.5] Direct check of _lt for small sd")
for sd_ in (1.0, 0.1, 0.01, 1e-3):
    print("    _lt(0, 0, %.4f, 4) = %+.4f" % (sd_, MF._lt(0.0, 0.0, sd_, 4.0)))
print("\n  A Student-t density at y=0 with tiny sd is astronomically large,")
print("  so -log q is hugely negative.  That is CORRECT behaviour for a")
print("  density, but it means the model's scale is collapsing at some steps.")
print("  The fix is a floor on sd AND a calibration that actually fires.")
print("\n  floor in code = %.3f" % m.floor)
sdser = rec[:, 3]
print("  sd percentiles: p1=%.3f p5=%.3f p50=%.3f"
      % tuple(np.percentile(sdser, [1, 5, 50])))
print("  n steps with sd < 1.0: %d" % int((sdser < 1.0).sum()))
print("  n steps with sd < 5.0: %d" % int((sdser < 5.0).sum()))
print("\n" + "=" * 80)
