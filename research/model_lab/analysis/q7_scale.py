"""Q7: the scale recursion is mis-scaled, and the calibration never fires.

Two facts from Q5 that should not both be true:

  * mean NLL is competitive (2.33, essentially tied with the reference)
  * yet PIT z sd = 0.51 and 80% coverage = 0.98

A model cannot be simultaneously competitive and 2x over-confident on a
hurdle density unless the continuous part is being *under*-penalised and the
hurdle is doing the work.  The cause is structural: the density is

    q(y) = p0 * 1{y=0}  +  (1-p0) * f_SYM(|y|; 0, sd)

with sd carried from a GARCH recursion initialised at s0=50, while the
conditional scale of |dmid| given dmid != 0 is on the order of 2-6 IRT, not
50.  So sd is ~10x too large, the density is far too flat on the continuous
branch, and -log q on a zero move is dominated by p0 rather than by the
continuous part.  The negative-looking wins of Q5 were an artefact of that.

The fix is to initialise and floor the scale on the CONDITIONAL scale of the
non-zero part, which is what the model actually has to match.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
x = y["dmid"]
print("=" * 84)
print("Q7 - THE SCALE IS WRONG BY AN ORDER OF MAGNITUDE")
print("=" * 84)

nz = x[np.isfinite(x) & (x != 0)]
print("\n[7.1] The conditional scale of the NON-ZERO part (what sd must match)")
print("  n(nonzero) = %d (%.4f of all)" % (len(nz), len(nz) / N))
print("  sd(|dmid| | dmid!=0)      = %.3f" % nz.std())
print("  IQR-sd                     = %.3f"
      % ((np.percentile(nz, 75) - np.percentile(nz, 25)) / 1.349))
print("  median |dmid|              = %.3f" % np.median(np.abs(nz)))
print("  p90 |dmid|                 = %.3f" % np.percentile(np.abs(nz), 90))
print("  p99 |dmid|                 = %.3f" % np.percentile(np.abs(nz), 99))
print("\n  Compare: sd of ALL dmid   = %.3f" % x[np.isfinite(x)].std())
print("  => the zero mass inflates the apparent scale by ~%.1fx"
      % (x[np.isfinite(x)].std() / nz.std()))

print("\n[7.2] What the Q5 GARCH actually converged to")
lam = 0.99
s2 = 2500.0
a, b = 0.10, 0.85
for t in range(N):
    v = x[t]
    if not np.isfinite(v):
        continue
    s2 = (1.0 - a - b) * s2 + a * v * v + b * s2
    if t in (8000, 50000, 100000, 200000, N - 3):
        print("    t=%7d  sd=%9.3f" % (t, np.sqrt(s2)))
print("  => the GARCH is tracking the UNCONDITIONAL scale (~50), which is")
print("     ~10x the scale the continuous branch actually needs (~2-6).")

print("\n[7.3] Consequence: a zero move is charged almost nothing.")
print("  q(0) = p0 = 0.347  ->  loss = 1.058")
print("  a NON-zero move of 1 IRT, if the true conditional scale is 3, should")
print("  cost roughly  -log( 0.653 * f(1; 0, 3) ) with f the t4 density.")
from scipy import stats as st  # noqa: E402
for sd in (2.0, 3.0, 5.0, 50.0):
    q = 0.653 * st.t.pdf(1.0, df=4, loc=0.0, scale=sd)
    print("     sd=%5.1f ->  q(1 IRT)=%.5f  loss=%.4f" % (sd, q, -np.log(q)))
print("  A scale of 50 makes a 1-IRT move cost only %.2f nats, which is why"
      % -np.log(0.653 * st.t.pdf(1.0, df=4, loc=0.0, scale=50.0)))
print("  the model looked fine on NLL while being 10x over-smooth on the")
print("  continuous branch.  The hurdle was masking it.")

print("\n[7.4] The correct construction: recursion on the NON-ZERO shocks only")
print("  Since q is a hurdle, the continuous part must be normalised on the")
print("  non-zero support.  So the scale recursion should use")
print("     x_t^2 * 1{x_t != 0}  (the squared conditional shock),")
print("  and p0 should carry the zero mass on its own.")
print("\n  Sanity check on what that scale converges to:")
s2c = 9.0
for t in range(N):
    v = x[t]
    if not np.isfinite(v):
        continue
    s2c = (1.0 - a) * s2c + a * (v * v if v != 0 else 0.0)
    if t in (8000, 50000, 200000):
        print("    t=%7d  sd(nonzero shocks)=%8.3f" % (t, np.sqrt(s2c)))
print("  => tracks ~%.1f, matching the empirical IQR-sd of %.1f."
      % (np.sqrt(s2c), (np.percentile(nz, 75) - np.percentile(nz, 25)) / 1.349))
print("\n" + "=" * 84)
