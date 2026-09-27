"""Phase 6: diagnose the scale law properly.

Two problems with the naive fit in Phase 5:
  (1) the functional forms are only identified up to the free coefficient c,
      so comparing NLL across forms conflates shape with scale;
  (2) the PIT shows a large mass deficit at the median, i.e. the predictive
      scale is too wide in a specific sub-population.

This phase measures the conditional scale directly and non-parametrically:
  sd(y | eta) is estimated in eta-bins with no distributional assumption, and
  compared against the candidate shapes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import stats1

d = H.load()
y = H.derived(H.core_frame(d))
x = y["dmid"][1:]
eta = 0.5 * y["spread"][:-1]
act = y["ocur"][:-1]
stale = y["stale_ms"][:-1]

print("=" * 88)
print("PHASE 6 - NON-PARAMETRIC CONDITIONAL SCALE")
print("=" * 88)
print("N obs = %d\n" % len(x))

qs = np.quantile(eta, [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])
bins = np.concatenate([[-1.0], qs, [1e9]])
print("Robust (IQR-matched) scale of dmid conditional on the half-spread eta:")
print("  %-16s %-8s %-10s %-10s %-8s %-8s"
      % ("eta bin", "n", "eta_med", "IQR sd", "sd/eta", "sd/eta^0.5"))
rows = []
for i in range(len(bins) - 1):
    m = (eta >= bins[i]) & (eta < bins[i + 1]) & np.isfinite(x)
    if m.sum() < 200:
        continue
    xx = x[m]
    iqr = np.subtract(*np.percentile(xx, [75, 25]))
    sd = 1.349 * iqr
    em = np.median(eta[m])
    rows.append((em, sd, m.sum()))
    print("  [%7.1f,%8.1f) %-8d %-10.1f %-10.2f %-8.3f %-8.3f"
          % (bins[i], bins[i + 1], m.sum(), em, sd, sd / em, sd / np.sqrt(em)))

print("\n  log-log slope of sd on eta across bins:")
em = np.array([r[0] for r in rows]); sm = np.array([r[1] for r in rows])
w = np.array([r[2] for r in rows], dtype=float)
A = np.column_stack([np.log(em), np.ones(len(em))])
coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], np.log(sm) * np.sqrt(w), rcond=None)
print("    fitted slope = %.3f   intercept = %.3f" % (coef[0], coef[1]))
print("    (slope 1 => sd proportional to eta; slope 0.5 => sd ~ sqrt(eta))")

print("\n" + "-" * 88)
print("Is the scale also driven by TRADE OCCURRENCE within an eta bin?")
print("-" * 88)
med_eta = (np.abs(eta - np.median(eta)) < np.median(eta) * 0.5) & np.isfinite(x)
print("  %-10s %-10s %-10s %-10s" % ("group", "n", "IQR sd", "mean|d|"))
for lab, msk in [("no trades", med_eta & (act == 0)), ("trades", med_eta & (act > 0))]:
    xx = x[msk]
    if len(xx) < 100:
        continue
    iqr = np.subtract(*np.percentile(xx, [75, 25]))
    print("  %-10s %-10d %-10.2f %-10.2f" % (lab, len(xx), 1.349 * iqr, np.mean(np.abs(xx))))

print("\n" + "-" * 88)
print("Is the scale driven by STALENESS within an eta bin?")
print("-" * 88)
qs2 = np.quantile(stale[med_eta], [0.33, 0.66])
for i, (lab, msk) in enumerate([
        ("stale low", med_eta & (stale <= qs2[0])),
        ("stale mid", med_eta & (stale > qs2[0]) & (stale <= qs2[1])),
        ("stale high", med_eta & (stale > qs2[1]))]):
    xx = x[msk]
    if len(xx) < 100:
        continue
    iqr = np.subtract(*np.percentile(xx, [75, 25]))
    print("  %-10s n=%-8d med_stale=%-7.0f IQR sd=%-8.2f"
          % (lab, len(xx), np.median(stale[msk]), 1.349 * iqr))

print("\n" + "-" * 88)
print("Where is the excess zero mass? P(dmid=0) by eta bin")
print("-" * 88)
for i in range(len(bins) - 1):
    m = (eta >= bins[i]) & (eta < bins[i + 1]) & np.isfinite(x)
    if m.sum() < 200:
        continue
    print("  eta in [%7.1f,%8.1f): P(dmid=0) = %.4f   n=%d"
          % (bins[i], bins[i + 1], np.mean(x[m] == 0), m.sum()))
