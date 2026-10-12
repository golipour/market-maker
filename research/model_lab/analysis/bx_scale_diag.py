"""Why does the hurdle-t scale blow up on Nobitex BTC_IRT?

The scale is EWMA-matched to the SECOND MOMENT of the non-zero support.  One
100k-tick print contributes 1e10 to a sum whose other terms are O(1e2).  This
script quantifies the tail dominance and tests robust alternatives.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import t as _tdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import bx_data as D  # noqa: E402

DATASETS = [("nobitex", "USDT_IRT"), ("nobitex", "BTC_IRT"),
            ("bitwana", "USDT_IRT")]


def banner(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100, flush=True)


def load(ex, sym):
    return D.derived(D.build(D.load_series(ex, sym)))


def e1():
    banner("E1  TAIL DOMINANCE of the non-zero move distribution")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        d = y["dmid_n1"]
        d = d[np.isfinite(d)]
        nz = d[d != 0.0]
        a = np.abs(nz)
        m2 = float(np.mean(nz ** 2))
        i = int(np.argmax(nz ** 2))
        m2_ex = float(np.mean(np.delete(nz ** 2, i)))
        print("  %s" % (ex + "/" + sym))
        print("    n=%d  nonzero=%d  median|d|=%.1f  p90=%.1f  p99=%.1f  max=%.1f"
              % (d.size, nz.size, np.median(a), np.percentile(a, 90),
                 np.percentile(a, 99), a.max()))
        print("    E[x^2]=%.4g  excl. largest=%.4g  ratio=%.1fx  rms=%.1f"
              % (m2, m2_ex, m2 / max(m2_ex, 1e-12), np.sqrt(m2)), flush=True)


def e2():
    banner("E2  ROBUST SCALE ESTIMATORS on the non-zero support")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        d = y["dmid_n1"]
        d = d[np.isfinite(d)]
        nz = d[d != 0.0]
        a = np.abs(nz)
        cap = float(np.percentile(a, 99.9))
        m2w = float(np.mean(np.minimum(nz, cap) ** 2))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        print("  %-22s rms=%10.1f  rms(wins99.9)=%9.1f  IQR=%8.1f  p75=%8.1f  med=%7.1f"
              % (ex + "/" + sym, np.sqrt(np.mean(nz ** 2)), np.sqrt(m2w), iqr,
                 float(np.percentile(a, 75)), float(np.median(a))), flush=True)
    print("\n  For nu=2.5 the t-scale from m2 is s = sqrt(m2 * 0.2), so a single")
    print("  outlier moves s by orders of magnitude.  Winsorising or quantile")
    print("  matching bounds it.")


def e3():
    banner("E3  TAIL INDEX: which nu reproduces the empirical |dmid| quantiles?")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        d = y["dmid_n1"]
        d = d[np.isfinite(d)]
        nz = d[d != 0.0]
        a = np.abs(nz)
        qs = [75, 90, 95, 99]
        emp = [float(np.percentile(a, p)) for p in qs]
        print("  %s" % (ex + "/" + sym))
        print("    empirical: %s" % "  ".join("p%d=%.1f" % (p, v)
                                              for p, v in zip(qs, emp)))
        best = None
        for nu in (2.5, 3.0, 4.0, 6.0, 10.0, 20.0):
            q75 = float(_tdist.ppf(0.875, df=nu))
            s = emp[0] / q75
            pred = [float(_tdist.ppf(0.5 + p / 200.0, df=nu)) * s for p in qs]
            mape = float(np.mean([abs(pred[i] - emp[i]) / emp[i]
                                  for i in range(len(qs))]))
            if best is None or mape < best[1]:
                best = (nu, mape)
            row = "  ".join("p%d=%.1f" % (p, v) for p, v in zip(qs, pred))
            print("      nu=%5.1f  s=%11.2f  pred %s   MAPE=%.3f"
                  % (nu, s, row, mape), flush=True)
        print("    BEST nu=%.1f (MAPE=%.3f)" % best, flush=True)


def main():
    e1()
    e2()
    e3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
