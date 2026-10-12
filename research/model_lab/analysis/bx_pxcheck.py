"""Sanity-check the extracted price levels before trusting any backtest."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import bx_data as D


def main():
    for ex, sym in [("nobitex", "USDT_IRT"), ("nobitex", "BTC_IRT"),
                    ("bitwana", "USDT_IRT")]:
        d = D.load_series(ex, sym)
        mid = d["mid"]
        print("  %-22s n=%6d  mid[0]=%.4f  mid[-1]=%.6f  spread: p1=%.6f p50=%.6f p99=%.6f"
              % (ex + "/" + sym, mid.size, mid[0], mid[-1],
                 np.percentile(d["spread"], 1), np.percentile(d["spread"], 50),
                 np.percentile(d["spread"], 99)))
        step = np.median(np.diff(d["t_end_ms"]))
        print("      step=%dms  unique mids=%d  min_mid=%.4f  dmid uniq(1..6)=%s"
              % (step, np.unique(mid).size, mid.min(),
                 sorted(set(np.round(np.diff(mid), 6).tolist()))[:6]))
        print("      tick size: bid/ask grid =", np.unique(np.round(d["best_bid"], 6))[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
