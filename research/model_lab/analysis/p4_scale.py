"""Phase 4: the conditional scale. Is there ANY predictable scale signal?

Candidates tested, each causal and each with an explicit economic meaning:
  sd_t = ewma(|dmid|^2)                 pure volatility clustering
  sd_t = c * half-spread                liquidity-proportional
  sd_t = ewma + b*|dmid_{t-1}|          Hasbrouck volatility feedback
  sd_t = c * halfspread * f(staleness)  stale book -> wider uncertainty
  sd_t = g(recent activity)             flow-driven regime

The decision statistic is the paired block-bootstrap CI of the loss
difference, and the calibration table checks the answer is not merely a
scale artefact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import preq, stats1
from research.model_lab.analysis.models_mid import ewma_update
from research.model_lab.analysis.preq import BaseModel


class ScaleModel(BaseModel):
    """Student-t innovation, zero mean, plug-in conditional scale.

    `mode` selects the mechanism.  `beta` is a recursive (EWMA) coefficient, so
    even when beta is free the estimator is online and its variance is visible
    in the prequential score rather than hidden in a fit.
    """

    name = "scale: plugin"

    def __init__(self, mode="ewma", lam=0.99, floor=1.0, nu=4.0, beta0=0.0,
                 start_sd=50.0, gclip=(0.2, 5.0)):
        self.mode, self.lam, self.floor, self.nu = mode, lam, floor, nu
        self.beta0, self.start_sd, self.gclip = beta0, start_sd, gclip
        self.s, self.w = 0.0, 0.0
        self.bs, self.bw = 0.0, 0.0

    def _eta(self, y, t):
        e = 0.5 * y["spread"][t]
        return e if np.isfinite(e) and e > 0 else 50.0

    def _beta(self):
        return self.bs / self.bw if self.bw > 0 else self.beta0

    def predict(self, t, y):
        base_sd = self.s / self.w if self.w > 0 else self.start_sd
        base_sd = max(base_sd, self.floor)
        eta = self._eta(y, t)
        b = self._beta()
        if self.mode == "ewma":
            g = 1.0
        elif self.mode == "spread":
            g = eta / 50.0
        elif self.mode == "etd":
            prev = y["dmid"][t] if t < self.N - 1 else np.nan
            g = 1.0 + b * (abs(prev) / max(base_sd, 1e-9) if np.isfinite(prev) else 0.0)
        elif self.mode == "stale":
            st = y["stale_ms"][t]
            g = 1.0 if not np.isfinite(st) else min(max(1.0 + b * (st - 2000.0) / 2000.0,
                                                         self.gclip[0]), self.gclip[1])
        elif self.mode == "flow":
            oc = y["ocur"][t]
            g = 1.0 + b * (oc - 0.55)
        else:
            raise ValueError(self.mode)
        sd = float(np.clip(base_sd * g, self.floor, 1e4))
        return 0.0, sd, self.target(y, t)

    @property
    def N(self):
        return self._N

    def update(self, t, y):
        self._N = len(y["dmid"])
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        self.s, self.w = ewma_update(self.s, self.w, min(x * x, 1e7), self.lam)
        if self.mode in ("etd", "stale", "flow"):
            # EWMA-regress squared normalised move on the driver
            eta = self._eta(y, t)
            base = max(self.s / self.w if self.w > 0 else self.start_sd ** 2, 1e-6)
            if self.mode == "etd":
                prev = y["dmid"][t] if t < self._N - 1 else np.nan
                drv = abs(prev) if np.isfinite(prev) else 0.0
            elif self.mode == "stale":
                st = y["stale_ms"][t]
                drv = (st - 2000.0) / 2000.0 if np.isfinite(st) else 0.0
            else:
                drv = y["ocur"][t] - 0.55
            u = x * x / base
            self.bs, self.bw = ewma_update(self.bs, self.bw, np.clip(u * drv, -50, 50), self.lam)


d = H.load()
y = H.derived(H.core_frame(d))
print("=" * 88)
print("PHASE 4 - THE CONDITIONAL SCALE (Student-t nu=4, zero mean, online beta)")
print("=" * 88)
print("N = %d  warmup = 4000   target: dmid over [t,t+1]\n" % y["N"])

configs = [
    ("sd = EWMA (no driver)", dict(mode="ewma")),
    ("sd = c * half-spread", dict(mode="spread")),
    ("sd ~ |dmid_prev| (vol feedback)", dict(mode="etd")),
    ("sd ~ book staleness", dict(mode="stale")),
    ("sd ~ trade occurrence", dict(mode="flow")),
]

losses = {}
for name, kw in configs:
    r = preq.Runner(y, warmup=4000).run(ScaleModel(lam=0.99, nu=4.0, **kw))
    losses[name] = r.loss
    print(preq.fmt(preq.summarize(r.loss, name)))

base_name = "sd = EWMA (no driver)"
base = losses[base_name]
print("\n" + "-" * 88)
print("Paired loss difference vs the pure-EWMA scale (95% block-bootstrap CI)")
print("-" * 88)
for name in losses:
    if name == base_name:
        continue
    df = preq.paired(losses[name], base)
    mean, lo, hi, n = stats1.moving_block_ci(df, B=1500, seed=5)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-34s d=%+.4f CI[%+.4f,%+.4f]  %s" % (name, mean, lo, hi, v))

m = ScaleModel(mode="etd", lam=0.99, nu=4.0)
preq.Runner(y, warmup=4000).run(m)
print("\n  fitted volatility-feedback beta (EWMA) = %+.4f" % (m.bs / m.bw if m.bw else 0))
m2 = ScaleModel(mode="flow", lam=0.99, nu=4.0)
preq.Runner(y, warmup=4000).run(m2)
print("  fitted flow-occurrence beta (EWMA)     = %+.4f" % (m2.bs / m2.bw if m2.bw else 0))
m3 = ScaleModel(mode="stale", lam=0.99, nu=4.0)
preq.Runner(y, warmup=4000).run(m3)
print("  fitted staleness beta (EWMA)           = %+.4f" % (m3.bs / m3.bw if m3.bw else 0))
