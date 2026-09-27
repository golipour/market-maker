"""Phase 2: the predictability ladder."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from research.model_lab.analysis import harness as H
from research.model_lab.analysis import preq, stats1
from research.model_lab.analysis import models_mid as M
from research.model_lab.analysis import models_mid2 as M2
from research.model_lab.analysis import models_struct as S

d = H.load()
y = H.derived(H.core_frame(d))
print("=" * 88)
print("PHASE 2 - PREDICTABILITY LADDER (prequential, paired, block-bootstrap)")
print("=" * 88)
print("N = %d  warmup = 4000   target: dmid over [t,t+1], IRT ticks" % y["N"])
print("CI: moving-block bootstrap, 1500 resamples, block ~ n^(1/3)\n")

ladder = [
    ("L1  EWMA Gaussian sd, zero mean", lambda: M.EWMAStatic(lam=0.99)),
    ("L2  EWMA Student-t, zero mean", lambda: M.StudentEWMA(lam=0.99, nu=4.0)),
    ("L3  Laplace (mean|x|) EWMA", lambda: M2.ExponentialTail(lam=0.99)),
    ("L4  Student-t, sd=c*halfspread", lambda: M2.StudentSpread(lam=0.99, nu=4.0)),
    ("L5  Student-t + vol feedback", lambda: M2.ETDStudent(lam=0.99, nu=4.0)),
    ("L6  MA(1) mean + EWMA sd", lambda: M.MA1EWMA(lam=0.995)),
    ("L7  microprice-anchored mean", lambda: S.MicropriceAnchor(lam=0.995)),
    ("L8  STRUCTURAL quote-mixture", lambda: S.QuoteMixture(lam=0.99, nu=4.0)),
]

losses = {}
for name, ctor in ladder:
    r = preq.Runner(y, warmup=4000).run(ctor())
    losses[name] = r.loss
    print(preq.fmt(preq.summarize(r.loss, name)))

base_name = "L1  EWMA Gaussian sd, zero mean"
base = losses[base_name]
print("\n" + "-" * 88)
print("Paired loss difference vs L1 (negative = better); 95% block-bootstrap CI")
print("-" * 88)
for name in losses:
    if name == base_name:
        continue
    df = preq.paired(losses[name], base)
    mean, lo, hi, n = stats1.moving_block_ci(df, B=1500, seed=1)
    t, p = stats1.dm_test(df)
    sig = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "not resolved")
    print("  %-36s d=%+.4f CI[%+.4f,%+.4f] p=%.1e  %s"
          % (name, mean, lo, hi, p, sig))

print("\n" + "-" * 88)
print("Fitted coefficients after the full run")
print("-" * 88)
m6 = M.MA1EWMA(lam=0.995)
preq.Runner(y, warmup=4000).run(m6)
print("  MA(1):      dmid_t = %+.3f %+.4f*dmid_(t-1)" % (m6.b[0], m6.b[1]))
m7 = S.MicropriceAnchor(lam=0.995)
preq.Runner(y, warmup=4000).run(m7)
print("  microprice: dmid_t = %+.3f %+.4f*(micro-mid)/(half-spread)"
      % (m7.b[0], m7.b[1]))
print("  (classical order-imbalance theory predicts the slope ~ -0.5..-1.0)")
