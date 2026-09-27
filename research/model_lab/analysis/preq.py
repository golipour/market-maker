"""Prequential (online) evaluation engine.

Contract for every candidate model:
  at instant t use ONLY information available at t;
  emit a predictive density for the move over [t, t+h];
  score it against the realised move;
  only then update.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

LOG2PI = float(np.log(2.0 * np.pi))


def log_normal(y, mu, sd):
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return -0.5 * (z * z) - np.log(sd) - 0.5 * LOG2PI


def log_student_t(y, mu, sd, nu):
    from scipy.special import gammaln
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    lc = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
          - 0.5 * (np.log(nu * np.pi) + np.log(sd)))
    return lc - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)


def log_discrete_normal(y, mu, sd):
    sd = np.maximum(sd, 1e-9)
    return np.log(np.maximum(
        stats.norm.cdf((y + 0.5 - mu) / sd)
        - stats.norm.cdf((y - 0.5 - mu) / sd), 1e-300))


def log_hurdle_normal(y, mu, sd, p0):
    p0 = np.clip(p0, 1e-6, 1.0 - 1e-6)
    out = np.log(np.where(y == 0, p0, 1.0 - p0))
    pos = y > 0
    if pos.any():
        sub = log_discrete_normal(np.maximum(y, 1), mu, sd) - np.log1p(-p0)
        out = np.where(pos, sub, out)
    return out


class BaseModel:
    """Subclasses implement predict/update. `nu=None` means Gaussian."""

    name = "base"
    h = 1
    nu = None

    def target(self, y, t, h=None):
        """The move realised over [t, t+h], indexed at the decision instant t.

        CORRECT ALIGNMENT: at decision instant t the newest observed move is
        dmid[t] = mid[t]-mid[t-1].  The forecast target is therefore
        dmid[t+h], NOT dmid[t-h].  Using the past move as the target is
        leakage, and using it as a feature is fine.
        """
        h = self.h if h is None else h
        return y["dmid"][t + h] if (t + h) < len(y["dmid"]) else np.nan

    def predict(self, t, y):
        raise NotImplementedError

    def update(self, t, y):
        raise NotImplementedError


class Runner:
    def __init__(self, y, warmup=3000, start=0, end=None):
        self.y, self.N = y, y["N"]
        self.warmup, self.start = warmup, start
        self.end = y["N"] if end is None else end
        self.loss = np.full(self.N, np.nan)
        self.mu = np.full(self.N, np.nan)
        self.sd = np.full(self.N, np.nan)
        self.p0 = np.full(self.N, np.nan)

    def run(self, model):
        y = self.y
        nu = getattr(model, "nu", None)
        for t in range(self.start, self.end):
            if t < self.warmup:
                model.update(t, y)
                continue
            out = model.predict(t, y)
            # Contract:
            #   (mu, sd, target)                 -> plain density with `nu`
            #   (mu, sd, p0, target)             -> hurdle, p0 = P(target == 0)
            # A model with a custom density returns a `logpdf` method and a
            # 3-tuple; `logpdf` then takes (target, mu, sd).
            if len(out) == 4:
                m, s, p0, tgt = out
            elif len(out) == 3 and hasattr(model, "logpdf"):
                m, s, tgt = out
                p0 = None
            else:
                m, s, tgt = out
                p0 = None
            if np.isfinite(tgt):
                self.mu[t], self.sd[t] = m, s
                self.p0[t] = p0 if p0 is not None else np.nan
                if p0 is not None:
                    if tgt == 0:
                        self.loss[t] = -float(np.log(max(p0, 1e-12)))
                    else:
                        llc = log_student_t(abs(tgt), m, s, nu if nu else 4.0)
                        self.loss[t] = -float(llc - np.log(max(1.0 - p0, 1e-12)))
                elif hasattr(model, "logpdf"):
                    self.loss[t] = -float(model.logpdf(tgt, m, s))
                elif nu is not None:
                    self.loss[t] = -log_student_t(tgt, m, s, nu)
                else:
                    self.loss[t] = -log_normal(tgt, m, s)
            model.update(t, y)
        return self


def summarize(loss, name=""):
    L = loss[np.isfinite(loss)]
    if len(L) == 0:
        return {"name": name, "n": 0}
    q = np.percentile(L, [50, 90, 99])
    return {"name": name, "n": int(len(L)), "mean": float(L.mean()),
            "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(L.max())}


def fmt(s, ref=None):
    if not s or s.get("n", 0) == 0:
        return "  %-40s (no scores)" % (s or {}).get("name", "?")
    d = "  d=%+.4f" % (s["mean"] - ref["mean"]) if ref and ref.get("n", 0) else ""
    return ("  %-40s n=%7d mean=%8.4f med=%8.4f p90=%8.3f p99=%8.3f max=%9.2f%s"
            % (s["name"], s["n"], s["mean"], s["median"], s["p90"], s["p99"],
               s["max"], d))


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m] - b[m]
