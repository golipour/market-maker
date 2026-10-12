"""Mixture price models: production incumbent vs lattice-integrated challengers.

All models share the production state recursion (EM gate + weighted RLS +
EWMA variances) so that the ONLY thing being compared is the likelihood:
a continuous point density versus a lattice-integrated pmf.
"""
from __future__ import annotations

import math

import numpy as np

from research.model_lab.analysis.bx_price_models import (
    expit,
    lattice_logpmf,
    cont_logpdf,
    logaddexp,
)


class MixtureFamily:
    """Shared production recursion, parameterised by the scoring likelihood."""

    lattice = False

    def __init__(self, *, decay=0.99, init_sigma=50.0, student_df=3.0,
                 min_refresh_weight=0.15, rls_p0=1e6, sigma_min=1e-6):
        self.gamma = max(1e-6, min(0.9999, float(decay)))
        self.alpha = 1.0 - self.gamma
        self.nu = max(2.1, float(student_df))
        self.min_rw = max(0.0, min(0.999, float(min_refresh_weight)))
        self.sigma_min = float(sigma_min)
        self.gate_beta = np.zeros(2)
        self.mkt_beta = np.zeros(2)
        self.mkt_P = np.eye(2) * float(rls_p0)
        self.ref_mean = 0.0
        self.mkt_var = float(init_sigma) ** 2
        self.ref_var = float(init_sigma) ** 2
        self.h = 1.0

    def bind(self, y):
        self.h = float(y["tick"])

    # -- state -------------------------------------------------------------
    def q_market(self, total_flow: float) -> float:
        lf = min(math.log1p(max(0.0, float(total_flow))), 12.0)
        if not math.isfinite(lf):
            lf = 0.0
        x = np.array([1.0, lf])
        score = float(np.nan_to_num(self.gate_beta @ x))
        cap = max(0.001, 1.0 - self.min_rw)
        return min(cap, max(0.001, expit(score)))

    def predict(self, y, t):
        flow = y["tot_n1"][t]
        mech = y["dmid_n1"][t]
        q = self.q_market(0.0 if not np.isfinite(flow) else flow)
        b0, b1 = self.mkt_beta
        mu = b0 + b1 * (0.0 if not np.isfinite(mech) else mech)
        return {"q": q, "mu_mkt": mu, "sig_mkt": math.sqrt(self.mkt_var),
                "mu_ref": self.ref_mean, "sig_ref": math.sqrt(self.ref_var),
                "nu": self.nu, "h": self.h}

    def _branch_lls(self, tgt, p):
        """Return (market_ll, refresh_ll) with mixture weights applied."""
        em = tgt - p["mu_mkt"]
        er = tgt - p["mu_ref"]
        if self.lattice:
            a = math.log(max(p["q"], 1e-12)) + lattice_logpmf(
                tgt, p["mu_mkt"], p["sig_mkt"], None, p["h"])
            b = math.log(max(1.0 - p["q"], 1e-12)) + lattice_logpmf(
                tgt, p["mu_ref"], p["sig_ref"], p["nu"], p["h"])
        else:
            a = math.log(max(p["q"], 1e-12)) + cont_logpdf(tgt, p["mu_mkt"], p["sig_mkt"], None)
            b = math.log(max(1.0 - p["q"], 1e-12)) + cont_logpdf(
                tgt, p["mu_ref"], p["sig_ref"], p["nu"])
        return a, b

    def logpmf(self, tgt, p):
        a, b = self._branch_lls(tgt, p)
        return logaddexp(a, b)

    # -- adaptation --------------------------------------------------------
    def _weighted_rls(self, beta, P, x, target, w):
        w = max(0.0, min(1.0, float(w)))
        if w <= 1e-6:
            return beta, P
        sc = math.sqrt(w)
        xw = x * sc
        yw = target * sc
        px = P @ xw
        denom = self.gamma + float(xw @ px)
        gain = px / denom
        beta = beta + gain * (yw - float(xw @ beta))
        P = (P - np.outer(gain, xw) @ P) / self.gamma
        P = 0.5 * (P + P.T)
        return beta, P

    def update(self, y, t):
        tgt = y["dmid_n1"][t]
        if not np.isfinite(tgt):
            return
        flow = y["tot_n1"][t]
        mech = y["dmid_n1"][t]
        if not np.isfinite(mech):
            mech = 0.0
        lf = min(math.log1p(max(0.0, float(flow) if np.isfinite(flow) else 0.0)), 12.0)
        gx = np.array([1.0, lf])
        p = self.predict(y, t)
        a, b = self._branch_lls(tgt, p)
        tot = logaddexp(a, b)
        mkt_resp = math.exp(min(a - tot, 0.0))
        q = p["q"]
        self.gate_beta = np.clip(self.gate_beta + self.alpha * (mkt_resp - q) * gx, -20.0, 20.0)
        rr = 1.0 - mkt_resp
        x = np.array([1.0, mech])
        em = tgt - p["mu_mkt"]
        er = tgt - p["mu_ref"]
        self.mkt_beta, self.mkt_P = self._weighted_rls(
            self.mkt_beta, self.mkt_P, x, tgt, mkt_resp)
        self.ref_mean = self.ref_mean + self.alpha * rr * (tgt - self.ref_mean)
        self.mkt_var = max((1.0 - self.alpha * mkt_resp) * self.mkt_var
                           + self.alpha * mkt_resp * em * em, self.sigma_min ** 2)
        self.ref_var = max((1.0 - self.alpha * rr) * self.ref_var
                           + self.alpha * rr * er * er, self.sigma_min ** 2)


class IncumbentMixture(MixtureFamily):
    """PRODUCTION: continuous mixture density on a discrete lattice."""

    lattice = False
    name = "INCUMBENT  q*N + (1-q)*t  (continuous)"


class LatticeMixture(MixtureFamily):
    """C1: identical recursion, lattice-integrated pmf."""

    lattice = True
    name = "CHALLENGE  q*N + (1-q)*t  (lattice)"
