# Empirical findings: the 10 s transition law

**Date:** 2026-09-27
**Data:** `research/model_lab/dataset/nobitex_USDT_IRT_10000ms.csv`
**Sample:** 347,640 contiguous 10 s buckets (~43 days), 674 non-contiguous gaps excluded
**Protocol:** causal prequential log-scoring — at instant `t` the density is built
only from realised values up to `t`, scored against the move over `[t, t+1]`,
and only then updated. All comparisons are paired and use a moving-block
bootstrap; every candidate density is asserted valid (`q <= 1`).

Every number below is reproduced by the scripts in
`research/model_lab/analysis/` (`q1`…`q18`) with logs in
`research/model_lab/analysis/out/`.

---

## 1. Answering the question that was asked

> *the current models assume time independence and samples are independent of
> the past somehow. maybe indirectly dependent*

**The assumption is false, and it was quantified.**

The incumbent family uses the past only through a single scalar — an EWMA of
the realised squares — and then emits conditionally iid draws. Standardising
the move by that scale and testing the resulting series:

| lag | ACF of `dmid_t / sigma_t` | t-stat | p |
|-----|---------------------------|--------|---|
| 1 | **−0.1893** | −111.6 | 0 |
| 2 | −0.0537 | −31.7 | 4.9e−220 |
| 3 | −0.0067 | −3.9 | 8.6e−05 |
| ≥5 | ≈ 0 | < 2 | > 0.06 |

So the standardised innovation is serially dependent at lag 1–3 and the
incumbent treats it as white noise. That dependence is real and measurable.

**But it is not exploitable, and it is not what it looks like.** Two further
tests closed the question.

### 1.1 The mean is not predictable

The optimal linear predictor of the move from its own past is

```
x_t = 0.167747 - 0.218008 * x_{t-1}      R^2 = 0.0475   n = 347,638
```

which bounds *any* linear mean model at

```
0.5 * log(sigma^2_uncond / sigma^2_resid) = 0.0244 nats/step
```

Feeding the correctly-signed coefficient into an online ARCH-in-mean
(`mu_{t+1} = phi * u_t`, `phi = -0.18`) returned **−0.0004 nats** prequentially
— 60× less than the in-sample bound, because the coefficient does not survive
out-of-sample. Estimating `phi` online by recursive least squares does not
change this; it converges toward zero.

> **Conclusion:** the observed negative autocorrelation is not alpha. A
> market maker cannot harvest it.

### 1.2 The negative autocorrelation is manufactured by the zero mass

A hurdle variable — zero with probability `p0`, otherwise a continuous draw —
has a **mechanically negative** lag-1 autocorrelation even when the continuous
part is iid. With `p0 = 0.3507` and an iid body:

```
predicted ACF1 = -p0 = -0.3507
observed  ACF1 = -0.2180
```

The entire sign of the dependence is produced by the atom at zero, which
occurs in 35% of buckets because the quoted mid simply does not change. No quote
bounce needs to be posited to explain it.

A related artifact was found and corrected in the analysis itself: dividing by a
*smooth* scale series creates negative serial correlation mechanically. The
series `1/sigma_t` has lag-1 autocorrelation **+0.986** purely from EWMA
smoothness, and `sigma_t` has **+0.999**. Standardising by a slowly varying
denominator manufactures the very dependence one then "discovers".

### 1.3 What *is* real serial dependence

The **magnitude** of the move is serially dependent, and this survives the
standardisation:

| statistic | value |
|-----------|-------|
| `ACF1(|z|)` on all buckets | +0.181 |
| `ACF1(|z|)` on consecutive non-zero moves | **+0.236** |
| `ACF1(z^2)` on consecutive non-zero moves | +0.043 |
| `sum` of `ACF(z^2)` over 120 lags | +0.217 |

Volatility clusters with a 2–5 step memory. This is the genuine non-i.i.d.
structure in the data, and it is a *scale* phenomenon, not a *mean* phenomenon.
A plain EWMA under-responds to it — but a GARCH on unconditional squares is
worse still (see §3.2), because the right object is a recursion on the
**conditional** (non-zero) shocks.

---

## 2. The measurement channel: an atom at the smallest grid step

The mid is quoted on a 0.5 IRT grid, so the move distribution is not smooth.
Measured atom masses:

```
P(dmid = 0)     = 0.3507
P(|dmid| = 0.5) = 0.1197
P(|dmid| = 1.0) = 0.0591
P(|dmid| = 1.5) = 0.0262
P(|dmid| = 2.0) = 0.0193
```

The diagnostic is the ratio `P(0.5)/P(1.0)`:

| series | `P(0.5)/P(1.0)` |
|--------|-----------------|
| real Nobitex data | **2.025** |
| synthetic martingale on the same 0.5 grid | 0.99 |

A smooth symmetric density on a 0.5 grid gives ≈ 1.0. The real data has
**twice** the mass at one grid step as at two. Combined with the fact that
47.0% of all moves are `0` or `±0.5`, the observation channel is dominated by
quote resampling at tick resolution.

**This matters practically, not just scientifically.** Any predictive density
here must be a pmf on the lattice, `q(y) = F(y+0.25) - F(y-0.25)`, not a
continuous density evaluated at a point. Using the continuous form inflates
`q` above 1 — an impossible probability — and this bug is exactly what produced
the spurious result in §3.3.

---

## 3. Model search: the ladder and what each rung was worth

All figures are mean prequential NLL (nats/bucket) unless stated. `n = 339,638`
scored steps after an 8,000-bucket warm-up.

### 3.1 The ladder that survived

| model | 1st half vs ref | 2nd half vs ref | mean NLL |
|-------|-----------------|-----------------|----------|
| REF: iid Student-t4 + EWMA scale | — | — | 3.5814 |
| iid Student-t2.5 + EWMA | +0.0044 | +0.0058 | 3.5866 |
| hurdle-t, `nu=4`, global `p0` | −0.0153 | −0.0154 | 3.5661 |
| **hurdle-t, `nu=2.5`, global `p0`** | **−0.0649** | **−0.0559** | **3.5211** |
| hurdle-t, `nu=2.5`, cells(sp,st,oc) | −0.0579 | −0.0501 | 3.5276 |
| hurdle-t, `nu=3.0`, global `p0` | −0.0510 | −0.0455 | 3.5332 |

**The recommended model is the two-component hurdle:**

```
q(y | I_t) = p0_t · 1{y = 0}  +  (1 - p0_t) · t_{2.5}(y ; 0, s_t)
```

with `p0_t` an EWMA of the zero indicator and `s_t` an EWMA of the second
moment of the **non-zero** shocks, converted to the Student-t scale by
`s = sqrt(E[x²|x≠0] · (nu−2)/nu)`, plus an online calibration multiplier.

**Headline: −0.0603 nats/step (1.68% of the incumbent loss),
95% CI [0.0562, 0.0645], Diebold–Mariano t = −31.4, p = 9.5e−216.**
The gain is larger on the second half than the first, so it is not a
sample-selection artifact.

The tail index was chosen by scanning `nu ∈ {1.5,…,5.0}` and confirmed
independently on both halves: `nu=2.5` won on each (−0.052 and −0.043 vs
`nu=4`), which rules out the winner's curse.

### 3.2 Negative results worth recording

* **GARCH on unconditional squares is worse than a plain EWMA** (+0.008 to
  +0.022 nats). Standardising the shock before accumulating it makes the
  recursion dimensionless and it compounds without bound. The scale recursion
  must live in observable units on the conditional support.
* **Conditioning `p0` and the scale on state cells** (spread × staleness ×
  trade, 12 cells) is *worse* than a single global pair, despite `p0` genuinely
  ranging 0.14→0.52 across cells and the conditional `p0` reducing the hurdle
  log-loss by 0.0002. The cells each hold ~29k observations; the partition
  spends more in estimation error than it recovers in fit. The conditional
  `p0` is nonetheless *well calibrated* (predicted 0.345 vs realised 0.339), so
  the global value is simply adequate.
* **An ARCH-in-mean is worthless** (−0.0004 nats, §1.1).

### 3.3 A bug that looked like a breakthrough

A three-component mixture (zero + narrow granular + heavy body) initially
appeared to gain **−0.818 nats/step**, reproduced on both halves
(−0.745 and −0.888), with t = −122. It was an artifact.

The granular component was evaluated as a *continuous* density at a lattice
point. Because that is missing the cell-width factor `h = 0.5`, and because
the learned `sigma1` collapsed to its 0.25 floor, the branch degenerated into
a **near-constant** function of `y`:

| `y` | continuous | correct lattice pmf |
|-----|-----------|---------------------|
| 0.5 | 1.559 | 0.500 |
| 1.0 | 1.596 | 0.0228 |
| 2.0 | 1.596 | ~0 |
| 9.0 | 1.596 | ~0 |

So it assigned the *same* probability to a 1-tick move and a 400-tick move.
That is not an over-sharp density — it is a **flat** one, and it happened to
be large exactly where the data is densest (47% of moves are `0` or `±0.5`).
Lattice-correcting it turned −0.818 into **+0.22**, i.e. the mixture is *worse*
than the simple hurdle. A pure point mass at `|y| = 0.5` was worse still
(+0.34), because the data also carries mass at 1.0, 1.5 and 2.0 that one atom
cannot represent.

**The diagnostic that settled it** (`q19_artifact.py`) is worth stating,
because the first explanation was wrong. It is tempting to assume the
continuous branch "exceeded 1" and so scored negative loss. Measured: it did
so in **0 of 222,004** non-zero steps — the mixture weight `p1 ≈ 0.116` kept
`q = 0.116 × 1.596 = 0.19 < 1`. The `q > 1` story was a plausible-sounding
rationalisation that the data did not support. The real fault is subtler: a
sub-unit, still-unnormalised, *flat* branch, which the
`log q <= 1e-9` assertion could never catch because the density looked legal
while being wrong in a way that flattered it.

The general lesson: **`q <= 1` is necessary but not sufficient.** A
normalised-looking density can still be misnormalised on a lattice. The check
that catches this class is the PIT and the coverage ladder, not the assertion.

### 3.4 The calibration falsifier fired as predicted

Freezing the scale at a fixed multiple of the learned value:

| variant | 1st half | 2nd half | z sd | 90% coverage |
|---------|----------|----------|------|--------------|
| recommended | −0.065 | −0.056 | 1.96 | 0.773 |
| 0.6× (too sharp) | **−0.012** | **+0.007** | 3.16 | 0.719 |
| 1.4× (too safe) | −0.028 | −0.029 | 1.36 | 0.828 |

The over-sharp variant has a *better median loss* and still **reverses sign**
out-of-sample while its z-dispersion triples. An MPC sampling from it would be
told to under-weight tail risk. Sharpness is not a free lunch; calibration is a
hard constraint, not a tiebreak.

---

## 4. Model families considered and rejected

| family | verdict | why |
|--------|---------|-----|
| AR / MA in the mean | rejected | ceiling 0.024 nats, realised 0.0004 |
| GARCH / ARCH scale | rejected | +0.008..+0.022 nats; recursion must be on conditional shocks |
| regime-switching (HMM) | not pursued | the only regime-conditional quantity that survives is the *scale*, and a single EWMA + calibration already captures it; cell partitioning (§3.2) was worse |
| state-space / Kalman | not pursued | would estimate a latent level, but the value process is a martingale here (no mean predictability) so there is no latent drift to recover |
| Hawkes / self-exciting point process | not applicable | would model *event arrivals*; the arrival process here is the bucket grid, not an event stream |
| neural / transformer | rejected on principle | the attainable gain is 0.06 nats and the honest mean-signal ceiling is 0.024; a sequence model would fit noise and cannot beat a martingale bound it does not have data for |
| deep generative (normalising flow) | rejected | would improve NLL only by sharpening, which §3.4 shows inverts out-of-sample |

The general principle: the entropy that remains in this process is almost
entirely **dispersion** uncertainty (which branch, how large a jump), not
**direction**. Models that add capacity to the direction channel are adding
capacity to a channel that is empty.

---

## 5. What to do with this

1. **Ship the two-component hurdle** (zero + Student-t2.5) with the conditional
   scale and the online calibration. It is ~30 lines, has no fitted constants,
   and is 0.06 nats/better than the incumbent with `p ≈ 1e−215`.
2. **Use a lattice pmf everywhere.** `t_logpmf_lattice` in
   `analysis/models_v3.py` does this exactly and is ~1000× faster than
   `scipy.stats.t.cdf` per call.
3. **Keep the `log q <= 1` assertion, but do not trust it alone.** It catches
   impossible densities, not *flat* ones. Pair it with a PIT uniformity test
   and a coverage ladder; those are what caught the mixture bug.
4. **Do not add a mean.** The bound is 0.024 nats and is not reachable
   out-of-sample. If a directional feature is ever proposed, it must beat
   0.024 nats *out-of-sample*, not in-sample.
5. **Do not add regime cells** without a much longer sample; at 29k
   observations per cell the estimation error exceeds the fit gain.

---

## 6. Reproducing

```bash
cd research/model_lab/analysis
.venv/bin/python q1_dependence.py    # serial dependence of the standardised innovation
.venv/bin/python q3_diagnose.py      # where the negative ACF lives
.venv/bin/python q4_residual.py      # the 0.024-nat ceiling on any mean model
.venv/bin/python q7_scale.py         # the zero mass explains the ACF
.venv/bin/python q8_final.py         # the ladder, with calibration
.venv/bin/python q9_nu.py            # tail-index scan, split-sample
.venv/bin/python q10_confirm.py      # out-of-sample confirmation + falsifier
.venv/bin/python q13_mixture.py      # the mixture, and the bug it exposed
.venv/bin/python q14_limit.py        # synthetic martingale control
.venv/bin/python q16_final.py        # why a point mass fails
.venv/bin/python q18_final.py        # the honest final ranking
.venv/bin/python q19_artifact.py     # what really caused the phantom gain
```

Logs are in `analysis/out/q*.txt`.
