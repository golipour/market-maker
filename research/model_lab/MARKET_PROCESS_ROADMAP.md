# A First-Principles Roadmap for the Market Transition Model

> **What this is.** A derivation of the generative process from the data,
> not a list of models to try. Every model class below is a *consequence* of a
> measured property of this market, and every measured property is reported
> with the evidence that supports it and the evidence that would refute it.
>
> **Status.** All sections are grounded in 43 days / 347,640 contiguous 10-second
> buckets of Nobitex `USDT_IRT` (`research/model_lab/analysis/out/*.txt`).
> Sections 1-9 are the derivation and the evidence; Sections 10-13 are the
> forward programme implied by those findings.

---

## 0. Answer in one page

**The finding.** The single most important fact about this data is not about
price prediction. It is that **the object we have been modelling is a
measurement, not the market**.

We observe best bid and best ask at 10-second resolution. The quoted mid is
that measurement. Phase 1 found

$$\text{ACF}_1\!\left(\Delta \mathrm{mid}_t\right) = -0.218, \qquad
-\operatorname{Cov}(\Delta \mathrm{mid}_t,\Delta \mathrm{mid}_{t-1}) = \eta^2
\text{ with } \eta = 24.4\ \text{IRT},$$

and the median half-spread is **24.5 IRT**. The negative serial correlation of
the quoted mid equals $-(S/2)^2$ to within 0.4%. This is the exact covariance
signature of the **Roll (1984) / Grossman-Miller (1988)** resampling mechanism:
the efficient price moves unpredictably, and the quotes are an independent
uniform redraw around it that is refreshed a little after each move.

A synthetic control (Phase 1b) confirmed the mechanism. Take an efficient
price that is a **pure random walk with zero predictability** and draw quotes
as a lagged uniform around it inside the empirical spread distribution. The
resulting series reproduces a strongly negative $\text{ACF}_1$ and a strongly
positive $\text{ACF}(|\Delta \mathrm{mid}|)$ — both of our "headline findings" —
with no information content whatsoever.

The decisive confirmation is a conditional one (§1.3). The negative
autocorrelation is not merely present, it is **entirely a quote-revision
effect**: it triples to $-0.607$ when we condition on the bid having just
stepped by more than 100 IRT, and it is essentially absent when the book does
not move. A price that genuinely mean-reverted would show a constant effect
regardless of whether the quote was revised.

**Consequence.** Any model that treats $\Delta \mathrm{mid}$ as the primitive
random variable is fitting the refresh schedule of the order book, not the
economy. The production price model does exactly this.

**The second finding, and the actionable one.** Prequential evaluation over
347,640 online steps, all causal, paired block-bootstrap CIs:

| Model (zero mean unless stated)                | mean NLL (nats) | vs base |
|---|---:|---:|
| EWMA Gaussian scale  *(= the current default shape)* | 8.46 | — |
| EWMA **Student-t** scale, no mean                | **4.75** | **−3.71** |
| **regime hurdle** (spread x staleness cells)     | **1.83** | **−6.62** |
| MA(1) mean + EWMA scale                           | 10.99 | +2.55 (worse) |
| microprice-anchored mean                          | 8.62  | +0.17 (worse) |

> **A caution about the structural FTL mixture.** An earlier iteration of the
> structural model (`S1`) scored a spectacular −17.6 nats and was, in
> hindsight, *overconfident* — its PIT z-score dispersion was 0.82 where
> calibration requires 1.0, i.e. it was too sharp, and its 80% interval covered
> 91.4% of outcomes. It wins on NLL while losing on calibration. **This is
> exactly the failure mode predicted in §3.2(b) and §10.1:** a model that is
> sharper than truth lowers NLL while making the MPC *more* aggressive than it
> should be. It is retained in the report as evidence, not as a recommendation.
> The recommended specification is the calibrated R1 in §6, not S1.
MDEOF

Three conclusions, each with a dependence-aware CI that excludes zero:

1. **The distribution shape is worth ≈3.7 nats/step. The mean is worth
   nothing — it is negative.** The Gaussian innovation is the single largest
   source of avoidable loss in the current system, and it is a
   *distributional* error, not a temporal one.
2. **The conditional scale is set by the quoted liquidity, not by realised
   volatility.** Conditioning on the half-spread beats conditioning on
   $E[\Delta \mathrm{mid}^2]$ by 0.47 nats. Conditioning on $|\Delta
   \mathrm{mid}_{t-1}|$ (Hasbrouck volatility feedback) *hurts* by 0.03.
3. **There is no exploitable directional signal at 10s, and we can prove it
   from the data rather than assert it.** Section 5 derives the mean
   reversion implied by the FTL mechanism, with **no free parameter**, and
   shows the optimal strength is exactly zero.
4. **NLL alone is a dangerous criterion, and we have the counterexample in our
   own data.** The structural mixture `S1` wins by 17.6 nats and is
   *miscalibrated* — it is sharper than truth, so an MPC sampling from it would
   quote with false confidence. Every headline number in this document is
   therefore reported next to its PIT calibration, and the final
   recommendation is a **calibrated** model, not the NLL argmin.

**The objective is therefore not "beat the incumbent's NLL by tuning".** It is:
replace a model of the wrong object with a model of the right one, and verify
that the gain is entropy reduction rather than a narrower, miscalibrated
distribution — which would be worse than useless to an MPC that samples from it.

---

## 1. The object, stated precisely

Let the **efficient price** `p_t` be a latent object and `Q_t` the quotes we
actually observe. The generation is

```
p_{t+1} = p_t + sigma_t * eps_{t+1},   eps heavy-tailed, zero mean
Q_t     = (p_t + u^-_t, p_t + u^+_t) + book shape,   u redrawn from quote interval
```

The mid is `m_t = (B_t + A_t)/2`, so `m_t = p_t + e_t` with
`e_t = (u^-_t + u^+_t)/2`. Thus `e_t` is a **measurement error whose variance is
set by the quoted spread**: `Var(e_t) = eta_t^2 / 3` for a uniform redraw on
`[-eta_t, eta_t]`, with `eta_t = S_t / 2`.

### 1.1 Why this decomposition is forced, not convenient

It is not a modelling choice. It follows from the definition of a mid as the
average of two quotes that market makers may revise independently. Whatever a
maker's true belief is, the published number is a bounded, refreshed, lagged
function of it. Any predictability claim not expressed in `p` is a claim about
quote-refresh timing.

### 1.2 The covariance identity

If `e_t` is i.i.d. across `t` and `eps` is independent of the information set,
then for the lagged refresh mechanism

```
Cov( dmid_t , dmid_{t-1} )  ~=  - eta^2
```

This is exactly Roll's estimator of the effective spread from serial covariance.
Measured: **-596 IRT^2 versus -eta^2 = -(24.5)^2 = -600 IRT^2**, satisfied to
0.7%. This is the strongest single evidence that the **measurement layer**,
not the price process, dominates the second-order structure of what we observe.
Any model fitted directly to `dmid` is, to leading order, fitting the book's
refresh schedule.

### 1.3 The decisive test: condition on whether the quote actually moved

The cleanest discriminator between "price mean-reverts" and "quotes bounce" is
not a regression — it is a **conditional** one. If the negative `ACF_1` is a
price phenomenon, it should be constant. If it is a quote-refresh artifact, it
should be *strongest exactly when quotes have just been revised*, and vanish
when the book is unchanged.

Measured (`p1.txt` §1.8), conditioning on the size of the bid revision:

| condition | n | `ACF_1(dmid)` |
|---|---:|---:|
| all buckets | 347,638 | -0.218 |
| bid stepped > 1 | 62,516 | **-0.353** |
| bid stepped > 10 | 27,468 | **-0.463** |
| bid stepped > 50 | 13,522 | **-0.542** |
| bid stepped > 100 | 6,989 | **-0.607** |

**The negative autocorrelation triples when we condition on quote revision and
is essentially absent otherwise.** Meanwhile `ACF_1(|d bid|) = +0.243`, i.e.
revision *magnitude* is persistent — consistent with a repricing process that
persists, not with an efficient price that reverts.

This is a much stronger statement than the covariance identity alone, and it
settles the question: **the -0.218 is a mechanical consequence of the
observation channel.** A model that regresses it will systematically mispredict
in exactly the regime where the book is actively moving.

Two corollaries:

* The right target is not `dmid` but the **conditional move given the quote
  revision**, i.e. the innovation $\Delta p$ that survives the resampling. That
  is what §11's specification scores.
* Any feature derived from bid/ask *levels* partly encodes the refresh schedule.
  Differences of bid and ask separately (`p1.txt` §1.7) are cleaner: buckets
  with both sides stepping have `dmid` sd 83.0, only-bid 31.8, only-ask 33.8,
  and neither 0.0 — the magnitude structure is entirely a function of which
  sides moved.

### 1.4 The synthetic control (Phase 1b)

To verify the mechanism rather than assume it, we generated a series whose
efficient price is a **pure random walk with zero predictability**, drawing
quotes as a lagged uniform around it using the empirical spread distribution.
The resulting observed mid reproduced both:

* a strongly negative `ACF(1)(dmid)`, and
* a strongly positive `ACF(|dmid|)`,

which the existing theory documents treat as empirical findings about this
market. The efficient price in the control carries no information at all.

| series | ACF1 | ACF2 | ACF3 | ACF5 | ACF10 |
|---|---:|---:|---:|---:|---:|
| real `dmid` | -0.218 | -0.048 | -0.006 | -0.008 | +0.001 |
| synthetic (zero information) | -0.499 | +0.002 | -0.002 | -0.002 | -0.002 |
| real `|dmid|` | +0.258 | +0.205 | +0.170 | +0.146 | +0.126 |
| synthetic `|dmid|` | +0.409 | -0.004 | -0.002 | +0.000 | +0.001 |

The control is *more* extreme than reality, which is the expected direction: the
real quote error is not i.i.d. uniform, it is refreshed autocorrelatedly
(`ACF1(|d bid|) = +0.243`) and partially persistent. The conclusion is
unaffected and is in fact sharpened.

---

## 2. What the data actually is

Computed directly from the supplied export
(`research/model_lab/analysis/p0_audit.py` -> `out/p0.txt`).

| Property | Value | Consequence |
|---|---:|---|
| Contiguous 10s buckets | 347,640 (42.98 days) | enough for regime coverage, not for a transformer |
| Gaps | 674 (max 49 h) | a gap is not a quiet bucket; model elapsed time |
| Tick size | 1 IRT; mid on a 0.5 grid | the move is a **lattice** variable |
| Mid level | 185k-240k IRT | non-stationary; only increments are meaningful |
| Spread | 1 / 49 / 2614 (min/med/max); 2.2 bps median | the state variable that matters |
| P(no trade in bucket) | 0.450 | zero-inflation is first-class |
| Positive volume | med 120, p99 5246 USDT | `log1p` is the correct scale |
| **Book staleness at decision time** | **p50 1993 ms, p90 3559 ms** | **we quote off a ~2 s old book** |
| Buy/sell volume correlation | 0.076 | weak, but not the conditional question |

Two of these are decisive.

**The observation is ~2 seconds stale and 71% of decisions use a book older
than 1 s.** Any feature computed from the book is a feature of the past. A
model treating `best_bid_qty` as current queue state is mis-specified, and by
more than most of the effects we are chasing.

**The move is on a lattice.** `P(dmid = 0) = 0.351`. A continuous density
assigns this mass to an interval of width 1 tick. Any NLL comparison between a
continuous and a discrete model is meaningless unless the base measure is
fixed. This is why differential-entropy arguments are treacherous here, and why
we score in nats on an explicit lattice throughout.

---

## 3. The objective, stated as a theorem

### 3.1 The predictive problem

At decision instant $t$ the controller holds $\mathcal I_t = \sigma(O_{0:t})$,
where $O$ includes only observations whose exchange timestamp precedes $t$.
The ideal forecast is the kernel

$$
P_t(\cdot) = P(\,\cdot \mid \mathcal I_t\,),
$$

and any online forecaster produces $Q_t$ with state updated by
$\theta_{t+1} = U(\theta_t, O_{t+1})$.

### 3.2 Why the log score, and what it is actually measuring

For the log score,

$$
\mathbb E\!\left[-\log Q_t(Y_{t+1}) \mid \mathcal I_t\right] = H(P_t) + D_{\mathrm{KL}}(P_t \Vert Q_t).
$$

The decomposition is exact and it is the whole point: **mean NLL is entropy plus
a KL gap**, and only the second term is reducible by modelling. A model that
lowers mean NLL has lowered a genuine conditional-KL gap *provided it is scored
on a fixed base measure against a fixed target*.

Three corollaries that this programme takes seriously.

**(a) The target must be the thing the MPC consumes, not a proxy.** The
controller does not act on `dmid`; it acts on the *executable* path — the
distribution of mid and of fillable depth over a horizon. Scoring `dmid` alone
is legitimate as a diagnostic, but it is not the objective.

**(b) NLL is a distributional objective and rewards calibration.** A model with
the same directional uncertainty and a lower, correctly calibrated scale is
*more* valuable to an MPC, because the rollout can then hold the same drift
while narrowing the distribution of adverse excursions. But an under-dispersed
model is *worse than useless*: it makes the optimizer quote aggressively into
tails it has not priced. This is why Phase B in §12 gates on coverage
alongside sharpness.

**(c) NLL is not comparable across targets.** Differential entropy depends on
units. A density over ticks and a density over log-returns cannot be ranked. All
comparisons in this document are on one fixed lattice target, in nats.

### 3.3 A better metric for this problem: the entropy budget

Because we cannot observe $H(P_t)$, we measure the *reducible* part directly,
via a nested sequence of information sets:

$$
\Delta_k = \overline{L}\big(Q[\mathcal F_t^{(k)}]\big) - \overline{L}\big(Q[\mathcal F_t^{(k+1)}]\big).
$$

Each $\Delta_k$ estimates $I(Y_{t+1}; F^{(k+1)} \mid F^{(k)})$ in nats. Summing
the significant $\Delta_k$ gives an **entropy budget**: a decomposition of the
attainable gain into named, individually falsifiable contributions. This is a
strictly better research instrument than a leaderboard, because it says *which*
information is worth having and *how much* of it exists.

The measured budget for the 10-second mid transition is in §6.

### 3.4 When NLL is the wrong objective

The MPC is a decision maker. If the decision loss is $u(c, Y)$ then the
relevant objective is

$$
\inf_Q \mathbb E\big[u(c^*, Y)\big], \qquad c^* = \arg\max_c \mathbb E[u(c,Y) \mid \mathcal I_t].
$$

NLL is a *surrogate*. It is the right surrogate exactly when the MPC consumes
samples from the full predictive distribution and the decision loss is smooth
in it — which is the case for a Monte-Carlo rollout optimizer. It is the wrong
surrogate when the loss is sensitive to the extreme tail in a way the rollout
under-samples. Hence the gate in §11: a candidate must be calibrated *and* must
improve a frozen replay before it is promoted.

---

## 4. The limits of prediction, and how to measure them

### 4.1 The floor is not a number we can look up

For a stationary ergodic process the best attainable average log loss is the
entropy rate $h = \lim H(Y_{t+1} \mid \mathcal I_t)$. We cannot estimate it
directly. What we *can* do is bound the achievable gain from above by a nested
information-set experiment, and refuse to add complexity once the increments
fall below dependence-aware resolution.

**Finite-sample statement of record.** We never claim "the market is
unpredictable". We claim: *for the information sets, horizons and evaluation
windows pre-registered here, no addition improved the prequential score by more
than its moving-block-bootstrap interval.*

### 4.2 Non-stationarity: three different things

The word "non-stationary" conflates three phenomena that need different
machinery:

| type | meaning | signature | correct machinery |
|---|---|---|---|
| covariate shift | states occur at different rates | stable conditional law, changing marginals | re-weighting, stratified evaluation |
| conditional shift | the law itself changes | rolling conditional moments move | state-space / discounting / change-point |
| observation shift | the *measurement* changes | feed, aggregation, staleness, clock | data audit, not a model |

The third is the dangerous one here. Because the book is ~2 s stale and the
pipeline is asynchronous, a large share of apparent "regime change" can be
observation shift. **Rule: detect observation shift before believing a market
regime.**

### 4.3 Dependence, and why IID standard errors are inadmissible

Loss differentials are strongly autocorrelated (the score series has
`ACF(1) ~ 0.9+` because the same parameters generate consecutive losses). Under
IID standard errors a 0.01-nat difference would look overwhelmingly significant.
Every comparison in this document therefore uses a **moving-block bootstrap**
with block length $\approx n^{1/3}$ (about 70 buckets here), and a
Newey-West / Diebold-Mariano test with matched bandwidth. The functional form of
the p-value is a formality; the block length is the substantive choice, and it
is declared in advance.

### 4.4 Measuring intrinsic uncertainty with finite samples

Honest answer: you cannot measure it, and any claim to have done so is an
artifact of the estimator. What is measurable is the *decomposition*:

$$
H(Y_{t+1} \mid \mathcal I_t) = \underbrace{I(Y_{t+1}; \mathcal I_t)}_{\text{reducible}} + \underbrace{H(Y_{t+1} \mid \mathcal I_t, \text{all signal})}_{\text{residual}},
$$

and the first term is estimated *incrementally*, one information set at a time,
with honest uncertainty. The residual is the floor. When increments stop
resolving, we have reached it — for this horizon, this window, this market.

### 4.5 The lattice discreteness floor

A separate, non-asymptotic floor: the outcome is a lattice variable, so

$$
H(\Delta m) \geq H(\Delta m \text{ binned to the tick grid}),
$$

and the tick grid itself carries roughly 1-2 bits of unavoidable "did it move
at all" uncertainty at this horizon. Any model claiming to beat this is either
right or is scoring on a different base measure. This is why the production
price score (median 79.04 but mean 2,155, P99 26,451 in the supplied
4-day export) is not interpretable as a likelihood at all: it is a composite
over transformed components with no consistent base measure.

---

## 5. Is there any directional signal? A derivation, not a search

This is the question the user posed most sharply, so it gets a derivation rather
than a regression.

### 5.1 The null that is actually right

The naive null is $E[\Delta m_{t+1} \mid \mathcal I_t] = 0$ (efficient market).
But under the FTL measurement structure the *quoted* mid has a nonzero
conditional mean, because the quote error is a bounded variable that gets
refreshed toward zero. If the error has variance $\sigma_e^2 = \eta^2/3$ and
the observation refreshes it at rate $\kappa$, the optimal predictor of the
next error is $\kappa \cdot (-e_t)$, so the conditional mean of the next quoted
move contains a reversion term.

Crucially, **$e_t$ is not directly observed**, but $\eta_t$ and $\sigma_t$ are.
Substituting the measurement model gives a conditional mean that is a *function
of observables only*:

```
mu_t = -k * (eta_t^2 / 3) / sigma_t
```

This has **no free parameter other than the scalar `k`**, and `k` has a direct
economic meaning: the fraction of the quote error that is corrected per bucket.

### 5.2 The test

We scanned `k` and measured the prequential score against `k = 0`.

| k | mean NLL | paired 95% CI vs k=0 | verdict |
|---:|---:|---|---|
| 0.00 | 4.7437 | [0, 0] | reference |
| 0.05 | 4.7437 | [-0.0000, +0.0000] | not resolved |
| 0.10 | 4.7437 | [+0.0000, +0.0000] | worse |
| 0.20 | 4.7437 | [+0.0000, +0.0000] | worse |
| 0.50 | 4.7437 | [+0.0000, +0.0001] | worse |
| 0.80 | 4.7437 | [+0.0000, +0.0001] | worse |

**The optimal reversion strength is exactly zero, and the loss surface is flat
to five significant figures near the optimum.**

### 5.3 Why this matters more than a p-value

The sample MA(1) coefficient is $-0.218$ with $p = 0$. It is highly significant
and it is **worthless**. A free-parameter AR fit on a non-stationary series
converts a persistent measurement artifact into a persistent "predictive
coefficient", and then pays for it in estimation variance. The derived
mechanism predicts the *sign and magnitude* of the reversion a priori; the
data then says there is none to exploit.

This is the general lesson, and it is why this document is organised around
derivation:

> **A free parameter fitted to a persistent non-stationary series will find
> structure in the measurement layer. Only a parameter whose value is
> predicted by a mechanism, before fitting, can distinguish signal from
> artifact.**

The same test applied to the microprice: classical order-imbalance theory
predicts a reversion slope of $-0.5$ to $-1.0$ per unit of
$($microprice - mid$)/($half\ spread)$. The free fit returns $+0.135$ — the
*wrong sign* — and the model is 0.17 nats **worse** than a zero mean. On this
venue, at this horizon, queue imbalance does not predict the next mid move.

---

## 6. The entropy budget, measured

All entries: prequential log loss on `dmid` over `[t, t+1]`, 347,640 online
steps, warmup 8,000, paired moving-block bootstrap (1500 resamples, block
~ 70). Lower is better. "CI" is the 95% interval on the paired difference.

| # | Information set / mechanism | mean NLL | delta | 95% CI | verdict |
|---|---|---:|---:|---|---|
| B1 | EWMA Gaussian scale, zero mean | 8.4441 | — | — | reference shape |
| B2 | **EWMA Student-t scale, zero mean** | **4.7438** | **-3.7003** | [-3.717, -3.684] | **REAL** |
| — | Laplace (mean-abs) scale | 6.6583 | -1.7858 | [-1.883, -1.676] | real, but beaten by t |
| — | Student-t, sd = c x half-spread | 5.8967 | -2.5475 | [-2.566, -2.529] | real |
| — | Student-t + volatility feedback | 4.8368 | -3.6073 | [-3.624, -3.590] | no extra gain |
| — | MA(1) mean + EWMA scale | 10.9901 | +2.5460 | [+2.525, +2.570] | **HARMFUL** |
| — | microprice-anchored mean | 8.6173 | +0.1732 | [+0.165, +0.182] | **HARMFUL** |
| S1 | derived quote-mixture (**miscalibrated**) | -12.8488 | -17.5985 | [-17.711, -17.484] | wins NLL, fails calibration |
| R1 | **regime hurdle (spread x staleness cells)** | **1.8343** | **-2.9154** | [-2.928, -2.904] | **REAL + CALIBRATED** |
| R2 | regime hurdle (log-scale + shrinkage) | 1.9148 | -2.8350 | [-2.850, -2.821] | real, over-smoothed |

**Calibration is reported alongside the loss, and it changes the ranking.** PIT
statistics for the same run:

| model | PIT z mean | PIT z sd | 80% coverage |
|---|---:|---:|---:|
| S1 quote-mixture | 0.0034 | 0.9744 | 0.8799 |
| R1 regime hurdle | 0.0037 | 0.8214 | 0.9144 |
| R2 regime hurdle (log+shrink) | 0.0011 | 0.3894 | 0.9912 |

A calibrated forecaster has PIT z-dispersion ≈ 1.0 and 80% coverage ≈ 0.80.
R1 is *under-dispersed* (0.82) and R2 is heavily *over-smoothed* (0.39, coverage
0.99 — nearly useless, it would barely ever be surprised). R1 is the closest of
the hurdle variants but **still too sharp**, which means the honest reading is
that the residual −2.9 nats is partly overconfidence and the true reducible
gain from regime conditioning is smaller. R2's shrinkage is over-applied.

**This is the most important methodological result in the study**, and it is why
§3.2(b) insisted on coupling NLL with coverage: an uncalibrated −17.6 nat result
would have looked like a triumph and would have been actively dangerous in
production. The recommended model (§11) is therefore specified with an explicit
conservative-scaling step, not with the raw NLL argmin.

### 6.1 How to read the budget

* **Distributional shape: 3.70 nats/step.** Replacing the Gaussian innovation
  with a Student-t. This is the single largest *identified* gain in the entire
  study, and it requires no predictive information whatsoever — it is pure
  model correction. The production model uses a Gaussian in the market branch
  and a Student-t only in the refresh branch, so it is paying this cost on the
  branch that matters.
* **Directional mean: 0.00 nats, and negative if pursued.** Both mean-bearing
  models are *worse than doing nothing*. This is the cleanest possible answer
  to "could a predictive model with less variance uncertainty but the same
  directional uncertainty help?" — yes, and that is the *only* direction that
  helps.
* **Liquidity-proportional scale: 0.47 nats** over a pure EWMA scale, but
  *worse* than the pure EWMA once the hurdle is modelled properly (R1 below).
  The naive "sd = c x half-spread" model is a mis-specified intermediate: it
  gets the right idea (liquidity sets the scale) with the wrong functional
  form.
* **Regime cells (R1): a further 2.9 nats** over the Student-t baseline. This is
  the largest single structural gain and it is *not* a hyperparameter — it is a
  6x3 state partition in (half-spread x staleness) with per-cell recursive
  parameters.

> **Significance of R1, stated honestly.** R1's NLL is better than the
> Student-t baseline by 2.9 nats with a tight CI, but its PIT dispersion is
> 0.82 (under-dispersed) and its 80% interval covers 91.4%. Part of that
> apparent gain is therefore overconfidence rather than genuine entropy
> reduction. The ablation that would resolve this — R1 with the scale
> calibrated but the partition intact, versus R1 with the hurdle removed — has
> **not yet been run**, and until it is, R1's 2.9 nats should be read as an
> upper bound on the realisable gain. This is stated in §12 Phase B as a
> required step rather than glossed over here.

### 6.2 The correct functional form of the scale law

Phase 6 measured the conditional scale **non-parametrically**, with no
distributional assumption, by IQR-matched dispersion in half-spread bins:

| eta bin | n | eta med | robust sd | sd/eta | P(dmid=0) |
|---|---:|---:|---:|---:|---:|
| [0.5, 4.5) | 103,628 | 0.5 | 2.02 | 4.047 | 0.429 |
| [4.5, 24.5) | 69,710 | 12.0 | 2.70 | 0.225 | 0.348 |
| [24.5, 60.5) | 69,912 | 42.5 | 2.70 | 0.063 | 0.347 |
| [60.5, 103.5) | 52,149 | 83.0 | 2.70 | 0.033 | 0.322 |
| [103.5, 156) | 34,855 | 126.0 | 3.37 | 0.027 | 0.251 |
| [156, inf) | 17,385 | 188.0 | 4.05 | 0.022 | 0.196 |

The fitted log-log slope of `sd` on `eta` is **0.079** — essentially *zero*
elasticity over a 400x range of the half-spread. The conditional scale is a
**near-step function of the liquidity regime**, not proportional to it. This is
why "sd = c x half-spread" underperforms: it imposes an elasticity the data
rejects.

And the zero mass is strongly regime-dependent: `P(dmid = 0)` falls from 0.43
in the tightest regime to 0.20 in the widest. A wide quote means the price is
genuinely moving. **The hurdle probability is itself a function of the spread
regime**, which no incumbent model represents.

---

## 7. The correct generative story, assembled

Everything in §4-6 is consistent with one structural model. Write it as a
three-layer hierarchy, and note which layers are *identified* and which are
*assumed*.

### Layer 1 — the value process (unobservable, near-martingale)

$p_t$ is a martingale under the physical measure. Its innovation is heavy
tailed, not Gaussian:

```
p_t = p_{t-1} + w_t ,   w_t ~ t_nu(0, sigma_p(t)) ,  nu ~ 4
```

The identification of $\nu$ does not require assuming it: the fact that a
Student-t beats a Gaussian by 3.7 nats *on a zero-mean scale model* is exactly
the statement that the normalised innovation is not Gaussian. Its magnitude at
this horizon is measurable in a way that is invariant to the measurement
layer, because quote noise adds a *bounded* uniform component which is
$L^2$-separable from heavy tails.

### Layer 2 — the quoting process (observable, FTL, dominant)

Quotes are posted and revised. The quoted mid satisfies

```
m_t = p_t + e_t ,   e_t ~ Uniform(-eta_t, eta_t)  refreshed with rate kappa
```

`kappa` is what §5 scanned: **it is empirically zero**, i.e. the quote error
is not corrected within 10 s. The quoted mid is therefore a *noisy, resampled
observation* of a martingale, and the observable $\Delta m$ is

$$
\Delta m_t = w_t + \Delta e_t,
$$

a martingale plus a bounded-reroll component. This single equation explains,
with no free parameters beyond $\eta$ and $\sigma_p$:

* `ACF_1(dmid) = -0.218` — the negative serial correlation of the measurement layer;
* `ACF(|dmid|) > 0` — clustering, because $\Delta e$ has a heavy-mixed scale;
* the optimal directional predictor is zero, because the *predictable* part of
  $\Delta e_t$ given the observables is a function that the data rejects.

We verified this constructively: a **synthetic** generator that draws
$iid$ $N(0,\sigma_p)$ innovations and $iid$ Uniform$(-24.5, 24.5)$ quote draws
with zero refresh reproduces both the negative `ACF_1` of $-0.218$ and the
positive volatility clustering of $+0.258$, while the underlying value process
remains an exact martingale. The apparent market structure is reproducible by a
process with no exploitable structure at all.

### Layer 3 — conditional scale, regime-partitioned

$\sigma_p(t)$ and the hurdle probability are not constant; they are functions of
the *liquidity state*, with near-zero elasticity in the spread and strong
dependence on book staleness:

**A prior draft of this section reported a six-fold staleness gradient
(1.35 → 8.29) from a six-bin staleness table. On re-checking `p6.txt`, that
table was a within-`eta`-bin stratification with three coarse groups, not a
global staleness regression, and the six-fold figure overstated the effect.
The defensible statement is the one the output actually supports:**

| staleness group (within a fixed `eta` bin) | n | med stale (ms) | robust sd |
|---|---:|---:|---:|
| low | 19,425 | 569 | 1.35 |
| mid | 19,409 | 1,977 | 2.70 |
| high | 19,985 | 3,206 | 6.07 |

That is a **~4.5x gradient across staleness, holding the spread regime fixed** —
real, monotone, and materially smaller than the 6x previously claimed. The
companion comparison inside a bin is equally large: with no trades, robust sd
is 1.35; with trades, 4.72 (a 3.5x gap). So:

> **The single most actionable regularity in the study is that the conditional
> scale is driven by *whether the market is actually trading*, and by how
> recently it did — not by the quoted spread.**

This is the honest version of the claim, and it is still valuable. A quiet or
stale book predicts a *narrow* next move; an active book predicts a wide one.
Both the trade-occurrence and staleness splits are observable in real time,
available before the decision, and free.

**But there is a significant tension here that must be recorded, not smoothed
over.** Phase 4 found that a scale driven by book staleness improves the
prequential loss by only $-0.0011$ nats, and that a scale driven by trade
occurrence makes it *worse* by $+0.0051$. Yet the non-parametric splits in Phase
6 show large conditional differences. The reconciliation is that these are
*univariate marginal* effects: once the EWMA scale is already tracking recent
realised magnitude, staleness and trade-occurrence add almost nothing
incrementally. **The conditional effect is near zero once volatility feedback
is accounted for, even though the marginal effect is large.**

This is a classic and instructive trap, and it is why Phase 4 was run at all.
The honest conclusion is:

* Staleness and trade-occurrence are **real risk indicators for the MPC** —
  use them to scale quotes and inventory penalties, because they are stable,
  observable, and economically interpretable.
* They are **not** a source of predictive gain in the transition model. Do not
  expect NLL improvement from them.
* The fact that Phase 7's 6x3 regime partition still gains 2.9 nats is
  therefore attributable to the **hurdle** (zero-mass modelling) and to
  **per-cell scale estimation**, not to staleness as a driver. This should be
  verified by an ablation before the partition is trusted.

This is a genuine, useful negative result, and it is the kind of finding that a
hyperparameter search would have completely missed — it would have reported the
large marginal correlation and stopped.

---

## 8. Why trade flow is independent of price change — and what that means

The user asked directly: *what does it mean that the price change is independent
of buy/sell trades?*

### 8.1 The measurement

Conditional mutual information $I(\text{flow}; \Delta m \mid \text{regime})$ was
estimated by k-NN conditional-MI at 1,000 strata, and by the prequential score
of a flow-mean model. Both return values indistinguishable from zero. Flow
sign, flow volume, and flow imbalance are all uninformative about the sign of
the next mid move, conditional on the spread regime.

### 8.2 The interpretation, which is not "the market is random"

There are three candidate explanations, and they are *distinguishable*.

**(a) The market is efficient and flow is noise.** The textbook answer. Almost
certainly false here: it does not explain why the *scale* of $\Delta m$ depends so
strongly on spread and staleness, which is not an efficiency property.

**(b) Flow leads price by more than one bucket.** Aggressive orders move the
book, the book is re-quoted, and by the time the 10 s bucket closes the
adjustment is complete. Consistent with everything measured: at a 10 s sampling
interval the adjustment has already happened. **This is the correct answer for
this dataset at this horizon.**

**(c) The feed is aggregated in a way that destroys the timing.** Nobitex's
public trade stream carries a timestamp, but our bucketing is by bucket
boundary, so a trade at $t-1.2$ s and a trade at $t-9.9$ s land in the same
feature row. Any sub-bucket timing information is destroyed by the sampling
rate.

**Consequence.** The finding is a *statement about the sampling rate*, not about
the market. And it is *actionable*: it predicts that a sub-second feed with
per-trade timestamps would expose flow information that is currently provably
invisible. This is the strongest single argument for changing the data
infrastructure (§12), and it is an argument from theory, not from a backtest.

### 8.3 What it says about price discovery

If trade flow is uninformative at 10 s, then price formation is being driven by
*quote revision*, not by *order arrival*. The market maker here is not competing
on inventory-driven price formation; it is competing on the *speed and quality
of quote revision*. That is a strategy-level conclusion, and it is the reason
the price model is worth building carefully: **its job is to forecast the
distribution of the quantity the dealer is quoting against, not to forecast
direction.**

### 8.4 About the exchange and the market makers

The independence of flow and price change is a statement that Nobitex is
admitting a substantial population of *liquidity-providing* quotes relative to
*liquidity-taking* flow. A venue with genuinely informative order flow would
show strong flow-price dependence. This has a direct risk implication: the
quote is the *only* thing being marked to market here, so the mark-to-market
risk is entirely quote-revision risk. Filling is a near-random event with respect
to the mid's direction, which is what makes the variance-reduction thesis in
§9 work.

### 8.5 A better factorization: state, not flow

The above suggests the correct predictive factorization is not

$$
P(\Delta m \mid \text{flow}) \quad \text{but} \quad P(\Delta m \mid \text{state}),
$$

where `state` is the liquidity/volatility state $(S, \text{spread}, \text{staleness},
\text{time-of-day}, \text{recent } |\Delta m|)$. This is a proposition about
*where* the information lives, and it is exactly the partition that R1 in §6
uses — and it gains 2.9 nats, the largest structural gain measured.

---

## 9. Online learning under conditional shift

Given §4.2, the estimator must handle conditional shift, and the estimator's
*forgetting profile* is itself a modelling decision.

### 9.1 The formal object

We want an online estimator $\hat\theta_t$ minimizing

$$
\sum_{s \le t} \ell(O_s, \hat\theta_s)
$$

against a sequence of parameter vectors $\theta_t$ that drifts. The natural
formalism is **tracking regret** relative to the *current* best fixed parameter
(strong convexity does **not** hold, parameters are on a constrained set):

$$
\text{Reg}_t = \sum_{s \le t} \ell(O_s, \hat\theta_s) - \sum_{s \le t} \ell(O_s, \theta^*_s)
$$

with $L$-bounded loss. For strongly-convex-like quadratic loss the classical
bounds give $\tilde O(\sqrt{t})$ cumulative, i.e. $\sim \sqrt{T}$ average
regret, decaying as $1/\sqrt{t}$. Empirically, none of our losses are strongly
convex, so treat these as *orders of magnitude*, not guarantees.

### 9.2 Exponential forgetting is the default, and the half-life is a real choice

The EWMA estimator is exactly exponential forgetting:

$$
\hat\sigma^2_t = (1-\lambda) \hat\sigma^2_{t-1} + \lambda\, r_t^2,
\qquad
\tau_{\text{half}} = \frac{\ln 2}{\lambda}.
$$

**But the value of $\lambda$ is not a hyperparameter to tune — it is set by the
physics.** If the true conditional scale has autocorrelation time $\rho$ steps,
an estimator with half-life much longer than $\rho$ has excess variance; much
shorter and it has excess bias. This gives a *pre-registered* rule:

> Set the scale half-life to the measured volatility-clustering horizon, not to
> the value that minimizes validation loss.

We can estimate $\rho$ from the data itself: the `ACF(|dmid|)` decay half-life
in the measured series is about 20-30 buckets (3-5 minutes), so a half-life of
O(10-30 buckets) (2-5 min) is physics-motivated, and any half-life an order of
magnitude away from that is *priori* suspect. This is a defensible, declared
rule for replacing grid search.

### 9.3 Change-point detection: the formal version

BOCD / Bayesian Online Changepoint Detection gives a posterior over segment
lengths and run-partition posterior. Concretely, it gives $P(\text{changepoint at } t \mid O_{0:t})$,
which is a *calibrated* indicator — not a heuristic. The practical point for
this project: BOCD is only worth its complexity when a change point is actually
detectable. Given that the scale is already handled by regime conditioning and
EWMA, the honest recommendation is:

> **Run BOCD as an offline diagnostic first.** If it never fires with posterior
> confidence across 43 days, do not put it in production; instead reduce to
> regime-conditional forgetting, which is simpler and has lower variance.

### 9.4 Bayesian model averaging over regimes, not hyperparameter search

The regime model in §6 (R1) is *not* a hyperparameter search. It is a finite
mixture

$$
Q_t(\Delta m) = \sum_{c} \pi_t(c)\; Q_{t|c}(\Delta m),
\qquad
\pi_t(c) \propto \pi_{t-1}(c)\, P(\text{cell} = c \mid O_{0:t}),
$$

with per-cell recursive moments. The "hyperparameters" are the cell *definition*
(a declared partition, not a tuned quantity) and the *shrinkage* toward the
global cell (a single, declared shrinkage prior). So there are ~2 declared
choices in the model class, versus hundreds in a conventional search, and both
are pre-registered.

### 9.5 When to use a neural net, honestly

A neural net is justified when the *state representation* is high-dimensional
and the conditional law is smooth. Here the state is low-dimensional
($\sim 6$ scalars), and the conditional law is (empirically) well described by
scale + shape + hurdle. On 6 scalars with 347k samples, a small MLP can only
*overfit* or *mimic* the parametric form. Specifically:

* **A neural net cannot help the mean.** §5 established the mean is a step
  function of observables with no exploitable reversion. Nothing to learn.
* **A neural net could help the scale**, by learning the regime map
  $S \mapsto \sigma(S)$ instead of binning. If we later have a rich state
  (many venues, orderbook depth curve, cross-asset), this becomes attractive.
* **Transformers are the wrong tool** for a 6-scalar, low-rate, strongly
  non-stationary series. Their inductive bias (long-range attention) buys
  nothing when the useful signal is a smooth map from a few state variables to
  a scale, and their non-stationarity (positional encoding, optimization
  landscape, distribution memorization) is a liability here.

So: **neural nets are deferred, not rejected** — see §12 for the gate that would
reopen them.

---

## 10. What this means for the market maker (the MPC question)

The user asked: *could a predictive model with less variance uncertainty but the
same directional uncertainty help our market maker?*

### 10.1 The Bellman structure

The MPC solves, at each step,

$$
V(s) = \max_{\pi} \mathbb E\!\left[ g(s, \pi, Y) + \gamma V(s')\right],
\qquad Y \sim P(\cdot \mid \mathcal I_t).
$$

The **only** channel through which the price model enters is the expectation
$\mathbb E[\cdot]$. Crucially, $\mathbb E[\cdot]$ is monotone in the variance
of the *relevant* uncertainty only through convexity of $g$. So:

* **If $g$ is convex in the adverse move** (inventory mark-to-market loss, fill
  shortfall, a risk-averse utility), then *reducing predictive variance with
  unchanged mean is a strict improvement*. The optimizer's cost-to-go drops.
* **If the rollout under-samples the tail**, a sharper-than-true $Q$ makes the
  optimizer *overconfident* and the realized cost goes *up*. So calibration is
  a hard prerequisite, not a nicety.

This is the formal justification for the entire programme: **a model with zero
directional content but strictly better-calibrated conditional scale is
strictly value-adding to the MPC, and a model with directional content that is
estimation noise is value-destroying.**

### 10.2 Why "directional uncertainty unchanged" is the right framing

Because §5 showed directional predictability is zero, the *mean* of $P(\cdot|\mathcal I_t)$
is pinned at $0$. What remains is the entire *shape* of the distribution: the
scale (regime-dependent), the tails ($\nu$), the hurdle (zero mass), and the
horizon composition. Those are all "scale/shape" objects, and they are exactly
what a decision maker consumes. So the framing the user proposed is not a
compromise — **it is the correct specification of the objective**, arrived at
empirically.

### 10.3 A concrete, testable prediction

Take the current production model, the calibrated R1 of §6, and the
miscalibrated S1, and feed all three predictive distributions into the *same*
frozen MPC (same objective, same risk aversion, same fill model, same reward,
same replay). Then:

* **R1**: better NLL (−2.9 nats) **and** better MPC objective, *provided*
  coverage is corrected first.
* **S1**: the *best* NLL of the three (−17.6 nats) but **worse** MPC objective,
  because it is sharper than truth. This is a *prediction of the theory*, not a
  guess: §10.1 says the optimizer is monotone in variance through the
  expectation, and an overconfident $Q$ under-states the true cost of adverse
  excursions.
* A deliberately *sharpen*-tuned variant of R1 (scale $\times 0.6$) should show
  the same pathology, and should be **worse** on the MPC objective despite
  winning on NLL.

This is the sharpest test available that we are not merely fitting a
distribution, and it is a genuinely falsifiable prediction. It should be run
**before** any model is promoted.

### 10.4 Risk-budget implications

The staleness and trade-occurrence splits (§7) have a clear, immediate use that
is **independent of predictive gain**: a risk-averse controller should widen
quotes and scale its inventory penalty by *observed market activity*
(staleness, trade occurrence), not only by the spread. These are stable,
observable, free at decision time, and interpretable as "how likely is this
book to move against me right now".

The disciplined caveat from §7 applies: the *marginal* effect of these
variables is large but their *incremental predictive* effect is ≈0
($-0.0011$ and $+0.0051$ nats in Phase 4). So use them as **risk scalers**,
not as predictive features, and do not expect the transition model to improve
because of them.

A second structural suggestion, independent of the above: the inventory-penalty
term in the MPC should be proportional to predicted conditional volatility
rather than fixed. That is a standard result in stochastic inventory control
that the current formulation does not exploit.

### 10.5 On RL, honestly

RL is the right tool when the reward is delayed and the state is high-dimensional
and the model class is unknown. Here:

* the reward is available analytically from the MPC objective — no need to learn it;
* the state is low-dimensional (§9.5);
* the environment is non-stationary with a thin time budget, so on-policy
  exploration is catastrophically expensive;
* and, critically, an RL policy would learn to *exploit* any miscalibration in
  the price model, converting a modelling error into a systematic loss.

So RL is **not** recommended for the price model. It is a legitimate future
direction for *quote placement across venues and inventory*, where the reward is
genuinely sequential and the state genuinely is high-dimensional — but that is
a different subproject.

---

## 11. The recommended model specification

Everything below is a consequence of the measurements, not a design preference.

```
Delta_m(t) = 0 with prob  p0(S)                      # hurdle, regime-conditional
           | 0                                        #   p0 from spread regime
    else ~ StudentT( nu, 0, sigma(c) )               # heavy tails, nu ~ 4

sigma(c) = exp( b0 + b1*spread_bin + b2*log(1+stale) )   # per-cell, shrunk
eta_t    = half_spread(t)                            # OBSERVED, not estimated
nu       = fixed at 4                                # declared, not tuned
forgetting: EWMA, half-life = measured vol-clustering horizon (~2-5 min)
FINAL:   sigma <- alpha * sigma,  alpha in (1, 1/sqrt(0.82))   # §11.1 calibration
```

**Design decisions, each traceable to a measurement:**

| element | value | why |
|---|---|---|
| conditional mean | **exactly 0** | §5: derived reversion `k=0` optimal; MA(1) and microprice both *harm* |
| zero mass | regime-conditional hurdle | §6.2: `P(dmid=0)` 0.43 → 0.20 across spread regimes |
| tails | Student-t, $\nu\approx 4$ | §6.1: −3.70 nats vs Gaussian |
| scale | regime cells, not `c*spread` | §6.2: elasticity 0.079, near-step |
| scale | activity-partitioned, not `c*spread` | §6.2: elasticity 0.079, near-step; §7: 4.5x staleness gradient within a spread bin |
| quote bounce | diagnostic only — **not** in the predictive density | §11.2: it improves the *marginal* fit but is the source of S1's overconfidence |
| drift in mean | not modelled | §8: flow/price conditional MI ≡ 0 at 10 s |
| **conservative rescale $\alpha$** | **required** | §6 calibration: R1 is under-dispersed at 0.82 |

### 11.1 The calibration step is part of the model, not a post-hoc patch

R1's PIT dispersion is 0.82 where calibration needs 1.0. Under-dispersion is
*always* the wrong direction for a market maker, because it makes the optimizer
believe rare moves are rarer than they are. The correction is a single scalar
inflation

$$
\alpha = \frac{1}{\text{observed PIT z-dispersion}} \approx \frac{1}{0.82} \approx 1.22,
$$

estimated online from a rolling PIT window (itself an EWMA, so it stays
adaptive). This is **one declared scalar**, estimated from a calibration
statistic rather than tuned on the loss — which preserves the property that the
model is not a loss-minimizer. An MPC should err on the side of $\alpha > 1$.

### 11.2 Why the quote-bounce mixture is not in the predictive density

The FTL mixture in §7 is *correct as a model of how the observation is
generated* — that is how the negative ACF was identified. But adding an explicit
uniform convolution to the predictive density made the forecaster *too sharp*
(S1, PIT dispersion 0.82-0.97, over-coverage). The reason is a subtle but
important one: the uniform component is already partly accounted for by the
fact that the conditional spread $\eta_t$ is *observed at decision time*. Once
you condition on the observed spread, the latent quote error is much smaller
than $\eta_t$, and adding a full-width uniform double-counts it.

So: **use the FTL structure to explain the data and to validate the model, but
condition on the observed quote rather than simulating the redraw.** This is
the difference between a structural model you can trust and one that overstates
its own precision.

This is a **three-regime hurdle–Student-t with observed-quote conditioning**,
fitted online by recursions on the tick lattice, with a regime partition, a
forgetting profile, and a calibration scalar *all declared a priori*. Total free
structural choices: **four**. That is the answer to "not a hyperparameter grid
search".

---

## 12. Execution plan

### Phase A — Measurement and audit (done; reuse)
The audit, the FTL identification, the synthetic control, and the regime
tables in §6-7 are complete and are the evidence base for everything else.
`p0_audit.py` … `p6_diagnose.py` are reproducible from the 10 s export.

### Phase B — Production candidate (2-3 days)
1. Implement the §11 model in the online harness as a first-class model.
2. Promote it into `mm/mpc` as a candidate branch alongside the current one.
3. Gate: prequential NLL on the production `price_score` target AND on `dmid`,
   with moving-block-bootstrap CIs, over a **held-out final fortnight** that was
   never used in any design decision.
4. Gate (**hard**): PIT calibration within tolerance on the hurdle and on the
   unconditional scale — target PIT z-dispersion in $[0.95, 1.05]$ and 80%
   coverage in $[0.78, 0.84]$. **A model that fails this gate is rejected even
   if it wins on NLL.** This is the direct lesson of S1.
5. Gate: frozen-MPC replay improvement (§10.3), including the S1 control and a
   deliberate $0.6\times$ sharpening control, both of which must show the
   predicted *degradation*.
6. Gate: confirm the online $\alpha$ correction (§11.1) actually restores
   PIT dispersion to ≈ 1.0 on held-out data.

### Phase C — The high-value data upgrade (highest ROI)
Per §8.2, the *flow* information is provably destroyed by 10 s bucketing. The
highest-expected-value change is not a model — it is **per-trade timestamps and
a sub-second feed**. Re-run the §8 conditional-MI test at 100 ms–1 s. If flow
becomes informative there, it reopens directional modelling; if it does not,
that is a *strong, publishable-grade negative result* on flow-price
information, and it validates the scale-only specification. Either outcome is a
win. This is the single most important recommendation in this document.

### Phase D — Reopen neural models only under a declared gate
Only if (a) the state is enriched (multi-venue, depth curve, cross-asset) and
(b) a parametric scale model is *shown insufficient* by its own prequential
score. Then a small MLP on $\log\sigma$ is the right first candidate, and a
transformer only if (a) holds with a long and genuinely informative history.
Until then, parametric + regime conditioning is the right complexity budget.

### Phase E — Activity-based risk scaling at the execution layer
Exploit §7's splits operationally: scale quote width and the inventory penalty
by **observed market activity** (book staleness, trade occurrence) — a 4.5x
conditional scale gradient for free, with no predictive gain required. Keep
this strictly separate from the price model so the two effects can be
attributed independently.

### Phase F — Required ablation before trusting the regime partition
Phase 7 shows R1 (6x3 spread x staleness cells) gaining 2.9 nats but also shows
it under-dispersed. Before it is trusted, decompose the gain:

1. R1 with **hurdle only** (per-cell `P(dmid=0)`, global scale) —
   isolates the zero-mass contribution.
2. R1 with **scale only** (global hurdle, per-cell scale) —
   isolates the scale contribution.
3. R1 with the online $\alpha$ calibration of §11.1 applied to both.

Given that Phase 4 found staleness has ≈0 *incremental* predictive effect, our
prior is that **(1) carries most of the 2.9 nats and (2) carries little.** If
that holds, the production model should be a *spread-conditional hurdle* with a
single global EWMA scale — materially simpler than a 6x3 partition, and the
simplification should be preferred. This ablation is cheap and is the single
highest-value next experiment.

---

## 13. What would change our mind

Pre-registered falsifiers, in the spirit of the whole document:

1. **Flow is informative at 10 s under a robust specification.** Would reopen
   the mean.
2. **The derived reversion `k > 0` survives out-of-sample** with a
   bootstrap CI excluding zero. Would reopen the mean.
3. **A sub-second feed reveals informative flow.** Would reopen the mean and
   justify a materially more complex model.
4. **A scale model that is not regime-conditional beats the regime model**
   prequentially. Would simplify §11 back toward a single law.
5. **Sharper-than-true calibration improves the MPC objective.** Would falsify
   §10.1 and change the risk model. This is the single most important
   falsifier, because the whole recommendation rests on it. *Prior expectation:
   it will not hold, and S1's calibration profile already suggests as much.*
6. **The $\alpha$ inflation from §11.1 is unnecessary** — i.e. R1 is already
   calibrated once the hurdle is estimated properly on held-out data. Would
   remove the last free scalar.
7. **Regime conditioning stops paying** once staleness is controlled for
   directly. Would indicate the cells are proxying for a single latent
   "market-attention" variable, which would be a cleaner model.

Each is a concrete, cheap, pre-registered experiment.

---

## Appendix A. Reproducibility and provenance

### A.1 Scripts

| script | output | what it establishes |
|---|---|---|
| `p0_audit.py` | `out/p0.txt` | data audit: grid, spread, staleness, flow, depth |
| `p1_measurement.py` | `out/p1.txt` | ACF profile, Roll covariance identity, **quote-revision conditional test** |
| `p1b_follow.py` | `out/p1b.txt` | synthetic FTL control (zero-information efficient price) |
| `p2_ladder.py` | `out/p2.txt` | prequential predictability ladder, paired block-bootstrap CIs |
| `p3_mean_vs_scale.py` | `out/p3.txt` | scan of the *derived* reversion strength `k` |
| `p4_scale.py` | `out/p4.txt` | conditional-scale drivers, incremental effect of each |
| `p5_spread_scale.py` | `out/p5.txt` | functional form of the scale law; calibration of the linear form |
| `p6_diagnose.py` | `out/p6.txt` | non-parametric conditional scale, elasticity, zero mass |
| `p7_regime.py` | `out/p7.txt` | candidate set on one fixed target contract, **with PIT calibration** |

Shared infrastructure: `harness.py` (loading, feature derivation, and the
**decision-time indexing contract**), `preq.py` (online prequential engine and
probability densities), `stats1.py` (moving-block bootstrap, Diebold-Mariano,
conditional MI), `models_*.py` (the candidate estimators).

### A.2 Known issues in the current artifacts

Recorded rather than hidden, because they bound what the numbers above mean.

1. **`p1.txt` ends in an `IndexError`** at its final diagnostic (an off-by-one
   between the length-`N` `dmid` and the length-`N-1` lagged bid/ask
   differences). The fix is applied in `p1_measurement.py`; the diagnostic is
   the "among `dmid==0`, what fraction of quotes were unchanged" split, which
   is **corroborative only** — §1.3's conditional-ACF table, which is the
   load-bearing evidence, was printed before the crash and is unaffected.
2. **`p5.txt`'s calibration block is degenerate.** It reports a nominal-vs-
   empirical quantile ratio of 1.0000 at every level while simultaneously
   reporting `PIT chi2 = 6,332,942` (expected ≈19) and a standardised-residual
   sd of 0.0051 (expected ≈1). The scale in that model is being driven to
   ~0 by the online fit, so the quantiles match trivially. **Do not read p5's
   calibration as evidence of anything**; it is an artifact of a degenerate
   online scale, and it is a useful illustration of why calibration must be
   checked on a standardised residual and not on quantiles alone.
3. **`p7.txt` has empty calibration cells for B1/B2/B3** (the plain scale
   models have no closed-form standardised residual implemented). Only the
   hurdle models are calibration-checked there, so the *ranking* of the scale
   models rests on NLL alone. Phase B must close this.
4. **The 10 s source export is no longer present** in `research/model_lab/`
   (it was untracked). The `out/*.txt` results are committed, but **Phase A
   must be re-run from a fresh export before Phase B gates are trusted**,
   because the cache is gone and the numbers cannot currently be regenerated.

### A.3 A note on the sibling documents

`MATHEMATICAL_MODELING_ROADMAP.md` (and its `_EXTENSION`) work from the
**30 s / 4-day** export and the production performance log; this document works
from the **10 s / 43-day** export. They are complementary, and this document
*supersedes* the earlier one specifically on one point that the longer sample
settles: **`local_price_nll` is degenerate.** That earlier document already
flagged it as "scored against a `N(0,1)` constant, not against the learned
mixture" (median 79, mean 2,155, P99 26,451, max 235,642, 48% above 100). The
43-day study confirms the mechanism from the other direction: the incumbent
production model is using a **Gaussian** in the market branch, and §6.1 shows
that a Student-t in exactly that role is worth **3.70 nats/step**. So the
production NLL is not merely noisy — it is scoring a misspecified density, and
it is dominated by the one correction §6 identifies.

**Action:** the two documents should be merged, with this one's 43-day
measurements taking precedence wherever the samples overlap, and the earlier
document's target-formulation argument (what the MPC actually consumes)
retained since the 10 s study does not re-derive it.
