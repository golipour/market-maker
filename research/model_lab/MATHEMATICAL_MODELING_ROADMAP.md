# A Mathematical Roadmap for the Market-Maker Transition Model

> **What this document is for.** A first-principles development of the prediction problem
> we are actually trying to solve. It does not propose a hyperparameter grid, does not
> rank off-the-shelf models, and does not assume that "lower NLL on this CSV" is the
> objective. It tries to answer, in order:
>
> 1. *What is the object we are predicting, formally?*
> 2. *What does "prediction" cost (entropy rate) and how much of it can we actually remove?*
> 3. *What is the right joint factorisation of the target, given the data?*
> 4. *What does "online" mean as a mathematical constraint, not just an engineering rule?*
> 5. *Which families of estimators are consequences of (1)–(4), and which are just menus?*
> 6. *What does the market-maker downstream actually need from this prediction layer?*
>
> Every claim that should be checked against data is checked against the supplied
> four-day extract (`nobitex_USDT_IRT_30000ms_4d.csv`) and the matching production
> performance log (`..._model_performance_and_params.csv`). Numbers below were computed
> directly from those files; they are not inherited from prior documents.
>
> This document is deliberately upstream of `GUIDE.md` and complementary to
> `THEORETICAL_FOUNDATIONS.md`, `THEORETICAL_ROADMAP.md`, `MATHEMATICAL_ROADMAP.md`.
> It supersedes those only in places where the data audit reveals a contradiction.

---

## 0. Epistemological stance

The dominant temptation in this kind of work is to read off a leaderboard and call the
top entry "the best model." That conflates *competitive fit on this benchmark* with
*the right object for the market maker*. The two coincide only by accident. Three
positions follow.

**Position A — the target is not what we think.** What we are scoring today (per-bucket
volume and per-bucket mid-delta) is a *projection* of what the controller consumes
(VWAP paths, executable depth, markout distributions, fill probability under intervention,
inventory risk conditional on a quote set). A model that wins NLL on the projection
can lose MPC utility on the consumption object. The right target is a jointly defined
**decision-time conditional law of the next controlled transition**, not a marginal
density of a single bucket quantity.

**Position B — NLL is one number from a vector.** A model that achieves NLL 13.0 but
fails PIT calibration in the regime where the market maker holds inventory is not
"better" than one that achieves NLL 13.2 and is conditionally calibrated everywhere.
NLL is a strictly proper score *in expectation under the true law*; it is a necessary
diagnostic, not a sufficient ranking.

**Position C — the irreducible uncertainty is finite and estimable.** For a stationary
ergodic process the entropy rate \(h\) is a hard floor. For a non-stationary one we
cannot define \(h\) globally, but we can still ask, in any epoch, how close we are to
the local floor. The gap is the *amount of predictive information that remains
unharvested*, measured in nats per step. That number is the only honest way to bound
the size of the prize.

The remainder of the document expands each of these in mathematical form, anchored to
the data we actually have.

---

## 1. The data as it is — empirical surface

Source: `nobitex_USDT_IRT_30000ms_4d.csv`, 8700 rows, 2026-07-08 07:14:30 UTC →
2026-07-11 09:15:00 UTC, 30-second buckets.

| Diagnostic | Value | Reading |
|---|---:|---|
| regular 30s transitions | 8677 | nominal cadence preserved |
| gaps (60s / 90s / 420s / 3960s) | 14 / 4 / 2 / 1 | 22 non-canonical intervals |
| zero-flow buckets (total) | 4.60 % | much sparser than the GUIDE's "59 %"; period matters |
| zero-buy / zero-sell | 13.78 % / 17.00 % | sell side more often empty |
| positive total-flow: median / P90 / P99 / max | 441 / 2386 / 8958 / 39 123 USDT | heavy right tail |
| positive buy / sell median | 212 / 167 USDT | buy is larger when it occurs |
| mid-change mean / sd | −0.12 / 65.73 IRT | centred near zero |
| mid-change zero rate | 14.62 % | tick-quantised |
| mid-change skew / excess kurtosis | −0.355 / **7.18** | not Gaussian; heavy tails |
| Hill estimator \(|ΔM|\), top-k ∈ {50, 100, 200, 500, 1000} | ξ = 0.22 / 0.23 / 0.26 / 0.29 / 0.35 → α ≈ 4.6 / 4.3 / 3.9 / 3.5 / 2.8 | power-law tail with **α ∈ (3, 5)**: heavier than Gaussian, lighter than Cauchy; Student-t / NIG territory |
| Hill estimator total flow, top-k ∈ {50, 100, 500, 1000} | ξ = 0.31 / 0.36 / 0.55 / 0.62 → α ≈ 3.2 / 2.8 / 1.8 / 1.6 | tail **heavier than \(|ΔM|\)**; close to log-normal tail regime |
| ACF(ΔM)[1, 2, 5, 10, 20, 60, 120] | −0.238, −0.002, 0.012, 0.003, −0.018, 0.012, 0.011 | Roll bounce lives at lag 1, **memory dies by lag 2** |
| ACF(\(|ΔM|\))[1, 5, 20, 60, 120] | 0.229, 0.134, 0.066, 0.021, 0.014 | vol clustering decays in ~hours |
| ACF(total-flow)[1] | 0.287 | flow persists at lag 1 |
| ACF(spread)[1] | 0.516 | spread is highly persistent — slow regime |
| rolling 1h variance of ΔM: min / median / max | 283 / 3624 / 14 705 (ratio 52×) | conditional scale varies by 1–2 orders of magnitude |
| corr(buy, sell) | 0.148 | weak co-movement |
| corr(net-flow\(_t\), ΔM\(_t\)) | 0.032 | contemporaneous residual |
| **corr(net-flow\(_t\), ΔM\(_{t+1}\))** | **0.011** | **no forward information** |
| **corr(net-flow\(_{t-1}\), ΔM\(_t\))** | **0.171** | **backward mechanical-impact signature** |
| corr(\(|net_flow|\), \(|ΔM|\)) | 0.062 | weak magnitude coupling |
| quintile of net-flow\(_{t-1}\) → mean(ΔM\(_t\)) | −0.4, −2.5, −1.7, +0.3, **+3.7** | monotonic: prior bucket's flow moves *current* bucket's mid |
| quintile of \(|net_flow|_{t-1}\) → mean(\(|ΔM|_t\)) | 35.9, 36.0, 36.2, 37.6, **42.2** | weak scale coupling |

These numbers are not "interesting" by themselves. They constrain what any model can
claim. The most important facts:

1. **The forward flow → price signal is dead.** Anything claimed about "predicting next
   price from current flow" is at best a model of the *fraction of a bucket* between
   arrival and end-of-bucket settlement.
2. **The backward flow → price signal is alive** at lag 1, with magnitude rising
   monotonically with quintile. This is *book-walking impact*, already encoded in the
   mechanical-consumption step of the MPC. Its predictive information content is
   already consumed inside the simulator, **not** at the price-NLL layer.
3. **Volatility is the only dimension along which price is predictable.** ACF(\(|ΔM|\))
   is positive and slow; this is a GARCH phenomenon, not an autoregressive-mean
   phenomenon.
4. **Tails are heavy but not extreme.** α ∈ (3, 5) for \(|ΔM|\), (1.6, 3) for flow. These
   values determine the **right innovation family**: Student-t, Normal-Inverse-Gaussian,
   or symmetric-α-stable (the latter only if α < 2, which it is not). Gaussian is
   decisively wrong.
5. **Spread is a slow regime variable.** ACF(spread)[1] = 0.516 with mean ~50 IRT and
   P99 = 394 IRT — the wide-spread tail is genuinely long-lived. The data has at least
   two structural regimes: tight quoting and wide quoting.

The CSV does not carry the full order book, only best bid/ask and quantity. Anything
deeper requires `mm_events` from Postgres — a hard prerequisite, not an optional
enrichment. (See §14.)

---

## 2. What the production score actually says

The performance CSV is the only artefact in the repo that tells us what the live bot
thinks of its own predictions. **Most of it is broken as a measurement.** This must be
stated openly before we use any of it.

| Column | Behaviour observed | Interpretation |
|---|---|---|
| `local_market_flow_params_before_mu_buy` | evolves smoothly from 0.0 → 5.1 → 6.4 → back to 0.0 (one reset) | flow μ update is real EWMA |
| `local_market_flow_params_before_sigma_buy` | evolves 1.0 → 0.6 → 0.5 → … → 1.9; hits **1e-6** repeatedly | flow σ update is real but EWMA **starves to zero** when zero-flow runs dominate |
| `local_market_flow_nll` | median 13.06, mean 13.24, **9 scores = 690.78**, P99 21.46 | real on the body; the 9 worst scores are a **numerical-floor artefact** (probability = 0 in the JSON payload) |
| `local_price_params_before_sigma_mid` | **constant 1.0 over all 8697 rows** | price σ is a placeholder, not the learned value |
| `local_price_params_before_mu_mid` | **constant 0.0** | price μ is a placeholder |
| `local_price_nll` | median 79, **mean 2155**, P99 26 451, max 235 642, 48 % above 100 | degenerate: scored against N(0, 1) constant, not against the learned mixture |
| `local_price_skip_reason` | only "missing_previous_frame" × 6 | otherwise scoring pipeline runs |
| `local_market_flow_skip_reason` | none | flow scoring is complete |

### 2.1 What this means for the research programme

1. **The recorded flow NLL is a real but contaminated number.** Median 13.06 is
   informative; the 9 worst scores are *not* telling us "the market surprised the
   model by 690 nats" — they are telling us "the model's PMF code returned 0 because
   the standard-lognormal density at the observation was below the floating-point
   floor." Every one of those rows in the worst-20 list has `prob_buy=0.0` or
   `prob_sell=0.0` in the JSON payload. These are **scoring-pipeline failures**, not
   prediction failures.
2. **The recorded price NLL is meaningless.** With σ_mid always 1 and ΔM sometimes
   ±500 IRT, the score is \(\tfrac{1}{2}(500/1)^2 \approx 1.25 \times 10^5\). The
   "production price model" cannot be evaluated from this file; it must be re-run
   with the correct parameters before any inference.
3. **EWMA starvation is a real production failure mode.** σ_buy = 1e-6 occurs when
   a long sequence of zero-flow buckets arrives; the EWMA on log-variance collapses
   to its floor. The next non-zero observation is then scored at log-PMF
   ≈ −∞. This is a fixable algorithmic defect, but it must be fixed *before*
   any architectural change.

> **Action.** Before any new model is written, fix two things in the scoring pipeline:
> (i) replace the implicit PMF floor with a calibrated tail (e.g. Student-t log-PDF
> with a ν floor), and (ii) record the actual `params_before` for the price model
> instead of a placeholder. Until then, *the only usable diagnostic in the performance
> CSV is the body of the flow NLL, not its mean and not its tails*.

---

## 3. The mathematical object

We work on a filtered probability space \((\Omega, \mathcal{F}, (\mathcal{F}_t)_{t\ge 0},
P)\). At the end of bucket \(t\), the information set available at the next quote is
\(\mathcal{I}_t = \sigma(O_{0:t}, A_{0:t-1}, E_t)\), where

* \(O_t\) is the union of all exchange- and receipt-timestamped events that arrived
  before the close of bucket \(t\): best bid/ask, depth at each tick, last-trade print,
  and the reference venue state if used;
* \(A_t\) is our action: outstanding bids/asks at each tick, with queue position
  wherever known, inventory, fees, latency budget;
* \(E_t\) is a data-quality flag: gap, book-stale, snapshot-reconstruction mode,
  clock-skew, exchange outage.

The observation in the lab CSV is a *projection*

\[
x_t = g(O_t) = \bigl(V_t^+, V_t^-, M_t, S_t, \mathcal{B}_t^{\text{top}}, \tau_t\bigr),
\]

with the book reduced to (best_bid, best_ask, quantities) and τ_t the time-of-day
cyclic encoding. **The information used by the live bot is strictly larger than the
information recorded in the CSV.** The model lab is therefore evaluating models on a
*projected* filtration. This is acceptable for benchmarking, but it bounds the
*absolute* entropy that can be removed: any information in \(O_t \setminus x_t\)
cannot be recovered from the CSV alone.

### 3.1 The prediction kernel

The live model emits a probability kernel

\[
Q_t(\cdot) = Q(\cdot \mid \mathcal{F}_t;\,\theta_t),\qquad
\theta_{t+1} = U(\theta_t, x_{t+1}, A_t),
\]

with the scoring constraint that the score of \(x_{t+1}\) is computed against \(Q_t\)
*before* the update is applied. The walk-forward loop in `core/walk_forward.py`
implements this. The point is non-trivial: a model that ingests \(x_{t+1}\) and then
scores it against a posterior is a leak, not a prediction.

### 3.2 The interventional caveat

If our quotes change the displayed book or the fill probability, the *observational*
distribution \(P(x_{t+1} \mid \mathcal{F}_t, A_t = a)\) is not the same as the
counterfactual \(P(x_{t+1} \mid \mathcal{F}_t, \mathrm{do}(A_t = a))\). For a small
maker on a liquid local venue the distinction is approximately negligible; for any
maker that regularly leans on one side it is not. The current model lab does not
condition on \(A_t\) at all. **That is a real, identifiable misspecification** that
should be measured before claiming a model is "decision-sufficient" — i.e., that its
predictions are good enough to drive an action.

### 3.3 The target as a joint outcome

What the MPC consumes is not a single number but a path:

\[
Y_{t,h} = \bigl\{(V^+_{t+j}, V^-_{t+j}, M_{t+j}, S_{t+j}, \mathcal{B}_{t+j})\bigr\}_{j=1}^{h}.
\]

A simulator that produces credible paths of \(Y_{t,h}\) is the *operational* object.
The two marginal scores the lab currently reports — flow NLL on \((V^+, V^-)\) and
price NLL on residuals — are factors in a *valid* chain rule

\[
-\log Q(Y_{t+1}) = -\log Q(V^+, V^- \mid \mathcal{F}_t) - \log Q(\Delta M, \Delta S, \Delta\mathcal{B} \mid V^+, V^-, \mathcal{F}_t).
\]

But **the factorisation is exact only if** \(\Delta M\) and \((V^+, V^-)\) are
conditionally independent given \(\mathcal{F}_t\) *and the realised flow of the
*current* bucket*. The data (§1) is consistent with the much weaker condition that
the *next*-bucket flow and next-bucket mid change are conditionally independent given
\(\mathcal{F}_t\) — the price has already incorporated the bucket's flow by close.
That is the factorisation the live MPC uses; it is defensible at the *transition*
level but not at the marginal level. See §6.

---

## 4. Score rules

The logarithmic score \(S(Q, x) = -\log Q(x)\) is the only **local proper score**:
its expectation under the true distribution \(P\) is uniquely minimised at \(Q = P\).
The CRPS \(\int (F_Q(y) - \mathbf{1}\{y \ge x\})^2 dy\) is proper but not local; the
energy/variogram score is its multivariate generalisation; the interval/quantile score
is proper but only for the targeted quantile.

For an *online generative* model the right primary score is the log score. The reasons
are not aesthetic but operational:

* **strict properness** ensures that the empirical leaderboard is monotone in the
  KL divergence to the truth;
* **decomposability over products** means the joint log score is the sum of the
  marginal log scores under any factorisation;
* **sharpness is rewarded only when paired with calibration**: a sharper \(Q\) that
  is wrong gets a worse log score, exactly as desired.

But the log score has four known problems:

1. **Tail dominance.** A single near-zero probability observation dominates the mean.
   The 9 / 8697 worst scores in the production file are responsible for the entire
   gap between the *median* flow NLL of 13.06 and any future "improved" mean. **Always
   report mean and median; report P99 and max separately.**
2. **Score-direction confusion.** Minimising log score *under the wrong metric* can
   reward a model that is sharp but miscalibrated. Calibration must be reported
   alongside.
3. **Unit sensitivity.** Continuous log scores depend on the choice of units. Two
   models on ΔM in IRT vs. on log returns cannot be compared without the change of
   variables. **The target and units must be frozen before any model is compared.**
4. **Discretisation loss.** If the recorded flow volume is integer lots and the
   model emits a continuous lognormal density, the score is the density evaluated at
   the integer plus a Jacobian (the bin width). The CSV carries fractional volumes
   like 585.54 that are rounded to 586 in the score; this rounding is part of the
   probabilistic model and must be made explicit.

> **Recommendation.** Use log score as the primary statistical objective. Add a
> *calibration vector* of mandatory diagnostics: randomised PIT (or rank histogram)
> for the discrete components; PIT and interval coverage for the continuous
> components; energy score for the joint; CRPS as a robust univariate diagnostic;
> a *tail-weighted* log score (observations in the worst 5 % of conditional
> predicted quantile) for the parts the MPC actually fears. A model that improves
> log score by becoming less calibrated is *not* an improvement.

---

## 5. The entropy floor

The best attainable per-step NLL on a stationary ergodic process is the **entropy
rate** \(h = \lim_T H(x_{T+1} \mid \mathcal{F}_T)\). For a non-stationary process
\(h\) is not a single number; the best attainable on an epoch of length \(W\) is

\[
h_W = \frac{1}{W} \sum_{t=1}^{W} H(x_{t+1} \mid \mathcal{F}_t),
\]

which is itself estimated from data and is uncertain. The point of the floor is
*not* its precise value but the bound it puts on improvement.

### 5.1 What the data says about the floor

A back-of-envelope estimate, with all assumptions stated:

| Component | Family assumed | Estimated conditional entropy | Notes |
|---|---|---:|---|
| ΔM, marginal | Student-t, ν=4, σ=65 | ≈ 4.20 nats | heavy-tail innovation; lower bound under independence |
| ΔM conditional on ΔM\(_{t-1}\) (AR(1)) | t-innovation | ≈ 4.16 nats | lag-1 mean captures a few percent of variance |
| ΔM conditional on (ΔM\(_{t-1}\), spread quintile, imbalance sign) | t-innovation per cell | ≈ 4.05 nats | regime helps marginally |
| Buy volume, positive part | Lognormal, μ=5.5, σ=1.5 (matched to data) | ≈ 4.6 nats | continuous part |
| Buy volume, full hurdle | Bernoulli × Lognormal | ≈ 0.6 + 0.86·4.6 ≈ 4.6 nats | hurdle reduces effective entropy because zero is a single point |
| Buy volume conditional on V\(^+_{t-1}\), spread | Hurdle-lognormal with state-dependent μ | ≈ 4.4 nats | small gain from conditioning |

So the *floor* on each marginal is somewhere between **4 and 5 nats per step**, and
the *gain* over a marginal hurdle-lognormal achievable by adding state-dependent
parameters is **0.1–0.3 nats per step** for each component. The total gain over a
well-calibrated hurdle-lognormal is **0.5–1.0 nats per step across the joint target**
at best. A transformer cannot beat this. A neural density estimator cannot beat this.
A better *online update* might.

> **The single most important empirical bound of this document.** Any future
> "improvement" of more than ~1 nat/step over a calibrated hurdle-lognormal with
> state-dependent μ is suspect. The only way to win more is to:
> (a) widen the target (use a more decision-relevant outcome), or
> (b) exploit information the lab CSV does not carry (full book, reference venue, queue).

### 5.2 Why this is not the same as "the model cannot improve"

The floor above is for *marginal* targets. The market maker's PnL is a function of
*joint* path outcomes: a sharp fill-probability × markout distribution × inventory
trajectory. Even if every marginal component is near-optimal, a *jointly miscalibrated
model* can be improved by a factor that does not appear in any marginal NLL. This
is the §6 question: is the factorisation valid?

---

## 6. The factorisation question

The current production score factorises as

\[
\log Q(x) = \log Q(V^+, V^-) + \log Q(\Delta M, \Delta S, \Delta\mathcal{B} \mid V^+, V^-).
\]

This is exact iff \(\Delta M_{t+1}\) is conditionally independent of \(V^+_{t+1},
V^-_{t+1}\) given \(\mathcal{F}_t\) *and the realised flow of the same bucket*. We
test the weaker, weaker, weaker versions of this:

### 6.1 H1: \(V^+ \perp V^- \mid \mathcal{F}_t\)

**Test statistic**: corr(buy, sell) = 0.148 over the 4-day sample. **Not** a
conditional-independence test, but informative. A strict test would bin by spread
quintile × volume tercile and check whether the conditional cross-correlation is
zero in each cell; we expect residual co-movement in cells with large flow, where
informed participants may hit both sides.

**Verdict**: weakly rejected. The independent-side assumption is not exactly wrong
but is *biased*: under one bucket per 30 s the joint buy/sell emission captures
~1 % of total variance via co-movement. **Whether this matters for the MPC is an
open question; the price side is the more important calibration target.**

### 6.2 H2: \(V_{t+1} \perp \Delta M_{t+1} \mid \mathcal{F}_t\)

**Test**: corr(net-flow\(_t\), ΔM\(_{t+1}\)) = 0.011, with 95 % CI roughly ±0.022
under a HAC estimator. **Cannot reject the null at any reasonable level.**

But: corr(net-flow\(_{t-1}\), ΔM\(_t\)) = 0.171, monotonically increasing across
quintiles. So **flow at \(t-1\) predicts ΔM\(_t\)**, but flow at \(t\) does *not*
predict ΔM\(_{t+1}\). The mechanism is straightforward: by the time the bucket closes,
the price has already absorbed the bucket's flow via within-bucket mechanical
consumption. The signal is *backward* (cause-effect, where cause is the previous
bucket's flow and effect is the current bucket's mid move), not forward.

**Verdict**: the factorisation \(P(\Delta M \mid V_{\text{current}}) \cdot P(V)\) is
justified at the transition level for the next-step joint target. This is a deep
property of the market, not a modelling choice.

### 6.3 H3: \(\Delta M_t \perp \Delta M_{t+k} \mid \mathcal{F}_t\) for \(k \ge 2\)

**Test**: ACF(ΔM)[2] = −0.002, [5] = 0.012, [10] = 0.003, [20] = −0.018, [60] = 0.012,
[120] = 0.011. **Cannot reject serial independence for k ≥ 2 at conventional levels
with a 4-day sample.**

But: ACF(\(|ΔM|\))[2] = 0.169, [5] = 0.134, [20] = 0.066. The *conditional scale* is
predictable up to about 1 hour; the *conditional mean* is not.

**Verdict**: the second-order structure is a GARCH phenomenon, not an AR phenomenon.
A model that captures conditional variance (volatility) without conditional mean is
*asymptotically* mean-reverting to the unconditional mean, which is what the data
shows.

### 6.4 H4: \(V_{t+1} \perp \mathcal{F}_t\)

**Test**: ACF(total-flow)[1] = 0.287, ACF(buy)[1] = 0.272, ACF(sell)[1] = 0.164.
**Decisively rejected.** Flow has memory at lag 1 (and probably more). The current
hurdle-lognormal with EWMA on (μ, σ) does capture *some* of this through the drift
of μ and σ, but the EWMA on the parameters is a single-timescale smoother and
under-weights short-run bursts.

> **Implication for the model class.** A model that ignores
> the autocorrelation of flow is leaving 0.1–0.3 nats per step of predictive
> information on the table — small but real. A first-order autoregressive term on
> the hurdle mean (a "soft" AR(1) on the latent log-volume intensity) is the minimal
> fix; a Hawkes process on event arrivals is the maximal one.

### 6.5 The full factorisation that follows

Putting §6.1–6.4 together:

\[
P(x_{t+1} \mid \mathcal{F}_t) = P(V^+_{t+1} \mid \mathcal{F}_t) \cdot P(V^-_{t+1} \mid V^+_{t+1}, \mathcal{F}_t) \cdot P(\Delta M_{t+1}, \Delta S_{t+1}, \Delta\mathcal{B}_{t+1} \mid V^+_{t+1}, V^-_{t+1}, \mathcal{F}_t),
\]

with the third factor further factorising as

\[
P(\Delta M \mid \cdot) \cdot P(\Delta S \mid \Delta M, \cdot) \cdot P(\Delta\mathcal{B} \mid \Delta M, \Delta S, \cdot),
\]

provided the test in §6.2 holds. The **third factor is essentially mechanical** for
this venue: the price has already moved with the flow inside the bucket, and what
remains at bucket close is refresh noise, snapshot/reconstruction effects, and
reference-venue drift that the local book cannot observe. *This is exactly what the
existing production `conditional_mechanical_mixture` model is trying to capture.* Its
NLL is suspicious for the §2 reasons, but its *structural choice* is right.

---

## 7. Stationarity and the meaning of "online"

### 7.1 A non-stationarity taxonomy

The data is non-stationary in at least four ways. Each demands a different model
response, and the production model responds correctly to only one.

| Type | Definition | Diagnostic in our data | Required model behaviour |
|---|---|---|---|
| I (unit root in level) | \(M_t \sim I(1)\); ΔM is stationary | mid wanders 700 IRT over 4 days; ΔM has sd 65 | model ΔM, not \(M_t\) (the lab already does this) |
| II (slow drift in parameters) | μ_buy(t), σ_buy(t) vary smoothly | mu_buy drifts 5.0 → 6.4 over 4 days; σ_buy drifts 0.5 → 1.9 | EWMA or discounted Bayes on (μ, σ) (the lab already does this) |
| III (structural breaks) | parameter values jump | one row with mu_buy_b = 0 in the middle of the run; rolling 1h variance of ΔM varies 52× | change-point detection; reset posterior on detection |
| IV (regime persistence) | spread quintile, vol quintile | ACF(spread)[1] = 0.516; ACF(\(\|ΔM\|\))[1] = 0.229 | regime filter on observable state (spread, imbalance, time-of-day) |

The production model handles II via EWMA with decay 0.95. It does not handle III
(visible in the production file as a single reset row and as the 1e-6 σ starvation).
It does not handle IV explicitly (although the `conditional_mechanical_mixture` *gate*
on flow volume is a weak form of IV handling).

### 7.2 The tracking regret theorem

For an online learner with the cumulative regret

\[
R_T = \sum_{t=1}^{T} \bigl(\ell_t(Q_t) - \ell_t(Q_t^*)\bigr)
\]

against the best *time-varying* comparator \(Q_t^*\) with total parameter variation
\(C_T = \sum_t \|\theta_t^* - \theta_{t-1}^*\|_2\), the lower bound is

\[
R_T \ge \Omega(T^{2/3} C_T^{1/3})
\]

(Besbes–Gur–Zeevi, 2015). For our data, a rough estimate of \(C_T\) on the flow
parameters is the observed range of μ_buy (≈ 1.5 nats) × number of distinct regime
switches (≈ 4 days × 1 switch/day ≈ 4). With \(C_T \approx 6\), \(T = 8700\),
the bound is \(R_T \ge \Omega(8700^{2/3} \cdot 6^{1/3}) \approx \Omega(1950)\) nats
cumulative — about **0.22 nats/step**. This is the irreducible tracking error of any
algorithm with that variation budget. A more refined model that reduces the parameter
variation (e.g. by detecting regime structure explicitly) reduces the budget and the
bound. A more complex model that does *not* reduce the variation (e.g. a deeper
neural net on the same EWMA features) does not.

### 7.3 What "online dynamic model" actually means

The phrase as used in the production code means: update (μ, σ, p_0) by EWMA after
each observation. The phrase as used in the user's request — and in any
mathematically defensible research programme — means something stronger:

> An **online dynamic model** is a sequence of conditional probability kernels
> \(Q_t\) together with an update \(U\) such that
> (a) \(Q_t\) is a function of \(\mathcal{F}_t\) only;
> (b) the update has *bounded state* (state complexity ≤ some function of the
> data, not the data itself);
> (c) the update is *causally admissible* (uses only \(O_{0:t}\), never
> \(O_{t+1}\));
> (d) the *tracking regret* of the sequence against any causally admissible
> comparator with declared variation budget is bounded.

The current production model satisfies (a)–(c) but not (d), because nothing bounds
its tracking regret against a model that simply resets to the empirical quantiles of
the last 200 buckets. That second model would be a legitimate comparator, and the
EWMA does not provably beat it.

The mathematical constraint (d) is what "online" *should* mean, and it is what
forces a *structured* model class — not because EWMA is wrong but because a
declaration of regret forces you to ask "what class of competitors am I tracking?"

---

## 8. The correct online update — what it looks like

The EWMA update is a degenerate Kalman filter on a single state variable with a
random-walk transition. It has three structural deficiencies:

* **point-estimate only**: the Kalman filter maintains a covariance \(\Sigma_t\);
  the EWMA collapses it to a scalar that is not even exposed to the score;
* **one timescale**: a single decay parameter \(\gamma\) for every state;
* **no support guarantee**: when the data starves one variable to zero, the variance
  estimate collapses to zero, and the model assigns probability zero to anything
  outside its support.

The right upgrade is *not* a deep neural net. It is a Bayesian online update with
three additions:

### 8.1 Discounted conjugate Bayes

For a univariate exponential-family emission with natural parameter \(\eta_t\) and
sufficient statistic \(T(x_t)\), the *discounted* Bayesian update (already
classical, see e.g. Van Erven et al. 2016) is

\[
P(\eta \mid T_{0:t}) \propto P(\eta)^{\gamma^t} \cdot \prod_{s=1}^{t} P(x_s \mid \eta)^{1-\gamma^t}
\]

where \(\gamma \in (0, 1)\) is the *forgetting factor*. The prior decays
geometrically and the likelihood has total weight \(1 - \gamma^t\) on the current
data. The posterior is still conjugate for the right prior; for a Gaussian emission
with Normal–Inverse-Gamma prior, for example, the update has closed form. **The
posterior mean is what EWMA approximates, and the posterior variance is what EWMA
discards.** Discarding the posterior variance is provably suboptimal under any
risk-sensitive loss; under log-loss it is suboptimal exactly when the score is
nonlinear in the parameter.

For our flow model, the lognormal hurdle is approximately log-Gaussian when μ is
moderate, so a Normal–Inverse-Gamma prior on (log μ, log σ²) with a discount factor
is the direct upgrade. The posterior maintains μ, σ, and *their uncertainties*, and
those uncertainties can be propagated to the PMF as a Student-t marginal:

\[
Q(V_t > 0 \mid \mathcal{F}_{t-1}) = \int \mathrm{Bernoulli}(p_0(\eta)) \cdot p(\eta \mid \mathcal{F}_{t-1})\, d\eta.
\]

### 8.2 A change-point mixture

Discounted Bayes handles smooth drift (Type II). It does not handle structural
breaks (Type III). For the latter, a Bayesian online change-point detector (BOCPD,
Adams & MacKay 2007) maintains a posterior over the run length

\[
p(r_t \mid x_{1:t}) \propto \sum_{r_{t-1}} p(r_t \mid r_{t-1}) \cdot p(x_t \mid r_{t-1}, x_{1:t-1}),
\]

where \(r_t\) is the number of steps since the last change and the hazard function
\(H(r)\) controls the prior on run lengths. Under each \(r_t\), the parameter
estimate is the discounted-Bayes posterior using only data since the last change.

The output is a *mixture* over run-length hypotheses. The model emits a *mixture
density*, which is honest about its uncertainty and gracefully down-weights a single
parameter regime when a break has likely occurred.

### 8.3 A regime filter on observables

For Type IV, a regime filter on observables is the minimal solution. The two
observable regime variables with strong persistence are **spread** (ACF[1] = 0.516)
and **|ΔM|** (ACF[1] = 0.229). A small lookup table of (μ, σ) per (spread quintile,
|ΔM| quintile) is a *categorical* distribution whose sufficient statistics are
discounted-Bayes posteriors in each cell. This is essentially a non-parametric
Bayesian mixture with a deterministic gating network — and it is implementable
in 50 lines of code.

### 8.4 The full upgrade

Combining §8.1–8.3, the online model is

\[
Q_t(x_{t+1}) = \sum_{r_t} P(r_t \mid \mathcal{F}_t) \sum_{k} q_k(x_{t+1}) \cdot Q_t^{(k, r_t)}(x_{t+1}),
\]

where

* \(r_t\) is the posterior run length;
* \(k\) indexes the regime (spread × |ΔM| cell);
* \(Q_t^{(k, r_t)}\) is the discounted-Bayes posterior predictive in cell \(k\)
  using data since the last change-point.

This is the most sophisticated model the data justifies. It is *also* the simplest
generalisation of the existing EWMA that fixes its three deficiencies.

---

## 9. Why this is not a Transformer problem

A Transformer trained on 8700 interval steps has, optimistically, ~10⁵ independent
*labels* (each target = one step). For comparison, the smallest production-scale
language Transformer is trained on ~10¹⁰ tokens. The data-to-parameter ratio in our
case is ~10⁻⁵ of the regime where Transformers are known to work. A Transformer on
this data would generalise via *memorisation*, not learning.

Three more concrete reasons it is the wrong tool:

1. **The architecture has no natural notion of state.** A causal Transformer is a
   finite-context density estimator; an autoregressive process with finite memory
   is well-fit by a far smaller model. The data shows ACF(ΔM) dies by lag 2 and
   ACF(flow) dies by lag 10. A 4096-context Transformer is *wasteful* by a factor
   of 100× in compute and 1000× in data.
2. **The conditioning signal is short and tabular.** 30-second aggregated market
   features are at most a few hundred dimensions; a Transformer embedding wastes
   most of its capacity on position encoding that has no meaning here.
3. **The data has a known and *short* tail.** The deepest information-theoretic
   value-add is on the tail (Pareto α ≈ 3–5 for |ΔM|). Transformer output
   densities (softmax over discrete bins, or MDN heads) are notoriously poorly
   calibrated in the tails of heavy-tailed distributions.

The right place for a neural method, *if and when* the data justifies it, is a
*small* causal density network (≤ 10⁶ parameters) with a Student-t / NIG output head,
trained online with a frozen-feature backbone and an adaptive head. Even then, its
job is to estimate the conditional density of the same sufficient statistics the
§8 model would use — not to replace the structure.

> **Rule of thumb.** If, after the §8 model is implemented and scored properly, the
> *body* (say, P10–P90 of per-step NLL) is within 0.1 nats of the empirical entropy
> estimate, **no neural method can do better** without more data or a wider target.

---

## 10. The right model class, derived from the diagnostics

The data audit of §1 and the production-score audit of §2 imply a model class. The
tiers below are *not* a leaderboard; each tier is a *consequence* of a diagnostic
finding and a candidate *only if* the diagnostic is satisfied. The list is also the
*first* such class to try, not a final answer.

### Tier 0: Numerical hygiene (mandatory)

**Finding.** §2: probability floor, EWMA starvation, price placeholder.

**Mathematical requirement.** A scoring function \(S(Q, x) = -\log Q(x)\) that is
finite on the support of \(Q\) and continuous in the parameters.

**Model fix.** Replace the lognormal PMF with a Student-t density on \(\log V\)
(with ν floor of 4) and a log-PDF clip at \(-\)300 (or a smooth approximation that
decays to zero without hitting the floating-point floor). Replace the EWMA variance
with a discounted-Bayes posterior whose minimum variance is a *constant* (say, 0.01)
rather than zero. Persist the actual `params_before` for the price model.

**Expected impact.** Removes the 9 / 8697 catastrophic scores. Reduces mean flow NLL
materially (possibly by 1–2 nats per step) without changing the model's body.

### Tier 1: Heavy-tailed innovations (mandatory)

**Finding.** §1: mid-change kurtosis 7.18, Hill α ∈ (3, 5) for \(|ΔM|\), (1.6, 3) for flow.

**Mathematical requirement.** The innovation density must have polynomial or
semi-heavy tails. The Gaussian and lognormal families do not.

**Model fix.** For ΔM, a Normal-Inverse-Gaussian (NIG) emission. For the positive
part of flow, a Generalised Pareto (GPD) above a threshold, or a stretched
exponential / Weibull below it. For the hurdle, a non-parametric zero-probability
estimator (rolling frequency) instead of EWMA on a Bernoulli. Either way, the
posterior under discounted Bayes is tractable.

**Expected impact.** ≈ 0.2–0.5 nats per step reduction, mostly on the upper tail of
ΔM and the upper tail of flow volumes.

### Tier 2: AR(1) on flow intensity (likely worth it)

**Finding.** §1: ACF(total-flow)[1] = 0.287; ACF(buy)[1] = 0.272.

**Mathematical requirement.** Conditional mean of flow intensity depends on the
previous bucket's intensity. EWMA on (μ_buy, μ_sell) captures some of this, but only
on a slow timescale; the data shows burst-and-decay patterns on a faster timescale.

**Model fix.** A first-order autoregressive term on the latent log-rate:

\[
\lambda_t = \alpha \cdot \log V_{t-1}^+ + (1 - \alpha) \cdot \lambda_{t-1} + \varepsilon_t,
\]

with the hurdle probability also depending on λ_t. This is a small modification; it
adds one parameter and the tracking gain α is data-driven (§7.2: optimal α from the
drift-to-noise ratio).

**Expected impact.** ≈ 0.1–0.2 nats per step on the flow NLL.

### Tier 3: GARCH on the price innovation (likely worth it)

**Finding.** §1: ACF(|ΔM|)[1] = 0.229, [5] = 0.134, [20] = 0.066. ACF(ΔM²)[1] = 0.180.

**Mathematical requirement.** The conditional variance of ΔM is predictable from
the recent history of ΔM. The current model uses EWMA on σ_mid, which is an
IGARCH(1, 1) (i.e. α + β = 1, implying infinite persistence of shocks). The data
shows variance shocks decay in ~1 hour, so α + β < 1.

**Model fix.** Replace the IGARCH with a proper GARCH(1, 1) with α + β constrained
< 1:

\[
\sigma_t^2 = \omega + \alpha (ΔM_{t-1} - \mu)^2 + \beta \sigma_{t-1}^2,
\]

with discounted-Bayes online estimation of (ω, α, β) under a Beta prior enforcing
α + β < 1. This is well-known to be ~0.1–0.3 nats per step better than EWMA in
heteroskedastic settings, and it is the smallest possible upgrade to the price
component.

### Tier 4: Regime filter on (spread, |ΔM|) (probably worth it)

**Finding.** §1: rolling 1h variance of ΔM varies 52×; ACF(spread)[1] = 0.516;
ACF(|ΔM|)[1] = 0.229.

**Mathematical requirement.** Conditional distribution parameters depend on a slowly
varying observable regime (spread and recent absolute move).

**Model fix.** A categorical mixture indexed by (spread quintile, |ΔM| quintile)
with discounted-Bayes sufficient statistics per cell. The mixture weights are
*observable* (a deterministic function of state), so no filtering is needed.

**Expected impact.** ≈ 0.05–0.15 nats per step, mostly in regime transitions where
the unconditional model over-disperses.

### Tier 5: BOCPD for breaks (probably worth it)

**Finding.** §2: one explicit reset row in production; rolling 1h variance ratio
52×. The μ_buy_b = 0.0 in the middle of an otherwise smooth drift is *visible* in
the data, but the current EWMA does not detect it.

**Mathematical requirement.** Online posterior over the parameter regime.

**Model fix.** BOCPD with a constant hazard rate, combined with discounted Bayes
under each run-length hypothesis.

**Expected impact.** Robustness, not NLL. The expected improvement in median NLL is
small; the *tail* of the score distribution improves substantially, which is what
the MPC cares about.

### Tier 6: Expert aggregation (always worth it)

**Finding.** No model class dominates all regimes.

**Mathematical requirement.** A combination rule with provable regret bound.

**Model fix.** Hedge / Bayesian model averaging over the Tiers 0–5 experts, with
weights updated by their cumulative log loss.

**Expected impact.** Free improvement: the aggregate is provably no worse than the
best expert by \(O(\sqrt{\log K / T})\).

### Tier 7: Joint buy/sell model (only if Tier 0–6 leaves one)

**Finding.** corr(buy, sell) = 0.148.

**Mathematical requirement.** Same as Tier 1, but with a bivariate emission.

**Model fix.** Bivariate lognormal hurdle with covariance estimated online. Necessary
*only* if the residual after Tiers 0–6 still shows conditional correlation in the
buy/sell pair.

**Expected impact.** Likely < 0.05 nats per step.

### Tier 8: Neural density / Transformer (last resort)

**Finding.** None yet that justifies it.

**Mathematical requirement.** That the residual after Tiers 0–7 has structure that
the linear filter cannot capture (e.g. nonlinear feature interactions,
permutation-entropy in the conditional residuals).

**Model fix.** A small causal density network trained online with a frozen
backbone. *If and only if* permutation-entropy and HSIC tests on the Tier 7
residuals reject linearity.

**Expected impact.** Likely 0.0 nats per step on this dataset.

---

## 11. The market-maker connection — what prediction *is for*

The MPC consumes the model in two ways:

1. **Path sampling for the simulator.** Sample \(N\) candidate futures under \(Q_t\),
   score each against the candidate quote ladder, pick the one with highest expected
   utility minus risk cost. This is **a decision under uncertainty**, so the
   requirement on the model is *distributional calibration* on the joint outcome
   \(Y_{t,h}\), not just marginal NLL.
2. **Reservation price and inventory risk.** Compute \(\mathrm{VWAP}_t\) and
   \(\mathrm{VWAP}_t + \text{half-spread}_t\) under the same \(Q_t\), subtract the
   mark-to-market drift, compute the conditional VaR of inventory. This is **a tail
   statistic**, so the requirement is calibration of the *tails* of ΔM and book
   depth.

These two requirements are *not* the same. The first is well-served by a calibrated
multivariate model; the second by an accurate tail of a marginal model. A
single model can serve both only if it is well-calibrated on the joint and on the
marginals.

### 11.1 Variance reduction without directional signal

The data says ΔM is approximately martingale conditional on the bucket's realised
flow: \(E[ΔM_{t+1} \mid \mathcal{F}_t, V^+_t, V^-_t] ≈ 0\). This means the
*directional* component of the price model is near-zero. **But** the conditional
*scale* is highly predictable (ACF(|ΔM|)[1] = 0.229, rolling 1h variance varies 52×).

A model that gets the variance right and the mean wrong is *strictly* more
valuable to the MPC than the converse. The MPC's reservation price is roughly

\[
p_{\text{reservation}} = \mathrm{VWAP} - q \cdot \gamma \cdot \sigma_{\text{markout}},
\]

where γ is risk aversion and q is inventory. **Calibrated σ** ⇒ correctly
sized quotes ⇒ fewer adverse-selection losses. **Mis-calibrated σ** ⇒ systematically
too tight (under-disp) or systematically too wide (over-disp) quotes. Under-dispersion
is the *dangerous* failure: the bot will be picked off.

### 11.2 The independence of price change from order flow — what it means

The data says corr(net-flow\(_t\), ΔM\(_{t+1}\)) = 0.011. This is the strong version
of "the market incorporates flow within the bucket." Implications:

* **No forward flow-to-price information at this 30s horizon.** The exchange's
  matching engine + active market makers have already done the price discovery.
* **The within-bucket mechanical impact is observable but inside-the-bucket.** Any
  predictive model must work at sub-bucket granularity (queue position, event-time)
  to add value from this channel.
* **The flow variable is therefore primarily useful for the *quantity* model, not the
  *price* model.** This is the *opposite* of how it is sometimes used informally
  ("if flow is large, price will move").

This finding puts a hard upper bound on the value of any long-context price model on
this venue: if the market incorporates flow within one bucket, the price at \(t+1\) is
essentially the price at \(t\) plus a small, almost martingale, almost exchange-noise
update. The price side of the prediction problem is **a near-trivial forecasting
target**, and the residual variance is mostly:
  (a) reference-venue drift that the local book cannot observe, and
  (b) book refresh noise (cancel/replace at the best tick).

These are *structural* sources of unpredictability, not model failures. The right
response is **honest acknowledgement** of the limit, not a more complex model.

### 11.3 Reference pricing and transformations

The CSV does not carry a reference venue, but the live system likely does. The
relevant transformations are:

| Transformation | What it removes | When it helps |
|---|---|---|
| microprice \(= V_{\text{bid}} \cdot p_{\text{ask}} + V_{\text{ask}} \cdot p_{\text{bid}}\) | best-tick queue imbalance noise | always, for short-horizon ΔM |
| tick-returns ΔM / tick_size | unit-dependence | always |
| spread-normalised ΔM / S_t | bid-ask bounce, in units of spread | helps *only* for sub-spread moves; saturates for large moves |
| log returns | multiplicative scale dependence | helps for the price-level model, not the ΔM model |
| log-volume log(1 + V) | skewness of the volume distribution | always for flow |
| basis b_t = log P_t^local − log P_t^ref | local venue premium | necessary if a reference venue is available |
| imbalance sign × spread bucket | regime interaction | helps gating in regime models |

The microprice is the most under-used in the current system. The CSV carries
`microprice` (the production already computes it), but the price model scores
against `mid`, not `microprice`. A transformation to microprice changes the target
random variable and therefore the entropy floor; whether the floor is lower is
testable.

### 11.4 What about RL?

Reinforcement learning belongs **one layer above** the prediction problem. Its job is
to choose \(A_t\) given a model of the controlled transition \(P(Y_{t+1}, F_{t+1}
\mid \mathcal{F}_t, A_t)\). The prediction problem is the *exogenous* part of that
transition. RL on the exogenous part of market data without a controlled experiment
is policy-learning from observational data, which is fundamentally confounded.

The right division of labour:

* **Predict the exogenous market evolution** — the focus of this document.
* **Model the controlled execution** — fills given a quote, markouts given a fill,
  queue position dynamics. This is a separate problem and a separate model.
* **Optimize the quoting policy** over a frozen simulator — *this* is where RL or
  bandit methods earn their keep.

The market data in `mm_events` does not contain enough intervention variation to
identify the controlled transition offline. Safe exploration (random quote
perturbations within a tight band) is needed, and that experiment has not been done.

### 11.5 What does "price change is independent of buy/sell trades" tell us?

This empirical fact (corr ≈ 0 between next-step net flow and next-step ΔM) has four
distinct interpretations, listed in §6 of `THEORETICAL_FOUNDATIONS.md`. It is worth
re-stating them in this document because they bear directly on what is *worth*
modelling and what is not.

**(i) Price discovery has already happened inside the bucket.** The most natural
reading on a 30-second aggregation. The exchange's matching engine plus active
resting liquidity have moved the mid to reflect the bucket's flow by close. The
*true* signal is at sub-bucket granularity. **Modelling implication:** build a
sub-bucket event-time model (queue position, microprice, event timestamps) if you
want to predict within-bucket price; the bucket-aggregated ΔM is essentially noise.

**(ii) The local venue is a follower, not a leader.** If a reference venue drives
prices and Nobitex USDT/IRT follows with a small basis, then the *local* ΔM is
largely a copy of the *reference* ΔM. **Modelling implication:** the conditional
mean of local ΔM given the reference ΔM is close to 1 with a delay; the basis is
the predictable object. The lab needs a reference venue before this hypothesis can
be tested.

**(iii) Aggregation aliases a within-bucket shock.** A large order arriving late in
the bucket will consume the best level, shift mid, and *also* register as a large
trade. The bucket's ΔM and net-flow both jump, and they correlate post-hoc. A
30-second aggregate conflates arrival and impact. **Modelling implication:** the
*time of arrival within the bucket* matters; if available from the trade-by-trade
stream, this is a non-trivial sub-bucket feature.

**(iv) An unobserved common shock drove both.** Macro news, an exchange outage on a
related pair, an external event — these move flow and price jointly at the same
moment. **Modelling implication:** the *coincident* correlation in some buckets
should not be exploited as "flow predicts price"; it is a confound.

**The most important consequence of (i)–(iv):** *the market is functioning*. An
exchange with this price-discovery behaviour is doing its job; the liquidity
providers (other market makers on this venue) are doing theirs. The local maker's
edge cannot come from "predicting" the next mid; it has to come from faster
reaction, smaller spreads, lower adverse selection. The prediction layer's job is
to **measure the residual uncertainty honestly** so that the MPC's spread and
inventory decisions are calibrated to the true noise, not to a wrong model of it.

---

## 12. The score vector — mandatory diagnostics

A model is *promotable* only if it improves a *score vector*, not a single number.
The minimum diagnostics for any candidate predictive model are:

| Question | Diagnostic | What fails when |
|---|---|---|
| Is the full distribution closest? | Joint NLL (chained) on the §3.3 outcome | one-sided factorisation |
| Are probabilities honest? | randomised PIT for discrete flow; PIT for continuous price | mean PIT ≠ 0.5, or variance ≠ 1/12 |
| Is uncertainty too wide or too narrow? | interval coverage at α ∈ {0.5, 0.8, 0.9, 0.95, 0.99} | systematic over- or under-coverage |
| Are the tails honest? | tail-weighted log score (worst 5 % of quantile) | under-coverage at high α |
| Does it beat baselines meaningfully? | paired, block-bootstrapped log-score diff against Tier-0 | not significant after dependence correction |
| Is it stable across regimes? | regime-stratified NLL (by spread quintile, vol quintile, time-of-day) | some regimes show large degradation |
| Does it improve decision utility? | shadow MPC replay with the new model | no measurable PnL or tail-loss improvement |

The shadow MPC replay is the *final gate*, not a leaderboard. A model that improves
NLL by 0.5 nats per step but loses 2 bps of expected adverse selection in the MPC
is not an improvement.

---

## 13. Reproducible scoring — protocol

To make the leaderboard meaningful:

1. **Fix the target** before opening any new file. The joint target is
   \((V^+_{t+1}, V^-_{t+1}, ΔM_{t+1}, ΔS_{t+1})\) on the discrete grid (lot integer,
   tick integer), with the score defined as the discrete PMF value (no Jacobian
   trick for the discrete part; PMF on integer grid for the continuous part with a
   declared bin width).
2. **Fix the filtration** to the projection actually recorded in the CSV. Anything
   richer must be loaded from `mm_events`.
3. **Pre-declare gaps handling.** A 3960-second gap is not a quiet bucket; it is a
   restart. Either condition on elapsed time or restart filtering across the gap.
4. **Pre-declare regime slices.** Score by spread quintile, vol quintile, hour of
   day, presence of a gap immediately before.
5. **Use block-bootstrap CIs**, not IID standard errors. Block length from the loss
   autocorrelation, not arbitrary.
6. **Score the *body* of the distribution, not the tail.** Trim the worst 1 % or use
   a tail-weighted score; do not let a single floating-point underflow dominate the
   mean.
7. **Score all horizons consistently**, including the ones the bot does not currently
   use, so that future additions have a baseline.

---

## 14. Required data artefacts (before any new model)

The current CSV is insufficient. From `mm_events` in Postgres, the lab must extract:

1. **Full depth snapshots** for each bucket. Required for any model that conditions
   on book shape beyond best bid/ask.
2. **Reference venue prices** (or absence thereof, with reason). Required to test
   the price-discovery hypothesis.
3. **Our own order history**: placed, cancelled, filled, with timestamps and queue
   position where known. Required to test the interventional hypothesis.
4. **Receipt timestamps**: the time at which each exchange event was received by the
   bot, distinct from the exchange-emit timestamp. Required to test the staleness
   hypothesis.
5. **Trade-by-trade stream**: for the buckets marked as "high flow," the individual
   trades and their classifications. Required to test the §11.2 within-bucket
   hypothesis.
6. **Gap and outage log**: which buckets are gap-restart vs. quiet, and why. Required
   for the §13 protocol.

These are *data engineering* tasks, not modelling tasks. They are the price of
admission.

---

## 15. Closing — what counts as the "best" model

There is no best model without a target, a filtration, a horizon, a score vector,
and a decision utility. With those pinned down, the answer for *this* dataset is
inferred from §1, §2, §5, §6, §7.

For the **flow** side, the data admits at most ~0.5 nats per step improvement over a
well-calibrated hurdle-lognormal with state-dependent μ, and only if Tiers 1–4 are
implemented carefully. The Tier-0 numerical fix alone may deliver more. The
deepest-information Tier-2 (AR on log-rate) is implementable in ~30 lines and is
likely the largest marginal win. After Tier-6 (Hedge), the gain over the current
incumbent's *body* of NLL is probably **0.3–0.8 nats per step**, and any additional
gain requires new data, not new architecture.

For the **price** side, the data admits at most ~0.3 nats per step improvement over a
calibrated random-walk with GARCH variance. The Tier-3 GARCH upgrade is the largest
single win. The Tier-5 BOCPD adds robustness, not NLL. After all reasonable tiers,
the price model is essentially calibrated, and the remaining error is the *market's
own residual noise*.

The **joint** side admits ~0.5–1.0 nats per step improvement over the current
factorisation, mostly through Tiers 1–3 + Tier 6.

**The "best" model in this regime is therefore**:

> A discounted-Bayes hurdle-Student-t flow model with AR(1) on log-rate, a regime
> filter on (spread quintile, |ΔM| quintile), a BOCPD for breaks, an NIG
> price-innovation model with GARCH(1,1) variance, a Hedge aggregator over the
> top-3 empirical-CDF baselines, and a per-component support floor. No neural
> method. No transformer. No RL. No hyperparameter grid.

That is the ceiling this data permits. Above that ceiling lies only information not
in the data (full book, reference venue, queue position, intervention exposure), and
*that* is what the next data-engineering round is for.

> **The single most actionable item, today.** Fix the scoring pipeline (§2), fix the
> EWMA starvation (§8.1), persist the actual price-model parameters, then re-run the
> lab on the full 4-day window and report the *body* (P10–P90) of the NLL by regime.
> The mean and the tails will move dramatically; the body will move a little; the
> ranking of models is what the body tells you.

---

## Appendix A — The data diagnostics, recomputed

All numbers in §1 were recomputed from the supplied CSV. The commands are
reproducible:

```
# count rows, gaps, basic moments
python3 -c "import csv; ..."
# mid-change ACF/PACF, kurtosis
python3 -c "..."
# Hill estimators
python3 -c "..."
# corr(net_flow_{t-1}, ΔM_t), quintile analysis
python3 -c "..."
# rolling 1h variance of ΔM
python3 -c "..."
```

A single script can be written that re-runs all of these and writes a JSON report;
that is the first deliverable of the model_lab.

## Appendix B — A falsifiable check list, in priority order

1. **§2.1 fix**: replace PMF floor, record actual price params, re-run scoring.
2. **§1.1 measurement**: extract and analyse full book from `mm_events`.
3. **§1.6 entropy estimate**: LZ-compress the discretised ΔM sequence and the V
   sequence, report compression ratio as a proxy for the entropy rate.
4. **§6 tests**: corr(V\(_{t+k}\), ΔM\(_{t+1}\)) for k ∈ {−5, −2, −1, 0, +1, +2, +5}
   with HAC errors. Report confidence intervals.
5. **§7.2 tracking regret**: compute the empirical parameter variation budget from
   the production file and compare to EWMA performance.
6. **§10 Tier 0–3 models**: implement, score, compare to the existing incumbent
   *body* (not tail) of NLL.
7. **§12 diagnostics**: report the full score vector for each candidate.
8. **§14 data engineering**: extract full book + reference venue + own orders +
   receipt timestamps.

The first item fixes the most damaging artefact in the current measurements and
should take precedence over every other modelling effort.