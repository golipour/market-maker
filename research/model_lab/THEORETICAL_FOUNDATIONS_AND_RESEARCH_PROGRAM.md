# Theoretical foundations and research programme

## Purpose and answer in one sentence

The object to discover is **not a parameter vector and not a named model**.  It is the
smallest online predictive state (Z_t), computable from information available at the
decision time, whose conditional distribution of the future is sufficiently accurate for
the market-making decisions that follow.  Model classes are consequences of evidence about
that state and its dynamics.

This document is deliberately upstream of `GUIDE.md`.  The guide specifies a useful
*implementation benchmark* (causal walk-forward scoring); this document specifies the
scientific programme that decides what ought to be benchmarked.  It also corrects a common
ambiguity: a low NLL on an arbitrarily transformed target is not by itself a solution to
the market-making problem.

## 1. Start with the right mathematical object

Work on a filtered probability space

\[
(\Omega,\mathcal F,(\mathcal I_t)_{t\geq0},P).
\]

At the end of bucket (t), let (O_t) be *all observations actually available before the
next quote decision*: timestamped trades, the order book, book age, reference venues,
calendar time, system health and our outstanding orders.  Let (A_t) be our quote/cancel
action.  The observed bucket representation currently used by the lab is only a projection

\[
X_t=(B_t^{+},B_t^{-}, M_t,S_t,\mathcal B_t,\ldots)=g(O_t),
\]

where (B^{+},B^{-}) are aggressive buy/sell volumes, (M) is a mid or reference price,
(S) is spread, and \(\mathcal B\) is the book.  The information set is
\(\mathcal I_t=\sigma(O_{0:t}, A_{0:t-1})\), not merely the last CSV row.

For a horizon (h), choose an outcome (Y_{t,h}) before choosing a model.  A minimal
useful environmental outcome is a jointly defined sequence of

\[
Y_{t,1:H}=\{(B^+_{t+j},B^-_{t+j},\Delta R_{t+j},
             \Delta S_{t+j},\Delta L_{t+j})\}_{j=1}^{H},
\]

where (R) is the price used to mark inventory and \(L\) denotes executable liquidity or
book-shape shocks.  This is more meaningful than “predict the mid”: the MPC needs adverse
selection, consumption of displayed liquidity, and price/shape evolution jointly.  It may
be too large to estimate directly; that is a representation problem, not a licence to
silently change the target.

An online forecaster produces a probability kernel

\[
Q_t(dy\mid\mathcal I_t,A_t), \qquad
\theta_{t+1}=U(\theta_t,O_{t+1},A_t),
\]

and scores (Y_{t,1}) **before** assimilating it.  Parameters are therefore not a fixed
historical fit: they are a posterior/filtering state.  In a non-stationary market the true
kernel is (P_t), so the goal is tracking (P_t), not estimating an eternal (P).

There is an important causal qualification.  If our quotes can change displayed liquidity,
trade arrivals, or the exchange's book, historical observations are drawn under a behaviour
policy, not automatically under a new policy.  The correct counterfactual is

\[
P(Y_{t,h}\mid \mathcal I_t,\operatorname{do}(A_t=a)),
\]

not necessarily (P(Y_{t,h}\mid\mathcal I_t,A_t=a)).  A pure “market evolution” model
should either use a period in which our impact is negligible, include our action and
inventory as inputs, or explicitly model this intervention.  Otherwise a model can learn
our own footprint and call it prediction.

## 2. What is the objective?

### 2.1 The primary scientific objective: conditional distribution, not point accuracy

For a fixed, fully specified target and base measure, the logarithmic score is

\[
L_{\log}(Q_t,Y_{t+1})=-\log q_t(Y_{t+1}\mid\mathcal I_t).
\]

It is strictly proper: in expectation under the true conditional law it is uniquely
minimized by (q_t=p_t).  The excess expected score has an exact interpretation:

\[
E[L_{\log}(Q_t,Y)\mid\mathcal I_t]-H(P_t)
 =D_{KL}(P_t\Vert Q_t)\geq0.
\]

Thus causal/prequential NLL is the right primary score **when the MPC consumes samples
from the whole forecast distribution**.  It rewards calibrated tails and dependence rather
than merely getting the direction or conditional mean right.

The benchmark must report the additive chain-rule components on the same outcome space:

\[
-\log q(Y)= -\log q(B^+,B^-)
              -\log q(\Delta R,\Delta S,\Delta L\mid B^+,B^-).
\]

This is an exact factorisation if the second term conditions on flow.  It does *not* assume
price and flow are independent.  In contrast, the product
\(q(B^+)q(B^-)\) assumes conditional independence of the two sides and pays an irreducible
penalty (I(B^+;B^-\mid\mathcal I_t)) if that assumption is false.  This is the relevant
question for the current independent-side flow model.

### 2.2 Why NLL alone is insufficient

NLL has four non-negotiable comparability requirements.

1. The outcome, conditioning set, horizon, censoring rule, and score period must be
   identical.  A density for a four-dimensional shape vector cannot be compared to a density
   for only \(\Delta M\).
2. Units and resolution must be fixed.  Differential NLL changes under a change of units;
   use a PMF on contract/lot and tick grids, or include the correct bin-width/Jacobian term.
   The supplied CSV has fractional volumes while the logged flow observation is rounded
   (e.g. 585.54 becomes 586); this discretisation is part of the probabilistic model and
   must be explicit.
3. Missing buckets are neither zero trades nor ordinary 30-second transitions.  Gap-aware
   horizons must be separated or represented by their elapsed time.
4. A very flexible density can improve a score while producing bad samples in a tail that
   matters to inventory.  Proper score is necessary, not sufficient for deployment.

Accordingly, maintain a **score vector**, with NLL first:

| Question | Prequential diagnostic |
|---|---|
| Is the full distribution closest? | Joint NLL and its flow/price/shape decomposition |
| Are probabilities honest? | randomized PIT/rank histograms, conditional coverage, calibration slope, and multivariate rank tests |
| Is uncertainty too wide or too narrow? | sharpness conditional on calibrated coverage; interval score |
| Are quantities with no tractable density useful? | CRPS (univariate), energy/variogram score (multivariate) |
| Are rare adverse events covered? | a proper tail-weighted CRPS or threshold-event log score, reported separately |
| Does it improve the actual policy? | locked-down shadow-MPC expected utility, inventory risk and tail loss |

The last row is a **deployment gate**, not a replacement training target.  It prevents
optimizing a backtest policy by accident.  If a model has the same directional uncertainty
but lower, correctly calibrated variance, it can still be very valuable: the MPC can quote
tighter with the same directional drift because its distribution of adverse excursions and
fillable depth is narrower.  If it is merely *under-dispersed*, it will look sharp and make
the MPC dangerously aggressive; coverage and tail score distinguish these cases.

### 2.3 Decision sufficiency is the final abstraction test

Let (c) be the MPC action and \(u(c,Y)\) its realised utility, including fills, fees,
inventory and risk.  A representation (Z_t=f(\mathcal I_t)) is decision-sufficient for a
class \(\mathcal C\) when

\[
\arg\max_{c\in\mathcal C}E[u(c,Y)\mid\mathcal I_t]
=\arg\max_{c\in\mathcal C}E[u(c,Y)\mid Z_t].
\]

This is the formal reason to seek a latent state rather than “all possible features.”  A
principled compression target is an information bottleneck: minimize
\(I(Z_t;\mathcal I_t)\) subject to a bound on predictive or decision loss.  The book may
be high-dimensional in raw coordinates but low-dimensional relative to this objective.

## 3. Limits of prediction: what can and cannot be learned

For a stationary ergodic, discrete process, the best attainable average log loss with the
entire past is its entropy rate

\[
h=\lim_{t\to\infty}H(Y_{t+1}\mid Y_{0:t},\text{available covariates}).
\]

No method—transformer, RL, or oracle architecture search—can beat (h).  The value of
adding a legal feature set (F_t) beyond state (Z_t) is exactly conditional mutual
information:

\[
I(Y_{t+1};F_t\mid Z_t)
= H(Y_{t+1}\mid Z_t)-H(Y_{t+1}\mid Z_t,F_t).
\]

This gives a clean interpretation of a properly evaluated NLL improvement: it is an
estimate of reducible uncertainty in nats per bucket, not proof of a causal mechanism.

But entropy is not magic.  For continuous prices it is differential entropy, which depends
on scale; for a non-stationary process there may be no single entropy rate; and finite data
cannot identify an unrestricted conditional law.  “No observed improvement” means only
that any improvement is below the resolution of this data, this horizon, and this model
comparison.  It does not prove unpredictability.  The defensible finite-sample statement is
a confidence/credible interval for *prequential score difference against a locked baseline*,
using dependence-respecting blocks.

The practical substitute for unknowable (h) is a nested sequence of information sets:

\[
\varnothing\subset Z^{(0)}_t\subset Z^{(1)}_t\subset\cdots\subset\mathcal I_t.
\]

Compare their best robust online predictors on untouched future blocks.  The improvement
from (Z^{(k)}) to (Z^{(k+1)}) estimates the information value of that additional state;
when gains vanish with uncertainty bounds, stop increasing state complexity.  This is a
model-agnostic, operational definition of “how much entropy can be removed.”

## 4. Non-IID data changes the theory

The useful null is not IID.  The competing descriptions are:

\[
P_t(Y_{t+1}\mid\mathcal I_t)\quad\text{may be}
\begin{cases}
P(Y_{t+1}\mid Z_t) & \text{stationary Markov after state construction},\\
P(Y_{t+1}\mid Z_t,R_t) & \text{finite/continuous latent regimes},\\
P_{u}(Y_{t+1}\mid Z_t),\ u=t/T & \text{locally stationary drift},\\
P_t\ \text{with abrupt jumps} & \text{change points},\\
\text{an arbitrary sequence} & \text{adversarial online setting}.
\end{cases}
\]

Each statement can be rejected or supported only locally and provisionally.  A level price
is likely non-stationary; a return, spread-normalised return, residual relative to a
reference venue, or book-coordinate shock may be closer to stable.  “Closer” must mean
stable conditional distribution, not an attractive histogram.

Under drift, judge an online learner by **dynamic regret** rather than convergence to a
single parameter:

\[
\sum_{t=1}^{T}\ell_t(Q_t)-\sum_{t=1}^{T}\ell_t(Q_t^*)
\quad\text{where }Q_t^*\text{ is the best comparator allowed to move over time}.
\]

The comparator's variation budget must be declared.  Exponential forgetting is reasonable
only for gradual movement; it is a poor approximation to a jump.  Use a posterior over
run-lengths/change points when evidence supports breaks, and retain a conservative fallback
distribution during uncertainty.  With no trustworthy probabilistic stationarity
assumption, aggregate a small set of validated forecasters online by log-loss/expert
aggregation; its regret guarantee is then more honest than a fictitious generative claim.

## 5. Empirical facts from the supplied four-day extract

The file `nobitex_USDT_IRT_30000ms_4d.csv` contains 8,700 rows from 2026-07-08 07:14:30 UTC
to 2026-07-11 09:15:00 UTC.  These numbers are descriptive, calculated directly from that
file, and are not claims about the full market:

| Diagnostic | Observation | Consequence |
|---|---:|---|
| cadence | 8,677 30s transitions; 22 gaps (up to 3,960s) | Do not treat every row pair as the same transition. |
| zero total-flow buckets | 4.6% | This sample is not the 59%-zero regime described in `GUIDE.md`; zero-flow baselines remain a control, not a plausible dominant model here. |
| positive total-flow median / P90 / P99 / max | 441 / 2,386 / 8,959 / 39,123 USDT | flow is highly skewed and tail-sensitive. |
| \(\operatorname{acf}_1\)(total flow), \(\operatorname{acf}_{120}\) | 0.287, 0.107 | flow is plainly not IID; long/latent activity state is worth testing. |
| \(\operatorname{acf}_1\)(mid change), \(\operatorname{acf}_1\)(absolute change) | -0.238, 0.229 | reversal/bid-ask mechanics and volatility clustering need separate treatment. |
| excess kurtosis of mid change | 7.19 | a fixed Gaussian innovation is an inadequate tail null. |
| block SD of mid changes | 48.8–74.8 IRT | conditional scale changes materially across this short sample. |
| correlation buy vs sell volume | 0.148 | independent-side flow is questionable, but this is not a conditional-independence test. |
| correlation signed *contemporaneous* flow vs mid change | 0.171 | realised flow and realised price share information. |
| correlation signed flow at (t) vs change at (t+1) | 0.032 | little linear one-bucket-ahead signal survives; nonlinear/conditional signal and timing still require tests. |

For a deliberately coarse direction target \(D\in\{\downarrow,0,\uparrow\}\), the
empirical unconditional entropy is 1.453 bits.  Conditioning on the previous direction
reduces it by only 0.021 bits in-sample; spread quintile reduces it by 0.020 bits; a
75-cell combination of prior direction, imbalance and spread reduces it by 0.052 bits but
has cells with as few as 15 observations.  This is a warning against proclaiming a
directional edge.  It is not an entropy-rate estimate and it has not been held out.

The logged production performance deserves an immediate data-quality/calibration audit,
not an architecture race.  Across the supplied file the flow NLL has median 13.06, P99
21.46 and nine extreme scores above 100 (maximum 690.78), consistent with cold-start or
support-floor events.  The price score has median 79.04 but mean 2,155, P99 26,451 and
4,209 scores above 100; its logged standardized residuals have SD about 65.6.  Because
that price score combines several transformed components, these values are not comparable
to flow NLL or to another target.  They *do* show that support, scale, missing-data and
conditional calibration must be audited before reading its mean as model quality.

The export does not include raw depth levels, despite reporting depth counts.  Full-book
state discovery and mechanical counterfactual tests must therefore query `mm_events` in
Postgres rather than infer book shape from this CSV.

## 6. The pre-model research programme

This phase produces falsifiable findings and a target contract.  It is not hyperparameter
search.  Every plot/table is computed on chronological blocks, with intervals resampled in
blocks rather than shuffled.

### A. Establish the observation and target contract

* Record exchange timestamps, receipt timestamps, bucket boundaries, book snapshot age,
  missingness, tick/lot sizes, trade-side classification, aggregation rules and whether our
  orders are present.  Measure how much a stale book predicts apparent future moves.
* Split true 30-second transitions from (60s+) gaps.  Either condition on elapsed time or
  restart filtering across a gap.  Never label a long gap as a quiet bucket.
* Define a canonical discrete target for each horizon (e.g. tick return, integer lots,
  discretised book coordinates) and a canonical joint score.  Store raw data so the mapping
  is reproducible.
* Choose the reference price deliberately.  A local mid includes quote bounce and local
  liquidity noise; an external/reference mid tests price discovery; a microprice tests
  short-horizon queue imbalance.  They answer different questions and must not be conflated.

### B. Measure predictability before fitting rich functions

For each target (flow occurrence, joint side volume, signed flow, return, volatility,
spread, executable depth) compute a **predictive-information curve** as history increases:

\[
\Delta_k = \overline L(Q[Z_{t-k:t-1}])-
           \overline L(Q[Z_{t-k-1:t-1}]).
\]

Use deliberately simple, regularized conditional estimators plus blocked prequential tests;
the purpose is detecting information, not declaring their functional form correct.  Include
calendar, stale-book indicators, reference-market changes, previous flow, book imbalance,
spread/depth and our own action where legal.  Plot gains by horizon.  A gain only at one
bucket is microstructure timing; a gain persisting for 20 buckets is a different latent
state claim.

Test conditional—not marginal—independences:

\[
I(B^+_{t+1};B^-_{t+1}\mid Z_t),\qquad
I(\Delta R_{t+1};B_{t+1}\mid Z_t),\qquad
I(\Delta R_{t+1};B_{t-k:t}\mid Z_t).
\]

Estimate their practical counterpart by the out-of-sample log-score gain of a joint or
augmented predictor over a restricted one.  Assess it with stationary/block bootstrap or
time-block permutation, preserving serial dependence.  A Pearson correlation near zero is
not evidence of conditional independence.

### C. Characterise distribution, memory, and change

* Compare empirical conditional CDFs and tail exceedance rates by activity/spread/calendar
  state; estimate tail-index stability only where sample size supports it.  The question is
  whether a single innovation family has stable standardized residuals.
* Inspect ACF/PACF, nonlinear dependence and long-memory diagnostics for both levels and
  transformations; test residual whiteness only after conditioning on a candidate state.
* Run rolling distributional-distance/change-point diagnostics on conditional mean, scale,
  zero rate, tails and score.  Report detection delay and false-alarm rate prospectively;
  retrospective break dates alone are not usable online.
* Estimate intrinsic/effective state dimension with out-of-sample reconstruction/prediction
  curves, not visual embeddings.  If a handful of stable book coordinates retain all score
  gain, a huge raw-book network is unjustified.
* Measure the horizon at which each target becomes indistinguishable from an unconditional
  forecast.  This establishes how far MPC rollouts may rely on informative dynamics before
  reverting to a conservative unconditional/regime forecast.

### D. Resolve the market-structure questions carefully

“Price change is independent of buy/sell trades” can mean four different things:

1. Contemporaneous realised flow does not explain the same-bucket price change after book
   state and timing.  That would question the mechanical-impact proxy.
2. Past flow adds no information about *next* price after current state.  Then flow is not
   a useful predictive input at that horizon, even though it may mechanically move the book
   within a bucket.
3. Local price does not respond but reference price moves first.  Then the local venue is
   following price discovery elsewhere, and reference venue state is missing.
4. Both respond to an unobserved common shock.  Association is not a causal flow effect.

The current 30s correlations are compatible with all four.  Resolve them using event-time
ordering where possible, reference-venue features, lag/lead conditional tests, and a model
that conditions on the pre-trade book.  Do not interpret a regression coefficient as a
structural impact curve without that design.  If the mechanical branch has a negative or
zero conditional response, it may reveal replenishment, aggregation aliasing, adverse
selection, a stale snapshot, or an unsuitable reference price—not necessarily irrational
market makers.

## 7. Model families follow from the findings

The question is always “which assumption survived the preceding tests?”  These are not a
flat menu of models to grid-search.

| Supported empirical structure | Mathematical representation and online update |
|---|---|
| Stable low-dimensional, approximately linear, conditionally light-tailed shocks | state-space model; Kalman/recursive least squares is the optimal linear-Gaussian filter and a strong falsifiable baseline. |
| Discrete activity/spread/volatility states with persistent transitions | switching state-space/HMM; filter regime probabilities rather than hard-cluster then regress. |
| Smoothly drifting parameters | dynamic generalised linear/state-space model with discounting; estimate tracking loss and uncertainty in the drift. |
| Abrupt recurrent breaks | Bayesian online change-point/run-length filter plus a reset/heavy-tail fallback. |
| Self-exciting event arrivals and marks | marked point process in event time; bucket volumes are its integrals, so this is justified only when event timestamps show excitation beyond bucket artefacts. |
| Strong nonlinear predictive information in a stable, finite state window | conditional normalizing flow/mixture density network or distributional neural state-space model, trained/updated online with a calibrated uncertainty head. |
| Long context has repeatable value across many independent days/markets | causal transformer used as a conditional-density/state encoder, with strict temporal pretraining and online adaptation; not an in-context excuse to leak future data. |
| No stable distributional law but several validated experts | online Bayesian/exponential expert aggregation under log loss, with a dynamic-regret guarantee. |

Neural networks and transformers are estimators of a conditional kernel, not answers to the
identification problem.  They need substantially more independent regime coverage than four
days, explicit capacity control, posterior/ensemble uncertainty, and evaluation against the
same target contract.  Their ability to reduce training NLL says nothing about an entropy
gap unless it persists in future regimes.

RL belongs one layer later.  It can optimize a quoting **policy** given a trustworthy
transition model or safely gathered interactive data.  It does not identify the exogenous
market kernel from passive historical data, and offline RL here is especially vulnerable to
policy/action confounding and unobserved fills.  Use it only after the predictive simulator
passes calibration and counterfactual validity gates.

## 8. Baselines that genuinely locate the difficulty

Baselines are nested information hypotheses, not weak straw men:

\[
Q^{(0)}(Y)\;\to\;Q^{(1)}(Y\mid\text{calendar})\;\to\;
Q^{(2)}(Y\mid\text{last state})\;\to\;
Q^{(3)}(Y\mid\text{short history})\;\to\;
Q^{(4)}(Y\mid\text{book/reference/action state}).
\]

At each arrow report paired, block-resampled prequential log-score difference, calibration,
and conditional tail score.  This shows whether the value is from seasonality, persistence,
book state, or a proposed nonlinear representation.  Include the following controls:

* empirical regime-conditional distribution (the entropy/no-memory reference);
* random walk/reference random walk for price; persistence/zero and a **joint** side-flow
  distribution for flow;
* a robust heavy-tailed conditional-scale model; and
* an oracle diagnostic that permits information which is intentionally unavailable at quote
  time.  Its score is an upper-bound diagnostic, never a deployable result.

The current hurdle-lognormal with independent sides and the conditional mechanical mixture
are useful members of this ladder, not the definition of the ladder.

## 9. Online evaluation and selection protocol

1. Freeze the target contract, transformations, scoring code, data-quality rules, baselines
   and candidate *assumptions* before inspecting the final evaluation period.
2. Divide time chronologically into many contiguous evaluation episodes spanning quiet,
   active and stressed regimes.  Warm-up only with data before each episode.  Pretraining,
   if used, must precede the episode and be versioned.
3. At every step: form forecast from the pre-update filter; store a reproducible predictive
   distribution/sample seed; score the realised observation; then update.  Score all
   horizons consistently and do not silently skip hard cases.
4. Compare paired losses on identical timestamps.  Use moving-block/stationary bootstrap or
   sequential confidence sequences, not IID standard errors.  Report score over time and by
   regime, tail, staleness and gap status, not only one mean.
5. Fail a model that wins mean NLL by becoming conditionally miscalibrated, has unbounded
   catastrophic scores, or loses on the risk-relevant conditional subset.  A conservative
   mixture fallback is part of the production forecast.
6. Shadow-run the unchanged MPC with forecast distributions from the incumbent and candidate.
   Only then compare realised/simulated decision metrics under an agreed risk constraint.

Selection should permit online mixtures rather than a permanent “winner.”  Maintain a
posterior/weight over structurally distinct, calibrated experts and publish the current
weight, regime belief, forecast entropy, data-quality state and abstention/fallback state to
the controller.  The controller must know when its uncertainty is epistemic (state/model
uncertain) as well as aleatoric (market intrinsically random).

## 10. Concrete first deliverables

Before adding a new predictive architecture, produce these durable artefacts from Postgres:

1. A versioned data/target contract and gap/staleness report.
2. A chronological “market atlas”: marginal and conditional distributions, tails, calendar
   effects, memory, change points, reference-venue lead/lag, and action exposure.
3. A predictive-information report: nested-state prequential score gains, horizons and
   blocked uncertainty intervals for every proposed input.
4. A dependence report: joint buy/sell flow, flow-to-price conditional tests, residual
   autocorrelation, and a statement of which factorisations survive.
5. A calibration report for the incumbent, including PIT/coverage and worst-regime/tail
   scores.  The extreme current price-score behaviour makes this first priority.
6. A one-page model-assumption decision record: which of the rows in §7 is supported, what
   it predicts, what would falsify it, and what online update/fallback it requires.

Only then is implementation work well posed.  The likely near-term result from the supplied
sample is not “use a transformer.”  It is: make time and measurement semantics correct;
model a joint, heavy-tailed, conditionally heteroskedastic process; determine whether a
persistent latent activity/reference-price state explains the observed dependence; and
preserve calibrated uncertainty through the MPC.  If those investigations leave a stable,
material score gap for long context or nonlinear book interactions, a neural state-space
model or transformer becomes a justified estimator rather than a hopeful experiment.
