# A first-principles research programme for the market transition model

## Thesis

The task is not to find a distribution with the lowest NLL on one CSV. It is to
construct, update, and falsify a **decision-time conditional law** for the next
market transition, then determine whether a better law improves market-making
decisions after adverse selection, inventory, and execution costs.

At decision time `t`, let `F_t` contain only data genuinely available to the
controller, and let an online model emit

\[
Q_t(\cdot)=Q(\cdot\mid\mathcal F_t;\theta_t),\qquad
\theta_{t+1}=U(\theta_t,O_{t+1}).
\]

The unknown ideal is `P_t^*(.) = P(. | F_t)`. The programme is to answer,
in this order:

1. What is `F_t`, and what future random object must the controller simulate?
2. Which available information measurably reduces uncertainty about it?
3. How stable is that reduction under dependence and non-stationarity?
4. Does an improved, calibrated transition law improve decision utility?

NLL is central evidence for questions 2–3. It is not by itself the objective
of a market maker.

## 1. Define the object before choosing a model

### 1.1 The filtration is a product requirement

Define the information set explicitly:

\[
\mathcal F_t=\sigma\{O_s,A_s:s\leq t;E_t\}.
\]

`O` includes only book/trade/reference observations available by exchange and
receipt timestamps; `A` includes outstanding/filled own orders, inventory,
fees and queue age where known; `E` is data quality: book age, feed reset,
snapshot/reconstruction mode, clock skew and missing intervals. An observation
that belongs to the same nominal bucket but arrived after the decision is not
in `F_t`.

The lab's score-before-update rule is necessary but insufficient protection
against leakage. Every feature needs an availability timestamp and every score
record needs the data/feature version used to form it.

### 1.2 Forecast a joint transition, not “the price”

Let `B_t` be the local book, and let `Z_t` be a representation sufficient to
derive all quantities the MPC uses: mid, spread, executable VWAP by depth and
book shape. A useful next-step outcome is

\[
Y_{t+1}=(V^+_{t+1},V^-_{t+1},Z_{t+1},R_{t+1},M_{t+1}),
\]

where `V+`/`V-` are aggressive buy/sell flow, `R` is reference price or local
basis, and `M` contains book/data-validity outcomes. The object needed by a
controller is, in general,

\[
P_t(Y_{t+1}\mid\mathcal F_t,A_t).
\]

`A_t` matters: our quote can change displayed depth, fills, queue position and
therefore measured outcomes. Omitting it from an external-market component is
a small-maker approximation to be tested, not assumed.

A natural simulator factorisation is

\[
\begin{aligned}
P_t(Y_{t+1}\mid\mathcal F_t,A_t)
&=P_t(V_{t+1}\mid\mathcal F_t,A_t)\\
&\quad P_t(Z_{t+1},R_{t+1},M_{t+1}\mid V_{t+1},\mathcal F_t,A_t).
\end{aligned}
\]

This fits the present MPC: sample aggressive flow, mechanically consume the
book, then sample the conditional book/refill transition.

The guide's flow NLL scores the first factor. Its residual and conditional
mechanical-mixture price scores condition on the *realised* interval flow and
score a version of the second factor. This is valid for the factorisation; it
is not valid to select a production simulator from separately ranked flow and
price NLLs. The primary generative statistic must be the chained joint score

\[
\ell_{t+1}^{joint}=\log Q(V_{t+1}\mid\mathcal F_t)
+\log Q(Z_{t+1},R_{t+1},M_{t+1}\mid V_{t+1},\mathcal F_t).
\]

Keep the terms separately for diagnosis, but compare candidates using the sum
on one fixed target representation.

### 1.3 A structural hypothesis

Start from a partially observed controlled process:

\[
X_{t+1}\sim K_t(\cdot\mid X_t,A_t),\qquad
O_{t+1}\sim G_t(\cdot\mid X_{t+1},A_t),
\]

with a latent state such as

\[
X_t=(F_t^{ref},b_t,\nu_t,\lambda_t^+,\lambda_t^-,L_t,c_t).
\]

These are, respectively, reference/efficient price, local basis, volatility,
buy/sell demand intensity, latent liquidity supply, and regime/data state.
This is not a claim that each coordinate is separately identifiable. It gives
falsifiable roles: marked point-process demand, limit-order replenishment and
cancellation, delayed local price discovery, and observation error.

The present mechanical-consumption-plus-residual model is a restricted member
of this class. Its residual combines replenishment, cancellations, external
price movement, snapshot error, and missing mechanics. A large residual is
evidence of a missing state or poor measurement, not evidence for Gaussianity.

## 2. Predictability and its limits

### 2.1 Conditional entropy is the theoretical bound

With a stationary law and logarithmic score,

\[
\inf_Q\mathbb E[-\log Q(Y_{t+1}\mid\mathcal F_t)]
=H(Y_{t+1}\mid\mathcal F_t),
\]

attained by the true conditional law. A restricted predictor pays expected
conditional KL error. The best possible incremental value of feature `W_t` is

\[
I(Y_{t+1};W_t\mid\mathcal F_t)
=H(Y_{t+1}\mid\mathcal F_t)-H(Y_{t+1}\mid\mathcal F_t,W_t).
\]

Estimate this operationally as the held-out **prequential log-score
difference** between nested, timestamp-correct online information sets. Do not
try to estimate high-dimensional book entropy directly.

There is no finite-sample test proving the intrinsic entropy of an unrestricted
non-stationary process. A defensible conclusion is only: no specified available
feature improved a pre-registered forecast by more than its dependent-sample
uncertainty bound in the tested periods.

For continuous targets, differential entropy and density NLL depend on units
and coordinates. Do not compare raw NLL of ticks, log returns, residual VWAP,
and a book embedding. Fix one target/base measure or include the transformation
Jacobian. For discrete flow score probability mass at recorded resolution.

### 2.2 Direction, scale, and value are distinct

It is possible that

\[
\mathbb E[r_{t+1}\mid\mathcal F_t]\simeq0,
\qquad\operatorname{Var}(r_{t+1}\mid\mathcal F_t)=\sigma_t^2
\text{ is predictable}.
\]

Calibrated conditional scale can change quote width, depth, inventory limits
and risk-boundary probability, even without directional edge. It does not
automatically justify tighter quotes. For an action `a`, fill indicator `F`,
and post-fill markout `K`, the economically critical distributions are

\[
P(F,K\mid\mathcal F_t,A_t=a),\qquad
\mathbb E[K\mid F=1,\mathcal F_t,A_t=a].
\]

The conditional covariance of fill and future price (adverse selection) can
matter more than unconditional price accuracy.

### 2.3 What flow-price independence would say

The relevant null is not low correlation but

\[
R_{t+1}\perp V_{t+1}\mid\mathcal F_t.
\]

If it survives conditional lead/lag tests, recorded local aggressive flow adds
no incremental reference-price information beyond the current book/reference
state. Possible explanations: reference-venue price discovery, rapid passive
replenishment, sign/aggregation error, an inappropriate bucket, or mostly
uninformed local trades. It does not deny mechanical consumption of local depth;
mechanical impact and informational impact are different claims.

For a locally lagged market, model a time-aligned basis instead:

\[
\log P_t^{local}=\log P_{t-\delta_t}^{ref}+b_t+\epsilon_t,
\]

where `delta_t` is an explicit availability/staleness delay.

## 3. Metrics and model selection

Use prequential joint log score as the primary statistical score. Add, rather
than replace it with, these diagnostics:

| Question | Required diagnostic |
|---|---|
| Are probabilities calibrated? | randomized PIT for discrete flow, PIT/rank histograms and conditional coverage by regime |
| Are tails safe? | interval/quantile coverage and a pre-declared tail-weighted log score or CRPS |
| Are components jointly plausible? | energy/variogram score and calibration of signed-flow/return dependence |
| Is a gain real? | paired cumulative loss difference with a contiguous-block bootstrap interval |
| Does it persist? | time-block, volatility/liquidity/staleness and post-change score slices |
| Does the model know uncertainty? | calibration conditional on forecast scale and stale-data state |

CRPS and energy score are useful when density normalisation is awkward. They
are not universally superior: log score is strictly proper and estimates the
conditional-KL objective. Select the score before seeing challengers.

The final decision metric is a different object. For policy `pi` and realised
net PnL `G`, specify before evaluation

\[
J(\pi)=\mathbb E[G_\pi]-\rho\,\mathcal R(G_\pi)
-C_{inventory}-C_{churn}-C_{obligations}.
\]

The risk measure and transition must include fees, latency, cancellations,
terminal liquidation and post-fill adverse selection. A model is promoted only
after it is calibrated probabilistically *and* improves a frozen decision
replay/shadow experiment on expected utility and tail loss.

Counterfactual policy evaluation also needs action overlap: historical behaviour
must have placed varied quotes in comparable states. Without it, offline PnL
claims under a new policy are extrapolation. RL does not cure missing
counterfactual data; it requires a valid controlled transition or safe,
logged exploration.

## 4. Dependence, drift, and genuine online learning

There may be no constant `P*`. Compare against a dynamic oracle and report
discounted dynamic regret,

\[
\sum_{t=1}^T w_t\{-\log Q_t(Y_t)+\log Q_t^*(Y_t)\},
\]

where the comparator's allowable changes are declared. An EWMA decay therefore
encodes a market-change assumption, not a harmless hyperparameter.

Separate three changes: covariate shift (states occur at different rates),
conditional shift (the transition law changes), and observation shift (feed,
aggregation, tick/lot or clock behaviour changes). Detect the third before
calling a surprise a market regime change.

Use no IID standard errors. Preserve temporal dependence with contiguous block
resampling, a stationary bootstrap, or a HAC/Diebold-Mariano-type test of the
loss differential. Choose and sensitivity-test block length from loss
autocorrelation. Model elapsed time or break gaps; unequal intervals are not
silently IID samples.

An online solution need not be an EWMA, but it must obey

\[
\text{forecast from }\mathcal F_t\rightarrow\text{observe }O_{t+1}
\rightarrow\text{update bounded, versioned state}.
\]

The state may be a posterior/filter, change-point mixture, expert ensemble, or
neural state-space representation. A high-capacity model additionally needs a
strictly historical replay buffer, update schedule, latency budget, rollback
rule, and conservative uncertainty mechanism. Otherwise it is an offline model
periodically refit, not an online solution.

## 5. Evidence-led construction path

This is a sequence of hypotheses to test, not a grid of candidate algorithms.
Advance only if the prior stage has a stable and economically material result.

### A. Identify the data and target

Create an immutable decision-time table with event ID, exchange/receipt/source
times, raw and materialized book versions, trade de-duplication, aggregation
version, actions/queue/fills and data-quality state. Audit cadence/gaps,
duplicates, crossed/invalid books, stale snapshots, depth truncation, trade-book
alignment, volume units, and whether own orders enter observations.

The referenced 30-second CSV, performance CSV and `THEORY_*.md` files are not
present in this checkout. The local parquet archive contains 10-second and
60-second Nobitex MarketEvents, not the requested 30-second series. Therefore
this document makes no invented empirical claim from them. Reconstruct the
series from Postgres or add the files before interpreting statistics.

### B. Test whether any available feature helps

Build online nulls: a seasonal/clock-time zero-inflated joint flow law,
martingale/persistence price law with adaptive scale, mechanical book consumption
plus broad refill noise, and a basis persistence law. Add information sets in a
fixed causal order: own past, flow, book state, reference venue, then action.

For every target and horizon report

\[
\Delta_h=\overline{L_h(Q^{base})-L_h(Q^{base+feature})}.
\]

Use time/block-shifted feature placebos as negative controls. A feature that
also appears predictive after its availability is broken is usually an
alignment leak. This produces a predictability atlas rather than a correlation
table.

### C. Test a low-dimensional predictive state

Use transformations for a reason: log returns/tick-normalised distances remove
price units; `log1p` volume plus zero and trade-count indicators separate
occurrence from mark size; cumulative depth by normalised distance or basis
coefficients expresses book shape; microprice-mid, imbalance, executable-VWAP
slope, spread and local-reference basis express geometry; causal multi-scale
exponential histories express memory.

Fit a transparent adaptive conditional distribution/filter and test whether
residuals retain autocorrelation, feature dependence, joint miscalibration or
regime instability. This is a variance-efficient reference and tells us which
mechanism is missing.

### D. Add complexity only to explain rejected residual hypotheses

If residual tests show persistent volatility/liquidity/reference regimes, use a
latent state-space filter with explicit adaptation and uncertainty. If nonlocal
path dependence remains, a causal neural filter (recurrent state-space model or
causally masked transformer) may parameterise the *same conditional density*,
not merely a point prediction. It must beat an online state-space baseline with
the same target, information, and deployment rules.

Four days of 30-second data contain at most 11,520 nominal intervals and far
fewer effective independent regimes. That is normally inadequate for training
a flexible transformer from scratch. Longer, diverse, version-consistent data
or carefully separated pretraining is a prerequisite, not a tuning choice.

An ensemble gate is justified only when experts demonstrably win in recurrent,
observable regimes; the gate itself is a prequentially scored forecast.

### E. Model controlled execution separately

When own quotes matter, use

\[
P(Y,F,K\mid\mathcal F,A)=P(Y^{ext}\mid\mathcal F)
P(F,K\mid Y^{ext},\mathcal F,A).
\]

This makes clear that price prediction is not a fill/queue model. Structural
execution modelling, safe randomized quote perturbations, or supported causal
off-policy estimation are possible identification paths. RL is only an
optimizer over such a learned/observed controlled process.

## 6. Protocol and deliverables

1. Freeze target, filtration, horizons, score, baselines, regime slices and
   minimum practical effect before testing a challenger.
2. Run chronological prequential replay: forecast, store reproducible
   distribution/seed, score, then update. Do not tune repeatedly on one period.
3. Use rolling-origin development and a later untouched period; report paired
   block-resampled uncertainty, worst block and regime table.
4. Require calibration before sharpness. Narrower distributions are better only
   if their conditional coverage is correct.
5. Compare statistically credible models in a frozen execution replay, then
   shadow and limited live deployment with pre-declared rollback rules.

The first deliverables are therefore: (i) the 30-second data-contract/audit;
(ii) a versioned joint target and chained score in the lab; (iii) a
predictability atlas with negative controls; (iv) calibrated adaptive nulls;
(v) an explicit residual hypothesis that decides the next mechanism to add;
and (vi) an independently measured action/fill/markout layer.

## Decision rules

- No “best” model exists without a target, filtration, horizon and decision
  loss.
- No price model wins from conditional-on-realised-flow NLL without the paired
  flow factor and joint-path evaluation.
- No finite sample proves intrinsic uncertainty; report a bounded empirical
  predictability claim instead.
- No gain is trusted without dependence-aware uncertainty, calibration and a
  later untouched period.
- Neural networks, transformers and RL earn their place only by resolving a
  documented residual dependency or controlled-action limitation.
- If predictable information is genuinely small, a broad calibrated online
  mixture and conservative execution policy is the correct answer—not a
  failure to search enough hyperparameters.
