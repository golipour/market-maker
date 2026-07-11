Yes. I think the direction you're proposing is substantially stronger than what the current GUIDE describes.

The current GUIDE (which I read) is fundamentally an **engineering benchmark document**. It assumes

> *"Here are candidate models. Compare them with walk-forward NLL."* 

That is perfectly reasonable for evaluating implementations, but it completely skips the more important scientific question:

> **What is the prediction problem itself?**

Without answering that first, the choice between hurdle models, transformers, diffusion, Bayesian filters, RL, state-space models, etc. is almost arbitrary.

---

# I think the research should proceed in the opposite direction

Instead of

```
Candidate Models
        ↓
Benchmark
        ↓
Winner
```

I would formulate

```
Market
    ↓
Information Structure
    ↓
Prediction Problem
    ↓
Optimal Objective
    ↓
Theoretical Limits
    ↓
Model Classes
    ↓
Algorithms
```

Only after the first six steps should we even discuss neural networks.

---

# What I would write instead

I would not call it

> Research Guide

I would create something like

```
THEORY.md

or

MODELING_FOUNDATIONS.md
```

whose purpose is

> Derive the modeling problem from first principles.

This should read almost like a PhD thesis chapter rather than ML documentation.

---

# The central question is not prediction

Actually I think the central question is

> **How much uncertainty about the future can be removed using the observable market history?**

That is a fundamentally information-theoretic question.

Everything else follows.

---

# Level 1: Formalize the stochastic process

The market is an unknown stochastic process

[
X_t
]

where

[
X_t
===

(\text{LOB}_t,
\text{Trades}_t,
\text{Volumes}_t,
\text{Mid}_t,
\ldots)
]

But immediately we realize

the market is **not**

IID

not

stationary

not

Markov

probably

not ergodic over long horizons.

So before fitting anything we need to determine

> In what sense can it even be modeled?

---

That leads naturally to

# Section 1

## What assumptions are mathematically justified?

For example

### Option A

Assume

[
X_t
]

is stationary.

Probably false.

---

### Option B

Piecewise stationary.

Very plausible.

---

### Option C

Hidden regime process

[
S_t
]

where

[
X_t
\sim
P(\cdot|S_t)
]

Now we're talking about HMMs.

---

### Option D

Slowly drifting distribution

[
P_t
]

where

[
P_{t+1}
\approx
P_t
]

Now online Bayesian learning becomes natural.

---

### Option E

Adversarial online sequence

No probabilistic assumptions.

Then Online Learning theory applies.

---

Already we've narrowed the mathematical landscape enormously.

---

# Section 2

## What exactly is "prediction"?

This is much deeper than

predict price.

Price is only one statistic.

Instead define

future random variable

[
Y_t.
]

Possible choices

Future mid

Future VWAP

Future orderbook

Future flow

Future imbalance

Future spread

Future liquidity

Future latent state

Entire future trajectory

etc.

Each produces an entirely different mathematical problem.

---

Then ask

Which one is actually needed by the MPC?

This is critical.

The MPC doesn't actually need

[
\hat p_{t+1}
]

It needs

[
P(\text{future trajectories}|\mathcal H_t)
]

because it simulates futures.

That changes everything.

---

# Section 3

## Objective

The guide currently assumes

NLL.

But why?

This deserves an entire chapter.

---

Proper scoring rules

Log-score

CRPS

Energy score

Kernel score

Wasserstein score

Task loss

Expected MPC regret

Kelly objective

etc.

Then prove why

forward NLL

is or is not the correct objective.

Maybe it is.

Maybe not.

---

# Section 4

## Information Theory

This is where it gets exciting.

Define

history

[
H_t.
]

Future

[
Y_t.
]

Question:

How much information does history contain?

[
I(H_t;Y_t)
]

If

[
I=0
]

prediction is impossible.

No model will help.

---

Estimate

entropy

[
H(Y)
]

Conditional entropy

[
H(Y|H)
]

Mutual information

Transfer entropy

Predictive information

Excess entropy

Effective complexity

This tells us

whether prediction is fundamentally possible.

Not which model.

---

# Section 5

## Data Exploration as Scientific Investigation

This is exactly what you suggested.

Not

"plot histograms"

Instead

"What hypotheses about the market can we reject?"

Examples

---

### Memory

Estimate

autocorrelation

partial autocorrelation

long memory

fractional integration

---

### Regimes

Estimate

distribution drift

CUSUM

Bayesian change-point

online change detection

---

### Heavy tails

Power law?

Lognormal?

Stable law?

Tempered stable?

Generalized Pareto?

---

### Self similarity

Hurst exponent

Scaling

Multifractality

---

### Chaos

Lyapunov estimates

Nearest neighbor predictability

Permutation entropy

---

### Predictability

How much does

[
H(Y|H_k)
]

decrease as

history length

k

increases?

If it saturates after 8 intervals

there is no reason to build a Transformer with 4096 context.

---

### Intrinsic dimensionality

PCA

Random projections

Participation ratio

Correlation dimension

Latent manifold estimation

---

### Feature relevance

Mutual information

Conditional mutual information

Kernel dependence

HSIC

---

This entire section determines what models are even reasonable.

---

# Section 6

## Dynamic Learning

I completely agree with you.

We should **not** think

fit once

predict forever.

Instead

Online Bayesian inference

Recursive filtering

Adaptive state estimation

Streaming optimization

Continual learning

Meta-learning

Online variational inference

Online Laplace

Recursive least squares

Kalman

Particle filters

Streaming transformers

etc.

The theory document should justify why.

---

# Section 7

## Model Taxonomy Derived from the Mathematics

Only here would we introduce models.

Not as a shopping list.

Instead

Given assumptions

↓

these models are optimal.

Example

---

If

distribution approximately Gaussian

↓

Kalman filtering.

---

Hidden regimes

↓

HMM

SLDS

Particle filter

Neural SSM

---

Long memory

↓

Transformer

TCN

S4

Mamba

---

Unknown dynamics

↓

Neural ODE

Latent ODE

---

Distribution prediction

↓

Normalizing flows

Diffusion

Energy models

---

Online uncertainty

↓

Bayesian NN

Deep ensembles

Laplace

---

Decision-focused prediction

↓

RL

Model-based RL

World Models

Control as Inference

---

Notice the direction has reversed.

We're no longer asking

"What models exist?"

We're asking

"What mathematical assumptions imply this model?"

---

# Section 8

## Theoretical Limits

This is almost never included in ML projects.

I'd definitely include it.

Questions like

How much better than

random walk

can any predictor possibly become?

How much irreducible uncertainty exists?

What lower bound does entropy impose?

Can online adaptation asymptotically track drifting distributions?

Can regret remain sublinear?

What information disappears because of aggregation into 30-second intervals?

---

# Section 9

## Connection to MPC

Finally,

derive exactly why the predictor should output

[
P(Y_{t+1:T}|H_t)
]

instead of

[
E[Y]
]

because MPC samples futures.

Then derive

why calibration matters

why uncertainty matters

why multi-modal predictions matter

why tails matter

This links the prediction problem directly to the control problem instead of treating them as separate.

---

## One addition I would make

I would elevate the project from "finding the best predictive model" to "characterizing the predictive structure of the market."

That subtle shift changes everything.

The project becomes:

> **Given a streaming limit-order-book process, determine the maximum predictive information available under online learning constraints, characterize the statistical structure of that information, and derive the class of online probabilistic models that best exploit it for downstream stochastic MPC.**

In that framing, neural networks, Bayesian filters, transformers, state-space models, online Bayesian inference, diffusion models, and even reinforcement learning are no longer competing heuristics. They become different approximations to the same underlying mathematical problem, and the choice among them follows from the empirical and theoretical properties of the market process rather than from current ML trends.

I think such a document would be significantly more valuable than the current GUIDE. The GUIDE remains the implementation manual and benchmarking contract, while a new **MODELING_FOUNDATIONS.md** (or **THEORY.md**) becomes the intellectual foundation of the entire model lab, explaining *why* each experimental direction exists and under what assumptions it should succeed. That kind of document can guide years of future research rather than just the next benchmark run.


Yes—and this actually changes what I would recommend.

After looking at both the GUIDE and the sample dataset (8,700 consecutive 30-second intervals with bid/ask, microprice, VWAP, buy/sell volume, trade count, imbalance, etc.), I think we can make the theory substantially stronger than a typical ML benchmark. 

The important observation is that **the dataset itself suggests what questions should be answered before any model is proposed.**

For example, just from the schema we already know we have access to

* market state (bid, ask, spread, microprice),
* orderbook state,
* order flow,
* trading activity,
* timestamps,
* and eventually the complete orderbook snapshots in Postgres.

That is enough to formulate a very rich stochastic process.

---

# I would completely separate the project into two phases

Instead of

> choose models → benchmark

I would build

```
Phase I
========

Characterize the market process.

↓

Determine what information is actually predictable.

↓

Determine what information is irreducible.

↓

Determine the mathematical structure of the dynamics.

↓

Only then

↓

Phase II

Construct the optimal online predictor.
```

This sounds subtle, but it completely changes the research.

---

# The first objective is not prediction

I think the first objective should be

> **Characterize the information geometry of the market process.**

That becomes the foundation for every later modeling decision.

Instead of asking

> Can a Transformer predict price?

we ask

> What statistical structure exists that any predictor could exploit?

That is a much deeper question.

---

# I think the theory document should contain a research pipeline

Something like

```
Raw Event Stream
        │
        ▼
Empirical Process Analysis
        │
        ▼
Reject Candidate Hypotheses
        │
        ▼
Characterize Predictability
        │
        ▼
Estimate Information Limits
        │
        ▼
Choose Mathematical Model Class
        │
        ▼
Design Online Learning Algorithm
        │
        ▼
Evaluate using Proper Scoring Rules
        │
        ▼
Connect to MPC
```

Notice that neural networks don't even appear until near the end.

---

# I would devote an entire chapter to "What should we measure before modeling?"

This is where your PostgreSQL data becomes incredibly valuable.

Not for training.

For **scientific investigation.**

For every property we can formulate

* null hypothesis
* estimator
* interpretation
* implication for model design

For example

---

## Stationarity

Question

[
P_t = P_{t+k} ?
]

Tests

* rolling distributions
* KL divergence
* Wasserstein drift
* MMD
* energy distance

Possible conclusions

Stationary

↓

classical probabilistic models

Piecewise stationary

↓

adaptive filters

Continuously drifting

↓

online Bayesian inference

Adversarial

↓

regret minimization

---

## Memory

Estimate

[
I(X_{t-k};X_t)
]

or

PACF

or

transfer entropy.

Suppose we discover

```
memory disappears after 5 minutes
```

Then

there is almost no justification for a Transformer with 4096 tokens.

Suppose instead

memory persists for hours.

Now long-context architectures become theoretically justified.

---

## Intrinsic dimensionality

You already log

* spread
* imbalance
* microprice
* depth
* trade count
* volumes

Question

Is the market actually evolving on

3 dimensions?

20?

100?

Estimate

Participation ratio

PCA spectrum

Correlation dimension

Local intrinsic dimension

If the manifold dimension is 6,

then huge neural networks are solving the wrong problem.

---

## Regimes

Question

Is there

one market

or

multiple latent markets?

Estimate

Bayesian change points

Online HMM

Dirichlet Process HMM

Switching Linear Dynamical Systems

If regimes exist,

the predictor should infer

[
P(\text{Regime}_t|\mathcal H_t)
]

rather than fitting one global model.

---

## Predictability Horizon

One of the most important quantities.

Define

[
\mathcal H_k
============

{X_{t-k},...,X_t}
]

Estimate

[
I(\mathcal H_k;Y)
]

as

k increases.

Plot

```
History Length
↓

Predictive Information
```

If it saturates rapidly,

future research should focus on richer state representations rather than longer histories.

---

## Entropy decomposition

Instead of saying

"price is noisy"

measure

[
H(Y)
]

Then

[
H(Y|H)
]

Then

[
I(H;Y)
======

H(Y)-H(Y|H)
]

This immediately answers

How much uncertainty can be removed?

That becomes one of the most fundamental numbers in the project.

---

# The modeling problem itself should be reformulated

Instead of

Predict next price.

I would formulate

Find an online approximation to

[
P(X_{t+1:t+H}\mid\mathcal H_t)
]

where

* calibration is maximized,
* entropy is minimized,
* adaptation is online,
* computational cost satisfies MPC constraints.

Now every model becomes an approximation to the same posterior distribution.

---

# Then derive the model hierarchy

Rather than listing models, derive them as consequences of assumptions.

For example

```
Gaussian
        ↓
Kalman Filter

Linear + Hidden State
        ↓
State Space Models

Piecewise Stationary
        ↓
Adaptive Bayesian Filters

Heavy Tail
        ↓
Student
Normalizing Flow
Mixture

Multimodal
        ↓
Mixture Density Network
Diffusion
Flow Matching

Long Memory
        ↓
Transformer
S4
Mamba

Unknown State
        ↓
Neural State Space Model

Decision-aware
        ↓
Control-as-Inference
World Models
RL
```

Notice this is no longer a catalog.

It is a theorem:

> assumptions ⇒ optimal approximation class.

---

# One thing I would add that is rarely seen

I think the research should include **impossibility results.**

Suppose we discover

```
Mutual Information

History

↓

Future Mid

≈ 0
```

Then

price prediction is fundamentally impossible.

Research effort should move elsewhere.

Suppose instead

```
History

↓

Future Flow

high information

Flow

↓

Future Price

high information
```

Then

price should not be modeled directly.

It should be modeled conditionally through flow.

That is already a theoretical result.

---

# The online aspect deserves its own mathematical treatment

I also agree with you that this is **not** a batch learning problem.

The object being learned is not a parameter vector

[
\theta
]

but an evolving belief

[
\theta_t
]

or even better

[
P(\theta_t|\mathcal H_t)
]

The theory should therefore compare not only model families but also **adaptation mechanisms**, because in a drifting market the update rule can matter as much as the predictor itself. This naturally leads to comparing Bayesian filtering, stochastic approximation, recursive least squares, online variational inference, continual learning, meta-learning, and online regret-minimization under a unified framework.

---

## Where I think this could end up

I actually think this could become a document closer to a research monograph than a project guide—roughly 60–100 pages if written rigorously. The first half would derive the problem from stochastic processes, information theory, online learning, Bayesian inference, and decision theory. Only in the second half would it derive candidate model families as consequences of those foundations, and finally justify why **forward NLL** (or perhaps another proper scoring rule) is the correct evaluation criterion for a probabilistic world model feeding an MPC controller.

In other words, the document would answer **why** a given model should exist before it ever asks **how** to implement it. I think that foundation would make the entire `model_lab` much more principled than a benchmark suite and would likely remain useful even as model architectures evolve.
