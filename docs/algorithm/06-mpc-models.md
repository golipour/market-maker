# 06 — Stochastic models: how the MPC imagines the next horizon

The optimizer is **model-predictive**: it simulates `H` future steps under a
*stochastic model* of order flow and price. The model has two pieces:

1. a **price model** — produces `(Δmid, P(next event))` per step;
2. a **order-flow model** — produces `(V₊, V₋)` per step.

Both are pluggable through the `PriceModel` and `OrderFlowModel` protocols. The
runtime always uses the same *family* of models for the *real* data and the
*simulated* data, so the optimizer is consistent with what the aggregation
pipeline produces.

## 6.1 Notation

- `H` — horizon (number of decision steps simulated into the future).
- `B` — Monte Carlo batch size (number of paths).
- `V₊, V₋` — buy and sell market volumes over a single step.
- `Δmid` — change in mid over a single step.
- `vwap, vwap_buy, vwap_sell` — volume-weighted average prices for the step.

## 6.2 Price models (`mm/mpc/price_torch.py`)

### 6.2.1 `PriceModel` protocol

```1:40:mm/mpc/price_torch.py
class PriceModel(Protocol):
    def sample(self, state: StateTensor, horizon: int) -> tuple[Tensor, Tensor]:
        """Return (Δmid of shape (B, H), log-prob of shape (B, H))."""
    def log_prob(self, realized: Tensor) -> Tensor: ...
    def update(self, realized_step) -> None: ...
```

The optimizer needs both `sample` (used in simulation) and `log_prob` (used in
importance-sampling corrections and diagnostics). `update` is called on the
realized step after each interval closes.

### 6.2.2 `GaussianPriceModel`

The simplest model: `Δmid ∼ 𝒩(0, σ²)` with `σ` learned from realized mid
changes via EWMA. Used in unit tests and as a fallback.

### 6.2.3 `TorchResidualPriceModel`

A two-stage model:

1. Use a *mechanical* mixture model (fixed-depth mid/spread/shape deltas driven
   by `V₊, V₋` and the resting book).
2. Learn a residual distribution around the mechanical prediction via EWMA.

The residual is Gaussian with a learned variance. The implementation keeps a
running mean and variance for the residual, updated on each realized step.

### 6.2.4 `TorchConditionalMechanicalMixturePriceModel`

The full production model. It has:

- fixed-depth mid/spread/shape deltas learned per `(V₊, V₋, imbalance, book
  shape)` bucket;
- a residual model that can be either Gaussian or Student-t (configurable);
- an RLS (recursive least squares) update for the bucket weights so the model
  adapts online without storing history.

`sample` returns `Δmid` per Monte Carlo path; `log_prob` is used by the
optimizer to score the realized path against the simulated paths.

## 6.3 Order-flow models (`mm/mpc/market_order_flow_torch.py`)

### 6.3.1 `OrderFlowModel` protocol

```1:35:mm/mpc/market_order_flow_torch.py
class OrderFlowModel(Protocol):
    def sample(self, state: StateTensor, horizon: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (V₊, V₋, log-prob) of shape (B, H) each."""
    def log_prob(self, realized_step) -> Tensor: ...
    def update(self, realized_step) -> None: ...
```

The order-flow model produces a *joint* sample of buy and sell volumes for
each simulated step.

### 6.3.2 Hurdle models

The order-flow distribution is *hurdle*: a mass at zero and a continuous
positive distribution. The hurdle log-likelihood is

```
log p(V) = log p0  if V = 0
        = log(1 − p0) + log p⁺(V)   otherwise
```

Implementations:

- `TorchHurdleLogNormalModel` — `log V | V > 0 ∼ 𝒩(μ, σ²)`.
- `TorchHurdleLogStudentTModel` — heavy-tailed alternative.
- `TorchHurdleGammaModel` — `V | V > 0 ∼ Gamma(α, β)`.
- `TorchHurdleWeibullModel` — `V | V > 0 ∼ Weibull(k, λ)`.
- `TorchHurdleGeneralizedParetoModel` — `V | V > 0 − u ∼ GPD(ξ, σ)`.

Each model learns its parameters from the realized `V₊` and `V₋` via EWMA on
the positive part and a separate EWMA on the zero-mass probability.

### 6.3.3 Cross-correlation

The two sides are not independent. Each `sample` call draws `V₊` and `V₋` from
a *bivariate* distribution; the implementation factorizes as
`p(V₊, V₋) = p(V₊) · p(V₋ | V₊)` with a learned conditional that captures the
directional pressure visible in `imbalance`.
