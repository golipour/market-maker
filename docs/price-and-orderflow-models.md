# Price And Orderflow Models

This document describes the stochastic models that feed the MPC controller:
what each model represents, what data updates it, how its parameters are
learned, and where the runtime uses the sampled paths.

## Runtime Wiring

The controller owns four model slots:

| Slot | Config key | Main implementation | Meaning |
|------|------------|---------------------|---------|
| Local market orderflow | `controller.local_market_flow_model` | `mm.mpc.market_order_flow_torch` | Buy/sell market-order volume on the local venue |
| Local price | `controller.local_price_model` | `TorchResidualPriceModel` | Residual VWAP move not explained by market-order consumption |
| Global market orderflow | `controller.global_market_flow_model` | `mm.mpc.market_order_flow_torch` | Buy/sell market-order volume on the global/reference venue |
| Global price | `controller.global_price_model` | `GlobalGaussianPriceModel`, `GaussianGlobalDeltaModel`, or residual model | Global buy/sell VWAP deltas or global VWAP residuals |

`MpcStrategyEngine.__post_init__()` builds these models through
`_build_controller()` in `mm/strategies/mpc_engine.py`. The local models are
always built when the controller is enabled. Global models are only passed into
the optimizer when a global binding exists and either liquidity sourcing is
enabled or `controller.reference_price_source == "global"`.

Per decision, `MpcStrategyEngine.decide()` follows this order:

1. Build a `State` from the latest local/global `StrategyInputFrame`.
2. Score model observations with pre-update parameters.
3. Update global price, market-flow models, then local residual price.
4. Refresh diagnostics with post-update parameters.
5. Run `MonteCarloMPCTensorBatched.optimize(state)`.
6. Publish `StrategyDecision.model_updates` and `model_scores`.

That order is deliberate: diagnostics report whether the observation was likely
under the old model, while the optimizer uses the freshly updated parameters.

## Data Used By The Models

All model updates come from `StrategyInputFrame`, which is assembled from market
events. The relevant fields are:

| Data | Source in frame/state | Used by |
|------|-----------------------|---------|
| `buy_volume`, `sell_volume` | `frame.local` / `frame.global_` | Market orderflow models |
| Current local/global orderbook | `frame.local.orderbook`, `frame.global_.orderbook` | VWAP state and residual price updates |
| Previous frame orderbook | `MpcStrategyEngine._previous_frame` | Residual price model |
| Local/global buy/sell VWAP | `vwap_from_raw_orderbook(..., target_depth=state.vwap_depth)` | Global price model, state valuation, rollouts |
| Inventory-derived VWAP depth | `state.vwap_depth` | Residual and global VWAP models |

`state.vwap_depth` is the absolute inventory being marked. In single-venue mode
it is `abs(local_inventory)`. In liquidity-sourcing mode it is
`abs(local_inventory + global_inventory)`. This keeps the learned and simulated
VWAP reference tied to the size of inventory the optimizer is actually marking.

## Market Orderflow Models

Market orderflow models represent random next-step aggressive buy and sell
volume:

```text
(V_buy, V_sell) ~ model parameters for the venue
```

They are independent per side in the current implementation. In optimizer
rollouts, sampled buy volume consumes the ask side of the book and sampled sell
volume consumes the bid side.

### Runtime-Supported Kinds

`_build_market_flow_model()` supports these `kind` values for local and global
market flow:

| `kind` | Class | Parameters returned by `get_params()` |
|--------|-------|----------------------------------------|
| `hurdle_lognormal` | `TorchHurdleLogNormalModel` | `p0_buy`, `mu_buy`, `sigma_buy`, `p0_sell`, `mu_sell`, `sigma_sell` |
| `hurdle_gamma` | `TorchHurdleGammaModel` | `p0_*`, `shape_*`, `scale_*` |
| `hurdle_weibull` | `TorchHurdleWeibullModel` | `p0_*`, `wb_k_*`, `wb_lambda_*` |
| `gamma` | `TorchDiscreteGammaModel` | `shape_*`, `scale_*` |
| `weibull` | `TorchWeibullModel` | `wb_k_*`, `wb_lambda_*` |

`mm.mpc.market_order_flow_torch` also contains `TorchHurdleLogStudentTModel`
and `TorchHurdleGeneralizedParetoModel`. They implement update/sample/PMF
methods, but they are not currently exposed through `_build_market_flow_model()`
or scoring dispatch.

### What The Distributions Mean

For hurdle models, each side first samples whether the interval has zero
aggressive flow:

```text
P(V = 0) = p0
P(V > 0) = 1 - p0
```

Conditional on positive flow, the volume is sampled from the configured positive
distribution:

| Model | Positive-volume law |
|-------|---------------------|
| Hurdle lognormal | `V = LogNormal(mu, sigma)` |
| Hurdle Gamma | `V = Gamma(shape, scale)` |
| Hurdle Weibull | `V = Weibull(k, lambda)` |
| Hurdle log-t, module only | `V = exp(StudentT(df, mu, sigma))` |
| Hurdle GPD, module only | `V = u + GeneralizedPareto(xi, beta)` |

The optimizer uses continuous PyTorch samples. Scoring and the legacy
`HurdleOrderFlow` helper use rounded integer volumes and discrete PMFs based on
CDF mass around each integer.

### Parameter Learning

All market-flow models are online EWMA estimators with decay `gamma = decay`.
Newer data has weight `1 - gamma`; larger `decay` changes more slowly.

For every frame, the model receives a step tuple containing only:

```text
t_end_ms, buy_volume, sell_volume
```

The orderbook fields in that tuple are unused by the market-flow models.

For hurdle models:

```text
p0_next = gamma * p0_prev + (1 - gamma) * 1[V == 0]
```

Positive-volume parameters update only when `V > 0`.

For Gamma models, the estimator maintains exponentially weighted sums of
positive volumes and squared positive volumes:

```text
S  = gamma * S  + V
S2 = gamma * S2 + V^2
W  = gamma * W  + 1
mean = S / W
var  = max(param_eps^2, S2 / W - mean^2)
shape = mean^2 / var
scale = var / mean
```

For Weibull models, the estimator computes the same EWMA mean and variance,
then derives `k` from the Weibull coefficient of variation by numerical
inversion and derives `lambda` from the mean:

```text
lambda = mean / Gamma(1 + 1/k)
```

For lognormal/log-t models, the estimator applies EWMA moments to
`log(V)` for positive volumes:

```text
mu = EWMA mean(log(V))
sigma = sqrt(EWMA var(log(V)))
```

For the module-only GPD model, positive exceedances above `u` update EWMA
moments of `V - u`; `xi` and `beta` are method-of-moments estimates.

## Price Models

Price models do not predict an exchange price feed directly. They model the
remaining stochastic movement of buy/sell VWAP reference paths used by MPC.

### Local Residual Price

The local price model is `TorchResidualPriceModel`, configured by:

```json
"local_price_model": {"kind": "residual", "residual_decay": 0.95}
```

It compares two VWAP paths between consecutive local frames:

| Path | Meaning |
|------|---------|
| Realized VWAP | Buy/sell VWAP on the current observed orderbook |
| MOF-only VWAP | VWAP after consuming observed market-order flow on the previous orderbook |

The residual is:

```text
e_buy  = realized_buy_vwap  - mof_only_buy_vwap
e_sell = realized_sell_vwap - mof_only_sell_vwap
```

`reference_prices_from_transition_ticks()` computes both paths. It consumes
buy flow from asks and sell flow from bids on the previous book, then compares
that mechanically updated book with the current book. The residual therefore
captures net limit-order replenishment/cancellation, book motion, and other
price movement not explained by aggressive flow alone.

The residual model maintains EWMA mean and variance per side:

```text
mu_side_next = gamma * mu_side_prev + (1 - gamma) * e_side
sigma2_side_next = gamma * sigma2_side_prev
                 + (1 - gamma) * (e_side - mu_side_next)^2
```

`get_params()` returns per-side residual parameters plus pooled mid-style
diagnostic parameters:

```text
mu_mid = (mu_buy + mu_sell) / 2
sigma_mid = sqrt((sigma_buy^2 + sigma_sell^2) / 2)
```

The optimizer calls `sample_residual((n_samples, horizon), ...)`, which returns
independent Gaussian residual shocks for buy and sell VWAPs:

```text
eps_buy  ~ Normal(mu_buy, sigma_buy)
eps_sell ~ Normal(mu_sell, sigma_sell)
```

In local-only MPC these shocks are added to the mechanically simulated VWAP
paths after sampled market-order flow moves the book.

### Global Price

The default global price model is:

```json
"global_price_model": {"kind": "ewma_gaussian", "decay": 0.95}
```

`GlobalGaussianPriceModel` observes current global buy/sell VWAPs and learns
Gaussian deltas:

```text
d_buy  = current_global_buy_vwap  - previous_global_buy_vwap
d_sell = current_global_sell_vwap - previous_global_sell_vwap
```

It maintains EWMA first and second moments for each delta:

```text
m1_next = decay * m1_prev + (1 - decay) * d
m2_next = decay * m2_prev + (1 - decay) * d^2
sigma = max(sigma_min_quote, sqrt(max(0, m2 - m1^2)))
```

`get_params()` returns `mu_delta_buy`, `sigma_delta_buy`,
`mu_delta_sell`, and `sigma_delta_sell`. `sample_deltas()` draws Gaussian delta
shocks for rollout paths.

`global_price_model.kind = "gaussian"` builds `GaussianGlobalDeltaModel`, a
static zero-mean Gaussian increment model with configured `sigma_buy` and
`sigma_sell`.

`global_price_model.kind = "residual"` reuses `TorchResidualPriceModel` on the
global orderbook transition. That path is supported by the runtime updater, but
the joint optimizer currently consumes global price through `sample_residual()`
when present; it does not call `sample_deltas()` in `joint_optimizer.py`.

## Where The Models Are Used

### State Construction

`_state_from_frame()` computes current local/global mid, orderbook-relative
depth, observed market volumes, current buy/sell VWAPs, inventory/cash, and
`reference_price`.

`reference_price_source` selects whether consolidated wealth is marked against
the local or global sell-side VWAP path:

```text
wealth = local_cash + global_cash
       + (local_inventory + global_inventory) * reference_price
```

### Optimizer Rollouts

In `MonteCarloMPCTensorBatched.optimize()`:

```text
buy_vol, sell_vol = local_market_flow_model.sample((n_samples, horizon))
eps_buy, eps_sell = local_price_model.sample_residual((n_samples, horizon))
```

The rollout consumes sampled market-order flow from the book, applies candidate
quote actions and execution assumptions, then adds residual VWAP shocks to the
simulated buy/sell reference prices.

In joint local/global MPC (`mm/mpc/joint_optimizer.py`), local and global market
flow are both sampled when global models are active:

```text
local_buy_vol, local_sell_vol = local_market_flow_model.sample(...)
global_buy_vol, global_sell_vol = global_market_flow_model.sample(...)
local_eps_buy, local_eps_sell = local_price_model.sample_residual(...)
global_eps_buy, global_eps_sell = global_price_model.sample_residual(...)  # residual global model path
```

The global EWMA delta model exposes `sample_deltas()`. The strategy/runtime
builds and updates it, and existing docs describe it as the global VWAP delta
model. Check the active optimizer path before relying on those deltas for a
given rollout mode.

### Diagnostics And Scoring

`mm.models.scoring.score_frame_models()` scores the newest observation against
pre-update parameters:

| Score key | Observation | Score |
|-----------|-------------|-------|
| `local_market_flow` | local buy/sell volume | joint side PMF negative log likelihood |
| `global_market_flow` | global buy/sell volume | joint side PMF negative log likelihood |
| `local_price` | local mid change | Gaussian NLL using `mu_mid`, `sigma_mid` |
| `global_price` | global buy/sell VWAP deltas | Gaussian NLL using global delta params |

Scores and parameter snapshots are published on `StrategyDecision`:

```text
decision.model_updates.local_market_flow.params
decision.model_updates.local_price.params
decision.model_updates.global_market_flow.params
decision.model_updates.global_price.params
decision.model_scores[...].params_before / params_after / nll
```

Grafana dashboards consume these diagnostics through persisted decisions and
controller diagnostics events.

## Configuration Surface

Typical controller config:

```json
{
  "reference_price_source": "global",
  "local_price_model": {"kind": "residual", "residual_decay": 0.95},
  "local_market_flow_model": {"kind": "hurdle_lognormal", "decay": 0.95},
  "global_market_flow_model": {"kind": "hurdle_lognormal", "decay": 0.95},
  "global_price_model": {"kind": "ewma_gaussian", "decay": 0.95}
}
```

Important knobs:

| Knob | Applies to | Effect |
|------|------------|--------|
| `decay` | Market-flow models, global EWMA Gaussian price | Larger values adapt more slowly |
| `residual_decay` | Local/global residual price model | Larger values adapt residual mean/variance more slowly |
| `param_eps` | Market-flow models | Floors variance-like terms and logs |
| `init_p0`, `init_mu`, `init_sigma` | Hurdle lognormal/log-t | Cold-start zero probability and positive-flow log moments |
| `init_mean`, `init_shape` | Gamma/hurdle Gamma | Cold-start positive-flow Gamma moments |
| `init_mean`, `init_k` | Weibull/hurdle Weibull | Cold-start positive-flow Weibull moments |
| `sigma_min`, `sigma_min_quote` | Price models | Minimum residual/delta volatility |

`local_price_model.vwap_depth` was removed. VWAP depth is derived from current
inventory through `state.vwap_depth`.

## Practical Interpretation

Market-flow parameters answer: "How much aggressive buy/sell volume should the
optimizer expect over the next interval?"

Residual price parameters answer: "After accounting for aggressive volume
walking the displayed book, how much unexplained VWAP movement remains?"

Global price delta parameters answer: "How much does the global VWAP reference
usually move from one interval to the next?"

Together, the models let MPC roll out book state, fills, inventory, cash, and
marked wealth under many plausible next-step market paths instead of optimizing
only against the current static orderbook.
