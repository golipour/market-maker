# Price Horizon Modeling Notes

This note documents the current local price horizon model, improvement options,
and the recommended simpler model for a global liquidity-source venue.

## Current Local Price Horizon Model

The primary/local exchange price horizon is not modeled as a standalone process
such as `mid_next = mid + noise`. It is modeled through orderbook evolution:

```text
current orderbook
  + sampled market order flow
  + strategy quote depth
  + residual price noise
  -> simulated future VWAPs
```

The runtime builds an MPC `State` from the current primary orderbook:

- `mid`: midpoint from best bid/ask
- `ob_rel`: relative bid/ask depth around mid
- `buy_vwap`: VWAP on the ask side
- `sell_vwap`: VWAP on the bid side
- `buy_vol` and `sell_vol`: observed interval market flow
- inventory and cash from private state

During each Monte Carlo rollout step:

1. The market-flow model samples future buy and sell market-order volumes.
2. The residual price model samples buy-side and sell-side residual shocks.
3. `soft_fill_batched` simulates market-flow consumption against the orderbook
   plus the strategy's quoted depth.
4. The next market-only book is derived after consumption.
5. Next buy/sell VWAPs are computed from that simulated book.
6. Residual shocks are added to those VWAPs.
7. Wealth is marked using the simulated sell-side VWAP.

In simplified form:

```text
buy_flow_h, sell_flow_h = sample_market_flow()
eps_buy_h, eps_sell_h = sample_residuals()

book_next = consume_book(book_h, buy_flow_h, sell_flow_h, own_quotes)

buy_vwap_h = vwap_ask(book_h) + eps_buy_h
sell_vwap_h = vwap_bid(book_h) + eps_sell_h

buy_vwap_next = vwap_ask(book_next) + eps_buy_next
sell_vwap_next = vwap_bid(book_next) + eps_sell_next
```

This is valuable for the local venue because thin-book price impact matters. If
sampled buy flow consumes asks, the future buy VWAP rises. If sampled sell flow
consumes bids, the future sell VWAP falls.

## Implemented Improvement: True VWAP Residual Updates

The runtime now updates `TorchResidualPriceModel` using true VWAP residuals
instead of a simple midpoint delta.

The residual model compares:

```text
realized VWAP on the current orderbook
  vs
VWAP predicted by applying observed market flow to the previous orderbook
```

For each interval:

```text
e_buy = realized_buy_vwap - market_flow_only_buy_vwap
e_sell = realized_sell_vwap - market_flow_only_sell_vwap
```

Those residuals update the EWMA means and variances:

```text
mu_buy, sigma_buy
mu_sell, sigma_sell
```

This keeps the orderbook-fill mechanism as the structural price-impact model
and uses the residual model only for unexplained price movement and liquidity
replenishment effects.

## Further Local Model Improvements

### Joint Residual Distribution

Current residual sampling treats buy and sell residuals independently. A better
model would estimate a joint two-dimensional Gaussian:

```text
[eps_buy, eps_sell] ~ Normal(mu, covariance)
```

This would preserve correlation between bid-side and ask-side price moves and
reduce unrealistic spread behavior.

### Mid And Spread Decomposition

Another robust alternative is to model mid and spread separately:

```text
mid_next = mid_model(...)
spread_next = positive_spread_model(...)

buy_vwap_next = mid_next + spread_next / 2
sell_vwap_next = mid_next - spread_next / 2
```

This makes it easier to enforce sane bid/ask relationships.

### Spread Guard

Even before a full mid/spread model exists, the rollout should prevent impossible
or pathological sampled states, for example:

```text
buy_vwap >= sell_vwap
```

When residual shocks violate this relationship, clamp or resample according to a
clear policy.

### Limit-Order Replenishment

The current horizon mostly consumes the displayed book. It does not explicitly
simulate passive liquidity refill. A future model should add a limit-order-flow
or replenishment step:

```text
book_next = consumed_book + sampled_limit_order_replenishment
```

This matters more as horizon increases.

### Regime Awareness

Residual volatility and market-flow intensity should eventually depend on
observable regime features:

- recent volatility
- spread width
- top-of-book depth
- imbalance
- time of day
- stale-data or degraded-feed indicators

The first version can use simple EWMA parameters. Later versions can switch
between quiet, normal, and stressed regimes.

### Horizon Discipline

Without replenishment and regime modeling, long horizons can drift into
unrealistic book states. Until the model is richer, prefer short horizons such
as `1..3` for local exchange decisions.

## Recommended Global Exchange Model

The global venue should be simpler than the local venue.

In the liquidity-sourcing design, the global exchange is mostly:

- a reference price source
- a deep liquidity source
- a venue where source orders track a tight external market
- less sensitive to our small orders, assuming configured source size is modest

So the global model should not initially copy the full local orderbook-fill
simulation.

### Global Reference Price Process

Start with an EWMA Gaussian mid-price model:

```text
global_mid_next = global_mid + delta_global
delta_global ~ Normal(mu_global, sigma_global^2)
```

Track spread separately:

```text
global_spread_next = EWMA(global_spread)

global_bid_ref = global_mid_next - global_spread_next / 2
global_ask_ref = global_mid_next + global_spread_next / 2
```

This is easier to calibrate and more stable than simulating the full global
orderbook.

### Optional Local-Global Basis

For liquidity sourcing, the local/global basis is more important than the global
book microstructure:

```text
basis = local_reference_price - global_reference_price
```

Useful future model:

```text
basis_next = mean_reverting_basis_model(basis)
```

This helps the optimizer learn when cross-exchange spread is structural versus
temporary noise.

### Global Fill Model

Use a simple maker-fill probability based on quote distance from global mid:

```text
distance_bps = abs(order_price - global_mid) / global_mid * 10000
p_fill = sigmoid((reach_bps - distance_bps) / temperature)
```

or:

```text
p_fill = exp(-k * distance_bps)
```

Fill quantity should be capped by:

- target order quantity
- configured maximum source size
- observed top-depth or VWAP-depth capacity
- available global cash/base in the ledger

### Global Slippage And Capacity

For source orders that are effectively taker-like or replace-aggressive, use a
conservative slippage model:

```text
effective_buy_price = global_ask_ref + slippage_bps
effective_sell_price = global_bid_ref - slippage_bps
```

Slippage should increase with requested size relative to observed depth.

### Suggested First Implementation

Add a `GlobalReferencePriceModel` with:

- EWMA `mu_delta`
- EWMA `sigma_delta`
- EWMA spread
- optional basis tracking
- `sample_mid_path`
- `sample_bid_ask_refs`

Add a separate `GlobalExecutionModel` with:

- distance-based maker fill probability
- source-size caps
- simple slippage for aggressive orders

This keeps global modeling understandable while giving the joint MPC enough
signal to decide:

- when global inventory is cheap enough to source
- when local distribution is rich enough to quote
- when global volatility or spread makes sourcing unattractive
- when venue-specific inventory/cash constraints require backing off

## Modeling Split

The recommended split is:

```text
local exchange:
  detailed orderbook impact
  sampled market flow
  true VWAP residuals
  future replenishment model

global exchange:
  simple reference mid process
  EWMA spread
  basis model
  distance-based fill probability
  conservative capacity/slippage caps
```

This avoids overfitting the global venue and keeps the first liquidity-sourcing
MPC tractable.
