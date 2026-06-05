# Orderflow VWAP Regression Experiments

This note summarizes a set of offline tests on whether short-horizon price
changes can be improved by adding linear order-flow regressions to the current
mechanical orderbook price model.

The goal is to make the experiments reviewable by a market-microstructure or
statistical modeling expert.

## Context

The current price model is not a direct regression from order flow to price.
It first applies observed or sampled market-order flow mechanically to the
orderbook, then models the remaining unexplained movement as Gaussian residual
noise.

For one interval:

```text
observed_vwap_t = VWAP(current observed orderbook)

mechanical_vwap_t =
  VWAP(previous orderbook after consuming observed market-order flow)

residual_t = observed_vwap_t - mechanical_vwap_t
```

The runtime model currently estimates residual mean and variance with EWMA:

```text
residual_t ~ Normal(mu_side, sigma_side^2)
```

The experiments below test whether the residual or full VWAP delta can be
better explained by current and lagged trade-volume and orderbook features.

## Data

Source:

```text
Postgres table: mm_events
event_type: MarketEvent
symbol: USDT_IRT
local venue: wallex
global venue: nobitex
event interval: 10000 ms
```

The latest sampled event in the reported runs was:

```text
2026-06-03T16:00:50+00:00
```

Most tests used rolling 30 minute windows. With a 10 second event interval,
each 30 minute window has about 181 rows before lag filtering.

For train/test metrics:

```text
train split: first 70% of each rolling window
test split: last 30% of each rolling window
```

The split is chronological, not shuffled.

## Metrics And Baselines

The main metrics are:

```text
test_R2
test_RMSE
test directional accuracy
```

There are two baselines currently printed by the script:

1. Mean-delta baseline:

```text
predicted_delta_t = mean(train_delta)
```

2. Mechanical-only baseline:

```text
predicted_delta_t = mechanical_vwap_t - observed_vwap_{t-1}
```

The printed `baseline RMSE` is the mean-delta baseline, not zero delta. A
zero-delta baseline should be added:

```text
predicted_delta_t = 0
```

This is important because short-horizon price deltas usually have near-zero
mean, so beating zero-delta is a useful sanity check.

## Experiment 1: Direct Mid-Price Return Regression

Early tests modeled mid-price log return directly from signed order flow:

```text
return_t =
  beta0
  + beta1 signed_flow_t
  + beta2 signed_flow_{t-1}
  + beta3 signed_flow_{t-2}
  + beta4 signed_flow_{t-3}
  + error_t
```

where:

```text
signed_flow_t = buy_volume_t - sell_volume_t
```

Result:

- The signal was weak.
- Wallex sometimes showed a negative current signed-flow coefficient.
- Nobitex sometimes showed a positive current signed-flow coefficient.
- Smaller windows such as 15 minutes were unstable.
- This was useful diagnostically, but too weak to replace the mechanical
  orderbook model.

Interpretation:

Direct return regression loses information about book shape, depth, spread, and
nonlinear consumption of liquidity.

## Experiment 2: Top-Of-Book Next-Step Model

This model predicted the next best bid or ask delta using current order flow and
current book features.

For bid:

```text
best_bid_{t+1} - best_bid_t =
  beta0
  + beta1 sell_volume_t
  + beta2 sell_volume_t / best_bid_qty_t
  + beta3 sell_volume_t / bid_depth_5_t
  + beta4 spread_t
  + beta5 queue_imbalance_5_t
  + beta6 (microprice_t - mid_t)
  + error_t
```

For ask:

```text
best_ask_{t+1} - best_ask_t =
  beta0
  + beta1 buy_volume_t
  + beta2 buy_volume_t / best_ask_qty_t
  + beta3 buy_volume_t / ask_depth_5_t
  + beta4 spread_t
  + beta5 queue_imbalance_5_t
  + beta6 (microprice_t - mid_t)
  + error_t
```

This was the model that worked best for Nobitex in earlier tests.

Latest 30 minute Nobitex result:

| Target | Test R2 | Model RMSE | Mean Baseline RMSE |
|---|---:|---:|---:|
| best_bid next delta | 0.117 | 43.16 | 46.08 |
| best_ask next delta | 0.118 | 34.40 | 36.75 |

Interpretation:

The book-aware next-step top-of-book model found a real but modest signal on
Nobitex.

Open concern:

The meaning of "next step" must be checked carefully against event construction.
If the `MarketEvent` volume in row `t` summarizes trades that occurred between
book snapshots, then using row `t` features to predict `t+1` may accidentally
align the target better than the intended current-step model.

## Experiment 3: Current-Step Top-Of-Book Model

We then changed the target to current-step price movement:

```text
observed_delta_t = price_t - price_{t-1}
```

The first implementation used current book features from row `t`. This produced
some apparent Nobitex signal:

Latest 30 minute Nobitex result:

| Target | Test R2 | Model RMSE | Mean Baseline RMSE |
|---|---:|---:|---:|
| best_bid current delta | 0.117 | 42.78 | 45.66 |
| best_ask current delta | 0.049 | 35.40 | 36.42 |

Interpretation:

This may contain feature leakage, because current book features are observed
after the same interval whose price movement is being predicted.

## Experiment 4: Mechanical Residual For Best Bid/Ask

We tested residual modeling:

```text
residual_t = observed_price_t - mechanical_price_t
```

where:

```text
mechanical_price_t =
  price from book_{t-1} after consuming current market-order flow
```

Result:

- Raw volume residual models were not robust.
- Wallex sometimes improved.
- Nobitex generally did not improve, especially on ask-side residuals.
- Adding book features helped Nobitex bid in some windows, but not enough to be
  robust.

Important implementation fix:

When market flow consumed all stored levels, the mechanical simulator was fixed
to return the deepest available level price rather than the original best price.

## Experiment 5: VWAP Residual Model

We switched from best bid/ask to side-specific VWAP:

```text
buy_vwap  = VWAP on ask side
sell_vwap = VWAP on bid side
```

Default VWAP depth:

```text
1000 units
```

Mechanical residual target:

```text
mechanical_buy_vwap_t =
  VWAP(asks of book_{t-1} after consuming buy_volume_t)

mechanical_sell_vwap_t =
  VWAP(bids of book_{t-1} after consuming sell_volume_t)

residual_t = observed_vwap_t - mechanical_vwap_t
```

Initial residual features:

```text
buy_volume_t, sell_volume_t,
buy_volume_{t-1}, sell_volume_{t-1},
buy_volume_{t-2}, sell_volume_{t-2}
```

Representative 30 minute result:

| Venue | Target | Test R2 | Full Model RMSE | Mechanical RMSE | Mean Baseline RMSE |
|---|---|---:|---:|---:|---:|
| Wallex latest | buy_vwap | 0.215 | 63.03 | 75.01 | 71.65 |
| Wallex latest | sell_vwap | 0.109 | 55.80 | 59.25 | n/a |
| Nobitex latest | buy_vwap | n/a | 8.97 | 6.00 | n/a |
| Nobitex latest | sell_vwap | n/a | 36.48 | 35.61 | n/a |

Interpretation:

VWAP residual regression helped Wallex in some windows, especially buy-side
VWAP, but it did not beat the mechanical model on Nobitex.

## Experiment 6: Current VWAP Delta From Previous-Step Features

To avoid leakage, we tested:

```text
observed_delta_t = vwap_t - vwap_{t-1}
```

with only previous-step inputs:

```text
1,
sell_volume_{t-1},
sell_volume_{t-1} / best_bid_qty_{t-1},
sell_volume_{t-1} / bid_depth_5_{t-1},
spread_{t-1},
queue_imbalance_5_{t-1},
microprice_{t-1} - mid_{t-1}
```

Ask-side used analogous buy-side features.

Result:

| Venue | Target | Latest 30m Test R2 |
|---|---|---:|
| Nobitex | buy_vwap | -0.423 |
| Nobitex | sell_vwap | -0.111 |
| Wallex | buy_vwap | -0.004 |
| Wallex | sell_vwap | -0.018 |

Interpretation:

This formulation did not reproduce the good Nobitex next-step result. That
suggests the successful next-step model may be benefiting from event timing,
target alignment, or top-of-book behavior that does not transfer to VWAP.

## Experiment 7: Complete VWAP Delta Model

We then modeled full current VWAP delta directly, including the mechanical
component as a feature:

```text
observed_delta_t = vwap_t - vwap_{t-1}
```

Model:

```text
observed_delta_t =
  beta0
  + beta1 mechanical_delta_t
  + beta2 buy_volume_t
  + beta3 sell_volume_t
  + beta4 buy_volume_t / best_ask_qty_t
  + beta5 buy_volume_t / ask_depth_5_t
  + beta6 sell_volume_t / best_bid_qty_t
  + beta7 sell_volume_t / bid_depth_5_t
  + beta8 spread_t
  + beta9 queue_imbalance_5_t
  + beta10 (microprice_t - mid_t)
  + lagged versions
  + error_t
```

This model overfit badly in several windows, especially when depth-ratio
features were included.

Interpretation:

The depth-ratio features are highly unstable when displayed depth is small.
They can create very large feature values and high leverage points.

## Experiment 8: Complete VWAP Delta Without Depth-Ratio Features

We removed these features for all included lags:

```text
buy_volume / best_ask_qty
buy_volume / ask_depth_5
sell_volume / best_bid_qty
sell_volume / bid_depth_5
```

The remaining model is:

```text
observed_delta_t =
  beta0
  + beta1 mechanical_delta_t
  + beta2 buy_volume_t
  + beta3 sell_volume_t
  + beta4 spread_t
  + beta5 queue_imbalance_5_t
  + beta6 (microprice_t - mid_t)
  + lagged versions
  + error_t
```

### Lags = 1, Latest 30 Minutes

| Venue | Target | Test R2 | Model RMSE | Mechanical RMSE | Mean Baseline RMSE | Zero-Delta RMSE |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy_vwap | -0.030 | 71.54 | 74.34 | 71.01 | 70.61 |
| Wallex | sell_vwap | 0.325 | 46.64 | 58.90 | 56.77 | 56.81 |
| Nobitex | buy_vwap | -1.241 | 9.65 | 5.95 | 6.99 | 6.45 |
| Nobitex | sell_vwap | -0.090 | 36.46 | 35.29 | 35.03 | 34.94 |

### Lags = 2, Latest 30 Minutes

| Venue | Target | Test R2 | Model RMSE | Mechanical RMSE | Mean Baseline RMSE | Zero-Delta RMSE |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy_vwap | -0.134 | 75.75 | 75.01 | 71.65 | 71.25 |
| Wallex | sell_vwap | 0.128 | 53.33 | 59.25 | 57.12 | 57.15 |
| Nobitex | buy_vwap | -2.434 | 12.06 | 6.00 | 7.04 | 6.51 |
| Nobitex | sell_vwap | -0.086 | 36.74 | 35.61 | 35.34 | 35.26 |

Interpretation:

Removing ratio features improves stability versus the full model, but the model
still is not robust. The best recent result is Wallex sell-side VWAP with
`lags=1`, where the regression beats both mechanical-only and mean-delta
baselines, and also beats zero-delta. Nobitex still prefers mechanical-only
VWAP for buy-side in the latest window, but sell-side does not beat zero-delta;
this reinforces that zero-delta is a necessary baseline at 10 second horizons.

## Experiment 9: Online Best Bid/Ask Mechanical-Delta Model

To handle nonstationarity more realistically, we tested an online prequential
model instead of a fixed train/test split.

Target:

```text
observed_delta_t = best_price_t - best_price_{t-1}
```

For bid:

```text
mechanical_delta_t =
  best_bid_after_consuming_sell_volume_t_from_book_{t-1}
  - best_bid_{t-1}
```

For ask:

```text
mechanical_delta_t =
  best_ask_after_consuming_buy_volume_t_from_book_{t-1}
  - best_ask_{t-1}
```

Model:

```text
observed_delta_t ~ Normal(beta0_t + beta1_t mechanical_delta_t, sigma_t^2)
```

Evaluation is online:

1. Predict sample `t` using parameters from samples `< t`.
2. Score RMSE and Gaussian NLL.
3. Update `beta` with recursive least squares and forgetting factor.
4. Update residual variance with EWMA decay.

The zero-delta baseline is:

```text
observed_delta_t ~ Normal(0, sigma_zero_t^2)
```

where `sigma_zero_t^2` is an EWMA of squared deltas around zero.

We also compare against an independent adaptive Gaussian delta model with no
mechanical feature:

```text
observed_delta_t ~ Normal(mu_t, sigma_mean_t^2)
```

where `mu_t` is an EWMA mean of recent deltas and `sigma_mean_t^2` is an EWMA
variance around that adaptive mean. This is the same online Gaussian structure
as the mechanical-delta model, but with `beta1_t = 0`.

The 48 hour query did not contain a dense 48 hours of data. It produced about
3,400 scored 10 second samples per venue, roughly 9.4 hours of scored samples.

### Decay Sweep, Lookback = 48h

| Venue | Side | Decay | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| Wallex | bid | 0.90 | 114.30 | 55.58 | 72.24 | 30.665 | 112.594 |
| Wallex | ask | 0.90 | 83.70 | 69.25 | 129.52 | 11.377 | 12.175 |
| Nobitex | bid | 0.90 | 78.26 | 62.08 | 62.31 | 6.467 | 6.048 |
| Nobitex | ask | 0.90 | 59.45 | 53.53 | 52.72 | 5.900 | 5.771 |
| Wallex | bid | 0.95 | 58.61 | 55.58 | 72.24 | 15.785 | 16.664 |
| Wallex | ask | 0.95 | 76.01 | 69.25 | 129.52 | 6.494 | 6.494 |
| Nobitex | bid | 0.95 | 62.81 | 62.08 | 62.31 | 5.479 | 5.497 |
| Nobitex | ask | 0.95 | 54.81 | 53.53 | 52.72 | 5.412 | 5.353 |
| Wallex | bid | 0.98 | 56.50 | 55.58 | 72.24 | 7.490 | 7.548 |
| Wallex | ask | 0.98 | 71.79 | 69.25 | 129.52 | 5.880 | 5.858 |
| Nobitex | bid | 0.98 | 61.72 | 62.08 | 62.31 | 5.445 | 5.471 |
| Nobitex | ask | 0.98 | 52.53 | 53.53 | 52.72 | 5.361 | 5.369 |
| Wallex | bid | 0.99 | 56.04 | 55.58 | 72.24 | 6.133 | 6.149 |
| Wallex | ask | 0.99 | 70.82 | 69.25 | 129.52 | 5.727 | 5.718 |
| Nobitex | bid | 0.99 | 61.14 | 62.08 | 62.31 | 5.470 | 5.497 |
| Nobitex | ask | 0.99 | 52.18 | 53.53 | 52.72 | 5.372 | 5.397 |

### Independent Gaussian Comparison

The slower decays were the only plausible settings, so the independent
Gaussian comparison is most useful for `0.98` and `0.99`.

| Venue | Side | Decay | Mechanical-Feature RMSE | Mean-Gaussian RMSE | Zero RMSE | Mechanical-Feature NLL | Mean-Gaussian NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Wallex | bid | 0.98 | 56.50 | 55.89 | 55.58 | 7.490 | 7.515 | 7.548 |
| Wallex | ask | 0.98 | 71.79 | 69.75 | 69.25 | 5.880 | 5.869 | 5.858 |
| Nobitex | bid | 0.98 | 61.72 | 62.58 | 62.08 | 5.445 | 5.477 | 5.471 |
| Nobitex | ask | 0.98 | 52.53 | 53.93 | 53.53 | 5.361 | 5.375 | 5.369 |
| Wallex | bid | 0.99 | 56.04 | 55.75 | 55.58 | 6.133 | 6.144 | 6.149 |
| Wallex | ask | 0.99 | 70.82 | 69.51 | 69.25 | 5.727 | 5.721 | 5.718 |
| Nobitex | bid | 0.99 | 61.14 | 62.33 | 62.08 | 5.470 | 5.501 | 5.497 |
| Nobitex | ask | 0.99 | 52.18 | 53.73 | 53.53 | 5.372 | 5.400 | 5.397 |

Interpretation:

- The online framing is more appropriate than a fixed train/test split for this
  nonstationary problem.
- Faster decay `0.90` is too reactive and unstable.
- Wallex still does not clearly beat zero-delta or the independent adaptive
  Gaussian. The mechanical feature improves strongly over raw mechanical-only,
  but does not add enough information beyond an adaptive mean/variance model.
- Nobitex improves modestly over zero-delta with slower decay, especially at
  `0.98` and `0.99`.
- Nobitex also improves over the independent adaptive Gaussian, which suggests
  the top-of-book mechanical feature contains some real information there.
- Nobitex ask is the clearest case where online regression and raw mechanical
  delta are both close; raw mechanical-only is still competitive.

## Raw Data Diagnosis

After the online tests, we inspected the latest one hour of raw orderbook and
market-flow data directly. The goal was to understand why some features looked
correlated in simple analysis but failed in online prediction.

This diagnostic used:

```text
target = fixed-depth VWAP delta
depth = 100
interval = 10 seconds
lookback = latest 1 hour
venues = wallex, nobitex
```

The VWAPs were recomputed from raw orderbook levels, not taken from the
persisted `payload.vwap` field.

### Candidate Features Checked

We looked at simple correlations and online RLS behavior for:

```text
mechanical_delta_t
delta_price_t-1
buy_volume_t
sell_volume_t
signed_flow_t = buy_volume_t - sell_volume_t
total_flow_t = buy_volume_t + sell_volume_t
spread_t
spread_t-1
spread_delta_t = spread_t - spread_t-1
queue_imbalance_5_t-1
microprice_minus_mid_t-1
top/book depth features
```

### Wallex Observations

Wallex did not show a stable mechanical relationship.

For buy-side fixed-depth VWAP:

```text
corr(mechanical_delta_t, delta_price_t) ~= +0.008 over the full latest hour
```

But the rolling subwindow correlations were unstable:

```text
+0.59, +0.06, +0.25, +0.37, -0.37, -0.03
```

So a feature can look useful in part of the hour and then flip sign later.
This is a poor setting for an unconstrained online linear model.

The strongest Wallex buy-side correlation was:

```text
corr(spread_delta_t, delta_price_t) ~= +0.525
```

However, `spread_delta_t = spread_t - spread_t-1` uses the current orderbook.
For a current-step target `price_t - price_t-1`, that may contain information
from the same interval as the target. It is therefore not clearly usable as a
pre-trade predictive feature without a timestamp audit.

For sell-side Wallex, the correlations were weak:

```text
corr(mechanical_delta_t, delta_price_t) ~= +0.057
corr(buy_volume_t, delta_price_t) ~= -0.166
corr(spread_t-1, delta_price_t) ~= +0.088
corr(queue_imbalance_5_t-1, delta_price_t) ~= +0.015
corr(microprice_minus_mid_t-1, delta_price_t) ~= +0.019
```

This explains why adding queue imbalance and microprice features hurt the
online model. They were not stable enough to justify the extra degrees of
freedom.

### Nobitex Observations

Nobitex showed a clearer mechanical relationship:

```text
buy-side:
  corr(mechanical_delta_t, delta_price_t) ~= +0.355

sell-side:
  corr(mechanical_delta_t, delta_price_t) ~= +0.496
```

This is consistent with the repeated online result that
`mechanical_delta_t` helps Nobitex more than Wallex.

However, even for Nobitex there were large jump events with zero mechanical
delta:

```text
buy-side examples:
  delta_price_t ~= +361 while mechanical_delta_t = 0
  delta_price_t ~= -360 while mechanical_delta_t = 0

sell-side examples:
  delta_price_t ~= +317 while mechanical_delta_t = 0
  delta_price_t ~= +304 while mechanical_delta_t = 0
```

These events dominate RMSE but are not explained by mechanical consumption of
the previous displayed book. They may reflect quote refreshes, cancellations,
book shifts, or event-timing effects rather than aggressive-flow consumption.

### Why Correlated Features Failed Online

The main reason is that correlation was not enough. Several features showed
moderate full-window correlation but failed in prequential evaluation because:

1. The relationship changed sign across subwindows.

2. The target distribution was jumpy and sparse:

```text
many intervals had near-zero delta,
few intervals had large jumps.
```

3. Large jumps often occurred when candidate features were zero or ordinary.

4. Some apparently strong features, especially `spread_delta_t`, may use
current-book information and could leak the target interval.

5. Faster online decay made the model chase recent jumps and high-leverage
volume observations, causing RMSE and NLL to worsen.

In particular, very small decay values such as `0.90`, `0.80`, and `0.50`
were unstable for the shared linear model:

```text
mechanical_delta_t
signed_flow_t
total_flow_t
spread_t-1
```

The online model became too reactive, and its coefficients/variance estimates
were dominated by rare large moves.

### Neutral Interpretation

The data suggests two different mechanisms may be present:

1. Mechanical displayed-book consumption.

This is visible on Nobitex and partially visible in some Wallex buy-side
subwindows.

2. Book jump / quote refresh movement.

This appears to explain many large VWAP or top-of-book moves, especially when
`mechanical_delta_t = 0`. The current linear Gaussian model does not explicitly
separate this regime.

Therefore, the failure of the shared feature model does not necessarily mean
there is no signal. It may mean the model class is too simple:

```text
single Gaussian mean = beta' features
single EWMA residual variance
```

may be inadequate for data that alternates between quiet near-zero intervals
and jump intervals.

## Main Findings

1. The mechanical orderbook model is hard to beat on Nobitex VWAP.

Nobitex VWAP deltas are often predicted better by mechanical orderbook
consumption alone than by adding linear residual features.

2. Wallex shows more residual structure.

Wallex sometimes benefits from residual regression, especially for sell-side
VWAP in the latest 30 minute window.

3. The next-step top-of-book Nobitex model worked better than current-step VWAP.

This is suspicious and should be reviewed carefully. It may reflect real
predictability, but it may also reflect event timestamp alignment or the fact
that top-of-book and VWAP are very different targets.

4. Current book features can leak information.

If the feature is computed from book `t` and the target is `price_t -
price_{t-1}`, the feature may already contain the result of the interval.

5. Depth-ratio features are dangerous without clipping or transforms.

Features such as `volume / top_depth` can explode when top depth is small.
Removing them helped. Better alternatives may be clipping, log transforms,
rank transforms, or nonlinear tree models with careful validation.

6. More lags did not clearly help.

`lags=2` usually added variance. `lags=1` was cleaner in the no-ratio model.

## Recommended Next Tests

The recommended order is methodological cleanup first, then broader model
experiments. Two methodological issues could materially change the
interpretation of the current results.

### Priority 1: Fix Evaluation And Alignment

1. Add zero-delta baseline everywhere.

Report:

```text
RMSE(model)
RMSE(mechanical-only)
RMSE(train-mean-delta)
RMSE(zero-delta)
```

This should be reported for every model, venue, side, and rolling window. A
model should not be considered useful unless it beats both mechanical-only and
zero-delta on out-of-sample windows.

2. Audit timestamp alignment, especially for the Nobitex next-step model.

For each row, confirm exactly whether:

```text
volume_t
book_t
book_{t-1}
```

represent beginning-of-interval, end-of-interval, or interval-aggregate data.
This is the most important check before trusting next-step or current-step
results.

Specific question to resolve:

```text
Does volume_t occur before book_t, after book_t, or between book_{t-1} and book_t?
```

The earlier Nobitex top-of-book next-step model worked better than the
current-step VWAP models. Before drawing a modeling conclusion from that, verify
that it is not caused by an off-by-one alignment effect or by using features
that already contain information from the target interval.

### Priority 2: Broaden Rolling Evaluation

After the baseline and timestamp checks are fixed, rerun the main candidates
across many 30 minute windows, not only the latest few windows.

A reasonable minimum bar is one full trading day of non-overlapping 30 minute
windows. At a 10 second event interval, that gives roughly:

```text
24 hours * 60 minutes / 30 minutes = 48 windows
```

Use at least 48 non-overlapping windows before calling a signal stable. More is
better, especially if the venue has regime changes by time of day.

Report distributions, not only single-window results:

```text
median RMSE improvement
25th/75th percentile improvement
worst-window degradation
fraction of windows beating zero-delta
fraction of windows beating mechanical-only
```

This is necessary because the current evidence is based on too few windows to
distinguish a stable signal from a lucky recent interval.

### Priority 3: Try Mid/Spread/Shape Decomposition

The next model family to test should decompose price movement into mid, spread,
and VWAP shape rather than directly predicting buy/sell VWAP deltas.

Candidate decomposition:

```text
mid_t = (best_bid_t + best_ask_t) / 2
spread_t = best_ask_t - best_bid_t

buy_vwap_shape_t = buy_vwap_t - mid_t
sell_vwap_shape_t = mid_t - sell_vwap_t
```

Then model:

```text
mid_delta_t = model(order flow, imbalance, microprice signal)
spread_t or spread_delta_t = positive / constrained model
buy_vwap_shape_t = model(depth and ask-side shape)
sell_vwap_shape_t = model(depth and bid-side shape)
```

This may be more stable than separately regressing buy-side and sell-side VWAP
deltas, because it separates common market movement from spread/shape changes.
It also makes it easier to enforce sane relationships such as:

```text
buy_vwap_t >= mid_t >= sell_vwap_t
```

Evaluate this decomposition with the same rolling 30 minute train/test setup
and the same baselines:

```text
zero-delta
train-mean-delta
mechanical-only
```

### Priority 4: Test Regime Or Gated Models

The raw data suggests a single linear Gaussian may be mixing quiet intervals
and jump intervals. A simple next candidate is a gated model:

```text
if active_regime_t:
    delta_t ~ Normal(beta0 + beta1 mechanical_delta_t, sigma_active_t^2)
else:
    delta_t ~ Normal(0, sigma_quiet_t^2)
```

Possible gates:

```text
abs(mechanical_delta_t) > threshold
total_flow_t > threshold
spread_t-1 > threshold
thin_book_indicator_t-1
```

This would let the model avoid predicting movement during quiet states while
still using mechanical impact when orderbook consumption is informative.

Questions for expert review:

```text
Should the jump/quiet state be explicit?
Should the model be hurdle-style: P(move != 0) and size conditional on move?
Should mechanical_delta_t enter only conditional on an active regime?
```

### Priority 5: Keep Residual Modeling As The Production Candidate

Use residual-only modeling for production candidates.

Prefer:

```text
observed_delta_t = mechanical_delta_t + residual_prediction_t
```

or:

```text
residual_t = observed_delta_t - mechanical_delta_t
residual_t = beta' features + error_t
```

This preserves the causal liquidity-consumption model.

### Priority 6: Add Robust Feature Handling

For depth-ratio style variables, test:

```text
log1p(volume / depth)
clip(volume / depth, percentile limits)
volume / max(depth, floor)
indicator for thin book
```

### Priority 7: Consider Regularized Or Robust Regression

Ridge, Huber regression, or Bayesian shrinkage could help, but only after
confirming timestamp alignment and adding zero-delta baselines.

## Current Practical Conclusion

Do not replace the mechanical orderbook model with a direct linear order-flow
price model.

The most defensible direction is:

```text
VWAP_delta_t =
  mechanical_delta_t
  + residual_model(features known before or during the interval)
```

For Wallex, a simple residual model may be worth more testing. For Nobitex, the
current evidence says mechanical-only VWAP is still the stronger baseline,
except for the earlier top-of-book next-step model, whose alignment should be
audited before using it.

The immediate next step is not another feature tweak. It is to fix the
evaluation methodology: add zero-delta baselines everywhere and audit the
timestamp alignment of the successful Nobitex next-step model. After that,
evaluate the mid/spread/shape decomposition over many rolling 30 minute windows.
# Cross-Venue Delta-Lag Test

Question:

```text
Does Wallex price follow Nobitex price changes after one or more 10-second steps?
```

Implementation added to the offline tool:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --cross-venue-delta-lags \
  --local-exchange wallex \
  --global-exchange nobitex \
  --symbol USDT_IRT \
  --interval-ms 10000 \
  --lookbacks 1h,6h,24h,48h \
  --lags 2 \
  --vwap-depth 100
```

The target is:

```text
Wallex delta_price_t
```

The first pass used contemporaneous and lagged Nobitex deltas:

```text
Nobitex delta_price_t
Nobitex delta_price_t-1
Nobitex delta_price_t-2
```

A cleaner follow/lead-lag test excluded the current Nobitex delta and used only:

```text
Nobitex delta_price_t-1
Nobitex delta_price_t-2
Nobitex delta_price_t-3
```

Metrics are chronological train/test split results. Zero-delta remains the main
baseline.

## Result Summary

The relationship is weak overall.

For the lag-only model, the best recurring pocket was Wallex buy VWAP versus
lagged Nobitex buy VWAP:

| Window | Target | Test RMSE | Zero RMSE | Test R2 |
|---|---|---:|---:|---:|
| 1h | Wallex buy VWAP delta | 38.97 | 39.34 | 0.018 |
| 6h | Wallex buy VWAP delta | 33.28 | 33.36 | 0.005 |
| 24h | Wallex buy VWAP delta | 26.46 | 26.73 | 0.020 |
| 48h | Wallex buy VWAP delta | 34.51 | 34.60 | 0.005 |

This is directionally consistent, but economically small.

Most top-of-book targets did not improve over zero-delta:

| Window | Target | Test RMSE | Zero RMSE | Test R2 |
|---|---|---:|---:|---:|
| 24h | Wallex mid delta | 21.04 | 20.87 | -0.016 |
| 24h | Wallex best bid delta | 25.83 | 25.67 | -0.013 |
| 24h | Wallex best ask delta | 28.11 | 27.65 | -0.033 |
| 48h | Wallex mid delta | 29.43 | 29.46 | 0.002 |
| 48h | Wallex best bid delta | 32.40 | 32.27 | -0.008 |
| 48h | Wallex best ask delta | 39.13 | 39.04 | -0.005 |

The contemporaneous-plus-lag model showed slightly stronger in-sample
coefficients in a few cases, but this is less useful for prediction because
`Nobitex delta_t` is measured over the same 10-second interval as
`Wallex delta_t`.

## Interpretation

There is no strong evidence that Wallex top-of-book price changes are simply a
delayed regression on Nobitex price changes.

There may be a small lead-lag effect for buy-side VWAP shape or ask-side
liquidity, but the effect is much smaller than the noise:

```text
typical lag-only R2: 0.0% to 2.0%
typical RMSE improvement over zero: very small
```

This means Nobitex delta lags alone are not enough to fix the Wallex model. If
used, they should be treated as a weak auxiliary feature, probably combined with
Wallex quote-refresh/depth-change regime indicators rather than used as a
standalone predictor.

# Gated Same-Step Cross-Venue Delta Test

Question:

```text
If Wallex follows Nobitex inside the same 10-second bucket, does the relation
become useful only during active Nobitex price-move buckets?
```

Model:

```text
if abs(Nobitex_delta_t) < train_percentile_threshold:
    predicted_Wallex_delta_t = 0
else:
    predicted_Wallex_delta_t = beta0 + beta1 * Nobitex_delta_t
```

The gate threshold is estimated from the train segment only. Tested thresholds:

```text
50th, 70th, 80th, 90th, 95th percentile of abs(Nobitex_delta_t)
```

## 48h Results

Best gate per target:

| Target | Best Gate | Active Fraction | Gated RMSE | Zero RMSE | Ungated RMSE | Gated R2 |
|---|---:|---:|---:|---:|---:|---:|
| Wallex mid | p80 | 0.230 | 29.45 | 29.47 | 29.45 | 0.001 |
| Wallex best bid | p70 | 0.370 | 32.39 | 32.35 | 32.39 | -0.003 |
| Wallex best ask | p50 | 0.543 | 38.78 | 39.02 | 38.77 | 0.012 |
| Wallex buy VWAP | p50 | 0.449 | 34.59 | 34.58 | 34.54 | -0.001 |
| Wallex sell VWAP | p50 | 0.593 | 27.77 | 27.72 | 27.80 | -0.004 |

The best 48h result is still the ask-side relation:

```text
Wallex best ask from Nobitex best ask:
    gated RMSE  = 38.78
    zero RMSE   = 39.02
    ungated RMSE = 38.77
```

The gate did not improve on the ungated model for best ask. For most other
targets, the gated model was similar to or worse than zero-delta.

## Gated Cross-Venue Interpretation

The same-step Nobitex delta has a weak contemporaneous relation with Wallex,
mostly on the ask side. But hard-gating on the magnitude of Nobitex's same-step
delta does not materially strengthen the signal.

This suggests the missing Wallex feature is probably not simply:

```text
large Nobitex move -> Wallex follows
```

The more likely missing component is still Wallex-specific quote update
behavior: quote refresh, cancellations, depth collapse, or market-maker
repositioning.

# Quote-Refresh Event Diagnostic

Question:

```text
Are Wallex price jumps often book refresh/replacement events rather than
mechanical orderbook-consumption events?
```

Diagnostic definition:

```text
mechanically_quiet_t = abs(mechanical_delta_t) <= 1e-9
quiet_sigma = std(price_delta_t | mechanically_quiet_t)
large_unmechanical_t = mechanically_quiet_t and abs(price_delta_t) > 2 * quiet_sigma
strict_quote_refresh_t = large_unmechanical_t and total_flow_t == 0
```

For each strict quote-refresh candidate, the script also summarizes:

```text
spread_t-1, spread_t, spread_delta_t
bid_depth_5_t-1, bid_depth_5_t
ask_depth_5_t-1, ask_depth_5_t
UTC hour distribution
inter-event gaps
lag-1 event autocorrelation
```

## Wallex 48h Results

| Target | Mechanically Quiet | Large Unmechanical | Zero-Flow Large | Share Of Large | Lag-1 Autocorr |
|---|---:|---:|---:|---:|---:|
| mid | 71.0% | 82 / 2504 | 23 / 82 | 28.0% | 0.035 |
| best bid | 89.6% | 95 / 2504 | 15 / 95 | 15.8% | 0.061 |
| best ask | 78.7% | 88 / 2504 | 22 / 88 | 25.0% | -0.009 |
| buy VWAP | 70.3% | 81 / 2504 | 25 / 81 | 30.9% | 0.071 |
| sell VWAP | 81.7% | 73 / 2504 | 11 / 73 | 15.1% | -0.004 |

The strict no-flow quote-refresh subset exists, but it is not the majority of
large unmechanical Wallex jumps. Since `total_flow_t == 0` in 36.1% of all
Wallex 48h intervals, the no-flow share among large unmechanical events is not
especially elevated.

## Wallex Event Shape

The strongest no-flow refresh signatures are on the ask/mid/buy side.

For Wallex 48h best ask strict quote-refresh candidates:

```text
spread_before median: 203
spread_after median:  102
spread_delta median: -103

bid_depth5_before median: 1754.54
bid_depth5_after median:   789.78

ask_depth5_before median: 2041.01
ask_depth5_after median:  1295.39
```

For Wallex 48h buy VWAP strict quote-refresh candidates:

```text
spread_before median: 51
spread_after median:  56
spread_delta median:  -9

ask_depth5_before median: 657.22
ask_depth5_after median:  601.48
```

Time clustering was weak:

```text
Wallex 48h lag-1 autocorr:
    mid:       0.035
    best bid:  0.061
    best ask: -0.009
    buy VWAP:  0.071
    sell VWAP:-0.004
```

Top UTC hours for Wallex 48h were concentrated around `09:00-14:00`, but the
event counts are small enough that this should be treated as descriptive, not a
stable time-of-day signal.

## Comparison With Nobitex

The same diagnostic on Nobitex actually showed a stronger strict no-flow
quote-refresh signature:

| Target | Nobitex Zero-Flow Large / Large |
|---|---:|
| mid | 72 / 101 = 71.3% |
| best bid | 67 / 128 = 52.3% |
| best ask | 80 / 130 = 61.5% |
| buy VWAP | 63 / 107 = 58.9% |
| sell VWAP | 72 / 117 = 61.5% |

This does not mean Wallex has no quote-refresh behavior. Rather, the stricter
definition `large move + mechanical_delta=0 + total_flow=0` is not sufficient
to explain most Wallex failures.

## Diagnostic Interpretation

The result changes the hypothesis:

```text
Wallex jumps are often unmechanical, but not usually zero-flow.
```

So the next diagnostic should split large unmechanical events into:

```text
1. zero-flow refresh
2. nonzero-flow but mechanical_delta=0
3. nonzero-flow with mechanical_delta in the wrong direction
4. nonzero-flow where mechanical_delta is much smaller than realized delta
```

The second and fourth groups are likely where Wallex is failing: trades occur,
but the displayed previous book still does not explain the next displayed book.
That is consistent with quote refresh/cancellation/repositioning happening
during the same interval as trades, not only in empty intervals.

# Two-Second Interval Data

New `MarketEvent` data was added with:

```text
interval_ms = 2000
```

The purpose was to see whether shorter buckets separate market-order
consumption from limit-order refresh/replacement better than the previous
10-second data.

Important caveat:

```text
The available 2-second sample currently contains only about 750-830 usable
scored rows per venue, roughly 25-28 minutes of dense data.
```

Therefore these results are early diagnostics, not stable 24h/48h validation.

## Wallex: Mechanical-Delta-Only VWAP

Model:

```text
delta_vwap_t = beta0 + beta1 * mechanical_delta_t
```

Online decay:

```text
decay = 0.99
VWAP depth = 100
```

Wallex results:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Final beta1 |
|---|---:|---:|---:|---:|---:|---:|
| buy VWAP | 8.813 | 8.716 | 8.790 | 3.908 | 3.901 | -0.0167 |
| sell VWAP | 8.013 | 7.966 | 8.090 | 4.399 | 4.413 | -0.0077 |

The mechanical-delta-only model still did not beat zero-delta on RMSE.

## Wallex: Decomposed Hard-Gated VWAP

Model:

```text
mid / spread / shape decomposition
hard-gated quiet regime
default threshold gate
```

Wallex results:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Active Fraction |
|---|---:|---:|---:|---:|---:|---:|
| buy VWAP | 8.627 | 8.562 | 8.636 | 3.959 | 3.862 | 0.065 |
| sell VWAP | 7.960 | 7.836 | 7.958 | 4.231 | 4.367 | 0.065 |

This did not beat zero on RMSE. The sell-side NLL improved versus zero, which
suggests some variance/regime calibration benefit, but not better point
prediction.

## Wallex: Best Bid/Ask Mechanical Delta

Model:

```text
delta_best_price_t = beta0 + beta1 * mechanical_delta_t
```

Wallex results:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Final beta1 |
|---|---:|---:|---:|---:|---:|---:|
| best bid | 8.685 | 8.634 | 8.954 | 4.300 | 4.319 | -0.0009 |
| best ask | 15.835 | 15.692 | 16.268 | 4.198 | 4.192 | -0.0037 |

Top-of-book mechanical impact also did not beat zero-delta.

## Wallex: Same-Step Nobitex Cross Dependence

Model:

```text
Wallex_delta_t = beta0 + beta1 * Nobitex_delta_t
```

2-second same-step results:

| Target | Test RMSE | Zero RMSE | Test R2 | beta1 |
|---|---:|---:|---:|---:|
| mid | 5.692 | 5.670 | -0.0087 | -0.128 |
| best bid | 8.501 | 8.485 | -0.0055 | 0.016 |
| best ask | 7.546 | 7.493 | -0.0143 | -0.287 |
| buy VWAP | 1.293 | 1.275 | -0.0288 | -0.0117 |
| sell VWAP | 7.956 | 7.939 | -0.0063 | 0.014 |

Same-step Nobitex dependence did not improve Wallex out-of-sample. Some
in-sample coefficients were negative, so this is not currently a useful
Wallex predictor.

## Wallex: First Useful 2-Second Signal

The first model that improved Wallex was the lower-dimensional VWAP feature
model:

```text
delta_vwap_t =
    beta0
  + beta1 * mechanical_delta_t
  + beta2 * signed_flow_t
  + beta3 * total_flow_t
  + beta4 * spread_t-1
```

Features:

```text
mechanical_delta_t
signed_flow_t = buy_volume_t - sell_volume_t
total_flow_t = buy_volume_t + sell_volume_t
spread_t-1
```

Wallex results:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---:|---:|---:|---:|
| buy VWAP | 8.363 | 8.056 | 8.128 | 3.743 | 3.706 |
| sell VWAP | 7.166 | 7.405 | 7.518 | 4.145 | 4.188 |

This is the first clear Wallex improvement, but it is one-sided:

```text
sell VWAP improved
buy VWAP did not
```

The final Wallex sell-side beta vector was:

```text
intercept:          0.256214
mechanical_delta:   0.008024
signed_flow_t:     -0.113440
total_flow_t:      -0.115366
spread_t-1:         0.001176
```

The current-flow-only version was unstable and much worse, so the lower
dimensional signed/total/spread-prev formulation is preferred over raw
buy/sell flow at this sample size.

## Two-Second Quote-Refresh Diagnostic

The same quote-refresh diagnostic became much cleaner at 2-second resolution.

Wallex 2-second data:

```text
zero-flow intervals: 718 / 768 = 93.5%
```

Among large unmechanical jumps:

| Target | Large Unmechanical | Zero-Flow Large | Share Of Large |
|---|---:|---:|---:|
| mid | 17 | 13 | 76.5% |
| best bid | 13 | 10 | 76.9% |
| best ask | 12 | 11 | 91.7% |
| buy VWAP | 13 | 13 | 100.0% |
| sell VWAP | 16 | 14 | 87.5% |

This is much stronger evidence for quote refresh/replacement than the earlier
10-second analysis. At 10 seconds, many unmechanical jumps had nonzero flow,
probably because market orders and quote refreshes were mixed into the same
bucket. At 2 seconds, large unmechanical Wallex jumps are mostly zero-flow.

## Updated Interpretation

The 2-second data supports the decomposition:

```text
orderbook change =
    market-order consumption component
  + limit-order refresh / cancellation / replacement component
```

The market-order component is what the mechanical simulator tries to model:

```text
previous book + market orders -> mechanically reduced book -> projected price move
```

For Wallex, most large book/price changes in the 2-second data are not explained
by market-order consumption. They look more like limit-order quote refresh or
market-maker repositioning.

The next modeling implication is:

```text
1. Keep mechanical market-order impact as a separate component.
2. Add a separate quote-refresh/jump component, especially for zero-flow
   intervals.
3. Do not force the mechanical market-order model to explain refresh-dominated
   Wallex book moves.
4. Treat the sell-side 2-second signed/total/spread-prev result as promising
   but not yet production-stable until more 2-second history exists.
```

# Bitpin Data Comparison

New Bitpin data was available for:

```text
exchange = bitpin
symbol = USDT_IRT
interval_ms = 10000
rows = 1346
time range = 2026-06-04 22:19:20 UTC to 2026-06-05 02:46:40 UTC
```

Wallex did not overlap this period, so the direct comparison in this run is:

```text
Bitpin vs Nobitex over the same late 2026-06-04 / early 2026-06-05 window.
```

## Bitpin Quote-Refresh Diagnostic

Using the same definition:

```text
large_unmechanical =
    abs(mechanical_delta_t) <= 1e-9
    and abs(price_delta_t) > 2 * quiet_sigma

strict_quote_refresh =
    large_unmechanical
    and total_flow_t == 0
```

Bitpin 24h lookback, which is effectively the full available Bitpin sample:

| Target | Mechanically Quiet | Zero Flow | Large Unmechanical | Zero-Flow Large | Share Of Large |
|---|---:|---:|---:|---:|---:|
| mid | 92.0% | 72.9% | 12 / 1345 | 7 / 12 | 58.3% |
| best bid | 96.2% | 72.9% | 32 / 1345 | 16 / 32 | 50.0% |
| best ask | 95.7% | 72.9% | 14 / 1345 | 5 / 14 | 35.7% |
| buy VWAP | 92.0% | 72.9% | 13 / 1345 | 4 / 13 | 30.8% |
| sell VWAP | 94.4% | 72.9% | 23 / 1345 | 15 / 23 | 65.2% |

Interpretation:

```text
Bitpin has many mechanically quiet intervals, like the other venues.
But large unmechanical events are less frequent than Wallex 10s and Nobitex.
The zero-flow share of large unmechanical events is mixed by side.
```

One data-quality warning appeared in the Bitpin quote-refresh diagnostic:

```text
Some best-ask / buy-VWAP refresh candidates had negative spread_before values.
```

This suggests crossed/stale book snapshots or timestamp alignment issues in
some Bitpin intervals. Those intervals should be filtered or separately labeled
before production modeling.

## Bitpin Mechanical-Delta-Only VWAP

Model:

```text
delta_vwap_t = beta0 + beta1 * mechanical_delta_t
```

Online decay:

```text
decay = 0.99
VWAP depth = 100
```

Bitpin results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | beta1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 3.715 | 4.179 | 3.627 | 3.935 | 3.950 | 0.837 |
| 1h | sell VWAP | 5.380 | 3.287 | 3.234 | 3.926 | 3.897 | 1.065 |
| 4h | buy VWAP | 29.018 | 28.190 | 29.483 | 4.679 | 4.688 | 0.787 |
| 4h | sell VWAP | 6.078 | 6.110 | 6.084 | 3.782 | 3.788 | 1.045 |
| full | buy VWAP | 28.544 | 27.732 | 29.002 | 4.728 | 4.733 | 0.787 |
| full | sell VWAP | 6.047 | 6.040 | 6.014 | 3.971 | 3.977 | 1.045 |

Interpretation:

```text
Bitpin buy VWAP has a strong short-window mechanical relationship.
But over the full available sample, large jumps dominate RMSE and zero wins.

Bitpin sell VWAP mechanical_delta is near useful, but not a stable RMSE win.
```

## Bitpin Signed/Total/Spread Feature Model

Model:

```text
delta_vwap_t =
    beta0
  + beta1 * mechanical_delta_t
  + beta2 * signed_flow_t
  + beta3 * total_flow_t
  + beta4 * spread_t-1
```

Bitpin results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 3.888 | 4.179 | 3.627 | 3.939 | 3.950 |
| 1h | sell VWAP | 6.977 | 3.287 | 3.234 | 3.955 | 3.897 |
| 4h | buy VWAP | 32.393 | 28.190 | 29.483 | 4.891 | 4.688 |
| 4h | sell VWAP | 6.046 | 6.110 | 6.084 | 3.760 | 3.788 |
| full | buy VWAP | 31.844 | 27.732 | 29.002 | 4.917 | 4.733 |
| full | sell VWAP | 6.058 | 6.040 | 6.014 | 3.710 | 3.977 |

The feature model is useful mainly for Bitpin sell-side variance/NLL and mildly
for 4h sell RMSE. It is harmful for Bitpin buy VWAP over the longer sample.

## Bitpin Decomposed Hard-Gated VWAP

Model:

```text
mid / spread / shape decomposition
hard-gated quiet regime
default threshold gate
```

Bitpin results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Active Fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 3.657 | 4.192 | 3.638 | 4.270 | 3.954 | 0.285 |
| 1h | sell VWAP | 3.221 | 3.294 | 3.241 | 4.270 | 3.901 | 0.285 |
| 4h | buy VWAP | 12.255 | 8.840 | 12.509 | 4.324 | 4.134 | 0.259 |
| 4h | sell VWAP | 6.085 | 6.131 | 6.111 | 3.657 | 3.757 | 0.259 |
| full | buy VWAP | 12.066 | 8.720 | 12.321 | 4.479 | 4.169 | 0.263 |
| full | sell VWAP | 6.017 | 6.064 | 6.043 | 3.679 | 3.933 | 0.263 |

Interpretation:

```text
The decomposed model is the best Bitpin sell-side candidate so far.
It improves sell VWAP RMSE and NLL over the full available sample.

It improves buy VWAP in the latest 1h, but fails over the full sample because
large buy-side jumps dominate.
```

## Bitpin Same-Step Nobitex Cross Dependence

Model:

```text
Bitpin_delta_t = beta0 + beta1 * Nobitex_delta_t
```

Out-of-sample results were not useful. Full-sample lookback:

| Target | Test RMSE | Zero RMSE | Test R2 | beta1 |
|---|---:|---:|---:|---:|
| mid | 2.465 | 2.412 | -0.046 | -0.0058 |
| best bid | 2.884 | 2.863 | -0.016 | 0.0011 |
| best ask | 3.764 | 3.756 | -0.0048 | 0.0045 |
| buy VWAP | 3.868 | 3.869 | -0.0007 | 0.0075 |
| sell VWAP | 3.150 | 3.131 | -0.013 | 0.0018 |

Bitpin is not simply following Nobitex same-step deltas in this sample.

## Cross-Venue Comparison

Current evidence:

```text
Nobitex:
    mechanical/orderbook impact is meaningful, especially buy-side.

Wallex:
    2-second data showed strong zero-flow quote-refresh behavior.
    One promising model: sell VWAP with mechanical + signed/total/spread_prev.

Bitpin:
    sell-side decomposed hurdle model is promising.
    buy-side mechanical signal exists in the latest 1h but is not stable over
    the full available sample.
    some crossed/stale spread states need filtering.
```

Practical implication:

```text
Do not use one universal venue profile.

Nobitex:
    keep mechanical/decomposed model as leading candidate.

Wallex:
    separate market-order impact from quote-refresh jumps.

Bitpin:
    start with decomposed hard-gated sell-side model,
    filter crossed/stale books,
    collect more data before trusting buy-side behavior.
```
