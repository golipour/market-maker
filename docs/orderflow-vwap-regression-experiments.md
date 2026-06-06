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

# Tabdeal Data Comparison

New Tabdeal data was available for:

```text
exchange = tabdeal
symbol = USDT_IRT
interval_ms = 10000
rows = 634
time range = 2026-06-05 17:13:30 UTC to 2026-06-05 19:25:30 UTC
```

The useful Tabdeal sample is about 2.2 hours. It overlaps Nobitex, but not the
current Wallex or Bitpin samples.

## Tabdeal Quote-Refresh Diagnostic

Using the same definition:

```text
large_unmechanical =
    abs(mechanical_delta_t) <= 1e-9
    and abs(price_delta_t) > 2 * quiet_sigma

strict_quote_refresh =
    large_unmechanical
    and total_flow_t == 0
```

Full available Tabdeal sample:

| Target | Mechanically Quiet | Zero Flow | Large Unmechanical | Zero-Flow Large | Share Of Large |
|---|---:|---:|---:|---:|---:|
| mid | 79.1% | 22.3% | 24 / 637 | 8 / 24 | 33.3% |
| best bid | 88.2% | 22.3% | 23 / 637 | 6 / 23 | 26.1% |
| best ask | 89.6% | 22.3% | 25 / 637 | 7 / 25 | 28.0% |
| buy VWAP | 77.4% | 22.3% | 22 / 637 | 8 / 22 | 36.4% |
| sell VWAP | 75.7% | 22.3% | 25 / 637 | 10 / 25 | 40.0% |

Interpretation:

```text
Tabdeal is much more trade-active than Wallex 2s and Bitpin.
Zero-flow intervals are only about 22%.
Large unmechanical moves are mostly not zero-flow events.
```

So Tabdeal does not look like a pure quote-refresh venue. The displayed book
still changes in ways not fully explained by mechanical consumption, but those
changes happen mostly during active-flow intervals.

## Tabdeal Mechanical-Delta-Only VWAP

Model:

```text
delta_vwap_t = beta0 + beta1 * mechanical_delta_t
```

Online decay:

```text
decay = 0.99
VWAP depth = 100
```

Tabdeal results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | beta1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 24.403 | 26.033 | 24.702 | 4.730 | 4.775 | 0.539 |
| 1h | sell VWAP | 25.361 | 27.436 | 25.267 | 5.275 | 4.808 | 0.650 |
| full | buy VWAP | 59.809 | 63.223 | 59.547 | 5.393 | 5.450 | 0.549 |
| full | sell VWAP | 43.269 | 47.590 | 43.427 | 5.123 | 5.214 | 0.660 |

Interpretation:

```text
Tabdeal has a strong mechanical market-order impact signal.
Both buy and sell VWAP beat zero-delta on RMSE over the available sample.
```

The raw mechanical prediction itself is already very strong, and the learned
coefficient is stable around:

```text
buy beta1  ~= 0.55
sell beta1 ~= 0.66
```

## Tabdeal Signed/Total/Spread Feature Model

Model:

```text
delta_vwap_t =
    beta0
  + beta1 * mechanical_delta_t
  + beta2 * signed_flow_t
  + beta3 * total_flow_t
  + beta4 * spread_t-1
```

Tabdeal results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 34.297 | 26.033 | 24.702 | 4.990 | 4.743 |
| 1h | sell VWAP | 27.372 | 27.606 | 25.266 | 5.086 | 4.810 |
| full | buy VWAP | 60.579 | 63.069 | 59.401 | 5.377 | 5.445 |
| full | sell VWAP | 43.243 | 47.527 | 43.320 | 5.120 | 5.213 |

This feature set does not improve Tabdeal buy and is not clearly better than
mechanical-only. It is acceptable on sell, but the simpler mechanical model is
cleaner.

## Tabdeal Decomposed Hard-Gated VWAP

Model:

```text
mid / spread / shape decomposition
hard-gated quiet regime
default threshold gate
```

Tabdeal results:

| Window | Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Active Fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| 1h | buy VWAP | 23.630 | 26.262 | 24.906 | 4.734 | 4.749 | 0.825 |
| 1h | sell VWAP | 24.296 | 27.701 | 25.266 | 4.761 | 4.809 | 0.825 |
| full | buy VWAP | 58.171 | 63.018 | 59.351 | 5.429 | 5.444 | 0.777 |
| full | sell VWAP | 42.110 | 47.479 | 43.250 | 5.119 | 5.211 | 0.777 |

Interpretation:

```text
This is the best Tabdeal model so far.
It beats both zero-delta and raw mechanical on both buy and sell VWAP.
```

The active fraction is high:

```text
latest 1h: 82.5%
full sample: 77.7%
```

This matches the quote-refresh diagnostic: Tabdeal is trade-active, not mostly
quiet/zero-flow.

## Tabdeal Same-Step Nobitex Cross Dependence

Model:

```text
Tabdeal_delta_t = beta0 + beta1 * Nobitex_delta_t
```

Latest 1h had a short-window signal:

| Target | Test RMSE | Zero RMSE | Test R2 | beta1 |
|---|---:|---:|---:|---:|
| mid | 25.477 | 26.054 | 0.043 | 0.0798 |
| sell VWAP | 26.410 | 28.010 | 0.111 | 0.0776 |

But this signal did not survive over the full available sample:

| Target | Test RMSE | Zero RMSE | Test R2 | beta1 |
|---|---:|---:|---:|---:|
| mid | 30.522 | 29.257 | -0.090 | -0.0053 |
| sell VWAP | 30.634 | 29.116 | -0.108 | 0.0057 |

Interpretation:

```text
Tabdeal may sometimes move with Nobitex inside the same 10-second bucket,
but this is not stable enough to use as a primary model.
```

## Updated Cross-Venue Comparison

Current evidence:

```text
Nobitex:
    mechanical/decomposed model remains strong.

Wallex:
    quote-refresh dominated in 2-second data.
    market-order impact model only has a tentative sell-side signal.

Bitpin:
    sell-side decomposed hurdle is promising.
    buy-side mechanical signal appears only in short windows and needs more
    data and crossed-book filtering.

Tabdeal:
    strongest new result.
    mechanical market-order impact is clearly useful.
    decomposed hard-gated VWAP is best on both buy and sell.
```

Practical venue profile implication:

```text
Tabdeal should be treated more like Nobitex than Wallex.

For Tabdeal:
    start from the decomposed hard-gated VWAP model,
    keep mechanical_delta as a core feature,
    do not add extra flow/spread features until more data confirms they help.
```

# Decomposed Two-Regime Mechanical-Feature Model

After comparing the venue behavior, a simpler unified model was tested:

```text
decomposed components:
    mid_delta
    log_spread_delta
    log_buy_shape_delta
    log_sell_shape_delta

market regime:
    component_delta_j ~ Normal(beta_j0 + beta_j1 * mechanical_component_delta_j,
                               sigma_market_j^2)

refresh regime:
    component_delta_j ~ Normal(mu_refresh_j, sigma_refresh_j^2)
```

The first implementation uses a hard refresh label:

```text
refresh_t =
    total_flow_t == 0
    and abs(mechanical_mid_delta_t) <= 1e-9

otherwise:
    market_t = true
```

Evaluation:

```text
online / prequential
decay = 0.99
VWAP depth = 100
lookback = all available 10000ms rows per venue
```

This is different from the older decomposed hurdle model. The older model uses
the full mechanical decomposed state as an anchor plus residual means. This new
model uses each mechanical component delta as a linear feature.

## Results

| Venue | Scored N | Market Fraction | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 21586 | 0.519 | buy | 83.827 | 83.623 | 103.711 | 6.641 | 6.934 |
| Wallex | 21586 | 0.519 | sell | 75.895 | 75.622 | 85.237 | 8.271 | 6.474 |
| Nobitex | 24497 | 0.449 | buy | 10415.2 | 10388.6 | 10388.5 | 5835.11 | 108.724 |
| Nobitex | 24497 | 0.449 | sell | 10412.8 | 10385.9 | 10385.7 | 653.855 | 135.459 |
| Bitpin | 1283 | 0.261 | buy | 11.063 | 8.720 | 12.321 | 4.440 | 4.169 |
| Bitpin | 1283 | 0.261 | sell | 7.246 | 6.064 | 6.043 | 3.710 | 3.933 |
| Tabdeal | 701 | 0.756 | buy | 56.910 | 59.592 | 56.568 | 5.391 | 5.340 |
| Tabdeal | 701 | 0.756 | sell | 41.301 | 46.371 | 41.808 | 6.899 | 5.196 |

## Interpretation

The unified two-regime model is not yet a replacement for the best existing
venue-specific models.

What worked:

```text
Tabdeal:
    improves RMSE on buy and sell versus zero.
    sell also improves versus raw mechanical.

Wallex buy:
    improves NLL versus zero, but not RMSE.

Bitpin sell:
    improves NLL versus zero, but not RMSE.
```

What failed:

```text
Nobitex all-history:
    unstable and not usable in this form.

Wallex sell:
    worse than zero on RMSE and NLL.

Bitpin buy:
    worse than zero.
```

The failure mode is important:

```text
The hard refresh gate is too crude.
```

`total_flow == 0 and mechanical_mid_delta == 0` does not capture the actual
refresh/replacement regime consistently across venues. For Wallex 2-second
data it was meaningful, but for 10-second data and for active venues like
Tabdeal, refresh-like book replacement can happen inside nonzero-flow
intervals.

The second issue is numerical/statistical:

```text
The decomposed log component RLS can produce poor NLL on long jumpy samples.
```

This showed up most clearly on all-history Nobitex. The full Nobitex history
contains enough level shifts / discontinuities / stale states that the simple
online component model is not robust without additional filtering or robust
variance updates.

## Current Conclusion

The form is still conceptually right:

```text
market-order component:
    linear in mechanical_delta

quote-refresh component:
    separate adaptive jump/refresh distribution
```

But the implementation should not use the current hard refresh rule as the
final gate.

Next version should use either:

```text
1. a soft mixture probability q_t

   delta_t ~
       (1 - q_t) * Normal(beta0 + beta1 * mechanical_delta_t, sigma_market^2)
     + q_t       * Normal(mu_refresh, sigma_refresh^2)

2. or a safer hard gate based on multiple signals:

   total_flow_t == 0
   abs(mechanical_delta_t) small
   abs(observed spread/depth replacement label) from offline training
   stale/crossed book filter
   venue-specific minimum active-flow condition
```

For now:

```text
Tabdeal:
    keep decomposed hard-gated model as best.

Nobitex:
    keep previous decomposed/mechanical model on rolling windows, not this
    all-history two-regime version.

Wallex:
    use the two-regime idea, but train a better quote-refresh gate.

Bitpin:
    use sell-side decomposed model; filter crossed/stale states before buy-side
    modeling.
```

# Soft-Mixture Gate With Continuous Sections

The next test replaced the hard refresh label with a learned soft mixture gate:

```text
q_refresh,t = sigmoid(
    a0
  + a1 * 1[total_flow_t == 0]
  + a2 * abs(mechanical_mid_delta_t)
  + a3 * spread_{t-1}
)
```

For numerical stability in the online update, the implementation uses the same
formula with scaled continuous inputs:

```text
x_t = [
    1,
    1[total_flow_t == 0],
    clip(abs(mechanical_mid_delta_t) / 1000, 0, 10),
    clip(spread_{t-1} / 1000, 0, 10)
]

q_refresh,t = sigmoid(a_t' x_t)
q_market,t = 1 - q_refresh,t
```

The model remains decomposed:

```text
mid_delta
log_spread_delta
log_buy_shape_delta
log_sell_shape_delta
```

with two components:

```text
market component:
    component_delta_j ~ Normal(beta_j0 + beta_j1 * mechanical_component_delta_j,
                               sigma_market_j^2)

refresh component:
    component_delta_j ~ Normal(mu_refresh_j, sigma_refresh_j^2)
```

The gate is updated online from posterior mixture responsibility. The market
component is updated with weighted RLS, and the refresh component is updated
with weighted EWMA.

Evaluation settings:

```text
interval = 10000ms
lookback = all available rows
continuous sections = true
section split gap > 1.5 * interval
min raw rows per section = 100
decay = 0.99
gate learning rate = 0.02
VWAP depth = 100 USDT
```

Important caveat:

```text
min raw rows per section = 100
```

does not guarantee 100 scored rows after warmup and validity filtering. Some
short sections still have very small scored samples, so the section win counts
are useful as diagnostics but not as final production evidence.

## Aggregate Results Across Continuous Sections

Here `mean_market_prob` is the average learned probability of the market-order
component, i.e. `1 - q_refresh`.

| Venue | Sections | Scored N | Mean Market Prob | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Wins vs Zero | Wins vs Mechanical |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.727 | buy | 414.662 | 51.203 | 85.407 | 4.889 | 5.139 | 17/59 | 40/59 |
| Wallex | 59 | 9564 | 0.727 | sell | 312.022 | 37.389 | 56.500 | 4.695 | 4.940 | 19/59 | 46/59 |
| Nobitex | 63 | 11699 | 0.657 | buy | 779.465 | 41.135 | 39.837 | 4.755 | 5.060 | 33/63 | 26/63 |
| Nobitex | 63 | 11699 | 0.657 | sell | 771.698 | 55.512 | 53.464 | 4.922 | 5.260 | 35/63 | 25/63 |
| Bitpin | 1 | 1071 | 0.892 | buy | 13.210 | 9.273 | 13.338 | 2.436 | 3.942 | 0/1 | 1/1 |
| Bitpin | 1 | 1071 | 0.892 | sell | 8.716 | 6.472 | 6.455 | 2.946 | 3.409 | 0/1 | 0/1 |
| Tabdeal | 1 | 594 | 0.243 | buy | 1216.000 | 51.588 | 48.945 | 5.342 | 5.187 | 0/1 | 0/1 |
| Tabdeal | 1 | 594 | 0.243 | sell | 218.133 | 40.997 | 38.479 | 5.230 | 5.091 | 0/1 | 0/1 |

## Interpretation

The soft gate does learn useful uncertainty/regime information:

```text
Wallex:
    NLL improves versus zero on both buy and sell.
    It beats mechanical RMSE in most sections.

Nobitex:
    NLL improves versus zero on both buy and sell.
    It beats zero RMSE in a little over half the sections.

Bitpin:
    NLL improves strongly, especially buy-side.
```

But the model is not point-prediction safe:

```text
Wallex:
    aggregate RMSE is much worse than zero because a few sections explode.

Nobitex:
    many sections work, but several extreme sections dominate aggregate RMSE.

Tabdeal:
    clear failure; both RMSE and NLL are worse than zero and mechanical.
```

The key lesson is that this sigmoid gate improves likelihood calibration but
does not yet control the mean prediction well enough. The log-shape
decomposition can still generate very large price-space errors when the gate
assigns meaningful probability to the wrong component.

Practical conclusion:

```text
Do not ship this soft-mixture model as the production point predictor.

Potentially keep the idea for variance / scenario generation, because NLL
improvement means the mixture is learning some distributional structure.

Before another gate experiment, add robust guards:
    minimum scored rows per section,
    stale/crossed book filters,
    caps on price-space component deltas,
    robust or bounded log-shape updates,
    and separate evaluation of point prediction vs rollout uncertainty.
```

# Soft-Mixture Failure Diagnostic And Guarded Point Mean

The bad RMSE sections were inspected with per-row diagnostics. The important
pattern was consistent across Wallex, Nobitex, and Tabdeal:

```text
posterior market responsibility ~= 0
but prior q_market often remained 0.3 to 0.7
and the market component sometimes predicted huge deltas.
```

Example from Wallex before the guard:

```text
observed buy delta      ~= 3.9
predicted buy delta     ~= 20941.6
mechanical buy delta    ~= 996
market buy delta        ~= 32010
refresh buy delta       ~= 3.7
q_market                ~= 0.654
posterior market_resp   ~= 0.000
```

So the likelihood/NLL was good because the mixture likelihood could assign the
observation to the refresh component after seeing the outcome. The RMSE was bad
because the point forecast used the prior mixture mean:

```text
point_mean = q_market * market_mean + q_refresh * refresh_mean
```

That lets an unstable market component contaminate the pre-observation point
prediction even when the likelihood later gives that component zero posterior
responsibility.

## Guard Tested

A robust point-mean guard was added. It does not change the mixture likelihood
or the online responsibility update. It only changes the point prediction used
for RMSE:

```text
if abs(market_delta_side - mechanical_delta_side)
   > cap_mult * max(spread_{t-1}, 1):

    drop the market component from the point mean

else:
    use the original soft mixture point mean
```

In shorthand:

```text
if market component looks implausible relative to mechanical:
    point_mean = refresh_mean
else:
    point_mean = q_market * market_mean + q_refresh * refresh_mean
```

This is different from changing the probabilistic model. The NLL remains the
same because the mixture likelihood is still evaluated with both components.

## Guarded Results

Settings:

```text
guard mode = drop
cap_mult = 1.0
all other settings unchanged
```

| Venue | Sections | Scored N | Side | Guarded RMSE | Unguarded RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | buy | 51.310 | 414.662 | 51.203 | 85.407 | 4.889 | 5.139 |
| Wallex | 59 | 9564 | sell | 37.344 | 312.022 | 37.389 | 56.500 | 4.695 | 4.940 |
| Nobitex | 63 | 11699 | buy | 40.716 | 779.465 | 41.135 | 39.837 | 4.755 | 5.060 |
| Nobitex | 63 | 11699 | sell | 54.922 | 771.698 | 55.512 | 53.464 | 4.922 | 5.260 |
| Bitpin | 1 | 1071 | buy | 9.185 | 13.210 | 9.273 | 13.338 | 2.436 | 3.942 |
| Bitpin | 1 | 1071 | sell | 6.369 | 8.716 | 6.472 | 6.455 | 2.946 | 3.409 |
| Tabdeal | 1 | 594 | buy | 51.704 | 1216.000 | 51.588 | 48.945 | 5.342 | 5.187 |
| Tabdeal | 1 | 594 | sell | 41.117 | 218.133 | 40.997 | 38.479 | 5.230 | 5.091 |

Threshold sensitivity:

| Venue | cap_mult | Side | Guarded RMSE | Zero RMSE |
|---|---:|---|---:|---:|
| Wallex | 0.5 | buy | 51.296 | 51.203 |
| Wallex | 0.5 | sell | 37.306 | 37.389 |
| Wallex | 2.0 | buy | 51.553 | 51.203 |
| Wallex | 2.0 | sell | 37.434 | 37.389 |
| Nobitex | 0.5 | buy | 40.820 | 41.135 |
| Nobitex | 0.5 | sell | 55.051 | 55.512 |
| Nobitex | 2.0 | buy | 40.439 | 41.135 |
| Nobitex | 2.0 | sell | 54.979 | 55.512 |

## Interpretation

The RMSE explosion had two separate causes:

```text
1. Artificial explosion:
   unstable market component contaminates the mixture mean.
   The drop guard largely fixes this.

2. Real jump error:
   observed quote-refresh jumps happen while both market and refresh means are
   near zero.
   The guard does not solve this, because there is no predictive signal in the
   current feature set for the jump arrival.
```

After the guard, the worst Wallex errors are no longer artificial 20k+ IRT
predictions. They are real observed jumps, for example:

```text
observed buy delta ~= 1118.7
predicted buy delta ~= 12.2
```

This means the next modeling problem is not "fix the mean regression"; it is
"model jump arrival risk." The current soft mixture already helps NLL because
it can carry fatter distributional uncertainty, but it cannot reliably predict
the exact direction and timing of quote-refresh jumps.

## Updated Recommendation

Use the model in two layers:

```text
1. Point / drift prediction:
   use the guarded point mean with market_guard_mode = drop.
   cap_mult around 0.5-1.0 is safer for Wallex.
   cap_mult around 1.0-2.0 is slightly better for Nobitex.

2. Rollout uncertainty:
   keep the soft mixture likelihood / variances.
   Use the refresh component as the jump-risk distribution.
```

For a single shared setting across venues:

```text
market_guard_mode = drop
cap_mult = 1.0
```

This setting is not always better than zero on every side, but it removes the
catastrophic RMSE failure while preserving most of the NLL benefit.

# Independent EM Mixture Without Spread Component

To simplify the previous model, another version was tested:

```text
No conditional sigmoid gate.
No spread component in the decomposition.
```

The state becomes:

```text
mid_t = (best_bid_t + best_ask_t) / 2

buy_shape_t = buy_vwap_t - mid_t
sell_shape_t = mid_t - sell_vwap_t
```

The modeled component deltas are:

```text
mid_delta_t
log_buy_shape_delta_t
log_sell_shape_delta_t
```

Spread is still used only as a sanity filter to avoid crossed/invalid books,
but it is not modeled and is not used as a gate feature.

The regime probability is a single learned parameter:

```text
pi_market,t = P(market regime)
```

The likelihood is:

```text
p(y_t) =
    pi_market,t       * Normal(y_t | beta0 + beta1 * mechanical_delta_t,
                               sigma_market^2)
  + (1 - pi_market,t) * Normal(y_t | mu_refresh,
                               sigma_refresh^2)
```

Online EM update:

```text
gamma_t = P(market | observed y_t)

pi_market,t+1 =
    decay * pi_market,t
  + (1 - decay) * gamma_t

market beta:
    weighted RLS with weight gamma_t

refresh mean:
    weighted EWMA with weight 1 - gamma_t
```

Settings:

```text
interval = 10000ms
continuous sections = true
decay = 0.99
initial sigma = 50
VWAP depth = 100 USDT
min raw rows per section = 100
```

Two point-prediction modes were tested:

```text
mean:
    point_delta = pi_market * market_delta
                + (1 - pi_market) * refresh_delta

map:
    point_delta = market_delta  if pi_market >= 0.5
                = refresh_delta otherwise
```

The NLL is identical for both point modes because the same mixture likelihood
is used.

## Results: Mixture Mean Point Prediction

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.409 | buy | 245.753 | 51.203 | 85.407 | 4.955 | 5.139 |
| Wallex | 59 | 9564 | 0.409 | sell | 158.610 | 37.389 | 56.500 | 4.770 | 4.940 |
| Nobitex | 63 | 11699 | 0.380 | buy | 296.051 | 41.135 | 39.837 | 4.771 | 5.060 |
| Nobitex | 63 | 11699 | 0.380 | sell | 418.051 | 55.512 | 53.464 | 4.975 | 5.260 |
| Bitpin | 1 | 1071 | 0.745 | buy | 10.476 | 9.273 | 13.338 | 3.091 | 3.942 |
| Bitpin | 1 | 1071 | 0.745 | sell | 7.382 | 6.472 | 6.455 | 3.310 | 3.409 |
| Tabdeal | 1 | 594 | 0.077 | buy | 301.567 | 51.588 | 48.945 | 5.207 | 5.187 |
| Tabdeal | 1 | 594 | 0.077 | sell | 97.475 | 40.997 | 38.479 | 5.117 | 5.091 |

The mean point predictor still has RMSE explosions. Therefore learning a
single independent `pi_market` does not by itself solve the point-prediction
problem if the point estimate averages in a bad component.

## Results: MAP / Hurdle-Style Point Prediction

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.409 | buy | 51.086 | 51.203 | 85.407 | 4.955 | 5.139 |
| Wallex | 59 | 9564 | 0.409 | sell | 37.492 | 37.389 | 56.500 | 4.770 | 4.940 |
| Nobitex | 63 | 11699 | 0.380 | buy | 40.944 | 41.135 | 39.837 | 4.771 | 5.060 |
| Nobitex | 63 | 11699 | 0.380 | sell | 55.317 | 55.512 | 53.464 | 4.975 | 5.260 |
| Bitpin | 1 | 1071 | 0.745 | buy | 14.528 | 9.273 | 13.338 | 3.091 | 3.942 |
| Bitpin | 1 | 1071 | 0.745 | sell | 9.373 | 6.472 | 6.455 | 3.310 | 3.409 |
| Tabdeal | 1 | 594 | 0.077 | buy | 51.743 | 51.588 | 48.945 | 5.207 | 5.187 |
| Tabdeal | 1 | 594 | 0.077 | sell | 41.149 | 40.997 | 38.479 | 5.117 | 5.091 |

## Interpretation

This version is much cleaner:

```text
No hand-built sigmoid gate.
No spread component.
No deviation threshold.
```

MAP point prediction solves the artificial RMSE explosion in a statistically
cleaner way than the previous deviation guard. It behaves like a true hurdle
model: use one component for the expected path, but keep the full mixture for
likelihood and simulation.

However, removing spread is not clearly free:

```text
Wallex:
    NLL still improves versus zero, but less than the spread/sigmoid version.

Nobitex:
    NLL still improves versus zero, but is slightly worse than the previous
    soft-mixture model.

Bitpin:
    NLL remains better than zero, but much worse than the previous soft-mixture
    with spread.

Tabdeal:
    still not useful; NLL and RMSE are both slightly worse than zero.
```

So the cleanest candidate so far is:

```text
Independent EM mixture probability.
No conditional sigmoid gate.
Use MAP/hurdle point prediction, not mixture mean.
But consider keeping spread in the decomposed state or at least as a
distributional feature, because removing it weakened NLL on several venues.
```

## Why Tabdeal NLL Got Worse

This result looks surprising because earlier Tabdeal experiments showed strong
mechanical dependence.

Earlier mechanical-only Tabdeal model:

```text
delta_vwap_t = beta0 + beta1 * mechanical_delta_t
```

Full-sample results:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | beta1 |
|---|---:|---:|---:|---:|---:|---:|
| buy VWAP | 59.809 | 63.223 | 59.547 | 5.393 | 5.450 | 0.549 |
| sell VWAP | 43.269 | 47.590 | 43.427 | 5.123 | 5.214 | 0.660 |

Earlier decomposed hard-gated Tabdeal model:

| Target | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL | Active Fraction |
|---|---:|---:|---:|---:|---:|---:|
| buy VWAP | 58.171 | 63.018 | 59.351 | 5.429 | 5.444 | 0.777 |
| sell VWAP | 42.110 | 47.479 | 43.250 | 5.119 | 5.211 | 0.777 |

So the mechanical signal is real. The new independent EM/no-spread model failed
for a different reason:

```text
mean_market_prob = 0.077
```

The model learned that Tabdeal was mostly refresh regime, even though the
earlier hard-gated model found an active fraction around:

```text
0.777
```

This is likely a self-starvation failure:

```text
1. market component starts weak / untrained
2. early posterior market responsibility gamma_market becomes small
3. market RLS receives almost no update weight
4. market component stays weak
5. pi_market keeps decaying toward refresh
```

In other words, the EM mixture can collapse into the refresh component before
the market component has learned the strong Tabdeal mechanical relationship.

A second issue is that the no-spread version removed an important part of the
Tabdeal book geometry. The best Tabdeal result came from:

```text
mid / spread / shape decomposition
```

not from:

```text
mid / shape only
```

So spread was probably not redundant for Tabdeal. It helped explain how the
book geometry changed when market orders interacted with displayed liquidity.

A third issue is that the EM likelihood used a joint buy/sell regime. If
mechanical explains one side well but the other side poorly in a step, the
joint market likelihood can lose and push responsibility to refresh. The
earlier mechanical-only model evaluated buy and sell separately, so it did not
penalize the market signal this way.

Updated Tabdeal interpretation:

```text
Tabdeal should not be treated as refresh-dominated.
It is one of the most mechanically active venues in the sample.
The independent EM/no-spread model underfit it because pi_market collapsed.
```

Potential fixes for the next EM version:

```text
1. initialize pi_market high, e.g. 0.75
2. enforce a pi_market floor, e.g. 0.25 or 0.40
3. use a Beta prior / pseudo-count update for pi_market
4. update market beta with a minimum responsibility weight,
   e.g. max(gamma_market, 0.05)
5. restore spread as a modeled component
6. consider separate regime responsibilities for buy and sell sides
```

# Flow-Gated Mixture With Heavy-Tailed Refresh

Another version tested a conditional `pi_market`, but kept the training
MLE/EM-based.

Instead of an independent constant:

```text
pi_market,t = constant EWMA parameter
```

the model used:

```text
pi_market,t = sigmoid(a0 + a1 * log1p(total_flow_t))
```

where:

```text
total_flow_t = buy_volume_t + sell_volume_t
```

The sigmoid parameters are trained online from the posterior market
responsibility:

```text
gamma_t = P(market | observed y_t)

a_t+1 = a_t + learning_rate * (gamma_t - pi_market,t) * x_t
```

This is an online MLE/EM-style update: the latent regime posterior `gamma_t`
acts as the soft label for the logistic gate.

The decomposed state remained the no-spread version:

```text
mid_delta_t
log_buy_shape_delta_t
log_sell_shape_delta_t
```

The market component stayed Gaussian:

```text
market:
    Normal(beta0 + beta1 * mechanical_delta_t, sigma_market^2)
```

The quote-refresh component was tested as either Gaussian or Student-t:

```text
refresh Gaussian:
    Normal(mu_refresh, sigma_refresh^2)

refresh Student-t:
    StudentT(df=3, loc=mu_refresh, scale derived from sigma_refresh)
```

Point prediction used MAP/hurdle mode:

```text
if pi_market >= 0.5:
    point_delta = market_delta
else:
    point_delta = refresh_delta
```

Settings:

```text
interval = 10000ms
continuous sections = true
decay = 0.99
gate learning rate = 0.02
initial sigma = 50
VWAP depth = 100 USDT
min raw rows per section = 100
```

## Results: Student-t Refresh, df=3

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.239 | buy | 51.944 | 51.203 | 85.407 | 4.686 | 5.139 |
| Wallex | 59 | 9564 | 0.239 | sell | 37.773 | 37.389 | 56.500 | 4.485 | 4.940 |
| Nobitex | 63 | 11699 | 0.263 | buy | 41.407 | 41.135 | 39.837 | 4.563 | 5.060 |
| Nobitex | 63 | 11699 | 0.263 | sell | 55.585 | 55.512 | 53.464 | 4.805 | 5.260 |
| Bitpin | 1 | 1071 | 0.142 | buy | 9.256 | 9.273 | 13.338 | 2.839 | 3.942 |
| Bitpin | 1 | 1071 | 0.142 | sell | 6.520 | 6.472 | 6.455 | 2.832 | 3.409 |
| Tabdeal | 1 | 594 | 0.107 | buy | 51.806 | 51.588 | 48.945 | 4.824 | 5.187 |
| Tabdeal | 1 | 594 | 0.107 | sell | 41.181 | 40.997 | 38.479 | 4.831 | 5.091 |

## Results: Gaussian Refresh

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.371 | buy | 50.961 | 51.203 | 85.407 | 5.001 | 5.139 |
| Wallex | 59 | 9564 | 0.371 | sell | 38.167 | 37.389 | 56.500 | 4.802 | 4.940 |
| Nobitex | 63 | 11699 | 0.387 | buy | 40.734 | 41.135 | 39.837 | 4.791 | 5.060 |
| Nobitex | 63 | 11699 | 0.387 | sell | 55.105 | 55.512 | 53.464 | 5.000 | 5.260 |
| Bitpin | 1 | 1071 | 0.694 | buy | 14.498 | 9.273 | 13.338 | 3.125 | 3.942 |
| Bitpin | 1 | 1071 | 0.694 | sell | 9.222 | 6.472 | 6.455 | 3.338 | 3.409 |
| Tabdeal | 1 | 594 | 0.109 | buy | 51.761 | 51.588 | 48.945 | 5.271 | 5.187 |
| Tabdeal | 1 | 594 | 0.109 | sell | 41.157 | 40.997 | 38.479 | 5.177 | 5.091 |

Student-t with `df=5` was also checked on Wallex/Nobitex. It was weaker than
`df=3` on NLL:

| Venue | df | Side | RMSE | Zero RMSE | NLL | Zero NLL |
|---|---:|---|---:|---:|---:|---:|
| Wallex | 5 | buy | 52.129 | 51.203 | 4.840 | 5.139 |
| Wallex | 5 | sell | 38.128 | 37.389 | 4.647 | 4.940 |
| Nobitex | 5 | buy | 44.853 | 41.135 | 4.673 | 5.060 |
| Nobitex | 5 | sell | 57.586 | 55.512 | 4.867 | 5.260 |

## Interpretation

The heavy-tailed refresh distribution worked for NLL:

```text
Student-t refresh improves NLL versus zero on every tested venue and side.
It is especially useful for Tabdeal, where Gaussian refresh was worse than
zero NLL but Student-t became better than zero NLL.
```

However, the flow-gated `pi_market` remained too conservative:

```text
mean_market_prob:
    Wallex  ~= 0.239
    Nobitex ~= 0.263
    Bitpin  ~= 0.142
    Tabdeal ~= 0.107
```

So the model learned a useful distributional jump/refresh component but still
under-used the mechanical market component for point prediction. This is why
RMSE mostly stayed close to zero-delta and did not recover Tabdeal's strong
mechanical dependence.

The important improvement is conceptual:

```text
The quote-refresh component should be heavy-tailed.
Gaussian refresh under-prices jump risk.
```

But for point/drift prediction:

```text
flow alone is not enough to identify the market regime.
```

Next likely improvement:

```text
Keep Student-t refresh.
Restore spread in the decomposition.
Use a stronger prior/floor for pi_market so active venues like Tabdeal do not
collapse into refresh.
Possibly use side-specific market responsibilities.
```

# Two-Gaussian Refresh Approximation

Because Student-t sampling can be more expensive inside batched MPC rollouts,
a cheaper heavy-tail approximation was tested:

```text
refresh:
    (1 - p_jump) * Normal(mu_refresh, sigma_quiet^2)
  + p_jump       * Normal(mu_refresh, sigma_jump^2)
```

This keeps the same outer market/refresh model:

```text
pi_market,t = sigmoid(a0 + a1 * log1p(total_flow_t))

market:
    Normal(beta0 + beta1 * mechanical_delta_t, sigma_market^2)

refresh:
    two-Gaussian mixture
```

Training remains online MLE/EM:

```text
gamma_market = P(market | y_t)
gamma_jump   = P(refresh_jump | y_t, refresh)

gate parameters:
    logistic online MLE update using gamma_market

market beta:
    weighted RLS using gamma_market

refresh p_jump:
    EWMA update using gamma_jump

refresh variances:
    weighted EWMA for quiet and jump components
```

Settings:

```text
interval = 10000ms
continuous sections = true
decay = 0.99
gate learning rate = 0.02
initial sigma = 50
VWAP depth = 100 USDT
point mode = MAP/hurdle
```

Two initial jump-width multipliers were tested:

```text
sigma_jump / sigma_quiet = 8
sigma_jump / sigma_quiet = 5
```

## Two-Gaussian Results, jump_mult = 8

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.394 | buy | 51.893 | 51.203 | 85.407 | 4.829 | 5.139 |
| Wallex | 59 | 9564 | 0.394 | sell | 37.858 | 37.389 | 56.500 | 4.680 | 4.940 |
| Nobitex | 63 | 11699 | 0.391 | buy | 41.123 | 41.135 | 39.837 | 4.649 | 5.060 |
| Nobitex | 63 | 11699 | 0.391 | sell | 55.472 | 55.512 | 53.464 | 4.853 | 5.260 |
| Bitpin | 1 | 1071 | 0.765 | buy | 14.642 | 9.273 | 13.338 | 2.736 | 3.942 |
| Bitpin | 1 | 1071 | 0.765 | sell | 9.402 | 6.472 | 6.455 | 3.173 | 3.409 |
| Tabdeal | 1 | 594 | 0.106 | buy | 51.819 | 51.588 | 48.945 | 4.339 | 5.187 |
| Tabdeal | 1 | 594 | 0.106 | sell | 41.187 | 40.997 | 38.479 | 4.920 | 5.091 |

## Two-Gaussian Results, jump_mult = 5

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.384 | buy | 51.938 | 51.203 | 85.407 | 4.817 | 5.139 |
| Wallex | 59 | 9564 | 0.384 | sell | 37.872 | 37.389 | 56.500 | 4.670 | 4.940 |
| Nobitex | 63 | 11699 | 0.383 | buy | 41.140 | 41.135 | 39.837 | 4.630 | 5.060 |
| Nobitex | 63 | 11699 | 0.383 | sell | 55.582 | 55.512 | 53.464 | 4.833 | 5.260 |
| Bitpin | 1 | 1071 | 0.764 | buy | 14.644 | 9.273 | 13.338 | 2.741 | 3.942 |
| Bitpin | 1 | 1071 | 0.764 | sell | 9.400 | 6.472 | 6.455 | 3.175 | 3.409 |
| Tabdeal | 1 | 594 | 0.106 | buy | 51.819 | 51.588 | 48.945 | 4.306 | 5.187 |
| Tabdeal | 1 | 594 | 0.106 | sell | 41.187 | 40.997 | 38.479 | 4.878 | 5.091 |

## Student-t vs Two-Gaussian

Best Student-t result used:

```text
df = 3
```

Comparison by NLL:

| Venue | Side | Student-t NLL | Two-Gauss NLL, mult=8 | Two-Gauss NLL, mult=5 | Zero NLL |
|---|---|---:|---:|---:|---:|
| Wallex | buy | 4.686 | 4.829 | 4.817 | 5.139 |
| Wallex | sell | 4.485 | 4.680 | 4.670 | 4.940 |
| Nobitex | buy | 4.563 | 4.649 | 4.630 | 5.060 |
| Nobitex | sell | 4.805 | 4.853 | 4.833 | 5.260 |
| Bitpin | buy | 2.839 | 2.736 | 2.741 | 3.942 |
| Bitpin | sell | 2.832 | 3.173 | 3.175 | 3.409 |
| Tabdeal | buy | 4.824 | 4.339 | 4.306 | 5.187 |
| Tabdeal | sell | 4.831 | 4.920 | 4.878 | 5.091 |

Comparison by RMSE:

| Venue | Side | Student-t RMSE | Two-Gauss RMSE, mult=8 | Two-Gauss RMSE, mult=5 | Zero RMSE |
|---|---|---:|---:|---:|---:|
| Wallex | buy | 51.944 | 51.893 | 51.938 | 51.203 |
| Wallex | sell | 37.773 | 37.858 | 37.872 | 37.389 |
| Nobitex | buy | 41.407 | 41.123 | 41.140 | 41.135 |
| Nobitex | sell | 55.585 | 55.472 | 55.582 | 55.512 |
| Bitpin | buy | 9.256 | 14.642 | 14.644 | 9.273 |
| Bitpin | sell | 6.520 | 9.402 | 9.400 | 6.472 |
| Tabdeal | buy | 51.806 | 51.819 | 51.819 | 51.588 |
| Tabdeal | sell | 41.181 | 41.187 | 41.187 | 40.997 |

## Interpretation

The two-Gaussian refresh is a viable cheaper approximation:

```text
It improves NLL versus zero on every venue and side.
It avoids Student-t sampling cost.
It is simple to sample in batched Torch:
    Bernoulli mask + one Normal draw + sigma selection.
```

But Student-t remains the stronger general distributional model:

```text
Student-t wins NLL on Wallex buy/sell, Nobitex buy/sell, Bitpin sell,
and Tabdeal sell.
```

Two-Gaussian wins some one-sided cases:

```text
Bitpin buy
Tabdeal buy
```

The point RMSE story is mostly unchanged. The flow-gated model still underuses
the mechanical market component, so RMSE remains near zero-delta and does not
recover the known Tabdeal mechanical advantage.

Practical production tradeoff:

```text
If rollout speed is the priority:
    use two-Gaussian refresh, jump_mult around 5.

If distributional fit / NLL is the priority:
    use Student-t refresh with df around 3.

Either way:
    improve the market gate / pi_market prior before expecting better RMSE.
```

# Flow-Gated Mixture With Spread Gate Features

The previous flow-gated Student-t model used:

```text
pi_market,t = sigmoid(a0 + a1 * log1p(total_flow_t))
```

Two spread-feature variants were tested:

```text
prev:
    pi_market,t =
        sigmoid(a0
              + a1 * log1p(total_flow_t)
              + a2 * log1p(spread_{t-1}))

both:
    pi_market,t =
        sigmoid(a0
              + a1 * log1p(total_flow_t)
              + a2 * log1p(spread_{t-1})
              + a3 * abs(mechanical_log_spread_delta_t))
```

where:

```text
mechanical_log_spread_delta_t =
    log(mechanical_spread_t) - log(spread_{t-1})
```

The refresh distribution remained:

```text
Student-t, df = 3
```

and point prediction remained:

```text
MAP / hurdle mode
```

## Results Compared With No Spread Gate Feature

| Venue | Spread Features | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Wallex | none | 0.239 | buy | 51.944 | 51.203 | 85.407 | 4.686 | 5.139 |
| Wallex | none | 0.239 | sell | 37.773 | 37.389 | 56.500 | 4.485 | 4.940 |
| Wallex | prev | 0.111 | buy | 51.467 | 51.203 | 85.407 | 4.629 | 5.139 |
| Wallex | prev | 0.111 | sell | 37.593 | 37.389 | 56.500 | 4.417 | 4.940 |
| Wallex | both | 0.111 | buy | 51.466 | 51.203 | 85.407 | 4.629 | 5.139 |
| Wallex | both | 0.111 | sell | 37.598 | 37.389 | 56.500 | 4.417 | 4.940 |
| Nobitex | none | 0.263 | buy | 41.407 | 41.135 | 39.837 | 4.563 | 5.060 |
| Nobitex | none | 0.263 | sell | 55.585 | 55.512 | 53.464 | 4.805 | 5.260 |
| Nobitex | prev | 0.109 | buy | 42.853 | 41.135 | 39.837 | 4.506 | 5.060 |
| Nobitex | prev | 0.109 | sell | 56.940 | 55.512 | 53.464 | 4.768 | 5.260 |
| Nobitex | both | 0.110 | buy | 42.710 | 41.135 | 39.837 | 4.506 | 5.060 |
| Nobitex | both | 0.110 | sell | 56.861 | 55.512 | 53.464 | 4.767 | 5.260 |
| Bitpin | none | 0.142 | buy | 9.256 | 9.273 | 13.338 | 2.839 | 3.942 |
| Bitpin | none | 0.142 | sell | 6.520 | 6.472 | 6.455 | 2.832 | 3.409 |
| Bitpin | prev | 0.056 | buy | 9.267 | 9.273 | 13.338 | 2.757 | 3.942 |
| Bitpin | prev | 0.056 | sell | 6.524 | 6.472 | 6.455 | 2.797 | 3.409 |
| Bitpin | both | 0.056 | buy | 9.267 | 9.273 | 13.338 | 2.757 | 3.942 |
| Bitpin | both | 0.056 | sell | 6.524 | 6.472 | 6.455 | 2.797 | 3.409 |
| Tabdeal | none | 0.107 | buy | 51.806 | 51.588 | 48.945 | 4.824 | 5.187 |
| Tabdeal | none | 0.107 | sell | 41.181 | 40.997 | 38.479 | 4.831 | 5.091 |
| Tabdeal | prev | 0.038 | buy | 51.798 | 51.588 | 48.945 | 4.752 | 5.187 |
| Tabdeal | prev | 0.038 | sell | 41.177 | 40.997 | 38.479 | 4.765 | 5.091 |
| Tabdeal | both | 0.038 | buy | 51.798 | 51.588 | 48.945 | 4.752 | 5.187 |
| Tabdeal | both | 0.038 | sell | 41.177 | 40.997 | 38.479 | 4.764 | 5.091 |

## Interpretation

Adding previous spread to the gate improves NLL almost everywhere:

```text
Wallex:
    buy NLL  4.686 -> 4.629
    sell NLL 4.485 -> 4.417

Nobitex:
    buy NLL  4.563 -> 4.506
    sell NLL 4.805 -> 4.768

Bitpin:
    buy NLL  2.839 -> 2.757
    sell NLL 2.832 -> 2.797

Tabdeal:
    buy NLL  4.824 -> 4.752
    sell NLL 4.831 -> 4.765
```

But it also lowers the learned market probability:

```text
Wallex:  0.239 -> 0.111
Nobitex: 0.263 -> 0.109
Bitpin:  0.142 -> 0.056
Tabdeal: 0.107 -> 0.038
```

That means spread helps the distributional model identify jump/refresh risk,
but it makes the gate even more conservative about using the market-mechanical
component. RMSE therefore remains near zero-delta and does not recover the
mechanical dependence, especially on Tabdeal.

The mechanical spread-delta feature adds almost nothing beyond previous spread:

```text
prev ~= both
```

So the useful spread signal is mainly the existing book state:

```text
spread_{t-1}
```

# Decay-Only Binned Flow Gate

To remove the extra `gate_learning_rate` hyperparameter, a decay-only gate was
tested.

Instead of:

```text
pi_market,t = sigmoid(a0 + a1 * log1p(total_flow_t))
```

with a separate logistic learning rate, the gate uses bins of
`log1p(total_flow_t)`:

```text
bin_t = bucket(log1p(total_flow_t), edges=[0, 2, 4, 6, 8])
pi_market,t = pi_market[bin_t]
```

After observing the interval, the mixture likelihood gives the posterior market
responsibility:

```text
gamma_market,t = P(market | observed buy/sell VWAP deltas)
```

Only the selected bin is updated, using the same decay as the rest of the
online model:

```text
pi_market[bin_t] =
    decay * pi_market[bin_t]
  + (1 - decay) * gamma_market,t
```

Settings:

```text
interval = 10000ms
continuous sections = true
decay = 0.99
initial sigma = 50
VWAP depth = 100 USDT
point mode = MAP/hurdle
refresh distribution = Student-t(df=3)
spread gate features = none
```

## Results

| Venue | Sections | Scored N | Mean Market Prob | Side | RMSE | Zero RMSE | Mechanical RMSE | NLL | Zero NLL | Wins vs Zero | Wins vs Mechanical |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Wallex | 59 | 9564 | 0.368 | buy | 52.383 | 51.203 | 85.407 | 4.684 | 5.139 | 13/59 | 46/59 |
| Wallex | 59 | 9564 | 0.368 | sell | 38.538 | 37.389 | 56.500 | 4.479 | 4.940 | 19/59 | 50/59 |
| Nobitex | 63 | 11699 | 0.349 | buy | 58.324 | 41.135 | 39.837 | 4.555 | 5.060 | 19/63 | 17/63 |
| Nobitex | 63 | 11699 | 0.349 | sell | 56.155 | 55.512 | 53.464 | 4.808 | 5.260 | 14/63 | 17/63 |
| Bitpin | 1 | 1071 | 0.137 | buy | 17.722 | 9.273 | 13.338 | 2.811 | 3.942 | 0/1 | 0/1 |
| Bitpin | 1 | 1071 | 0.137 | sell | 11.232 | 6.472 | 6.455 | 2.840 | 3.409 | 0/1 | 0/1 |
| Tabdeal | 1 | 594 | 0.222 | buy | 108.067 | 51.588 | 48.945 | 4.935 | 5.187 | 0/1 | 0/1 |
| Tabdeal | 1 | 594 | 0.222 | sell | 42.875 | 40.997 | 38.479 | 4.949 | 5.091 | 0/1 | 0/1 |

## Interpretation

The binned EWMA gate achieves the goal of removing the separate gate learning
rate, and it keeps NLL better than zero on every venue and side.

However, it does not solve the point-prediction problem:

```text
Wallex:
    NLL is slightly better than the sigmoid Student-t gate.
    RMSE is worse than zero on both sides.
    The model still beats the raw mechanical baseline in most sections because
    Wallex mechanical deltas are often toxic.

Nobitex:
    NLL is slightly better than the sigmoid Student-t gate.
    Buy-side RMSE becomes much worse, suggesting the binned gate over-uses the
    market component in some jumpy sections.

Bitpin:
    NLL remains strong, but RMSE deteriorates badly.

Tabdeal:
    NLL remains better than zero, but RMSE fails badly on buy.
    This is another sign that the current market/refresh point selection is not
    recovering Tabdeal's mechanical dependence.
```

Compared with the sigmoid Student-t gate, mean market probability rose:

```text
Wallex:  0.239 -> 0.368
Nobitex: 0.263 -> 0.349
Bitpin:  0.142 -> 0.137
Tabdeal: 0.107 -> 0.222
```

This is useful diagnostically: simply increasing market probability does not
guarantee better RMSE. The larger issue is the MAP point rule and the fact that
the market and refresh components can imply very different point moves even
when the likelihood remains well calibrated.

Practical conclusion:

```text
The binned gate is attractive for runtime simplicity because it uses only the
main decay parameter.

But as tested here, it should not replace the sigmoid gate for point prediction.
It is acceptable as a simpler probability/NLL model, not yet as a better
portfolio-valuation point model.
```

not the simulated mechanical spread change.

Updated conclusion:

```text
Use spread_{t-1} as a distributional / regime-risk feature.
Do not expect it alone to improve point RMSE.
To recover mechanical alpha, combine spread features with a pi_market prior,
floor, or side-specific mechanical responsibility.
```
