# Decomposed Hurdle VWAP Experiments

This note documents the experiments inspired by the expert recommendation to
move from independent buy/sell VWAP residual models toward:

```text
mid / spread / VWAP-shape decomposition
quiet-vs-active residual regimes
```

The purpose is exploratory. These results are from the offline regression tool,
not production runtime.

## Motivation

The earlier models predicted buy-side and sell-side VWAP deltas independently.
That can produce structurally invalid simulated states, such as sell VWAP
crossing above buy VWAP.

The proposed decomposition is:

```text
mid_t = (best_bid_t + best_ask_t) / 2
spread_t = best_ask_t - best_bid_t

buy_vwap_shape_t = buy_vwap_t - mid_t
sell_vwap_shape_t = mid_t - sell_vwap_t
```

This separates:

```text
common market movement -> mid
liquidity / book shape -> spread and VWAP shape
```

It also allows spread and shape to be modeled as positive quantities.

## Implementation

Added experimental CLI mode:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap
```

The model recomputes VWAPs from raw orderbook levels.

For these runs:

```text
VWAP depth = 100
decay = 0.99
evaluation = online / prequential
```

At each step:

1. Predict current delta using parameters from previous samples.
2. Score RMSE and Gaussian NLL.
3. Update residual means and variances with EWMA decay.

The model compares against:

```text
zero-delta Gaussian baseline
mechanical-only VWAP delta
```

## Decomposed Hurdle Model

The model computes a mechanical state from the previous orderbook after
consuming current market flow:

```text
mechanical_mid_t
mechanical_spread_t
mechanical_buy_shape_t
mechanical_sell_shape_t
```

It then estimates residuals in decomposed space:

```text
mid_residual_t = observed_mid_t - mechanical_mid_t

log_spread_residual_t =
  log(observed_spread_t) - log(mechanical_spread_t)

log_buy_shape_residual_t =
  log(observed_buy_shape_t) - log(mechanical_buy_shape_t)

log_sell_shape_residual_t =
  log(observed_sell_shape_t) - log(mechanical_sell_shape_t)
```

Using log residuals for spread and shape keeps reconstructed spread/shape
positive.

The reconstructed predictions are:

```text
pred_mid_t = mechanical_mid_t + estimated_mid_residual

pred_buy_shape_t =
  mechanical_buy_shape_t * exp(estimated_log_buy_shape_residual)

pred_sell_shape_t =
  mechanical_sell_shape_t * exp(estimated_log_sell_shape_residual)

pred_buy_vwap_t = pred_mid_t + pred_buy_shape_t
pred_sell_vwap_t = pred_mid_t - pred_sell_shape_t
```

## Hurdle Gate

Two gate configurations were tried.

### Default Gate

The active regime is selected if:

```text
total_flow_t > 0
or abs(mechanical_mid_delta_t) > 1e-9
```

This produced active fractions around:

```text
Wallex 1h:   0.575
Nobitex 1h: 0.550
Wallex 48h: 0.639
Nobitex 48h: 0.538
```

### Stricter Gate

The active regime is selected if:

```text
total_flow_t > 500
or abs(mechanical_mid_delta_t) > 1
```

This produced active fractions around:

```text
Wallex 1h:   0.167
Nobitex 1h: 0.217
Wallex 48h: 0.271
Nobitex 48h: 0.220
```

## Results: Default Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --vwap-depth 100
```

### Latest 1h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 41.48 | 27.07 | 44.40 | 5.067 | 4.788 |
| Wallex | sell | 28.11 | 23.31 | 27.86 | 4.793 | 4.589 |
| Nobitex | buy | 42.48 | 47.18 | 42.54 | 5.161 | 5.368 |
| Nobitex | sell | 65.98 | 71.59 | 67.55 | 5.627 | 5.724 |

### Latest 48h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 102.42 | 58.76 | 118.53 | 5.656 | 5.238 |
| Wallex | sell | 61.23 | 35.57 | 63.31 | 5.358 | 4.946 |
| Nobitex | buy | 39.18 | 42.08 | 39.42 | 5.213 | 5.272 |
| Nobitex | sell | 53.99 | 57.59 | 55.57 | 5.401 | 5.416 |

## Results: Stricter Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --vwap-depth 100 \
  --hurdle-flow-threshold 500 \
  --hurdle-mechanical-threshold 1
```

### Latest 1h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 40.01 | 27.07 | 44.40 | 4.951 | 4.788 |
| Wallex | sell | 28.23 | 23.31 | 27.86 | 4.705 | 4.589 |
| Nobitex | buy | 42.59 | 47.18 | 42.54 | 5.169 | 5.368 |
| Nobitex | sell | 65.58 | 71.59 | 67.55 | 5.648 | 5.724 |

### Latest 48h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 97.31 | 58.76 | 118.53 | 5.406 | 5.238 |
| Wallex | sell | 62.70 | 35.57 | 63.31 | 5.142 | 4.946 |
| Nobitex | buy | 39.19 | 42.08 | 39.42 | 5.205 | 5.272 |
| Nobitex | sell | 53.97 | 57.59 | 55.57 | 5.385 | 5.416 |

## Interpretation

The decomposed hurdle model is structurally better than independent buy/sell
VWAP residuals because it keeps spread and shape positive.

Empirically:

```text
Nobitex improved versus zero and mechanical-only.
Wallex did not improve versus zero.
```

For Nobitex, the model improved both buy and sell VWAP over the 48h window:

```text
buy RMSE:  42.08 zero -> 39.18 model
sell RMSE: 57.59 zero -> 53.99 model
```

For Wallex, the model was worse than zero:

```text
buy RMSE:  58.76 zero -> 102.42 model
sell RMSE: 35.57 zero -> 61.23 model
```

The stricter gate reduced active-regime frequency and improved Wallex somewhat,
but not enough to beat zero.

## Likely Issue

The first decomposed hurdle implementation still applied mechanical movement
inside the quiet regime. For Wallex, mechanical movement often appears
unreliable. Therefore we tested a hard-gated version where the quiet regime
predicts near-zero movement directly, instead of:

```text
mechanical movement + quiet residual
```

A stronger next version would be:

```text
if quiet:
    pred_buy_delta = 0
    pred_sell_delta = 0
else:
    use decomposed mechanical + residual model
```

This would let the gate decide whether mechanical impact is trusted at all.

## Results: Hard-Gated Quiet Regime

Hard-gated quiet regime:

```text
if quiet:
    pred_buy_delta = 0
    pred_sell_delta = 0
else:
    use decomposed mechanical + active residual model
```

The quiet and active regimes keep separate variance estimates.

### Hard-Gated Quiet With Stricter Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --hard-gated-quiet \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --vwap-depth 100 \
  --hurdle-flow-threshold 500 \
  --hurdle-mechanical-threshold 1
```

#### Latest 1h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 39.82 | 27.07 | 44.40 | 4.942 | 4.788 |
| Wallex | sell | 27.69 | 23.31 | 27.86 | 4.683 | 4.589 |
| Nobitex | buy | 42.57 | 47.18 | 42.54 | 5.171 | 5.368 |
| Nobitex | sell | 66.96 | 71.59 | 67.55 | 5.676 | 5.724 |

#### Latest 48h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 97.31 | 58.76 | 118.53 | 5.408 | 5.238 |
| Wallex | sell | 62.59 | 35.57 | 63.31 | 5.126 | 4.946 |
| Nobitex | buy | 39.74 | 42.08 | 39.42 | 5.213 | 5.272 |
| Nobitex | sell | 54.78 | 57.59 | 55.57 | 5.386 | 5.416 |

### Hard-Gated Quiet With Default Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --hard-gated-quiet \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --vwap-depth 100
```

#### Latest 1h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 41.51 | 27.07 | 44.40 | 5.068 | 4.788 |
| Wallex | sell | 28.09 | 23.31 | 27.86 | 4.793 | 4.589 |
| Nobitex | buy | 42.45 | 47.18 | 42.54 | 5.160 | 5.368 |
| Nobitex | sell | 66.10 | 71.59 | 67.55 | 5.630 | 5.724 |

#### Latest 48h

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 102.46 | 58.76 | 118.53 | 5.662 | 5.238 |
| Wallex | sell | 61.24 | 35.57 | 63.31 | 5.357 | 4.946 |
| Nobitex | buy | 39.04 | 42.08 | 39.42 | 5.210 | 5.272 |
| Nobitex | sell | 54.14 | 57.59 | 55.57 | 5.404 | 5.416 |

### Hard-Gated Interpretation

Hard-gating quiet movement did not fix Wallex.

It slightly improved some Wallex sell-side numbers, but Wallex remained much
worse than zero-delta on both 1h and 48h windows.

For Nobitex, the hard-gated model remained useful:

```text
Nobitex 48h default gate:
  buy RMSE:  42.08 zero -> 39.04 model
  sell RMSE: 57.59 zero -> 54.14 model
```

This suggests the issue for Wallex is not only quiet-regime mechanical leakage.
The active gate itself appears to select states where mechanical/order-flow
features are still unreliable, or Wallex price changes are driven by quote
refresh/cancellation jumps not captured by the current gate.

## Related Feature Tests

Several online linear feature sets were also tested on fixed-depth VWAP.

### Mechanical + Spread Features

Features:

```text
mechanical_delta_t
spread_t
mechanical_delta_t-1
spread_t-1
```

Latest 1h, fixed depth 100, decay 0.99:

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 43.57 | 42.19 | 43.34 | 5.219 | 5.185 |
| Wallex | sell | 21.12 | 20.25 | 49.68 | 4.629 | 4.592 |
| Nobitex | buy | 30.10 | 30.54 | 27.82 | 5.050 | 4.889 |
| Nobitex | sell | 56.66 | 79.52 | 71.74 | 5.489 | 5.803 |

This feature set was useful for Nobitex sell, but not robust across all sides.

### Mechanical + Previous Delta

Features:

```text
mechanical_delta_t
delta_price_t-1
```

Latest 1h, fixed depth 100, decay 0.99:

| Venue | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|
| Wallex | buy | 44.71 | 36.96 | 37.83 | 5.239 | 5.015 |
| Wallex | sell | 21.96 | 21.57 | 51.79 | 4.622 | 4.606 |
| Nobitex | buy | 27.09 | 29.25 | 26.47 | 4.788 | 4.838 |
| Nobitex | sell | 60.78 | 69.18 | 62.44 | 5.497 | 5.628 |

This helped Nobitex, especially sell-side, but not Wallex.

### Mechanical + Signed/Total Flow + Previous Spread

Features:

```text
mechanical_delta_t
signed_flow_t
total_flow_t
spread_t-1
```

Small decay values were tested because of suspected nonstationarity.

Result:

```text
decay 0.95: mixed, not clearly better than zero
decay 0.90: unstable
decay 0.80: unstable
decay 0.50: numerically/pathologically unstable
```

Fast adaptation did not solve the problem. It made the model chase sparse jumps
and high-leverage flow observations.

## Current Recommendation

The decomposed hurdle direction is worth keeping because it solves the
structural buy/sell VWAP consistency problem.

For Nobitex, the decomposed hurdle model is promising and repeatedly beats
zero-delta on 48h online evaluation.

For Wallex, neither mechanical-quiet nor hard-gated-quiet versions beat
zero-delta. The next Wallex experiment should focus on improving the gate or
detecting quote-refresh/jump regimes, rather than adding more linear residual
features.

Potential next gates:

```text
large spread_delta_t, after timestamp audit
top-of-book depth collapse
book replacement / quote refresh indicator
large change in depth distribution with low trade flow
stale/crossed/thin book state indicator
```

The production implication may be venue-specific profiles:

```text
Nobitex:
    decomposed mechanical + hurdle residual remains promising

Wallex:
    default to zero/adaptive Gaussian unless a reliable active jump gate is found
```

## Second Gate Pass

After the expert review, two additional gate families were implemented:

```text
depth_percentile:
    buy side active if ask_depth_5_t-1 is below a depth percentile
    sell side active if bid_depth_5_t-1 is below a depth percentile

flow_to_depth:
    buy side active if buy_volume_t / ask_depth_5_t-1 exceeds threshold
    sell side active if sell_volume_t / bid_depth_5_t-1 exceeds threshold
```

The hard-gated quiet logic is side-specific:

```text
if buy side quiet:
    predicted buy VWAP delta = 0
if sell side quiet:
    predicted sell VWAP delta = 0
```

### Book Depth Collapse Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --hard-gated-quiet \
  --hurdle-gate-mode depth_percentile \
  --hurdle-depth-percentile 20 \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --online-initial-sigma 50 \
  --vwap-depth 100
```

#### Latest 1h

| Venue | Active Fraction | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---|---:|---:|---:|---:|---:|
| Wallex | 0.406 | buy | 32.04 | 27.07 | 44.40 | 4.826 | 4.788 |
| Wallex | 0.406 | sell | 27.12 | 23.31 | 27.86 | 4.722 | 4.589 |
| Nobitex | 0.344 | buy | 44.07 | 47.18 | 42.54 | 5.272 | 5.368 |
| Nobitex | 0.344 | sell | 72.34 | 71.59 | 67.55 | 5.761 | 5.724 |

#### Latest 48h

| Venue | Active Fraction | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---|---:|---:|---:|---:|---:|
| Wallex | 0.359 | buy | 83.42 | 58.76 | 118.53 | 5.445 | 5.238 |
| Wallex | 0.359 | sell | 45.54 | 35.57 | 63.31 | 5.110 | 4.946 |
| Nobitex | 0.349 | buy | 40.41 | 42.08 | 39.42 | 5.253 | 5.272 |
| Nobitex | 0.349 | sell | 55.26 | 57.59 | 55.57 | 5.394 | 5.416 |

### Flow-to-Depth Gate

Command:

```bash
python3 -m mm.tools.orderflow_price_regression \
  --online-decomposed-hurdle-vwap \
  --hard-gated-quiet \
  --hurdle-gate-mode flow_to_depth \
  --hurdle-flow-depth-ratio 0.05 \
  --lookbacks 1h,48h \
  --online-decay 0.99 \
  --online-initial-sigma 50 \
  --vwap-depth 100
```

#### Latest 1h

| Venue | Active Fraction | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---|---:|---:|---:|---:|---:|
| Wallex | 0.250 | buy | 40.58 | 27.07 | 44.40 | 4.973 | 4.788 |
| Wallex | 0.250 | sell | 27.93 | 23.31 | 27.86 | 4.611 | 4.589 |
| Nobitex | 0.267 | buy | 42.85 | 47.18 | 42.54 | 5.248 | 5.368 |
| Nobitex | 0.267 | sell | 66.64 | 71.59 | 67.55 | 5.663 | 5.724 |

#### Latest 48h

| Venue | Active Fraction | Side | Model RMSE | Zero RMSE | Mechanical RMSE | Model NLL | Zero NLL |
|---|---:|---|---:|---:|---:|---:|---:|
| Wallex | 0.331 | buy | 96.93 | 58.76 | 118.53 | 5.541 | 5.238 |
| Wallex | 0.331 | sell | 54.32 | 35.57 | 63.31 | 5.074 | 4.946 |
| Nobitex | 0.264 | buy | 39.73 | 42.08 | 39.42 | 5.212 | 5.272 |
| Nobitex | 0.264 | sell | 54.75 | 57.59 | 55.57 | 5.394 | 5.416 |

A stricter flow-to-depth threshold of `0.25` produced lower active fractions
but did not beat zero-delta either. For Wallex 48h, buy RMSE was `88.40`
and sell RMSE was `53.01`, still worse than zero-delta.

### Second Gate Interpretation

The new gates improved Wallex relative to the earlier hard-gated strict model,
but still did not beat zero-delta:

```text
Wallex 48h buy:
    zero RMSE          58.76
    strict hard gate   97.31
    depth p20 gate     83.42
    flow/depth 0.25    88.40

Wallex 48h sell:
    zero RMSE          35.57
    strict hard gate   62.59
    depth p20 gate     45.54
    flow/depth 0.25    53.01
```

The book-depth-collapse gate is the best Wallex gate tested so far, but it is
still not good enough. This supports the expert diagnosis that Wallex movement
is not well explained by aggressive flow mechanically consuming displayed
liquidity.

For Nobitex, the default hard-gated decomposed model remains the strongest
overall candidate. The added depth and flow-to-depth gates still beat zero in
some 48h cases, but they do not clearly improve over the default gate.

## Nobitex Rolling 30-Minute Evaluation

The expert suggested evaluating at least 48 non-overlapping 30-minute windows.
The current database did not contain 48 dense usable windows for this test.
Only 7 windows passed the minimum-row and continuity checks.

Improvement is reported as:

```text
baseline RMSE - model RMSE
```

Positive values mean the model is better than the baseline.

### Default Hard Gate

| Side | Baseline | Windows | Median Improvement | P25 | P75 | Worst | Positive Fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| buy | zero | 7 | 4.419 | 3.198 | 6.022 | 1.100 | 1.000 |
| sell | zero | 7 | 1.445 | 0.143 | 8.879 | -0.995 | 0.714 |
| buy | mechanical | 7 | 0.156 | -0.147 | 0.822 | -0.739 | 0.571 |
| sell | mechanical | 7 | 0.225 | 0.213 | 1.052 | 0.134 | 1.000 |

### Other Gates

| Gate | Active Mean | Buy vs Zero Median | Sell vs Zero Median | Buy vs Mechanical Median | Sell vs Mechanical Median |
|---|---:|---:|---:|---:|---:|
| strict hard | 0.219 | 3.702 | 1.251 | 0.064 | 0.125 |
| depth p20 hard | 0.371 | 1.880 | 0.495 | -1.991 | -1.197 |
| flow/depth 0.05 hard | 0.260 | 3.607 | 0.681 | -0.296 | 0.008 |

The rolling check is encouraging but not conclusive. The default hard gate beat
zero in all 7 Nobitex buy windows and 5 of 7 Nobitex sell windows. It also beat
mechanical-only in all 7 sell windows. However, because only 7 usable windows
were available, this does not yet satisfy the proposed 48-window stability bar.

## Updated Next Steps

For Wallex:

```text
1. Do not ship the decomposed mechanical hurdle model yet.
2. Treat zero/adaptive Gaussian as the current baseline to beat.
3. Continue gate research, especially quote-refresh and cancellation-like
   indicators.
4. Use spread_delta_t only as a supervised labeling diagnostic unless timestamp
   alignment proves it is available before prediction.
```

For Nobitex:

```text
1. Keep the default hard-gated decomposed model as the leading candidate.
2. Collect or backfill enough dense data for >=48 non-overlapping 30m windows.
3. Re-run the rolling distribution and inspect worst-window degradation before
   production use.
```

## Cross-Venue Wallex From Nobitex Test

We also tested whether Wallex prices are following Nobitex order-flow/book
signals.

The online model predicts Wallex current deltas from aligned Nobitex features:

```text
target:
    Wallex delta_t

features:
    1
    Nobitex mechanical_delta_t
    Nobitex mechanical_delta_t-1
    Nobitex buy_volume_t
    Nobitex sell_volume_t
    Nobitex spread_t
    Nobitex buy_volume_t-1
    Nobitex sell_volume_t-1
    Nobitex spread_t-1
```

The rows are aligned by exact timestamp. The comparison baselines are:

```text
zero-delta Gaussian
independent adaptive Gaussian mean
Wallex own mechanical-only delta
```

### Cross-Venue Results, Decay 0.99

Latest aligned timestamp:

```text
2026-06-04T14:01:20+00:00
```

| Window | Target | Model RMSE | Zero RMSE | Mean RMSE | Wallex Mechanical RMSE | Model NLL | Zero NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| 1h | best bid | 33.64 | 26.71 | 26.84 | 30.60 | 6.334 | 4.729 |
| 1h | best ask | 34.33 | 27.79 | 27.93 | 48.96 | 4.984 | 4.837 |
| 1h | buy VWAP | 32.58 | 27.11 | 27.24 | 44.47 | 4.946 | 4.783 |
| 1h | sell VWAP | 27.56 | 23.35 | 23.46 | 27.90 | 6.488 | 4.590 |
| 24h | best bid | 33.43 | 32.38 | 32.51 | 44.98 | 4.944 | 4.896 |
| 24h | best ask | 39.05 | 37.41 | 37.56 | 71.45 | 5.110 | 5.001 |
| 24h | buy VWAP | 36.10 | 34.20 | 34.33 | 64.48 | 5.026 | 4.921 |
| 24h | sell VWAP | 31.14 | 30.03 | 30.14 | 41.97 | 4.804 | 4.760 |
| 48h | best bid | 40.54 | 37.55 | 37.68 | 65.68 | 5.133 | 5.025 |
| 48h | best ask | 68.10 | 59.61 | 59.87 | 126.34 | 5.581 | 5.278 |
| 48h | buy VWAP | 64.03 | 58.77 | 59.02 | 118.55 | 5.444 | 5.238 |
| 48h | sell VWAP | 37.45 | 35.57 | 35.68 | 63.32 | 5.140 | 4.946 |

The same test with slower decay `0.999` was slightly more stable but still did
not beat zero-delta:

| Window | Target | Model RMSE | Zero RMSE | Mean RMSE | Wallex Mechanical RMSE |
|---|---|---:|---:|---:|---:|
| 24h | best bid | 32.75 | 32.38 | 32.40 | 44.98 |
| 24h | best ask | 38.30 | 37.41 | 37.43 | 71.45 |
| 24h | buy VWAP | 35.23 | 34.20 | 34.21 | 64.48 |
| 24h | sell VWAP | 30.39 | 30.03 | 30.05 | 41.97 |
| 48h | best bid | 39.59 | 37.55 | 37.57 | 65.68 |
| 48h | best ask | 62.80 | 59.61 | 59.64 | 126.34 |
| 48h | buy VWAP | 61.60 | 58.77 | 58.80 | 118.55 |
| 48h | sell VWAP | 36.70 | 35.57 | 35.59 | 63.32 |

### Cross-Venue Interpretation

Nobitex features explain some of the error in Wallex's own mechanical baseline,
because the cross-venue model is usually better than Wallex mechanical-only.
However, it does not beat the much stronger zero/adaptive-Gaussian baselines.

So the current evidence does not support adding a direct Wallex-from-Nobitex
linear regression to production. If there is cross-venue information, it is
too weak, too nonlinear, or too regime-dependent for this simple online linear
model.
