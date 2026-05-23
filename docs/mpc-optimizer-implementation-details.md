# MPC Optimizer Implementation Details

This document describes the implemented optimizer used by the low-frequency market maker controller. It focuses on `MonteCarloMPCTensorBatched` in `mm/mpc/mpc_tensor_batched.py` and its integration through `MpcStrategyEngine` in `mm/strategies/mpc_engine.py`.

The optimizer is a tensor-batched Monte Carlo model predictive controller. It does not choose exchange orders directly. It chooses continuous cumulative bid and ask depth curves over spread from mid, simulates future market-flow paths, evaluates expected terminal objective, and returns the best first-step quote-depth action. The strategy layer later turns that continuous curve into a discrete ladder of limit orders.

## Integration Point

The production call path is:

```text
StrategyRunner.on_frame
  -> clean_strategy_frame
  -> MpcStrategyEngine.decide
  -> _state_from_frame
  -> score/update online models
  -> MonteCarloMPCTensorBatched.optimize(state)
  -> build_orders_from_sigmoid_with_report
  -> StrategyDecision + ControllerDiagnostics
```

The optimizer public API is:

```python
action = optimizer.optimize(state)
```

Compatibility API with diagnostics:

```python
action, diagnostics = optimizer.optimize_with_diagnostics(state)
optimizer.last_diagnostics
```

`optimize()` is a wrapper around `optimize_with_diagnostics()`.

## Input State

The optimizer receives `mm.mpc.mpc_state.State`.

| Field | Meaning |
| --- | --- |
| `mid` | Current primary mid price, quote per base |
| `ob_rel` | Cleaned market-only orderbook as `(spread, qty)` tuples |
| `buy_vol`, `sell_vol` | Realized interval primary market-flow volumes |
| `inventory`, `cash` | Primary venue base inventory and quote cash |
| `buy_vwap`, `sell_vwap` | Primary reference VWAPs |
| `hedge_buy_vwap`, `hedge_sell_vwap` | Optional hedge venue VWAP references |
| `hedge_inventory`, `hedge_cash` | Optional hedge venue inventory/cash |

`MpcStrategyEngine._state_from_frame()` builds this after own-order subtraction, so `ob_rel` should represent market liquidity excluding the bot's own resting orders.

Reference VWAP depth is controlled by controller `vwap_depth`, falling back to the price model target depth.

## Orderbook Projection

The optimizer projects the cleaned relative book onto a fixed spread grid:

```text
grid_levels = ceil(spread_max / spread_step)
spread_grid = [spread_step, 2 * spread_step, ..., grid_levels * spread_step]
```

Book projection uses two representations:

- cumulative tensors from `OrderbookRelative` for the first fill simulation
- marginal per-bucket arrays for simulated future book state

Bid spread is:

```text
spread = mid - bid_price
```

Ask spread is:

```text
spread = ask_price - mid
```

Only non-negative spreads inside `spread_max` are kept.

Bucket policy:

```text
bucket index = ceil(spread / spread_step) - 1
```

## Continuous Action Parameterization

The optimizer chooses cumulative quote depth as a mixture of sigmoids for each side:

```text
D_side(spread) = sum_i L_i * sigmoid(k_i * (spread - x0_i))
```

For each component:

| Parameter | Meaning |
| --- | --- |
| `L` | depth scale |
| `k` | slope/sharpness |
| `x0` | center spread |

One horizon-step action has this flat layout:

```text
[bid_L1, bid_k1, bid_x0_1, ..., bid_Ln, bid_k_n, bid_x0_n,
 ask_L1, ask_k1, ask_x0_1, ..., ask_Ln, ask_k_n, ask_x0_n]
```

For horizon `H` and `n_components = n`:

```text
len(x_torch) = H * 6 * n
```

Only the first horizon action is returned. Later horizon actions are planning variables.

## Effective Depth Convention

Raw sigmoid mixtures can have non-zero depth at spread zero. The implemented executable convention is:

```text
effective_depth(spread) = max(0, raw_depth(spread) - raw_depth(0))
```

This convention is used by:

- soft fill simulation
- quote-size penalty
- quoted-spread penalty
- optimizer action diagnostics
- ladder order conversion

Reviewers should treat effective depth as the true cumulative order size. Raw depth is useful for optimization parameters and warm starts, not direct order sizing.

## Positive Parameters And Bounds

`L` and `k` are optimized in raw unconstrained tensor space and converted to positive values with `softplus`.

In the default parameterization, bounds are specified in effective positive parameter space:

```text
L in [1e-6, L_max]
k in [1e-6, k_max]
x0 in [-x0_max, x0_max]
```

They are converted to raw optimizer-space bounds by `_bounds_effective_to_raw()`. Each closure clamps `x_torch` to raw bounds before rollout.

When `quote_size_softmax_L=true`, `L` slots are interpreted as logits instead of softplus depth scales. In that mode, `L` bounds pass through as logit bounds and `mm.mpc.l_param` materializes effective `(L, k, x0)` values with structural quote-size constraints.

If no custom bounds are supplied:

- default mode: `L_max` defaults to `1e5`, increased when quote size targets require it
- `quote_size_softmax_L=true`: `L` logits use `[-30, 30]`
- `k_max = 100`
- `x0_max = spread_max`

## Initialization And Warm Start

Initialization order inside `optimize_with_diagnostics()`:

1. If `warm_start=True` and `_last_action` exists, use the previous best action.
2. Else if `quote_size_target_per_side` is positive, use `_initial_action_uniform_marginal_qrange()`.
3. Else use a handcrafted default with increasing `L`, decreasing `k`, and spread centers across `[1, spread_max]`.

The selected one-step action is tiled over the whole horizon:

```python
x0_arr = np.tile(one_action, horizon)
```

Important implementation detail:

- Even with `warm_start=True`, the code writes a freshly tiled action into `_x_torch` at the start of each optimize call.
- If `warm_start=False`, Adam optimizer state is reset each call.
- If `warm_start=True`, the previous best action is remembered, and Adam state can be retained when compatible.
- With `quote_size_softmax_L=true`, initialization uses softmax-logit `L` parameters and may use spread-structured raw center parameters.

## Monte Carlo Samples

The optimizer draws future paths with shape:

```text
(n_samples, horizon)
```

Sampled variables:

| Variable | Source model |
| --- | --- |
| buy market volume | market-flow model |
| sell market volume | market-flow model |
| primary buy VWAP residual | price model |
| primary sell VWAP residual | price model |
| hedge buy VWAP delta | hedge price model |
| hedge sell VWAP delta | hedge price model |

If `fixed_mc_samples=True`, samples are drawn once before the Adam loop and reused across epochs. This gives a deterministic objective surface for that optimizer call.

If `fixed_mc_samples=False`, samples are redrawn inside each closure. That can behave like stochastic optimization, but initial/final/best objective diagnostics are noisier.

## Online Models Used By Optimizer

The optimizer depends on models already updated by `MpcStrategyEngine` just before optimization.

### Market-Flow Model

Supported config kinds in the strategy bridge:

- `gamma`
- `weibull`
- `hurdle_gamma`
- `hurdle_weibull`
- `hurdle_lognormal`

The optimizer calls model `sample()` when available. Volumes are modeled as future market buy and sell flow.

### Primary Price Model

The production price model is `TorchResidualPriceModel`.

It samples residual shocks around VWAPs rather than directly replacing the book process. In rollouts:

- at horizon step 0, upstream state VWAPs are used
- after step 0, VWAPs are recomputed from the simulated book arrays
- sampled residuals are added to buy/sell VWAPs

`MpcStrategyEngine` updates the model using EWMA on primary mid changes.

### Hedge Price Model

Optional hedge models:

- `legacy` / `ewma_gaussian` / `hedge_gaussian_rep`: `HedgeGaussianPriceModel`
- `gaussian`: `GaussianHedgeDeltaModel`
- disabled: no hedge price model

Hedge deltas are sampled only as future VWAP shocks. Live hedge execution is separate from this optimizer.

## Per-Epoch Optimization Loop

Core loop:

```python
for epoch in range(adam_epochs):
    loss = optimizer.step(closure)
    if loss < best_loss:
        best_x = x_torch.detach().clone()
        best_epoch = epoch
```

The optimizer minimizes:

```text
loss = -J
```

So lower loss means higher objective.

The closure:

1. Clears gradients.
2. Clamps raw parameters to raw bounds.
3. Optionally projects total quote size.
4. Draws or reuses Monte Carlo samples.
5. Rolls out every sample for every horizon step.
6. Computes terminal objective `J`.
7. Records initial/final/best objective metrics.
8. Backpropagates `loss = -J`.
9. Clips gradients by `adam_grad_clip`.
10. Counts nonfinite loss/gradient events.

## Per-Step Rollout

For each horizon step `h`:

1. Select `action_h` from the horizon action tensor.
2. Add quote-shape regularizers for this action:
   - marginal-depth concentration
   - quote-size target
   - quoted spread at reference depth
3. Read sampled market buy/sell flow and price residuals for step `h`.
4. Build current mid from simulated best bid/ask levels.
5. Build current primary buy/sell VWAPs.
6. Simulate market-flow interaction with own quotes via `soft_fill_batched()`.
7. Update primary inventory and cash from simulated fills.
8. If hedge rollout is enabled, simulate an immediate hedge trade for the primary fill delta.
9. Compute next mid and next VWAPs from the post-flow market book.
10. Compute primary and hedge step rewards.
11. Carry market book, primary state, and hedge state to the next horizon step.

The simulated book is market-only after each step. Unfilled own quote remainder is not carried as resting liquidity into the next simulated step.

## Soft Fill Model

`soft_fill_batched()` is the differentiable fill transition.

Inputs:

- action parameters for one step
- sell and buy VWAP tensors
- cumulative market bid and ask depth
- sell and buy market-flow volumes
- spread grid
- maker fee
- soft fill temperature
- soft fill activation, either `sigmoid` or `hardsigmoid`

Process:

1. Evaluate effective cumulative bid and ask quote depth.
2. Convert cumulative quote depth to marginal levels.
3. Convert cumulative market book to marginal levels.
4. Combine market levels and our quote levels.
5. Soft-consume combined depth using the configured reach activation.
6. Split consumed depth between market liquidity and our quotes by level-size share.
7. Remove only consumed market liquidity from the next market book.
8. Convert our filled levels into cash and inventory deltas.

The reach logic is:

```text
reach_bid_raw = sell_vol - combined_cumulative_bid_depth
reach_ask_raw = buy_vol - combined_cumulative_ask_depth
scaled_reach = normalized_reach / soft_fill_temp
fill_fraction = sigmoid(scaled_reach)      # default
fill_fraction = hardsigmoid(scaled_reach)  # optional
consumed_level = combined_level * fill_fraction
```

`hardsigmoid` can eliminate asymptotic ghost fills from far-from-mid quote levels because it reaches exact zero for sufficiently negative scaled reach.

Economic interpretation:

- market sells hit bid-side liquidity, so our bid fills increase inventory and reduce cash
- market buys hit ask-side liquidity, so our ask fills decrease inventory and increase cash

Implemented cash/inventory updates:

```text
cash_delta = ask_fill_cash * (1 - maker_fee) - bid_fill_cash
inv_delta = bid_fill_qty * (1 - maker_fee) - ask_fill_qty
```

Review caveat:

- This is a differentiable liquidity-share model, not a queue-position model. It assumes proportional sharing with other liquidity at the same spread bucket.

## Hedge Rollout Logic

If `hedge_rollout_enabled` is true, the optimizer simulates an immediate hedge after each primary fill delta.

Let:

```text
d_inv = primary inventory delta from simulated fills
```

Then:

- if `d_inv > 0`, primary inventory increased, so sell hedge
- if `d_inv < 0`, primary inventory decreased, so buy hedge

The hedge cash update uses hedge VWAP next:

```text
trade_cash_hedge = where(
  d_inv >= 0,
  d_inv * hedge_sell_vwap_next,
  d_inv * hedge_buy_vwap_next
)
hedge_cash_next = hedge_cash_t + trade_cash_hedge * (1 - hedge_fee)
hedge_inv_next = hedge_inv_t - d_inv
```

The hedge reward is:

```text
hedge_wealth = hedge_cash + hedge_inventory * hedge_sell_vwap
hedge_reward = hedge_wealth_next - hedge_wealth_t
```

Total step reward is:

```text
step_reward = primary_step_reward + hedge_reward_weight * hedge_reward
```

Important distinction:

- Hedge rollout is an objective assumption.
- Live hedging is performed by `HedgeExecutionService` only when `hedging.enabled` is true.
- The optimizer does not emit hedge orders.

## Reward And Terminal Objective

Primary wealth is marked to sell-side VWAP:

```text
W = cash + inventory * sell_vwap
step_reward_main = W_next - W
```

Across the horizon:

```text
total_reward = sum(step_reward_main + hedge_reward_weight * hedge_reward)
```

`terminal_objective()` computes:

```text
J =
  expected_pnl_weight * reward_term
  - inventory_target_penalty_weight * inventory_target_term
  - pnl_volatility_penalty_weight * volatility_term
  - pnl_cvar_penalty_weight * soft_cvar_term
  - quote_shape_concentration_penalty_weight * concentration_term
  - quote_size_target_penalty_weight * size_term
  - quote_spread_penalty_weight * spread_term
```

Reported weighted objective components:

| Component | Meaning |
| --- | --- |
| `expected_pnl` | weighted mean reward term |
| `inventory_target_penalty` | negative inventory target miss term |
| `pnl_volatility_penalty` | negative reward standard-deviation term |
| `pnl_cvar_penalty` | negative soft lower-tail risk term |
| `quote_shape_concentration_penalty` | negative marginal-depth concentration term |
| `quote_size_target_penalty` | negative quote-size target term |
| `quote_spread_penalty` | negative quoted-spread term |

These components are designed to sum to `objective.value`.

### Component Normalization

When `objective_normalize_components=true`, the objective uses dimensionless terms:

- reward, volatility, and soft CVaR are divided by `wealth_scale`
- inventory target miss is divided by `inventory_scale`
- size and spread penalties are computed as relative errors upstream
- concentration is normalized into `[0, 1]`

`wealth_scale = max(eps, abs(cash) + abs(inventory * sell_vwap))`.

`inventory_scale = max(eps, abs(target_inventory), quote_size_target_per_side, 1)`.

When normalization is disabled, reward and risk terms stay in quote units and inventory target penalty reverts to squared deviation.

### Soft CVaR

`objective_soft_cvar()` computes a differentiable lower-tail risk term:

```text
w_i = softmax(-temperature * reward_i)
soft_tail_mean = sum_i w_i * reward_i
soft_cvar = -soft_tail_mean
```

Low-reward paths receive more weight. The objective subtracts `pnl_cvar_penalty_weight * soft_cvar_term`.

## Quote Shape Regularizers

### Marginal-Depth Concentration

`_action_concentration_penalty_from_marginal_depth()`:

- computes effective cumulative depth on the spread grid
- converts cumulative depth to marginal per-level depth
- computes the Herfindahl index of per-level depth shares
- optionally normalizes the index into `[0, 1]`

Interpretation:

```text
0 = depth spread uniformly across levels
1 = depth concentrated in one level
```

The previous `quote_shape_uniformity_penalty_weight` name is treated as a compatibility alias. New configs should use `quote_shape_concentration_penalty_weight`.

### Quote-Size Target

`_action_size_penalty_qrange()` penalizes:

```text
|Q_bid(spread_max) - target| / target
+ |Q_ask(spread_max) - target| / target
```

when objective normalization is enabled. Without normalization it falls back to squared error. `Q_side` is effective cumulative depth.

### Spread At Target Depth

`_action_spread_at_depth()`:

- evaluates quoted bid and ask depth VWAPs from the controller curve
- fills `quote_depth_target_for_spread` base units on each side of the curve
- computes quoted width in basis points:

  ```text
  spread_bps = (ask_depth_vwap - bid_depth_vwap) / mid * 10000
  ```

- returns one-sided relative excess over `quote_spread_target_bps` when normalization is enabled

This penalizes the controller curve being wider than the configured quoted-spread target at the reference depth. It does not inspect the market BBO.

## Total Quantity Projection

The optimizer supports optional hard projection toward a quote-size target:

| Config | Meaning |
| --- | --- |
| `quote_size_target_per_side` | Target cumulative depth per side over `[0, spread_max]` |
| `hard_project_quote_size_target` | Rescale depth parameters toward `quote_size_target_per_side` |
| `quote_size_min_per_side`, `quote_size_max_per_side` | Lower/upper cumulative depth bounds per side |
| `hard_project_quote_size_range` | Clamp projected depth into the configured min/max range |

Projection exists in both numpy and torch helper forms. It is applied before closure rollout and when building returned best/final action slices.

## Structural Softmax L Parameterization

When `quote_size_softmax_L=true`, raw `L` values are not direct depth scales. They are logits decoded by `LParamConfig`.

Without spread-structured settings:

- component centers `x0` are mapped into `[0, spread_max]`
- `L` logits are softmax weights
- effective `L` values are solved so `D_eff(spread_max) = quote_size_target_per_side`

With `quote_spread_target` and `quote_spread_ref_depth`:

- first bid and ask centers are split by a softmax so their sum is `quote_spread_target`
- the first component carries `quote_spread_ref_depth` at its center
- remaining components fill the tail to the quote-size target at `spread_max`
- `quote_spread_k_epsilon` sets a floor on the first component sharpness

This mode embeds some quote-size and spread structure into the parameterization itself, instead of relying only on soft penalties or post-optimization projection.

## Returned Action

The returned action is the best first-step action:

```text
best_x[:single_len]
```

It is converted with:

```python
optimizer._flat_to_effective_action(...)
```

In default mode this delegates to `SigmoidMixtureAction.from_flat_positive(...).as_effective()`. In `quote_size_softmax_L` mode it materializes the structured parameters through `sigmoid_action_from_flat_raw(...)`. In both cases downstream `depth_bid()` and `depth_ask()` expose executable effective cumulative depth.

If `warm_start=True`, this best action is stored as `_last_action`.

## Conversion To Limit Orders

The optimizer returns a continuous curve. `MpcStrategyEngine` then calls:

```python
build_orders_from_sigmoid_with_report(...)
```

Conversion stages:

1. Evaluate effective cumulative depth at each spread level.
2. Convert cumulative depth to marginal order quantity:

   ```text
   qty_i = max(0, cumulative_i - cumulative_{i-1})
   ```

3. Build bid prices as `mid - spread`.
4. Build ask prices as `mid + spread`.
5. Quantize price/qty to venue instrument constraints.
6. Drop below-min-qty and below-min-notional orders.
7. Cap to `max_orders_per_side`, keeping largest quantities per side.

This means the optimizer controls the shape, while venue constraints and caps decide the final discrete order set.

## Optimizer Diagnostics

`OptimizerRunDiagnostics` is produced every optimizer call and embedded in controller diagnostics.

Top-level payload:

| Field | Meaning |
| --- | --- |
| `config` | Optimizer config snapshot |
| `timing` | sample draw, train, total seconds, best epoch |
| `objective.initial` | first closure objective snapshot |
| `objective.final` | last closure objective snapshot |
| `objective.best` | best objective snapshot |
| `objective.improvement` | best/final minus initial |
| `objective.contribution_fractions` | final-stage absolute contribution shares by component |
| `action.initial` | initial effective depth and raw params |
| `action.final` | final effective depth and raw params |
| `action.best` | best effective depth and raw params |
| `gradients` | nonfinite loss/gradient counts |
| `rollout` | sample count, horizon, hedge enabled, optimizer log |
| `best_epoch` | epoch index for best objective |

Objective snapshot fields:

| Field | Meaning |
| --- | --- |
| `value` | scalar objective `J` |
| `loss` | `-J` |
| `components` | weighted objective terms |
| `contribution_fractions` | `abs(component) / sum(abs(components))` |
| `expected_total_reward` | raw mean horizon reward before objective penalties |
| `expected_next_reward` | raw mean first-step reward |
| `main_reward_mean` | mean primary reward |
| `hedge_reward_mean` | mean hedge reward |

Action snapshot fields:

| Field | Meaning |
| --- | --- |
| `effective_depth.bid` | cumulative bid depth by spread/price |
| `effective_depth.ask` | cumulative ask depth by spread/price |
| `raw_params.flat` | raw flat first-step sigmoid parameters |

## Important Configuration Knobs

| Config | Effect |
| --- | --- |
| `n_components` | Expressiveness of each side's depth curve |
| `n_samples` | Monte Carlo sample count |
| `horizon` | Planning steps |
| `adam_epochs` | Gradient steps per decision |
| `adam_lr` | Adam learning rate |
| `adam_grad_clip` | Gradient clipping threshold |
| `fixed_mc_samples` | Deterministic vs stochastic objective per call |
| `warm_start` | Reuse previous best action as initialization |
| `spread_max` | Maximum spread domain for book/action projection |
| `spread_step` | Spread grid resolution |
| `soft_fill_temp` | Smoothness of fill transition |
| `soft_fill_activation` | `sigmoid` or `hardsigmoid` fill fraction |
| `expected_pnl_weight` | Weight on expected reward |
| `pnl_volatility_penalty_weight` | Penalty on reward volatility |
| `pnl_cvar_penalty_weight` | Penalty on soft lower-tail reward risk |
| `cvar_softmax_temperature` | Sharpness of soft CVaR tail weighting |
| `inventory_target_penalty_weight` | Penalty on expected terminal inventory target miss |
| `target_inventory` | Desired terminal inventory |
| `quote_size_target_per_side` | Target cumulative quote depth per side |
| `quote_size_target_penalty_weight` | Soft size target penalty |
| `hard_project_quote_size_target` | Hard rescaling toward quote-size target |
| `quote_size_min_per_side`, `quote_size_max_per_side` | Hard size range bounds when enabled |
| `hard_project_quote_size_range` | Hard clamp to size range |
| `quote_size_softmax_L` | Structural softmax size parameterization |
| `quote_shape_concentration_penalty_weight` | Marginal depth concentration penalty |
| `quote_spread_penalty_weight` | Quoted-spread target penalty |
| `quote_spread_target_bps` | Maximum target quoted width in basis points |
| `quote_depth_target_for_spread` | Cumulative depth used for quoted spread VWAP |
| `quote_spread_target` | Structural first-center sum in quote units when softmax L is enabled |
| `quote_spread_ref_depth` | Structural depth at first centers when softmax L is enabled |
| `quote_spread_k_epsilon` | Sharpness floor for structural first components |
| `objective_normalize_components` | Scale objective terms into dimensionless units |
| `hedge_rollout_enabled` | Include simulated hedge trades in objective |
| `hedge_reward_weight` | Weight of simulated hedge reward |
| `hedge_fee` / `execution_model.hedge_fee` | Fee used in hedge rollout |
| `device` | Torch device, usually `cpu` in runtime config |

## Numerical Safeguards

Implemented safeguards:

- softplus for positive `L` and `k`
- optional softmax materialization for `L`
- raw bound clamping every closure
- NaN/Inf sanitization in terminal objective
- objective normalization by wealth and inventory scales
- gradient clipping
- nonfinite loss and gradient counters
- minimum sigma in scoring/model code
- optional fixed Monte Carlo samples for stable diagnostics

Remaining numerical review points:

- Some optimizer mutation paths still assign through `x_torch.data`; projection paths mostly use `copy_()` under `torch.no_grad()`.
- Very large quote currencies can make unnormalized rewards large; `objective_normalize_components=true` reduces but does not eliminate calibration needs.
- `fixed_mc_samples=False` can make best/final objective comparisons noisy.
- Soft-fill sigmoid temperature can create either overly smooth fills or near-discontinuous gradients.

## Market-Making Assumptions

The optimizer assumes:

- market order flow consumes depth on a spread grid
- our quotes share fills proportionally with other liquidity in the same bucket
- unfilled own quotes are not carried into later simulated steps
- primary inventory is marked to sell-side VWAP
- hedge rollout, if enabled, is immediate and exact after simulated primary fills
- venue constraints are handled after optimization, not inside the objective except through quote-size projection and configured penalties

These are modeling choices, not exchange facts. They should be validated against realized fill and PnL data.

## Review Questions For Market-Making Experts

- Is sell-side VWAP the right mark for all inventory states?
- Should inventory marking switch by long/short inventory?
- Should the objective include adverse selection from microprice, imbalance, or cross-venue basis?
- Does proportional fill sharing approximate the venue queue well enough?
- Should old resting quotes be modeled as persistent inventory/liquidity across horizon steps?
- Should hedge rollout be disabled unless live fill-driven hedging is healthy?
- Are quote-size and spread penalties dimensionally calibrated for `USDT_IRT`?
- Should the action include asymmetric inventory skew beyond the terminal inventory penalty?
- Should spread-grid levels be tied to venue tick size rather than arbitrary quote-unit steps?
- Should market-flow volume be modeled as continuous rather than rounded integer volume?

## Review Questions For Software Engineers

- Do diagnostics always expose enough information to reproduce a decision?
- Should optimizer model updates and optimization be split into separate services/classes?
- Should `MpcStrategyEngine` call `optimize_with_diagnostics()` directly instead of reading `last_diagnostics`?
- Should `.data` mutations be replaced fully with `torch.no_grad()` and `copy_()`?
- Should optimizer config be validated with a stricter schema?
- Should objective component sums be asserted in tests for all active penalty combinations?
- Should `fixed_mc_samples=True` also use a deterministic seed per frame for replay reproducibility?
- Should optimizer timing and nonfinite counters become direct Prometheus metrics?

## Debugging Guide

When a quote looks wrong, inspect `ControllerDiagnostics.optimizer`:

1. Check `objective.best.value` and `objective.improvement.best_minus_initial`.
2. Check which component dominates `objective.best.components`.
3. Compare `action.initial`, `action.final`, and `action.best`.
4. Verify effective depth reaches intended `quote_size_target_per_side`.
5. Check `rollout.hedge_enabled` matches expectations.
6. Check `gradients.nonfinite_loss_count` and `gradients.nonfinite_grad_count`.
7. Compare continuous action depth to action pipeline ladder totals.
8. Check whether venue quantization or capping removed most of the curve.
9. Check model scores under `ControllerDiagnostics.models`.
10. Check risk and execution gate rejections after optimizer output.

Useful files:

| Concern | File |
| --- | --- |
| Optimizer implementation | `mm/mpc/mpc_tensor_batched.py` |
| Optimizer diagnostics | `mm/mpc/optimizer_diagnostics.py` |
| State/action types | `mm/mpc/mpc_state.py` |
| Objective | `mm/mpc/reward_tensor.py` |
| Strategy integration | `mm/strategies/mpc_engine.py` |
| Model scoring | `mm/models/scoring.py` |
| Order conversion | `mm/core/order_constraints.py` |
| Controller diagnostics | `mm/controller/context.py` |
