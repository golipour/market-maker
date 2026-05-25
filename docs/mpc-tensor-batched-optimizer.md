# Batched Tensor MPC Optimizer

This document explains `mm/mpc/mpc_tensor_batched.py`: how the batched Monte Carlo MPC optimizer converts a market state into a continuous quote-depth action, how the objective is built, how Adam iterates, and what diagnostics should be exposed.

## Purpose

`MonteCarloMPCTensorBatched` is the production MPC optimizer used by `MpcStrategyEngine`. It chooses a multi-level bid/ask quote-depth curve by simulating many possible future market paths in a single PyTorch batch, then optimizing the quote parameters by gradient ascent on expected terminal objective.

The public API is:

```python
action = optimizer.optimize(state)
```

where `state` is an `mm.mpc.mpc_state.State`, and `action` is a `SigmoidMixtureAction`. The strategy layer later discretizes that continuous depth curve into actual ladder orders.

## Input Data

The optimizer receives a `State` with:

| Field | Meaning |
| --- | --- |
| `mid` | current primary mid price |
| `ob_rel` | market-only relative book: bid/ask spread from mid plus base quantity |
| `inventory`, `cash` | primary venue position state |
| `buy_vwap`, `sell_vwap` | current primary reference VWAPs |
| `hedge_buy_vwap`, `hedge_sell_vwap` | optional hedge venue VWAP references |
| `hedge_inventory`, `hedge_cash` | optional hedge venue position state |

The state is already built by `MpcStrategyEngine._state_from_frame()` after own-order subtraction. That means `ob_rel` should represent market liquidity excluding the bot's own open orders.

## Action Parameterization

The optimizer does not directly choose individual order prices and quantities. It chooses cumulative bid and ask depth curves over spread from mid.

Each side is a mixture of sigmoids:

```text
D_side(spread) = sum_i L_i * sigmoid(k_i * (spread - x0_i))
```

For each component:

- `L`: depth scale
- `k`: slope/sharpness
- `x0`: center spread

The flat action layout for one horizon step is:

```text
[bid_L1, bid_k1, bid_x0_1, ..., bid_Ln, bid_kn, bid_x0_n,
 ask_L1, ask_k1, ask_x0_1, ..., ask_Ln, ask_kn, ask_x0_n]
```

For horizon `H`, the optimizer variable is:

```text
x_torch length = H * 6 * n_components
```

Only the first horizon action is returned. The remaining horizon actions are planning variables.
The returned object is an effective action view: its `depth_bid()` and `depth_ask()`
methods expose executable cumulative depth `D(spread) - D(0)`, while the underlying
flat parameters remain the raw sigmoid-mixture parameters for warm starts and debugging.

`L` and `k` are optimized in raw unconstrained space and converted to positive values with `softplus`. Bounds are given in effective positive space and converted to raw space by `_bounds_effective_to_raw()`.

Important convention: the real quoted cumulative depth is not raw `D(spread)`. It is:

```text
effective_depth(spread) = D(spread) - D(0)
```

This matters because a sigmoid mixture can have non-zero raw depth at spread zero. Any fill simulation, marginal ladder conversion, size target, or spread-at-depth penalty must use the baseline-shifted effective depth.

## Initialization

`optimize()` initializes one action and tiles it across the horizon.

Initialization order:

1. If `warm_start=True` and `_last_action` exists, use the previous best action.
2. Else, if `quote_size_target_per_side` is positive, call `_initial_action_uniform_marginal_qrange()`.
3. Else, use a simple handcrafted default:
   - increasing `L` per component
   - decreasing `k`
   - `x0` spread centers distributed across `[1, spread_max]`

The uniform initialization tries to create a nearly flat marginal quote-size distribution over the spread grid. It chooses small `k`, centers components near the middle of the spread range, estimates `L` from the desired per-level quantity, and rescales so:

```text
D(spread_max) - D(0) ~= quote_size_target_per_side
```

Then the one-step action is repeated for every horizon step:

```python
x0_arr = np.tile(one_action, horizon)
```

If `warm_start=False`, Adam optimizer state is reset each `optimize()` call. If `warm_start=True`, the previous action is remembered, but this implementation still writes a freshly tiled `x0_arr` into `_x_torch` each call.

## Orderbook Projection

The primary orderbook is projected onto a fixed spread grid:

```text
spread_grid = [spread_step, 2*spread_step, ..., grid_levels*spread_step]
grid_levels = ceil(spread_max / spread_step)
```

Two book forms are built:

- cumulative tensors from `OrderbookRelative` for initial soft filling
- marginal per-level tensors repeated over `n_samples` for future simulated book state

The book arrays are market-only. The bot's quotes are added during simulation and removed again after the simulated market order flow consumes liquidity.

## Monte Carlo Samples

The optimizer draws future random paths for:

- market buy volume
- market sell volume
- primary price residual buy shock
- primary price residual sell shock
- hedge buy VWAP delta
- hedge sell VWAP delta

Shapes are:

```text
(n_samples, horizon)
```

If `fixed_mc_samples=True`, samples are drawn once before the Adam loop and reused every epoch. This creates a deterministic objective for the optimization call.

If `fixed_mc_samples=False`, samples are redrawn inside every closure. This trains against a noisy objective and can be useful as stochastic optimization, but it makes initial/final objective diagnostics less directly comparable unless the diagnostic also records the sampling mode.

## Optimization Loop

The main loop is:

```python
for epoch in range(adam_epochs):
    loss = optimizer.step(closure)
    if loss < best_loss:
        best_x = x_torch.detach().clone()
```

The closure:

1. Clears gradients.
2. Clamps raw parameters to raw bounds.
3. Applies optional total-quantity projection.
4. Draws or reuses Monte Carlo samples.
5. Rolls out every sample for every horizon step.
6. Builds terminal objective `J`.
7. Stores objective components.
8. Backpropagates `loss = -J`.
9. Clips gradients.

The optimizer minimizes `loss`, which is the negative of the desired objective. Lower loss means higher objective.

## Per-Step Rollout

For each horizon step `h`, the optimizer:

1. Selects `action_h` from `x_torch`.
2. Adds action regularizers for this horizon step:
   - marginal depth uniformity
   - size target penalty
   - spread-at-depth penalty
3. Reads sampled buy/sell market volumes and price residuals for step `h`.
4. Builds current mid from simulated best bid/ask.
5. Builds current primary buy/sell VWAP:
   - at `h=0`, uses upstream `state.buy_vwap` and `state.sell_vwap`
   - after `h=0`, recomputes VWAP from simulated book arrays
   - then adds sampled residual shocks
6. Calls `soft_fill_batched()` to simulate market-flow interaction with the combined market book plus our quotes.
7. Updates inventory and cash from our simulated fills.
8. Optionally simulates hedge trades if hedge dynamics are enabled.
9. Computes next mid and next VWAPs.
10. Computes primary and hedge step rewards.
11. Carries book, cash, inventory, and hedge state into the next horizon step.

## Soft Fill Model

`soft_fill_batched()` is the differentiable execution transition.

It:

1. Converts raw cumulative quote depth to effective depth `D(s) - D(0)`.
2. Converts effective cumulative quote depth to marginal level sizes.
3. Converts cumulative market-book depth to marginal market levels.
4. Combines market-book levels and our quote levels.
5. Soft-consumes combined depth using a sigmoid reach function:

```text
reach = market_order_volume - combined_cumulative_depth
consumed_level = combined_level * sigmoid(normalized_reach / soft_fill_temp)
```

6. Splits consumed volume between the market book and our quotes in proportion to their level sizes.
7. Returns:
   - next market-only bid levels
   - next market-only ask levels
   - cash delta from our fills
   - inventory delta from our fills

Own fill cash is priced from the clean orderbook mid and the spread grid:

```text
bid_fill_price(s) = clean_mid - s
ask_fill_price(s) = clean_mid + s
```

This avoids hard fill/no-fill discontinuities and keeps gradients flowing through the quote-depth parameters.

## Reward And Objective

At each step, primary wealth is marked using sell-side VWAP:

```text
W = cash + inventory * sell_vwap
step_reward_main = W_next - W
```

If hedge is enabled, hedge wealth is:

```text
hedge_W = hedge_cash + hedge_inventory * hedge_sell_vwap
hedge_reward = hedge_W_next - hedge_W
```

Total trajectory reward is:

```text
total_reward = sum_h(step_reward_main + hedge_reward_weight * hedge_reward)
```

The terminal objective is built by `terminal_objective()`:

```text
J =
  expected_pnl_weight * E[total_reward]
  - inventory_target_penalty_weight * (E[terminal_inventory] - target_inventory)^2
  - pnl_volatility_penalty_weight * Std(total_reward)
  - quote_shape_uniformity_penalty_weight * uniformity_penalty
  - quote_size_target_penalty_weight * size_penalty
  - quote_spread_penalty_weight * spread_penalty
```

The reported objective components are weighted terms:

| Component | Meaning |
| --- | --- |
| `expected_pnl` | weighted expected total reward |
| `inventory_target_penalty` | weighted terminal inventory target penalty |
| `pnl_volatility_penalty` | weighted reward volatility penalty |
| `quote_shape_uniformity_penalty` | weighted marginal-depth uniformity penalty |
| `quote_size_target_penalty` | weighted quote-size target penalty |
| `quote_spread_penalty` | weighted spread-at-depth penalty |

The scalar optimized value is:

```text
J = sum(components.values())
loss = -J
```

## Action Regularizers

### Uniformity

`_action_uniformity_l2_from_marginal_depth()` compares marginal quoted depth to a uniform target:

```text
q_uniform = quote_size_target_per_side / spread_max
```

It penalizes how far bid/ask marginal depth is from that uniform distribution.

### Size

`_action_size_penalty_qrange()` penalizes the difference between each side's total quoted depth and `quote_size_target_per_side`:

```text
(Q_bid - target)^2 + (Q_ask - target)^2
```

where:

```text
Q_side = D_side(spread_max) - D_side(0)
```

### Spread At Depth

`_action_spread_at_depth()` estimates the spread where effective cumulative depth `D(s) - D(0)` reaches `quote_depth_target_for_spread`, using a differentiable soft-argmin over a dense grid. The penalty is:

```text
spread_bid(target_depth) + spread_ask(target_depth)
```

This encourages desired depth to sit closer to mid when `quote_spread_penalty_weight` is positive.

## Output

After all epochs:

1. The best raw parameter vector is selected by lowest loss.
2. Only the first horizon slice is kept.
3. Optional flat projection is applied again.
4. The raw first-slice parameters are converted to positive effective parameters with `SigmoidMixtureAction.from_flat_positive()`.
5. The action is returned.

`MpcStrategyEngine` then converts this continuous depth curve into ladder orders on the configured spread grid.

## Current Diagnostics

The optimizer currently stores several private diagnostic fields:

| Field | Meaning |
| --- | --- |
| `_last_J_components` | most recent closure's weighted objective components |
| `_last_J_components_initial` | first closure's weighted objective components |
| `_last_E_dW` | expected total reward for best action when tracked |
| `_last_E_dW_next` | expected first-step reward for best action when tracked |
| `_last_main_reward_mean` | main venue reward mean |
| `_last_hedge_reward_mean` | hedge reward mean |
| `_last_main_reward_mean_initial` | main reward mean at first closure |
| `_last_hedge_reward_mean_initial` | hedge reward mean at first closure |

These are useful, but they are incomplete as a stable diagnostics API. In particular, `_last_J_components` can represent the last closure, while `best_x` can come from an earlier epoch. For diagnostics, record initial, final-last, and best snapshots explicitly.

For hedge-specific objective behavior and the gap between simulated hedging and live hedge orders, see [Hedging In The MPC Objective](hedging-in-mpc.md).

## Recommended Diagnostics Event Fields

Add an explicit optimizer diagnostics snapshot to the proposed `ControllerDiagnostics` event:

```text
optimizer:
  kind
  n_samples
  horizon
  n_components
  spread_max
  spread_step
  fixed_mc_samples
  warm_start
  adam_lr
  adam_epochs
  adam_grad_clip
  device
  dtype
  timing:
    sample_draw_sec
    train_sec
    total_sec
  objective:
    initial:
      value
      loss
      components
      expected_total_reward
      expected_next_reward
      main_reward_mean
      hedge_reward_mean
    last:
      value
      loss
      components
    best:
      epoch
      value
      loss
      components
      expected_total_reward
      expected_next_reward
      main_reward_mean
      hedge_reward_mean
  gradients:
    grad_norm_before_clip
    grad_norm_after_clip
    nonfinite_loss_count
    nonfinite_grad_count
  action:
    initial_first_slice
    best_first_slice
    projected_total_q_bid
    projected_total_q_ask
```

The objective initial and optimal values are especially important. They answer:

- Did optimization improve the controller objective?
- Which objective component improved or got worse?
- Did the optimizer mainly improve reward, inventory, risk, quote size, or spread?
- Did the result come from the best epoch or just the final epoch?
- Is a bad quote caused by model inputs or by optimizer failure?

Recommended scalar metrics:

- `mm_optimizer_objective_value{stage="initial|best|last"}`
- `mm_optimizer_loss{stage="initial|best|last"}`
- `mm_optimizer_objective_component{stage,component}`
- `mm_optimizer_objective_improvement = best.value - initial.value`
- `mm_optimizer_best_epoch`
- `mm_optimizer_nonfinite_loss_total`
- `mm_optimizer_grad_norm`

## Technical Improvement Suggestions

### 1. Return A Public Diagnostics Object

Replace private-field scraping with:

```python
@dataclass(frozen=True)
class OptimizerDiagnostics:
    initial: ObjectiveSnapshot
    last: ObjectiveSnapshot
    best: ObjectiveSnapshot
    timing: dict[str, float]
    config: dict[str, object]
```

Then expose either:

```python
action, diagnostics = optimizer.optimize_with_diagnostics(state)
```

or:

```python
optimizer.last_diagnostics
```

This avoids reaching into `_last_*` fields from `MpcStrategyEngine`.

### 2. Track Best Components Correctly

Currently, best scalar reward fields are tracked, but component fields are not clearly stored for the best epoch. Add:

```python
best_components
best_epoch
best_objective_value
initial_objective_value
last_objective_value
```

Use the same fixed Monte Carlo samples for initial and best evaluation when `fixed_mc_samples=False`, or label comparisons as stochastic.

### 3. Vectorize Sample Drawing

`draw_samples()` loops over `n_samples * horizon` in Python and calls model sampling repeatedly. This is likely one of the easiest performance wins.

Prefer model APIs that return tensors shaped `(n_samples, horizon)` directly:

```python
buy_vol, sell_vol = local_market_flow_model.sample((n_samples, horizon), ...)
eps_buy, eps_sell = local_price_model.sample_residual((n_samples, horizon), ...)
```

The hedge model already supports batched `sample_deltas()`.

### 4. Avoid `.data` Mutation

The code uses `x_torch.data = ...` and in-place `.data[...]` updates during projection. This can silently bypass autograd safety checks.

Prefer:

```python
with torch.no_grad():
    x_torch.copy_(torch.clamp(x_torch, lower, upper))
```

For projection, use `copy_()` or a pure tensor projection function inside `torch.no_grad()`.

### 5. Separate Rollout From Optimizer Control Flow

The closure is large and hard to inspect. Split it into pure-ish helpers:

- `prepare_optimizer_inputs(state)`
- `draw_mc_samples()`
- `rollout(params, samples, prepared_state)`
- `objective_from_rollout(rollout)`
- `snapshot_objective(...)`

This would make it easier to test objective behavior without running Adam.

### 6. Make Printing Optional

`optimize()` currently prints batch/train/total timing unconditionally. Route this through structured diagnostics or logger instead. Unconditional stdout makes production logs noisy and forces `MpcStrategyEngine` to redirect stdout.

### 7. Cache Static Tensors

Some tensors are recreated every `optimize()` call:

- spread grid
- bound tensors
- dense spread grid for spread-at-depth
- zero tensors with fixed shape

Cache them by `(device, dtype, grid_levels, horizon, n_samples, n_components)` when configs do not change.

### 8. Review Projection Math

`_project_quote_size_torch()` rescales raw `L` parameters, not effective `softplus(L)` values. Because `softplus(a*x)` is not equal to `a*softplus(x)`, the projection is approximate and nonlinear.

If strict total quantity is important, project in effective space and transform back through inverse softplus, or make size control purely a differentiable penalty instead of a hard projection.

### 9. Clarify Naming

Several names mix objective value, loss, and reward:

- `_last_E_dW` is expected total reward, not full objective.
- `best_loss` is `-J`.
- `_last_J_components` are weighted objective components.
- `J_mean_raw` is mean total reward before objective penalties.

Use names like:

- `objective_value`
- `loss_value`
- `expected_total_reward`
- `expected_first_step_reward`
- `weighted_components`

### 10. Add Unit Tests For Objective Accounting

Suggested tests:

- `sum(components) == objective_value`
- `loss == -objective_value`
- initial/best/last snapshots are all present
- with zero epochs or one epoch, best and initial semantics are clear
- with `fixed_mc_samples=True`, objective improvement is reproducible
- projection keeps total quoted quantity within tolerance
- `soft_fill_batched()` conserves level volume split as expected

## Readability Improvement Roadmap

1. Add optimizer diagnostics dataclasses and capture initial/last/best objective snapshots.
2. Extract rollout and objective helpers without changing behavior.
3. Vectorize sample drawing.
4. Replace `.data` mutation.
5. Add unit tests around snapshots and objective components.
6. Replace stdout printing with diagnostics/logging.

These changes would make the optimizer much easier to debug while preserving the current market-making behavior.
