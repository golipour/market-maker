# MPC Optimizer Implementation Details

This document describes the implemented MPC controller after the
liquidity-sourcing migration. The main classes are:

- `mm.strategies.mpc_engine.MpcStrategyEngine`
- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched`
- `mm.mpc.joint_optimizer.optimize_joint_with_diagnostics`
- `mm.mpc.joint_mpc`

The optimizer still chooses continuous sigmoid quote-depth curves over spread
from mid. In single-venue mode it chooses local bid/ask curves. In liquidity
sourcing mode it chooses four curves per horizon step: local bid/ask and global
bid/ask. The strategy layer discretizes the first-step curves into
`JointTargetBook`; `Action` is derived from that book for execution reconcile.

## Integration Point

Production call path:

```text
StrategyRunner.on_joint_frame
  -> StrategyRunner._process_input_frame
  -> clean_strategy_frame
  -> MpcStrategyEngine.decide
  -> _state_from_frame
  -> score online models before update
  -> update local/global models
  -> MonteCarloMPCTensorBatched.optimize(state)
  -> single-venue optimizer or joint optimizer
  -> build_joint_target_book_from_sigmoid
  -> StrategyDecision + JointTargetBook + Action
```

Public API:

```python
action = optimizer.optimize(state)
```

Diagnostics API:

```python
action, diagnostics = optimizer.optimize_with_diagnostics(state)
optimizer.last_diagnostics
```

`optimize()` is a wrapper around `optimize_with_diagnostics()`. When
`liquidity_sourcing_enabled` is true, `MonteCarloMPCTensorBatched` dispatches to
`optimize_joint_with_diagnostics()`.

## Input State

The optimizer receives `mm.mpc.mpc_state.State`.

| Field | Meaning |
| --- | --- |
| `mid` | Current local mid price, quote per base |
| `ob_rel` | Cleaned local market-only book as `(spread, qty)` tuples |
| `buy_vol`, `sell_vol` | Realized local market-flow volumes |
| `inventory`, `cash` | Local base inventory and quote cash |
| `buy_vwap`, `sell_vwap` | Local reference VWAPs |
| `global_ob_rel` | Optional cleaned global market-only book |
| `global_buy_vol`, `global_sell_vol` | Optional global market-flow volumes |
| `global_inventory`, `global_cash` | Optional global base inventory and quote cash |
| `global_buy_vwap`, `global_sell_vwap` | Optional global reference VWAPs |
| `reference_price` | Price used to mark consolidated wealth |
| `global_active` | Whether the global curves participate in rollout |

`MpcStrategyEngine._state_from_frame()` builds this state after own-order
subtraction. If private state is missing, balances default to zero; hard risk can
then reject missing or stale private state before live execution.

The global venue is active only when:

- `liquidity_sourcing.enabled` is true
- `liquidity_sourcing.mode == "liquidity_sourcing"`
- a global market event is present
- global status is `ready` or `stale`

Risk and execution apply stricter freshness requirements before live global
orders are allowed.

## Orderbook Projection

Both local and global books use the same spread grid:

```text
grid_levels = ceil(spread_max / spread_step)
spread_grid = [spread_step, 2 * spread_step, ..., grid_levels * spread_step]
```

Bid spread:

```text
spread = mid - bid_price
```

Ask spread:

```text
spread = ask_price - mid
```

Only non-negative spreads inside `spread_max` are kept. Bucket index:

```text
bucket index = ceil(spread / spread_step) - 1
```

The first rollout step uses cumulative tensors projected from `OrderbookRelative`.
Later steps use marginal per-bucket arrays produced by the fill transition.

## Action Parameterization

One venue action contains bid and ask sigmoid mixtures:

```text
D_side(spread) = sum_i L_i * sigmoid(k_i * (spread - x0_i))
```

One venue step layout for `n_components = n`:

```text
[bid_L1, bid_k1, bid_x0_1, ..., bid_Ln, bid_k_n, bid_x0_n,
 ask_L1, ask_k1, ask_x0_1, ..., ask_Ln, ask_k_n, ask_x0_n]
```

Single-venue horizon layout:

```text
len(x_torch) = horizon * 6 * n
```

Joint-venue step layout concatenates local and global venue actions:

```text
joint_step = [local_action, global_action]
len(joint_step) = 2 * 6 * n
len(x_torch) = horizon * 2 * 6 * n
```

The optimizer returns the local first-step `SigmoidMixtureAction`. In joint mode
it also stores:

- `_last_joint_action`
- `_last_global_sigmoid_action`
- `_last_raw_flat`, containing the best joint first-step raw vector when warm
  start is enabled

## Effective Depth

Raw sigmoid mixtures can have non-zero depth at zero spread. The executable
convention is:

```text
effective_depth(spread) = max(0, raw_depth(spread) - raw_depth(0))
```

Effective depth is used by:

- fill simulation
- quote-size penalties
- quoted-spread penalties
- structural softmax decode
- diagnostics
- target-book order conversion

Reviewers should treat effective depth as the true cumulative quote size.

## Positive Parameters And Structural Size

In default parameterization, `L` and `k` are raw unconstrained values mapped to
positive values with `softplus`. Bounds are specified in positive parameter
space and converted to raw optimizer bounds.

When `quote_size_softmax_L=true`, raw `L` slots are logits rather than literal
depth scales. `mm.mpc.l_param` decodes raw parameters so executable depth at
`spread_max` equals `quote_size_target_per_side`. In this mode hard quote-size
projection is skipped because size is enforced by reparameterization.

Spread-structured softmax mode remains supported when both are positive:

- `quote_spread_target`
- `quote_spread_ref_depth`

That mode reserves the first component on each side as a spread leg and maps the
first bid/ask centers so their sum is capped by `quote_spread_target`.

## Initialization And Warm Start

Single-venue initialization order:

1. Use previous best action if `warm_start=True` and available.
2. Else use `_initial_action_uniform_marginal_qrange()` if
   `quote_size_target_per_side` is positive.
3. Else use the handcrafted default with increasing `L`, decreasing `k`, and
   centers across `[1, spread_max]`.

Joint initialization uses `initial_joint_action_flat()`, which builds one venue
initializer and concatenates it for local and global actions. The one-step raw
vector is tiled across the horizon.

Important details:

- Each optimize call writes a freshly tiled action into `_x_torch`.
- If `warm_start=False`, Adam optimizer state is reset.
- If `warm_start=True`, Adam state can be retained.
- Joint warm start stores and reloads the joint first-step raw vector, not only
  the local action.

## Monte Carlo Samples

Single-venue optimizer samples:

| Variable | Source model |
| --- | --- |
| local buy/sell market volume | `local_market_flow_model` |
| local VWAP residuals | `local_price_model` |
Joint optimizer samples:

| Variable | Source model |
| --- | --- |
| local buy/sell market volume | `local_market_flow_model` |
| global buy/sell market volume | `global_market_flow_model` |
| local VWAP residuals | `local_price_model` |
| global VWAP residuals | `global_price_model` |

If `fixed_mc_samples=True`, samples are drawn once before the Adam loop and
reused for the whole optimizer call. If false, samples are redrawn inside each
closure.

## Online Models

`MpcStrategyEngine` owns and updates:

| Model | Config | Update |
| --- | --- | --- |
| local market flow | `controller.local_market_flow_model` | current local buy/sell volumes |
| local residual price | `controller.local_price_model` | EWMA on local market event price step |
| reference/global price | `controller.global_price_model` | global VWAP deltas or residual update |
| global market flow | `controller.global_market_flow_model` | current global buy/sell volumes |

Model scoring is prequential: `score_frame_models()` evaluates the observation
using parameters from before the frame update, and
`refresh_score_params_after()` records the post-update parameters.

Supported market-flow kinds:

- `gamma`
- `weibull`
- `hurdle_gamma`
- `hurdle_weibull`
- `hurdle_lognormal`

Supported global price kinds:

- `ewma_gaussian`
- `gaussian`
- `residual` for liquidity-sourcing global residual modeling
- `none` / `disabled`

## Single-Venue Rollout

When `liquidity_sourcing_enabled=false`, `MonteCarloMPCTensorBatched` uses the
local-only rollout:

1. Select one local action per horizon step.
2. Apply action shape regularizers.
3. Sample local market flow and price residuals.
4. Build current VWAPs from the simulated local book.
5. Run `soft_fill_batched()` against the local book.
6. Update local inventory and cash from simulated fills.
7. Compute terminal objective from local wealth and penalties.

## Joint Liquidity-Sourcing Rollout

When `liquidity_sourcing_enabled=true`, `optimize_joint_with_diagnostics()` runs
a four-curve rollout.

For each horizon step:

1. Split the joint step tensor into `local_h` and `global_h`.
2. Draw local and global market-flow volumes.
3. Draw local and global VWAP residuals.
4. Compute local and global VWAPs from simulated books.
5. Run `soft_fill_batched()` for the local book with `depth_scale=1.0`.
6. Run `soft_fill_batched()` for the global book with
   `depth_scale=1.0` when global is active, otherwise `0.0`.
7. Update local and global cash/inventory independently.
8. Mark consolidated wealth to the global sell VWAP path.
9. Accumulate normalized consolidated returns.
10. Carry both market-only books into the next step.

Unfilled own quote remainder is not carried as resting own liquidity into later
rollout steps. The simulated books after each step are market-only books.

## Soft Fill Model

`soft_fill_batched()` is the differentiable fill transition used by both
single-venue and joint rollouts.

Inputs include:

- one venue action parameters
- sell and buy VWAP tensors
- cumulative market bid/ask depth
- sell and buy market-flow volumes
- spread grid
- maker fee
- soft fill temperature and activation
- `depth_scale`

Process:

1. Evaluate effective cumulative bid and ask quote depth.
2. Convert quote and market cumulative depth to marginal levels.
3. Combine market liquidity and own quotes.
4. Soft-consume combined depth using `sigmoid` or `hardsigmoid` reach.
5. Allocate consumed volume between market liquidity and own quote liquidity by
   level-size share.
6. Remove consumed market liquidity from the next market book.
7. Convert own filled levels into cash and inventory deltas.

Economic convention:

- market sells hit bid-side liquidity, so our bid fills increase inventory and
  reduce cash
- market buys hit ask-side liquidity, so our ask fills decrease inventory and
  increase cash

Implemented cash/inventory updates:

```text
cash_delta = ask_fill_cash * (1 - maker_fee) - bid_fill_cash
inv_delta = bid_fill_qty * (1 - maker_fee) - ask_fill_qty
```

This is a differentiable liquidity-share model, not a venue queue-position
model.

## Joint Objective

Joint MPC uses consolidated wealth:

```text
W = local_cash + global_cash
    + (local_inventory + global_inventory) * reference_price
```

Initial wealth:

```text
W_0 = max(1e-8, W(state_0))
```

Rollout return:

```text
local_returns = total_local_delta_wealth / W_0
global_returns = total_global_delta_wealth / W_0
```

Terminal objective:

```text
J =
  local.objective_weight
    * entropic_return_beta(local_returns, local.return_risk_aversion_beta)
  + global_active
    * global.objective_weight
    * entropic_return_beta(global_returns, global.return_risk_aversion_beta)
  - local.inventory_target_penalty_weight * local_inventory_capacity_penalty
  - global_active
    * global.inventory_target_penalty_weight * global_inventory_capacity_penalty
```

Implemented joint components:

| Component | Meaning |
| --- | --- |
| `risk_adjusted_return` | Sum of local and active global weighted return terms |
| `local_risk_adjusted_return` | Local weighted entropic return term |
| `global_risk_adjusted_return` | Global weighted entropic return term |
| `inventory_capacity_penalty` | Total inventory penalty |
| `local_inventory_penalty` | Local inventory penalty contribution |
| `global_inventory_penalty` | Global contribution, masked when inactive |

Joint diagnostics also record:

- `reference_price`
- `wealth_0`
- `global_active`
- resolved `local_optimizer` / `global_optimizer` configs
- first-step local/global fill means
- `global_action_flat`

The joint objective does not use the older local-only expected-PnL, volatility,
CVaR, quote-size, and quoted-spread component set. Those still apply in the
single-venue branch.

## Legacy Single-Venue Objective

The local-only objective remains:

```text
J =
  expected_pnl_weight * reward_term
  - inventory_target_penalty_weight * inventory_target_term
  - pnl_volatility_penalty_weight * volatility_term
  - pnl_cvar_penalty_weight * soft_cvar_term
  - quote_shape_concentration_penalty_weight * concentration_term
  - quote_size_target_penalty_weight * size_term
  - quote_spread_penalty_weight * quoted_spread_term
```

When `objective_normalize_components=true`, reward, volatility, and CVaR are
scaled by wealth, inventory miss by an inventory scale, quote-size and spread
terms are relative, and concentration is normalized into `[0, 1]`.

Soft CVaR weights lower-reward paths with a softmax over negative rewards. The
quote-shape term is a Herfindahl-style marginal-depth concentration index. The
quoted-spread term computes the controller curve width at
`quote_depth_target_for_spread`; it does not inspect market BBO.

## Per-Epoch Optimization Loop

Both branches minimize:

```text
loss = -J
```

Core loop:

```python
for epoch in range(adam_epochs):
    loss = optimizer.step(closure)
    if loss < best_loss:
        best_x = x_torch.detach().clone()
        best_epoch = epoch
```

The closure:

1. Clears gradients.
2. Clamps raw parameters to raw bounds.
3. Optionally projects total quote size in default single-venue mode.
4. Draws or reuses Monte Carlo samples.
5. Rolls out every sample for every horizon step.
6. Computes objective `J`.
7. Records initial/final/best metrics.
8. Backpropagates `loss = -J`.
9. Clips gradients by `adam_grad_clip`.
10. Counts nonfinite loss/gradient events.

## Target-Book Conversion

The optimizer output is not sent directly to venues.

`build_joint_target_book_from_sigmoid()` performs:

1. Evaluate local effective cumulative depth on the spread ladder.
2. Convert cumulative depth to marginal limit orders.
3. Apply local instrument tick/lot/min-notional constraints.
4. Cap local orders per side.
5. In liquidity-sourcing mode, repeat the same process for the stored global
   sigmoid action and global instrument constraints.
6. Return a `JointTargetBook`.
7. Convert the joint book to `Action`.

The local `ActionPipelineReport` includes continuous depth, ladder totals,
venue-quantized totals, capped totals, target-book totals, risk totals, and
execution totals. In joint mode, `target_book` summarizes local plus global
orders.

## Optimizer Diagnostics

`OptimizerRunDiagnostics` includes:

| Field | Meaning |
| --- | --- |
| `config` | optimizer kind, grid, horizon, samples, joint/global flags |
| `timing` | sample draw, train, total seconds, best epoch |
| `objective.initial/final/best` | objective value, loss, weighted components, reward means |
| `objective.improvement` | best/final minus initial |
| `objective.contribution_fractions` | final-stage absolute component shares |
| `action.initial/final/best` | effective local depth curves and raw local params |
| `gradients` | nonfinite loss/gradient counts |
| `rollout` | branch-specific rollout metadata |

In joint mode:

- `config.joint_venue` is true
- `config.global_active` records whether global curves were enabled this frame
- `rollout.liquidity_sourcing` is true
- `rollout.consolidated_wealth` is true
- `rollout.global_action_flat` stores the best global first-step raw action
- local action snapshots remain the main action snapshots

`MpcStrategyEngine` embeds this payload in both `StrategyDecision.optimizer` and
`ControllerDiagnostics.optimizer`.

## Execution And Risk Boundaries

The optimizer does not decide which orders are live. It proposes target books.

After optimization:

- `prepare_execution_joint_book()` strips or caps orders according to execution
  config.
- `NetDeltaRiskEngine` approves/rejects the target-book transition against
  ledger state.
- `compute_venue_execution_gates()` independently gates local/global placement.
- `route_joint_target_book_with_report()` executes or previews each venue book.

This means `liquidity_sourcing.enabled` affects the optimizer and target-book
construction, while `liquidity_sourcing.global_live_orders` controls whether
global orders may reach the global `OrderManager`.

## Current Caveats

- Joint action diagnostics currently expose local action snapshots directly and
  global raw action via `rollout.global_action_flat`.
- Joint MPC uses the same inventory target/scale resolver for local and global
  inventory penalties.
- The joint objective uses consolidated wealth and inventory capacity penalties;
  it does not yet include all quote-shape and tail-risk regularizers.
- The fill model remains a differentiable proportional-share approximation, not
  a queue model.
- Same-quote consolidated wealth is assumed by the current implementation; if
  local/global quote currencies differ, an explicit conversion provider is
  needed before live deployment.
