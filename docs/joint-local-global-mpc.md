# Joint Local/Global MPC Models And Optimizer

This document describes the current joint local/global MPC path used when
`liquidity_sourcing.enabled=true` and `liquidity_sourcing.mode="liquidity_sourcing"`.
It focuses on how the local and global models feed simulation, how the global
action is simulated, what the joint objective is, and how venue-scoped optimizer
parameters are configured.

Code map:

- Runtime strategy entry: `mm/strategies/mpc_engine.py`
- Optimizer class: `mm/mpc/mpc_tensor_batched.py`
- Joint rollout: `mm/mpc/joint_optimizer.py`
- Joint action/objective helpers: `mm/mpc/joint_mpc.py`
- Target-book conversion: `mm/strategies/target_book_builder.py`
- Example config: `mm/configs/market_maker.runtime.example.json`

## Runtime Path

One joint frame goes through:

```text
StrategyRunner
  -> MpcStrategyEngine.decide(frame)
  -> build State from local/global market data and ledger balances
  -> score model observations before update
  -> update local/global online models
  -> MonteCarloMPCTensorBatched.optimize_with_diagnostics(state)
  -> optimize_joint_with_diagnostics(state)
  -> discretize first-step local/global sigmoid curves into JointTargetBook
  -> publish StrategyDecision + JointTargetBook + Action
```

Model scoring is prequential: diagnostics score the current observation using
the model parameters from before the update. The optimizer then uses the updated
model parameters for the current decision.

## Model Inputs

The current controller has four first-class models:

| Model | Config key | Used for |
| --- | --- | --- |
| Local market flow | `controller.local_market_flow_model` | Simulated local buy/sell market-order volume |
| Local price | `controller.local_price_model` | Simulated local VWAP residuals |
| Global market flow | `controller.global_market_flow_model` | Simulated global buy/sell market-order volume |
| Global price | `controller.global_price_model` | Simulated global VWAP residuals or deltas |

The local and global market-flow models are independent instances. The global
fill probability is not derived from local flow anymore.

The price models do not fetch future exchange prices. They generate residuals
around the simulated book-derived VWAP path. Exchange APIs only provide the
current observed frame and ledger state before optimization starts.

## Joint Action Shape

The optimizer chooses sigmoid quote-depth curves. One venue action contains a
bid curve and an ask curve. In joint mode, one horizon step is:

```text
joint_step = [local_bid_ask_curves, global_bid_ask_curves]
```

With `n_components = n`, one venue uses `6 * n` raw parameters:

```text
bid: L_i, k_i, x0_i for i=1..n
ask: L_i, k_i, x0_i for i=1..n
```

Joint mode uses twice that:

```text
len(joint_step) = 2 * 6 * n
len(full_optimizer_vector) = horizon * 2 * 6 * n
```

The first optimized step is converted into:

- local target orders from the local sigmoid action
- global target orders from the global sigmoid action, only when global is
  active and a global frame/constraints are available

The local and global curves use the same sigmoid-mixture parameterization, but
their quote-size, spread-shaping, soft-fill, and objective parameters are
configured separately.

## Global Active Mask

The joint optimizer can run while the global part is inactive. The mask is:

```text
global_mask = 1.0 if state.global_active else 0.0
```

When global is inactive:

- global soft-fill depth is multiplied by `0.0`
- global simulated fills are effectively zero
- global inventory penalty is masked out
- no global sigmoid action is exposed to target-book construction

When global is active:

- global soft-fill depth is multiplied by `1.0`
- global inventory/cash evolve from simulated global fills
- global target orders can be emitted into `JointTargetBook.global_`

## Simulation: Local Vs Global

Local and global use the same simulation mechanics:

1. Draw market-flow samples for the venue.
2. Draw VWAP residual samples for the venue.
3. Compute current VWAPs from the venue's simulated market-only book.
4. Add the venue price residuals.
5. Run `soft_fill_batched()` against that venue's book.
6. Update that venue's cash and inventory.
7. Carry the venue's remaining market-only book into the next horizon step.

The important differences are the inputs and the mask:

| Aspect | Local | Global |
| --- | --- | --- |
| Book | `state.ob_rel` | `state.global_ob_rel` |
| Flow model | `local_market_flow_model` | `global_market_flow_model` |
| Price model | `local_price_model` | `global_price_model` |
| Cash/inventory | `state.cash`, `state.inventory` | `state.global_cash`, `state.global_inventory` |
| Soft-fill depth scale | `1.0` | `1.0` if active, else `0.0` |
| Inventory penalty | Always included | Included only if active |

`soft_fill_batched()` is not an exchange queue-position simulator. It is a
differentiable liquidity-share model:

- market sells consume bid-side liquidity, including our bid quotes
- market buys consume ask-side liquidity, including our ask quotes
- own fills produce cash and inventory deltas
- unfilled own quote remainder is not carried as own resting liquidity into the
  next simulated horizon step

So the global side is not special because of an exchange API. It is special
because it uses its own global book, global flow model, global price model, and
is gated by `global_active`.

## Reference Price And Consolidated Wealth

Joint MPC does not optimize separate local PnL plus global PnL. It marks one
consolidated portfolio:

```text
W = local_cash + global_cash
    + (local_inventory + global_inventory) * reference_price
```

The reference path is selected by `controller.reference_price_source`:

- `global`: mark inventory and objective returns with the simulated global
  sell-side VWAP path, falling back to local if no global VWAP is available.
- `local`: mark inventory and objective returns with the simulated local
  sell-side VWAP path.

Local and global future VWAP paths are still simulated separately from their own
books and price models. Only the selected path is used as the valuation
reference. A global reference path can be used even when
`liquidity_sourcing.enabled=false`; in that mode global public data affects the
reference price, but global inventory, cash, fills, and actions stay out of the
portfolio.

VWAP depth is not a fixed controller config. The runtime derives it each frame
from `abs(local_inventory)`, or from `abs(local_inventory + global_inventory)`
when liquidity sourcing is enabled. If the requested depth is larger than
visible book depth, the VWAP calculation consumes only the available displayed
depth.

## Joint Objective

The current joint objective is:

```text
local_returns = total_local_delta_wealth / initial_consolidated_wealth
global_returns = total_global_delta_wealth / initial_consolidated_wealth

J =
  local.objective_weight
    * entropic_return_beta(local_returns, local.return_risk_aversion_beta)
  + global_active
    * global.objective_weight
    * entropic_return_beta(global_returns, global.return_risk_aversion_beta)
  - local.inventory_target_penalty_weight
    * local_inventory_capacity_penalty
  - global_active
    * global.inventory_target_penalty_weight
    * global_inventory_capacity_penalty
```

The optimizer minimizes:

```text
loss = -J
```

Objective components reported in diagnostics:

| Component | Meaning |
| --- | --- |
| `risk_adjusted_return` | Sum of local and active global weighted return terms |
| `local_risk_adjusted_return` | Local weighted entropic return term |
| `global_risk_adjusted_return` | Global weighted entropic return term, masked when inactive |
| `inventory_capacity_penalty` | Combined weighted inventory penalty |
| `local_inventory_penalty` | Local weighted inventory contribution |
| `global_inventory_penalty` | Global weighted inventory contribution, masked when inactive |

## What "Mixed Objective" Means Here

The mixed objective is the configurable sum of local and global return terms,
plus their own inventory penalties:

```text
local_weight * local_objective + global_weight * global_objective
```

The return contribution weights are:

- `controller.optimizer.local.objective_weight`
- `controller.optimizer.global.objective_weight`

Each venue return is still marked in the same consolidated accounting system:
local and global inventory are valued against the rollout reference price. That
means the local/global tradeoff comes from:

- simulated local fills
- simulated global fills
- the reference price used to mark combined inventory
- local/global objective weights
- local/global inventory penalty weights
- the global active mask

If you set `global.objective_weight=0`, global fills can still affect global
inventory penalty and target-book diagnostics, but the global return term is not
rewarded. If `global_active=false`, the global return and inventory terms are
both masked out.

## Configured Joint Objective Weights

These current joint objective parameters are configured under
`controller.optimizer.local` and `controller.optimizer.global`:

```json
{
  "optimizer": {
    "local": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_target_penalty_weight": 0.000001,
      "target_inventory": 0.0
    },
    "global": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_target_penalty_weight": 0.000001,
      "target_inventory": 0.0
    }
  }
}
```

Meaning:

| Parameter | Current effect in joint mode |
| --- | --- |
| `objective_weight` | Venue return-term multiplier |
| `return_risk_aversion_beta` | Venue risk aversion for entropic return |
| `inventory_target_penalty_weight` | Venue inventory-capacity penalty weight |
| `target_inventory` | Venue inventory target in base units |

The inventory target is used directly:

```text
local_target_inventory = local.target_inventory
global_target_inventory = global.target_inventory
```

## Separate Optimizer Parameters

These optimizer parameters are separate for local and global:

```json
{
  "optimizer": {
    "local": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_target_penalty_weight": 0.000001,
      "target_inventory": 0.0,
      "quote_size_min_per_side": 0.3,
      "quote_size_max_per_side": 10.0,
      "quote_capacity_ratio": 1.0,
      "quote_spread_max": 100.0,
      "quote_spread_min_depth": 0.3,
      "quote_spread_k_epsilon": 0.01,
      "soft_fill_temp": 0.1,
      "soft_fill_activation": "hardsigmoid",
      "soft_fill_queue_penalty": 1.0
    },
    "global": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_target_penalty_weight": 0.000001,
      "target_inventory": 0.0,
      "quote_size_min_per_side": 0.3,
      "quote_size_max_per_side": 10.0,
      "quote_capacity_ratio": 1.0,
      "quote_spread_max": 100.0,
      "quote_spread_min_depth": 0.3,
      "quote_spread_k_epsilon": 0.01,
      "soft_fill_temp": 0.1,
      "soft_fill_activation": "hardsigmoid",
      "soft_fill_queue_penalty": 1.0
    }
  }
}
```

Current behavior:

| Parameter | Venue-scoped? | Notes |
| --- | --- | --- |
| `objective_weight` | Yes | Venue multiplier in mixed objective |
| `return_risk_aversion_beta` | Yes | Venue entropic-return risk aversion |
| `inventory_target_penalty_weight` | Yes | Venue inventory penalty weight |
| `target_inventory` | Yes | Venue inventory target in base units |
| `quote_size_min_per_side` | Yes | Venue sigmoid depth lower range |
| `quote_size_max_per_side` | Yes | Venue sigmoid depth upper range |
| `quote_capacity_ratio` | Yes | Venue side max ratio: bid max uses `cash / mid`, ask max uses inventory, both floored at `quote_size_min_per_side` |
| `quote_spread_max` | Yes | Venue structured spread cap |
| `quote_spread_min_depth` | Yes | Venue structured minimum depth |
| `quote_spread_k_epsilon` | Yes | Venue spread-parameter numerical guard |
| `soft_fill_temp` | Yes | Venue fill softness |
| `soft_fill_activation` | Yes | Venue fill activation |
| `soft_fill_queue_penalty` | Yes | Venue queue-at-back penalty; default `1` treats one worse public level as ahead, `0` is pro-rata |

Separate today:

| Surface | Local | Global |
| --- | --- | --- |
| Market-flow model | `local_market_flow_model` | `global_market_flow_model` |
| Price model | `local_price_model` | `global_price_model` |
| Market book | local event | global event |
| Ledger balances/orders | local ledger | global ledger |
| Instrument constraints | local symbol spec | global symbol spec |
| Execution planner | `execution.local` | `execution.global` |
| Hard risk limits | `controller.risk.local` | `controller.risk.global` |
| Live placement switch | normal local live gate | `liquidity_sourcing.global_live_orders` |

## Implementation Notes

`MonteCarloMPCTensorBatched` owns two venue parameter bundles:

- `local_optimizer`
- `global_optimizer`

Joint rollout uses:

- separate local/global `LParamConfig`
- separate local/global raw bounds
- separate local/global soft-fill settings
- separate local/global inventory targets and penalty weights
- separate local/global risk-aversion betas and objective weights

Flat venue-specific fields under `controller.optimizer` are rejected. Put them
under `controller.optimizer.local` and `controller.optimizer.global`.

## Practical Reading Of Diagnostics

For the joint optimizer:

- `optimizer.config.global_active` tells whether the global action participated.
- `optimizer.config.reference_price` is the initial marking price.
- `optimizer.config.reference_price_source` tells which VWAP path marked
  objective returns.
- `optimizer.config.vwap_depth` is the initial inventory-derived VWAP depth.
- `optimizer.config.wealth_0` is the normalization base for returns.
- `optimizer.config.local_optimizer` and `optimizer.config.global_optimizer`
  show the resolved venue-scoped optimizer parameters.
- `optimizer.objective.*.components.risk_adjusted_return` is the return term.
- `optimizer.objective.*.components.local_risk_adjusted_return` is the local
  return contribution.
- `optimizer.objective.*.components.global_risk_adjusted_return` is the global
  return contribution.
- `optimizer.objective.*.components.local_inventory_penalty` is the local
  inventory term.
- `optimizer.objective.*.components.global_inventory_penalty` is the global
  inventory term.
- `optimizer.rollout.local_fill_*_mean` and `global_fill_*_mean` show expected
  first-step simulated fills.
- `optimizer.rollout.global_action_flat` is the optimized global first-step raw
  vector when joint mode runs.

If `global_inventory_penalty` is zero and global fill means are zero, check
`global_active`, frame readiness, global staleness, and global model outputs.
`liquidity_sourcing.global_live_orders` is an execution switch after
optimization; it does not change the optimizer simulation itself.
