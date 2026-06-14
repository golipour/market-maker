# Incremental Improvements for Profitability, Inventory Control, and Book Quality

## Goal

Improve the existing market maker so that it produces:

- Higher risk-adjusted profitability
- Lower inventory risk
- Better order-book quality
  - tighter spread
  - deeper book
  - more balanced bid/ask presence
  - fewer empty-sided books

while preserving the current architecture and optimizer framework.

The objective should become a tunable trade-off:

```text
Objective =
    RiskAdjustedReturn
  + λ_book * BookQuality
```

where:

- `RiskAdjustedReturn` remains the primary profitability objective.
- `BookQuality` measures spread, depth, balance, and quote presence.
- `λ_book` controls the trade-off.

Importantly, many structural failures should not be handled through objective weights at all. They should be enforced through hard constraints because they represent invalid books rather than preferences.

---

# Design Philosophy

Current architecture already has:

- differentiable rollout
- sigmoid quote curves
- dynamic inventory-aware capacities
- entropic certainty equivalent objective
- downstream risk validation

The largest weakness is:

```text
Optimizer is rewarded almost exclusively for wealth generation.
```

As a result it can legitimately choose:

- very wide spreads
- one-sided books
- almost no liquidity
- excessive inventory accumulation

if those maximize the certainty equivalent.

The best improvements are therefore:

1. Add hard structural constraints.
2. Add small objective extensions.
3. Reuse existing rollout state.
4. Avoid architectural changes.

---

# Priority 1: Hard Minimum Two-Sided Quoting Constraint

## Failure Mode

Optimizer can collapse one side.

Examples:

- only bids
- only asks
- almost zero depth on one side

Book quality becomes poor.

---

## Location

### File

```text
mm/mpc/mpc_tensor_batched.py
```

### Function

Where action parameters are converted into effective quote curves before rollout.

Likely near:

```python
build_action(...)
decode_action(...)
action_to_depth(...)
```

depending on exact implementation.

---

## Change

Enforce:

```text
Q_bid_total >= min_side_depth
Q_ask_total >= min_side_depth
```

after capacity limits.

Current code already computes:

```text
bid_max
ask_max
bid_min
ask_min
```

Increase lower bounds:

```text
bid_min = max(bid_min, min_side_depth)
ask_min = max(ask_min, min_side_depth)
```

whenever capacity allows.

---

## Why Constraint Instead of Objective

This is not a preference.

A one-sided book is simply invalid for market making.

No tuning weight needed.

---

## Effort

Low

~20-40 LOC

---

## Dependencies

None

---

# Priority 2: Hard Maximum First Quote Distance

## Failure Mode

Optimizer can earn good CE by placing quotes extremely far away.

Produces:

- huge spread
- almost zero fill probability
- poor market quality

---

## Location

### File

```text
mm/mpc/mpc_tensor_batched.py
```

or

```text
mm/mpc/action_parameterization.py
```

where first sigmoid center is decoded.

---

## Change

Require:

```text
best_bid_distance <= spread_cap
best_ask_distance <= spread_cap
```

For example:

```text
spread_cap = k * current_market_spread
```

or

```text
spread_cap = fixed_bps
```

Enforce directly in parameter decoding.

---

## Why Constraint

Quotes beyond some distance are not useful liquidity.

No objective weight required.

---

## Effort

Low

---

## Dependencies

None

---

# Priority 3: Hard Bid/Ask Balance Constraint

## Failure Mode

Optimizer may quote:

```text
Bid depth = 10
Ask depth = 0.5
```

without actually intending inventory rebalancing.

Creates unstable books.

---

## Location

Quote-capacity stage.

---

## Change

Constrain:

```text
depth_ratio =
max(Q_bid_total,Q_ask_total)
/
min(Q_bid_total,Q_ask_total)
```

Require:

```text
depth_ratio <= max_balance_ratio
```

Example:

```text
max_balance_ratio = 3
```

unless inventory exceeds threshold.

Inventory exception:

```text
if abs(inventory) > inventory_skew_threshold:
    allow stronger imbalance
```

---

## Why Constraint

This is structural book quality.

Not a reward preference.

---

## Effort

Low

---

## Dependencies

None

---

# Priority 4: Inventory Penalty in Objective

## Failure Mode

Current system only handles inventory indirectly through:

- marking
- capacity
- skew

Optimizer may still accumulate inventory.

---

## Location

### File

```text
mm/mpc/reward_tensor.py
```

and

```text
mm/mpc/joint_mpc.py
```

---

## Change

Add terminal inventory penalty:

```text
InvPenalty =
E[
(Q_terminal / inventory_scale)^2
]
```

New objective:

```text
J =
CE_beta(R)
-
λ_inv * InvPenalty
```

---

## Reason

Inventory risk is the largest missing objective term.

---

## Tunable Weight

```text
λ_inv
```

---

## Effort

Low

---

## Dependencies

None

---

# Priority 5: Book Quality Objective

## Failure Mode

Current objective has no notion of market quality.

---

## Location

### File

```text
mm/mpc/reward_tensor.py
```

---

## Change

Add:

```text
BookQuality =
w_spread * SpreadScore
+
w_depth * DepthScore
+
w_balance * BalanceScore
```

where

### Spread Score

Prefer tighter first quotes:

```text
SpreadScore =
-E[
best_bid_distance
+
best_ask_distance
]
```

---

### Depth Score

Reward displayed liquidity:

```text
DepthScore =
E[
total_bid_depth
+
total_ask_depth
]
```

---

### Balance Score

Reward symmetry:

```text
BalanceScore =
-E[
|Q_bid_total - Q_ask_total|
]
```

---

Then:

```text
J =
CE_beta(R)
-
λ_inv * InvPenalty
+
λ_book * BookQuality
```

---

## Tunable Weights

Primary user controls:

```text
λ_book
λ_inv
```

Only two scalars.

---

## Effort

Medium

---

## Dependencies

None

---

# Priority 6: Inventory-Aware Book Quality

## Failure Mode

Pure balance rewards can fight inventory management.

---

## Location

Same objective code.

---

## Change

Define desired imbalance:

```text
target_skew =
f(inventory)
```

Example:

```text
target_skew =
tanh(inventory / inventory_scale)
```

Then:

```text
BalanceScore
=
-
(skew_actual - target_skew)^2
```

instead of forcing equal depth.

---

## Benefit

Book quality and inventory management become aligned.

---

## Effort

Medium

---

## Dependencies

Inventory already available in rollout state.

---

# Priority 7: Fill-Efficiency Objective

## Failure Mode

Optimizer may create depth that never trades.

---

## Location

### File

```text
mm/mpc/reward_tensor.py
```

---

## Change

Use rollout fill statistics already produced by:

```text
soft_fill_batched()
```

Define:

```text
FillEfficiency =
fills
/
displayed_depth
```

Reward:

```text
λ_fill * FillEfficiency
```

---

## Benefit

Encourages useful liquidity rather than decorative depth.

---

## Effort

Low-Medium

---

## Dependencies

Fill metrics already available.

---

# Priority 8: Multi-Step Inventory Drift Penalty

## Failure Mode

Terminal inventory only penalizes end state.

Path risk remains.

---

## Location

Reward calculation.

---

## Change

Add:

```text
InventoryPathPenalty
=
E[
Σ_h Q_h²
]
```

Objective:

```text
J =
CE_beta(R)
-
λ_inv_path * InventoryPathPenalty
```

---

## Benefit

Reduces inventory excursions.

---

## Effort

Low

---

## Dependencies

Inventory trajectory already exists.

---

# Priority 9: Quote Persistence Penalty

## Failure Mode

Optimizer may oscillate between books.

Creates churn.

---

## Location

Optimizer objective.

---

## Change

Penalize deviation from current action:

```text
ActionChangePenalty
=
||a_t - a_prev||²
```

Objective:

```text
J =
...
-
λ_churn * ActionChangePenalty
```

---

## Benefit

Lower cancellations.

Lower fees.

More stable liquidity.

---

## Effort

Low

---

## Dependencies

Previous action already available through warm-start state.

---

# Recommended Final Objective

Keep hard constraints for structural quality:

```text
1. Minimum depth per side
2. Maximum first quote distance
3. Maximum bid/ask imbalance ratio
```

These require no weights.

Then optimize:

```text
J
=
CE_beta(Return)
-
λ_inv      * InventoryPenalty
-
λ_path     * InventoryPathPenalty
+
λ_book     * BookQuality
+
λ_fill     * FillEfficiency
-
λ_churn    * ActionChangePenalty
```

For simplicity, start with:

```text
J
=
CE_beta(Return)
-
λ_inv * InventoryPenalty
+
λ_book * BookQuality
```

and keep all structural book-quality requirements as hard constraints.

This provides the largest improvement in profitability-vs-book-quality behavior with the smallest architectural disruption.