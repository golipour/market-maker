```markdown
# Incremental Improvements for Balanced, Low-Risk, High-Quality Market Making

Based on the code-derived system documentation, the current MPC optimizer maximizes a risk-adjusted return (Entropic Certainty Equivalent) but lacks explicit mechanisms to guarantee book quality (depth/spread) or enforce hard safety boundaries on inventory and drawdowns. 

Below are four concrete, incrementally implementable proposals. They prioritize **hard constraints** (which require no weight tuning) to fix critical failure modes, alongside **one tunable objective extension** to explicitly balance return against book quality.

---

## 1. Hard Terminal Inventory Constraint via Rollout Penalty
**Goal:** Less risky, balanced book. Prevents the optimizer from exploring paths that lead to dangerous inventory accumulation, without needing to tune a quadratic penalty weight.
- **File:** `mm/mpc/mpc_tensor_batched.py` (or `mm/mpc/joint_mpc.py` for joint mode)
- **Function:** The inner rollout loop (where `Q_{i,h+1}` and `X_{i,h+1}` are computed).
- **Description of Change:** Instead of adding a soft quadratic penalty to the objective, enforce a hard constraint by assigning a catastrophic negative reward to any Monte Carlo path that breaches a configured absolute inventory limit (`Q_max`). This naturally forces the Adam optimizer to avoid these regions of the action space without requiring Lagrangian multipliers or weight tuning.
- **Pseudocode/Math:**
  ```python
  # Inside the horizon rollout loop for path i, step h:
  Q_next = Q_current + inventory_delta
  
  # Hard constraint: if inventory exceeds safe bounds, invalidate the path
  if abs(Q_next) > Q_hard_max:
      R_i = -1e9  # Catastrophic penalty ensures this path is rejected by Adam
      break       # Early exit for this Monte Carlo path
  
  # Otherwise, continue normal wealth calculation
  W_h = X_next + Q_next * mark(Q_next)
  ```
- **Implementation Effort:** Low (10–15 lines of code).
- **Prerequisites:** Add `inventory_hard_max` to `ControllerRuntimeConfig` (can be derived from venue limits or set as a multiple of `quote_size_max_per_side`).

---

## 2. Tunable Book-Quality Objective (Spread & Depth)
**Goal:** Achieve "good depth and spread" via a tunable trade-off between risk-adjusted return and book quality.
- **File:** `mm/mpc/reward_tensor.py` (and `mm/mpc/joint_mpc.py` for joint mode)
- **Function:** The terminal objective computation (where `J = CE_beta(R)` is calculated).
- **Description of Change:** Introduce a `BookQualityScore` that penalizes the optimizer if the resulting target book has a spread wider than a target or lacks depth at the best levels. This is combined with the existing return objective using a scalar weight `w_quality`.
- **Pseudocode/Math:**
  ```python
  # 1. Calculate standard risk-adjusted return
  J_return = CE_beta(R) 
  
  # 2. Calculate Book Quality Penalty (evaluated on the expected/mean action)
  # spread_penalty: 0 if spread <= target, linearly increases otherwise
  spread_penalty = max(0.0, (mean_spread - target_max_spread) / target_max_spread)
  
  # depth_penalty: 0 if depth >= target, linearly increases otherwise
  depth_penalty = max(0.0, (target_min_depth - mean_best_level_depth) / target_min_depth)
  
  J_quality = - (w_spread * spread_penalty + w_depth * depth_penalty)
  
  # 3. Combine objectives
  # w_return is implicitly 1.0, w_quality is a tunable scalar (e.g., 0.1 to 0.5)
  J_final = J_return + (w_quality * J_quality)
  
  loss = -J_final
  ```
- **Implementation Effort:** Medium (requires extracting mean spread/depth from the continuous action tensor before discretization).
- **Prerequisites:** Add `target_max_spread`, `target_min_depth`, and `w_quality` to `ControllerRuntimeConfig`. Requires the optimizer to have access to the mean action's implied book shape (already available via the sigmoid parameterization).

---

## 3. Hard Constraint on First-Level Quote Depth (Guaranteed Depth)
**Goal:** Ensure "good depth" at the top of the book deterministically, without relying on the optimizer to learn it or tuning objective weights.
- **File:** `mm/mpc/mpc_strategy_engine.py` (or wherever `MpcStrategyEngine` is defined)
- **Function:** `build_joint_target_book_from_sigmoid` (the discretization step).
- **Description of Change:** After converting the continuous cumulative depth curve into marginal order sizes, enforce a hard floor on the quantity of the first level (best bid and best ask). If the optimizer suggests less, clamp it to the minimum. To maintain budget feasibility, proportionally reduce deeper, less competitive levels or cap the total notional.
- **Pseudocode/Math:**
  ```python
  def build_joint_target_book_from_sigmoid(continuous_action, config, ledger):
      marginal_sizes = compute_marginal_from_cumulative(continuous_action)
      
      for side in [BID, ASK]:
          # Hard constraint: guarantee minimum depth at the best level
          if marginal_sizes[side][0] < config.min_depth_per_side:
              shortfall = config.min_depth_per_side - marginal_sizes[side][0]
              marginal_sizes[side][0] = config.min_depth_per_side
              
              # Rebalance: subtract shortfall from the deepest levels to respect total capacity
              for i in reversed(range(1, len(marginal_sizes[side]))):
                  deduction = min(shortfall, marginal_sizes[side][i])
                  marginal_sizes[side][i] -= deduction
                  shortfall -= deduction
                  if shortfall <= 0:
                      break
                      
      return build_venue_target_book(marginal_sizes, ...)
  ```
- **Implementation Effort:** Low (modifies existing discretization logic).
- **Prerequisites:** Add `min_depth_per_side` to `ExecutionRuntimeConfig` or `ControllerRuntimeConfig`.

---

## 4. Hard Drawdown Constraint in Rollout
**Goal:** Less risky. Prevents the optimizer from accepting strategies with high expected return but catastrophic tail risk (a known gap noted in the documentation).
- **File:** `mm/mpc/mpc_tensor_batched.py` 
- **Function:** The inner rollout loop (same location as Proposal 1).
- **Description of Change:** Enforce a hard maximum drawdown limit per Monte Carlo path. If the simulated wealth at any horizon step drops below a configured fraction of initial wealth, the path is immediately invalidated. Like Proposal 1, this requires no weight tuning in the objective function.
- **Pseudocode/Math:**
  ```python
  # Inside the horizon rollout loop for path i, step h:
  W_h = X_h + Q_h * mark(Q_h)
  
  # Hard constraint: max drawdown limit (e.g., 5%)
  if W_h < W_0 * (1.0 - config.max_allowed_drawdown):
      R_i = -1e9  # Invalidate path
      break
  ```
- **Implementation Effort:** Low (5 lines of code).
- **Prerequisites:** Add `max_allowed_drawdown` (float, e.g., 0.05) to `ControllerRuntimeConfig`.

---

## Summary of Prioritization

1. **Implement Proposal 1 & 4 First (Hard Constraints):** These directly address the documentation's explicit note that "Current code does not include an explicit quadratic inventory penalty... or drawdown constraint." They require **zero weight tuning**, are trivial to implement, and immediately eliminate catastrophic tail-risk failure modes.
2. **Implement Proposal 3 Next (Hard Depth Floor):** This guarantees "good depth" at the best levels deterministically during discretization, ensuring the bot remains competitive without disrupting the core MPC math.
3. **Implement Proposal 2 Last (Tunable Book Quality):** Once the hard safety rails are in place, this provides the fine-grained, scalar-weighted dial (`w_quality`) to smoothly trade off between pure profit maximization and providing superior market liquidity (tighter spreads, deeper books).
```