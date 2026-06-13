# Concrete Incremental Improvements for Profitable, Balanced, Less Risky Order Books

This document proposes three concrete, minimally invasive changes to the existing market‑making codebase. Each change targets a critical failure mode that degrades book quality (depth and spread) or increases risk, while preserving the current architecture. The trade‑off between the original risk‑adjusted return objective and a new book‑quality objective is controlled by a single scalar weight. Hard constraints are used where possible to avoid weight tuning.

---

## Proposal 1: Add a Book‑Quality Reward Term to the MPC Objective

**Failure mode**  
The optimizer maximises only risk‑adjusted terminal wealth. It may choose to quote very wide spreads or provide little depth at tight spreads because those actions do not directly improve PnL. This leads to poor liquidity, low fee revenue, and an unattractive order book.

**Change**  
Introduce a configurable term that rewards placing quantity close to the mid price. The term is added to the per‑step wealth delta during Monte Carlo rollouts, making the overall objective a weighted sum of wealth change and book quality.

**Affected files and functions**

- `mm/runtime/config.py` – extend `ControllerRuntimeConfig`
- `mm/mpc/mpc_tensor_batched.py` – modify `_compute_rollout_returns()`
- `mm/mpc/joint_optimizer.py` – modify `_rollout_series()` (joint mode)
- `mm/mpc/joint_mpc.py` – update objective construction (if needed)

**Detailed design**

1. **Configuration** (in `ControllerRuntimeConfig`):
   ```
   book_quality_weight: float = 0.0        # lambda
   book_quality_tightness_scale: float = 0.01  # relative spread where weight halves (e.g., 1% = 0.01)
   book_quality_norm: float = 1000.0       # scaling factor, e.g., typical notional in quote asset
   ```

2. **Book quality score function** (pseudocode, differentiable):
   ```
   def book_quality_score(action_params, mid, spread_grid, tick_size, norm):
       # action_params: tensor of raw sigmoid parameters for one venue, one step
       # spread_grid: 1D tensor of absolute spreads (mid - bid, ask - mid)
       # mid: scalar tensor for current step
       # returns scalar tensor: total weighted notional
       
       # Convert raw action to cumulative depth at each grid point
       cum_depth = sigmoid_mixture_depth(action_params, spread_grid)
       # Marginal quantities (first level = cum_depth[0], then differences)
       marginal = diff(cum_depth, prepend=0)
       
       # Relative spreads (unitless)
       rel_spread = spread_grid / mid
       # Exponential weight: exp(-rel_spread / tightness_scale)
       weight = torch.exp(-rel_spread / tightness_scale)
       
       # Notional of each level: quantity * mid
       notional = marginal * mid
       
       # Weighted sum, then normalized
       score = (weight * notional).sum() / norm
       return score
   ```

3. **Integration into rollout** (in `_compute_rollout_returns`):
   - After computing `wealth_delta` for step `h` and path `i`, compute:
     ```
     score = book_quality_score(local_action_h, mid_h, spread_grid, book_quality_norm)
     augmented_delta = wealth_delta + book_quality_weight * score
     ```
   - Accumulate `augmented_delta` instead of `wealth_delta` into the per‑path cumulative return.
   - For joint mode, compute the score for local and global actions separately, weight them by the same `book_quality_weight` (or separate weights), and add to the respective venue wealth deltas before consolidating.

4. **Scaling considerations**  
   - The `book_quality_norm` should be set to a typical notional value (e.g., `max_order_size * mid`). Default to 1000 USDT.  
   - The `book_quality_weight` is then a dimensionless coefficient. Start with `0.01` and tune.

**Effort estimate**  
- 1–2 days for implementation, including unit tests and a simple integration test.

**Prerequisites**  
- None. The change is purely additive and does not require new infrastructure.

---

## Proposal 2: Hard Constraint – Minimum Depth at the Innermost Spread Level

**Failure mode**  
The optimizer can produce a cumulative depth curve that has near‑zero quantity at the tightest spread (the first level). This results in an empty top‑of‑book, making the strategy effectively non‑competitive. Because this is a hard constraint, it requires no tuning.

**Change**  
After building the continuous sigmoid depth curve, project it so that the cumulative depth at the smallest configured spread is at least a configurable minimum quantity. The projection preserves the total depth and the shape as much as possible.

**Affected files and functions**

- `mm/mpc/action.py` – add `enforce_min_first_level_depth()`  
- `mm/application/mpc_engine.py` – call the function in `build_joint_target_book_from_sigmoid()` (or directly in `MpcStrategyEngine.decide()` after generating the best action)

**Detailed design**

1. **Configuration** (add to `ControllerRuntimeConfig`):
   ```
   min_first_level_quantity: float = 0.0   # minimum base quantity at the first spread level (absolute)
   ```

2. **Projection logic** (pseudocode):
   ```
   def enforce_min_first_level_depth(cum_depth_fn, spread_grid, min_qty):
       # cum_depth_fn: callable that returns cumulative depth at any spread (e.g., the sigmoid mixture)
       # spread_grid[0] is the smallest positive spread
       # returns adjusted raw parameters or directly modified cumulative depth values
       
       current_depth_at_first = cum_depth_fn(spread_grid[0])
       if current_depth_at_first >= min_qty:
           return  # no change
       
       # Scale factor needed to raise first level to min_qty
       scale = min_qty / current_depth_at_first
       
       # Apply scaling to all L (level) parameters and to the total depth logit?
       # Simpler: scale the total depth (u_depth) so that the entire curve is multiplied by scale.
       # This preserves the shape (relative depth distribution).
       # In the sigmoid mixture, the total depth is a separate logit parameter u_s_depth.
       # We can increase u_s_depth so that the new total depth = old_total_depth * scale.
       # Because depth scales linearly with the softmax‑normalised L_eff.
       
       new_total_depth = old_total_depth * scale
       # Compute new u_s_depth = logit( (new_total_depth - min_total) / (max_total - min_total) )
       # where min_total = quote_size_min_per_side, max_total = quote_size_max_per_side
       # Clamp to [logit_min, logit_max]
       # Replace the u_s_depth in the raw action vector.
   ```
   - Apply this separately for bid and ask.
   - The function modifies the raw action parameters **before** discretization.

3. **Integration point** – in `MpcStrategyEngine.decide()`, after obtaining the best action from the optimizer and before calling `build_joint_target_book_from_sigmoid`, call `enforce_min_first_level_depth()` on both local and global actions.

**Effort estimate**  
- 0.5–1 day, mostly for testing the projection with edge cases (e.g., when `min_qty` exceeds the maximum possible depth).

**Prerequisites**  
- Access to the raw action parameters after optimisation (already available).

---

## Proposal 3: Dynamic Spread Cap Based on Recent Volatility

**Failure mode**  
The static `spread_max` (maximum allowed quote spread) does not adapt to market conditions. When volatility is low, wide spreads are unnecessarily uncompetitive. When volatility spikes, the static cap may be too tight, causing the strategy to quote unsafe levels and suffer adverse selection.

**Change**  
Before each decision frame, compute a short‑term volatility estimate from the local market feed and dynamically cap the spread used by the MPC. The cap is still a hard constraint (no weight to tune).

**Affected files and functions**

- `mm/application/strategy_runner.py` – modify `_on_joint_frame()` to compute volatility and pass a dynamic `spread_max` to the controller.
- `mm/mpc/mpc_state.py` – allow `spread_max` to be overridden per frame (or pass it through `ControllerRuntimeConfig` as a callable).
- `mm/runtime/config.py` – add `volatility_spread_multiplier` and `volatility_window_seconds`.

**Detailed design**

1. **Configuration** (add to `ControllerRuntimeConfig`):
   ```
   volatility_spread_multiplier: float = 2.0   # number of standard deviations for spread cap
   volatility_window_seconds: float = 60.0     # lookback for mid price returns
   max_spread_volatility_factor: float = 0.05  # spread cap = max(static_min, vol_mult * vol * mid)
   static_spread_max_absolute: float           # existing spread_max, used as upper bound
   ```

2. **Volatility estimation** – in `StrategyRunner`, maintain a rolling buffer of mid prices from local `MarketEvent`s (already published). At each frame:
   - Compute log returns over the window `volatility_window_seconds`.
   - Compute standard deviation of returns (annualised not needed, use raw).
   - Estimate `volatility_estimate = std(log_returns) * mid` (absolute price volatility).
   - Dynamic spread cap = `min(static_spread_max_absolute, volatility_spread_multiplier * volatility_estimate)`.
   - Clamp to a reasonable minimum (e.g., 2× tick size).

3. **Pass to MPC** – The `State` constructor already accepts a `spread_max` argument. Modify the call in `MpcStrategyEngine._build_state()` to use the dynamic value from the runner.

**Effort estimate**  
- 0.5–1 day, assuming market events already contain mid prices (they do). No new data sources required.

**Prerequisites**  
- Access to a rolling buffer of mid prices – trivial to add in `StrategyRunner`.

---

## Summary of Recommendations

| Proposal | Type | Weight tuning needed? | Effort | Impact |
|----------|------|----------------------|--------|--------|
| 1. Book‑quality reward | Soft objective | One scalar (`book_quality_weight`) | 1–2 days | Directly improves depth at tight spreads, increases fee revenue |
| 2. Minimum depth at first level | Hard constraint | None | 0.5–1 day | Guarantees a non‑empty top‑of‑book |
| 3. Dynamic spread cap | Hard constraint | None | 0.5–1 day | Reduces adverse selection, adapts to volatility |

**Implementation order** (least architectural disruption first):  
2 → 3 → 1  

Proposal 2 is the simplest and most robust (hard constraint). Proposal 3 is also a hard constraint and improves risk management. Proposal 1 provides the tunable trade‑off between profitability and book quality; start with a small weight (e.g., 0.01) and increase gradually.

All changes are backward‑compatible (new config fields default to disabling the feature). They extend existing components without replacing them, as requested.