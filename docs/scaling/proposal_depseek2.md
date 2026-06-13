## Classification of Existing Book‑Quality & Inventory Safety Mechanisms

| Mechanism | Location | Current Role | Classification | Justification |
|-----------|----------|--------------|----------------|----------------|
| **Imbalance skew floors** (`bid_floor` / `ask_floor` based on inventory imbalance) | `mm/mpc/mpc_tensor_batched.py` (capacity computation) | Forces a minimum quote size on the side opposite to inventory – a lower bound, not a hard cap. | **(b) Graded preference → tunable soft objective** | A hard floor can over‑constrain near‑neutral inventory; a soft penalty on imbalance in quoted depth is more flexible and can be weighted. |
| **Capacity ratio** (`quote_capacity_ratio * cash / mid`) | Same | Caps max quote size by available cash or inventory. | **(a) Hard structural constraint** | This is a feasibility limit: quoting more than available collateral would cause settlement failure. Keep as is. |
| **Spread caps** (`spread_max`, `x0_eff = spread_max * sigmoid(...)`) | `mm/mpc/mpc_tensor_batched.py` (parameter decoder) | Enforces maximum spread (tick‑to‑mid). | **(a) Hard structural constraint** | Essential for regulatory and risk limits; must never be violated. |
| **Min / max size per side** (`quote_size_min_per_side`, `quote_size_max_per_side`) | Config and capacity bounds | Hard limits on per‑side quote quantity. | **(a) Hard structural constraint** | Operational minima (e.g., exchange min order size) and maxima (risk limit). |
| **Fractional‑inventory sizing** (dynamic capacity per rollout step) | Same, in `soft_fill_batched` and capacity update | Re‑computes max bid/ask from updated cash/inventory each step. | **(c) Replace with different architecture** | Current approach uses a fixed action shape that can exceed later capacity; better to **re‑normalise the action’s depth curve each step** to respect time‑varying capacity. |

---

## Concrete Proposals

### 1. Adverse Selection – Coupled Flow / Price Innovations

**Goal**  
Make market‑order flow and the resulting price impact statistically dependent **within each Monte Carlo path**, so that adverse selection (e.g., large buy order → higher sell VWAP) is structurally present.

**Current behaviour**  
Flow models (`gamma`, `weibull`, etc.) sample `buy_vol` / `sell_vol` independently. Price models (`residual`, `conditional_mechanical_mixture`) sample independent residuals. No coupling.

**Proposed dependency structure**  
Use a **Gaussian copula** with a single latent factor per side. For each side `s ∈ {buy, sell}`:

```python
# Sample two correlated standard normals
rho = config.adverse_selection_correlation   # e.g. 0.6
z_flow = N(0,1)
z_price = rho * z_flow + sqrt(1 - rho**2) * N(0,1)

# Transform to flow volume (using inverse CDF of the flow model)
volume_s = flow_model.sample_given_shock(z_flow)   # new method

# Transform to price residual (using inverse CDF of the price model)
residual_s = price_model.sample_residual_given_shock(z_price)
buy_vwap = mechanical_buy_vwap + residual_buy
```

**File / function changes**  
- `mm/mpc/market_order_flow_torch.py`: add `sample_given_shock(z: Tensor) -> Tensor` to each model class. For Gamma / Weibull / hurdle models, implement by sampling from the unconditional distribution and then applying a Gaussian copula transform (or simpler: keep unconditional sample but correlate the underlying Gaussian innovations).  
- `mm/mpc/price_models.py`: add `sample_residual_given_shock(z: Tensor)` – similarly map a standard normal shock to the residual distribution (for Gaussian residual models this is direct; for mixture models we need to correlate the branch selection).  
- `mm/mpc/joint_optimizer.py` and `mm/mpc/mpc_tensor_batched.py`: in the rollout loops (e.g., `_rollout_paths`), replace independent sampling with the coupled version.

**Pseudocode (within rollout loop, for each path & step)**  

```python
# Pre-draw common shocks for all steps? Or per step. Per step is simpler.
z_buy_flow = torch.randn(n_paths, device=device)
z_buy_price = rho * z_buy_flow + (1 - rho**2)**0.5 * torch.randn(n_paths, device=device)
buy_vol = flow_model.sample_given_shock(z_buy_flow)  # shape (n_paths,)
buy_vwap_residual = price_model.sample_residual_given_shock(z_buy_price)
buy_vwap = mechanical_buy_vwap + buy_vwap_residual
# Similarly for sell side, optionally with independent or same correlation.
```

**Effort estimate**  
High – requires modifying four model classes, adding inverse‑CDF transforms for non‑Gaussian flows, and updating two rollout functions.  
**Break into sub‑steps** (≤ 200 lines each):  
1. Add `sample_given_shock` to Gaussian and Gamma models (simplest).  
2. Extend to Weibull and hurdle models (inverse transform sampling).  
3. Add shock‑conditional sampling to residual price models.  
4. Modify rollout loops to use coupled sampling (one function per optimizer).  
5. Add configuration parameter and unit tests for correlation = 0 (backward compatible).

**Prerequisites**  
- Refactor flow models to expose the underlying uniform or normal latent variable that drives sampling.  
- Decide on temporal correlation (shocks can be i.i.d. per step or persistent). Persistent requires stateful models.

---

### 2. Inventory Risk – Operational Capacity Term with Zone of Indifference

**Rationale**  
Even with CVaR on wealth changes, model misspecification or short horizons can allow extreme inventory accumulation. An explicit inventory penalty, flat near a target band and growing only outside, acts as a safe backstop.

**Proposed term** (added to the reward / objective)  

Let `Q_net = Q_local + Q_global` (or just local if no global).  
Define:

```
inv_penalty = 0                                      if |Q_net| <= Q_target_band
            = (|Q_net| - Q_target_band)^2           if Q_target_band < |Q_net| <= Q_max
            = (Q_max - Q_target_band)^2 + huge      if |Q_net| > Q_max   (hard constraint)
```

But a soft quadratic above the band is enough. The penalty is subtracted from the wealth change (or added as a negative term in the objective).

**Current objective** (in `mm/mpc/reward_tensor.py`) is `CE_beta(R)`. We modify to:

```
J = CE_beta(R) - λ_inv * inv_penalty(Q_net_terminal)   # terminal inventory
   - λ_inv_step * average_over_horizon(inv_penalty(Q_net_h))  # optional stepwise
```

**Files to change**  
- `mm/mpc/reward_tensor.py`: add new function `inventory_penalty(q, band, lambda)` and incorporate into `compute_risk_adjusted_return` or into the main objective.  
- `mm/mpc/mpc_tensor_batched.py` and `mm/mpc/joint_mpc.py`: pass the penalty terms through the objective builder.  
- Configuration: add `inventory_penalty_band` (in base units) and `inventory_penalty_weight`.

**Stopgap if coupling not done**  
Use the same term but with a relatively high weight (e.g., 0.01 per unit²) to discourage extreme inventory. This works without coupled flow‑price because it directly penalises inventory regardless of why it accumulated.

**Effort**  
Low – ~50 lines across two files. No new data dependencies.

**Prerequisite**  
None – can be implemented immediately.

---

### 3. Price‑Shading vs Size‑Capping – Hard Capacity Enforced per Rollout Step

**Problem**  
Current actions are absolute depth curves. Later steps may have less cash/inventory than when the action was planned, yet the action’s total depth is not reduced, leading to quotes that cannot be backed.

**Proposed change**  
In each rollout step, after computing the raw sigmoid depth curve `D_s(x)`, re‑scale the whole curve so that `D_s(spread_max)` does not exceed the step‑specific capacity `bid_max_h` (for bids) or `ask_max_h`.

**Implementation (in `soft_fill_batched` or a new function `rescale_depth_to_capacity`)**

```python
def rescale_depth_to_capacity(depth_curve, spread_max, max_capacity):
    # depth_curve is a function from spread to cumulative depth
    total_quoted = depth_curve(spread_max)
    if total_quoted > max_capacity:
        scale = max_capacity / total_quoted
        def scaled_curve(x):
            return depth_curve(x) * scale
        return scaled_curve
    return depth_curve
```

Call this for each side every step, using the updated `cash_h` and `inventory_h`. The action’s shape (relative depth distribution) is preserved but the level is capped.

**Files to change**  
- `mm/mpc/joint_optimizer.py` and `mm/mpc/mpc_tensor_batched.py`: inside the step loop, before calling `soft_fill_batched`, rescale the action’s cumulative depth function.  
- Because the action is represented as parameters, we need to evaluate `D_s(spread_max)` efficiently – pre‑compute grid points.

**Interaction with `quote_spread_max`**  
The spread cap is a hard limit on the x‑axis. The rescaling only affects the y‑axis (size). The two are orthogonal. Skew is still driven by the relative depth on bid vs ask and by the `x0` centers; with a wide enough spread max (e.g., 200 bps for crypto), skew can shift effective prices without changing total size.

**Effort**  
Medium – ~100 lines, plus changes to the action representation to allow efficient evaluation of total depth.

**Prerequisite**  
Current action decoder must expose a method `total_depth_at_spread_max(params)`. This is already computable from the mixture of sigmoids.

---

### 4. Book‑Quality Soft Objective Terms

**Goal**  
Encourage tight spread, sufficient depth, two‑sided presence, balanced depth, and a smooth depth profile. These are **soft preferences** (graded), not hard constraints.

**Proposed terms** (each multiplied by a tunable weight `w_<term>` and added to the reward per step)

| Term | Formula (per venue, per step) | Scaling |
|------|------------------------------|---------|
| **Spread tightness** | `- (spread_bid + spread_ask) / (2 * spread_max)` | Range [0,1], penalty linear in spread |
| **Total depth** | `min(1, (bid_depth + ask_depth) / depth_target)` | Bounded reward up to a target |
| **Two‑sided presence** | `1` if both sides have non‑zero quote, else `0` (or `-1` if missing) | Binary reward/penalty |
| **Balance** | `- |bid_depth - ask_depth| / (bid_depth + ask_depth + eps)` | Range [-1,0], zero when balanced |
| **Depth‑profile uniformity** | Negative of variance of marginal depths across spread levels (normalised by mean depth) | Penalises large gaps |

All terms are dimensionless and bounded. Combine with the PnL objective:

```
J_total = J_pnl + Σ w_k * BookQualityTerm_k
```

**Interaction with inventory zone of indifference**  
- When `|Q_net| <= Q_target_band`, inventory penalty is zero, so the book‑quality terms can dominate – encouraging a tight, deep, balanced book.  
- When `|Q_net|` grows beyond the band, the inventory penalty increases (quadratically) and, through the action’s capacity rescaling (proposal #3), will automatically reduce the allowed size on the risky side. The book‑quality terms still apply but have less room to act because the hard capacity cap shrinks.  
- No discontinuity: the inventory penalty is differentiable, and the capacity rescale is a continuous (multiplicative) operation.

**Files to change**  
- `mm/mpc/reward_tensor.py`: add `compute_book_quality_reward(state, action, step)` that returns a scalar.  
- `mm/mpc/mpc_tensor_batched.py` / `joint_optimizer.py`: include this reward in the stepwise reward accumulation.  
- Configuration: add weights (e.g., `book_quality.spread_weight = 0.1`).

**Effort**  
Medium – ~150 lines across two files. Requires computing depth‑profile uniformity from the action’s sigmoid mixture (can be done on a fixed grid).

**Prerequisites**  
- The `TradingState` or rollout state must provide the current quoted depths per spread level.  
- The current spread grid is already defined in `ControllerRuntimeConfig.orderbook_grid`.

---

## Order of Implementation (by impact per engineering hour)

1. **Inventory penalty (zone of indifference)** – immediate risk backstop, low effort, no dependencies.  
2. **Hard capacity rescaling per step** – fixes a structural flaw, medium effort, independent.  
3. **Book‑quality soft terms** – directly improves spreads/depth, medium effort, can be tuned after (1) and (2).  
4. **Adverse selection coupling** – most impactful for profitability, but high effort; should be done after the above are stable.

**Dependencies**:  
- (4) does not require (1)–(3), but (1) and (3) provide safety while (4) is being developed.  
- (2) makes (3) more effective because capacity is respected.  
- (1) can be added even before (2) as a stopgap.