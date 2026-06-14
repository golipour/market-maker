Based on the provided system documentation, here is a structured, incrementally implementable proposal to achieve a profitable, balanced, and less risky order book. The proposals are ordered by **expected impact per engineering hour**, with dependencies clearly noted.

---

### Part 1: Classification of Existing Mechanisms
Before introducing new terms, existing mechanisms must be reclassified to ensure the optimizer’s search space is well-defined.

| Mechanism | Classification | Justification |
| :--- | :---: | :--- |
| **Capacity Ratio** (`quote_capacity_ratio * cash / mid`) | **(a) Hard Structural Constraint** | Prevents insolvency. Must remain a hard cap in the decoder/parameterization. Violating this is a binary failure state, not a graded preference. |
| **Spread Caps** (`quote_spread_max`) | **(a) Hard Structural Constraint** | Defines the valid action space for the sigmoid parameterization (`x0` bounds). Keeping this hard prevents the optimizer from proposing economically nonsensical, infinitely wide spreads. |
| **Min/Max Size per Side** (`quote_size_min_per_side`, `quote_size_max_per_side`) | **(a) Hard Structural Constraint** | Enforces exchange limits and minimum viable liquidity. Should remain as box constraints on the raw optimizer variables or post-softmax normalization. |
| **Imbalance Skew Floors** (`skew_strength * max(-imbalance, 0)`) | **(b) Tunable Soft Objective** | Currently applied as a hard multiplier on `bid_min`/`ask_min`. This creates discontinuous, heuristic size reductions. It should be replaced by a smooth price-shading mechanism and a soft inventory penalty, allowing the optimizer to trade off skew vs. size organically. |
| **Fractional-Inventory Sizing** (implicit in current dynamic capacity) | **(c) Replace with Architectural Approach** | Heuristically shrinking size based on inventory is a blunt instrument. It should be replaced by **(1)** hard-capping size at available capacity in the decoder, and **(2)** using a smooth, zone-of-indifference inventory penalty in the objective to drive price-shading instead of arbitrary size truncation. |

---

### Part 2: Concrete Proposals (Ordered by Impact / Engineering Hour)

#### Proposal 1: Inventory Risk Separation (Zone-of-Indifference Backstop)
**Impact/Hour:** Very High (Foundational, low effort, immediate risk reduction)  
**Dependencies:** None (Prerequisite for Proposals 2 & 3)  
**Files to Change:** `mm/mpc/reward_tensor.py` (`CE_beta` computation), `mm/mpc/joint_mpc.py` (joint objective), `mm/runtime/config.py` (add inventory risk config).

**Concept:** Separate inventory risk into (1) wealth-variance (already handled implicitly by the entropic certainty equivalent `CE_beta` once adverse selection is coupled) and (2) an explicit, non-volatility-scaled operational-capacity term. This acts as a backstop against model misspecification and short-horizon blindness.

**Math / Pseudocode:**  
Define a smooth, zone-of-indifference penalty $C(Q_{net})$ that is exactly zero within a safe band $[Q_{min}, Q_{max}]$, and grows quadratically (bounded asymmetric growth) outside it:
```python
# In mm/mpc/reward_tensor.py or joint_mpc.py
def inventory_capacity_penalty(Q_net, q_min_safe, q_max_safe, lambda_inv, k_smooth=10.0):
    # softplus(x) = log(1 + exp(x)), smooth approximation of max(0, x)
    # Penalty is 0 when q_min_safe <= Q_net <= q_max_safe
    excess_long = torch.nn.functional.softplus(k_smooth * (Q_net - q_max_safe))
    excess_short = torch.nn.functional.softplus(k_smooth * (q_min_safe - Q_net))
    
    # Quadratic growth outside the band ensures strong repulsion from hard limits
    penalty = lambda_inv * (excess_long**2 + excess_short**2)
    return penalty

# Update objective J:
# J = w_L * CE_beta(R^L) + active_G * w_G * CE_beta(R^G) - inventory_capacity_penalty(Q_net, ...)
```
**Interim Stopgap:** If the full rollout coupling (Proposal 4) is delayed, add a simple quadratic penalty $-\lambda_{inv} Q_{net}^2$ directly to the per-path return $R_i$ before computing $CE_\beta$. This immediately forces mean-reversion without waiting for the copula implementation.  
**Effort:** 1–2 days.  
**Prerequisites:** Define `q_min_safe` and `q_max_safe` in `ControllerRuntimeConfig` based on venue collateral limits and historical fill rates.

---

#### Proposal 2: Price-Shading vs. Size-Capping Decoder Change
**Impact/Hour:** High (Directly improves book quality and prevents heuristic size collapse)  
**Dependencies:** Proposal 1  
**Files to Change:** `mm/mpc/mpc_tensor_batched.py` (`LParamConfig` or action decoding logic), `mm/mpc/joint_optimizer.py`.

**Concept:** Currently, inventory imbalance shrinks order sizes via `bid_min`/`ask_min` floors. Instead, enforce a **hard cap** on total quote size based on actual available inventory/cash in the decoder, and use the inventory imbalance **exclusively to shift the center ($x_0$)** of the sigmoid curves (price shading). This ensures the book remains deep but is priced away from the mid to discourage unwanted fills.

**Math / Pseudocode:**  
```python
# In the action decoder (e.g., inside LParamConfig or build_joint_target_book_from_sigmoid)
# 1. Hard-cap total depth at available capacity (Binary Feasibility)
Q_bid_raw = compute_total_depth_from_logits(u_bid_depth)
Q_bid_eff = torch.minimum(Q_bid_raw, max_cash / mid_price) # Never violates solvency

Q_ask_raw = compute_total_depth_from_logits(u_ask_depth)
Q_ask_eff = torch.minimum(Q_ask_raw, max_inventory)

# 2. Price-shading via x0 shift (Graded Preference)
# imbalance is normalized to [-1, 1]
# lambda_skew is a new tunable scalar weight
x0_bid_eff = base_x0_bid - lambda_skew * imbalance * quote_spread_max
x0_ask_eff = base_x0_ask + lambda_skew * imbalance * quote_spread_max

# Ensure x0_eff remains within [-quote_spread_max, quote_spread_max] via clamp
x0_bid_eff = torch.clamp(x0_bid_eff, -quote_spread_max, 0.0)
x0_ask_eff = torch.clamp(x0_ask_eff, 0.0, quote_spread_max)
```
**Interaction with `quote_spread_max`:** The current config uses `quote_spread_max` to bound `x0`. If `lambda_skew * quote_spread_max` is large, it could push the effective spread to the absolute limit, compressing the book. **Check:** Ensure `quote_spread_max` in the config is at least $2\times$ the expected maximum skew offset, leaving room for the base spread. If not, increase `quote_spread_max` or decouple the skew offset from `quote_spread_max` by introducing a dedicated `max_skew_bps` parameter.  
**Effort:** 2 days.  
**Prerequisites:** Audit current `quote_spread_max` values in `market_maker.runtime.example.json` to ensure they accommodate the new shading logic without hitting hard bounds prematurely.

---

#### Proposal 3: Book-Quality Terms as Soft Weighted Objectives
**Impact/Hour:** Medium-High (Directly optimizes for the stated "balanced, good depth" goal)  
**Dependencies:** Proposals 1 & 2  
**Files to Change:** `mm/mpc/reward_tensor.py`, `mm/mpc/joint_mpc.py`, `mm/runtime/config.py`.

**Concept:** Introduce dimensionless, soft penalty terms for book quality. To solve the **dimensional-scaling problem** (mixing quote-currency PnL with base-currency inventory or abstract geometry), all book-quality terms must be normalized to a unitless ratio or scaled by `mid_price * total_depth` to align with the wealth metric $W_h$.

**Math / Pseudocode:**  
```python
def book_quality_penalty(Q_bid, Q_ask, spread_bid, spread_ask, lambda_book, lambda_balance, lambda_tightness):
    eps = 1e-6
    
    # 1. Balance Penalty (Unitless): Penalize asymmetry in provided depth
    total_depth = Q_bid + Q_ask + eps
    balance_ratio = (Q_bid - Q_ask) / total_depth
    penalty_balance = lambda_balance * (balance_ratio ** 2)
    
    # 2. Two-Sided Presence Penalty (Unitless): Heavy penalty if either side is near zero
    # Use softplus to keep it differentiable
    penalty_onesided = lambda_book * (torch.nn.functional.softplus(-Q_bid / eps) + 
                                      torch.nn.functional.softplus(-Q_ask / eps))
    
    # 3. Spread Tightness (Scaled to quote currency): Encourage tighter spreads relative to mid
    # Normalized by mid_price to make it dimensionless
    avg_spread = (spread_bid + spread_ask) / 2.0
    penalty_tightness = lambda_tightness * (avg_spread / mid_price)
    
    return penalty_balance + penalty_onesided + penalty_tightness

# Total Objective:
# J = CE_beta(R) - inventory_capacity_penalty(Q_net) - book_quality_penalty(...)
```
**Interaction with Inventory Risk (Smooth Handoff):**  
Because `inventory_capacity_penalty` uses a **zone of indifference** (zero within $[Q_{min}, Q_{max}]$), when net inventory is near the target, the inventory penalty vanishes. In this regime, the `book_quality_penalty` dominates the gradient, forcing the optimizer to produce a deep, balanced, two-sided book.  
When inventory breaches the safe band, `inventory_capacity_penalty` grows quadratically. Because it is scaled by $\lambda_{inv}$ (which should be set higher than $\lambda_{book}$), it smoothly overpowers the book-quality terms, forcing the price-shading mechanism (Proposal 2) to skew the book and reduce fills, without creating a discontinuous "on/off" switch.  
**Effort:** 2–3 days.  
**Prerequisites:** Historical backtesting infrastructure to tune $\lambda_{book}$, $\lambda_{balance}$, and $\lambda_{inv}$ to find the Pareto frontier between PnL and book-quality metrics.

---

#### Proposal 4: Adverse Selection Coupling in Monte Carlo Rollout
**Impact/Hour:** Medium (Highest absolute impact on profitability/risk, but highest engineering effort)  
**Dependencies:** None (can be developed in parallel, but should be gated behind a config flag until validated).  
**Files to Change:** `mm/mpc/market_order_flow_torch.py` (flow models), `mm/mpc/mpc_tensor_batched.py` (`MonteCarloMPCTensorBatched.optimize_with_diagnostics`), `mm/mpc/reward_tensor.py`.

**Concept:** Currently, market flow ($V_{buy}, V_{sell}$) and price innovations ($\epsilon_{buy}, \epsilon_{sell}$) are sampled independently. This ignores adverse selection: large market flow often correlates with adverse price moves (e.g., informed trading pushes the mid away, picking off our quotes). We must couple them within each Monte Carlo path using a **Gaussian Copula** or **Conditional Mean Shift**.

**Math / Dependency Structure (Gaussian Copula Approach):**  
Instead of independent sampling, we sample a joint standard normal vector $(Z_v, Z_p) \sim \mathcal{N}(0, \Sigma)$, where $\Sigma = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$.  
1. Transform $Z_v$ to flow volume via the inverse CDF of the flow model (e.g., Gamma or Weibull): $V = F_{flow}^{-1}(\Phi(Z_v))$.  
2. Use $Z_p$ directly for the price residual: $\epsilon = \mu_{price} + \sigma_{price} Z_p$.  
This ensures that when $Z_v$ is high (large volume), $Z_p$ is correlated (e.g., $\rho > 0$ for buys pushing price up), structurally embedding adverse selection into the rollout.

**Pseudocode Implementation (Sub-steps to manage >200 line complexity):**  
*Sub-step 4.1: Extend Flow Models (`mm/mpc/market_order_flow_torch.py`)*  
Add a `sample_joint` method to the flow model classes that accepts a standard normal tensor $Z_v$ instead of sampling internally.
```python
# In TorchGammaFlowModel or similar
def sample_from_z(self, Z_v: torch.Tensor, n_samples: int, horizon: int) -> torch.Tensor:
    # Z_v is shape (n_paths, horizon)
    # Apply inverse CDF (PPF) of Gamma distribution using estimated shape/scale
    # PyTorch doesn't have native Gamma PPF, so use a rational approximation or 
    # switch to Conditional Mean Shift (Sub-step 4.1b) for easier autograd.
    pass
```
*Sub-step 4.1b (Pragmatic Alternative): Conditional Mean Shift*  
If Gamma PPF is too complex for autograd, use a conditional mean shift, which is differentiable and simpler:
```python
# Sample flow V_s independently as before
V_s = self.sample_flow(n_samples, horizon) 

# Compute standardized flow anomaly
V_mean = self.ewma_mean_s
V_std = torch.sqrt(self.ewma_var_s + 1e-6)
flow_z = (V_s - V_mean) / V_std

# Condition price residual mean on flow anomaly
# alpha is a learned or config-driven adverse selection coefficient
epsilon_mean = self.mu_price + self.alpha * flow_z
epsilon = epsilon_mean + self.sigma_price * torch.randn_like(V_s)
```

*Sub-step 4.2: Update Rollout (`mm/mpc/mpc_tensor_batched.py`)*  
Modify `soft_fill_batched()` or the rollout loop to consume the jointly sampled `(V_s, epsilon)` pairs per path, ensuring the `market_order_volume` and the `mid` update for that specific path reflect the correlated adverse move.

*Sub-step 4.3: Objective Adjustment (`mm/mpc/reward_tensor.py`)*  
Once coupled, the wealth variance $Var(W_h)$ will naturally increase during adverse selection events. The existing `CE_beta` (Entropic Certainty Equivalent) will automatically penalize this variance. No new explicit adverse selection penalty term is needed; the coupling makes the existing CVaR-like term effective.

**Effort:** 4–6 days (High complexity due to PyTorch autograd requirements and distribution math).  
**Prerequisites:**  
1. Historical data to estimate the correlation coefficient $\rho$ or conditional coefficient $\alpha$ between interval volume and VWAP residuals per symbol/venue.  
2. Unit tests in `tests/unit` to verify that the joint sampler produces the correct marginal distributions and the intended correlation structure without breaking the computational graph.

---

### Summary of Implementation Order
1. **Week 1:** Implement **Proposal 1** (Inventory Zone-of-Indifference). It is a low-risk, high-reward change to the reward tensor that immediately stabilizes the optimizer.
2. **Week 2:** Implement **Proposal 2** (Decoder Price-Shading). This requires modifying `LParamConfig` and verifying `quote_spread_max` limits. It pairs perfectly with Proposal 1.
3. **Week 3:** Implement **Proposal 3** (Book-Quality Soft Terms). With the decoder and inventory backstop in place, adding dimensionless book-quality penalties will smoothly guide the optimizer toward balanced, deep books without risking insolvency.
4. **Week 4–5:** Implement **Proposal 4** (Adverse Selection Coupling). Start with the pragmatic Conditional Mean Shift (Sub-step 4.1b) to validate the concept with minimal autograd friction, then iterate toward the full Gaussian Copula if higher fidelity is required.