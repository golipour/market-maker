Balancing risk-adjusted returns with structural book quality in a differentiable MPC framework is an excellent, albeit deeply complex, optimization challenge. Relying strictly on constraints often leads to brittle corner-case behavior, while relying purely on PnL objectives can lead to over-fitted, sparse order books.

Based on the provided codebase documentation, here is a concrete proposal to migrate the system from heuristic capacity bounds to a mathematically grounded, multi-objective formulation.

---

### 1. Classification of Existing Constraints

Before adding new terms, we must clean up the existing action parameterizations to ensure the optimizer isn't fighting hardcoded heuristics.

* **Capacity Ratio (Scaling max size by cash/inventory):** *(a) Hard structural constraint.* You fundamentally cannot quote sizes exceeding your wallet balance. This must remain a strict bound in the tensor parameterization (`LParamConfig`) to prevent invalid rollouts.
* **Spread Caps (`quote_spread_max`):** *(a) Hard structural constraint.* Unbounded spreads result in mathematically valid but practically useless quotes. The exchange rules and sanity constraints require a ceiling.
* **Min/Max Size per Side (`quote_size_min/max`):** *(c) Replace.* While exchange minimums are a hard constraint, the *maximum* size should emerge dynamically from the optimizer balancing risk versus reward, rather than being a static config variable.
* **Imbalance Skew Floors (`bid_floor = ... * (1 + skew * max(-imb, 0))`):** *(c) Replace.* This is a deterministic heuristic overriding the MPC's probabilistic insight. Skew should emerge naturally from the explicit inventory penalty and side-aware liquidation marking.

---

### 2. Adverse Selection: Coupling Flow and Price Samplers

Currently, market order flow (`V_s`) and price innovations (`epsilon`) are sampled independently. This assumes noise trading, meaning the optimizer overestimates profits because it doesn't realize that large fill volumes usually occur right before the price moves against the bot.

**Proposal:** Introduce a Gaussian Copula to couple the residuals in the rollout phase. If a large buy volume is sampled, a positive price innovation should be heavily favored.

**Location:** `mm/mpc/joint_optimizer.py` (in the rollout loop) and `mm/mpc/market_order_flow_torch.py`.

**Math / Pseudocode:**
We link the samplers using a correlation parameter $\rho$. We draw standard normal latent variables for both flow ($Z_{\text{flow}}$) and price ($Z_{\text{price}}$):

$$Z_{\text{flow}} \sim \mathcal{N}(0, I)$$

$$Z_{\text{price}} \sim \mathcal{N}(0, I)$$

We correlate them:


$$Z_{\text{correlated\_price}} = \rho Z_{\text{flow}} + \sqrt{1 - \rho^2} Z_{\text{price}}$$

For the flow model (e.g., Weibull), we use the inverse CDF to map the latent variable to volume:


$$V_{\text{buy}} = F_{\text{Weibull}}^{-1}(\Phi(Z_{\text{flow}}))$$

For the `TorchResidualPriceModel`, we apply the correlated innovation:


$$\epsilon_{\text{buy\_vwap}} = \mu_{\text{buy}} + \sigma_{\text{buy}} Z_{\text{correlated\_price}}$$

**Implementation Sub-steps (High Effort):**

1. Add a `rho` (correlation) parameter to the global controller config.
2. Refactor `market_order_flow_torch.py` to accept a pre-sampled $Z$ tensor in its `.sample()` method.
3. Refactor `TorchResidualPriceModel` to accept a pre-sampled $Z$ tensor.
4. In `joint_optimizer.py`, generate $Z_{\text{flow}}$ and $Z_{\text{price}}$, apply the correlation matrix, and pass the explicit $Z$ tensors to the respective models during the horizon rollout.

---

### 3. Inventory Risk: The "Zone of Indifference" Backstop

Wealth-variance risk (CVaR) handles the financial risk of holding inventory, but short-horizon MPC often suffers from "horizon blindness"—it doesn't see the disaster looming at step $H+1$.

**Proposal:** Add an explicit, non-financial operational penalty for straying too close to inventory limits. This should be a "zone of indifference" shape: flat near the target, but growing exponentially near limits to force extreme price-shading.

**Location:** `mm/mpc/reward_tensor.py` (single venue) and `mm/mpc/joint_mpc.py` (joint venue).

**Math / Pseudocode:**
Let $Q$ be net inventory, $T$ be the target inventory (usually 0), and $W$ be the allowed variance width.


$$P(Q) = \lambda \max(0, |Q - T| - W)^2$$

```python
# Pseudocode for reward_tensor.py or joint_mpc.py
def compute_inventory_penalty(inventory, target=0.0, width=10.0, lambda_penalty=0.5):
    deviation = torch.abs(inventory - target)
    excess = torch.clamp(deviation - width, min=0.0)
    return lambda_penalty * (excess ** 2)

# Subtracted from the objective:
loss = - (CE_beta(R) - compute_inventory_penalty(Q_final))

```

*Interim Stopgap:* If the Copula coupling (Item 2) is not built yet, you must set $\lambda$ exceptionally high to manually protect against adverse selection.

---

### 4. Price-Shading vs. Size-Capping

Currently, dynamic capacity constraints dynamically slice off the total quantity $Q_{s,\text{total}}$ based on available cash. This forces size-capping rather than allowing the optimizer to shift the quote further out (price-shading).

**Proposal:** Let the decoder hard-cap $Q_{s,\text{total}} \le \text{actual\_wallet\_balance}$, but remove heuristic imbalance skews. The zone-of-indifference penalty (Item 3) will naturally induce the optimizer to push the $x_0$ centers of the sigmoid curves away from the mid price to avoid fills when inventory is dangerous.

**Location:** `mm/mpc/mpc_tensor_batched.py` inside `LParamConfig`.

**Crucial Interaction with `quote_spread_max`:**
If `quote_spread_max` is set too tight (e.g., 10 bps), the optimizer cannot physically shade the price far enough away to avoid fills when $P(Q)$ is screaming. Ensure `quote_spread_max` is wide enough (e.g., 50-100 bps) so the optimizer has the geometric "room" to move quotes back, creating a one-sided book naturally.

---

### 5. Book-Quality Terms

To prevent the optimizer from making a mathematically profitable but structurally awful book (e.g., one tiny order at the mid, and zero depth behind it), we add graded preferences.

**Proposal:** Introduce dimensionless soft objective terms.

**Location:** `mm/mpc/joint_mpc.py`.

**Math / Pseudocode:**
Since PnL is in quote-currency and inventory penalty is in base-currency squared, we must normalize the book quality to be dimensionless:


$$\text{Spread Score} = 1 - \frac{\sum x_{0,\text{bid}} + \sum x_{0,\text{ask}}}{2 \cdot \text{quote\_spread\_max}}$$

$$\text{Depth Score} = \frac{Q_{\text{bid,total}} + Q_{\text{ask,total}}}{2 \cdot \text{quote\_size\_max}}$$

```python
# Joint Objective Construction
J = w_L * CE_beta(R_L) + ...
J += w_spread * spread_score + w_depth * depth_score - P(Q)

```

**Smooth Handoff:**
Because $P(Q) = 0$ inside the indifference zone $W$, the `w_spread` and `w_depth` terms will entirely dominate behavior when inventory is healthy, naturally tightening the spread and increasing depth. The moment inventory breaches $W$, the quadratic $P(Q)$ rapidly scales up, overriding the linear book-quality terms and forcing the quotes to widen out (shading).

---

### 6. Suggested Execution Roadmap (By ROI)

1. **Price-Shading & Explicit Inventory Risk (High Impact, Low Effort - 4-8 hours):**
* Remove heuristic skew floors.
* Implement the quadratic $P(Q)$ indifference zone in `joint_mpc.py`.
* Verify `quote_spread_max` limits allow enough room for shading.


2. **Book-Quality Soft Terms (Medium Impact, Low Effort - 4-6 hours):**
* Add dimensionless spread and depth scores to the objective function.
* Expose scalar weights (`w_spread`, `w_depth`) in `ControllerRuntimeConfig`.


3. **Adverse Selection Copula Coupling (High Impact, High Effort - 20-30 hours):**
* Refactor samplers in `market_order_flow_torch.py` and `TorchResidualPriceModel`.
* Implement latent variable correlation in the `joint_optimizer.py` horizon loop.



---

Given how `LParamConfig` currently maps raw tensor outputs to sigmoid components via softmax, do you want to explore adjusting the `quote_spread_min_depth` constraint to ensure the soft book-quality terms don't inadvertently spread the liquidity too thin across the tail components?