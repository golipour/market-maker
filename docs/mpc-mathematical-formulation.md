# Mathematical Formulation of the MPC Objective

This document provides a formal mathematical representation of the Model Predictive Control (MPC) optimization problem implemented in the market-maker. It details the objective function $J$, its decomposition into profitability and penalty terms, and the bi-level optimization method used to solve it.

## 1. Problem Definition

At each decision time $t$, the controller solves a **finite-horizon stochastic program** over differentiable quote parameters. The problem incorporates soft constraints for book quality and quoting capacity via penalties, with one adaptive Lagrange multiplier (for viability).

Formally, the controller approximates the following bi-level problem:

**Outer Problem (Dual Update):**
Update the dual state $\theta$ based on the availability diagnostics from the previous step.

**Inner Problem (Primal Solve):**
Maximize the scalar objective $J(u; \xi_t, \theta_t)$ over the action parameters $u$, given the exogenous state $\xi_t$ and fixed dual variable $\theta_t$:
$$ u^* = \arg\max_{u \in \mathcal{U}(\xi_t)} J(u; \xi_t, \theta_t) $$

---

## 2. State and Action Spaces

### 2.1 Exogenous State ($\xi_t$)
The state $\xi_t$ encapsulates the market and wallet conditions at time $t$:
*   $p_t$: Mid price (quote anchor).
*   $\mathcal{B}_t$: Market orderbook in relative spread coordinates.
*   $b_t$: Total base balance (e.g., USDT).
*   $q_t$: Total quote balance (e.g., IRT).
*   $V_t^{\text{buy}}, V_t^{\text{sell}}$: Market flow observations for stochastic models.

### 2.2 Decision Variables ($u$)
The optimizer does not directly output limit orders. Instead, it outputs a parameter vector $u$ that defines continuous cumulative depth curves using a **mixture of sigmoids**.

For a horizon $H$ and $K$ components per side, the action $u$ is stacked across time steps:
$$ u = [u^{(0)}, u^{(1)}, \dots, u^{(H-1)}] $$

Each slice $u^{(h)}$ contains parameters for the bid and ask sides. For a given side $s \in \{\text{bid}, \text{ask}\}$, the parameters decode to mixture weights $L_i$, steepness $k_i$, and centers $x_{0,i}$ for $i = 0, \dots, K-1$. 

The continuous cumulative depth at a given spread is defined as:
$$ D_s(\text{spread}) = \sum_{i=0}^{K-1} L_i \cdot \sigma\big(k_i \cdot (\text{spread} - x_{0,i})\big) $$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid.
The executable (effective) depth is $\tilde{D}_s(\text{spread}) = \max(0, D_s(\text{spread}) - D_s(0))$.

### 2.3 Feasible Set $\mathcal{U}(\xi_t)$
Hard constraints bounding the feasible set $\mathcal{U}(\xi_t)$ are enforced via capacity clamping and box constraints on $u$, ensuring that simulated depth never exceeds available wallet balances:
$$ Q_{\max}^{\text{bid}} = \min(Q_{\text{cap}}, q_t / p_t), \quad Q_{\max}^{\text{ask}} = \min(Q_{\text{cap}}, b_t) $$

---

## 3. Objective Function Decomposition

The scalar objective $J$ is a Lagrangian relaxation that maximizes risk-adjusted returns while penalizing poor book structure (Validity) and capacity over-extension (Viability).

$$ J(u; \xi_t, \theta_t) = CE_{\text{eff}}(u; \xi_t) - \mu \big[B_{\min} - B(u^{(0)}; \xi_t)\big]_+ - \lambda(\theta_t) \mathcal{S}(u; \xi_t) $$

Where $[x]_+ = \max(0, x)$ is the hinge function.

### 3.1 Term 1: Risk-Adjusted Return ($CE_{\text{eff}}$)
This term represents the profitability objective. For each Monte Carlo path $i \in \{1 \dots N\}$, the rollout operator simulates $H$ steps, yielding a path return $R_i$:
$$ R_i = \frac{\sum_{h=0}^{H-1} \Delta W_{i,h}}{W_0} $$
where $W$ is the mark-to-market wealth. 

The optimizer maximizes the **Entropic Certainty Equivalent (CE)** of these returns, governed by the risk-aversion parameter $\beta$:
$$ CE = \min_i(R_i) - \frac{1}{\beta} \log \sum_{i=1}^N \exp\Big(-\beta \cdot \big(R_i - \min_j(R_j)\big)\Big) + \frac{1}{\beta} \ln(N) $$

Optionally, $CE$ is normalized by an Exponentially Weighted Moving Average (EWMA) scale factor $s$ to form $CE_{\text{eff}} = CE / s$.

### 3.2 Term 2: Validity Penalty (Book Quality)
This term penalizes the structural shape of the immediate quote (at horizon step $h=0$) if it falls below a minimum quality threshold $B_{\min}$. 

*   **$B(u^{(0)}; \xi_t) \in [0, 1]$**: The book quality score, composed of a weighted average of:
    *   **Tightness**: Closeness of the first centers to the mid.
    *   **Presence**: Depth near the touch.
    *   **Balance**: Symmetry of bid and ask quote notionals.
    *   **Uniformity**: Smoothness of the marginal depth distribution.
*   **$\mu$**: A fixed, configured penalty scale.

$$ \text{Validity Penalty} = -\mu \cdot \max\big(0, B_{\min} - B(u^{(0)}; \xi_t)\big) $$

### 3.3 Term 3: Viability Penalty (Availability Shortfall)
This term enforces soft two-sided quoting capacity. It penalizes actions that, after the *first* simulated fill, leave the wallet with insufficient margins to continue quoting on both sides.

1. **Quote-Notional Margins**: Evaluated after the step $h=1$ fill.
   $$ m_{\text{bid}}^{(i)} = q_1^{(i)} - q_{\min}, \quad m_{\text{ask}}^{(i)} = b_1^{(i)} \cdot p_1^{(i)} - q_{\min} $$
   where $q_{\min}$ is the minimum viable operational quote size.
2. **Soft Availability ($\tilde{A}$)**:
   $$ a_s^{(i)} = \sigma\left(\frac{m_s^{(i)}}{\delta}\right) \quad \text{for } s \in \{\text{bid}, \text{ask}\} $$
   $$ \tilde{A}_1^{(i)} = a_{\text{bid}}^{(i)} \cdot a_{\text{ask}}^{(i)} $$
   where $\delta$ is the margin buffer scale.
3. **Mean Shortfall ($\mathcal{S}$)**:
   $$ \mathcal{S}(u; \xi_t) = \frac{1}{N} \sum_{i=1}^N \max\big(0, A^* - \tilde{A}_1^{(i)}\big) $$
   where $A^*$ is the availability target (e.g., $0.99$).

$$ \text{Viability Penalty} = -\lambda(\theta_t) \cdot \mathcal{S}(u; \xi_t) $$

---

## 4. Optimization Method

The controller solves this system using a bi-level strategy.

### 4.1 Inner Loop: Adam Optimizer (Primal Solve)
For a fixed dual variable $\theta_t$, the optimizer runs a configured number of Adam epochs ($E$) to find $u^*$:
$$ u^* \approx \arg\min_u \big( -J(u; \xi_t, \theta_t) \big) $$

**Rollout and Soft Fills:**
Gradients are computed via backpropagation through a Monte Carlo rollout. Since exact limit-order queue matches are discrete and non-differentiable, the rollout uses a **soft-fill model**:
$$ \text{fill}_s = \min\big(\tilde{D}_s(\text{spread}), V_s\big) $$
where $V_s$ is the sampled market flow, smoothed by a temperature parameter to maintain differentiability.

**Warm Start:**
The initial guess for $u$ is blended between a default prior and the optimal $u$ found in the *previous* decision step, accelerating Adam convergence.

### 4.2 Outer Loop: Dual Update
The viability multiplier $\lambda = \exp(\theta)$ is updated *once per decision frame* using subgradient ascent on the availability gap.

After the inner problem completes, the realized mean availability $\bar{A}_1 = \frac{1}{N} \sum_i \tilde{A}_1^{(i)}$ is computed from the optimal paths. The dual state updates via:
$$ \text{gap}_t = A^* - \bar{A}_1 $$
$$ \theta_{t+1} = \theta_t + \eta \cdot \text{gap}_t $$
$$ \lambda_{t+1} = \exp(\theta_{t+1}) $$
where $\eta$ is the dual learning rate. 

If availability is below target ($\bar{A}_1 < A^*$), $\theta$ increases, making the viability penalty strictly harsher in the next decision step, naturally forcing the optimizer to quote more conservatively to preserve inventory/cash.