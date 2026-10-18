# Matching Engine and Flow Conservation in MPC

This document details the mathematical formulation of exchange order matching inside the Monte Carlo MPC simulator (`soft_fill_batched` in `mm/mpc/mpc_tensor_batched.py`), the root cause of the historical "ghost fills" bug, and the volume-conserving execution transition.

---

## 1. System Architecture: Flow Sampling vs. Execution Matching

The MPC rollout pipeline separates execution into two distinct stages:

1. **Order Flow Sampling (Statistical Estimation)**:
   The market flow model (e.g. Hurdle-Lognormal or Poisson) draws sample trade volumes $V^{(s)} \ge 0$ for each Monte Carlo trajectory $s \in \{1, \dots, S\}$ and each horizon step $h \in \{1, \dots, H\}$.
2. **Transition Matching Engine (Physical Execution)**:
   Given realized taker volume $V^{(s)}$ arriving at the exchange, the matching engine determines which resting quotes get consumed, updates the public orderbook state, and computes portfolio cash/inventory deltas:
   $$\text{ob\_bids\_next}, \text{ob\_asks\_next}, \Delta \text{Cash}, \Delta \text{Inv} = \text{soft\_fill\_batched}(\dots)$$

---

## 2. The Historical "Ghost Fills" Bug

### The Old Formulation
Previously, consumption at each price level was calculated by evaluating a continuous "reach" variable:
```python
# DEPRECATED / BUGGY FORMULATION
comb_cum = torch.cumsum(comb_level, dim=2)
reach_raw = vol - comb_cum
neg_scale = torch.clamp((-reach_raw).max(dim=2, keepdim=True).values, min=eps)
reach = reach_raw / neg_scale
fill_frac = _soft_fill_fraction(reach, soft_temp, activation)
consumed_level = comb_level * fill_frac
```

### The Root Defect
When incoming market volume was identically zero ($V = 0$):
1. `reach_raw` $= - \text{comb\_cum} < 0$.
2. `neg_scale` was normalized by the **maximum deficit across the entire orderbook**, which equals the total depth of the book (e.g. $500$ USDT).
3. At the top of the book, depth is small (e.g. $2$ USDT). The normalized reach was:
   $$\text{reach} = \frac{-2.0}{500.0} = -0.004$$
4. With `soft_temp = 0.1`, `reach / soft_temp = -0.04`.
5. PyTorch's `F.hardsigmoid(-0.04) = (-0.04 + 3) / 6 = 0.493` ($49.3\%$).

Even though $V = 0.00$ USDT, the simulator manufactured a $49.3\%$ fill on top-level quotes out of thin air on every single rollout step. This led the optimizer to hallucinate infinite passive liquidity, accumulate unhedged inventory on thin venues like Bitwana, and suffer adverse selection when resting quotes failed to fill in live trading.

---

## 3. The Flow-Conserving Matching Formulation

Any valid matching simulator must obey the **physical law of volume conservation**:
$$\sum_{i=0}^{K-1} c_i \le \min\left(V,\; \sum_{i=0}^{K-1} d_i\right)$$
where $d_i = b_i + u_i$ is combined depth (public orderbook $b_i$ plus our quote $u_i$) at grid level $i$.

### A. Exact Prefix Consumption
Let cumulative combined depth up to level $i$ be:
$$C_i = \sum_{j=0}^i d_j, \qquad C_{-1} = 0$$

The cumulative volume consumed up to level $i$ is:
$$\text{cons\_cum}_i = \min(V, C_i)$$

The marginal volume consumed at level $i$ is:
$$c_i = \max\big(0,\; \text{cons\_cum}_i - \text{cons\_cum}_{i-1}\big)$$

#### Mathematical Properties
1. **Zero Volume Invariant**: If $V = 0$, $\text{cons\_cum}_i = \min(0, C_i) = 0 \implies c_i = 0$ for all levels.
2. **Capacity Invariant**: $c_i \le C_i - C_{i-1} = d_i$ at every level.
3. **Conservation Invariant**: $\sum_{i=0}^{K-1} c_i = \min(V, C_{K-1}) \le V$.
4. **Price Priority**: Volume is completely consumed at level $i$ before any volume reaches level $i+1$.

---

## 4. Queue Allocation Models

Once aggregate level consumption $c_i$ is determined, it is partitioned between public orderbook depletion $r_i$ and our maker fill $f_i$:
$$c_i = r_i + f_i, \qquad 0 \le r_i \le b_i, \qquad 0 \le f_i \le u_i$$

### 1. `back_of_queue` (FIFO Priority at Level)
We join the book behind the existing public depth $b_i$ at that price level. The market order must completely consume the public depth at level $i$ before filling our quote:
$$r_i = \min(c_i, b_i)$$
$$f_i = c_i - r_i = \max(0,\; c_i - b_i)$$

- If $c_i \le b_i$: Public liquidity absorbs the entire trade; our quote fill $f_i = 0$.
- If $c_i > b_i$: Public liquidity is exhausted ($r_i = b_i$); the remaining volume fills our quote ($f_i = c_i - b_i \le u_i$).

### 2. `pro_rata` (Proportional Sharing)
Volume is shared proportionally to displayed capacity:
$$f_i = c_i \frac{u_i}{b_i + u_i}$$
$$r_i = c_i - f_i = c_i \frac{b_i}{b_i + u_i}$$

Both models guarantee $r_i + f_i = c_i$, ensuring exact conservation and consistency with the post-trade public book state:
$$b_i^{\text{next}} = b_i - r_i = \max(0,\; b_i - r_i)$$

---

## 5. Differentiability and Autograd Gradients

A common misconception is that exact matching produces zero gradients everywhere. 

In this codebase, our quoted depth $u_i(\theta)$ is a continuous, smooth function of the logistic mixture parameters $\theta = (L_j, k_j, x_{0,j})$ decoded via `mm/mpc/state_tensor.py`:
$$u_i(\theta) = D(s_i; \theta) - D(s_{i-1}; \theta)$$

When backpropagating from terminal wealth $J$ through the matching engine:
$$\frac{\partial f_i}{\partial \theta} = 
\begin{cases}
0 & \text{if unreached } (c_i \le b_i) \\
\frac{\partial u_i}{\partial \theta} & \text{if fully filled } (c_i \ge b_i + u_i) \\
\frac{\partial (c_i - b_i)}{\partial \theta} & \text{if partially filled}
\end{cases}$$

Under `pro_rata`, non-zero smooth gradients flow whenever $c_i > 0$:
$$\frac{\partial f_i}{\partial u_i} = c_i \frac{b_i}{(b_i + u_i)^2} > 0$$

`torch.autograd.gradcheck` confirms full analytical agreement between finite differences and backward autograd across both queue models.

---

## 6. Deprecations and Compilation Performance

- **Deprecated Parameters**: `soft_fill_temp` and `soft_fill_activation` are deprecated. Exact volume conservation eliminates the need for heuristic smoothing temperatures or sigmoid activations.
- **Compiler Compatibility**: The kernel uses only elementwise primitives and prefix pads (`cumsum`, `minimum`, `pad`, `relu`), eliminating the `.max()` reduction barriers of the old code.
- **Benchmarks**:
  - Eager forward pass: ~7x faster.
  - Backward autograd pass: ~2.3x faster.
  - Decision latency under `torch.compile` (Inductor): **~10.0 ms per decide** (0 recompiles).
