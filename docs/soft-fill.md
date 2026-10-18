Here is the exact mathematical chain from the input parameters \(\theta\) to the objective function \(J(\theta)\), showing precisely **which function is discrete**, why its gradient is **zero almost everywhere**, and why optimization fails without a smooth relaxation.

---

### 1. The Pipeline: From Parameters to Objective

#### Step 1: Input Parameters \(\theta\)
The optimizer optimizes a parameter vector \(\theta = (L_j, k_j, x_{0,j})\). In `mm/mpc/state_tensor.py`, these parameters define a continuous logistic mixture curve for cumulative quoted depth as a function of spread \(s\):
\[
D(s; \theta) = \sum_{j=1}^M L_j \, \sigma\big(k_j (s - x_{0,j})\big)
\]
* \(x_{0,j}\) sets where the quote is placed (its spread / price offset).
* \(L_j\) sets the quote volume.

#### Step 2: Discretization onto the Spread Grid
The exchange orderbook lives on a discrete price grid \(s_0 < s_1 < \dots < s_K\). The marginal quote size placed at level \(i\) is:
\[
q_i(\theta) = D(s_i; \theta) - D(s_{i-1}; \theta)
\]
Notice that \(q_i(\theta)\) is smooth and continuously differentiable with respect to \(\theta\).

#### Step 3: Sampled Market Volume
At each step of the MPC rollout, the model draws a **specific numerical sample** of taker volume from the order flow distribution:
\[
V \sim P_{\text{flow}} \quad \longrightarrow \quad \text{e.g., } V = 3.5 \text{ USDT} \quad (\text{or } V = 0.0)
\]
*Crucial point:* In Monte Carlo rollouts, inside a specific trajectory, \(V\) is **not** a continuous probability density function \(p(V)\); it is a **fixed scalar number** for that scenario.

#### Step 4: The Fill Function (Where the Discreteness Lives)
Let \(b_i\) be the public orderbook depth ahead of us. The cumulative depth ahead of our quote at level \(i\) is:
\[
Q_{\text{ahead}} = \sum_{j=0}^{i-1} (b_j + q_j) + b_i
\]
How much of our quote \(q_i\) gets filled by the market order of volume \(V\)?

In a **discrete/exact matching engine**, the fill \(f_i\) is governed by an **indicator function (Heaviside step function \(\Theta\))**:
\[
f_i(V, \theta) = 
\begin{cases}
0 & \text{if } V \le Q_{\text{ahead}}(\theta) \quad &\text{(market order ran out of steam before reaching us)} \\
V - Q_{\text{ahead}}(\theta) & \text{if } Q_{\text{ahead}} < V < Q_{\text{ahead}} + q_i \quad &\text{(partially filled)} \\
q_i(\theta) & \text{if } V \ge Q_{\text{ahead}}(\theta) + q_i \quad &\text{(completely filled)}
\end{cases}
\]

Compactly, this is:
\[
f_i(V, \theta) = \max\Big(0,\; \min\big(q_i(\theta),\; V - Q_{\text{ahead}}(\theta)\big)\Big)
\]

#### Step 5: Cash, Inventory, and Terminal Objective \(J\)
The fills determine the portfolio change:
\[
\Delta \text{Cash} = \sum_i f_i \cdot P_i, \qquad \Delta \text{Inv} = \sum_i f_i
\]
Terminal wealth is:
\[
W(\theta) = \text{Cash}_0 + \Delta \text{Cash}(\theta) + \big(\text{Inv}_0 + \Delta \text{Inv}(\theta)\big) \cdot P_{\text{VWAP}}
\]
And the optimizer minimizes loss \(L(\theta) = -J(\theta)\), where \(J\) is expected utility or entropic return:
\[
J(\theta) = \frac{1}{S} \sum_{s=1}^S \left( W^{(s)}(\theta) - \frac{\beta}{2} \big(W^{(s)}(\theta)\big)^2 \right)
\]

---

### 2. Why the Gradient \(\frac{\partial J}{\partial \theta} = 0\) Almost Everywhere

By the chain rule of calculus:
\[
\frac{\partial J}{\partial \theta} = \sum_{\text{samples } s} \sum_{\text{levels } i} \frac{\partial J}{\partial f_i^{(s)}} \cdot \frac{\partial f_i^{(s)}}{\partial \theta}
\]
The derivative of the objective with respect to cash and inventory (\(\frac{\partial J}{\partial f_i}\)) is nonzero. **The bottleneck is entirely \(\frac{\partial f_i}{\partial \theta}\).**

Let's look at the derivative of the discrete fill function \(f_i\) with respect to the placement parameter \(\theta\) (for example, the spread offset \(x_0\)):

```
Fill f_i
  q_i |                    +----------------------- (Case 2: fully filled)
      |                   /
      |                  /  (Slope exists only on this razor-thin edge)
    0 +-----------------+ (Case 1: zero fill)
      0            Q_ahead       Q_ahead + q_i        Volume V
```

#### Case 1: \(V < Q_{\text{ahead}}(\theta)\) (Most common in illiquid books)
The market order volume was \(2.0\) USDT, but there were \(5.0\) USDT ahead of you in the queue.
* Your fill is \(f_i = 0\).
* Now, suppose PyTorch nudges your parameter \(\theta\) slightly: you tighten your spread by \(0.1\) bps.
* The queue ahead of you shrinks from \(5.0\) to \(4.99\) USDT.
* Does your fill change? **No, \(f_i\) is still exactly 0.**
\[
\frac{\partial f_i}{\partial \theta} = \lim_{\Delta \theta \to 0} \frac{0 - 0}{\Delta \theta} = \mathbf{0}
\]

#### Case 2: \(V > Q_{\text{ahead}} + q_i\) (Market order completely filled you)
The market order was \(50.0\) USDT, and your order was completely consumed.
* Your fill is \(f_i = q_i\).
* PyTorch nudges your spread offset parameter \(x_0\).
* The market order still easily blows through your quote: fill remains \(q_i\).
\[
\frac{\partial f_i}{\partial x_0} = \mathbf{0}
\]

#### The Result: The "Staircase" Loss Landscape
Plotted against the placement parameter \(x_0\), the objective function \(J(x_0)\) looks like a **staircase**:

```
Objective J
    ^
    |                       +----------- (Step 2: Filled in 2 scenarios)
    |                       |
    |           +-----------+            (Step 1: Filled in 1 scenario)
    |           |
    +-----------+                        (Step 0: Filled in 0 scenarios)
    +-----------------------------------> Placement Parameter x_0
      Slope = 0   Slope = ∞   Slope = 0
```

1. **Everywhere on the flat steps**: The gradient \(\nabla_\theta J = 0\). Gradient descent cannot take a single step because \(\theta \leftarrow \theta - \eta \cdot 0 = \theta\). The optimizer is frozen.
2. **At the step edges**: The derivative is undefined (Dirac delta \(\infty\)). If an optimizer hits it, the gradient explodes.

---

### 3. Why "Soft-Fill" Was Introduced

To allow gradient descent (Adam / L-BFGS) to work, you must replace the flat step function with a **smoothly increasing curve**:

```
Fill f_i
  q_i |                         . - - - - (Soft fill)
      |                     . '
      |                 . '
      |             . '
    0 + - - - - . '
      +----------------------------------->
```

On a smooth curve:
* If you improve your price by \(0.1\) bps, your fill probability increases from \(12.1\%\) to \(12.4\%\).
* \(\frac{\partial f_i}{\partial \theta} > 0\).
* PyTorch now has a non-zero gradient vector pointing in the direction of higher expected fill and higher spread capture!

### Summary

| Concept | What is it? | Why does it matter? |
| :--- | :--- | :--- |
| **Input Parameters \(\theta\)** | Continuous weights \((L, k, x_0)\) in PyTorch. | They can be differentiated smoothly. |
| **Sampled Volume \(V\)** | Fixed scalar draw per Monte Carlo scenario. | Because it's a fixed number per rollout, matching against it behaves like a deterministic threshold. |
| **The Discrete Fill Function** | The `min` / `max` / indicator function \(\mathbf{1}_{\{V > Q\}}\). | It is flat almost everywhere, giving **gradient = 0** and paralyzing gradient descent. |
| **The Soft Fill Relaxation** | Replacing the hard step with a smooth approximation. | Restores non-zero gradients so PyTorch can steer quotes. |
| **The Flaw in Current Code** | It made it soft by normalizing against the whole book. | It made it smooth, but broke the physical constraint that \(V=0 \implies \text{Fill}=0\). |