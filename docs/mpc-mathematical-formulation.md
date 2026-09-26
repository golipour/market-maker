# Mathematical Formulation of the MPC Objective

This document is the compact mathematical statement of the market-maker MPC
as implemented in `mm/mpc/`. It describes the constrained stochastic program
the controller approximates, the scalar Lagrangian $J$ that Adam actually
maximizes, and the outer dual update on the viability multiplier $\lambda$.

Implementation detail, diagnostics, and config commentary live in
[`mpc-optimization-objective.md`](mpc-optimization-objective.md). The
stochastic flow and price models sampled inside $\Phi$ are stated in
[`mpc-price-and-flow-models.md`](mpc-price-and-flow-models.md). The shared
relative spread grid, soft-fill depth, and ladder discretization are stated
in [`mpc-spread-grid-and-depth.md`](mpc-spread-grid-and-depth.md). Source of
truth for the formulas below is the code, not older research notes.

Primary references:

- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched`
- `mm.mpc.joint_optimizer.optimize_joint_with_diagnostics`
- `mm.mpc.objective_terms` (validity, viability, CE normalization)
- `mm.mpc.reward_tensor.objective_entropic_return`
- `mm.mpc.l_param` (sigmoid-mixture decode)

---

## 1. Decision problem

At each controller decision $t$, the optimizer chooses a differentiable
quote-parameter vector $u$ given exogenous state $\xi_t$. The **design
intent** is a finite-horizon stochastic program:

$$
\begin{aligned}
\max_{u \in \mathcal{U}(\xi_t)} \quad
  & CE_{\mathrm{eff}}(u;\xi_t) \\
\text{s.t.} \quad
  & \mathbb{E}\bigl[S_1(u;\xi_t)\bigr] \le \varepsilon, \\
  & B\bigl(u^{(0)};\xi_t\bigr) \ge B_{\min}
    \quad\text{(optional)}.
\end{aligned}
$$

- $CE_{\mathrm{eff}}$ is an entropic certainty equivalent of simulated
  horizon return (profitability, including price and fill risk).
- $S_1$ is a next-step **quadratic reserve deficit** on total wallet
  balances (two-sided quoting capacity / survival).
- $B$ is an $h{=}0$ **book-quality** score (presence, tightness, balance,
  uniformity). Validity is a presence/shape gate, not a profit term.

Inequalities are **not** hard constraints in the solver. They enter $J$ as
one-sided penalties. Viability uses an **adaptive Lagrange multiplier**
$\lambda_t$, held fixed during Adam and updated once after the decision.

The inner problem is non-convex. Adam returns a **local** best-epoch
solution, not a global optimum.

---

## 2. State and action

### 2.1 Exogenous state $\xi_t$

| Symbol | Code | Meaning |
| --- | --- | --- |
| $p_t$ | `mid` | Quote-per-base anchor |
| $\mathcal{B}_t$ | `ob_rel` | Market book on the relative-spread grid |
| $b_t$ | `inventory` | Total base (free + locked) |
| $q_t$ | `cash` | Total quote (free + locked) |
| $\omega_t$ | `buy_vol`, `sell_vol` | Flow observations for MC models |

Viability and capacity use **total** balances. Locked funds are not treated
as gone. Free-only wallets, tick/lot rules, and exchange min-notional are
outside this formulation.

With `fixed_mc_samples = true`, the same Monte Carlo draws are reused across
all Adam epochs of one `optimize()` call.

### 2.2 Decision variable $u$

The optimizer does not choose individual limit orders. It chooses parameters
of **cumulative depth curves** on a fixed spread grid.

Let $H =$ `horizon` and $K =$ `n_components`. One venue slice is
$u^{(h)} \in \mathbb{R}^{6K+3}$:

$$
u = \bigl[u^{(0)},\, u^{(1)},\, \ldots,\, u^{(H-1)}\bigr] \in \mathbb{R}^{H(6K+3)}.
$$

For side $s \in \{\mathrm{bid},\mathrm{ask}\}$ the slice materializes to
mixture weights $L_i$, steepness $k_i$, and centers $x_{0,i}$
($i=0,\ldots,K-1$), plus a size logit at $S_{\max}$ and an optional
reservation-skew logit.

**Validity** uses only $u^{(0)}$. **CE** and **viability** use the full
horizon stack through the rollout.

Joint / liquidity-sourcing mode concatenates local and global venue blocks
with separate $K$.

### 2.3 Decode: sigmoid mixture

On side $s$,

$$
D_s(\delta) = \sum_{i=0}^{K-1} L_i\,\sigma\bigl(k_i(\delta - x_{0,i})\bigr),
\qquad
\tilde{D}_s(\delta) = \max\bigl(0,\, D_s(\delta) - D_s(0)\bigr),
$$

where $\sigma(z) = (1+e^{-z})^{-1}$ and $\delta$ is spread from mid in
quote currency. $\tilde{D}_s$ is executable cumulative depth.

Total size at the outer grid edge is a logit between a configured floor and
a **wallet-clamped** cap:

$$
Q_s^{\max} = \min\bigl(Q_{\mathrm{cap}},\; q_t/p_t\bigr)\ \text{(bid)},
\qquad
Q_s^{\max} = \min\bigl(Q_{\mathrm{cap}},\; b_t\bigr)\ \text{(ask)}.
$$

Capacity floors at decode are zero: one-sided quoting is feasible. Optional
`book_quality.hard_structure` further constrains first-touch centers.

The spread grid is $\{\,j\Delta : j=1,\ldots,\lceil S_{\max}/\Delta\rceil\,\}$
with $S_{\max}=$ `spread_max`, $\Delta=$ `spread_step`. Full geometry,
public-book bucketing, soft fill, and execution ladder: see
[`mpc-spread-grid-and-depth.md`](mpc-spread-grid-and-depth.md).

### 2.4 Hard feasible set $\mathcal{U}(\xi_t)$

Enforced by decode / clamping, **not** by terms in $J$:

- box bounds on raw parameters,
- $\tilde{D}_s \ge 0$,
- wallet capacity $Q_s^{\max}$ as above.

Tick, lot, min-notional, net-delta, and allocation caps are applied
**after** MPC, on the target book.

---

## 3. Stochastic rollout $\Phi$

Given $u$ and $\xi_t$, the closure draws $N=$ `n_samples` paths
$\varepsilon_i$ from the configured price and flow models and simulates
$H$ steps:

$$
\bigl(b_{i,h},\, q_{i,h},\, p_{i,h},\, W_{i,h}\bigr)_{h=0}^{H}
= \Phi(u,\,\xi_t,\,\varepsilon_i).
$$

Index $h=0$ is the pre-action wallet. Index $h=1$ is the wallet after
the first simulated fill (viability uses this step only).

Per step, `soft_fill_batched` matches sampled market flow against the
combined (market + quoted) book with a temperature and queue model
(`soft_fill_temp`, `soft_fill_activation`, `soft_fill_queue_model`), applies
maker fees, updates the public book, and marks wealth with inventory-aware
VWAP:

$$
W_{i,h} = q_{i,h} + b_{i,h}\cdot \mathrm{mark}_{i,h}.
$$

Wealth change on a step is the accounting identity

$$
\Delta W = \Delta q + b\,\Delta p + \Delta b\, p + \Delta b\,\Delta p.
$$

---

## 4. Implemented objective $J$

The solver maximizes the Lagrangian relaxation

$$
J(u;\xi_t,\lambda_t)
=
CE_{\mathrm{eff}}(u;\xi_t)
- \mu\bigl[B_{\min} - B(u^{(0)};\xi_t)\bigr]_+
- \lambda_t\,\mathcal{S}(u;\xi_t),
$$

where $[x]_+ = \max(0,x)$.

| Term | When | Adaptive? |
| --- | --- | --- |
| $CE_{\mathrm{eff}}$ | full-horizon path returns | optional EWMA scale on CE only |
| Validity $\mu[B_{\min}-B]_+$ | decoded book at $h=0$ | no ($\mu$ is config) |
| Viability $\lambda\mathcal{S}$ | next-step wallet $h=1$ | yes ($\lambda$ between frames) |

If `validity.penalty_scale = 0` (or the block is absent), the validity term
is omitted. If `viability` is absent or $\lambda=0$, viability is omitted.

There is **no** linear inventory penalty $W_{\mathrm{inv}}\cdot b$ in $J$.
That term was removed: it had no physical units and fought CE even when
both sides remained above $q_{\min}$.

### Joint mode

Consolidated simulated wealth drives a single $CE_{\mathrm{eff}}$. Validity
and viability are **per venue**:

$$
\begin{aligned}
J = CE_{\mathrm{eff}}
&- \mu_\ell[B_{\min,\ell}-B_\ell]_+ - \lambda_\ell\,\mathcal{S}_\ell \\
&- \mu_g[B_{\min,g}-B_g]_+ - \lambda_g\,\mathcal{S}_g.
\end{aligned}
$$

---

## 5. Certainty equivalent

For path $i$,

$$
R_i = \frac{1}{W_0}\sum_{h=0}^{H-1}\Delta W_{i,h},
\qquad
W_0 = \max\bigl(10^{-8},\, q_0 + b_0\cdot\mathrm{mark}_0\bigr).
$$

The entropic CE with risk aversion $\beta=$ `return_risk_aversion_beta` is

$$
CE = -\frac{1}{\beta}\log\left(\frac{1}{N}\sum_{i=1}^{N} e^{-\beta R_i}\right).
$$

The implementation uses a numerically stable shift by $\min_i R_i$, which
is algebraically identical:

$$
CE = \min_i R_i
- \frac{1}{\beta}\log\left(\frac{1}{N}\sum_{i=1}^{N}
  e^{-\beta(R_i-\min_j R_j)}\right).
$$

This is a soft minimum over path returns: it penalizes left-tail outcomes
without requiring a separate inventory-variance term. If every $R_i=0$,
then $CE=0$.

When `objective_normalize.enabled` is true, $CE$ is divided by an EWMA
scale $s$ from **previous** decisions (fixed for the whole Adam loop):

$$
CE_{\mathrm{eff}} = \frac{CE}{s},
\qquad
s \leftarrow \alpha_{\mathrm{CE}}\,s + (1-\alpha_{\mathrm{CE}})|CE|.
$$

Validity $\mu$ and viability $\lambda$ are **not** scaled by $s$.

---

## 6. Validity (book-quality gate)

Let $B(u^{(0)};\xi_t)\in[0,1]$ be the weighted average of four sub-scores
on the **current** decoded book (not the horizon wallet):

- **presence** — min touch depth vs a reference quantity,
- **tightness** — $\exp(-(x_{0,\mathrm{bid},0}+x_{0,\mathrm{ask},0})/\tau_{\mathrm{spread}})$,
- **balance** — symmetry of bid/ask quote notionals,
- **uniformity** — smoothness of marginal depth.

$$
\text{validity penalty}
= -\mu\cdot\max\bigl(0,\, B_{\min}-B\bigr).
$$

High $B$ is not rewarded; only shortfall below $B_{\min}$ is penalized.
This is the **presence** requirement: the optimizer is discouraged from
leaving an empty or structurally poor book.

---

## 7. Viability (quadratic reserve deficit)

Viability is the soft form of “keep enough total wallet to quote both
sides.” It is evaluated on **total** balances after the **first** simulated
fill (path column 1). Terminal-horizon recovery does not mask an immediate
hole. Pre-action $h=0$ is excluded.

### 7.1 Squared hinge

Let $q_{\min}=$ `min_viable_quote` (quote currency, **per side**):

$$
\begin{aligned}
d_{\mathrm{bid},1}^{(i)}
&= \frac{\bigl[q_{\min}-q_1^{(i)}\bigr]_+}{q_{\min}}, \\
d_{\mathrm{ask},1}^{(i)}
&= \frac{\bigl[q_{\min}-b_1^{(i)}p_1^{(i)}\bigr]_+}{q_{\min}}, \\
S_1^{(i)}
&= \bigl(d_{\mathrm{bid},1}^{(i)}\bigr)^2 + \bigl(d_{\mathrm{ask},1}^{(i)}\bigr)^2.
\end{aligned}
$$

Batch shortfall:

$$
\mathcal{S}(u;\xi_t) = \frac{1}{N}\sum_{i=1}^{N} S_1^{(i)}.
$$

Properties that matter for Adam:

- $S=0$ whenever both sides sit **above** $q_{\min}$ (no inventory
  headwind; this is not $W_{\mathrm{inv}}$).
- Below the floor the slope **steepens** with depth, so a crash does not
  flatten the gradient the way a saturating sigmoid product did.
- One dead side is enough to produce a penalty; the other side’s surplus
  does not cancel it.

Diagnostic health $1/(1+S)$ may be logged as
`viability_min_availability`. It is **not** used in $J$ or the dual.

Legacy keys `availability_target` $A^*$ and `margin_buffer` $\delta$ have been
removed from the configuration schema, as they do not enter $\mathcal{S}$ or the dual.

### 7.2 Explicit recovery objective

When a reserve deficit exists at step $h=0$ ($d_{s, 0} > 0$ for $s \in \{\mathrm{bid}, \mathrm{ask}\}$), the optimizer can be configured to actively recover the missing balance rather than merely avoiding further depletion. The expected recovery shortfall is:

$$
\mathcal{S}_{\mathrm{rec}}(u;\xi_t) = \mathbb{E}\left[ \sum_{s \in \{\mathrm{bid}, \mathrm{ask}\}, d_{s, 0} > 0} \left[ d_{s, 1} - (1 - \rho)\,d_{s, 0} \right]_+^2 \right]
$$

where $\rho \in [0, 1]$ is the configured `recovery_fraction` (default `0.0`).
- If $\rho = 0.0$ (default), $\mathcal{S}_{\mathrm{rec}} = 0$, preserving the standard floor constraint.
- If $\rho > 0.0$, the optimizer pays a quadratic penalty whenever post-fill deficit $d_{s, 1}$ fails to reduce the initial deficit $d_{s, 0}$ by at least fraction $\rho$. This provides an explicit gradient pulling the optimizer toward fills that replenish the depleted side.

### 7.3 Lagrangian penalty

$$
\text{viability penalty} = -\lambda_t\,\mathcal{S}(u;\xi_t) - \lambda_t\,\mathcal{S}_{\mathrm{rec}}(u;\xi_t).
$$

$\lambda_t$ is **fixed** for every Adam epoch of decision $t$. Gradients
flow: action $\to$ first-step fills $\to$ $(b_1,q_1,p_1)$ $\to$
$(S_1, S_{\mathrm{rec}, 1})$ $\to$ $J$.

### 7.4 Why this stays soft (spread widening)

A hard capacity clamp that forces quote size to zero when the wallet is
near $q_{\min}$ would empty the book. That fights validity, which wants
volume on the book.

With both penalties active, the typical interior response is: keep orders
present (validity) but **push them deeper** so simulated fill probability
falls (viability). The Lagrange term changes the CE–fill tradeoff; it does
not shut the quote down.

---

## 8. Outer dual on $\lambda$

After the inner solve, the dual is updated **once per frame** from an EWMA
of the realized shortfall. This is **not** $\lambda=e^\theta$.

$$
\begin{aligned}
\bar{S}_{t+1}
&= \alpha\,\bar{S}_t + (1-\alpha)\,\mathcal{S}_t
  \quad\text{(first observation: }\bar{S}=\mathcal{S}\text{)}, \\
\lambda_{t+1}
&= \mathrm{clip}\bigl(\lambda_t + \eta(\bar{S}_{t+1}-\varepsilon),\; 0,\; \lambda_{\max}\bigr).
\end{aligned}
$$

| Symbol | Config | Role |
| --- | --- | --- |
| $\varepsilon$ | `viability.shortfall_tolerance` | constraint slack $\mathbb{E}[S_1]\le\varepsilon$ |
| $\eta$ | `viability.lambda_learning_rate` | dual step |
| $\alpha$ | `viability.dual_ewma_decay` | noise filter on $\mathcal{S}$ |
| $\lambda_{\max}$ | `viability.lambda_max` | roof (default $100$) |
| $\lambda_0$ | `viability.lambda_init` | start, clipped to $[0,\lambda_{\max}]$ |

Dynamics:

- $\bar{S}>\varepsilon$ $\Rightarrow$ $\lambda$ rises $\Rightarrow$
  next frame penalizes deficit more strongly (typically wider spread).
- $\bar{S}<\varepsilon$ $\Rightarrow$ $\lambda$ falls.
- $\lambda=0$ applies no viability penalty, but the dual can turn it back
  on; there is no $\theta\le 0 \Rightarrow \lambda=0$ snap from log-space.

The EWMA stops one noisy Monte Carlo batch from slamming $\lambda$. The
roof stops the old $e^{30}\approx 10^{13}$ lockup. Because $S$ itself
supplies a steep primal gradient below $q_{\min}$, a bounded linear
$\lambda$ is enough; the exponential map was a workaround for vanishing
sigmoid tails.

Joint mode keeps separate $(\lambda,\bar{S})$ for local and global.

---

## 9. Bi-level algorithm

```text
INNER (this frame):   u* ≈ arg max_u  J(u; ξ_t, λ_t)     Adam, λ_t frozen
OUTER (after frame):  λ_{t+1} = clip(λ_t + η (S̄ − ε), 0, λ_max)
```

Warm-start blends the previous $u^*$ into the next initialization.
Cancellation returns the last completed epoch, not a partial step.

There is no inner dual loop during Adam.

---

## 10. What is not in $J$

| Concern | Where it lives |
| --- | --- |
| Tick / lot / min-notional | order validation, reconcile |
| Net-delta / transition risk | `NetDeltaRiskEngine` |
| Allocation / virtual-ledger caps | registry / halt |
| Stale frames, private-data freshness | strategy runner |
| Negative-CE action policy | outside $J$ (post-optimize) |
| Realized live PnL | fills on the venue, not the MC closure |

The MPC output is a **target book**. Downstream layers decide live orders.

---

## 11. Symbol to config

| Symbol | Config / state |
| --- | --- |
| $H,N,\beta$ | `horizon`, `n_samples`, `return_risk_aversion_beta` |
| $K$ | `local`/`global` `n_components` |
| $S_{\max},\Delta$ | `spread_max`, `spread_step` |
| $Q_{\min},Q_{\mathrm{cap}}$ | `cumulative_qty_floor_at_spread_max`, `cumulative_qty_cap_at_spread_max` |
| $B_{\min},\mu$ | `validity.min_score`, `validity.penalty_scale` |
| $q_{\min}$ | `viability.min_viable_quote` |
| $\varepsilon,\eta,\alpha,\lambda_{\max},\lambda_0$ | `shortfall_tolerance`, `lambda_learning_rate`, `dual_ewma_decay`, `lambda_max`, `lambda_init` |
| $s,\alpha_{\mathrm{CE}}$ | `objective_normalize` |
| $\lambda_t$ | MPC instance state (`_viability_lambda_*`), not persisted across process restart |
