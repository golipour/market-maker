# Mathematical Formulation of the Spread Grid and Depth Curves

This document is the compact mathematical statement of the **two-ladder**
geometry used by the controller:

1. a **computation ladder** (internal MPC / simulation grid), and
2. an **execution ladder** (live-order sample of continuous depth).

It also covers how the public book and the continuous quote action \(D(\delta)\)
are projected onto the computation ladder, how soft fills consume that depth,
and how a *fixed* continuous \(D\) becomes a live ladder of orders on the
execution sample set.

Counterpart docs:

- [`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md) — objective \(J\) and action \(u\)
- [`mpc-price-and-flow-models.md`](mpc-price-and-flow-models.md) — stochastic inputs of \(\Phi\)

Source of truth is the current code, not older single-grid notes.

Primary references:

- `mm.runtime.config.OrderbookGridConfig` — computation ladder
- `mm.runtime.config.ExecutionGridConfig` — execution ladder
- `ControllerRuntimeConfig.execution_spread_ladder()` — resolved live spreads
- `mm.mpc.mpc_tensor_batched._spread_grid_tensor`, `_book_cumulative_from_ob_rel`, `soft_fill_batched`
- `mm.mpc.state_tensor`, `mm.mpc.l_param`
- `mm.core.order_constraints.ladder_orders_from_sigmoid_with_report`
- `mm.strategies.target_book_builder.build_joint_target_book_from_sigmoid`
- `mm.strategies.mpc_engine.MpcStrategyEngine.decide` (wires both ladders)

---

## 1. Two ladders (architecture)

The continuous action is a sigmoid-mixture cumulative depth \(D(\delta)\) in
quote currency from mid. **Two discrete ladders** sample that same continuum
for different jobs:

\[
D(\delta)
\;\longrightarrow\;
\begin{cases}
\{s_j\}^{\mathrm{comp}}
  & \text{computation: book, soft fill, CE, validity, Adam} \\[4pt]
\{s_k\}^{\mathrm{exec}}
  & \text{execution: live limit prices / quantities}
\end{cases}
\]

| Name in prose | Config | Contract `role` | What it affects |
| --- | --- | --- | --- |
| **Computation ladder** | `controller.orderbook_grid` | `mpc_simulation` | State bucketing, soft fill, marks, objective, Adam |
| **Execution ladder** | `controller.execution_grid` | `execution_only` | Live order prices after \(D\) is fixed |

### 1.1 Stage → which ladder

| Stage | Ladder |
| --- | --- |
| State construction (absolute book → relative \(\delta\)) | computation (`orderbook_grid.spread_max`) |
| MPC book tensors / soft fill / book-derived marks | computation |
| Continuous action inside Adam / optimizer action snapshots | computation |
| `continuous_action_depth` diagnostics | **execution** (sample of fixed \(D\)) |
| Live ladder / `action_pipeline` | **execution** |

### 1.2 Default and independence

When `execution_grid` is omitted, or `mode=uniform` with **no**
`spread_step` / `spread_max` overrides, the execution ladder **mirrors**
`orderbook_grid.spread_ladder()` (legacy single-grid behaviour).

Changing **only** `execution_grid` does **not** change the Monte Carlo
objective. Changing `orderbook_grid` still does: it is part of the economic
problem, not a display preference.

Hard domain rule (enforced in `ExecutionGridConfig.resolved_spread_max`):

\[
S_{\max}^{\mathrm{exec}} \le S_{\max}^{\mathrm{comp}}.
\]

Every resolved execution ladder is strictly increasing in
\((0, S_{\max}^{\mathrm{exec}}]\) and **always includes** the outer edge
\(S_{\max}^{\mathrm{exec}}\) so the size constraint at the edge remains
visible when sampling \(D\).

### 1.3 Wiring in `decide()`

```text
execution_ladder = config.execution_spread_ladder()
                 = execution_grid.spread_ladder(canonical=orderbook_grid)

continuous_action_depth  ← evaluate D̃ on execution_ladder
JointTargetBook          ← ladder_orders_from_sigmoid(..., spread_ladder=execution_ladder)
diagnostics.orderbook_grid  ← orderbook_grid.as_contract()      # role: mpc_simulation
diagnostics.execution_grid  ← execution_grid.as_contract(...)   # role: execution_only
```

Adam / soft fill still build their spread tensor from `orderbook_grid`
(`spread_max`, `spread_step`) only.

---

## 2. Relative spread coordinate

Let \(p_t\) be the mid (quote per base). For a standing limit at absolute
price \(P\):

\[
\delta =
\begin{cases}
p_t - P, & \text{bid}, \\
P - p_t, & \text{ask}.
\end{cases}
\]

All computation and execution depth is a function of \(\delta\) in **quote
currency** (e.g. IRT on USDT_IRT), not ticks and not percent. Venue tick /
lot quantization happens **after** marginalization
(`apply_venue_constraints`).

---

## 3. Computation ladder (`orderbook_grid`)

Config (`OrderbookGridConfig`):

| Key | Symbol | Meaning |
| --- | --- | --- |
| `spread_max` | \(S_{\max}^{\mathrm{comp}}\) | Outer edge of the relative book / quote domain |
| `spread_step` | \(\Delta\) | Uniform bucket width |

Number of buckets / evaluation levels:

\[
M = \max\bigl(1,\, \lceil S_{\max}^{\mathrm{comp}}/\Delta \rceil\bigr).
\]

Upper edges (the computation sample points):

\[
s_j^{\mathrm{comp}}
=
\min\bigl(S_{\max}^{\mathrm{comp}},\, j\Delta\bigr),
\qquad j = 1,\ldots,M.
\]

Bucket \(j\) covers spreads in \((s_{j-1}^{\mathrm{comp}}, s_j^{\mathrm{comp}}]\)
with \(s_0^{\mathrm{comp}} = 0\). Cumulative market and quoted depth used by
soft fill are evaluated at the **upper edge** \(s_j^{\mathrm{comp}}\).

Implementation:

```text
OrderbookGridConfig.levels              → M
OrderbookGridConfig.spread_ladder()     → (s_1^comp, …, s_M^comp)
_spread_grid_tensor(..., M, Δ)          → same edges inside Adam
```

Example: \(S_{\max}^{\mathrm{comp}}=100\), \(\Delta=30\) → \((30,60,90,100)\).

This ladder is the **only** geometry that enters soft fill and the objective.

---

## 4. Projecting the public book onto the computation ladder

### 4.1 Absolute book → relative levels

From a raw venue book, each \((P,q)\) becomes \((\delta,q)\) using §2.
Levels with \(\delta \le 0\) or \(\delta > S_{\max}^{\mathrm{comp}}\) are
dropped or clipped by the relative-orderbook construction upstream of MPC
state.

### 4.2 Bucket index

Given a relative spread \(\delta > 0\):

\[
\mathrm{idx}(\delta)
=
\min\bigl(M-1,\, \max\bigl(0,\, \lceil \delta/\Delta \rceil - 1\bigr)\bigr).
\]

\(\delta \le 0\) maps to index \(0\). Same rule in `_spread_bucket_index` /
`state_to_tensor`.

### 4.3 Marginal and cumulative market depth

Let \(m_j^{\mathrm{bid}}\) (resp. ask) be the **sum of base qty** whose
\(\delta\) falls in computation bucket \(j\). Cumulative market book:

\[
C_j^{\mathrm{mkt}}
=
\sum_{k=1}^{j} m_k
\qquad
\bigl(\text{equivalently } \mathrm{cumsum}(m)\bigr).
\]

When building \(C\) from relative levels, the code accumulates into bucket
indices and forward-fills empty upper buckets so \(C_j\) is nondecreasing
(`_book_cumulative_from_ob_rel`).

**Consequence:** coarsening \(\Delta\) merges distinct venue prices into one
MPC bucket. That changes how much public depth sits ahead of a quote at
\(s_j^{\mathrm{comp}}\), and therefore changes simulated fill share and \(J\).

---

## 5. Continuous quote action: sigmoid mixture depth

The optimizer does **not** choose per-level sizes on either ladder. It chooses
parameters of a smooth cumulative depth curve per side
\(s \in \{\mathrm{bid},\mathrm{ask}\}\) with \(K=\) `n_components`:

\[
D_s(\delta)
=
\sum_{i=0}^{K-1}
L_i\,\sigma\bigl(k_i(\delta - x_{0,i})\bigr),
\qquad
\sigma(z)=(1+e^{-z})^{-1}.
\]

Executable cumulative depth subtracts the mid baseline:

\[
\tilde{D}_s(\delta)
=
\max\bigl(0,\, D_s(\delta) - D_s(0)\bigr).
\]

Decode details (softmax on \(L\), softplus on \(k\), optional
`book_quality.hard_structure`, reservation skew) live in `mm.mpc.l_param`
and [`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md).

### 5.1 Size constraint at the computation outer edge

Total executable size at \(S_{\max}^{\mathrm{comp}}\) is a depth logit between
a configured floor and a **wallet-clamped** cap:

\[
\tilde{D}_s(S_{\max}^{\mathrm{comp}})
\in
\bigl[Q_{\min},\, Q_s^{\max}(\xi_t)\bigr],
\]

with

\[
Q_{\mathrm{bid}}^{\max}
=
\min\bigl(Q_{\mathrm{cap}},\, q_t/p_t\bigr),
\qquad
Q_{\mathrm{ask}}^{\max}
=
\min\bigl(Q_{\mathrm{cap}},\, b_t\bigr)
\]

when a cap is configured (`cumulative_qty_cap_at_spread_max`); otherwise the
cap is wallet-only / floor-only as in `LParamConfig`.

So \(S_{\max}^{\mathrm{comp}}\) is not only a discretization horizon: it is
the abscissa at which the **required cumulative quantity** is enforced.
Increasing it gives the mixture more distance in which to park that quantity
(typically in a tail component), even if near-mid depth stays tiny.

Optional `hard_structure.action_max_spread` constrains only first-touch /
center geometry. It does **not** force the tail near mid.

---

## 6. Soft fill on the computation ladder

For each Monte Carlo path and horizon step, `soft_fill_batched` uses
**only** \(\{s_j^{\mathrm{comp}}\}\):

1. Evaluate \(\tilde{D}_s\) on \(\{s_j^{\mathrm{comp}}\}\) → cumulative quoted depth.
2. Marginal quoted size per bucket:

\[
q_j^{\mathrm{own}}
=
\tilde{D}(s_j^{\mathrm{comp}}) - \tilde{D}(s_{j-1}^{\mathrm{comp}}),
\qquad
\tilde{D}(s_0^{\mathrm{comp}})=0.
\]

3. Combined level depth with the market book:

\[
\ell_j = m_j + q_j^{\mathrm{own}}.
\]

4. Soft-consume sampled aggressive volume \(V\) against the combined
   cumulative book. With \(C_j^{\mathrm{comb}} = \sum_{k\le j}\ell_k\),

\[
r_j = V - C_j^{\mathrm{comb}},
\]

normalize \(r_j\) by its positive / negative range on the path, then

\[
f_j = a\bigl(r_j / \tau\bigr),
\qquad
\tau = \texttt{soft\_fill\_temp},
\]

where \(a\) is `sigmoid` or `hardsigmoid` (`soft_fill_activation`).
Consumed level volume is \(\ell_j f_j\).

5. Split consumed volume between market and own quotes. With
   `soft_fill_queue_model = back_of_queue`, the next-worse public level is
   treated as ahead of our order:

\[
\mathrm{share}_j^{\mathrm{own}}
=
\frac{q_j^{\mathrm{own}}}{\ell_j + m_{j+1}}
\quad
(m_{M+1}=0).
\]

Pro-rata mode drops the \(m_{j+1}\) term.

6. Own fills update cash / inventory at prices \(p_t \pm s_j^{\mathrm{comp}}\)
   with maker fee; the next **market-only** book is residual public depth.
   Resting own remainder is treated as canceled for the next simulated book
   (quotes are re-placed each decision).

**Consequence:** both \(\Delta\) and \(S_{\max}^{\mathrm{comp}}\) change which
\((s_j^{\mathrm{comp}}, m_j, q_j^{\mathrm{own}})\) the fill model sees. A
coarser computation ladder is a different fill operator, not a subsampled
evaluation of the same one. The execution ladder never enters this loop.

---

## 7. Execution ladder (`execution_grid`)

After Adam returns a **fixed** continuous \(D\), live orders are formed by
sampling on \(\{s_k^{\mathrm{exec}}\}\), **not** necessarily on
\(\{s_j^{\mathrm{comp}}\}\):

\[
q_k^{\mathrm{live}}
=
\max\bigl(0,\,
  \tilde{D}(s_k^{\mathrm{exec}}) - \tilde{D}(s_{k-1}^{\mathrm{exec}})
\bigr),
\]

with \(s_0^{\mathrm{exec}} = 0\). Bid price \(p_t - s_k^{\mathrm{exec}}\),
ask price \(p_t + s_k^{\mathrm{exec}}\). Zero marginal levels are dropped.
Then:

1. Cap largest-\(q\) orders per side to `max_orders_per_side`
   (`cap_orders_per_side_with_report`).
2. Venue tick / lot / min-qty / min-notional
   (`apply_venue_constraints`) — **after** the strategy ladder.
3. Risk / reconcile layers decide which orders actually go live.

`ladder_orders_from_sigmoid_with_report(..., spread_ladder=...)` takes the
execution ladder explicitly so the live book can diverge from the MPC grid
without changing soft fill.

### 7.1 Modes (`ExecutionGridConfig`)

| Mode | Geometry |
| --- | --- |
| `uniform` | Linear steps; optional `spread_step` / `spread_max` overrides; default = computation ladder |
| `log` | Geometric points on `[spread_min, spread_max]` with `levels` samples (`spread_min > 0`, `levels ≥ 2`) |
| `hybrid` | Linear inner band to `inner_linear_max` (step `inner_step` or computation \(\Delta\)), then geometric outer band from `spread_min` (or `inner_linear_max`) to \(S_{\max}^{\mathrm{exec}}\) |
| `explicit` | Caller-provided `spreads` (outer edge forced to \(S_{\max}^{\mathrm{exec}}\)) |

Normalization (`_normalize_execution_spreads`): drop non-finite / non-positive
points, reject any point above \(S_{\max}^{\mathrm{exec}}\), sort, unique, and
append the outer edge if missing.

### 7.2 Diagnostics

| Field | Content |
| --- | --- |
| `diagnostics.orderbook_grid` | Computation geometry (`role: mpc_simulation`) |
| `diagnostics.execution_grid` | Resolved live sample ladder (`role: execution_only`) |
| `continuous_action_depth` / `action_pipeline.continuous` | \(\tilde{D}\) evaluated on the **execution** ladder |
| Optimizer action snapshots | Remain on the **computation** ladder |

---

## 8. Why the book is sensitive to each ladder

### 8.1 Computation ladder still changes the decision problem

Changing `orderbook_grid.spread_max` / `spread_step` changes:

- market depth aggregation (§4),
- soft-fill levels (§6),
- the abscissa of the size constraint (§5.1).

Two configs with different \((S_{\max}^{\mathrm{comp}},\Delta)\) are **not**
the same optimization problem. Computation cost of soft fill / VWAP / book
tensors scales roughly with \(M\) (and with \(N \times H \times E\) Adam
work).

### 8.2 Execution ladder is mostly resolution / fingerprint

Changing only `execution_grid` rematerializes the **same** continuous \(D\)
onto a different sample set. Cumulative depth between neighbouring execution
points is preserved by construction (differencing \(\tilde{D}\)). Remaining
sensitivities are post-sample effects:

- `max_orders_per_side` keeps largest quantities, not nearest prices;
- tick / lot / min-notional rounding and duplicate-tick merges;
- cancel/replace churn from denser near-mid samples.

Execution-ladder density does **not** enter Adam runtime in a material way
(one pass after optimize).

### 8.3 \(S_{\max}^{\mathrm{comp}}\) as economic / risk horizon

Raising computation \(S_{\max}\) while keeping the size floor/cap at the
outer edge allows the mixture to satisfy
\(\tilde{D}(S_{\max}^{\mathrm{comp}})\ge Q_{\min}\) by placing almost all
size in the far tail. Near-mid emptiness is then a valid local optimum of
\(J\) unless validity / presence punish it.

### 8.4 Interaction with validity

Validity / book-quality scores use the decoded \(h{=}0\) curve (and optionally
combined market depth on the computation grid). If
`validity.penalty_scale = 0`, nothing in \(J\) requires near-mid presence;
CE (and soft viability) can push size outward within
\([0,S_{\max}^{\mathrm{comp}}]\). Neither ladder alone creates that presence
preference.

---

## 9. Design status

The computation / execution split is **implemented**:

1. **Computation ladder** \(\{s_j^{\mathrm{comp}}\}\) — uniform linear
   `orderbook_grid` for book bucketing, soft fill, and Adam.
2. **Execution ladder** \(\{s_k^{\mathrm{exec}}\}\) — optional
   uniform / log / hybrid / explicit sampler used only when building live
   orders (and continuous-depth diagnostics) from a fixed \(D\).

Bounded temporal jitter / randomization of the execution ladder is **not**
implemented; execution stays deterministic until replay/seed contracts exist.

Making the **computation** ladder non-uniform remains a larger change: every
`ceil(δ/Δ)` bucket rule must become an edge-lookup on irregular
\(\{s_j^{\mathrm{comp}}\}\).

---

## 10. Symbol → config / code

| Symbol | Config / code |
| --- | --- |
| \(S_{\max}^{\mathrm{comp}},\Delta,M\) | `orderbook_grid.spread_max`, `.spread_step`, `.levels` |
| \(\{s_j^{\mathrm{comp}}\}\) | `orderbook_grid.spread_ladder()`, `_spread_grid_tensor` |
| \(\{s_k^{\mathrm{exec}}\}\) | `execution_grid.spread_ladder(canonical=…)`, `execution_spread_ladder()` |
| \(S_{\max}^{\mathrm{exec}}\) | `execution_grid.resolved_spread_max(canonical)` (≤ computation max) |
| \(\mathrm{idx}(\delta)\) | `_spread_bucket_index` |
| \(m_j, C^{\mathrm{mkt}}\) | `_book_cumulative_from_ob_rel` / `_book_cumulative_from_arrays` |
| \(D,\tilde{D}\) | `l_param` + `effective_depth_*_tensor` |
| \(Q_{\min},Q_{\mathrm{cap}}\) | `cumulative_qty_floor_at_spread_max`, `cumulative_qty_cap_at_spread_max` |
| \(\tau,a\) | `soft_fill_temp`, `soft_fill_activation` |
| queue share | `soft_fill_queue_model` (`back_of_queue` / `pro_rata`) |
| live \(q_k\) | `ladder_orders_from_sigmoid_with_report(..., spread_ladder=execution_ladder)` |
