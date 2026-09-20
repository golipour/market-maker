# Mathematical Formulation of the Spread Grid and Depth Curves

This document is the compact mathematical statement of the **internal MPC
spread grid** and the optional **execution ladder**, how the public book and
the MPC quote action are projected onto the internal grid, how soft fills
consume that depth, and how the continuous action becomes a live ladder of
orders.

It is the counterpart of
[`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md) for the
geometry of $\mathcal{B}_t$ and $u$, and of
[`mpc-price-and-flow-models.md`](mpc-price-and-flow-models.md) for the book
side of the rollout operator $\Phi$. Source of truth is the current code, not
older research notes.

Primary references:

- `mm.runtime.config.OrderbookGridConfig` (MPC simulation grid)
- `mm.runtime.config.ExecutionGridConfig` (live-order sample ladder)
- `mm.mpc.mpc_tensor_batched._spread_grid_tensor`, `_book_cumulative_from_ob_rel`, `soft_fill_batched`
- `mm.mpc.state_tensor` (depth tensors, State → fixed-length book arrays)
- `mm.mpc.l_param` (sigmoid-mixture decode, size at $S_{\max}$)
- `mm.core.order_constraints.ladder_orders_from_sigmoid_with_report`
- `mm.strategies.target_book_builder.build_joint_target_book_from_sigmoid`

---

## 1. Role in the controller

Two config blocks define the geometry:

| Config | Role |
| --- | --- |
| `controller.orderbook_grid` | **Internal MPC grid** — book projection, soft fill, objective, Adam |
| `controller.execution_grid` | **Execution ladder** — samples continuous $D(\delta)$ into live orders |

| Stage | Grid used |
| --- | --- |
| State construction | `orderbook_grid` — project absolute venue book levels to relative spreads |
| MPC book tensors | `orderbook_grid` — bucket and accumulate market depth |
| Soft fill | `orderbook_grid` — match sampled flow against market + quoted depth |
| Book-derived marks | `orderbook_grid` — best bid/ask / VWAP from post-flow books |
| Continuous action (optimizer) | `orderbook_grid` — evaluate $D$ inside Adam / diagnostics snapshots |
| Live ladder / `action_pipeline.continuous` | `execution_grid` — turn $D$ into marginal limit orders |

When `execution_grid` is omitted or `mode=uniform` with no overrides, the
execution ladder **mirrors** `orderbook_grid.spread_ladder()` (legacy
single-grid behaviour). Changing only `execution_grid` does **not** change
the Monte Carlo objective; changing `orderbook_grid` still does.

---

## 2. Relative spread coordinate

Let $p_t$ be the mid (quote per base). For a standing limit at absolute price
$P$:

$$
\delta =
\begin{cases}
p_t - P, & \text{bid}, \\
P - p_t, & \text{ask}.
\end{cases}
$$

All MPC depth, fills, and ladders are functions of $\delta$ in **quote
currency** (IRT on USDT_IRT), not in ticks and not in percent. Venue
tick/lot quantization happens **after** discretization
(`apply_venue_constraints`).

---

## 3. Linear orderbook grid

Config (`OrderbookGridConfig`):

| Key | Symbol | Meaning |
| --- | --- | --- |
| `spread_max` | $S_{\max}$ | Outer edge of the relative book / quote domain |
| `spread_step` | $\Delta$ | Uniform bucket width |

Number of buckets / ladder levels:

$$
M = \max\bigl(1,\, \lceil S_{\max}/\Delta \rceil\bigr).
$$

Upper edges of the buckets (the evaluation points):

$$
s_j = \min\bigl(S_{\max},\, j\Delta\bigr),
\qquad j = 1,\ldots,M.
$$

Bucket $j$ covers spreads in $(s_{j-1}, s_j]$ with $s_0 = 0$. Cumulative
market and quoted depth are always evaluated at the **upper edge** $s_j$.

Implementation:

```text
_grid_levels(S_max, Δ)           → M
_spread_grid_tensor(..., M, Δ)   → (s_1,…,s_M) clamped to S_max
OrderbookGridConfig.spread_ladder() → same tuple (MPC / default execution)
ExecutionGridConfig.spread_ladder(canonical=…) → live sample ladder
```

Example: $S_{\max}=100$, $\Delta=30$ → $(30,60,90,100)$.

The live order ladder is produced separately by `ExecutionGridConfig` (§7).
When that config is the default uniform mirror, the live ladder equals
$\{s_j\}$ above.

---

## 4. Projecting the public book onto the grid

### 4.1 Absolute book → relative levels

From a raw venue book, each $(P,q)$ becomes $(\delta,q)$ using §2. Levels
with $\delta \le 0$ or $\delta > S_{\max}$ are dropped or clipped by the
relative-orderbook construction upstream of the MPC state.

### 4.2 Bucket index

Given a relative spread $\delta > 0$:

$$
\mathrm{idx}(\delta)
=
\min\bigl(M-1,\, \max\bigl(0,\, \lceil \delta/\Delta \rceil - 1\bigr)\bigr).
$$

$\delta \le 0$ maps to index $0$. This is
`_spread_bucket_index` / the same rule in `state_to_tensor`.

### 4.3 Marginal and cumulative market depth

Let $m^{\mathrm{bid}}_j$ (resp. ask) be the **sum of base qty** whose
$\delta$ falls in bucket $j$. Then the cumulative market book used by soft
fill is

$$
C^{\mathrm{mkt}}_j
=
\sum_{k=1}^{j} m_k
\qquad
\bigl(\text{equivalently } \mathrm{cumsum}(m)\bigr).
$$

When building $C$ from a list of relative levels already sorted by $\delta$,
the code accumulates into bucket indices and forward-fills empty upper
buckets so $C_j$ is nondecreasing
(`_book_cumulative_from_ob_rel`).

**Consequence:** coarsening $\Delta$ merges distinct venue price levels into
one MPC bucket. That changes how much public depth sits “ahead of” a quote
at $s_j$, and therefore changes simulated fill share.

---

## 5. Continuous quote action: sigmoid mixture depth

The optimizer does **not** choose per-level sizes. It chooses parameters of
a smooth cumulative depth curve per side. On side $s \in \{\mathrm{bid},\mathrm{ask}\}$
with $K=$ `n_components`:

$$
D_s(\delta)
=
\sum_{i=0}^{K-1}
L_i\,\sigma\bigl(k_i(\delta - x_{0,i})\bigr),
\qquad
\sigma(z)=(1+e^{-z})^{-1}.
$$

Executable cumulative depth subtracts the mid baseline:

$$
\tilde{D}_s(\delta)
=
\max\bigl(0,\, D_s(\delta) - D_s(0)\bigr).
$$

Decode details (softmax on $L$, softplus on $k$, optional
`book_quality.hard_structure` on first-touch centers, reservation skew) live
in `mm.mpc.l_param` and are summarized in
[`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md) §2.3.

### 5.1 Size constraint at the outer edge

Total executable size at $S_{\max}$ is a depth logit between a configured
floor and a **wallet-clamped** cap:

$$
\tilde{D}_s(S_{\max})
\in
\bigl[Q_{\min},\, Q_s^{\max}(\xi_t)\bigr],
$$

with

$$
Q_{\mathrm{bid}}^{\max}
=
\min\bigl(Q_{\mathrm{cap}},\, q_t/p_t\bigr),
\qquad
Q_{\mathrm{ask}}^{\max}
=
\min\bigl(Q_{\mathrm{cap}},\, b_t\bigr)
$$

when a cap is configured (`cumulative_qty_cap_at_spread_max`); otherwise the
cap is wallet-only / floor-only as implemented in `LParamConfig`.

So $S_{\max}$ is not only a discretization horizon: it is the abscissa at
which the **required cumulative quantity** is enforced. Increasing $S_{\max}$
gives the mixture more distance in which to park that quantity (typically
in a tail component), even if near-mid depth stays tiny.

Optional `hard_structure.action_max_spread` constrains only the
**first-touch / center** geometry. It does **not** force the tail to place
size near mid; the remaining depth may still concentrate near $S_{\max}$.

---

## 6. Soft fill on the grid

For each Monte Carlo path and horizon step, `soft_fill_batched`:

1. Evaluates $\tilde{D}_s$ on $\{s_j\}$ → cumulative quoted depth.
2. Converts to **marginal** quoted size per bucket:

$$
q^{\mathrm{own}}_j
=
\tilde{D}(s_j) - \tilde{D}(s_{j-1}),
\qquad
\tilde{D}(s_0)=0.
$$

3. Forms combined level depth with the market book:

$$
\ell_j = m_j + q^{\mathrm{own}}_j.
$$

4. Soft-consumes sampled aggressive volume $V$ against the combined
   cumulative book. With $C^{\mathrm{comb}}_j = \sum_{k\le j}\ell_k$,

$$
r_j = V - C^{\mathrm{comb}}_j,
$$

normalize $r_j$ by its positive / negative range on the path, then

$$
f_j = a\bigl(r_j / \tau\bigr),
\qquad
\tau = \texttt{soft\_fill\_temp},
$$

where $a$ is `sigmoid` or `hardsigmoid`
(`soft_fill_activation`). Consumed level volume is $\ell_j f_j$.

5. Splits consumed volume between market and own quotes. With
   `soft_fill_queue_model = back_of_queue`, the next-worse public level is
   treated as ahead of our order (conservative queue):

$$
\mathrm{share}^{\mathrm{own}}_j
=
\frac{q^{\mathrm{own}}_j}{\ell_j + m_{j+1}}
\quad
(m_{M+1}=0).
$$

Pro-rata mode drops the $m_{j+1}$ term.

6. Own fills update cash / inventory at prices $p_t \pm s_j$ with maker fee;
   the next **market-only** book is the residual public depth after market
   share of consumption. Resting own remainder is treated as canceled for
   the next simulated book (quotes are re-placed each decision).

**Consequence:** both $\Delta$ and $S_{\max}$ change which $(s_j, m_j,
q^{\mathrm{own}}_j)$ the fill model sees. A coarser grid is a different
fill operator, not a subsampled evaluation of the same one.

---

## 7. From continuous depth to live orders

After Adam, a **fixed** continuous $D$ is sampled on the **execution
ladder** $\{s^{\mathrm{exec}}_k\}$ (`controller.execution_grid`), not
necessarily on the internal MPC edges:

$$
q_k^{\mathrm{live}}
=
\max\bigl(0,\, \tilde{D}(s^{\mathrm{exec}}_k) - \tilde{D}(s^{\mathrm{exec}}_{k-1})\bigr).
$$

Bid price $p_t - s^{\mathrm{exec}}_k$, ask price $p_t + s^{\mathrm{exec}}_k$.
Zero marginal levels are dropped. Then:

1. Cap largest-$q$ orders per side to `max_orders_per_side`.
2. Venue tick / lot / min-qty / min-notional
   (`apply_venue_constraints`) — **after** the strategy ladder.
3. Risk / reconcile layers decide which orders actually go live.

`execution_grid` modes:

| Mode | Geometry |
| --- | --- |
| `uniform` | Linear steps (optional `spread_step` / `spread_max` overrides; default = MPC ladder) |
| `log` | Geometric points on `[spread_min, spread_max]` with `levels` samples |
| `hybrid` | Linear inner band to `inner_linear_max`, then geometric outer band |
| `explicit` | Caller-provided `spreads` (outer edge forced to `spread_max`) |

All execution spreads must lie in $(0, S_{\max}]$ of the MPC grid and
**always include** $S_{\max}$ so outer-edge size constraints remain visible.

Diagnostics:

- `diagnostics.orderbook_grid` — internal MPC geometry
- `diagnostics.execution_grid` — resolved live sample ladder
- `action_pipeline.continuous` — $\tilde{D}$ evaluated on the **execution** ladder

Optimizer action snapshots remain on the internal MPC grid.

---

## 8. Why the book is sensitive to $S_{\max}$ and $\Delta$

### 8.1 Internal grid still changes the decision problem

Changing `orderbook_grid.spread_max` / `spread_step` still changes:

- market depth aggregation (§4),
- soft-fill levels (§6),
- the abscissa of the size constraint (§5.1).

So two configs with different **internal** $(S_{\max},\Delta)$ are **not**
the same decision problem.

### 8.2 Execution ladder is mostly resolution / fingerprint

Changing only `execution_grid` rematerializes the **same** continuous $D$
onto a different sample set. Cumulative depth between neighbouring
execution points is preserved by construction (differencing $\tilde{D}$).
Remaining sensitivities are post-sample effects:

- `max_orders_per_side` keeps largest quantities, not nearest prices;
- tick / lot / min-notional rounding and duplicate-tick merges;
- cancel/replace churn from denser near-mid samples.

### 8.3 $S_{\max}$ as economic / risk horizon

Raising internal $S_{\max}$ while keeping the size floor/cap at the outer
edge allows the mixture to satisfy $\tilde{D}(S_{\max})\ge Q_{\min}$ by
placing almost all size in the far tail. Near-mid emptiness is then a
valid local optimum of $J$ unless other terms (validity / presence)
punish it.

### 8.4 Interaction with validity

Validity / book-quality scores use the decoded $h{=}0$ curve (and optionally
combined market depth). If `validity.penalty_scale = 0`, nothing in $J$
requires near-mid presence; CE (and any soft viability) can push size
outward within $[0,S_{\max}]$. Grid geometry alone cannot create that
presence preference.

---

## 9. Design status

The internal / execution split from §1 and §7 is **implemented**:

1. **Internal MPC grid** `{s_j}` — unchanged uniform linear geometry for
   book bucketing, soft fill, and Adam.
2. **Execution ladder** `{s^{\mathrm{exec}}_k}` — optional uniform / log /
   hybrid / explicit sampler used only when building live orders from a
   fixed continuous $D$.

Bounded temporal jitter / randomization of the execution ladder is **not**
implemented yet; keep execution deterministic until replay/seed contracts
exist.

Making the **internal** grid non-uniform remains a larger change: every
`ceil(δ/Δ)` bucket rule must become an edge-lookup on irregular `{s_j}`.

---

## 10. Symbol to config / code

| Symbol | Config / code |
| --- | --- |
| $S_{\max},\Delta,M$ | `orderbook_grid.spread_max`, `.spread_step`, `.levels` |
| $\{s_j\}$ | `orderbook_grid.spread_ladder()`, `_spread_grid_tensor` |
| $\{s^{\mathrm{exec}}_k\}$ | `execution_grid.spread_ladder(canonical=...)`, `ControllerRuntimeConfig.execution_spread_ladder()` |
| $\mathrm{idx}(\delta)$ | `_spread_bucket_index` |
| $m_j, C^{\mathrm{mkt}}$ | `_book_cumulative_from_ob_rel` / `_book_cumulative_from_arrays` |
| $D,\tilde{D}$ | `l_param` + `effective_depth_*_tensor` |
| $Q_{\min},Q_{\mathrm{cap}}$ | `cumulative_qty_floor_at_spread_max`, `cumulative_qty_cap_at_spread_max` |
| $\tau,a$ | `soft_fill_temp`, `soft_fill_activation` |
| queue share | `soft_fill_queue_model` (`back_of_queue` / pro-rata) |
| live $q_k$ | `ladder_orders_from_sigmoid_with_report(..., spread_ladder=...)` |
