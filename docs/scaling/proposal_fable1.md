# Book-Quality vs Risk-Adjusted Return: Incremental Improvement Proposals

Goal: a profitable, balanced, less risky order book with good depth and tight spread, as a
**tunable trade-off** between the existing risk-adjusted-return objective and a new
book-quality objective. The trade-off is controlled by **one scalar per venue**
(`book_quality_weight`), with optional fixed sub-weights. Everything below **repairs or
extends existing components** — no replacements.

Design principle (and the key simplification): things that must *always* hold are expressed
as **hard structural constraints** in the action parameterization (`LParamConfig`,
capacity bounds). Hard constraints need **no objective weight** and cannot be traded away
by the optimizer. Only genuinely negotiable preferences (tighter vs more profitable, deeper
vs less inventory risk) go into the objective behind the scalar weight.

Current state, from code:

- The terminal objective is *only* the entropic certainty equivalent of returns
  (`terminal_objective()` in `mm/mpc/reward_tensor.py`; `joint_terminal_objective()` in
  `mm/mpc/joint_mpc.py`). There is no book-shape term at all — the optimizer is free to
  park all depth at `spread_max` (far, never-filled "fake depth") or quote a heavily
  one-sided book whenever that looks marginally better in expectation.
- Hard structure already exists and works: spread-structured first-leg constraints
  (`quote_spread_max`, `quote_spread_min_depth`, `quote_spread_k_epsilon` in
  `mm/mpc/l_param.py`), per-side dynamic capacity bounds and imbalance skew floors
  (`_quote_capacity_bounds_batched()` in `mm/mpc/mpc_tensor_batched.py`).
- Discretization (`mm/core/order_constraints.py`) caps orders per side by **largest
  quantity**, which can drop the near-touch order and destroy the spread the optimizer
  chose.

Proposals are ordered by (criticality of failure mode) / (architectural disruption).

---

## P1 — Fix order-capping so the near-touch quote survives discretization

**Failure mode (critical, silent):** `cap_orders_per_side_with_report()` keeps the
largest-qty orders per side. With a spread-structured action, the first leg is a small
tight quote plus large far tail legs; when the ladder exceeds `max_orders_per_side`, the
tight quote is exactly the one dropped. The live book then has *worse* spread than the
optimizer intended, and no objective change can fix it because it happens after the
optimizer.

**File / function:** `mm/core/order_constraints.py`,
`cap_orders_per_side_with_report()` (used by `build_orders_from_sigmoid_with_report()`).

**Change (pseudocode):** keep the nearest-to-anchor order per side unconditionally, then
fill the remaining slots by quantity:

```python
def cap_side(side_orders, cap):
    if len(side_orders) <= cap:
        return side_orders
    # buys: nearest = highest price; sells: nearest = lowest price
    nearest = max(side_orders, key=price_key)          # price_key sign per side
    rest = sorted((o for o in side_orders if o is not nearest),
                  key=lambda o: o.qty, reverse=True)
    return [nearest] + rest[: cap - 1]
```

Optionally protect the top-K nearest (`K=1` default) via a constant; no config change
needed.

**Effort:** ~1–2 hours including updating `tests/unit/test_order_constraints.py` and
`test_action_pipeline.py` expectations.

**Prerequisites:** none.

---

## P2 — Book-quality term in the single-venue objective (the tunable trade-off)

**What:** add a differentiable book-quality score computed from the **decoded effective
first-step action** (h=0), and combine:

```text
J = CE_beta(R) + w_book * J_book        # w_book = optimizer.<venue>.book_quality_weight
```

`J_book` is deterministic given the action (no Monte Carlo variance), so it is added
*outside* the certainty equivalent. Keep `components["risk_adjusted_return"]` as the
**pure** return CE (see P7) and add new components.

**Math.** For side `s ∈ {bid, ask}` decode `(L, k, x0)` with the existing
`materialize_side_torch()` and evaluate cumulative depth on the existing spread grid
`g_0..g_M` (`spread_step`, `spread_max`):

```text
D_s(g) = Σ_i L_i * (σ(k_i (g − x0_i)) − σ(−k_i x0_i))      # existing component_delta_at_torch
m_s[j] = D_s(g_j) − D_s(g_{j−1})                            # marginal depth per level
Q_s    = D_s(spread_max)                                    # total quoted per side

tight_s   = (Σ_j m_s[j] * g_j) / (Q_s + ε) / spread_max     # depth-weighted spread, in [0,1]
balance   = ((Q_bid − Q_ask) / (Q_bid + Q_ask + ε))²        # in [0,1]
depth_s   = Q_s / (q_max_s + ε)                             # vs dynamic capacity cap, in [0,1]

J_book = − w_tight * 0.5 * (tight_bid + tight_ask)
         − w_bal   * balance
         + w_depth * 0.5 * (depth_bid + depth_ask)
```

All three terms are dimensionless and bounded, so `w_book` is comparable across symbols
and `R` (also dimensionless) trades off against it sanely. `w_tight, w_bal, w_depth`
default to 1.0 and are secondary knobs; the primary knob is `w_book` (default `0.0` =
exactly current behavior).

`q_max_s` for the depth term is the already-computed dynamic capacity bound at h=0
(`bid_quote_size_max_t`, `ask_quote_size_max_t`), so a venue with no inventory is not
penalized for an absent ask — the depth term asks for depth *relative to what is
affordable*, which is the "balanced given capacity" notion that matches the existing
capacity philosophy.

**Files / functions:**

1. `mm/mpc/reward_tensor.py` — new helper, pure function of the action tensor:

```python
def book_quality_terms(action_h, n_components, spread_grid, l_param,
                       bid_q_min, bid_q_max, ask_q_min, ask_q_max,
                       *, w_tight=1.0, w_bal=1.0, w_depth=1.0) -> tuple[Tensor, dict]
```

   Internally calls `materialize_side_torch(action_h, side_off=0 / 3n, ...)` and
   `component_delta_at_torch` per grid point (vectorize over the grid: shape `(n, M)`).

2. `mm/mpc/reward_tensor.py`, `terminal_objective()` — accept optional
   `book_quality: Tensor | None` and `book_quality_weight: float`:

```python
term_return = objective_entropic_return(returns, beta=...)
J = term_return + w_book * book_quality if book_quality is not None else term_return
components = {
    "risk_adjusted_return": term_return,        # stays pure — P7 depends on this
    "book_quality": w_book * book_quality,
    "book_tightness": ..., "book_balance": ..., "book_depth": ...,
}
```

3. `mm/mpc/mpc_tensor_batched.py`, the closure in `optimize_with_diagnostics()`
   (around line 1381 where `terminal_objective(returns, ...)` is called) — compute
   `book_quality_terms()` once per closure from `action_h` at `h == 0` (the slice
   `x_torch[:single_len]`) using the h=0 capacity bounds already computed
   (`bid_quote_size_min_t` … at the first step), then pass into `terminal_objective`.

**Observability for free:** new `components` entries flow automatically into
`OptimizerRunDiagnostics` epoch metrics and into Postgres
`mm_optimizer_component_facts` — no persistence change needed. A Grafana panel can be
added later via `mm/tools/generate_observability_v2_dashboards.py` (optional).

**Effort:** ~1 day including unit tests (`tests/unit/test_entropic_objective.py` gains a
`test_book_quality_*` family; gradient-flow test asserting `dJ_book/d(action) != 0`).

**Prerequisites:** none (PyTorch path only; numpy decode already exists for parity tests).

---

## P3 — Same book-quality term in the joint local/global objective

**Files / functions:**

- `mm/mpc/joint_mpc.py`, `joint_terminal_objective()` — accept per-venue
  `local_book_quality` / `global_book_quality` tensors and per-venue weights:

```text
J = w_L * CE_{β_L}(R^L) + active_G * w_G * CE_{β_G}(R^G)
    + w_book_L * J_book^L + active_G * w_book_G * J_book^G
```

- `mm/mpc/joint_optimizer.py`, the closure in `optimize_joint_with_diagnostics()`
  (around line 732) — decode the local and global h=0 action slices (the venue split
  already exists via `split_venue_action_flat` / per-venue `l_param`), call the P2 helper
  per venue with that venue's h=0 capacity bounds, pass into the objective.

Components gain `local_book_quality` / `global_book_quality` mirroring the existing
`local_risk_adjusted_return` / `global_risk_adjusted_return` naming.

**Effort:** ~0.5–1 day (mostly wiring + `tests/unit/test_phase7_joint_mpc.py` additions).

**Prerequisites:** P2.

---

## P4 — Hard balance-ratio constraint (no weight needed)

**What:** as you noted, hard constraints need no weight selection. A guaranteed
not-too-lopsided book can be a *structural* property of the action decode, like
`quote_spread_max` already is. New optional per-venue key
`quote_balance_ratio_max = ρ ≥ 1`: the decoded side totals must satisfy
`max(Q_bid, Q_ask) ≤ ρ · min(Q_bid, Q_ask)` — but **only up to capacity**, so a venue with
zero inventory can still bid (it must be able to acquire inventory).

**Math (differentiable clamp, applied after the two depth logits decode):**

```text
Q_bid_c = min(Q_bid, max(ρ · Q_ask, bid_min))
Q_ask_c = min(Q_ask, max(ρ · Q_bid, ask_min))
```

`min/max` are autograd-safe; the `bid_min/ask_min` floor prevents deadlock when one side's
capacity collapses to ~0 (otherwise both sides would be forced to 0).

**Files / functions:**

- `mm/mpc/mpc_tensor_batched.py`, `VenueOptimizerParams` — new field
  `quote_balance_ratio_max: float | None = None`, parsed in `from_mapping()`, emitted in
  `as_config()`, validated `>= 1.0` in `validate()`.
- Implementation point: a small helper `coupled_quote_sizes(...)` computing both side
  totals from the two depth logits and applying the clamp, used:
  - in the torch rollout: compute `Q_bid_c/Q_ask_c` per step and pass them as
    `quote_size_min == quote_size_max == Q_c` into the existing
    `materialize_side_torch(..., quote_size_min=, quote_size_max=)` overrides
    (no signature change to `materialize_side_*` needed);
  - in the numpy decode path `sigmoid_action_from_flat_raw()` in `mm/mpc/l_param.py`
    (same override mechanism), so diagnostics, the negative-objective floor action
    (`mm/strategies/negative_objective_policy.py` → `_min_sigmoid_action`), and the
    discretized target book all see the same constrained sizes.

**Why hard instead of relying on the P2 `balance` term:** the soft term shapes the
optimum; the hard ratio guarantees a worst case regardless of `w_book`. They compose: set
`ρ` to the maximum lopsidedness you will ever tolerate, then tune `w_book` freely inside
that envelope. (Same relationship as `quote_spread_max` vs the `tight` term.)

**Effort:** ~1 day (both decode paths + capacity-interaction tests in
`tests/unit/test_quote_size_softmax_L.py` and `test_mpc_spread_depth.py`).

**Prerequisites:** none (independent of P2/P3, composes with them).

---

## P5 — Config plumbing and strict-schema allow-listing

New keys are venue-scoped, matching the existing convention
(`controller.optimizer.local` / `controller.optimizer.global`):

```json
"optimizer": {
  "local": {
    "book_quality_weight": 0.3,
    "book_tightness_weight": 1.0,
    "book_balance_weight": 1.0,
    "book_depth_weight": 1.0,
    "quote_balance_ratio_max": 4.0
  }
}
```

**Files / functions (exact allow-lists that must all be updated, or the strict schema
rejects the keys):**

- `mm/mpc/mpc_tensor_batched.py` — `VenueOptimizerParams` fields, `from_mapping()`,
  `as_config()`, `validate()` (`book_quality_weight >= 0`, sub-weights `>= 0`,
  `quote_balance_ratio_max >= 1` or `None`).
- `mm/strategies/mpc_engine.py` — `_VENUE_OPTIMIZER_KEYS` (top of file) gains the five
  keys.
- `mm/runtime/config.py` — the `unknown controller.optimizer.<venue> keys` check
  (~line 1417) uses its own allowed-key set; add the same keys there.
- `mm/configs/market_maker.runtime.example.json` — document defaults
  (`book_quality_weight: 0.0` so existing configs are bit-for-bit unchanged).

**Effort:** ~2–3 hours including `tests/unit/test_runtime_config.py` and
`test_runtime_migration_contracts.py` additions.

**Prerequisites:** P2 (fields must exist before plumbing is useful).

---

## P6 — Validate spread-structured depth against venue minimums (hard-constraint coherence)

**Failure mode:** the existing hard guarantee — first leg carries
`quote_spread_min_depth` within `quote_spread_max` — is defeated downstream if that
quantity is below the venue's `min_qty` / `min_notional`: the tight order is rejected at
validation (`mm/execution/rules_gateway.py` path) and the live book loses its guaranteed
near-touch presence while the optimizer believes it is there.

**File / function:** `mm/mpc/mpc_tensor_batched.py`, `VenueOptimizerParams.validate()`
(it already receives the venue context) plus the call site in
`mm/strategies/mpc_engine.py` where `binding.local.min_qty` is available
(`_venue_optimizer_params` resolves it today). Add:

```python
if quote_spread_min_depth is not None and spec.min_qty is not None:
    if quote_spread_min_depth < spec.min_qty:
        raise ValueError("quote_spread_min_depth must be >= venue min_qty "
                         "so the guaranteed near-touch quote is placeable")
# analogous check against min_notional using a reference price if configured
```

Fail at startup, not silently at runtime.

**Effort:** ~2–3 hours including tests.

**Prerequisites:** none.

---

## P7 — Keep the negative-objective gate on the *pure* return component (contract)

`best_risk_adjusted_return()` in `mm/strategies/negative_objective_policy.py` reads
`components["risk_adjusted_return"]` and falls back to `objective_value`. After P2/P3 the
total `J` includes the book term, so the fallback would be wrong: a positive book term
could mask a negative expected return and keep the bot quoting at a loss.

**Change:** none to the policy itself **if** P2/P3 keep `risk_adjusted_return` as the
pure return CE (which they must — this is the contract). Add a regression test in
`tests/unit/test_negative_objective_policy.py`: with `book_quality_weight > 0` and a
configuration where `J > 0` but `risk_adjusted_return < 0`, the floor/zero policy still
triggers.

**Effort:** ~1–2 hours (test only).

**Prerequisites:** P2.

---

## P8 (optional, lowest priority) — Terminal-inventory risk term

Inventory risk is currently represented only indirectly (side-aware liquidation marking,
capacity bounds, imbalance skew). A direct, dimensionless option:

```text
inv_frac_H = Q_H * ref_H / W_0                # terminal inventory as fraction of wealth
J_inv = − mean_i( inv_frac_{H,i}² )
J = CE_beta(R) + w_book * J_book + w_inv * J_inv
```

computed from the already-available `inv_t` / `ref` at the last rollout step in both
closures — a few lines each.

**Caution / why it is last:** the codebase deliberately removed inventory-penalty knobs
(`inv_coef`, `target_inventory`, `inventory_threshold_penalty_weight` are *rejected* keys
in `_REMOVED_VENUE_OPTIMIZER_KEYS`, `mm/strategies/mpc_engine.py`). Reintroducing one
should be a conscious decision, under a new clearly-scoped key
(`terminal_inventory_weight`), defaulting to `0.0`. If "less risky" can be achieved by
raising the existing `return_risk_aversion_beta` plus the P4 hard ratio, prefer that — it
adds no new weight to select.

**Effort:** ~0.5 day. **Prerequisites:** P2 component-naming contract.

---

## Recommended order and tuning workflow

| # | Proposal | Disruption | Effort | Fixes |
|---|----------|-----------|--------|-------|
| 1 | P1 capping fix | none (bug-class fix) | hours | tight quote silently dropped |
| 2 | P6 min-depth validation | none (startup check) | hours | guaranteed quote unplaceable |
| 3 | P2 book term, single venue | additive, default-off | ~1 day | no book shaping at all |
| 4 | P5 config plumbing | additive | hours | — |
| 5 | P7 gate contract test | none | hours | loss-masking regression |
| 6 | P3 book term, joint | additive | ~1 day | parity for liquidity sourcing |
| 7 | P4 hard balance ratio | additive, default-off | ~1 day | worst-case lopsidedness |
| 8 | P8 inventory term | additive, contentious | ~0.5 day | optional direct inventory risk |

**Tuning `w_book`:** every objective component is persisted per frame in
`mm_optimizer_component_facts` (lineage-linked), and `mm/persistence/frame_replay.py` +
`mm/application/frame_simulator.py` already replay persisted frames offline. So the
weight can be swept offline against recorded frames — compare realized
`risk_adjusted_return` vs `book_tightness`/`book_balance`/`book_depth` per candidate
weight — before any live change. Prerequisite data: a few days of persisted frames with
the default `book_quality_weight: 0.0` build (P2 deployed dormant) so the components are
already being scored and logged.
