# MPC Market-Making Optimization: Objective, Mechanism, and Business Goals

This document describes the implemented MPC controller: business intent, mathematics, data flow, configuration surface, diagnostics, and known gaps. It is written for external expert review. It records what the code does and what is uncertain; it does not prescribe tuning values or a preferred redesign.

Related implementation references:

- `mm.strategies.mpc_engine.MpcStrategyEngine`
- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched`
- `mm.mpc.joint_optimizer.optimize_joint_with_diagnostics` (liquidity-sourcing mode)
- `mm.mpc.objective_terms` (book quality, validity, viability availability, dual λ, CE normalization)
- `docs/viability_term.md` (conceptual motivation for availability vs CE)
- `docs/validity_term.md` (validity gate design)
- `mm.mpc.reward_tensor` (risk-adjusted return)
- `docs/old/mpc-optimizer-implementation-details.md` (rollout timing, decoder, joint vs local-only differences)

---

## 1. Business goals (operational intent)

The controller targets **persistent market making**: quoting over many intervals rather than optimizing a single trade.

### 1.1 Two-sided quote availability

Operational intent includes:

- Bid and ask liquidity when wallet capacity allows.
- Wide spreads are not forbidden by the core return term; tightness enters only via **validity** when `validity.penalty_scale > 0`.
- Depleting quote (IRT) or base (USDT) on one side removes hard capacity on that side and ends two-sided quoting in simulation.

Implemented mechanisms that relate to this intent:

| Mechanism | Type | Role |
| --- | --- | --- |
| `bid_max = cash/mid`, `ask_max = inventory` | Hard | Cannot simulate quotes beyond wallet totals in rollout |
| `−μ · max(0, B_min − B)` | Soft (optional) | **Validity** — penalizes poor h=0 book shape (`validity` block) |
| `−λ · 𝒮` | Soft (optional) | **Viability** — penalizes low terminal two-sided **availability** along MC paths (`viability` block); `λ` adapts online |

`w_book` and `w_inv` / `inventory_zone` were **removed**. Book scoring still exists via `book_quality.soft_scoring`, but enters `J` only through the validity gate (§7.3–7.4), not as a direct reward term.

Current capacity decoder sets `bid_min = ask_min = 0`. There is **no hard minimum quote per side** at the capacity layer. `cumulative_qty_floor_at_spread_max` affects spread-structure decode at `spread_max`, but mins are clamped to feasible capacity and can be zero when a side has no free balance.

### 1.2 Profitability

Profitability enters through **risk-adjusted horizon return** on simulated mark-to-market wealth, with maker fees and a differentiable soft-fill model. The optimizer maximizes CE (see §7.1). A separate **negative-objective policy** (§9) may alter the action when CE is negative; that policy is outside `J`.

### 1.3 Availability vs profitability (design split)

The objective deliberately separates:

- **CE (§7.1)** — risk-adjusted **return** on simulated wealth (price + fill uncertainty).
- **Validity (§7.3)** — is the **h=0 quote shape** acceptable (spread/depth/balance/uniformity score)?
- **Viability (§7.5–7.6)** — after simulating orderflow and fills over the horizon, is **two-sided quoting capacity** still above target?

Viability uses **total** wallet balances (`free + locked`, see §3.1). It does not model exchange min-notional, tick size, or free-vs-locked separately. Registry allocation caps, risk limits, and physical exchange balances remain outside `J`.

### 1.4 Outside the optimizer objective

Not part of `J`:

- Tick, lot, min-notional validation
- Net-delta and transition risk gates
- Order reconcile, router, venue API errors
- Stale-frame skipping, private-data freshness rules
- Virtual vs physical ledger selection and allocation breach halts

The MPC outputs a **target book**; downstream layers decide live orders.

---

## 2. End-to-end decision pipeline

```text
Market + private snapshots
  → JointStrategyFrameAssembler (subtract own orders from public book)
  → MpcStrategyEngine.decide
      → State (inventory, cash, books, VWAPs, optional global fields)
      → update online flow / price models (prequential score + update)
      → MonteCarloMPCTensorBatched.optimize_with_diagnostics(state)
          OR optimize_joint_with_diagnostics (liquidity sourcing)
      → Adam on sigmoid parameters, maximize J
      → build_joint_target_book_from_sigmoid
      → StrategyDecision + JointTargetBook
  → negative-objective policy (optional)
  → risk engine
  → OrderManager reconcile → venue
```

Branch selection depends on `liquidity_sourcing` and `reference_price_source`; not all deployments use the same branch (see §15).

Frames may be skipped when market or private data exceeds configured age limits.

---

## 3. State vector and wallet sourcing

Optimizer input: `mm.mpc.mpc_state.State`.

| Field | Role |
| --- | --- |
| `mid` | Quote anchor (quote per base) |
| `ob_rel` | Market-only book as `(spread, qty)` within `spread_max` |
| `inventory`, `cash` | Base and quote totals used in rollout |
| `buy_vwap`, `sell_vwap` | VWAPs at inventory-derived depth |
| `buy_vol`, `sell_vol` | Market-flow observations for models |
| `global_*` | Second venue when liquidity sourcing is active |

### 3.1 How inventory and cash are populated

`MpcStrategyEngine._state_from_frame()` prefers venue ledger when `ledger.last_snapshot_ms > 0`; otherwise private snapshot balances. Diagnostics expose `state.local_inventory`, `state.local_cash`, wallet leg views, and `ledger_source` (`physical` or virtual).

**Blind spot:** Optimizer state can disagree with exchange UI balances when:

- Ledger snapshots stop updating or lag wall clock.
- `ledger_source` is virtual (allocation pool) vs full exchange wallet.
- Locked order collateral is represented differently in `free` vs `total` vs order-manager locks.
- Frame assembly uses stale private events while market is fresh.

Observed in production data: MPC diagnostics have shown `0/0` balances while operators report non-zero exchange balances; conversely diagnostics have shown heavily skewed wallets that may not match post-manual rebalance exchange state. Any expert review should treat **diagnostics state** and **exchange state** as distinct signals until reconciled.

### 3.2 Wealth marking in rollout

Per-step wealth uses inventory-aware marks: bid VWAP when short inventory, sell VWAP when long. Mark choice affects `W_0`, `R_i`, CE, and indirectly gradient direction.

---

## 4. Action representation

Sigmoid-mixture cumulative depth per side on a fixed spread grid:

```text
spread_grid = [spread_step, 2·spread_step, …, spread_max]
D_side(spread) = Σ_i L_i · sigmoid(k_i · (spread − x0_i))
effective_depth(spread) = max(0, D(spread) − D(0))
```

Horizon `H` stacks `H` venue actions (typically same initializer tiled across steps). Optimizer variables are raw logits/centers per component plus depth logit(s).

Spread-structure decode (`book_quality.hard_structure`) maps logits to sizes with constraints on first-touch spread and depth at `spread_max`.

---

## 5. Hard structural constraints (not in `J`)

### 5.1 Side capacity

```text
bid_max_qty = min(optional_quote_size_max_per_side, cash / mid)
ask_max_qty = min(optional_quote_size_max_per_side, inventory)
bid_min_qty = 0
ask_min_qty = 0
```

Zero `cash` → zero bid capacity in simulation. Zero `inventory` → zero ask capacity. This is the main hard guard against quoting a depleted side; it does not guarantee the other side stays quoted.

### 5.2 Decode geometry

`book_quality.hard_structure` can cap first-touch spread sum and minimum depth at the first center. `cumulative_qty_floor_at_spread_max` is a lot-size floor in structural sizing at `spread_max`, not an independent per-side minimum at the capacity layer.

---

## 6. Monte Carlo rollout

### 6.1 Closure structure

Each Adam closure:

1. Sample (or reuse) `n_samples` paths from market-flow and price-residual models.
2. Roll out `horizon` steps; accumulate mark-to-market `ΔW`.
3. Form `J`; backpropagate `loss = −J`.

`fixed_mc_samples = true` reuses the same random draws across all epochs in one optimize call.

### 6.2 Single-venue rollout

Per sample, per step `h`:

1. Decode action → effective depths.
2. `soft_fill_batched` against market-only book + sampled flow.
3. Update `inventory`, `cash` (fees applied).
4. Evolve public book.
5. Recompute VWAPs at `|inventory|`; add price residual.
6. Accumulate `ΔW` with inventory-aware marking.

```text
R_i = (Σ_h ΔW_i,h) / W_0
W_0 = max(1e-8, cash_0 + inventory_0 · mark_0)
```

### 6.3 Soft fill model

Differentiable proportional-fill approximation:

- Sells hit bids; buys hit asks.
- Share at level `i` depends on own vs public size; `back_of_queue` adds next-level public depth to denominator (more conservative than `pro_rata`).
- `soft_fill_temp` and `sigmoid` / `hardsigmoid` activation.

```text
cash_delta = ask_fill_notional · (1 − fee) − bid_fill_notional
inv_delta  = bid_fill_qty · (1 − fee) − ask_fill_qty
```

This is not a discrete queue simulator, not adverse-selection modeling, and not partial-fill correlation across levels.

### 6.4 Joint / liquidity-sourcing rollout

Local and global books evolve in parallel; consolidated wealth drives CE. VWAP timing differs from single-venue rollout (eaten-book effect can lag one step in joint mode — see old implementation doc). Joint objective omits several regularizers present in the legacy single-venue branch (§15).

---

## 7. Scalar objective `J`

The optimizer **maximizes** `J`. Adam minimizes `loss = −J`.

### 7.0 Combined form

**Local-only optimizer:**

```text
J = CE_eff − μ · max(0, B_min − B) − λ · 𝒮
```

**Joint / liquidity-sourcing optimizer** (consolidated CE; per-venue validity and viability):

```text
J = CE_eff
    − μ_local · max(0, B_min,local − B_local) − λ_local · 𝒮_local
    − μ_global · max(0, B_min,global − B_global) − λ_global · 𝒮_global
```

| Symbol | Config / state | Code component key |
| --- | --- | --- |
| `CE_eff` | `objective_normalize` | `risk_adjusted_return` |
| `B` | `book_quality` + `validity.min_score` | `validity_book_score` (unweighted); penalty in `validity_penalty` |
| `μ` | `validity.penalty_scale` | fixed per venue |
| `𝒮` | `viability` block + dual `λ` | `viability_penalty_raw` (unweighted); penalty in `viability_penalty` |
| `λ` | `exp(θ)` from persisted `θ` | `viability_lambda` (closure); `viability_lambda_learned` (post-update) |

**Removed:** `w_book`, `w_inv`, `inventory_zone`, and all `inventory_zone_*` diagnostics keys.

**Evaluation timing:**

| Term | When evaluated | Differentiable w.r.t. action? |
| --- | --- | --- |
| CE | Terminal closure over full horizon rollout | Yes (via fills, marks, fees) |
| Validity `B` | **Horizon step 0** decoded action only | Yes (h=0 action slice) |
| Viability `𝒮` | **Terminal** wallet after simulated horizon fills | Yes (via rollout path) |

During one `optimize()` call, **`λ` (via `θ`) is fixed** for all Adam epochs. It updates **once** after the decision completes (§7.6).

---

### 7.1 Risk-adjusted return (CE)

#### Per-path horizon return

For Monte Carlo path `i`, horizon steps `h = 0 … H−1`:

```text
ΔW_i,h = wealth after step h − wealth before step h   (inventory-aware marking)
R_i    = (Σ_h ΔW_i,h) / W_0
W_0    = max(1e-8, cash_0 + inventory_0 · mark_0)
```

`mark_0` uses bid VWAP when short inventory, sell VWAP when long, else quote anchor (see §3.2).

#### Entropic certainty equivalent

Per-path horizon returns `R_i`, batch size `N`, `β = return_risk_aversion_beta`:

```text
CE = min_i(R_i) − (1/β) · logsumexp_i(−β · (R_i − min_i(R_i))) + (1/β) · ln(N)
```

Properties:

- Soft minimum / entropic certainty equivalent over paths.
- Sensitive to tail paths, not only mean return.
- Units: dimensionless return after dividing by `W_0`.
- When all paths have zero tradeable capacity or zero `ΔW`, `R_i = 0` ∀i → **CE = 0**.

CE is computed from **simulated** fills and model-sampled flows, not realized live PnL.

---

### 7.2 CE normalization (`objective_normalize`)

When `objective_normalize.enabled`:

```text
CE_raw  = CE from §7.1 (before scaling)
s       = EWMA scale from prior decisions (fixed for entire Adam loop on this frame)
CE_norm = CE_raw / s
J       ← J − CE_raw + CE_norm
```

If `s ≤ 0` on first use, `s ← max(|CE_raw|, 1e-12)` for that frame.

EWMA update after the decision completes (`ce_ewma_decay = decay`):

```text
s_new = decay · s + (1 − decay) · |CE_raw|     if s > 0 and |CE_raw| > 0
s_new = max(|CE_raw|, 1e-12)                    if s ≤ 0 (bootstrap)
```

Diagnostics: `risk_adjusted_return_raw`, `risk_adjusted_return` (= `CE_norm`), `objective_ce_ewma_scale`.

**Asymmetry:** Only CE is EWMA-normalized. Validity uses fixed `μ`; viability uses adaptive `λ`. No joint normalization across terms.

---

### 7.3 Validity term (book-shape gate)

Validity penalizes a **low book-quality score** at the **current** (h=0) decoded action. It is **not** a reward for high `B`; only shortfall below `B_min` is penalized.

```text
validity_shortfall = max(0, B_min − B)        # B_min = validity.min_score
validity_penalty   = −μ · validity_shortfall  # μ = validity.penalty_scale
```

If `validity` is absent, `penalty_scale = 0`, or `book_score` is unavailable, the term is skipped.

`B` is the **book quality score** from `book_quality_score()` (§7.4). Diagnostics store `validity_book_score` (= `B`) and `validity_penalty_raw` (= shortfall).

---

### 7.4 Book quality score `B` (feeds validity only)

`B ∈ [0, 1]` at **horizon step 0**. Let `d_bid_touch`, `d_ask_touch` be effective quoted depth at touch spread `touch = spread_max · touch_spread_fraction`, and `q_bid`, `q_ask` cumulative depth at `spread_max`:

```text
presence   = min(d_bid_touch, d_ask_touch) / (q_ref + min(d_bid_touch, d_ask_touch))   # q_ref = presence_reference_qty

tightness  = exp(−(x0_bid,0 + x0_ask,0) / tightness_spread_scale)   # first sigmoid centers

q_bid_$    = q_bid · mid
q_ask_$    = q_ask · mid
balance    = 1 − |q_bid_$ − q_ask_$| / (q_bid_$ + q_ask_$ + ε)

uniformity = ½ · (U(marginals_bid) + U(marginals_ask))
U(m)       = 1 − Var(m) / (Mean(m)² + Var(m) + ε)

B = (v_presence·presence + v_tightness·tightness + v_balance·balance + v_uniformity·uniformity)
    / (v_presence + v_tightness + v_balance + v_uniformity)
B = clamp(B, 0, 1)
```

`measure = "quoted"` uses own decoded depth only. `measure = "combined"` adds market cumulative depth at each grid level.

Sub-scores are stored in diagnostics (`book_presence`, `book_tightness`, `book_balance`, `book_uniformity`) but **do not** enter `J` unless `validity` maps `B` into a penalty.

---

### 7.5 Viability term (availability shortfall)

Viability encodes **two-sided quoting capacity** after simulating orderflow and fills. It replaces the removed inventory-zone penalty (`w_inv · Z̄`).

Conceptual target (see `docs/viability_term.md`): maximize CE subject to maintaining enough **bid and ask capacity** to place minimum viable size `q_min`. The implementation uses a **soft availability** functional and a **Lagrangian penalty** with adaptive `λ` (§7.6).

#### 7.5.1 Wallet inputs

At each viability snapshot along a rollout path:

| Variable | Code field | Meaning |
| --- | --- | --- |
| `b_t` | `inventory` | Total base balance (USDT), `max(0, ·)` |
| `q_t` | `cash` | Total quote balance (IRT), `max(0, ·)` |
| `p_t` | `mid` | Quote per base, `max(1e-12, ·)` |

Totals use **reported wallet total** (`free + locked`) from ledger/snapshot (§3.1). **Free-only balances are not used.**

Config (`viability` block):

| Key | Symbol | Default | Role |
| --- | --- | --- | --- |
| `min_viable_quote` | `q_min` | `50000` | Minimum operational quote-notional required on **each** side |
| `margin_buffer` | `δ` | `5000` | Sigmoid scale for soft margins (same units as margins: quote currency) |
| `availability_target` | `A*` | `0.99` | Target terminal availability |
| `lambda_init` | `λ_0` | `0` | Initial multiplier (maps to `θ_0`, §7.6) |
| `lambda_learning_rate` | `η` | `0.02` | Dual learning rate |
| `shortfall_tolerance` | `ε` | `0.05` | Parsed and stored; **not used** in the current dual update rule |

**Removed keys** (rejected at load): `min_quote_qty`, `margin_sharpness`, `inventory_zone`, `w_inv`, `lambda_leakage`, etc.

#### 7.5.2 Quote-notional margins (both sides in quote currency)

```text
m_bid(t) = q_t − q_min
m_ask(t) = b_t · p_t − q_min
```

Interpretation:

- `m_bid > 0` → total quote exceeds minimum required to fund bids.
- `m_ask > 0` → base notional exceeds minimum required to fund asks.
- Margins are in **quote currency** (IRT for USDT_IRT), not unitless ratios.

#### 7.5.3 Soft side availability

Logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`:

```text
a_bid(t) = σ(m_bid(t) / δ)
a_ask(t) = σ(m_ask(t) / δ)
Ã(t)     = a_bid(t) · a_ask(t)     ∈ [0, 1]
```

`Ã` is **multiplicative**: one dead side drives availability to ~0 even if the other side is ample.

**Reference:** with `δ = 5000` IRT, `m = 0` gives `σ(0) = 0.5` per side, so `Ã = 0.25` at exactly `q_min` on both legs. Large positive margins saturate `σ` → 1.

Diagnostics (batch means over path/time unless noted):

| Key | Maps to |
| --- | --- |
| `viability_bid_margin` | mean `m_bid` |
| `viability_ask_margin` | mean `m_ask` |
| `viability_base_notional` | mean `b · p` |
| `viability_step_availability` | mean `Ã` |
| `viability_min_availability` | mean terminal `Ã_H` (see below) |

#### 7.5.4 Rollout path snapshots

During each Adam closure, for each MC sample, the rollout records wallet/mid snapshots:

```text
index 0:  (b_0, q_0, p_0)     initial wallet before horizon step 0
index 1:  after step 0 soft_fill + book evolution
…
index H:  after step H−1       terminal post-fill state
```

Path tensors have shape `(N_paths, H+1)`. Snapshots after each step use **post-fill** `(inventory, cash, mid)` from `soft_fill_batched` and evolved orderbook mids.

Viability reads **terminal availability only**:

```text
Ã_H^(i) = Ã(b_H^(i), q_H^(i), p_H^(i))     last column of path i
```

**Design note:** Shortfall uses `Ã_H`, not `min_h Ã_h` along the path. Simulated fills over the horizon can **improve** terminal availability relative to an early depleted step; the path minimum is intentionally **not** used (see `viability_path_shortfall` docstring).

#### 7.5.5 Per-path and batch shortfall

```text
S^(i) = max(0, A* − Ã_H^(i))
𝒮     = (1/N) Σ_i S^(i)        # viability_penalty_raw / viability_shortfall
```

```text
viability_penalty = −λ · 𝒮
```

Gradients flow: action → fills → terminal `(b_H, q_H, p_H)` → `Ã_H` → `𝒮` → `J`.

---

### 7.6 Dual optimization for `λ` (log-space)

Viability uses a **primal–dual** pattern: `λ` is not hand-tuned each frame like `w_inv`; it is **state** on the `MonteCarloMPCTensorBatched` instance, persisted across controller decisions.

#### 7.6.1 Multiplier parameterization

```text
λ = exp(θ)     θ ∈ ℝ
θ_0 = ln(max(λ_init, ε_floor))     ε_floor = 1e-300
```

If `λ_init = 0`, then `θ_0 = ln(ε_floor)` and `λ = 0` (no viability penalty until `θ` increases).

Implementation: `viability_theta_from_lambda_init`, `viability_lambda_from_theta` in `objective_terms.py`. State: `mpc._viability_theta_local`, `mpc._viability_theta_global` (joint mode).

#### 7.6.2 Closure vs between-decision updates

**Inside Adam closure** (each epoch, each `optimize()` call):

```text
λ_closure = exp(θ_t)              # θ_t fixed for all epochs
J        ← J − λ_closure · 𝒮     # combine_viability_term
```

`θ` does **not** update during Adam; only the action vector changes.

**After `optimize()` completes** (once per controller decision):

```text
Ā_H = mean_i(Ã_H^(i))            # from diagnostics: viability_min_availability
gap = A* − Ā_H                   # signed availability gap
θ_{t+1} = θ_t + η · gap
λ_{t+1} = exp(θ_{t+1})
```

Implementation: `update_viability_theta`. Joint mode updates local and global `θ` separately from `viability_min_availability` and `global_viability_min_availability`.

**Dual dynamics:**

| Situation | `gap` | Effect on `θ`, `λ` |
| --- | --- | --- |
| Terminal availability below target (`Ā_H < A*`) | `> 0` | `θ` increases → `λ` increases → stronger viability penalty next decision |
| Terminal availability above target (`Ā_H > A*`) | `< 0` | `θ` decreases → `λ` decreases → weaker penalty next decision |
| At target (`Ā_H = A*`) | `0` | `θ` unchanged |

There is **no** `ε` tolerance in the update rule (unlike classical `λ ← λ + η(S − ε)` on shortfall). `shortfall_tolerance` in config is currently **inert**.

#### 7.6.3 Diagnostics: which `λ` to read

| Key | When | Meaning |
| --- | --- | --- |
| `viability_lambda` | After optimize stamp | `λ` used in **this** decision's penalty (reconciled from `|penalty|/shortfall` when needed) |
| `viability_lambda_theta` | After optimize stamp | `θ` at start of this decision (before dual update) |
| `viability_lambda_learned` | After optimize stamp | `exp(θ_{t+1})` for **next** decision |
| `viability_lambda_theta_learned` | After optimize stamp | `θ_{t+1}` |

`stamp_viability_dual_diagnostics()` runs on initial/final/best snapshots after the dual update, overwriting closure-side `viability_lambda` keys so they match the multiplier that shaped `viability_penalty`.

**Identity (per decision):**

```text
viability_penalty ≈ − viability_lambda · viability_penalty_raw
```

Use `viability_lambda_learned` when correlating the **next** tick's penalty strength with the prior update.

#### 7.6.4 Constrained optimization interpretation

The implemented penalty is the Lagrangian form of:

```text
maximize_π  CE_eff(π)
subject to  E[Ã_H] ≥ A*     (softened via 𝒮)
```

`λ` rises when the constraint is violated (low terminal availability) and falls when the constraint is slack. Unlike fixed `w_inv`, the trade-off against CE **adapts**, but:

- Only **terminal** availability enters `𝒮` (not path-min).
- `λ` is fixed during the inner Adam solve (no inner dual loop).
- CE may still dominate if `λ` is small or `𝒮` is small (e.g. saturated sigmoid with miscalibrated `q_min`).

---

### 7.7 Combined objective and component keys

```text
J = CE_eff − μ · max(0, B_min − B) − λ · 𝒮        # local-only
```

Representative diagnostics keys:

| Key | Value stored |
| --- | --- |
| `risk_adjusted_return` | `CE_eff` |
| `risk_adjusted_return_raw` | `CE_raw` (if normalization on) |
| `objective_ce_ewma_scale` | EWMA scale `s` |
| `validity_penalty` | `−μ · max(0, B_min − B)` |
| `validity_penalty_raw` | `max(0, B_min − B)` |
| `validity_book_score` | `B` |
| `viability_penalty` | `−λ · 𝒮` |
| `viability_penalty_raw` | `𝒮` |
| `viability_shortfall` | same as `𝒮` in tensor components |
| `viability_min_availability` | `Ā_H` |
| `viability_lambda` / `viability_lambda_learned` | see §7.6.3 |

Joint mode prefixes global venue keys with `global_`.

---

### 7.8 Contribution fractions

From `compute_contribution_fractions()` on **weighted** terms only:

```text
fraction_k = |component_k| / Σ_j |component_j|
```

Keys in `OBJECTIVE_CONTRIBUTION_KEYS`:

- `risk_adjusted_return`
- `validity_penalty`
- `viability_penalty`
- `global_validity_penalty`
- `global_viability_penalty`

Raw sub-scores (`viability_penalty_raw`, `book_presence`, etc.) are **excluded**.

**Interpretation constraints:**

- Fractions are shares of **absolute weighted terms in `J`**, not economic PnL.
- If `CE_eff ≈ 0`, viability/validity can show ~100% share while absolute penalty remains tiny.
- `viability_lambda` in closure components should match `|viability_penalty| / |viability_penalty_raw|` when `𝒮 > 0`.

---

### 7.9 Worked viability example (illustrative units)

Take `q_min = 5×10⁶` IRT, `δ = 5×10⁵` IRT, `A* = 0.9`, terminal wallet after rollout:

| `q_H` (IRT) | `b_H` (USDT) | `p_H` | `m_bid` | `m_ask` | `Ã_H` (approx.) | `S^(i)` |
| --- | --- | --- | --- | --- | --- | --- |
| 18×10⁶ | 0.005 | 170k | 13×10⁶ | −8.5×10⁵ | ~0 | 0.9 |
| 10×10⁶ | 45 | 170k | 5×10⁶ | 2.65×10⁶ | ~0.88 | 0.02 |
| 10×10⁶ | 108 | 170k | 5×10⁶ | 13.4×10⁶ | ~0.99 | 0 |

With `λ = 0.01`, `𝒮 = 0.9` → `viability_penalty = −0.009`. With `λ = 1`, same shortfall → `−0.9`.

This shows why **`q_min` calibration in quote currency** dominates viability behavior: wrong `q_min` saturates `𝒮` or leaves it near zero regardless of action.

---
## 8. Optimization procedure

- Adam on flattened horizon action vector.
- `loss = −J`.
- `adam_epochs`, `adam_grad_clip`, optional `warm_start`.
- Best epoch = lowest loss; target book from best clamped vector (`last_eval_x`, not post-update `x_torch`).

Diagnostics count nonfinite loss/gradient events. No guarantee that Adam finds global optimum; nonconvexity from fills, sigmoids, and finite samples.

---

## 9. Post-optimizer policy

If `best_risk_adjusted_return < 0`, `negative_objective_action` may select:

| Mode | Behavior |
| --- | --- |
| `optimizer_action` | Use optimized curves |
| `floor_action` | Minimum feasible quote from decoder + capacity |
| `zero_action` | Zero depth |

This gate uses optimizer diagnostics, not live PnL. Interaction with validity, viability `λ`, and CE normalization is configuration-dependent.

---

## 10. Configuration surface (example deployment)

Illustrative values from MPC diagnostics snapshots (not authoritative; live registry may differ):

| Parameter | Example value |
| --- | --- |
| `kind` | `monte_carlo_mpc_tensor_batched` |
| `horizon` | 4 |
| `n_samples` | 64 |
| `spread_max` / `spread_step` | 200 / 20 |
| `fixed_mc_samples` | true |
| `adam_epochs` | 40 |
| `reference_price_source` | `local` or `global` |
| `objective_normalize.enabled` | true |
| `objective_normalize.ce_ewma_decay` | 0.9 |
| `validity.penalty_scale` | e.g. 0 (disabled) or >0 |
| `validity.min_score` | e.g. 0.5 — book score floor `B_min` |
| `viability.min_viable_quote` | e.g. 5×10⁶ IRT — `q_min` |
| `viability.margin_buffer` | e.g. 5×10⁵ IRT — sigmoid scale `δ` |
| `viability.availability_target` | e.g. 0.9 — `A*` |
| `viability.lambda_init` | e.g. 0.001 — initial `λ_0` |
| `viability.lambda_learning_rate` | e.g. 0.1 — dual step `η` |
| `return_risk_aversion_beta` | e.g. 0.1 |
| `cumulative_qty_floor_at_spread_max` | decoder cumulative qty floor at `spread_max` |
| `cumulative_qty_cap_at_spread_max` | optional decoder cumulative qty cap |
| `book_quality.soft_scoring.presence_reference_qty` | presence sub-score reference |
| `book_quality.soft_scoring.tightness_spread_scale` | tightness sub-score spread scale |
| `reservation_skew.max_abs_skew` | learnable reservation skew bound |
| `execution_model` maker fee | e.g. 0.0015 |
| `soft_fill_queue_model` | `back_of_queue` |

Separate `local_optimizer` and `global_optimizer` blocks exist in joint mode.

### 10.1 Optimizer parameter names

| Key | Role |
| --- | --- |
| `cumulative_qty_floor_at_spread_max` | Decoder total base qty floor at outer spread |
| `cumulative_qty_cap_at_spread_max` | Optional decoder total base qty cap |
| `book_quality.soft_scoring` | Sub-score weights and references for `B` (§7.4) |
| `validity` | `penalty_scale` (`μ`), `min_score` (`B_min`) |
| `viability` | `min_viable_quote`, `margin_buffer`, `availability_target`, `lambda_init`, `lambda_learning_rate` |
| `presence_reference_qty` | Presence sub-score reference qty |
| `tightness_spread_scale` | Tightness sub-score spread scale |
| `reservation_skew` | Learnable bid/ask curve skew block |
| `max_abs_skew` | Bound on reservation skew parameter |

Legacy keys (`w_inv`, `w_book`, `inventory_zone`, `min_quote_qty`, `quote_size_min_per_side`, etc.) are **rejected** at parse time. Registry Postgres configs were migrated via `mm/tools/migrate_optimizer_config_keys.py`.

---

## 11. Coupling: CE scale vs validity `μ` vs viability `λ`

### 11.1 What is implemented

| Quantity | Scale / units | Adaptive? |
| --- | --- | --- |
| CE (`risk_adjusted_return`) | Wealth-normalized return; optionally ÷ EWMA `|CE_raw|` | CE only, via `objective_normalize` |
| `−μ · max(0, B_min − B)` | `B ∈ [0,1]` × fixed `μ` | No — `validity.penalty_scale` |
| `−λ · 𝒮` | `𝒮 ∈ [0, A*]` × `λ = exp(θ)` | **Yes** — dual update on `θ` after each decision (§7.6) |

`μ` is a **static config scalar**. `λ` adapts to terminal availability gap `A* − Ā_H` but is **fixed during** each Adam solve.

### 11.2 Why tuning is non-obvious

CE magnitude moves with fees, spread, fill rate, `horizon`, `n_samples`, `β`, wealth, EWMA state, and wallet fidelity in `State`.

Viability magnitude moves with:

- `q_min` and `δ` (sigmoid calibration in **quote currency**)
- Terminal simulated wallet after fills (`b_H`, `q_H`, `p_H`)
- `A*` and current `λ`
- Whether totals include locked funds the exchange treats as unavailable for new quotes

Validity magnitude moves with h=0 decoded shape only; it does not see horizon path depletion.

There is **no single `μ` or `λ_init`** that yields a constant economic trade-off against CE across regimes without recalibrating `q_min`, `δ`, and dual dynamics.

### 11.3 Illustrative viability arithmetic (not a tuning guide)

With `A* = 0.9`, `𝒮 = 0.15` (mean terminal shortfall), and `|CE_eff| = 1`:

| `λ` | `\|−λ · 𝒮\|` | Share of `\|J\|` terms (CE + viability) |
| --- | --- | --- |
| 0.001 | 0.00015 | ~0.015% |
| 0.01 | 0.0015 | ~0.15% |
| 0.1 | 0.015 | ~1.5% |
| 1.0 | 0.15 | ~13% |

If `|CE_eff| = 0.01` with `λ = 0.1`, viability share rises to ~60%. If `CE_eff = 0`, viability can dominate **relative** contribution fractions while absolute penalty stays small.

**Dual steady state (sketch):** if `Ā_H` persistently sits below `A*`, `θ` drifts up and `λ` grows until the optimizer quotes more conservatively and terminal availability rises. If `q_min` is set far above realistic margins, `𝒮` stays near `A*` regardless of action → dual may raise `λ` without improving `Ā_H`.

### 11.4 Design space not implemented

- Inner dual loop during Adam (update `θ` each epoch)
- `shortfall_tolerance` in the dual update (`λ ← λ + η(𝒮 − ε)`)
- Path-min availability `min_h Ã_h` instead of terminal `Ã_H`
- Free-balance margins separate from locked collateral
- `μ` or `λ` proportional to EWMA `|CE_raw|`
- Hard minimum spread or minimum two-sided size in `J` (only soft validity/viability)

See `docs/viability_term.md` for design alternatives discussed but not wired.

---

## 12. Diagnostics map

| Field | What it reflects | Caveats |
| --- | --- | --- |
| `optimizer.objective.final.value` | Best `J` | Simulation objective, not realized PnL |
| `optimizer.objective.final.components` | Weighted terms | See §7.7–7.8 |
| `optimizer.objective.contribution_fractions` | Relative \|term\| shares | Misleading when terms are near zero |
| `optimizer.objective.best.expected_total_return` | Mean `R_i` over paths | Simulated |
| `state.local_inventory`, `local_cash` | Wallet fed to MPC | May lag exchange; virtual vs physical |
| `state.local_*_wallet` | Free/total/locked legs | Viability uses **total**; capacity uses same totals |
| `validity_book_score` | `B` at h=0 | Unweighted; penalty via `validity_penalty` |
| `viability_min_availability` | `Ā_H` (mean terminal `Ã`) | Drives dual update (§7.6) |
| `viability_penalty_raw` | `𝒮` | Unweighted shortfall |
| `viability_lambda_learned` | `λ` for **next** decision | Tracks dual state; `viability_lambda` should match prior tick's learned |

Postgres tables: `mm_optimizer_facts`, `mm_optimizer_component_facts`, `mm_events` diagnostics payloads.

---

## 13. Business goal → mechanism mapping (with gaps)

| Intent | Implemented linkage | Gap / blind spot |
| --- | --- | --- |
| Profitable quoting | Maximize simulated CE | Sim ≠ live; model error; fee/slippage gaps |
| Both sides quoted | Hard capacity; optional validity on `B` | `μ=0` → no soft book-shape gate; mins are 0 |
| Don't run out of IRT/USDT | Capacity zeros depleted side; viability on terminal `Ã_H` | Uses **total** balance; locked funds can hide free depletion |
| Wide spread OK | No BBO term in CE | Tightness only if validity enabled |
| Maintain quoting capacity | `−λ · 𝒮` on simulated paths | `q_min`/`δ` calibration; terminal-only `Ã_H`; small `λ` vs CE |
| Safe live operation | Risk + reconcile | Outside `J`; can reject MPC output entirely |

---

## 14. Blind spots, limitations, and open gaps

This section lists areas the documentation and diagnostics do not fully resolve. External review should treat these as first-class concerns.

### 14.1 State and ledger fidelity

- Optimizer wallet may be **virtual allocation**, not full exchange wallet (`ledger_source`).
- Ledger snapshot pipeline may stall; optimizer may optimize on **zeros or stale balances** while exchange holds funds.
- Manual exchange rebalance may not appear in diagnostics until ingest catches up.
- `inventory`/`cash` use reported wallet totals from ledger/snapshot; relationship to **locked in orders** depends on how balances are reported upstream.

### 14.2 Simulation vs live execution

- Soft fill is differentiable approximation; live fills depend on queue position, latency, cancels, and competitor behavior.
- Rollout uses **market-only** books; resting own orders from previous intervals are not carried as liquidity in future rollout steps.
- Horizon `H` and interval length (e.g. 30s) may not match time to fill or mean-revert inventory.
- MC sample count and fixed seeds affect gradient noise and chosen action.

### 14.3 Objective structure

- **CE normalization applies only to CE**; validity `μ` and viability `λ` sit in different units unless manually aligned.
- **Viability `λ` adapts** on terminal availability gap, but `shortfall_tolerance` is not wired into the dual update.
- **Viability shortfall uses terminal `Ã_H` only**, not minimum availability along the path.
- **Viability margins use total balances**, not free-only; exchange may reject quotes when funds are locked in resting orders.
- **`μ = 0`** disables validity; **`λ_init = 0`** starts with negligible viability penalty until dual raises `θ`.
- **Sigmoid calibration:** at `m = 0`, each side has `σ(0) = 0.5`, so `Ã = 0.25` when both margins are exactly zero.

### 14.4 Capacity vs intent

- `bid_min = ask_min = 0`: optimizer may choose **one-sided** quotes when CE favors it and capacity allows the other side.
- Capacity uses `cash` and `inventory` totals, not separate buffers for “minimum operational reserve”.
- Large rebalance trades in simulation can flip imbalance sign within short horizons → **churn** in objective landscape; no explicit turnover penalty in current `J`.

### 14.5 Economics

- Maker fee (e.g. 0.15%) vs typical spread width affects whether CE rewards quoting at all.
- Negative CE triggers optional post-policy; behavior may quote floor/zero while validity/viability still shaped the optimizer before gating.
- No explicit adverse selection or toxicity term in `J`.

### 14.6 Joint vs single-venue divergence

- **Liquidity sourcing** uses consolidated wealth CE; single-venue batched path uses local rollout (see code branch).
- Joint rollout has **different VWAP timing** than local-only (documented lag in old implementation doc).
- Joint objective **lacks** some regularizers still present in legacy single-venue objective (volatility, CVaR, quote-shape penalties — §15).
- `global_*` validity/viability blocks can differ from local; consolidated wallet may not match per-venue penalties.

### 14.7 Diagnostics interpretation

- Contribution fractions **≠** economic importance.
- `viability_min_availability` is mean terminal `Ã_H` over MC paths, not live wallet availability.
- Plot **`viability_lambda_learned`** for dual time series; closure `viability_lambda` may lag or disagree if stamping is inconsistent.
- `expected_total_return` is simulated mean return, not accounting PnL.
- Component facts store initial/final/best snapshots; Grafana may show final fractions while operator inspects best CE.

### 14.8 Configuration and policy layers

- Registry **allocation** (capital cap) is not the same as viability `q_min` (minimum quote-notional per side).
- Risk caps can block orders that MPC thought feasible if risk uses different ledger view or freshness rules.
- `quote_imbalance_skew_strength` appears in legacy config validation; current `_quote_capacity_bounds_*` does not apply imbalance skew to mins (mins stay 0).

### 14.9 Expert review dimensions (unresolved)

Areas where multiple valid approaches exist and the current code picks one:

| Topic | Current approach | Alternatives not in code |
| --- | --- | --- |
| Return vs capacity trade-off | Adaptive `λ` on terminal `𝒮`; fixed `μ` on `B` | Inner dual loop; CE-linked `μ`; constraint formulation |
| CE scale | Optional EWMA normalize CE only | Joint normalization; normalize all terms |
| Two-sided presence | Hard capacity + optional validity | Hard min size per side; path-min viability |
| Capacity target | Soft `Ã_H` vs `A*` with dual `λ` | Free-balance margins; inventory ratio targets |
| Risk in objective | Entropic CE over paths | Mean-variance, CVaR in terminal `J` (legacy branch only) |
| Horizon objective | Sum of `ΔW` then one CE; terminal viability | Per-step viability; discounting |

---

## 15. Legacy and branch-specific objectives

Not all code paths share the same `J`.

### 15.1 Primary path (`monte_carlo_mpc_tensor_batched`)

Terminal `J` = CE − validity − viability as in §7. Used when joint optimizer is not selected.

### 15.2 Joint path (`optimize_joint_with_diagnostics`)

Consolidated CE + per-venue validity and viability terms. See §6.4 and old implementation doc for timing differences.

### 15.3 Legacy single-venue objective (still in codebase)

Some documentation and older configs reference a richer local-only objective:

```text
J_legacy =
  w_pnl · E[reward]
  − w_vol · volatility
  − w_cvar · soft_CVaR
  − w_conc · concentration
  − w_size · size_penalty
  − w_spread · spread_penalty
```

Shortage/threshold inventory penalties in that branch are **no longer wired** into the primary tensor batched optimizer. Experts should confirm which branch a given deployment actually runs before mapping math to behavior.

---

## 16. Glossary

| Symbol | Definition |
| --- | --- |
| `Q`, `cash` | Base inventory and quote cash in optimizer state (totals) |
| `mid` | Quote per base |
| `W` | Mark-to-market wealth |
| `W_0` | Initial wealth `max(1e-8, cash_0 + Q_0·mark_0)` |
| `ΔW_i,h` | Per-step wealth change on path `i` |
| `R_i` | Per-path horizon return `Σ_h ΔW_i,h / W_0` |
| `CE`, `CE_raw` | Entropic risk-adjusted return over paths (raw before normalization) |
| `CE_eff` | CE after optional EWMA normalization (`risk_adjusted_return` in diagnostics) |
| `β` | `return_risk_aversion_beta` |
| `σ(x)` | Logistic sigmoid `1/(1+e^{−x})` |
| `B` | Book quality score ∈ [0, 1] at h=0 |
| `B_min` | `validity.min_score` |
| `μ` | `validity.penalty_scale` |
| `q_min` | `viability.min_viable_quote` |
| `δ` | `viability.margin_buffer` |
| `m_bid`, `m_ask` | Quote-notional margins `q − q_min`, `b·p − q_min` |
| `a_bid`, `a_ask` | Soft side availability `σ(m/δ)` |
| `Ã`, `Ã_H` | Product availability; terminal value on path |
| `A*` | `viability.availability_target` |
| `𝒮` | Mean path shortfall `max(0, A* − Ã_H)` (`viability_penalty_raw`) |
| `λ`, `θ` | Viability multiplier `λ = exp(θ)`; `θ` persisted on MPC instance |
| `η` | `viability.lambda_learning_rate` |
| `s` | CE EWMA scale when normalization enabled |
| `fraction_k` | `\|component_k\| / Σ_j \|component_j\|` over weighted objective keys |
