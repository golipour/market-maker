# MPC Market-Making Optimization: Objective, Mechanism, and Business Goals

This document describes the implemented MPC controller: business intent, mathematics, data flow, configuration surface, diagnostics, and known gaps. It is written for external expert review. It records what the code does and what is uncertain; it does not prescribe tuning values or a preferred redesign.

Related implementation references:

- `mm.strategies.mpc_engine.MpcStrategyEngine`
- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched`
- `mm.mpc.joint_optimizer.optimize_joint_with_diagnostics` (liquidity-sourcing mode)
- `mm.mpc.objective_terms` (book quality, inventory zone, normalization)
- `mm.mpc.reward_tensor` (risk-adjusted return)
- `docs/old/mpc-optimizer-implementation-details.md` (rollout timing, decoder, joint vs local-only differences)

---

## 1. Business goals (operational intent)

The controller targets **persistent market making**: quoting over many intervals rather than optimizing a single trade.

### 1.1 Two-sided quote availability

Operational intent includes:

- Bid and ask liquidity when wallet capacity allows.
- Wide spreads are not forbidden by the core return term; tightness is only rewarded if `w_book > 0`.
- Depleting quote (IRT) or base (USDT) on one side removes hard capacity on that side and ends two-sided quoting in simulation.

Implemented mechanisms that relate to this intent:

| Mechanism | Type | Role |
| --- | --- | --- |
| `bid_max = cash/mid`, `ask_max = inventory` | Hard | Cannot simulate quotes beyond wallet totals |
| `w_book · B` | Soft (optional) | Rewards presence, tightness, balance, uniformity at h=0 |
| `−w_inv · Z̄` | Soft (optional) | Penalizes imbalanced inventory **paths** in simulation |

Current capacity decoder sets `bid_min = ask_min = 0`. There is **no hard minimum quote per side** at the capacity layer. `quote_size_min_per_side` affects spread-structure decode at `spread_max`, but mins are clamped to feasible capacity and can be zero when a side has no free balance.

### 1.2 Profitability

Profitability enters through **risk-adjusted horizon return** on simulated mark-to-market wealth, with maker fees and a differentiable soft-fill model. The optimizer maximizes CE (see §7.1). A separate **negative-objective policy** (§9) may alter the action when CE is negative; that policy is outside `J`.

### 1.3 Inventory balance (50/50 economic value)

The inventory zone uses wallet imbalance:

```text
wallet_imbalance = (Q·mid − cash) / (|Q·mid| + |cash| + ε)
```

- `imbalance = 0` → equal **value** in base and quote (not equal units).
- `imbalance < 0` → more quote value than base value (cash-heavy in this convention).
- `imbalance > 0` → more base value than quote value.

Comfort band defaults: `[comfort_low, comfort_high] = [−0.2, +0.2]` with sigmoid ramps outside.

This 50/50 target is **in the MPC objective only**. Registry allocation caps, risk limits, and physical exchange balances are separate controls and may not match this target.

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

`book_quality.hard_structure` can cap first-touch spread sum and minimum depth at the first center. `quote_size_min_per_side` is a lot-size floor in structural sizing at `spread_max`, not an independent per-side minimum at the capacity layer.

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

```text
J = CE + w_book · B − w_inv · Z̄
    (+ global book / inventory terms in joint mode)
```

Evaluation timing:

- **CE, combined `J`:** from terminal closure using the candidate action vector at end of closure evaluation.
- **Book quality `B`:** decoded action at **horizon step 0 only**.
- **Inventory `Z̄`:** mean of `Z(imb)` over **simulated path** across horizon steps (and MC paths).

### 7.1 Risk-adjusted return (CE)

Per-path horizon returns `R_i` (wealth-normalized):

```text
CE = min_i(R_i) − (1/β) · ln( (1/N) Σ_i exp(−β · (R_i − min_i(R_i))) )
```

`β = return_risk_aversion_beta` (venue-scoped).

Properties:

- Soft minimum / entropic certainty equivalent over paths.
- Sensitive to tail paths, not only mean return.
- Units: dimensionless return after dividing by `W_0`.

CE is computed from **simulated** fills and model-sampled flows, not realized live PnL.

### 7.2 CE normalization (`objective_normalize`)

When `objective_normalize.enabled`:

1. `risk_adjusted_return_raw` = raw CE.
2. `risk_adjusted_return` = `CE_raw / s`, with `s` = EWMA of `|CE_raw|` from **prior decisions** (fixed during the Adam loop).
3. `J ← J − CE_raw + CE_norm`.

After the frame: `s_new = decay · s + (1 − decay) · |CE_raw|`.

**Asymmetry:** Only CE is scaled. `w_book` and `w_inv` are fixed scalars on their raw scores. No coupled normalization for inventory or book terms exists in code.

When `CE_raw = 0`, CE contributes nothing to `J`; inventory and book terms remain.

### 7.3 Book quality (`B`, weight `w_book`)

`B ∈ [0, 1]` at h=0 from weighted sub-scores:

| Sub-score | Construction (summary) |
| --- | --- |
| Presence | Touch depth vs `depth_target` |
| Tightness | `exp(−spread_sum / spread_target)` on first centers |
| Balance | Bid/ask notional symmetry at `spread_max` |
| Uniformity | Evenness of marginal depth across grid |

`measure = "quoted"` | `"combined"` (adds market depth at each level).

```text
book_quality_component = w_book · B
```

If `w_book = 0`, no soft reward for two-sided shape; only hard capacity applies.

### 7.4 Inventory zone penalty (`Z̄`, weight `w_inv`)

**Imbalance at each rollout step:**

```text
imb = (Q·mid − cash) / (|Q·mid| + |cash| + ε)
```

**Zone penalty:**

```text
Z(imb) = (a_lo · σ((comfort_low − imb) / d_lo) + a_hi · σ((imb − comfort_high) / d_hi)) / (a_lo + a_hi)
```

Default parameters (configurable per venue):

| Parameter | Default | Role |
| --- | --- | --- |
| `comfort_low` | −0.2 | Lower comfort edge |
| `comfort_high` | +0.2 | Upper comfort edge |
| `low_ramp_width` | 0.05 | Ramp below band |
| `high_ramp_width` | 0.05 | Ramp above band |
| `low_penalty_cap` | 1.0 | Low-side cap |
| `high_penalty_cap` | 0.6 | High-side cap |

`Z(imb)` is bounded in `[0, 1]` but **not zero at imb = 0** (sigmoid tails → floor ≈ 0.018 with default caps). Saturates ≈ 0.625 at extreme imbalance with default caps.

**Path aggregate:**

```text
Z̄ = mean_h,Z Z(imb_h)   (over horizon and MC batch in practice)
```

**Weighted term:**

```text
inventory_zone_penalty = −w_inv · Z̄
```

Diagnostics:

| Key | Meaning |
| --- | --- |
| `inventory_zone_penalty_raw` | `Z̄` (unweighted) |
| `inventory_zone_imbalance` | Mean `imb` along **simulated path** |
| `inventory_zone_low_penalty` / `high_penalty` | Side breakdown |

**Ambiguous case `imb = 0`:** Can mean economically balanced wallet, or `cash = inventory = 0` (formula gives 0), or near-zero wealth with numerical ε. `inventory_zone_imbalance` in diagnostics does not by itself distinguish these; read alongside `state.local_*` and wallet legs.

### 7.5 Combined objective and components

```text
J = CE_eff + w_book · B − w_inv · Z̄
```

`CE_eff` = normalized CE if enabled, else raw CE.

Stored weighted components (representative keys):

| Key | Value stored |
| --- | --- |
| `risk_adjusted_return` | CE_eff |
| `risk_adjusted_return_raw` | CE_raw (if normalization on) |
| `objective_ce_ewma_scale` | Scale `s` used |
| `book_quality` | `w_book · B` |
| `inventory_zone_penalty` | `−w_inv · Z̄` |
| `inventory_zone_penalty_raw` | `Z̄` |

### 7.6 Contribution fractions

```text
fraction_k = |component_k| / Σ_j |component_j|
```

Keys included: `risk_adjusted_return`, `book_quality`, `inventory_zone_penalty`, plus `global_*` variants in joint mode.

Interpretation constraints:

- Fractions are **shares of absolute weighted terms in `J`**, not shares of economic PnL or notional risk.
- Any term with value exactly 0 drops out of the denominator; remaining terms sum to 100%.
- Small absolute penalty can show 100% share if CE and other terms are 0.
- Large CE can dominate fractions even when inventory imbalance is extreme in the live wallet.

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

This gate uses optimizer diagnostics, not live PnL. Interaction with `w_inv` and CE normalization is configuration-dependent.

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
| `local_optimizer.w_inv` | varies (e.g. 0.001, 0.01) |
| `local_optimizer.w_book` | often 0 |
| `return_risk_aversion_beta` | e.g. 0.1 |
| `quote_size_min_per_side` | venue lot floor |
| `inventory_zone` | sigmoid band (see §7.4) |
| `execution_model` maker fee | e.g. 0.0015 |
| `soft_fill_queue_model` | `back_of_queue` |

Separate `local_optimizer` and `global_optimizer` blocks exist in joint mode.

---

## 11. Weight coupling: fixed `w_inv` vs CE scale

### 11.1 What is implemented

| Quantity | Scale / units | Adaptive? |
| --- | --- | --- |
| CE (`risk_adjusted_return`) | Wealth-normalized return; optionally divided by EWMA `|CE_raw|` | CE only, via `objective_normalize` |
| `w_book · B` | `B ∈ [0,1]` × fixed `w_book` | No |
| `−w_inv · Z̄` | `Z̄ ∈ [0,1]` × fixed `w_inv` | No |

`w_inv` and `w_book` are **static config scalars** until changed in registry. They do not auto-adjust to `|CE|`, `|CE_raw|`, wallet size, or live imbalance.

### 11.2 Why tuning is non-obvious

CE magnitude moves with:

- Fee vs spread vs fill rate in simulation
- `horizon`, `n_samples`, `β`
- Wealth level and mark prices
- CE EWMA state when normalization is on
- Whether wallet in state matches economic reality

`Z̄` magnitude moves with:

- Current and simulated path imbalance
- Zone geometry (comfort band, caps, ramp widths)
- Horizon length (path mean over steps)
- Same `imb = 0` floor for balanced and empty-degenerate states

There is **no single `w_inv`** that yields a constant economic trade-off against CE across regimes without additional structure.

### 11.3 Illustrative arithmetic (not a tuning guide)

For reference only, with `Z̄ ≈ 0.625` (saturated zone) and `Z̄ ≈ 0.018` (near floor):

| `w_inv` | `|−w_inv · Z̄|` at saturation | `|−w_inv · Z̄|` at floor |
| --- | --- | --- |
| 0.001 | 0.0006 | 0.000018 |
| 0.01 | 0.006 | 0.00018 |
| 0.05 | 0.031 | 0.0009 |
| 0.10 | 0.063 | 0.0018 |

Relative share in `J` depends on contemporaneous `|CE_eff|` and whether `w_book` is active. Example: if `|CE_eff| = 1`, `w_inv = 0.05`, saturated `Z̄`, inventory share ≈ 3%. If `|CE_eff| = 0.01`, same settings give ≈ 75% share. If `CE_eff = 0`, inventory share is 100% regardless of absolute penalty size.

### 11.4 Design space not implemented

The codebase does **not** include any of the following; each would be a design change:

- `w_inv` proportional to `|CE_raw|` or `s` (EWMA scale)
- Normalizing `Z̄` or `w_inv · Z̄` by the same `s` as CE
- Per-frame adaptive weights from live imbalance
- Separate weights for local vs global inventory in joint mode beyond existing `w_inv` per venue
- Hard minimum spread or minimum two-sided size in `J` (only soft `w_book` or capacity)

Experts may evaluate any of these (or other) couplings against the gaps in §15.

---

## 12. Diagnostics map

| Field | What it reflects | Caveats |
| --- | --- | --- |
| `optimizer.objective.final.value` | Best `J` | Simulation objective, not realized PnL |
| `optimizer.objective.final.components` | Weighted terms | See §7.5–7.6 |
| `optimizer.objective.contribution_fractions` | Relative \|term\| shares | Misleading when terms are near zero |
| `optimizer.objective.best.expected_total_return` | Mean `R_i` over paths | Simulated |
| `state.local_inventory`, `local_cash` | Wallet fed to MPC | May lag exchange; virtual vs physical |
| `state.local_*_wallet` | Free/total/locked legs | Order locks may not match operator mental model |
| `inventory_zone_imbalance` | Mean sim path imb | ≠ current wallet imb necessarily |
| `inventory_zone_penalty_raw` | `Z̄` | Same `Z̄` for imb=0 balanced vs empty |

Postgres tables: `mm_optimizer_facts`, `mm_optimizer_component_facts`, `mm_events` diagnostics payloads.

---

## 13. Business goal → mechanism mapping (with gaps)

| Intent | Implemented linkage | Gap / blind spot |
| --- | --- | --- |
| Profitable quoting | Maximize simulated CE | Sim ≠ live; model error; fee/slippage gaps |
| Both sides quoted | Hard capacity; optional `w_book` | `w_book=0` → no soft two-sided reward; mins are 0 |
| Don't run out of IRT/USDT | Capacity zeros depleted side | Doesn't prevent slow drift; locked funds not modeled in capacity formula as separate field |
| Wide spread OK | No BBO term in CE | Tightness only if `w_book > 0` |
| Rebalance to 50/50 value | `−w_inv · Z̄` on sim paths | Weak `w_inv` vs CE; churn risk if strong; target ≠ allocation config |
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

- **CE normalization applies only to CE**; `w_inv` and `w_book` sit in different units unless manually aligned.
- **Fixed `w_inv`** has no documented invariant (e.g. constant rebalance half-life or constant fraction of CE).
- **Inventory penalty is on simulated paths**, not explicitly on terminal wallet imbalance only.
- **Sigmoid floor at imb = 0** applies even when balanced; penalty never fully turns off.
- **`imb = 0` is ambiguous** (balanced, empty, or near-zero wealth).
- **`w_book = 0`** removes all soft encouragement of two-sided book shape.

### 14.4 Capacity vs intent

- `bid_min = ask_min = 0`: optimizer may choose **one-sided** quotes when CE favors it and capacity allows the other side.
- Capacity uses `cash` and `inventory` totals, not separate buffers for “minimum operational reserve”.
- Large rebalance trades in simulation can flip imbalance sign within short horizons → **churn** in objective landscape; no explicit turnover penalty in current `J`.

### 14.5 Economics

- Maker fee (e.g. 0.15%) vs typical spread width affects whether CE rewards quoting at all.
- Negative CE triggers optional post-policy; behavior may quote floor/zero while inventory penalty still shaped the optimizer before gating.
- No explicit adverse selection or toxicity term in `J`.

### 14.6 Joint vs single-venue divergence

- **Liquidity sourcing** uses consolidated wealth CE; single-venue batched path uses local rollout (see code branch).
- Joint rollout has **different VWAP timing** than local-only (documented lag in old implementation doc).
- Joint objective **lacks** some regularizers still present in legacy single-venue objective (volatility, CVaR, quote-shape penalties — §15).
- `global_optimizer.w_inv` and `local_optimizer.w_inv` can differ; consolidated imbalance in live wallet may not match per-venue penalties.

### 14.7 Diagnostics interpretation

- Contribution fractions **≠** economic importance.
- `inventory_zone_imbalance` is path mean, not necessarily `state` wallet imbalance.
- `expected_total_return` is simulated mean return, not accounting PnL.
- Component facts store initial/final/best snapshots; Grafana may show final fractions while operator inspects best CE.

### 14.8 Configuration and policy layers

- Registry **allocation** (capital cap) is not the same as MPC **50/50 imbalance target**.
- Risk caps can block orders that MPC thought feasible if risk uses different ledger view or freshness rules.
- `quote_imbalance_skew_strength` appears in legacy config validation; current `_quote_capacity_bounds_*` does not apply imbalance skew to mins (mins stay 0).

### 14.9 Expert review dimensions (unresolved)

Areas where multiple valid approaches exist and the current code picks one:

| Topic | Current approach | Alternatives not in code |
| --- | --- | --- |
| Return vs inventory trade-off | Fixed `w_inv`, fixed `w_book` | CE-linked weights, constraint formulation, Lagrangian on imbalance |
| CE scale | Optional EWMA normalize CE only | Joint normalization; normalize all terms; no normalization |
| Two-sided presence | Hard capacity + optional `w_book` | Hard min size per side; penalty for one-sided books |
| Inventory target | Sigmoid zone on value imbalance | Target inventory in units; dollar-neutral; band on base ratio |
| Risk in objective | Entropic CE over paths | Mean-variance, CVaR in terminal `J` (exists in legacy branch only) |
| Horizon objective | Sum of `ΔW` then one CE | Per-step rewards, terminal wealth only, discounting |

---

## 15. Legacy and branch-specific objectives

Not all code paths share the same `J`.

### 15.1 Primary path (`monte_carlo_mpc_tensor_batched`)

Terminal `J` = CE (+ optional `w_book`, `w_inv`) as in §7. Used when joint optimizer is not selected.

### 15.2 Joint path (`optimize_joint_with_diagnostics`)

Consolidated CE + per-venue `w_book` / `w_inv` terms. See §6.4 and old implementation doc for timing differences.

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
| `Q`, `cash` | Base inventory and quote cash in optimizer state |
| `mid` | Quote per base |
| `W` | Mark-to-market wealth |
| `R_i` | Per-path horizon return `ΣΔW / W_0` |
| `CE` | Entropic risk-adjusted return over paths |
| `CE_eff` | CE after optional EWMA normalization |
| `β` | `return_risk_aversion_beta` |
| `B` | Book quality score ∈ [0, 1] |
| `Z(imb)` | Sigmoid zone penalty ∈ [0, 1] |
| `Z̄` | Mean zone penalty over simulated path |
| `w_book`, `w_inv` | Fixed config weights on soft terms |
| `s` | CE EWMA scale when normalization enabled |
