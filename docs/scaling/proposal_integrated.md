# Integrated Market-Maker Optimizer Proposal

Merged architecture proposal synthesized from:

- `proposal_grok.md`
- `proposal_qwen2.md`
- `proposal_gemini2.md`
- `proposal_claude2.md`
- `proposal_gpt2.md`
- `**proposal_fable2.md**` (primary implementation backbone)

**Goal:** achieve a profitable, balanced, and less risky order book with good depth and spread — as a **tunable trade-off** between risk-adjusted return and book quality, controllable via a small set of scalar weights.

---

## 1. Executive Summary

The six proposals converge on one architecture:

1. **Keep feasibility hard** — cash, inventory, exchange limits, max size, and action-domain spread caps stay in the decoder and risk engine.
2. **Move preferences soft** — book shape and operational inventory safety become normalized objective terms.
3. **Inventory control via price shading**, not size floors — skew quotes by moving spread centers, not by forcing asymmetric minimum sizes.
4. **Adverse selection is structural** — couple flow and price innovations inside Monte Carlo rollouts so the entropic CE can see fill-then-adverse-move paths.

### Unified objective

```text
J = J_return + w_book * B - w_inv * Z
```


| Symbol            | Meaning                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| `J_return`        | Existing entropic certainty equivalent (`CE_beta(R)`), unchanged in form        |
| `B`               | Normalized book-quality composite in `[0, 1]`                                   |
| `Z`               | Operational inventory-zone penalty (trajectory-level), dimensionless            |
| `w_book`, `w_inv` | **Operator-facing dials** (per venue in `optimizer.local` / `optimizer.global`) |


Optional calibration constant `R_ref` (from historical `mm_optimizer_facts`) can scale `J_return` to O(1) so `w_book` and `w_inv` are directly comparable. This is a calibration constant, not a day-to-day operator dial.

**Keep `components["risk_adjusted_return"]` pure (CE only)** so `negative_objective_policy.py` and Grafana dashboards do not silently change meaning when book-quality terms are added.

---

## 2. Source Integration

What each proposal contributed to this merge:


| Source                   | Keep                                                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fable2**               | Two operator dials, normalized objective terms, decoder-level details, diagnostics plumbing, strict dependency order, spread-cap interaction analysis    |
| **GPT2**                 | Compute book quality on h=0 decoded action, keep CE diagnostics pure, demote `negative_objective_action` to safety valve                                 |
| **Claude2**              | Audit existing CMM flow→price wiring before building a full copula; add residual `beta_flow` only where needed; shared `objective_terms_torch.py` module |
| **Grok / Gemini / Qwen** | Hard feasibility vs soft preferences, zone-of-indifference inventory term, price shading over size capping, flow-price coupling                          |
| **All**                  | Remove imbalance skew floors once replacement terms are live; leave `NetDeltaRiskEngine` outside the optimizer                                           |


---

## 3. Mechanism Classification

**(a) = hard structural feasibility. (b) = soft tunable objective. (c) = replace architecturally.**


| Mechanism                              | Classification                    | Integrated treatment                                                                                                               |
| -------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Capacity / cash / inventory caps       | **(a)** hard                      | Keep in decoder. `quote_capacity_ratio < 1` is a *preference*, not feasibility — move to objective/reserve config or set ratio = 1 |
| `quote_size_max_per_side`              | **(a)** hard                      | Operational blast-radius cap; keep                                                                                                 |
| `quote_size_min_per_side`              | **Split**                         | Venue lot-size minimum stays hard (also enforced at `ExchangeRulesGateway`); two-sided presence becomes soft book-quality term     |
| Imbalance skew floors                  | **(c)** replace                   | Retire after inventory zone + price skew are validated; they force size-based rebalancing with gradient kinks                      |
| `quote_spread_max` + structured spread | **(a)** domain / **(b)** behavior | Keep as action-space bounds; treat spread-structured mode as contract-only unless exchange MM obligation requires it               |
| `negative_objective_action`            | **(c)** replace                   | Demote to circuit breaker (`zero_action` only at strongly negative CE threshold)                                                   |
| Independent flow / price sampling      | **(c)** replace                   | Conditional/coupled sampling so adverse selection appears in sampled paths                                                         |
| `NetDeltaRiskEngine`                   | **(a)** external                  | Independent hard backstop; do not move into objective                                                                              |
| Side-aware VWAP marking                | **Keep**                          | Valuation rule, not a constraint                                                                                                   |
| `soft_fill_batched()`                  | **Keep**                          | Simulator physics, unchanged except consuming coupled samples                                                                      |


---

## 4. Objective Terms (Detail)

All terms computed in shared module: `mm/mpc/objective_terms_torch.py` or `mm/mpc/book_quality.py`, imported by both `mm/mpc/reward_tensor.py` (single venue) and `mm/mpc/joint_mpc.py` (joint).

### 4.1 Risk-adjusted return — `J_return`

```text
J_return = CE_beta(R)                    # single venue
J_return = w_L * CE_{beta_L}(R^L) + active_G * w_G * CE_{beta_G}(R^G)   # joint
```

Wealth-variance inventory risk is priced here **once flow and price are coupled**. Today, independent sampling makes fill-then-adverse-move paths invisible to CE — coupling (§7) fixes that at the model layer, not with an extra penalty.

### 4.2 Book quality — `B`

Evaluated on **h=0 decoded action** (the only step executed) via `materialize_side_torch()` — no extra rollout cost, clean gradients.

```text
# Decoded curves: D_bid(s), D_ask(s); marginal m_side[j] on orderbook grid
Q_b = D_bid(spread_max);  Q_a = D_ask(spread_max)

presence   = smoothmin(D_bid(s_touch), D_ask(s_touch)) / (q_ref + smoothmin(...))   # [0,1)
tightness  = exp(-(x0_bid,1 + x0_ask,1) / s_ref)                                    # [0,1]
balance    = 1 - |Q_b - Q_a| / (Q_b + Q_a + eps)                                    # [0,1]
uniformity = 1 - Var_j(m[j]) / (mean_j(m[j])^2 + Var_j(m[j]))                       # [0,1]
             # constant-slippage shape; cousin of slippage_penalty_tensor()

B = v_p*presence + v_t*tightness + v_b*balance + v_u*uniformity     # v_* fixed, sum to 1
```

Operators tune `**w_book**` only. Sub-weights `v_*` ship as defaults (e.g. equal 0.25 each).

Each term flows into `components` dict → `OptimizerRunDiagnostics` → `mm_optimizer_component_facts` → Grafana.

**Do not add a separate two-sided-presence hard term** — presence is covered by soft `presence` plus structural min-size when feasible.

### 4.3 Inventory operational zone — `Z`

Separate from wealth-variance. Backstop against **short-horizon blindness** (`horizon = 2`) and model misspecification. **Not volatility-scaled** — if the model says σ ≈ 0, that is when you trust it least at boundaries.

**Shape (zone of indifference, C¹ at boundaries):**

```text
q = Q * mid / W0                      # value-fraction inventory, dimensionless

# Option A — power zones (Claude2):
g(Q) = 0                                    for Q in [Q_lo, Q_hi]
g(Q) = ((Q_lo - Q) / (Q_lo - Q_min))^p      for Q in (Q_min, Q_lo)
g(Q) = ((Q - Q_hi) / (Q_max - Q_hi))^p      for Q in (Q_hi, Q_max)

# Option B — sigmoid zones (Fable2):
Z(q) = a_lo * sigmoid((q_lo - q)/d_lo) + a_hi * sigmoid((q - q_hi)/d_hi)
```

Use **even power p ≥ 2** or smooth sigmoid — not `abs(Q - boundary)` (gradient kink at edge).

**Trajectory-level** (not terminal-only):

```text
Z_path = mean_i mean_h g(Q_{i,h})     # Q_{i,h} from rollout inv_t
J -= w_inv * Z_path
```

Joint mode: apply to consolidated `Q^L + Q^G` for wealth band; per-venue for capacity band (start with per-venue only).

Config: `inventory_zone: {q_lo, q_hi, d_lo, d_hi, a_lo, a_hi}` or `{Q_min, Q_lo, Q_hi, Q_max, p}` under each venue optimizer block. Set soft ramp **inside** `NetDeltaRiskEngine` hard min/max inventory limits.

### 4.4 Temporary stopgap (only if §7 is delayed)

Volatility-scaled interim term — **remove when coupling lands**:

```text
R_i += - kappa * sigma_step * sqrt(H) * |q_{i,H}|      # sigma from price model
```

Gated by config flag `adverse_selection_coupling_enabled: false`. Avoid double-counting once coupling is active.

---

## 5. Decoder and Price Shading

### 5.1 Hard size caps by construction

Current path almost correct: dynamic `ask_max` / `bid_max` flow into `quote_size_from_depth_logit_torch()`.

**Change in `_quote_capacity_bounds_batched()` / `_scalar()`:**

```text
# Current
ask_max = min(hard_cap, quote_capacity_ratio * inventory+)
bid_max = min(hard_cap, quote_capacity_ratio * cash+ / mid)

# Proposed
ask_max = min(hard_cap, max(inventory - inventory_reserve, 0))
bid_max = min(hard_cap, max(cash - cash_reserve, 0) / mid)
```

`quote_capacity_ratio` deprecated or hardcoded to 1.0; reserve preferences move to zone term or explicit reserve config.

Because `Q_s_total = Q_s_min + sigmoid(u) * (Q_s_max - Q_s_min)`, feasibility holds by construction.

### 5.2 Reservation price skew (core inventory mechanism)

Several source proposals describe the same idea with different math. The integrated design treats them as **equivalent views of one mechanism**: inventory pressure shifts where quotes are centered in price space, instead of shrinking size.

#### Reservation price (GPT2 / Grok framing)

Fair value stays at `quote_anchor` (from `mm/pricing/anchor.py`). Inventory shifts a **reservation price** used to place the ladder:

```text
# Normalized imbalance (matches current capacity computation)
imbalance = clamp((Q*mid - cash) / (|Q*mid| + |cash| + eps), -1, 1)

# Reservation price: long inventory → lower reservation (discourage buys, encourage sells)
reservation_price = quote_anchor - lambda_res * imbalance * quote_anchor

# Ladder around reservation, not fair anchor
bid_price(spread) = reservation_price - spread_bid
ask_price(spread) = reservation_price + spread_ask
```

At discretization, this is applied by passing a shifted anchor into `build_joint_target_book_from_sigmoid()` (`mm/strategies/target_book_builder.py` → `build_orders_from_sigmoid_with_report(..., anchor_price=...)`). Today `local_anchor_price` is the fair quote anchor from `MpcStrategyEngine.decide()`; reservation skew would use `reservation_price` instead (or an equivalent spread-space offset).

#### Spread-space skew (Qwen2 / Claude2 framing)

Same economics, expressed on sigmoid centers instead of absolute price:

```text
# Deterministic overlay (Qwen2) — config-driven, not optimizer-learned
x0_bid_eff = x0_bid_base + lambda_skew * imbalance * quote_spread_max
x0_ask_eff = x0_ask_base - lambda_skew * imbalance * quote_spread_max

# Optimizer-learned skew (Claude2) — preferred when zone term needs a dedicated dial
delta_skew = skew_max * tanh(u_skew)
x0_bid_eff = clamp(x0_bid_base + delta_skew, 0, quote_spread_max)
x0_ask_eff = clamp(x0_ask_base - delta_skew, 0, quote_spread_max)
```

Long inventory: widen bid (larger `x0_bid`), tighten ask (smaller `x0_ask`). **Verify sign** against fill-delta convention in `soft_fill_batched()` during implementation.

#### Three implementation paths (merged recommendation)


| Path                              | Mechanism                                                    | Where                                           | When to use                                                                   |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| **A — Implicit (Fable2 default)** | Zone term `Z` gradient flows through fill sim into free `x0` | Rollout closure in `mpc_tensor_batched.py`      | **Phase 1** — lowest risk; no new action dimension                            |
| **B — Reservation anchor shift**  | `reservation_price` at discretization                        | `mpc_engine.decide()`, `target_book_builder.py` | When you want interpretable, logged reservation skew independent of optimizer |
| **C — Learned `delta_skew`**      | New optimizer dimension per venue/step                       | `l_param.py`, vector layout in joint optimizer  | **Phase 2** — if diagnostics show Path A/B insufficient                       |


Path A and C are not mutually exclusive: the zone term supplies the *objective pressure*; Path C supplies an explicit, bounded skew channel the optimizer can use without fighting book-quality balance terms.

**Do not** reintroduce size-based skew (`bid_floor` / `ask_floor` with `quote_imbalance_skew_strength`). Remove after Path A or C is validated:

```text
bid_floor = q_min * (1 + skew_strength * relu(-imbalance))   # DELETE after validation
```

#### Config and diagnostics

New config knobs (per venue):

- `lambda_res` or `reservation_skew_strength` — anchor-shift magnitude (Path B)
- `lambda_skew` — deterministic x0 overlay (Qwen2 interim only)
- `skew_max` — bound for learned `delta_skew` (Path C)
- `max_skew_bps` — decouple skew headroom from `quote_spread_max` (Qwen2)
- `skew_spread_max` — allow one-sided retreat beyond symmetric cap while keeping fair-value spread tight (Claude2 option ii)

**Diagnostics** (optimizer facts / Grafana):

```text
reservation_shift = |reservation_price - quote_anchor|   # or |delta_skew| in spread units
reservation_shift / quote_spread_max                       # saturation check
```

Rule of thumb (GPT2): keep typical `reservation_shift / quote_spread_max < 0.3–0.5` so skew does not saturate the action domain. If saturation is frequent, widen `quote_spread_max` or use `skew_spread_max`.

**Discretization quantization (Fable2):** price shading becomes discrete ladder steps at execution (`spread_step` over `spread_max`). Coarse steps limit fine skew control — note in config validation when tuning reservation skew.

Raw vector length with Path C: `6*n_components + 3` per venue per step (was `+ 2`).

### 5.3 `quote_spread_max` headroom audit

Before finalizing skew design, pull recent `x0_`* distributions from `mm_optimizer_facts`:


| Typical x0 vs cap              | Action                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| < 50–60% of `quote_spread_max` | Simple clamp-based skew is sufficient                                                                   |
| > 70–80% of cap                | Skew saturates when most needed — widen `quote_spread_max` or add `skew_spread_max >= quote_spread_max` |


**Structured spread mode** (`quote_spread_min_depth > 0`): caps **sum** `x0_bid,1 + x0_ask,1 ≤ quote_spread_max`. Skew-as-reallocation works; widening both sides does not. Treat as contract-only. Example config has local `quote_spread_min_depth: 0.0` — sum cap does not bind locally.

Add diagnostic: `reservation_shift / spread_max` in optimizer diagnostics.

Ensure `spread_max` (orderbook grid) ≥ `quote_spread_max`; config validation warning if equal with structure enabled.

---

## 6. Regime Handoff (Smooth Transition)


| Regime                | Dominant term                      | Book behavior                                        |
| --------------------- | ---------------------------------- | ---------------------------------------------------- |
| Inside `[Q_lo, Q_hi]` | Book quality (`w_book * B`)        | Tight, balanced, deep, uniform                       |
| Approaching band edge | Inventory zone ramps (`w_inv * Z`) | Price skew: widen bad side, tighten rebalancing side |
| Hard boundary         | Decoder + `NetDeltaRiskEngine`     | Ask side vanishes at q=0; risk engine rejects        |


`w_inv` and `w_book` have **largely non-overlapping regimes**: inside the band, `Z ≡ 0` regardless of `w_inv` magnitude; `w_book` governs normal conditions, `w_inv` governs edge reaction.

Discontinuities removed: `negative_objective_action` jump, `relu`-kinked imbalance floors. Intentional remaining cliff: hard `ask_max = inventory` at q=0.

---

## 7. Adverse Selection Coupling

### 7.1 Problem

In `draw_samples()` (`mpc_tensor_batched.py`, `joint_optimizer.py`), flow and price randomness are **independent**. CE cannot penalize paths where large fills coincide with adverse price moves.

### 7.2 Step 0 — Audit CMM wiring (do first, ~0.5–1 day)

Check whether `TorchConditionalMechanicalMixturePriceModel.sample_dynamic_vwap()` receives **per-path, per-step simulated** `total_flow` and mechanical book state from `soft_fill_batched()` output — not stale scalars from the last observed frame.

If already wired: adverse coupling may exist for CMM; add regression test + diagnostics only.

If gap: wire simulated covariates at rollout step:

```python
total_flow_sim_h     = buy_vol_h + sell_vol_h
mechanical_mid_sim_h = mechanical_vwap_delta(book_{i,h}, buy_vol_h, sell_vol_h)
buy_vwap_h, sell_vwap_h = price_model.sample_dynamic_vwap(
    total_flow=total_flow_sim_h,
    mechanical_component=mechanical_mid_sim_h,
    ...
)
```

### 7.3 Residual model — flow-regression (if `residual` is active)

Maintain EWMA covariance between residual and net flow; sample conditionally:

```text
epsilon_side_{i,h} = mu_side
                     + beta_flow_side * (net_flow_{i,h} - flow_mean)
                     + sqrt(resid_var_side) * z_{i,h}
```

Flow sampling must precede residual sampling for same `(i, h)`.

### 7.4 Full copula (optional, higher fidelity)

Shared latent per path/step:

```text
g_{i,h} ~ N(0,1)

z_buy  = rho_f * g + sqrt(1-rho_f^2) * eta_b
z_sell = -rho_f * g + sqrt(1-rho_f^2) * eta_s
V_buy  = F_buy^{-1}(Phi(z_buy))
V_sell = F_sell^{-1}(Phi(z_sell))

# Residual: eps = mu + sigma * (rho_p*g + sqrt(1-rho_p^2)*zeta)
```

Online EWMA of `(signed_flow, price_innovation_residual)` for `rho_p`, `rho_f`. Config flag `controller.coupling: {enabled, decay, rho_floor, rho_cap}`.

**Pragmatic alternative:** conditional mean shift (Qwen2) if Gamma inverse-CDF is too costly for autograd.

Marginals preserved (copula property) → existing `score_frame_models()` stays valid.

### 7.5 Effort


| Sub-step                           | Effort      |
| ---------------------------------- | ----------- |
| A0 audit                           | 0.5–1 day   |
| A1 CMM wiring fix                  | 2–3 days    |
| A2 residual beta_flow              | 3–5 days    |
| A3 rollout ordering                | 1–2 days    |
| Full copula + inverse-CDF refactor | 1.5–2 weeks |


---

## 8. Policy Cleanup (P6)

With book-quality in objective, demote `negative_objective_action`:

- Default: `optimizer_action` (trust optimizer)
- Circuit breaker: `zero_action` only when `risk_adjusted_return < -tau` (return units)

Files: `mm/strategies/negative_objective_policy.py`, `mm/runtime/config.py`, example JSON.

---

## 9. Implementation Roadmap


| Order | Item                     | What changes                                                    | Effort    | Depends on           |
| ----- | ------------------------ | --------------------------------------------------------------- | --------- | -------------------- |
| 1     | Objective scaffold       | Normalized terms + diagnostics plumbing                         | 1–2 days  | —                    |
| 2     | Inventory zone           | Trajectory-level `Z` with comfort band                          | 1–3 days  | 1                    |
| 3     | Decoder cleanup          | True available caps; remove ratio preference                    | 0.5–1 day | 2 validated          |
| 4     | Spread headroom audit    | Check x0 saturation vs cap                                      | 0.5–1 day | —                    |
| 5     | Reservation / price skew | Implicit x0 first; reservation anchor or `delta_skew` if needed | 2–5 days  | 2, 4                 |
| 6     | Book-quality bundle      | `B` composite on h=0 action                                     | 2–3 days  | 1, 3                 |
| 7     | Adverse selection        | Audit → CMM/residual/copula                                     | 3–10 days | parallel from step 1 |
| 8     | Policy cleanup           | Demote negative objective; remove stopgap                       | 0.5–1 day | 1, 7                 |


**Critical path:** 1 → 2 → (3, 4 parallel) → 5 → 6

**Do not remove imbalance skew floors until 2 + 5 are validated in shadow/paper** — they are the only active inventory control today.

**Recommended first ship:** P1 + P2 (objective scaffold + inventory zone) — unlocks tunable `{w_book, w_inv}` trade-off with minimal modeling risk.

Overall: ~3–4 weeks for objective/decoder core; +3–10 days for adverse selection depending on price-model config.

---

## 10. Code Touchpoints


| Area             | File(s)                                                               | Change                                                             |
| ---------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Objective terms  | **New** `mm/mpc/objective_terms_torch.py` or `mm/mpc/book_quality.py` | Shared `inventory_operational_term()`, `book_quality_term()`       |
| Single objective | `mm/mpc/reward_tensor.py`                                             | `J = CE - w_inv*Z + w_book*B`                                      |
| Joint objective  | `mm/mpc/joint_mpc.py`                                                 | Same; consolidated Q for inventory                                 |
| Rollout closure  | `mm/mpc/mpc_tensor_batched.py`                                        | h=0 book terms; trajectory Z; capacity bounds                      |
| Joint rollout    | `mm/mpc/joint_optimizer.py`                                           | Mirror capacity, objective, coupling                               |
| Decoder          | `mm/mpc/l_param.py`                                                   | Feasibility by construction; optional `delta_skew`                 |
| Discretization   | `mm/strategies/target_book_builder.py`, `mpc_engine.decide()`         | Reservation anchor shift at `build_joint_target_book_from_sigmoid` |
| Flow models      | `mm/mpc/market_order_flow_torch.py`                                   | `sample_from_uniforms()` / conditional API                         |
| Price models     | `mm/mpc/price_torch.py`                                               | Conditional sampling; beta_flow for residual                       |
| Coupling         | **New** `mm/mpc/coupling.py` (optional)                               | Latent draw, copula, EWMA rho                                      |
| Strategy         | `mm/strategies/mpc_engine.py`                                         | Config keys, model update for rho                                  |
| Config           | `mm/runtime/config.py`                                                | `w_book`, `w_inv`, `inventory_zone`, `coupling`                    |
| Policy           | `mm/strategies/negative_objective_policy.py`                          | Circuit breaker only                                               |
| Tests            | `tests/unit/test`_*                                                   | Dimensional consistency, C¹ boundaries, marginal preservation      |
| Offline eval     | `mm/persistence/frame_replay.py`, `mm/application/frame_simulator.py` | Pareto frontier tuning                                             |


---

## 11. Config Additions (Sketch)

Per venue under `controller.optimizer.local` / `.global`:

```json
{
  "w_book": 0.002,
  "w_inv": 0.01,
  "inventory_zone": {
    "q_lo": -0.15,
    "q_hi": 0.15,
    "d_lo": 0.05,
    "d_hi": 0.05,
    "a_lo": 1.0,
    "a_hi": 0.6
  },
  "book_quality": {
    "spread_target": 50,
    "depth_target_quote": 1000000,
    "v_presence": 0.25,
    "v_tightness": 0.25,
    "v_balance": 0.25,
    "v_uniformity": 0.25
  },
  "coupling": {
    "enabled": false,
    "decay": 0.05,
    "rho_floor": -0.95,
    "rho_cap": 0.95
  },
  "reservation_skew_strength": 0.0,
  "skew_max": 50.0,
  "max_skew_bps": 100.0
}
```

Reject legacy keys per existing strict-migration pattern (`inv_coef`, `target_inventory`, `quote_capacity_ratio` as preference, etc.).

Optional calibration: `R_ref` — typical CE magnitude from `mm_optimizer_facts`, used to scale `J_return` so `w_book` and `w_inv` are O(1) operator dials (Claude2 §0).

---

## 12. Architecture Principle

> The **optimizer** decides how much PnL to sacrifice for book quality and how aggressively to move away from dangerous inventory.
>
> The **decoder** defines what is physically possible.
>
> The **risk engine** remains an independent hard stop.

That three-layer separation is the best common proposal across all source documents.

---

## 13. Dependency Graph

```text
P1 (book-quality objective + plumbing)
 ├── P2 (inventory zone term)
 │    └── P3 (decoder cleanup, retire skew floors)
 ├── P4 (stopgap vol-scaled inventory; interim until P7)
 └── P6 (demote negative_objective_action)

P7 (flow–price coupling) — independent track; start A0 audit in parallel; remove P4 when done
```

---

## 14. Success Criteria

1. Operator can tune `w_book` / `w_inv` and observe trade-off in Grafana optimizer component facts.
2. Near-target inventory: balanced, tight, deep books without heuristic floor skew.
3. Off-target inventory: smooth price skew without size collapse or objective kinks.
4. After coupling: left-tail mass in `R_i` increases for large `|Q|` states (CE responds correctly).
5. Offline replay shows improved fill quality / lower adverse-selection drag vs baseline.
6. No regression in risk-engine rejection rate or ledger/order-manager consistency.
7. `reservation_shift / quote_spread_max` stays below saturation threshold in normal inventory bands; visible in optimizer diagnostics when skew is active.

---

## 15. Coverage checklist (source → integrated)

Items explicitly merged from each proposal — use this to spot gaps during implementation review:


| Topic                                  | Grok   | Qwen         | Gemini        | Claude          | GPT                 | Fable       | In this doc |
| -------------------------------------- | ------ | ------------ | ------------- | --------------- | ------------------- | ----------- | ----------- |
| Flow–price coupling                    | ✓      | ✓            | ✓             | ✓ (audit-first) | ✓                   | ✓           | §7          |
| Zone inventory term                    | ✓      | ✓            | ✓             | ✓               | ✓                   | ✓           | §4.3        |
| Hard size cap by construction          | ✓      | ✓            | ✓             | ✓               | ✓                   | ✓           | §5.1        |
| **Reservation price skew**             | anchor | x0 + λ       | price shading | delta_skew      | **r = anchor − λq** | implicit x0 | **§5.2**    |
| Book-quality soft terms                | ✓      | ✓            | ✓             | ✓               | ✓                   | ✓           | §4.2        |
| Remove imbalance floors                | ✓      | ✓            | ✓             | ✓               | ✓                   | ✓           | §5.2        |
| `skew_spread_max` / headroom           | —      | max_skew_bps | spread room   | ✓               | 30–50% rule         | spread_step | §5.2–5.3    |
| `R_ref` normalization                  | —      | —            | —             | ✓               | —                   | —           | §1, config  |
| Negative objective demotion            | —      | —            | —             | —               | —                   | ✓ P6        | §8          |
| Stopgap vol × inventory²               | ✓      | ✓            | ✓             | ✓ B3            | ✓                   | ✓ P4        | §4.4        |
| `build_joint_target_book_from_sigmoid` | ✓      | —            | —             | ✓               | ✓                   | ✓           | §5.2, §10   |
| NetDeltaRiskEngine unchanged           | —      | —            | —             | ✓               | —                   | ✓           | §3          |


