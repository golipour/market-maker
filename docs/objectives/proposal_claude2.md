# MPC Market-Maker: Risk-Adjusted Return / Book-Quality Trade-off Proposal

**Scope note on file paths.** The system documentation confirms these modules: `mm/mpc/mpc_state.py`, `mm/mpc/mpc_tensor_batched.py`, `mm/mpc/joint_optimizer.py`, `mm/mpc/joint_mpc.py`, `mm/mpc/reward_tensor.py`, `mm/mpc/market_order_flow_torch.py`. The price-model classes (`TorchResidualPriceModel`, `TorchConditionalMechanicalMixturePriceModel`), `soft_fill_batched()`, and `build_joint_target_book_from_sigmoid()` are named in the docs but their containing files are not. I refer to these with an inferred path following the existing `..._torch.py` naming convention (e.g. `mm/mpc/price_model_torch.py`, `mm/mpc/fill_sim_torch.py`) and flag them as **(inferred — confirm path)**. Everything else should match the real tree.

---

## 0. Framing: one unified objective, two new scalar dials

Today:

```
single-venue:  J = CE_beta(R)
joint:         J = w_L * CE_{beta_L}(R^L) + active_G * w_G * CE_{beta_G}(R^G)
```

Proposed:

```
J_total = J_riskadj / R_ref   -   mu_inv * InvOp   +   lambda_bq * BQ
```

- `J_riskadj` is the existing CE-based term (single- or joint-venue), unchanged in form.
- `InvOp` is the new inventory operational-capacity term (Proposal B), normalized to roughly `[0, 1]` purely by config constants (`Q_min, Q_lo, Q_hi, Q_max`) — independent of volatility.
- `BQ` is the new book-quality composite (Proposal D), normalized to roughly `[0, 1]`.
- `R_ref` is a fixed config constant — a reference scale for "typical per-horizon CE magnitude" (e.g. estimated once from historical `mm_optimizer_facts` CE values, default ~1e-4–1e-3). Dividing `J_riskadj` by `R_ref` makes it O(1), so `mu_inv` and `lambda_bq` are also O(1) dials with the intuitive reading "how many units of normalized risk-adjusted-return am I willing to give up for a unit of book quality / inventory safety."

This gives exactly the "one or a small set of scalar weights" requested: **`{lambda_bq, mu_inv}`** as new top-level dials, plus the pre-existing `{beta_L, beta_G, w_L, w_G}`. `R_ref` is a calibration constant, not something an operator tunes day-to-day.

New shared module (new file, ~150 lines total across Proposals B & D):

`mm/mpc/objective_terms_torch.py` — houses `inventory_operational_term()`, `book_quality_term()`, and the stopgap variance term. Imported by both `mm/mpc/reward_tensor.py` (single-venue) and `mm/mpc/joint_mpc.py` (joint), so the two rollout paths share one implementation.

---

## 1. Classification of existing mechanisms

| # | Mechanism | Where (per docs) | Classification | Justification |
|---|---|---|---|---|
| 1 | `quote_size_min_per_side`, `quote_size_max_per_side` (box bounds on `Q_s_total`) | `mpc_tensor_batched.py`, `LParamConfig` | **(a) hard structural** | These define the domain of the `Q_s_total = Q_s_min + sigmoid(u)*(Q_s_max-Q_s_min)` interpolation. Without a well-defined `[Q_min,Q_max]` the parameterization is degenerate. Keep as-is. |
| 2 | `quote_spread_max` (and `quote_spread_min_depth`, first-leg slope floor) | `LParamConfig`, depth-grid clipping | **(a) hard structural**, value may need revisiting | Defines the domain of `x0_eff in [0, spread_max]` and the orderbook grid. The *role* stays hard; Proposal C re-examines whether the *value* leaves headroom for skew. |
| 3 | Raw-variable box constraints (`[-30,30]`, `[1e-6,100]`, `[-spread_max,spread_max]`) | `mpc_tensor_batched.py` Adam closure | **(a) hard structural** | Pure numerical/parameterization domain limits, orthogonal to this proposal. Unchanged. |
| 4 | `quote_capacity_ratio * max(inventory,0)` / `* max(cash,0)/mid` (dynamic `bid_max`/`ask_max`) | capacity computation, `mpc_tensor_batched.py` | **(c) replace** | This conflates a *feasibility* fact (you cannot quote more ask size than you hold) with a *risk preference* (only offer a fraction `quote_capacity_ratio` of it). Proposal C splits these: feasibility becomes a hard `inventory - reserve` cap (item below), and "how much of available inventory to actually offer" becomes a consequence of the skew/book-quality terms, not a multiplicative haircut on the cap. |
| 5 | Imbalance-driven `bid_floor`/`ask_floor` skew (`imbalance`, `skew_strength`) | capacity computation, `mpc_tensor_batched.py` | **(c) replace** | This is a *hard minimum-size floor* used as a proxy for inventory risk. It fights book-quality/balance terms (it forces an asymmetric minimum size regardless of what the objective would choose) and duplicates what the new inventory operational term + skew dimension (Proposals B & C) do properly, through price-offset rather than a forced size floor. Sequencing: remove only after B & C are live (see §6); until then it is the de-facto stopgap inventory control. |
| 6 | Two-sided presence (implicit, via `quote_size_min_per_side > 0`) | capacity computation | **(a) hard-when-feasible; no new soft term** | If `quote_size_min_per_side > 0` and capacity (`bid_max`/`ask_max`) is non-zero, both sides are structurally non-empty already. The only regime where a side legitimately goes to zero is when capacity itself is zero (no cash / no inventory) — a genuine feasibility boundary, not a quality defect. That boundary regime is exactly what Proposal B's zone-of-indifference term is for. **Recommendation: do not add a separate two-sided-presence soft term** — it would be redundant with #1 and with Proposal B. |
| 7 | `soft_fill_batched()` queue/fill mechanics | `mm/mpc/fill_sim_torch.py` (inferred) | **(a) hard structural (mechanism), unchanged** | This is the simulator's physics, not a tunable preference. Out of scope for this proposal except as the place where coupled sampling (Proposal A) needs its outputs consumed consistently. |
| 8 | Independent per-path sampling of flow vs. price innovations | `market_order_flow_torch.py` + price-model file | **(c) replace (architectural)** | Not a "constraint" in the classification sense, but the core architectural gap behind the adverse-selection ask. Proposal A replaces independent sampling with conditional/coupled sampling. |
| 9 | *(new)* Inventory operational capacity term | new, `objective_terms_torch.py` | **(b) soft tunable (backstop)** | Implemented as an additive penalty (`mu_inv * InvOp`), hence graded/tunable by classification — but its *zone-of-indifference* shape means it behaves like a near-hard constraint only near the operational edges, while being fully inert (zero gradient) inside the comfort band. See Proposal B. |
| 10 | *(new)* Book-quality terms (spread tightness, depth, balance, depth-profile uniformity) | new, `objective_terms_torch.py` | **(b) soft tunable** | Genuine preferences with no feasibility content — a wider/tighter/more-balanced book is "nicer" but never infeasible. See Proposal D. |
| 11 | *(new)* Decoupled skew dimension `delta_skew` | `LParamConfig` / `mpc_tensor_batched.py` | **(a) hard structural (new dimension)** | The variable itself is structurally box-constrained (`delta_skew in [-skew_max, skew_max]`, with `skew_max` chosen so `x0_eff` stays in `[0, spread_max]` by construction). What drives its *value* is soft (B & D); the *domain* is hard. See Proposal C. |

---

## 2. Proposal A — Couple the fill/flow sampler with the price-innovation sampler (adverse selection)

**This is the highest-effort item.** Before estimating new work, there is a cheap, high-leverage **Step 0 audit** that may shrink the remaining scope substantially.

### A0. Audit: is the existing flow→price conditioning in `conditional_mechanical_mixture` actually wired into rollout? (S, ~0.5–1 day)

The docs describe this model's branch gate as:

```
q_market = sigmoid(beta_gate[0] + beta_gate[1] * min(log1p(total_flow), 12))
```

and a market-branch regression `market_mean_component = beta_component[0] + beta_component[1] * mechanical_component` with an empirically observed `beta_component[1] ≈ 0.55–0.66`. **Both of these are exactly the flow→price coupling Proposal A is asking for** — if, and only if, during rollout `total_flow` and `mechanical_component` are computed from the *per-path, per-step simulated* flow/book (from `soft_fill_batched()`'s output for that path/step), not from a stale scalar carried over from the last observed frame.

**File/function (inferred — confirm path):** `mm/mpc/price_model_torch.py`, `TorchConditionalMechanicalMixturePriceModel.sample_dynamic_vwap()` (or whatever the rollout-sampling entry point is called), plus its call site in `mm/mpc/joint_optimizer.py` / `mm/mpc/mpc_tensor_batched.py`'s per-step rollout loop.

**Check:**
```python
# At the call site, inspect what is passed as `total_flow` / mechanical inputs:
price_model.sample_dynamic_vwap(
    total_flow=...,            # <- is this buy_vol[:,h]+sell_vol[:,h] (per-path, this step)
                                #    or a single scalar from the last observed MarketEvent?
    mechanical_component=...,  # <- is this recomputed from soft_fill_batched()'s book at (i,h)
                                #    or from the pre-rollout mechanical feature?
)
```

If `total_flow`/`mechanical_component` are per-path-step tensors of shape `(n_samples,)` or `(n_samples, horizon)` derived from the *sampled* `buy_vol`/`sell_vol` and the *simulated* book at step `h`, **adverse-selection coupling already exists for this model** and the remaining work is just confirmation + a regression test (effort: S, done).

If they are scalars carried from the last real frame (the more likely bug, since `total_flow`/mechanical features are naturally "observation-time" quantities), this is **A1**.

### A1. Wire simulated per-path flow/mechanical-state into the mixture model's gate and market branch (M, ~2–3 days, only if A0 finds the gap)

**File/function (inferred):** `mm/mpc/joint_optimizer.py` and `mm/mpc/mpc_tensor_batched.py`, inside the per-horizon-step rollout loop, plus `TorchConditionalMechanicalMixturePriceModel`'s sampling method signature in `mm/mpc/price_model_torch.py`.

Pseudocode change at the rollout step:
```python
# existing, per step h (vectorized over n_samples paths i):
buy_vol_h, sell_vol_h = buy_vol[:, h], sell_vol[:, h]          # (n_samples,)
book_{i,h}, fills = soft_fill_batched(book_{i,h-1}, own_quotes, buy_vol_h, sell_vol_h)

# NEW: derive per-path simulated covariates for the price model
total_flow_sim_h     = buy_vol_h + sell_vol_h                  # (n_samples,)
mechanical_mid_sim_h = mechanical_vwap_delta(book_{i,h}, buy_vol_h, sell_vol_h)  # existing mechanical-delta logic, applied to simulated book

buy_vwap_h, sell_vwap_h, branch_logits_h = price_model.sample_dynamic_vwap(
    total_flow=total_flow_sim_h,
    mechanical_component=mechanical_mid_sim_h,
    ...
)
```

Inside the model, `q_market_h = sigmoid(beta_gate0 + beta_gate1 * clamp(log1p(total_flow_sim_h), max=12))` then varies **per path**, so paths with simulated high flow get a higher probability of drawing from the market (mechanical-regression) branch, whose mean is itself a function of that path's simulated mechanical delta — i.e. paths where the strategy gets heavily filled (high simulated flow) are exactly the paths where price innovation is drawn with a mean shifted by `beta_component[1] ≈ 0.55–0.66 * mechanical_delta`. That *is* the adverse-selection dependency structure, expressed as conditional sampling rather than an explicit copula.

**Dependency-structure summary (math):**
```
Z_{i,h} = (buy_vol_{i,h}, sell_vol_{i,h})            ~ flow model, sampled first
total_flow_{i,h} = buy_vol_{i,h} + sell_vol_{i,h}
mech_{i,h}       = f(book_{i,h}, Z_{i,h})            deterministic given Z and book

branch_{i,h} ~ Bernoulli(sigmoid(beta_gate0 + beta_gate1 * log1p(total_flow_{i,h})))
price_innov_{i,h} | branch=market  ~ N(beta_component0 + beta_component1 * mech_{i,h}, sigma_market^2)
price_innov_{i,h} | branch=refresh ~ StudentT(refresh_mean, refresh_scale, nu)
```
This is a **conditional-sampling / shared-cause** structure: `Z` (flow) is sampled first and *causes* both the branch probability and the market branch's mean — no separate copula machinery needed.

### A2. For the `residual` price model: add a flow-regression term (L, ~3–5 days, only if `residual` is the active local/global price model)

The `residual` model has **no existing flow-conditioning** — its sampling is described as fully independent: `epsilon_side ~ N(mu_side, sigma_side^2)`.

**File/function (inferred):** `mm/mpc/price_model_torch.py`, `TorchResidualPriceModel`:
- training/update method (where `mu_side,t`, `sigma2_side,t` EWMAs are maintained)
- sampling method (where `epsilon_buy`, `epsilon_sell` are drawn)

**Training-side addition** — maintain EWMA covariance between residual and same-step net flow, per side:
```
flow_mean_t  = gamma_f * flow_mean_{t-1}  + (1-gamma_f) * net_flow_t
flow_var_t   = gamma_f * flow_var_{t-1}   + (1-gamma_f) * (net_flow_t - flow_mean_t)^2
cov_side_t   = gamma_f * cov_side_{t-1}   + (1-gamma_f) * (e_side,t - mu_side,t) * (net_flow_t - flow_mean_t)
beta_flow_side_t = cov_side_t / max(flow_var_t, eps)
```

**Rollout-side change** — sampling becomes conditional on the *simulated* per-path flow for that step:
```
resid_var_side = clamp(sigma2_side - beta_flow_side^2 * flow_var, min=floor)
epsilon_side_{i,h} = mu_side
                     + beta_flow_side * (net_flow_{i,h} - flow_mean)
                     + sqrt(resid_var_side) * z_{i,h},   z_{i,h} ~ N(0,1)
```
`net_flow_{i,h}` must be the **already-sampled** flow for path `i`, step `h` — i.e. flow sampling must happen strictly before residual sampling for the same `(i,h)` cell, and both must share the same `i,h` indexing (shape `(n_samples, horizon)`).

### A3. Rollout ordering / threading (S–M, ~1–2 days)

Across `mm/mpc/mpc_tensor_batched.py` (single-venue) and `mm/mpc/joint_optimizer.py` (joint, ×2 for local+global), ensure:
1. Flow tensors `buy_vol, sell_vol` of shape `(n_samples, horizon)` are sampled **once**, up front, as today.
2. Per-step `h`, before calling the price model, compute `net_flow[:, h]` and (for the mixture model) `mechanical_component` from the simulated book at `h`.
3. Pass these as covariates into whichever price-model sampling call is active (A1 or A2 path).
4. No change to `soft_fill_batched()` itself — it already consumes `buy_vol_h, sell_vol_h` and produces the simulated book; A3 only changes what's read *after* that call and fed into the price model for the *same* step.

### Effort roll-up for Proposal A
- A0 (audit): **S, ~0.5–1 day**, do first — determines whether A1 or A2 (or both, if local and global price models differ) is needed.
- A1 (mixture-model wiring): **M, ~2–3 days**, if A0 finds the gap.
- A2 (residual-model `beta_flow`): **L, ~3–5 days**, if `residual` is configured anywhere (local or global).
- A3 (ordering/threading): **S–M, ~1–2 days**, shared prerequisite for A1/A2.

Total realistic range: **3–10 engineering days** depending on which price models are actually configured (per docs, local price model is `residual` or `conditional_mechanical_mixture`; global adds `ewma_gaussian`/static `gaussian`/`residual`/`conditional_mechanical_mixture`/disabled — each active *global* residual/mixture model needs its own A1/A2 pass).

---

## 3. Proposal B — Inventory risk: split wealth-variance (CVaR) from operational capacity

### B1. Wealth-variance risk → already covered by `CE_beta(R)`, *conditional on Proposal A*

No code change here beyond Proposal A itself. Once flow and price innovations are coupled (Proposal A), Monte Carlo paths that accumulate a large position immediately before an adverse move become *representable* in the rollout — they show up as low-`R_i` tail paths, and `CE_beta(R)` (entropic CE ≈ CVaR-like for large `beta`) already discounts those tails. Today, independent sampling means such "fill-then-adverse-move" paths simply don't occur in the path ensemble, so no amount of objective re-weighting on top of `CE_beta` would fix it — the fix has to be in the sampler (Proposal A), not the objective.

**Action:** none, beyond A. **Verification:** after A lands, check that the *distribution* of `R_i` across paths widens (more left-tail mass) for states with large `|Q|`, relative to before — this is the signal that coupling is doing its job and that a separate variance penalty would now double-count.

### B2. Operational capacity term — zone-of-indifference, not volatility-scaled

**New file/function:** `mm/mpc/objective_terms_torch.py`, `inventory_operational_term(Q, Q_min, Q_lo, Q_hi, Q_max, p=2)`.

```
g(Q) =
  0                                     for Q in [Q_lo, Q_hi]              (comfort band — flat)
  ((Q_lo - Q) / (Q_lo - Q_min))^p       for Q in (Q_min, Q_lo)             (approaching short-side limit)
  ((Q - Q_hi) / (Q_max - Q_hi))^p       for Q in (Q_hi, Q_max)             (approaching long-side limit)
  -> saturating barrier as Q -> Q_min or Q -> Q_max  (e.g. clamp g to a large constant, or use p large / a 1/(1-x)-type term right at the boundary)
```

With `p = 2` (or any even `p >= 2`), `g` and `dg/dQ` are both continuous at `Q_lo` and `Q_hi` (both equal 0 at the boundary) — this is the **C¹ smoothness** needed for the handoff discussion in Proposal D. `Q_min`/`Q_max` should be configured with margin *inside* the existing `NetDeltaRiskEngine` hard `min inventory`/`max inventory` thresholds (§4 of the docs), so this term ramps up smoothly before the hard risk engine would ever reject a target book — an early-warning ramp in front of the existing hard wall, not a duplicate of it.

**Applied over the rollout, summed (not just terminal) so every horizon step gets gradient signal:**
```
InvOp_i = (1/H) * sum_{h=1..H} g(Q_{i,h})     # Q_{i,h} = Q^L_h (single venue) or Q^L_h+Q^G_h (joint, consolidated)
InvOp   = mean_i(InvOp_i)
```

Objective contribution: `- mu_inv * InvOp` (subtracted, since `g>=0` is a penalty). **Explicitly no `sigma`/`beta` dependence** — `g` is a pure function of `Q` and four config constants, as required.

**Why this is justified independently of B1:**
- *Model misspecification backstop*: if the price/flow models are wrong, CVaR-based control is miscalibrated by construction — but "don't let the bot run out of base asset to quote asks" is true regardless of model correctness.
- *Short-horizon blindness*: default `horizon = 2`. A 2-step CVaR cannot see a multi-step drift toward depletion; `g(Q)` is evaluated at the *current* simulated `Q`, so it reacts to where the path *is*, not just where it's *heading* over 2 steps.

**File/function to wire it in:**
- `mm/mpc/reward_tensor.py` — single-venue: add `J = CE_beta(R) - mu_inv * InvOp` where `InvOp` is computed from the rollout's `Q^L_{i,h}` series (these are already produced per the docs' single-venue rollout description: `Q_{i,h+1} = Q_{i,h} + inventory_delta_{i,h}`).
- `mm/mpc/joint_mpc.py` — joint: add the same term using consolidated `Q_net_{i,h} = Q^L_{i,h} + Q^G_{i,h}`, applied once (not per-venue) since "running out of operational capacity" is a consolidated-position concept.

**Effort:** M, ~2–3 days (new function + unit tests for the C¹ property at boundaries + wiring into both objective files + new config fields `inv_op.{q_min,q_lo,q_hi,q_max,p}` and `mu_inv`).

**Prerequisite data/infra:** none beyond what the rollout already produces (`Q_{i,h}` series). `Q_min,Q_lo,Q_hi,Q_max` should be derived from existing account-state fields (`inventory`, plus a target/neutral inventory — possibly 0, possibly a configured target) and ideally cross-checked against the risk engine's `min_inventory`/`max_inventory` config so the soft ramp sits strictly inside the hard limits.

### B3. Stopgap, only if Proposal A is deferred (S, ~1 day; explicitly temporary)

If A hasn't landed yet but B2 has, there remains a gap: CE_beta still under-penalizes "large `|Q|` right before an adverse move" because such paths aren't represented. Interim stopgap — a **volatility-scaled** quadratic term (the one term in this whole proposal that *should* scale with volatility, because it's standing in for what coupled CVaR would otherwise capture):

```
StopgapVarTerm_{i,h} = kappa * Q_{i,h}^2 * sigma_price^2
```
where `sigma_price` is the existing price-innovation volatility already estimated by whichever price model is active (`sigma2_side` from `residual`, or branch variances from the mixture model). Add as `- kappa * mean_i(mean_h(StopgapVarTerm))` to the same objective files, gated behind a config flag (`adverse_selection_coupling_enabled: false` → stopgap active; `true` → stopgap disabled) so it's a one-line toggle to remove once Proposal A lands, avoiding double-counting.

**Effort:** S, ~1 day. **Sequencing:** only build this if A's timeline is long; otherwise skip — it's pure throwaway work.

---

## 4. Proposal C — Price-shading vs. size-capping

### C1. Hard-cap quote size at *actual available* inventory/cash (replaces `quote_capacity_ratio`)

**File/function:** capacity computation in `mm/mpc/mpc_tensor_batched.py` (and the equivalent in `joint_optimizer.py` for the global venue), the block currently computing `bid_max`/`ask_max`.

Current:
```
bid_max = min(quote_size_max_per_side, quote_capacity_ratio * max(cash, 0) / mid)
ask_max = min(quote_size_max_per_side, quote_capacity_ratio * max(inventory, 0))
```

Proposed:
```
ask_max_hard = max(inventory - inventory_reserve, 0)        # inventory_reserve: small additive constant (fees/dust), config
bid_max_hard = max(cash - cash_reserve, 0) / mid             # cash_reserve: small additive constant, config

ask_max = min(quote_size_max_per_side, ask_max_hard)
bid_max = min(quote_size_max_per_side, bid_max_hard)
```

Because `Q_s_total = Q_s_min + sigmoid(u_s_depth) * (Q_s_max - Q_s_min)` already structurally bounds `Q_s_total <= Q_s_max`, simply redefining what `Q_s_max` (`ask_max`/`bid_max`) evaluates to gives "never violated by construction" for free — no new clamping logic needed, just a different (and tighter-to-truth) upper bound. `quote_capacity_ratio` is removed (or, if a transition period is wanted, hardcoded to `1.0` and left in config as a deprecated no-op).

**Effort:** S, ~1 day. **Note:** this *increases* available size relative to today (since `quote_capacity_ratio < 1` previously), which is exactly the point — size is no longer the lever for inventory risk, so it can be as large as is genuinely available; risk is pushed onto skew (C2) and the operational term (B2).

### C2. Decoupled skew dimension — inventory risk drives price-offset, not size

**File/function:** `mm/mpc/mpc_tensor_batched.py`, `LParamConfig` (parameterization), plus the raw-vector layout (currently `6*n_components + 2` per venue per step) and its consumer in the per-step rollout loop. Also touches `mm/mpc/joint_optimizer.py` (vector concatenation across local/global venues) and the discretization step `build_joint_target_book_from_sigmoid()` **(inferred — confirm path, likely `mm/application/strategy_runner.py` or a `mm/mpc/target_book_torch.py`)**.

**Parameterization change** — add one new raw scalar per venue per horizon step (`u_skew`), raw vector length becomes `6*n_components + 3`:

```
delta_skew = skew_max * tanh(u_skew)                    # in (-skew_max, skew_max), C^infty, smooth box

# existing spread-structured base offsets (from softmax budget, unchanged):
x0_bid_base, x0_ask_base  with  x0_bid_base + x0_ask_base <= quote_spread_max

# NEW: apply skew symmetrically
x0_bid_eff = clamp(x0_bid_base - delta_skew, 0, quote_spread_max)
x0_ask_eff = clamp(x0_ask_base + delta_skew, 0, quote_spread_max)
```

`skew_max` is a config constant chosen so that, combined with the typical range of `x0_*_base`, `x0_*_eff` rarely saturates at its clamp bounds in normal operation (see C3 for how to check this empirically). The `clamp` makes the structural guarantee `x0_*_eff in [0, quote_spread_max]` hold *by construction* regardless of `delta_skew`'s value — so this is correctly classified **(a) hard structural** for the *domain*, while *what value* `delta_skew` takes is driven by the soft terms (B2's `mu_inv * InvOp` and D's `Bal`).

**Sign convention:** `delta_skew > 0` ⇒ bid pulled toward mid (more aggressive buying discouraged... actually more aggressive *quoting* on bid = tighter bid = more likely to buy) is the wrong direction for "I'm long and want to sell" — pick the convention so that `delta_skew > 0` means *ask pulled toward mid / bid pushed away*, i.e. biases fills toward reducing long inventory. Concretely with the formula above (`x0_bid_eff = x0_bid_base - delta_skew`, `x0_ask_eff = x0_ask_base + delta_skew`): increasing `delta_skew` *decreases* the bid offset (tighter bid, more buy-fill-prone) — that's backwards for "I'm long." **Use the opposite sign**: `x0_bid_eff = x0_bid_base + delta_skew`, `x0_ask_eff = x0_ask_base - delta_skew`, so `delta_skew > 0` ⇒ wider bid (discourage buying) + tighter ask (encourage selling) ⇒ correct direction when long (`Q > Q_hi`, where `g(Q)`'s gradient pushes `delta_skew` positive). Get the sign right against the actual `Q_{i,h+1}` fill-delta convention in `soft_fill_batched()` during implementation/testing — this is a one-line detail but easy to get backwards.

**Effort:** M–L, ~3–5 days (new raw variable + `LParamConfig` plumbing through both single-venue and joint vector layouts + discretization update + tests confirming `x0_eff` domain guarantees under all `(u_skew, x0_base)` combinations).

**Dependencies:** B2 should land first (or at least be stubbed) — `delta_skew` needs *something* in the objective with non-zero gradient outside the comfort band to be meaningful; without B2, C2 adds a free dimension with only `D` (book balance) pulling it toward zero, which is the opposite of what's wanted.

### C3. `quote_spread_max` headroom check (S, ~0.5–1 day, do *before* finalizing C2's `skew_max`)

Pull recent `x0_bid_eff`/`x0_ask_eff` (or the underlying `x0_*_base`) distributions from `mm_optimizer_facts`/`mm_optimizer_component_facts` and compare to configured `quote_spread_max`.

- If typical `x0_base` sits at, say, **<50–60% of `quote_spread_max`** on each side, there's headroom: `skew_max` can be set to roughly `quote_spread_max - max(typical x0_base)` and C2's clamp-based design (§C2) is sufficient as written.
- If typical `x0_base` already sits at **>70–80% of `quote_spread_max`** (i.e. the "fair value" two-sided spread is already close to the cap, likely because `quote_spread_max` was calibrated tightly around maker-fee breakeven), then C2's clamp will frequently saturate `x0_*_eff` at `0` or `quote_spread_max` and skew will be **ineffective exactly when most needed**. In that case, two options:
  - **(i)** Increase `quote_spread_max` modestly and extend `orderbook_grid`/depth-grid clipping to match — low risk, since `quote_spread_max` is a domain bound, not itself a behavior; widening it doesn't force wider quotes, it just gives the optimizer room.
  - **(ii)** Introduce a separate, larger `skew_spread_max >= quote_spread_max` that bounds `x0_*_eff` specifically (i.e. `clamp(x0_base ± delta_skew, 0, skew_spread_max)` rather than `quote_spread_max`), while `x0_base` itself remains bounded by `quote_spread_max`. This lets the "fair value" component stay tight (good for book-quality spread tightness, Proposal D) while skew can push one side further out than the symmetric cap would otherwise allow. Requires extending the depth grid to `skew_spread_max` so the simulated book/fills are correctly represented out to that range.

**Recommendation:** do the audit (S) first; it's nearly free and determines whether C2 ships with the simple clamp or needs (ii)'s extra config dimension — cheaper to decide before building C2 than to retrofit.

---

## 5. Proposal D — Book-quality soft terms

### D1. Term definitions (all normalized to `[0,1]`, dimensionless)

**New file/function:** `mm/mpc/objective_terms_torch.py`, `book_quality_term(...)`, evaluated at the first rollout step (`h=0`, since that's what's actually placed on the book) using `x0_*_eff(0)`, `Q_*_total(0)`, and the depth-grid evaluation of `D_s(x)`.

```
# Spread tightness — reward quotes closer to mid than a configured target spread
T = clamp(1 - (x0_bid_eff(0) + x0_ask_eff(0)) / (2 * spread_target), 0, 1)

# Depth — measured in quote-notional (price * qty) so it's comparable across sides
# and doesn't mix base-currency units with price units
Dpt_bid = clamp(Q_bid_total(0) * mid / depth_target_quote, 0, 1)
Dpt_ask = clamp(Q_ask_total(0) * mid / depth_target_quote, 0, 1)
Dpt = 0.5 * (Dpt_bid + Dpt_ask)

# Balance — symmetry of the *quoted* book (distinct from inventory imbalance)
imb_book = (Q_bid_total(0)*mid - Q_ask_total(0)*mid) / (Q_bid_total(0)*mid + Q_ask_total(0)*mid + eps)
Bal = 1 - abs(imb_book)

# Depth-profile uniformity / constant-slippage — coefficient-of-variation of marginal increments,
# dimensionless by construction
delta_j_s = D_s(x_{j+1}) - D_s(x_j)   for grid points x_j in orderbook_grid, side s in {bid, ask}
CV_s = std(delta_j_s) / (mean(delta_j_s) + eps)
U = 0.5 * (1/(1+CV_bid) + 1/(1+CV_ask))
```

**Two-sided presence:** per §1 item 6, not a separate term — already structurally near-guaranteed by `quote_size_min_per_side > 0`, with the depletion edge case covered by B2.

**Composite:**
```
BQ = w_T*T + w_Dpt*Dpt + w_Bal*Bal + w_U*U,    sum(w_*) = 1
```
`w_T, w_Dpt, w_Bal, w_U` are **secondary** weights with sensible defaults (e.g. equal `0.25` each); not part of the user-facing "small set of scalars" — only `lambda_bq` (the overall multiplier on `BQ` in `J_total`) is meant to be a day-to-day operator dial.

**Dimensional-scaling resolution:** every raw quantity (`x0_eff` in price units, `Q_total` in base units, `D_s` increments in base units) is converted to a `[0,1]` ratio against a *fixed config target* (`spread_target`, `depth_target_quote`) or is naturally scale-free (`Bal`, `U` via CV). The composite `BQ in [0,1]` is then combined with `J_riskadj/R_ref` (also O(1)) via `lambda_bq` — see §0. This avoids ever directly adding "quote-currency PnL" to "base-currency-squared inventory" to "abstract geometric ratios"; everything is pre-normalized to dimensionless `[0,1]` before combination.

**Effort:** M, ~2–3 days (four small term functions + composite + config for `spread_target`, `depth_target_quote`, secondary weights, `lambda_bq` + wiring into `reward_tensor.py`/`joint_mpc.py` alongside B2).

**Prerequisites:** `depth_target_quote` and `spread_target` should be derived from recent market conditions (e.g. a rolling percentile of observed top-of-book depth/spread on the venue) rather than hardcoded — otherwise `BQ` will be persistently saturated at 0 or 1 across regime changes. This rolling-target estimation can reuse the existing EWMA infrastructure already present for flow/price models (no new statistical machinery, just a new EWMA channel fed from `MarketEvent`).

### D2. Interaction with the inventory zone-of-indifference — smooth handoff

`Bal` (book balance, wants `delta_skew ≈ 0`) and B2's `g(Q)` (wants `delta_skew` pushed in the inventory-reducing direction once `Q` is outside `[Q_lo,Q_hi]`) act on the *same* variable (`delta_skew`) with opposing preferences when `Q` is off-target. The handoff between these regimes is governed entirely by `dg/dQ`:

- **Inside `[Q_lo, Q_hi]`** (comfort band): `g(Q) ≡ 0` and (with `p>=2`) `dg/dQ ≡ 0` throughout the band. The inventory term contributes **exactly zero gradient** to `delta_skew` here — `delta_skew` is driven purely by `Bal` (pulling toward 0, i.e. a balanced book) and by whatever directional preference `CE_beta(R)` itself has (e.g. if the price model predicts a drift, some skew may still be "wanted" for pure-PnL reasons — that's a legitimate interaction, not a conflict to resolve).
- **Outside the band**: with `p=2`, `g(Q) = ((Q-Q_hi)/(Q_max-Q_hi))^2` for `Q>Q_hi` (symmetric for `Q<Q_lo`). At `Q=Q_hi`, `g=0` and `dg/dQ = 2(Q-Q_hi)/(Q_max-Q_hi)^2 |_{Q=Q_hi} = 0`. So **the gradient is continuous and starts at exactly zero at the boundary**, then grows smoothly (linearly, for `p=2`) as `Q` moves further from `Q_hi` toward `Q_max`. There is **no discontinuity** — the inventory term's influence on `delta_skew` ramps up continuously from zero, smoothly overtaking `Bal`'s influence as `|Q - Q_target|` grows.

**Implementation requirement to flag:** this smoothness depends on `g` being built from an **even power `>=2`** of `(Q - boundary)`, *not* `abs(Q - boundary)` (which has a gradient discontinuity — a kink — right at the boundary) and *not* a piecewise-linear ramp (continuous value but discontinuous gradient). Use `**2` (or a higher even power for sharper near-edge behavior) in `objective_terms_torch.py`.

**Relative scaling of `mu_inv` vs `lambda_bq`:** because `g(Q)=0` identically inside the band regardless of `mu_inv`'s magnitude, `mu_inv` only matters *outside* the band — it governs "how hard do we react once outside the comfort zone," while `lambda_bq` governs "how much PnL do we sacrifice for book quality while everything is fine." These two dials have **largely non-overlapping regimes of influence**, which is a desirable property: an operator can tune `lambda_bq` for normal-conditions book quality without it fighting `mu_inv`'s edge-case behavior, and vice versa.

---

## 6. Ordering, dependencies, and impact-per-hour

| Order | Item | Effort | Why this position |
|---|---|---|---|
| 1 | **A0** — audit mixture-model flow→price wiring | S (0.5–1 day) | Cheapest possible item; potentially reveals that most of Proposal A's value is one wiring fix away. Also determines whether B3's stopgap is needed at all. |
| 2 | **B2** — inventory operational term (zone-of-indifference) | M (2–3 days) | Foundational and *independent* of A/C/D — can ship immediately. Required as the gradient source for C2. Directly addresses "less risky... good depth" on its own. |
| 3 | **C1** — true-available-inventory hard cap (replace `quote_capacity_ratio`) | S (~1 day) | Small, low-risk, and frees up size headroom that C2/D1 depth terms need to be meaningful (today's `quote_capacity_ratio<1` artificially shrinks `Q_s_max`). |
| 4 | **C3** — `quote_spread_max` headroom audit | S (0.5–1 day) | Cheap empirical check that determines C2's design variant (simple clamp vs. `skew_spread_max`). Do before building C2 to avoid rework. |
| 5 | **C2** — decoupled `delta_skew` dimension | M–L (3–5 days) | The centerpiece of the price-shading/size-capping ask. Depends on B2 (for a meaningful gradient) and C1/C3 (for the capacity/domain it operates within). |
| 6 | **D1** — book-quality composite term | M (2–3 days) | Needs `delta_skew` (C2) and uncapped-by-ratio sizing (C1) to actually be expressible in the book — wiring it before C1/C2 would let it fight the old floor/ratio mechanisms. Can be developed in parallel with C2 and merged together. |
| 7 | **A1/A2/A3** — full adverse-selection coupling (mixture-model wiring or residual `beta_flow`) | M–L per active price model (2–5 days each) | Largest single item; scope depends entirely on A0's finding and on which price models (local/global) are configured. Sequenced after the objective-side work (B/C/D) because those are independent and lower-risk; A also benefits from B2 already being in place (B1's "no double-counting" argument is cleanest once both exist). |
| 8 | **B3** — stopgap variance term | S (1 day), *conditional* | Build **only if** A0/A1/A2 timeline is long *and* B2 alone is judged insufficient in the interim. Skip entirely if A lands within the same cycle as B2. |
| 9 | **§1 item 5 cleanup** — remove imbalance-driven `bid_floor`/`ask_floor` skew | S (~0.5 day) | Last, and gated: this floor is the *current* (crude) inventory control. Remove only once B2+C2 are live and validated — removing it earlier leaves a window with no inventory-risk mechanism at all. |

**Critical path:** B2 → (C1, C3 in parallel) → C2 → D1, with A's sub-items running in parallel starting from A0 and merging in before item 9's cleanup. Item 9 is the only hard gate: it must not land before C2 is validated in shadow/paper mode, since it's the removal of the only currently-active inventory control.

**Overall effort estimate:** roughly **3–4 engineering weeks** for items 1–6 and 8–9 (the "objective redesign" core), plus **3–10 additional days** for item 7 depending on price-model configuration — item 7 is the most variable and is recommended to start in parallel (from A0) but isn't on the critical path for shipping the tunable `{lambda_bq, mu_inv}` trade-off itself, since B1's benefit from A is a *quality* improvement to the existing CE term rather than a prerequisite for B2/C/D to function.
