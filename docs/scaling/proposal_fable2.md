# Optimizer Objective Improvement Proposal

Concrete, incrementally implementable improvements to achieve a profitable, balanced, and less risky order book with good depth and spread — as a tunable trade-off between a risk-adjusted return objective and a book-quality objective, controllable via one or a small set of scalar weights.

Grounded in direct inspection of:

- `mm/mpc/mpc_tensor_batched.py` (rollout closure)
- `mm/mpc/l_param.py` (decoder)
- `mm/mpc/market_order_flow_torch.py`, `mm/mpc/price_torch.py` (samplers)
- `mm/mpc/reward_tensor.py`, `mm/mpc/joint_mpc.py` (objectives)
- `mm/strategies/negative_objective_policy.py` (post-hoc policy)

**Terminology note:** The current risk term is an **entropic certainty equivalent** (`objective_entropic_return`), not CVaR. It plays the same role assigned to "the CVaR term" — it prices tail risk of the per-path return distribution. Risk that is not *present in the sampled paths* is invisible to it. That is exactly why the coupling item (P5) matters.

---

## 0. Proposal ordering by impact per engineering hour

| # | Proposal | Effort | Depends on |
|---|----------|--------|------------|
| P1 | Book-quality soft objective + master weights | ~1–2 days | — |
| P2 | Inventory zone-of-indifference term | ~1 day | P1 (plumbing) |
| P3 | Decoder cleanup: hard inventory cap, retire imbalance skew floors | ~0.5–1 day | P2 (zone term replaces floors) |
| P4 | Stopgap volatility-scaled inventory term | ~3–4 hours | P1; removed after P5 |
| P5 | Flow–price coupling (adverse selection in rollout) | ~1.5–2 weeks | — (but do P1–P3 first for ROI) |
| P6 | Demote `negative_objective_action` to safety valve | ~2 hours | P1 |

**Rationale for the order:** P1+P2 deliver the user-visible objective ("tunable trade-off via a small set of scalar weights") with small, low-risk diffs in two files that already have the right structure (a `components` dict flowing into diagnostics and Grafana). P5 is the highest-effort, highest-modeling-risk item; doing it last means the soft-objective scaffolding and its dashboards already exist to evaluate it.

---

## 1. Classification of existing mechanisms

**(a) = binary feasibility, keep/move into decoder. (b) = graded preference, convert to soft term. (c) = replace architecturally.**

### 1.1 Dynamic capacity caps

**Location:** `_quote_capacity_bounds_batched()` in `mm/mpc/mpc_tensor_batched.py` (lines 703–729):

```text
bid_max = min(hard_cap, ratio·cash⁺/mid)
ask_max = min(hard_cap, ratio·inventory⁺)
```

**Classification: (a) for the cap at `ratio = 1`, (b) for `ratio < 1`.**

"Cannot offer base you don't hold, cannot bid cash you don't have" is binary feasibility and is *already structural*: the cap enters the decoder via `quote_size_from_depth_logit_torch()` (`mm/mpc/l_param.py` lines 108–139), where `q_total = lo + sigmoid(u)·(hi − lo)` can never exceed `hi`. Keep that.

But `quote_capacity_ratio = 0.5` (the example config) is a *graded reserve preference* smuggled into a hard cap — it makes half the balance sheet permanently unusable regardless of opportunity. Split it: hard cap at `ratio = 1` (feasibility), and let the inventory-zone term (P2) express "don't deploy everything" as a graded cost.

**Justification:** A hard 50% reserve is not a feasibility fact; it is a risk preference, and risk preferences belong in the objective where the optimizer can trade them against expected return.

### 1.2 Imbalance skew floors

**Location:** Same function:

```text
bid_floor = q_min·(1 + skew·relu(−imbalance))
```

**Classification: (c) replace.**

This mechanism *raises the minimum quote size* on the rebalancing side. Three defects:

1. It forces size-based rebalancing (dumping/accumulating inventory by quoting bigger, i.e. paying adverse selection to rebalance) instead of price-based rebalancing (shading, i.e. getting *paid* to rebalance).
2. It has gradient kinks at `imbalance = 0` (the `relu`) and at the `min(floor, max)` crossover.
3. It fights the optimizer rather than informing it — the optimizer cannot decide the floor is wrong.

Replace with the zone-of-indifference term (P2), which produces skew through the objective gradient, primarily moving `x0` (price) rather than `L` (size).

The symmetric base `quote_size_min_per_side` is a different thing — see next item.

### 1.3 `quote_size_min_per_side`

**Classification: split (a)/(b).**

As "venue minimum order size," it is feasibility — but that is already enforced at the right place, the execution boundary (`ExchangeRulesGateway`). As "always show at least X on both sides," it is a *two-sided-presence preference* and currently it is a hard floor that forces quoting even when the model says quoting is toxic.

Convert the presence role into the soft presence term in P1; keep only the venue-lot-size floor structurally.

**Justification:** An MM with no contractual obligation should be able to pull one side when expected adverse selection exceeds the book-quality reward; a hard floor makes that impossible and is exactly the kind of constraint that turns a bad hour into a bad day.

### 1.4 `quote_size_max_per_side`

**Classification: (a) keep.**

Operational blast-radius cap (fat-finger/model-explosion bound). Binary by nature, cheap, already structural.

### 1.5 `quote_spread_max` + `quote_spread_min_depth` spread structure

**Location:** `_spread_centers_from_flat()` / `materialize_side_torch()` in `mm/mpc/l_param.py` (lines 230–251, 554–607):

```text
x0_bid,1 + x0_ask,1 ≤ quote_spread_max   (via scaled softmax)
first leg carries ≥ quote_spread_min_depth
```

**Classification: (a) *if and only if* it encodes a contractual MM obligation** (max quoted spread with min depth, as exchanges' MM agreements specify); **otherwise (b).**

Keep the structured decoder as an *opt-in contract mode* — it is well built (smooth, sum-capped, k-floored). For the non-contractual default, disable it (note the example config already effectively disables it locally via `quote_spread_min_depth: 0.0`) and express spread tightness as a soft term (P1).

**Justification:** A hard touch-spread cap with no obligation behind it forces tight quotes precisely in the regimes where tight quotes are most adversely selected.

### 1.6 `negative_objective_action` floor/zero policy

**Location:** `apply_negative_objective_policy()` in `mm/strategies/negative_objective_policy.py`.

**Classification: (c) replace, retain as safety valve.**

It is a *discontinuous post-hoc override*: at `risk_adjusted_return = 0⁻` the book jumps from the optimizer's solution to floor quotes or no quotes. Once book-quality has positive weight in the objective (P1), "slightly negative expected return but good book" beats "no book" *inside* the optimizer, continuously. Keep only `zero_action` behind a much more negative threshold as a circuit breaker. (P6.)

### 1.7 Inventory-derived VWAP depth and side-aware marking

Not a constraint; it is the valuation rule. **Keep unchanged.**

### 1.8 `NetDeltaRiskEngine` limits

Min/max inventory, notional caps, staleness gates — **(a)**, and correctly placed *outside* the optimizer as an independent backstop. Do not move these into the objective; their job is to catch the optimizer being wrong.

---

## 2. P1 — Book-quality soft objective with master weights

### Files / functions

- **New:** `mm/mpc/book_quality.py` (term computation)
- `mm/mpc/reward_tensor.py::terminal_objective()` (lines 129–150) — accept and combine new components
- `mm/mpc/joint_mpc.py::joint_terminal_objective()` (line 180) — same for joint mode
- `mm/mpc/mpc_tensor_batched.py` closure (lines 1147–1426) — compute terms from decoded h=0 action and pass into objective; extend `VenueOptimizerParams`
- `mm/strategies/mpc_engine.py::_VENUE_OPTIMIZER_KEYS` and `mm/runtime/config.py` — config plumbing

### Design

All terms are computed on the **h=0 decoded action** (the only step that is executed) via `materialize_side_torch()` — no extra rollout cost, clean gradients.

Every term is a **dimensionless ratio in [0, 1]**, which is the answer to the dimensional-scaling problem: the PnL term is already dimensionless (`R = ΔW/W₀`), so we put *everything* in return-equivalent units and never mix quote-currency, base-currency², or abstract geometry:

```text
# decoded curves: D_bid(s), D_ask(s) cumulative depth; marginal m_side[j] on grid
Q_b = D_bid(spread_max);  Q_a = D_ask(spread_max);  q_ref = configured reference size

presence   = smoothmin(D_bid(s_touch), D_ask(s_touch)) / (q_ref + smoothmin(...))   # [0,1)
tightness  = exp(-(x0_bid,1 + x0_ask,1) / s_ref)                                    # [0,1]
balance    = 1 - |Q_b - Q_a| / (Q_b + Q_a + eps)                                    # [0,1]
uniformity = 1 - Var_j(m[j]) / (mean_j(m[j])^2 + Var_j(m[j]))                       # [0,1]
             # = constant-slippage shape; normalized cousin of slippage_penalty_tensor()
             #   in mm/mpc/state_tensor.py (line 370)

B = v_p*presence + v_t*tightness + v_b*balance + v_u*uniformity     # v_* fixed, sum to 1
J = CE_beta(R) + w_book * B - w_inv * Z          # Z from P2
```

Operators tune **two scalars**: `w_book` and `w_inv` (per venue, in `optimizer.local` / `optimizer.global`, validated like existing keys). The sub-weights `v_*` ship as defaults and rarely move.

Interpretation: `w_book = 0.002` means "one full unit of book quality is worth 0.2% of wealth per horizon" — directly comparable to the CE term.

Add each term into the `components` dict that already flows to `OptimizerRunDiagnostics` → `mm_optimizer_component_facts` → the optimizer Grafana dashboard; this gives per-frame visibility of the trade-off for free.

### Caveat

`best_risk_adjusted_return()` in `negative_objective_policy.py` reads `components["risk_adjusted_return"]` — keep that component pure (CE only) so the policy and dashboards don't silently change meaning.

### Effort / prerequisites

- **Effort:** ~150–250 lines + tests
- **Prerequisites:** None

---

## 3. P2 — Inventory risk split: CE for wealth variance, explicit operational-capacity zone term

### Separation of concerns

1. **Wealth-variance inventory risk** is priced by the entropic CE **once price innovations correlate with fills** (P5). Today they don't (see §5), so the CE materially underprices holding inventory — hence P4 as a stopgap.

2. **Operational capacity** is a different risk that no wealth-distribution term can see: at `Q ≈ 0` the desk structurally cannot quote asks next frame (the decoder's own `ask_max = inventory` guarantees this), and at `Q ≈ Q_max` it cannot quote bids; with `horizon = 2` (~20 s of lookahead at 10 s intervals), the rollout is blind to the hours of lost quoting revenue that follow.

This term is a backstop against **short-horizon blindness and model misspecification**, and therefore must **not** be scaled by model volatility — if the model says σ ≈ 0, that is precisely when you trust it least at the boundaries.

### Shape — zone of indifference, bounded, asymmetric

```text
q = Q * mid / W0                      # value-fraction inventory, dimensionless
Z(q) = a_lo * sigmoid((q_lo - q)/d_lo) + a_hi * sigmoid((q - q_hi)/d_hi)
```

- Flat (`Z ≈ 0`, `Z' ≈ 0`) inside the band `[q_lo, q_hi]` — genuine indifference, not a quadratic that always pulls toward a point target (consistent with removal of old `target_inventory`/`inv_coef` keys from config).
- Bounded (saturates at `a_lo`, `a_hi`) — cannot dominate the objective into pathological actions.
- Asymmetric — `a_lo > a_hi` typically, because running out of base kills the ask side entirely while excess inventory still allows two-sided quoting.

### Placement

Penalize the *trajectory*, outside the CE (deterministic capacity cost, not a wealth-distribution feature):

```text
Z_path = mean_i mean_h Z(q_{i,h})     # q_{i,h} from inv_t already tracked in the closure
J = CE_beta(R) + w_book*B - w_inv*Z_path
```

`inv_t` and `mid_t` are already maintained per step in the closure (lines 1169–1365).

### Files / config

- New function in `mm/mpc/book_quality.py` or `reward_tensor.py`
- Accumulation in closure of `mpc_tensor_batched.py` and in `joint_optimizer.py`
- Joint mode: apply to consolidated `Q^L + Q^G` for wealth band, per-venue for capacity band (start with per-venue only)
- Config: `inventory_zone: {q_lo, q_hi, d_lo, d_hi, a_lo, a_hi}` under each venue optimizer block

### Effort / prerequisites

- **Effort:** ~80–120 lines + tests
- **Prerequisites:** P1 (objective plumbing)

### Stopgap if P5 is not done first (P4)

The CE-blindness gap is wealth-variance, so the interim term *should* be volatility-scaled (unlike the zone term) and per-path:

```text
R_i += - kappa * sigma_step * sqrt(H) * |q_{i,H}|      # sigma_step from price model innovation sigma, in return units
```

~30 lines in the closure, explicitly marked temporary, deleted when P5 lands. Without this, P1+P2 alone will let the optimizer warehouse inventory too cheaply.

- **Effort:** ~3–4 hours
- **Prerequisites:** P1

---

## 4. P3 — Hard size caps by construction; shading, not shrinking; `quote_spread_max` interaction

### Size hard cap

The requirement "quote size never exceeds available inventory, by construction" is *almost* already met: dynamic `ask_max` flows into the decoder's sigmoid range and is recomputed each rollout step from simulated `inv_t`.

Two fixes in `_quote_capacity_bounds_batched()` / `_scalar()` (`mm/mpc/mpc_tensor_batched.py` lines 703–755):

```text
ask_max = min(hard_cap, inventory+)              # remove ratio from ask side: feasibility is ratio=1
bid_max = min(hard_cap, cash+ / mid)             # same for bids
# deprecate quote_imbalance_skew_strength: bid_min = ask_min = venue_lot_floor
```

Reserve preferences move to the zone term; rejecting `quote_capacity_ratio ≠ 1` can follow the codebase's strict-migration pattern in `runtime/config.py`. Execution-side backstops (`ExchangeRulesGateway`, `NetDeltaRiskEngine`) stay as independent layers.

### Skew through price

With floors symmetric and the zone term active, the gradient `∂Z/∂q` reaches the action through the fill simulator: shifting `x0_bid` outward / `x0_ask` inward changes expected fills directionally at much lower cost than size changes (price-shading earns the spread on the rebalancing side; size-dumping pays it).

No new mechanism is needed — `x0` is already free in `[0, spread_max]` (sigmoid-mapped, line 571 of `l_param.py`); the term in P2 supplies the missing gradient.

### `quote_spread_max` interaction report

Findings from the decoder and example config:

1. In the example config, the **local** venue has `quote_spread_min_depth: 0.0`, so `spread_structured` is `False` (`l_param.py` lines 60–66) and the sum-cap **does not bind locally at all** — full shading range `[0, spread_max]` per side. The cap currently binds only on the global venue.

2. When the structure *is* enabled: the constraint caps the **sum** `x0_bid,1 + x0_ask,1 ≤ quote_spread_max` via softmax split. Asymmetry is unrestricted — one side can take ~the whole cap while the other drops to the tiny center floor (`_min_structured_center`). So skew-as-*reallocation* works. What the sum cap forbids is *widening both sides* — backing off entirely — and shading one side beyond `cap − x0_other,1`. Critically, the loaded side at the cap forces the *other* side toward the touch, i.e. maximum adverse-selection exposure exactly when inventory is already wrong.

3. **Tail components escape the cap:** `x0_i = x0_first + (spread_max − x0_first)·sigmoid(·)` for `i ≥ 2` can sit anywhere out to `spread_max`, and only `quote_spread_min_depth` must stay on the first leg. So with `quote_spread_min_depth` small relative to total size, most depth on the loaded side can retreat to the tails even under the cap.

4. **Recommendations:**
   - Treat structured mode as contract-only (per classification §1.5)
   - In contract mode, keep `quote_spread_min_depth` at the contractual minimum, not more, so tail legs carry inventory-skewed depth
   - Ensure `spread_max` comfortably exceeds `quote_spread_max` (example sets them equal locally, 200/200, which would leave *zero* tail room if structure were enabled — worth a config-validation warning)
   - Discretization: `spread_step: 20` over `spread_max: 200` gives 10 ladder points; price shading quantizes to 20 quote units at execution

### Effort / prerequisites

- **Effort:** ~0.5–1 day including config migration and test updates (`test_quote_size_softmax_L.py`, `test_mpc_spread_depth.py`)
- **Prerequisites:** P2 (zone term replaces skew floors)

---

## 5. P5 — Adverse selection structurally inside the rollout (highest effort)

### Current dependency structure

In `draw_samples()` (`mpc_tensor_batched.py` lines 1068–1096; mirrored at `joint_optimizer.py` lines 233–280), flow and price randomness are **independent**:

```python
price_innov = None
if hasattr(self.local_price_model, "sample_innovations"):
    price_innov = self.local_price_model.sample_innovations(...)
    eps_buy = torch_mod.zeros(...)
    eps_sell = torch_mod.zeros(...)
elif hasattr(self.local_price_model, "sample_residual"):
    eps_buy, eps_sell = self.local_price_model.sample_residual(...)
else:
    eps_buy = torch_mod.zeros(...)
    eps_sell = torch_mod.zeros(...)
return buy_vol, sell_vol, eps_buy, eps_sell, price_innov
```

Within a flow model, buy and sell are also drawn independently.

**Existing partial channel:** The CMM model already has a *mechanical* adverse-selection channel — `sample_dynamic_vwap()` conditions on the post-flow book (`mech_mid` from `_mid_from_book_best_levels_batched` after `soft_fill_batched`), so heavy one-sided flow mechanically moves the predicted price.

**What is missing:** Correlation between flow direction and the **innovation** — the informed-flow component where price *keeps going* after eating your quote. For the `residual` model there is no coupling at all.

### Proposed structure: shared latent factor per (path, step), Gaussian copula

```text
g_{i,h} ~ N(0,1)                                  # latent "informed direction" factor

# Flow side (copula via inverse CDF):
z_buy  =  rho_f * g + sqrt(1-rho_f^2) * eta_b     # eta_b, eta_s ~ N(0,1) iid
z_sell = -rho_f * g + sqrt(1-rho_f^2) * eta_s
V_buy  = F_buy^{-1}(Phi(z_buy))                   # F = model's marginal CDF (hurdle handled
V_sell = F_sell^{-1}(Phi(z_sell))                 #   naturally: V=0 iff Phi(z) < p0)

# Price side:
#   residual model: eps_buy = mu_b + sigma_b*(rho_p*g + sqrt(1-rho_p^2)*zeta_b), same for sell
#   CMM model:      z_market["mid"] := rho_p*g + sqrt(1-rho_p^2)*z    (other components unchanged)
```

Positive `g` ⇒ more market buys, fewer sells, **and** an upward price innovation: ask fills now co-occur with price appreciation in the same path. Adverse selection appears in the left tail of the per-path return distribution, where the entropic CE prices it — no penalty term, and the optimizer's response (wider/smaller asks when `rho` is high) emerges from the gradient.

Marginals are preserved exactly (copula property), so existing model scoring (`score_frame_models`) stays valid.

### Online estimation of `(rho_f, rho_p)`

EWMA correlation, updated where models already update per frame (`MpcStrategyEngine.decide()` path in `mm/strategies/mpc_engine.py`):

```text
s_t   = signed flow imbalance  = (V_buy_t - V_sell_t) / (V_buy_t + V_sell_t + eps)
r_t   = price innovation residual = observed_mid_delta_t - mechanical_mid_delta_t
                                   # CMM update already computes this pair
rho_p = EWMA_corr(s_t, r_t)
rho_f = calibrated from EWMA_corr of (V_buy, V_sell) ranks
```

### Sub-steps (exceeds 200 lines; split as)

1. **Inverse-CDF sampling refactor** (`mm/mpc/market_order_flow_torch.py`): add `sample_from_uniforms(U_buy, U_sell)` per model. Analytic for Weibull, lognormal, log-Student-t, GPD, and hurdle variants; Gamma lacks analytic icdf — use Wilson–Hilferty approximation or defer Gamma support. Keep `sample()` delegating for uncoupled mode. (~150–200 lines + tests.)

2. **Coupling module** (`new mm/mpc/coupling.py`): latent draw, copula assembly, `CouplingParams` with EWMA `rho` estimator and update/score functions. (~100–140 lines.)

3. **Estimator wiring**: update call in model-update path of `mpc_engine.py`; expose `rho_f`, `rho_p` through `ModelUpdateReport` / `mm_model_score_facts`. (~60–80 lines.)

4. **Rollout wiring**: rewrite both `draw_samples()` implementations to draw `g` once and thread through flow and price samplers; config flag `controller.coupling: {enabled, decay, rho_floor/cap}`; A/B via config. (~80–120 lines + contract tests for marginal preservation and induced correlation sign.)

### Effort / prerequisites

- **Effort:** ~1.5–2 weeks including tests
- **Prerequisites:** None (but P1–P3 recommended first for evaluation scaffolding)
- **Data/infra:** None new — `(s_t, r_t)` derivable from inputs the CMM update already touches; `mm_market_facts` allows offline backtesting of estimator before enabling

---

## 6. P6 — Demote `negative_objective_action`

With P1, replace the default `floor_action` by `optimizer_action` and keep `zero_action` behind a strongly negative threshold (e.g. `risk_adjusted_return < -tau`, `tau` in return units) as a circuit breaker.

**Files:** `apply_negative_objective_policy()` in `mm/strategies/negative_objective_policy.py`; config default in example JSON and `runtime/config.py`.

- **Effort:** ~2 hours
- **Prerequisites:** P1

---

## 7. Regime handoff: smooth transition between book-quality and inventory dominance

Both the balance term (P1) and the zone term (P2) are smooth (C∞) in the action and inventory respectively.

**Inside the band:** `∂Z/∂q ≈ a·e^{-(q_hi-q)/d}/d ≈ 0`, so the book-quality gradient — which always prefers a symmetric, tight, uniform book — is unopposed. **Near-target inventory ⇒ book quality dominates.**

**Approaching a band edge:** `|∂Z/∂q|` rises smoothly to its maximum `w_inv·a/(4d)` at the edge, progressively outweighing `w_book·∂B/∂(skew)` and tilting `x0`/depth allocation. **Off-target inventory ⇒ zone term dominates and skews the book.**

The crossover happens over a width of order `4d` in value-fraction inventory; setting `d ≈ 10–20%` of the band half-width gives a gradual handoff with no kink.

The two genuine discontinuities in the *current* system — the `negative_objective_action` jump and the `relu`-kinked imbalance floors — are both removed by P6 and P3, so the proposed system is smoother than the status quo.

The one residual discontinuity that remains, intentionally, is the hard feasibility cap (`ask_max = inventory`): at `q = 0` the ask side vanishes by construction — which is exactly the cliff the asymmetric `a_lo` zone wall exists to keep you away from.

---

## 8. Summary dependency graph

```text
P1 (book-quality objective)
 ├── P2 (inventory zone term)
 │    └── P3 (decoder cleanup, retire skew floors)
 ├── P4 (stopgap vol-scaled inventory; interim until P5)
 └── P6 (demote negative_objective_action)

P5 (flow–price coupling) — independent track; do after P1–P3 for ROI; remove P4 when done
```

---

## 9. Recommended next implementation step

Implement **P1 + P2** first:

- Exact diffs to `terminal_objective`, the rollout closure, `VenueOptimizerParams`, and config schema
- Unit tests for dimensional consistency and component diagnostics
- Update example config with default `w_book`, `w_inv`, and `inventory_zone` values

This unlocks the tunable trade-off immediately with minimal modeling risk.
