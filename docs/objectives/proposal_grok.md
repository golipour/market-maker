**Proposed improvements prioritize high-impact changes first (by expected P&L stability / book quality per engineering hour), with clear dependencies.** All changes aim for a tunable trade-off: primary objective remains risk-adjusted return (entropic CE / CVaR-like via beta), augmented by a small number of scalar book-quality weights (e.g., `w_book_quality: 0.0-1.0`, `w_balance: 0.0-0.5`, `w_uniformity: 0.0-0.3` in `ControllerRuntimeConfig`). These are added as additive terms in the terminal objective (or per-horizon for smoother gradients), normalized to comparable scale (e.g., via z-score over samples or reference to initial wealth / spread units).

This creates a Pareto front controllable via scalars: higher book-quality weights favor tighter/balanced/deeper books at expense of raw return (tested via offline replay in `mm/persistence/frame_replay.py` and `mm/application/frame_simulator.py`).

### Classification of Existing Mechanisms
From the doc (state construction, capacity, action parameterization, risk engine):

- **Imbalance skew floors** (`imbalance` clamp driving `bid_floor`/`ask_floor` in capacity): **(b) tunable soft objective term**. Justification: Current hard floor is a crude linear skew. Convert to soft penalty in objective (e.g., quadratic or zone-based deviation from target skew) for smoother optimization and tunability. Retain as weak lower bound in decoder for feasibility.

- **Capacity ratio** (`quote_capacity_ratio * max(cash/inv,0)`): **(a) hard structural constraint** in decoder/parameterization. Justification: Pure feasibility (can't quote more than you have). Enforce strictly in `LParamConfig` / capacity computation to prevent invalid actions by construction; no need for objective term.

- **Spread caps** (`quote_spread_max`, `spread_max` in grids/parameterization): **(a) hard structural** (keep in sigmoid centers / box constraints) + **(b) soft preference** via book-quality term for tightness. Justification: Hard cap prevents unrealistic quotes; softness via objective encourages tighter spreads when safe.

- **Min/max size per side** (`quote_size_min_per_side`, `quote_size_max_per_side`, floors): **(a) hard** for max (capacity) and min (in spread-structured first leg); **(b) soft** for overall via book-quality depth term. Justification: Max is feasibility; min can be relaxed into objective for better trade-off.

- **Fractional-inventory sizing** (implicit in dynamic VWAP depth / capacity from `abs(inventory)`): **(c) replace with zone-of-indifference operational term** (see below). Justification: Pure fractional scaling couples too tightly to short-horizon noise; better as explicit backstop.

### 1. Adverse Selection Coupling (Highest Effort, Highest Impact)
**Proposal**: Couple fill/flow sampler (`soft_fill_batched`, market order flow models) with price-innovation sampler (`sample_residual` / `sample_dynamic_vwap` / `sample_innovations` in price models) via **shared latent factor or conditional sampling** per MC path. Adverse selection emerges naturally: large aggressive flow (high sampled `buy_vol`/`sell_vol`) correlates with worse (adverse) price moves in the same path.

**Why structural (not separate penalty)**: Penalties are brittle and require tuning; joint sampling makes it a first-class feature of rollouts. Use a Gaussian copula or simple shared noise factor (e.g., one common `z ~ N(0,1)` scaled per model family) for tractability.

**Exact location**: Primarily `mm/mpc/joint_optimizer.py` (joint rollout) and `MonteCarloMPCTensorBatched.optimize_with_diagnostics` / rollout loop in the tensor file (single-venue fallback). Also touches `mm/mpc/market_order_flow_torch.py` (flow `.sample()`) and price models (`TorchResidualPriceModel`, `TorchConditionalMechanicalMixturePriceModel`).

**Pseudocode/Math change** (in rollout, per path `i`, step `h`):
```python
# Existing (independent)
flow_samples = flow_model.sample((n_samples, horizon))  # buy_vol, sell_vol
price_innovs = price_model.sample_innovations(...)     # eps_buy, eps_sell or deltas

# New: conditional / copula
z_shared = torch.randn(n_samples, horizon)  # or per-side
flow_s = flow_model.sample_conditional(z_shared)   # e.g., scale Gamma/Weibull params by z
innov_s = price_model.sample_conditional(flow_s, z_shared)  # e.g., mu_eps += rho * flow_deviation

# Then soft_fill_batched(..., flow_s) -> mechanical_vwap + innov_s
```

**Dependency structure**:
- Flow model exposes `sample_conditional(latent)` (add thin wrapper).
- Price model conditions on realized/ sampled flow (e.g., for residual: larger flow → larger |eps| or biased sign; for mechanical mixture: higher flow prob → market branch with adverse mean).
- Joint rollout passes shared latents across local/global.

**Effort estimate**: High (300-600+ LOC initially). **Break into sub-steps**:
1. (Low effort, ~50 LOC) Add shared `z` sampling + passthrough in rollout (no conditioning yet).
2. (~100 LOC) Extend flow models with optional conditional param shift.
3. (~150 LOC) Add conditional to residual/mechanical price models + tests.
4. (~100 LOC) Joint integration + diagnostics for correlation.
5. Tuning / ablation via replay simulator.

**Prerequisites**: Unit tests for models; offline frame replay for validation. Do **first** (foundational for inventory risk).

**Impact**: Highest — directly improves realism of simulated P&L, reduces over-quoting into toxic flow.

### 2. Inventory Risk Term (High Impact, Medium Effort)
**Proposal**: Split as specified.
- (1) Wealth-variance: Implicit via existing CVaR/entropic CE once adverse coupling is in (side-aware marking + flow/price correlation captures adverse selection variance).
- (2) **Explicit operational-capacity term** (backstop vs. model misspecification/short horizon): Zone-of-indifference shape. E.g., `penalty = w_inv * max(0, dist_to_band(Q_net, target_band))^p` where band is [target_low, target_high] (configurable, e.g., symmetric or skewed), p=2 near edges, flat=0 inside. Asymmetric growth (steeper near 0 or max_capacity).

Add to terminal objective (or per-horizon weighted): `J_new = J_return + w_inv_operational * operational_term + ...`

**File/Function**: `mm/mpc/reward_tensor.py` (or `joint_mpc.py` for joint case). Call from rollout objective.

**Pseudocode**:
```python
def operational_inventory_penalty(Q_net, target_band, w=1.0):
    dist = torch.relu(torch.abs(Q_net - target_band.mid) - target_band.half_width)
    # asymmetric: steeper near extremes
    growth = torch.where(Q_net < 0, dist**1.5, dist**2.0)  # example
    return w * growth.mean(dim=0)  # or CVaR-like
```

**Interim stopgap** (if coupling delayed): Simple quadratic around target inventory in state construction / objective (low effort).

**Effort**: Medium (~100-200 LOC + config). Prerequisite: Adverse coupling (or stopgap).

**Interaction with book-quality**: See below.

### 3. Price-Shading vs. Size-Capping (High Impact, Low-Medium Effort)
**Proposal**: Hard-cap quote size at available inventory/cash **by construction** in decoder (`build_joint_target_book_from_sigmoid` + capacity in optimizer). Inventory-risk term (esp. operational zone) primarily drives **price offset / skew** (via adjusted quote_anchor or sigmoid centers) rather than size reduction.

**Check interaction with `quote_spread_max`**: Current spread limits (from doc: sigmoid centers up to `spread_max`, structured first leg) leave **sufficient range** for skew if `quote_spread_max` >= 2-5x typical spread (common in configs). Skew via anchor shift or per-side x0 bias works within cap. If not, widen via config or soft objective.

**File/Function**: `MpcStrategyEngine.decide()` (post-optimizer) + `LParamConfig` / capacity computation in tensor optimizer + `build_joint_target_book_from_sigmoid` (likely in mpc files).

**Change**: In discretization / capacity:
```python
# Hard cap by construction
bid_max = min(..., available_quote / mid)
ask_max = min(..., available_base)
# Then in sigmoid -> marginal: clip sizes, prefer shifting x0 for inventory term
```

**Effort**: Low-Medium (~80-150 LOC). Depends lightly on inventory term.

**Impact**: Prevents invalid oversized quotes; lets optimizer shade prices for risk (better gradients).

### 4. Book-Quality Terms (Medium-High Impact, Low Effort)
**Proposal**: Add as **soft weighted objective terms**:
- Depth: positive reward for total quoted size (normalized to `quote_size_max`).
- Spread tightness: negative penalty on weighted-average spread (or distance from mid).
- Two-sided presence/balance: penalty on |bid_size - ask_size| or min(bid,ask) deficiency.
- Uniformity / constant-slippage: penalty on deviation from ideal cumulative depth curve (e.g., L2 vs. linear or power-law in spread space).

**Scaling solution**: Normalize each to [0,1] or relative to initial wealth / typical spread * size (quote-currency units). E.g., spread term * (typical_notional), depth * (spread_unit_value). Use sample-wise z-score for robustness.

**File/Function**: `mm/mpc/reward_tensor.py` (extend terminal `CE_beta`) + joint equivalent. Config in `ControllerRuntimeConfig`.

**Pseudocode**:
```python
book_quality = (
    w_depth * mean(quoted_depth / max_depth) +
    w_tight * mean(-avg_spread / ref_spread) +
    w_balance * mean(-abs(bid_qty - ask_qty) / target) +
    w_uniform * -l2_dev_from_ideal_curve(cum_depths)
)
J = w_return * CE_return + w_book * book_quality + w_inv_oper * operational_term
```

**Interaction with inventory zone**: Near target band (flat operational penalty=0), book-quality dominates → balanced/tight/deep book. Off-band, inventory term dominates → skew (one side deeper/tighter, other reduced via shading). Smooth handoff via overlapping zones + continuous penalties (no discontinuity if weights tuned; test via simulator). Near-zero inventory: operational term strong, quality secondary.

**Effort**: Low (~100 LOC + config/tuning). Can implement independently but best after inventory term.

**Prerequisites**: Replay simulator for offline tuning/ablation; Prometheus/Grafana for live monitoring of new metrics (add to observability subscriber).

### Ordering by Impact per Engineering Hour + Dependencies
1. **Adverse selection coupling** (foundational; highest long-term impact).
2. **Inventory operational term** (depends on 1; enables clean split).
3. **Hard size-capping + shading** (depends lightly on 2; quick wins on safety).
4. **Book-quality soft terms** (orthogonal but shines with above; easiest to tune last).
5. **Refinements** (skew floors → soft, spread caps tuning, config scalars, tests/dashboards).

**Overall rollout plan**: Start with offline replay validation on historical Parquet data. Add 1-2 scalars to strategy preset. Monitor via new optimizer facts / diagnostics. Expect improved fill quality, lower adverse selection drag, more consistent books, and tunable risk/quality frontier. Total high-confidence path: 500-1000 LOC + tuning.