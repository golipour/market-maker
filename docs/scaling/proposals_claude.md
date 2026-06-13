# Market-Maker Improvement Proposals

## Framing

The MPC objective currently optimises a single term:

```
J = CE_beta(R)          # single-venue
J = w_L·CE_beta_L(R^L) + active_G·w_G·CE_beta_G(R^G)   # joint
```

Inventory risk enters only *indirectly* — through side-aware VWAP liquidation marking,
dynamic capacity clipping, and downstream risk gates.  
Book quality (spread, depth) is entirely absent from the objective.

The proposals below add a **book-quality objective term**, tighten the existing
**hard-constraint layer** (which, as noted, requires no weight selection), and add
one or two **soft penalty terms** that fold into the existing loss with a single
scalar weight `lambda`.

The full tunable trade-off is therefore:

```
J_total = CE_beta(R)  +  lambda · J_book
loss    = -J_total
```

`lambda = 0` → pure risk-adjusted return (current behaviour).  
`lambda > 0` → progressively favours tighter spreads and balanced depth.

---

## Priority ordering

| # | What it fixes | Disruption | Effort |
|---|---------------|------------|--------|
| P1 | Inventory never reaches terminal constraint; asymmetric book degradation | None — adds a penalty term | S |
| P2 | Spread widens arbitrarily when fill probability is low | Minimal — adds a book-quality term | S |
| P3 | Bid/ask depth imbalance after one-sided fills | Minimal — extends existing skew logic | S |
| P4 | Spread-structured parameterisation not always enforced | Config only | XS |
| P5 | Hard min-cash / min-inventory gates fire reactively, not predictively | Extends reward_tensor | M |

---

## Proposal P1 — Quadratic Inventory Penalty in the Rollout Objective

### Failure mode addressed

`CE_beta(R)` marks long inventory at the sell VWAP and short inventory at the buy
VWAP. When the bid-ask spread is wide, the liquidation discount is small relative to
the return from earned spread — so the optimizer accumulates inventory without penalty.
Downstream risk gates then *reactively* kill one side of the book, producing
zero-depth on that side for the next frame.

### Change

**File:** `mm/mpc/reward_tensor.py`  
**Function:** the terminal return computation block (currently produces `R_i` then
calls `entropic_ce`).

Add a per-step inventory penalty to the per-path return before the CE aggregation:

```
# existing
R_i = sum_h (Delta W_{i,h}) / W_0

# proposed: subtract a convex inventory cost at every step
inv_penalty_h = gamma_inv * (Q_{i,h} / Q_scale)^2
R_i_penalised = sum_h [ (Delta W_{i,h} - inv_penalty_h) ] / W_0

J = CE_beta(R_i_penalised)
```

where:

- `gamma_inv` is a new config scalar, e.g. `controller.optimizer.inventory_penalty`
  (default `0.0`).
- `Q_scale` is `quote_size_max_per_side` (already available in the optimizer closure)
  so the penalty is dimensionless.
- The `Q_{i,h}` tensor is already computed inside `soft_fill_batched()` rollout loops.

**In the joint case** (`mm/mpc/joint_mpc.py`), apply the same penalty to `Q^L + Q^G`:

```
inv_penalty_h = gamma_inv * ((Q^L_{i,h} + Q^G_{i,h}) / Q_scale)^2
```

### Why a hard constraint is not enough here

The existing `min_inventory` / `max_inventory` risk gates are hard **post-hoc**
constraints — they drop an entire side after the fact.  
A smooth rollout penalty nudges the optimizer *before* those gates would fire,
keeping both sides of the book alive.  
The hard constraint remains in place as the safety backstop and requires no weight.

### Config change

```json
"optimizer": {
  "inventory_penalty": 0.002
}
```

### Effort: S (< 1 day)

Prerequisite: none. `Q_{i,h}` tensor already exists in the rollout.

---

## Proposal P2 — Book-Quality Reward Term (Spread + Depth Incentive)

### Failure mode addressed

The optimizer has no incentive to quote tight spreads when fill probability is low
(e.g. thin market, wide reference spread).  
`CE_beta(R)` is indifferent between quoting 1 bps and 100 bps from mid if expected
fills are the same.  
The result is a systematically wide quoted book.

### Change

**File:** `mm/mpc/reward_tensor.py`  
**Function:** same block as P1 — extend `R_i_penalised` with a book-quality bonus.

Define a differentiable book-quality scalar at step `h = 0` (first horizon step,
since that is the step that becomes the actual target book):

```
# effective_spread: distance from quote_anchor to the first non-zero bid and ask level
# computed from the action's cumulative depth curves D_bid(x) and D_ask(x)

spread_bonus_h0 = -kappa_spread * (x0_bid_eff + x0_ask_eff) / spread_max
# x0_bid_eff, x0_ask_eff are the first-component centres from LParamConfig,
# already differentiable tensors in the optimizer action vector

depth_bonus_h0  = kappa_depth * min(D_bid(spread_ref), D_ask(spread_ref)) / Q_scale
# D_s(spread_ref) = cumulative depth at a reference spread, e.g. 0.5 * spread_max
# "min" rewards balanced depth; unbalanced depth earns only partial credit

J_book = spread_bonus_h0 + depth_bonus_h0
```

The full objective becomes:

```
J_total = CE_beta(R_i_penalised)  +  lambda · J_book
loss    = -J_total
```

`lambda` is the single scalar weight exposed to config:

```json
"optimizer": {
  "book_quality_weight": 0.01
}
```

### Gradient path

`x0_bid_eff` and `x0_ask_eff` are outputs of `LParamConfig` which is already in the
autograd graph. `D_s(spread_ref)` is a sigmoid evaluation at a fixed point — also
already differentiable. No new tensors are needed.

### Why `lambda` is the right tuning knob here

`CE_beta(R)` is in units of normalised return (dimensionless bps-scale).
`J_book` is also normalised to `[-1, 1]` by construction above.
`lambda` therefore has an interpretable meaning: "how many bps of expected return am I
willing to give up per unit of book quality improvement."  
A value around `0.005 – 0.02` is a reasonable starting range.

### Effort: S (< 1 day)

Prerequisite: none. `x0_bid_eff`/`x0_ask_eff` tensors are already present in the
optimizer's action parameterisation path in `mm/mpc/mpc_tensor_batched.py`.

---

## Proposal P3 — Inventory-Conditioned Depth Skew Inside the Rollout

### Failure mode addressed

After one-sided fills the account becomes skewed (e.g. long inventory, low cash).
The existing `bid_floor` / `ask_floor` skew only affects the **capacity bounds** fed
into the optimizer — it does not tilt the *initial action warm-start*.  
The optimizer therefore spends Adam epochs climbing back from a symmetric warm-start,
often converging to a suboptimal local minimum that still quotes aggressively on the
long side (compounding inventory) rather than reducing the bid and widening the ask.

### Change

**File:** `mm/mpc/mpc_tensor_batched.py`  
**Function:** `_warm_start_action()` (or equivalent initialisation block before
`adam_epochs`).

Add an inventory-conditioned offset to the initial `x0` parameters:

```
imbalance = clamp((Q * mid - X) / (|Q * mid| + |X| + eps), -1, 1)
# positive imbalance = long/overbought

# tilt first-leg centres away from mid on the heavy side:
x0_bid_init  += skew_x0_strength * max(imbalance, 0) * spread_max
x0_ask_init  -= skew_x0_strength * max(-imbalance, 0) * spread_max
```

This does not constrain the optimizer — it only provides a better warm start so Adam
converges to a tighter, directionally-correct solution in the available epoch budget.

`skew_x0_strength` is a new config scalar:

```json
"optimizer": {
  "skew_x0_strength": 0.3
}
```

### Effort: S (< 1 day)

Prerequisite: `imbalance` is already computed for `bid_floor`/`ask_floor` capacity
skew — expose it upstream or recompute in the same function.

---

## Proposal P4 — Enforce `quote_spread_max` / `quote_spread_min_depth` via Config

### Failure mode addressed

The spread-structured parameterisation (`quote_spread_max`, `quote_spread_min_depth`,
`quote_spread_k_epsilon`) exists in `LParamConfig` but is only activated when **all
three** are set in config. In practice, operators often set `spread_max` without
setting `quote_spread_min_depth`, silently falling back to the unconstrained
parameterisation and allowing the optimizer to quote with zero depth at tight spreads.

### Change

**File:** `mm/runtime/config.py`  
**Function:** `runtime_config_from_mapping()` validation block.

Add a validation warning (or hard error) when `spread_max` is set but
`quote_spread_min_depth` is not:

```python
if controller_cfg.spread_max is not None and \
   controller_cfg.quote_spread_min_depth is None:
    logger.warning(
        "spread_max is set but quote_spread_min_depth is not — "
        "spread-structured parameterisation is inactive; "
        "set quote_spread_min_depth to enforce minimum book depth at the inside spread."
    )
```

Also document the three-field activation requirement in the example config.

**File:** `mm/configs/market_maker.runtime.example.json`  
Add `quote_spread_min_depth` and `quote_spread_k_epsilon` alongside `spread_max`
with explanatory comments.

### Effort: XS (< 2 hours)

Prerequisite: none.

---

## Proposal P5 — Soft Cash/Inventory Floor Penalty (Predictive, not Reactive)

### Failure mode addressed

`min_cash` and `min_inventory` in the risk engine are hard gates that fire
*after* the optimizer has already proposed a target book that violates them.
That frame's orders are dropped, the book goes one-sided for the interval, and
market impact from queue re-entry is paid at the next frame.

### Change

**File:** `mm/mpc/reward_tensor.py`  
**Function:** same return computation block.

Add smooth barrier penalties evaluated over the rollout path:

```
# cash floor barrier (soft)
cash_shortfall_h = relu(cash_floor_frac * X_0 - X_{i,h})
cash_penalty_h   = eta_cash * (cash_shortfall_h / X_0)^2

# inventory floor/ceiling barrier (soft)
inv_hi_shortfall_h  = relu(Q_{i,h} - inv_max_frac * Q_scale)
inv_lo_shortfall_h  = relu(inv_min_frac * Q_scale - Q_{i,h})  # negative inventory floor
inv_penalty_h       = eta_inv_bounds * (
                          (inv_hi_shortfall_h / Q_scale)^2
                        + (inv_lo_shortfall_h / Q_scale)^2
                      )
```

Both are subtracted from `R_i` before the CE aggregation, alongside the P1 penalty:

```
R_i_penalised = sum_h [
    Delta W_{i,h}
  - inv_penalty_h        # P1
  - cash_penalty_h       # P5
  - inv_penalty_h        # P5
] / W_0
```

New config fields under `controller.optimizer`:

```json
"cash_floor_frac":   0.05,   // penalty activates below 5 % of initial cash
"eta_cash":          0.001,
"inv_max_frac":      2.0,    // relative to quote_size_max_per_side
"inv_min_frac":     -2.0,
"eta_inv_bounds":   0.001
```

The hard `min_cash` / `max_inventory` risk gates remain unchanged — these soft
penalties are purely predictive and do not replace the safety backstop.

### Effort: M (1–2 days)

Prerequisite: `X_{i,h}` (cash) tensor across rollout steps — verify it is
materialised at each horizon step in the current rollout loop. If only the terminal
cash is retained, extend the loop to collect per-step cash tensors (small tensor
allocation change in `mm/mpc/mpc_tensor_batched.py`).

---

## Consolidated Config Surface

All new scalars live under `controller.optimizer` in the existing runtime JSON:

```json
"optimizer": {
  "inventory_penalty":   0.002,   // P1 — quadratic inventory cost per rollout step
  "book_quality_weight": 0.01,    // P2 — lambda: the single return/quality trade-off knob
  "skew_x0_strength":    0.3,     // P3 — warm-start skew strength (0 = no skew)
  "cash_floor_frac":     0.05,    // P5 — soft cash floor as fraction of initial cash
  "eta_cash":            0.001,   // P5 — cash floor penalty coefficient
  "inv_max_frac":        2.0,     // P5 — soft inventory ceiling (× Q_scale)
  "inv_min_frac":       -2.0,     // P5 — soft inventory floor (× Q_scale)
  "eta_inv_bounds":      0.001    // P5 — inventory bounds penalty coefficient
}
```

All default to `0.0` / `false`, making every change **backwards-compatible with
existing configs**.

---

## Relationship Between Hard Constraints and Soft Terms

```
Layer               Location                        Tuning
────────────────────────────────────────────────────────────────
Hard risk gates     mm/risk/net_delta.py            None needed (binary on/off)
  min_cash          ← already exists
  min_inventory     ← already exists
  max_inventory     ← already exists
  max_open_notional ← already exists

Soft rollout        mm/mpc/reward_tensor.py         eta_cash, eta_inv_bounds,
  barriers          mm/mpc/joint_mpc.py             gamma_inv (small, independent)

Book quality        mm/mpc/reward_tensor.py         lambda = book_quality_weight
  bonus             (single scalar, the main knob)
```

The key insight is that because the hard constraints require no weight selection,
they remain the reliable safety backstop regardless of how `lambda` or `gamma_inv`
are tuned. The soft terms are purely about *shaping the optimizer's gradient signal*
before those hard gates would otherwise fire.

---

## Implementation Order

1. **P4** (config validation) — zero-risk, fixes silent misconfiguration now.
2. **P2** (book-quality weight `lambda`) — the primary control knob; validates the
   gradient path and establishes the two-objective trade-off structure.
3. **P1** (inventory penalty) — compounds with P2 to keep the book symmetric.
4. **P3** (warm-start skew) — improves Adam convergence speed after inventory
   events; especially valuable with the small `adam_epochs = 10` default.
5. **P5** (soft cash/inv barriers) — reduces hard-gate activation frequency;
   implement after verifying per-step cash tensors are available.