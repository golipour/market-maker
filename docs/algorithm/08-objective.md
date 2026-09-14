# 08 — Reward and objective terms

The MPC objective is the sum of:

- the **risk-adjusted return** (certainty equivalent) from the simulated PnL,
- an **inventory penalty** that pushes the bot to flat,
- a **slippage penalty** that discourages quoting too far from the inside,
- a **validity penalty** that penalizes structurally bad books,
- a **viability penalty** that penalizes actions that would over-spend
  inventory or margin,
- an optional **objective normalization** that scales the risk-adjusted return
  by a learned or configured factor.

## 8.1 Notation

- `W_t = cash_t + X_t · mid_t` — wealth at step `t`.
- `ΔW_t = W_t − W_{t−1}` — change in wealth.
- `X_t` — net base inventory (signed: positive = long).
- `Q_t` — queue position of a quote (for queue-aware fill model).
- `τ` — book quality target.
- `λ_v, λ_θ` — weights for the validity and viability penalties.
- `β` — risk aversion coefficient (return_risk_aversion_beta).

## 8.2 PnL decomposition: `delta_pnl_detailed_tensor`

The per-step PnL is split into four terms that are useful for diagnostics:

```60:120:mm/mpc/reward_tensor.py
def delta_pnl_detailed_tensor(state_t, state_tm1, action_t, ...):
    dX = state_t.inventory - state_tm1.inventory
    dP = state_t.mid - state_tm1.mid
    dQ = dX                       # inventory change == net fills
    return DeltaPnL(
        cash_change   = (state_t.cash - state_tm1.cash),
        inventory_pnl = state_tm1.inventory * dP,
        new_position  = dQ * state_tm1.mid,
        cross_term    = dQ * dP,
    )
```

So `ΔW = Δcash + X_{t−1}·ΔP + ΔQ·P_{t−1} + ΔQ·ΔP`.

## 8.3 Inventory penalty: `inventory_pnl_term_tensor`

```
inv_pen = λ_inv · X_T · mid_T · sign(X_T)
```

A simple *soft* penalty on terminal inventory, evaluated at the end of the
horizon. The weight `λ_inv` is set in `VenueOptimizerParams.inventory_penalty`.

## 8.4 Slippage penalty: `slippage_pnl_term_tensor`

```
slip_pen = λ_slip · E_path[ Σ_t  ∂D_ask / ∂s · (s_t − mid) ]
```

A penalty on quoting too far from the inside, measured as the area under the
depth curve times the spread. The marginal depth `∂D/∂s` is computed inside
the optimizer by `depth_*_derivative_tensor` in `mm/mpc/state_tensor.py`.

## 8.5 Risk-adjusted return: `objective_entropic_return`

The certainty-equivalent (negative-entropy) return is

```
CE = (1/β) · log E_path[ exp(β · ΔW_T) ]
```

with `β` = `return_risk_aversion_beta` (per-venue). The convention in the code
is `mean − β/2 · var` (Cox-Lin-Hanson expansion to second order) which matches
the exponential CE to leading order and is more numerically stable.

The sign convention inside the optimizer is `J += ce` (we *maximize* the
certainty equivalent) and `J -=` the other penalties. Adam minimizes `-J`.

## 8.6 Validity: `book_quality_score` and `combine_validity_term`

`book_quality_score` evaluates the *structure* of the proposed book:

- **tightness** — average distance from the mid, weighted by quantity;
- **presence** — total quantity inside a reference band around the mid;
- **balance** — how symmetric the bid and ask sides are;
- **uniformity** — how smooth the depth curve is (penalizes spikes).

```60:100:mm/mpc/objective_terms.py
def book_quality_score(action, ob_rel, params):
    tightness = _tightness_score(action, ob_rel, params)
    presence  = _presence_score(action, ob_rel, params)
    balance   = _balance_score(action)
    uniformity = _uniformity_score(action)
    return (
        params.v_tightness * tightness
        + params.v_presence  * presence
        + params.v_balance   * balance
        + params.v_uniformity * uniformity
    )
```

`combine_validity_term` adds `λ_v · max(0, τ − score)²` to the objective.

## 8.7 Viability: `viability_path_shortfall` and `combine_viability_term`

`wallet_quadratic_deficit` measures a two-sided squared-hinge reserve
deficit on **total** balances. It is zero above `q_min` and steepens as a
side falls further below the floor:

```
d_bid = relu(q_min - cash) / q_min
d_ask = relu(q_min - inventory * mid) / q_min
S     = d_bid² + d_ask²
```

The MPC applies `−λ · E[S_1]` (next-step only). `update_viability_lambda`
then steps a bounded linear multiplier from an EWMA of `S`, so a noisy
Monte Carlo batch cannot explode `λ`.
