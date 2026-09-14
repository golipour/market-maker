# 07 — The MPC optimizer: Monte Carlo over a finite horizon

The optimizer lives in `mm/mpc/mpc_tensor_batched.py`. It is a *batch Monte Carlo*
optimizer: for every Adam epoch it draws `B` paths of length `H`, simulates each
path under the action, evaluates the terminal objective, and backpropagates
through everything to update the action parameters.

The same machinery is wrapped by the joint-venue variant in
`mm/mpc/joint_optimizer.py` which optimizes a single flat vector that contains
both local and global `SigmoidMixtureAction` parameters.

## 7.1 Notation

- `B` — Monte Carlo batch size.
- `H` — horizon (number of future steps).
- `K` — number of sigmoid components per side.
- `n_raw_local` = `3 * K_local` raw parameters per side for the local venue.
- `n_raw_global` = `3 * K_global` raw parameters per side for the global venue.
- `N_total` = total raw parameter count.
- `E` — number of Adam epochs per decision.
- `α` — Adam learning rate.
- `clip_norm` — global L2 gradient norm clip.

## 7.2 Top-level call

`MonteCarloMPCTensorBatched.optimize(state, n_samples, horizon, adam_epochs,
learning_rate, warm_start_action, cancel_event)` returns an
`OptimizerResult`:

- `action` — the final `SigmoidMixtureAction`;
- `best_objective` — best terminal objective seen across epochs;
- `best_risk_adjusted_return` — separated for the negative-objective policy;
- `objective_history` — per-epoch objective trace;
- `gradient_norm_history` — per-epoch gradient norm trace;
- `cancel_reason` — `None`, `"cancelled"`, or `"timeout"`.

## 7.3 The closure: one Adam epoch

Inside the optimizer loop, every epoch calls a `closure` that:

1. Decodes the current raw parameters into `(L, k, x0)` tensors.
2. Simulates the market over the horizon under the action.
3. Computes the terminal objective and the auxiliary penalties.
4. Backpropagates and lets Adam update the parameters.

## 7.4 Market simulation per step

For each step `t` of the horizon, the optimizer:

1. Draws `V₊, V₋` from the order-flow model and `Δmid` from the price model.
2. Computes the **soft fill** for the bot's quotes: at each tick distance `s`,
   the cumulative depth is `D(s)`; market orders at a given side drain the
   bot's queue at that side.
3. Updates the bot's inventory and cash:

   ```
   X_t   = X_{t−1} + fill_bid_t − fill_ask_t
   cash_t = cash_{t−1} + fill_ask_t · (mid_t − spread_ask/2)
                     − fill_bid_t · (mid_t − spread_bid/2)
   ```

4. Optionally updates the resting book (top-N) using the realized trades.

`soft_fill_batched` is the heart of the simulation: it is a *differentiable*
function of the action parameters and the sampled `V₊, V₋`. The fill at
distance `s` is `min(D(s) − D(0), V_side)` (a smooth min is used inside the
optimizer so the gradient is well-defined).

## 7.5 Terminal objective

The terminal objective combines:

- risk-adjusted return (certainty equivalent),
- inventory penalty,
- slippage penalty,
- validity penalty (book quality),
- viability penalty (expected availability shortfall).

`terminal_objective(state_traj, action_traj)` in `mm/mpc/reward_tensor.py`
computes the full scalar `J`. The sign convention is **higher J is better** —
the optimizer *minimizes* `-J`.

## 7.6 Viability & validity

Validity and viability are described in detail in [10-risk.md](10-risk.md). In
short:

- `validity_penalty = μ · max(0, B_min − book_quality_score)`
- `viability_penalty = λ · E[d_bid² + d_ask²]` (next-step quadratic deficit)

The optimizer adds both to the terminal objective. The `viability` dual
variable `λ` is updated externally in `update_viability_lambda` using a
bounded linear step on an EWMA of the deficit.

## 7.7 Cancellation and timeout

The optimizer checks the `cancel_event` at the end of every Adam epoch. The
runner sets the event when the wall-clock budget (`optimizer_timeout_ms`) is
exceeded. The optimizer then:

- sets `cancel_reason = "cancelled"`,
- returns the action at the *last* completed epoch (not the partial epoch), and
- the runner reports `"optimizer_status": "cancelled"` in the diagnostics.

The action returned by a cancelled optimization is always *finite* (see
[11-edge-cases.md](11-edge-cases.md)) because the warm-start re-validates the
raw vector and the decoder rejects NaN/Inf values during construction.

## 7.8 Warm-start

The optimizer accepts a `warm_start_action: np.ndarray | None` of shape
`(N_total,)` containing the raw flat parameter vector of the *previous*
decision. The runner stores this vector at the end of every decision (see
[11-edge-cases.md](11-edge-cases.md) for the persistence details) and feeds it
back in for the next one.

The blend is controlled by `WarmStartParams`:

- `enabled=false` (or `alpha=0`) → cold start (raw vector is the default
  initializer, computed by `default_initial_raw` from the current mid and
  config bounds).
- `alpha=1.0` → use the previous vector verbatim (legacy binary warm start).
- `alpha_mode="adaptive"` → `α = min(α_max, ‖∇u_warm‖ / (‖∇u_warm‖ + ‖∇u_default‖ + ε))`,
  i.e. start near the previous vector when its gradient is small relative to
  the default's, else start near the default.

`prepare_warm_start_initial_flat` tiles the per-step blended vector across the
horizon so the optimizer sees a stationary plan to start from.

`preserve_optimizer_state=true` keeps Adam's first and second moments across
decisions, which gives a substantial speed-up in practice (the optimizer
converges in 3-5 epochs when warm-started vs. 15-25 from cold).

## 7.9 Best-objective tracking

The optimizer keeps a running `best_objective` and the action that produced it.
At the end of `E` epochs it returns `best_action`, not the action from the last
epoch. This is robust to Adam overshooting in the last few epochs — common
when the gradients are very large early in training.

## 7.10 Joint-venue optimizer

`optimize_joint_with_diagnostics` in `mm/mpc/joint_optimizer.py` is the entry
point for joint mode. It:

1. Builds a joint state with `enrich_state_for_joint`.
2. Slices the warm-start vector into local and global halves.
3. Builds a single `JointSigmoidMixtureAction` whose parameters span both
   venues.
4. Runs the same Adam loop but with the joint reward
   `joint_terminal_objective(state, joint_action)`.
5. Returns the same `OptimizerResult`, with the action split back into local
   and global components for the target book builder.

The joint objective includes a *consolidated wealth* term so the optimizer
trades off between the two venues' PnL streams (see
[08-objective.md](08-objective.md)).
