# 11 — Edge cases and failure handling

This section is dedicated to the *non-happy-path* behavior of the system. The
runtime is designed to be *boring under failure*: it should never panic, and
the worst case is "no orders sent this interval".

## 11.1 Negative objective: the floor / zero policy

If the optimizer's terminal objective is negative (the MPC cannot find a
profitable plan), three policies are configurable:

- `"optimizer_action"` (default) — return the optimizer's plan anyway. The
  validity/viability penalties already discourage truly unprofitable plans,
  so a slightly-negative objective is often still a better response than
  doing nothing.
- `"floor_action"` — replace the action with the *minimum* depth that
  satisfies the validity/viability floors. The bot stays present but does
  not try to be clever.
- `"zero_action"` — return an action with zero depth on both sides. The
  target book is empty and the venue is cleared of resting orders.

`apply_negative_objective_policy` in `mm/strategies/negative_objective_policy.py`
implements the selection. The choice is per-actor and live-configurable.

The `risk_adjusted_return` used by the policy is the *best* value seen
across Adam epochs (not the final value), so the policy never punishes the
optimizer for Adam overshooting in the last epoch.

## 11.2 Finite-checks: `mm/mpc_utils/finite.py`

The optimizer can produce `NaN` or `Inf` values in the raw parameter vector
when:

- the gradient explodes (rare, but possible after a model miss);
- the warm-start vector is corrupt (e.g. a config change between runs that
  changed `n_components`);
- the order-flow model emits a very large sample (a heavy-tailed
  distribution combined with a small batch).

`mm/mpc_utils/finite.py` exposes:

- `is_finite_flat(arr)` — `True` iff every element of `arr` is finite.
- `replace_non_finite(arr, fill=0.0)` — returns a copy with `NaN`/`Inf`
  replaced by `fill`.
- `finite_flat_or_none(arr, expected_len)` — returns the array if it is
  finite and the right length, else `None`. The runner uses this when
  building the warm-start vector: if the previous decision is corrupt, the
  runner falls back to cold start (no warm-start).

The optimizer also clamps the raw parameters to `[-param_clip, param_clip]`
inside the L-param decoder so the Adam state cannot drift into a region where
the sigmoid saturates and the gradient vanishes.

## 11.3 Optimizer cancellation

Already covered in [07-mpc-optimizer.md](07-mpc-optimizer.md). The runner
sets the cancel event when `optimizer_timeout_ms` is exceeded, and the
optimizer exits at the next epoch boundary with `cancel_reason="cancelled"`.
The runner reports this in diagnostics and skips the decision cycle. The
**previous** target book stays on the venue.

## 11.4 Warm-start persistence

The warm-start vector lives in three places:

1. **In the `StrategyRunner`** — as the `_last_raw_flat` attribute, updated at
   the end of every decision.
2. **In the warm-start spy** (`mm/observability/warm_start_spy.py`) — a small
   publisher that writes the raw vector and a hash of it to the bus on every
   decision. The hash is the `decision_hash` exposed in
   `ControllerDiagnostics`.
3. **In the persistence layer** — on graceful shutdown the runner writes the
   raw vector to a per-actor key. On startup the runner reads it and uses it
   for the first decision. This is the only state that survives a process
   restart.

If the persisted vector is missing (first-ever startup) or corrupt (fails
`finite_flat_or_none`), the runner cold-starts.

## 11.5 Stale or missing private state

If the `PrivateStateSnapshot` for a venue is older than `max_ledger_age_ms`,
the `JointFrameAssembler` drops the frame and the runner does not act. The
control plane logs `reason="stale_ledger"`. The strategy keeps its last
target book on the venue; the venue's orders continue to age out or fill
naturally.

## 11.6 Stale global event

If the global event is older than `max_global_lag_ms`, the
`StrategyInputFrameAssembler` sets `global_status="stale"` and
`global_lag_ms`. The strategy engine still runs (it can degrade gracefully by
using the local mid as the reference) but the *consolidated wealth* term in
the joint objective is down-weighted (the optimizer still respects the
global mid but penalizes it when it is stale).

## 11.7 Reconciliation failures

The `Reconciler` is the loop that pulls open orders and fills from the venue
and writes them to the `Ledger`. If the venue rejects an order, the
reconciler marks it as `rejected` and removes it from the ledger. If the
venue fills an order, the reconciler updates the ledger and emits a
`FillDelta` payload on the bus (consumed by the model update loop and the
PnL aggregator).

The reconciler is *idempotent*: re-running it on the same set of events
produces the same ledger state. This is important because the venue's
private feed can deliver the same fill twice, and we must not double-count.

## 11.8 Process restart

On graceful shutdown the runtime:

1. stops the schedulers (no more `MarketEvent`s);
2. cancels the strategy runners (no more decisions);
3. waits for the in-flight optimizers to return or to time out (max
   `optimizer_timeout_ms`);
4. flushes the persistence layer (writes the last warm-start vector and
   diagnostic snapshots);
5. closes the bus connections.

On a *crash* (SIGKILL, OOM) the persistence layer holds the most recent
snapshot. On startup the runners read the snapshot and resume from there.
The optimizer cold-starts (the persisted warm-start vector is loaded if
present, else `None`).
