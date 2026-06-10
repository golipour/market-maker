# Predictability Hardening Solution

This document combines the code-derived architecture review, the MPC and
optimizer review, and `docs/vulnerabilities-and-mitigations.md` into a concrete
hardening plan for the low-frequency market maker.

The core conclusion is simple: the optimizer uses Monte Carlo simulation
internally, but the externally visible quoting behavior can still be
fingerprinted. A trader can observe timing, price ladder spacing, order counts,
order sizes, and cancel/place waves, then adapt around or against the bot.

The solution is not to make the MPC random everywhere. The MPC should continue
to compute the economic target. Randomness should be applied mainly in a
bounded, risk-aware target-book presentation layer and in live execution timing.
All randomized outputs must be validated by the same risk and venue rules as
normal outputs.

## Current Detectable Structure

The following properties are visible or inferable from the public order book:

1. Fixed decision cadence.
   `IntervalScheduler` aligns callbacks to exact wall-clock boundaries. This
   creates quote updates around predictable interval edges.

2. Fixed spread grid.
   `OrderbookGridConfig.spread_ladder()` creates a ladder at:

   ```text
   spread_step, 2 * spread_step, ..., spread_max
   ```

   The target builder then places bids at `anchor_mid - spread` and asks at
   `anchor_mid + spread`.

3. Deterministic state-to-action mapping.
   Given similar order book, model state, inventory, cash, and optimizer
   settings, the same MPC state will produce very similar sigmoid depth curves.
   If `fixed_mc_samples` is enabled, the optimizer objective inside a decision
   is deterministic for the sampled batch.

4. Deterministic discretization.
   The continuous sigmoid action is converted to marginal quantities at fixed
   ladder spreads. Capping keeps the largest quantities per side.

5. Predictable replacement behavior.
   When a venue uses `replace`, the bot can create repeated cancel/repost waves.
   Even `reconcile` can churn if price and quantity tolerances are zero.

6. Formulaic inventory response.
   Inventory-dependent quote capacity and skew are structurally sensible, but
   a large, persistent observer can infer how quoting changes as inventory
   changes.

## Exploitation Vectors

### Temporal Sniping

A trader waits for the bot to refresh, then trades against stale quotes before
the next interval when an external reference price has moved.

### Pennying and Queue Theft

A trader learns the quote ladder and places orders one tick inside the bot.
The bot either loses benign flow or repeatedly cancels and loses queue priority.

### Inventory Baiting

A trader repeatedly fills one side to push inventory toward a known stress
zone, then waits for predictable skewed quotes.

### Reference and VWAP Manipulation

On thin venues, small or wash-like trades can distort local trade VWAP. If the
bot trusts this too much, quotes can be pulled away from the true market.

### Footprint Classification

Repeated order counts, size ratios, ladder shape, and cancel cadence let another
participant classify the actor and build a response strategy.

## Design Principles

1. Keep the MPC deterministic enough to debug.
   The economic target should be explainable and replayable.

2. Randomize only within economic and risk bounds.
   A randomized quote must still pass min edge, inventory, cash, max notional,
   price distance, tick size, and venue limits.

3. Prefer relative configuration.
   Parameters should scale across symbols and markets using bps, percentages,
   interval fractions, tick counts, spread fractions, inventory utilization, and
   notional ratios.

4. Do not randomize feature windows by default.
   Market-event construction, model updates, and diagnostic timestamps should
   remain aligned and reproducible. Jitter the live execution action after the
   decision, not the raw market data windows.

5. Record random seeds and perturbation summaries.
   Private diagnostics should capture what was randomized. Public or shared
   channels should not leak exact randomization details.

## Recommended Architecture

Add three layers around the existing decision pipeline:

```text
clean frame
  -> MPC economic target
  -> deterministic target book
  -> target-book hardening layer
  -> risk validation
  -> venue preparation/rules
  -> execution timing hardening
  -> order manager sync
```

The hardening layer should be disabled by default, then enabled per actor or
venue after backtesting and dry-run observation.

### Layer 1: Target-Book Hardening

This layer receives a `JointTargetBook` and returns a modified `JointTargetBook`
plus diagnostics. It should operate on target orders before live execution.

Responsibilities:

- price jitter within a relative cap,
- size jitter while preserving side-level exposure approximately,
- probabilistic level thinning,
- optional child order splitting,
- near-equivalent level selection,
- minimum edge enforcement,
- max distance enforcement,
- deterministic replay via a stored seed.

The layer must never make a forbidden order valid by hiding it. It only creates
candidate target books. Risk and venue rules still decide what can be placed.

### Layer 2: Refresh Hysteresis

Before syncing, compare the hardened target book to live orders and the previous
approved target. If the change is economically small, hold current orders.

This is often more valuable than randomization because it reduces the public
cancel/repost signature and preserves queue position.

### Layer 3: Execution Timing Jitter

Add a bounded random delay before live sync. The delay should be configured as a
fraction of the decision interval and capped by data freshness and order stale
risk.

Do not delay emergency cancels, kill-switch actions, stale-reference pulls, or
risk-reducing actions.

## Relative Configuration Schema

Recommended config block:

```json
{
  "controller": {
    "hardening": {
      "enabled": true,
      "seed_mode": "run_actor_frame",
      "diagnostics_level": "summary",

      "refresh_hysteresis": {
        "enabled": true,
        "price_change_bps": 1.0,
        "qty_change_pct": 5.0,
        "notional_change_pct": 5.0,
        "side_notional_change_pct": 7.5,
        "max_hold_intervals": 3,
        "always_sync_on_side_flip": true,
        "always_sync_on_risk_reduction": true
      },

      "execution_timing": {
        "enabled": true,
        "jitter_interval_pct": 12.5,
        "max_jitter_ms": 750,
        "skip_jitter_when_frame_age_pct_of_interval_gt": 50.0,
        "skip_jitter_on_risk_reduction": true,
        "skip_jitter_on_cancel_only": true
      },

      "quote_price": {
        "enabled": true,
        "jitter_bps_of_mid": 0.5,
        "jitter_pct_of_half_spread": 8.0,
        "max_jitter_ticks": 2,
        "min_distance_from_touch_bps": 0.5,
        "max_distance_from_mpc_price_bps": 3.0,
        "never_cross_touch": true,
        "round_after_jitter": true
      },

      "quote_size": {
        "enabled": true,
        "per_order_jitter_pct": 10.0,
        "side_total_jitter_pct": 3.0,
        "max_order_inventory_pct": 5.0,
        "max_order_cash_pct": 5.0,
        "preserve_side_notional_within_pct": 5.0,
        "min_notional_multiplier": 1.25
      },

      "level_selection": {
        "enabled": true,
        "drop_level_probability_pct": 10.0,
        "max_dropped_side_notional_pct": 15.0,
        "eligible_outer_level_min_rank": 2,
        "near_equivalent_qty_pct": 15.0,
        "always_keep_best_level": true
      },

      "child_orders": {
        "enabled": false,
        "split_probability_pct": 20.0,
        "max_children_per_order": 3,
        "min_child_notional_multiplier": 1.5,
        "child_size_jitter_pct": 20.0
      },

      "inventory_response": {
        "enabled": true,
        "skew_strength_jitter_pct": 7.5,
        "threshold_jitter_pct": 5.0,
        "inventory_utilization_bucket_pct": 5.0,
        "only_reduce_risk_when_utilization_pct_gt": 80.0
      },

      "minimum_edge": {
        "enabled": true,
        "min_edge_bps": 3.0,
        "min_edge_fee_multiplier": 2.0,
        "min_edge_volatility_multiplier": 0.25,
        "reject_or_widen": "widen"
      }
    }
  }
}
```

All values above are intentionally relative:

- `*_bps` scales with price.
- `*_pct` scales with order size, interval, spread, inventory, cash, or notional.
- `*_multiplier` scales with venue minimums, fees, or volatility.
- `max_jitter_ticks` is the only absolute-like setting, but it scales through
  venue tick size.

## Parameter Semantics

### Seed Mode

`seed_mode` should support:

- `run_actor_frame`: deterministic replay inside one run, different across
  runs.
- `actor_day_frame`: stable intraday replay for incident analysis.
- `system_random`: least replayable, only for mature production.

Recommended default: `run_actor_frame`.

The seed material should include:

```text
run_id, actor_id, exchange, symbol, frame_step, t_end_ms, decision_hash
```

### Price Jitter

Use the smallest of the configured caps:

```text
price_jitter_abs = min(
  mid * jitter_bps_of_mid / 10000,
  half_spread * jitter_pct_of_half_spread / 100,
  tick_size * max_jitter_ticks,
  mpc_price * max_distance_from_mpc_price_bps / 10000
)
```

Then sample a side-safe perturbation:

```text
bid_price = bid_price - random_abs_jitter_or_small_inward_move
ask_price = ask_price + random_abs_jitter_or_small_inward_move
```

By default, favor outward or neutral movement. Inward movement should require a
minimum edge check. Never cross the touch and never violate configured max quote
distance.

### Size Jitter

Apply per-order size jitter, then renormalize per side:

```text
raw_qty_i = mpc_qty_i * Uniform(1 - p, 1 + p)
target_side_total = mpc_side_total * Uniform(1 - side_p, 1 + side_p)
qty_i = raw_qty_i * target_side_total / sum(raw_qty)
```

After that:

- cap by inventory and cash ratios,
- drop orders below venue minimum multiplied by `min_notional_multiplier`,
- rerun risk.

### Level Selection

Random level dropping should not remove too much exposure:

```text
sum(dropped_notional_side) <= side_notional * max_dropped_side_notional_pct / 100
```

`always_keep_best_level` should be true unless the strategy intentionally wants
to avoid top-of-book competition.

For capped ladders, instead of always keeping the largest quantities, use a
weighted sample among near-equivalent levels:

```text
eligible if qty >= max_side_qty * (1 - near_equivalent_qty_pct / 100)
weight = qty
```

### Refresh Hysteresis

Hold current live orders if all are true:

```text
abs(price_change) <= price_change_bps
abs(qty_change) <= qty_change_pct
abs(notional_change) <= notional_change_pct
side_notional_change <= side_notional_change_pct
held_intervals < max_hold_intervals
```

Bypass hysteresis for:

- side flips,
- risk reduction,
- stale reference protection,
- kill switch,
- cancel-only safety actions,
- orders that violate max age or max quote distance.

### Timing Jitter

Compute:

```text
jitter_ms = min(interval_ms * jitter_interval_pct / 100, max_jitter_ms)
delay = Uniform(0, jitter_ms)
```

Skip delay when:

- the frame is already old,
- the action reduces risk,
- the action is cancel-only,
- market data or private state is close to stale,
- a global reference has moved beyond a protection threshold.

## Structural Defenses

Randomization hides the fingerprint. It does not solve bad economics. The
following structural defenses should be implemented or verified separately.

### Reference Price Protection

Use a robust reference cascade:

1. trusted global/reference mid or microprice,
2. local clean orderbook microprice,
3. local VWAP only after outlier and dust filtering,
4. hold or widen when references diverge too much.

Suggested relative config:

```json
{
  "reference_protection": {
    "enabled": true,
    "local_global_divergence_bps": 10.0,
    "trade_outlier_bps_from_mid": 20.0,
    "min_trade_notional_multiplier": 2.0,
    "microprice_weight_when_local_thin_pct": 80.0,
    "hold_when_reference_stale_interval_pct": 150.0
  }
}
```

### Adverse Selection Protection

Add a fast path that can pull or widen without waiting for the full MPC loop:

```json
{
  "adverse_selection": {
    "enabled": true,
    "short_window_interval_pct": 25.0,
    "ofi_zscore_pull": 3.0,
    "global_move_bps_pull": 5.0,
    "spread_widen_multiplier": 2.0,
    "size_down_pct": 50.0,
    "cooldown_intervals": 2
  }
}
```

Actions:

- cancel stale exposed side,
- widen both sides,
- size down,
- hold current orders if already safe.

### Queue Preservation

Use `reconcile` where possible and set tolerances relative to market scale:

```json
{
  "controller": {
    "reconcile": {
      "price_tol_bps": 1.0,
      "qty_tol_rel": 0.05
    }
  },
  "execution": {
    "quoting": {
      "planner": "reconcile"
    }
  }
}
```

Do not cancel merely to move one tick unless the expected edge improvement is
large enough to compensate for lost queue priority.

### Inventory Baiting Protection

Inventory response should not be a single predictable threshold. Use buckets,
hysteresis, and active hedging:

```json
{
  "inventory_protection": {
    "enabled": true,
    "utilization_bucket_pct": 5.0,
    "panic_skew_start_pct": 85.0,
    "panic_skew_hysteresis_pct": 5.0,
    "hedge_start_pct": 70.0,
    "hedge_clip_pct_of_inventory": 25.0,
    "max_hedge_notional_pct_of_daily_volume": 1.0
  }
}
```

When inventory becomes risky, prefer controlled hedging or reduced quoting over
publicly offering obviously distressed prices.

## Implementation Plan

### Phase 1: Low-Risk Config Changes

1. Change quoting execution from `replace` to `reconcile` where the venue and
   private-state latency allow it.
2. Set nonzero reconcile tolerances:
   `price_tol_bps` and `qty_tol_rel`.
3. Increase `execution_after_steps` only for warmup, not as a hardening tool.
4. Keep `fixed_mc_samples` only when reproducibility is more important than
   stochastic robustness. For live trading, evaluate `fixed_mc_samples=false`
   after diagnostics are stable.

### Phase 2: Target-Book Hardening Module

Add a module such as:

```text
mm/strategies/target_book_hardening.py
```

Suggested API:

```python
def harden_joint_target_book(
    book: JointTargetBook,
    *,
    frame: StrategyInputFrame,
    ledger: JointLedgerSnapshot,
    config: TargetBookHardeningConfig,
    venue_specs: SymbolBinding,
    run_id: str,
    frame_step: int,
    decision_hash: str,
) -> tuple[JointTargetBook, TargetBookHardeningReport]:
    ...
```

Call it after `prepare_execution_joint_book()` creates the execution target and
before risk validation, or call it immediately after the MPC target is built and
then risk-check the hardened book. The latter is cleaner because diagnostics can
show:

```text
mpc_target -> hardened_target -> risk_approved_target -> rules_valid -> router_accepted
```

### Phase 3: Timing Jitter

Add execution sync delay inside `StrategyRunner._sync_joint_with_timeout()` or
immediately before it. The timeout budget must account for this delay.

Emergency actions must bypass timing jitter.

### Phase 4: Structural Protection

Add or verify:

- trade outlier and dust filtering,
- local/global divergence gates,
- OFI or toxicity fast path,
- dynamic spread widening,
- inventory bucket hysteresis,
- optional active hedging through liquidity sourcing.

### Phase 5: Observability and Backtesting

Add private diagnostics:

- hardening enabled flag,
- seed fingerprint, not raw seed if logs are shared,
- price jitter bps summary,
- size jitter pct summary,
- dropped level counts,
- held-by-hysteresis counts,
- timing delay ms,
- minimum-edge rejections,
- bypass reasons.

Backtest metrics:

- spread capture,
- fill toxicity,
- adverse selection after fills,
- cancel/place rate,
- queue-preservation rate,
- inventory variance,
- PnL volatility,
- rejection counts,
- stale quote hit rate.

## Safe Defaults

For first production trial:

```json
{
  "hardening": {
    "enabled": true,
    "refresh_hysteresis": {
      "enabled": true,
      "price_change_bps": 0.5,
      "qty_change_pct": 3.0,
      "notional_change_pct": 3.0,
      "max_hold_intervals": 2
    },
    "execution_timing": {
      "enabled": true,
      "jitter_interval_pct": 5.0,
      "max_jitter_ms": 250
    },
    "quote_price": {
      "enabled": true,
      "jitter_bps_of_mid": 0.25,
      "jitter_pct_of_half_spread": 3.0,
      "max_jitter_ticks": 1,
      "never_cross_touch": true
    },
    "quote_size": {
      "enabled": true,
      "per_order_jitter_pct": 5.0,
      "side_total_jitter_pct": 1.0,
      "preserve_side_notional_within_pct": 2.0
    },
    "level_selection": {
      "enabled": false
    },
    "child_orders": {
      "enabled": false
    }
  }
}
```

Then gradually increase:

- timing jitter to 10-15% of interval,
- per-order size jitter to 10-15%,
- price jitter to 1-2 ticks or 5-10% of half-spread,
- level selection only after confirming no edge loss.

## Important Non-Goals

Do not:

- randomize raw market-event windows as the first step,
- allow jitter to cross the touch accidentally,
- allow random sizing to exceed inventory or cash bounds,
- randomize away from stale-reference cancel actions,
- expose hardening diagnostics publicly,
- rely on randomization instead of minimum-edge checks.

## Final Recommendation

The strongest practical package is:

1. Use `reconcile` with tolerances to reduce cancel/repost fingerprinting.
2. Add target-book hardening after MPC and before risk validation.
3. Express all hardening knobs as bps, percentages, interval fractions,
   inventory/cash utilization, fee multipliers, volatility multipliers, or tick
   counts.
4. Add execution timing jitter only for non-emergency, non-risk-reducing syncs.
5. Add structural protections for reference manipulation, toxic flow, queue
   preservation, and inventory baiting.
6. Keep full private diagnostics for replay, but do not leak target and
   hardening internals to shared observers.

This preserves the MPC as the economic brain while making the visible quoting
surface less regular, less baitable, and less expensive to defend.
