# Liquidity Sourcing Architecture Upgrade Plan

Status: implemented. The current runtime contract is the clean local/global
layout described in `docs/low-frequency-market-maker-architecture.md`; the
migration notes below are preserved as historical planning context.

This plan is based on the current code review plus the design notes in
`docs/new_design.txt`. The core change is conceptual: the secondary venue should
no longer be modeled primarily as a reactive "hedge" venue. It should become an
optional liquidity source and reference venue inside a more general joint-venue
market-making architecture.

The target design must still support today's single-exchange mode. Liquidity
sourcing should be a superset that can be disabled by config without forking the
runtime, optimizer, or execution path.

## Current Code Review

The codebase already has useful foundations:

- `mm.runtime.engine` wires a single async runtime with bus-first data flow,
  persistence, observability, control plane, strategy runner, private state, and
  execution services.
- `mm.application.strategy_runner.StrategyRunner` provides one production
  decision path: frame preparation, MPC decision, risk validation, action
  publishing, live execution gates, and diagnostics.
- `mm.core.models.execution.Action` is already declarative: it represents desired
  resting orders, not imperative place commands.
- `mm.execution.reconcile` already contains a tolerance-based reconcile planner
  for matching target orders to live exchange orders.
- `mm.private_state.reconciler.PrivateStateReconciler` already normalizes private
  state into snapshots and per-event topics.
- `mm.execution.plan` and `mm.diagnostics` already provide structured pipeline
  reports, which is a good base for clearer operational observability.
- `mm.runtime.config` already models a primary source plus optional `hedge`
  source and private configs, which gives a migration path to rename concepts
  without immediately breaking runtime JSON files.

The current design also has structural limits that directly conflict with the
new goal:

- The secondary venue is named and wired as `hedge` across config, frame models,
  strategy diagnostics, risk limits, MPC rollout, and live fill-driven execution.
- `MonteCarloMPCTensorBatched` models the secondary venue as a reactive hedge
  leg: primary fills change primary inventory, then a deterministic hedge trade
  offsets that inventory. This is not liquidity sourcing.
- The optimizer action space only produces primary bid/ask quote curves. It does
  not produce independent local and global target books.
- `StrategyInputFrame` carries primary and hedge market/private snapshots, but
  it does not carry a first-class joint market state or ledger snapshot.
- `OrderManager` tracks optimistic live orders while private state snapshots and
  events also represent exchange truth. `ExecutionStateReconciler` mirrors
  private events into the manager, so there are still two state holders.
- `RuntimeRiskEngine` evaluates individual orders against primary wallet state
  and open notional. It does not evaluate a full transition from live book to
  target book, and it does not reason about locked collateral jointly across
  venues.
- Venue hard limits are checked in execution after risk, while order construction
  also applies instrument constraints. This makes rejection reasons harder to
  trace.
- Frame-level failures and order-level failures share the same broad funnel,
  which increases debugging cost.
- `HedgeExecutionService` is fill-driven and reactive. It may remain useful as a
  legacy emergency risk tool, but it should not be the foundation for
  cross-exchange profit capture.

## Target Architecture

### Naming

Use neutral roles in new APIs:

- `local`: the venue where the strategy primarily distributes liquidity.
- `global`: the optional liquidity source and reference venue.
- `single_venue`: local-only operation with global action curves masked out.

Keep legacy `primary` and `hedge` config aliases during migration. New code
should avoid adding more "hedge" concepts unless it specifically refers to the
legacy reactive hedge service.

### Event Flow

The final runtime should flow through state snapshots and target books:

```text
PublicMarketGateway(s)
  -> MarketEvent bus topics

PrivateAccountGateway(s)
  -> BalanceEvent / OrderUpdateEvent / FillEvent / PrivateSnapshot bus topics

PortfolioLedger
  -> JointLedgerSnapshot

FrameAssembler
  -> JointStrategyFrame(MarketState + JointLedgerSnapshot)

StrategyEngine
  -> JointTargetBook

NetDeltaRiskEngine
  -> ApprovedJointTargetBook

ExecutionPlanner(s)
  -> ExecutionPlan(cancel/place commands)

ExchangeRulesGateway
  -> ValidatedExecutionPlan

OrderRouter(s)
  -> exchange adapters
```

### State Ownership

The ledger should become the only in-process source of truth for portfolio and
open-order state.

Private account components should translate exchange payloads into immutable
events and snapshots. They should not own strategy-facing state.

Execution components should route commands and report adapter outcomes. They
should not own optimistic live-order truth. A command is considered real only
when exchange private state or a trusted acknowledgement event updates the
ledger.

### Ledger Snapshot Contract

Add new domain models under `mm/core/models/ledger.py` or `mm/ledger/models.py`:

```text
VenueLedger
  exchange
  symbol
  base_asset
  quote_asset
  balances
  live_orders
  locked_base
  locked_quote
  available_base
  available_quote
  last_private_event_ms
  last_snapshot_ms

JointLedgerSnapshot
  local: VenueLedger
  global: VenueLedger | None
  reference_price
  consolidated_base_inventory
  consolidated_quote_wealth
  created_at_ms
  freshness
```

The first version can be snapshot-driven and deterministic from existing
`private:state:*` topics. Later versions can add event replay, sequence checks,
and durable checkpointing.

### Strategy Frame Contract

Add a new frame model rather than expanding `StrategyInputFrame` indefinitely:

```text
JointStrategyFrame
  local_market: MarketEvent
  global_market: MarketEvent | None
  ledger: JointLedgerSnapshot
  mode: "single_venue" | "liquidity_sourcing"
  sync_status
  created_at_ms
```

Compatibility can be preserved by adapting `StrategyInputFrame` into
`JointStrategyFrame` while the old runner still exists.

### Strategy Output Contract

Introduce target books as the strategy output:

```text
TargetOrder
  exchange
  symbol
  side
  price
  qty
  order_type
  tags

VenueTargetBook
  exchange
  symbol
  actor_id
  orders
  policy

JointTargetBook
  actor_id
  local: VenueTargetBook
  global: VenueTargetBook | None
  mode
  diagnostics
```

`Action` can remain as a compatibility wrapper around target-book orders until
all execution code is migrated.

## Mathematical Design

### Objective

Replace reactive hedge reward with consolidated wealth marked to the global
reference price when available:

```text
W_t =
  local_quote_cash
  + global_quote_cash_in_local_quote_terms
  + (local_base + global_base) * reference_price
```

For same-quote symbols like `USDT_IRT` on both venues, quote conversion may be
identity. If the global venue quote currency differs, add an explicit FX or
conversion-rate provider before enabling live mode.

The MPC objective remains a certainty-equivalent return objective, but the
return is based on joint wealth:

```text
R_i = (W_i,H - W_0) / W_0

J = CE_beta(R) - lambda_inv * joint_inventory_penalty - lambda_capacity * capacity_penalty
```

### Joint Inventory Penalty

Penalize operational imbalance per venue, not just total inventory:

```text
joint_inventory_penalty =
  ((local_base - local_base_target) / local_base_scale)^2
  + mask_global * ((global_base - global_base_target) / global_base_scale)^2
```

Add capacity penalties for running out of venue-specific quote cash or base
inventory:

```text
capacity_penalty =
  shortage(local_available_base_for_sells)
  + shortage(local_available_quote_for_buys)
  + mask_global * shortage(global_available_base_for_sells)
  + mask_global * shortage(global_available_quote_for_buys)
```

### Simulation

The rollout should simulate independent maker fills on both venues:

- Sample local market flow from the local flow model.
- Sample global market flow from a global flow model.
- Apply local bid/ask curves to the local orderbook.
- Apply global bid/ask curves to the global orderbook.
- Update local and global cash/inventory independently.
- Mark consolidated terminal wealth to the reference price.

Single-venue mode should use the same implementation with `global_enabled =
False`, global depth fixed at zero, and global parameters detached or masked.

### Action Space

Current action:

```text
[local_bid_curve, local_ask_curve]
```

Liquidity sourcing action:

```text
[local_bid_curve, local_ask_curve, global_bid_curve, global_ask_curve]
```

The initial implementation can use the same curve family on both venues. After
that is stable, specialize global quoting to a simpler block-style curve if it
improves execution behavior.

## Execution Architecture

### Net-Delta Risk

Replace per-order risk filtering with target-book transition risk:

```text
risk_input = LedgerSnapshot + JointTargetBook
risk_output = ApprovedJointTargetBook | TargetBookRejected
```

Risk should answer:

- If we move from current live orders to this target book, what collateral will
  be locked per venue?
- Does the transition respect available cash, available inventory, max order
  notional, max open notional, min cash, and min inventory?
- Are market and private data fresh enough for the configured mode?
- Is global state required for this mode, or may local-only trading continue?

This fixes additive quoting and collateral double-counting because the risk
engine evaluates the final desired state instead of each new order in isolation.

### Execution Planners

Split planning from routing:

```text
ExecutionPlanner.plan(ApprovedTargetBook, LedgerSnapshot) -> ExecutionPlan
OrderRouter.execute(ExecutionPlan) -> AdapterCommandResult
```

Provide at least two planners:

- `ToleranceReconcilePlanner`: keeps live orders within configured price/qty
  tolerance, cancels orphans, places gaps. Use this for local venue queue
  preservation and rate-limit control.
- `AtomicReplacePlanner`: cancels current scoped orders before placing the new
  target book, or uses native cancel-replace if the exchange supports it. Use
  this for global/source venue when quote freshness is more important than queue
  position.

Planner selection should be per venue:

```json
"execution": {
  "local": {"planner": "reconcile"},
  "global": {"planner": "replace"}
}
```

### Target Book Diff Contract

The order diff must be computed from the approved desired state and the ledger's
exchange truth:

```text
ApprovedTargetBook + LedgerSnapshot.live_orders -> ExecutionPlan
```

It must not use `OrderManager` optimistic state as an input. During migration,
`OrderManager` may call the diff code as a facade, but the source of truth for
live orders should be the ledger snapshot.

Diff inputs:

- `ApprovedTargetBook`: venue-scoped desired resting orders after strategy and
  net-delta risk approval.
- `LedgerSnapshot.live_orders`: current exchange-confirmed live orders.
- `actor_id`: scope key for strategy-owned orders.
- venue policy: `reconcile` or `replace`.
- tolerance config: `price_tol_bps`, `qty_tol_rel`.

Diff output:

```text
ExecutionPlan
  exchange
  symbol
  actor_id
  policy
  keep: LiveOrderRef[]
  cancel: CancelCommand[]
  place: PlaceCommand[]
  rejected: CommandRejection[]
  diagnostics
```

The canonical reconcile algorithm:

1. Scope live orders from the ledger by `actor_id`, `exchange`, `symbol`, and
   side. Ignore all orders owned by other actors.
2. Scope target orders from the approved target book by the same exchange and
   symbol.
3. Match targets to live orders per side:
   - buys are processed from highest price to lowest price
   - sells are processed from lowest price to highest price
   - only unmatched live orders are eligible
   - live order must have the same side, exchange, and symbol
   - price difference must be within `price_tol_bps`
   - quantity difference vs remaining quantity must be within `qty_tol_rel`
4. For each target order:
   - if the best eligible live order exists, add it to `keep`
   - otherwise create a `PlaceCommand`
5. Any scoped live order not matched to a target becomes a `CancelCommand`.

The match score should prefer the closest executable equivalent, not merely the
first eligible order:

```text
price_bps = abs(target.price - live.price) / max(abs(target.price), abs(live.price), eps) * 10000
qty_rel = abs(target.qty - live.remaining_qty) / max(target.qty, live.remaining_qty, eps)
score = price_bps + qty_rel * qty_weight
```

Use a stable `qty_weight` config or keep the current reconcile behavior during
the first migration. Ties should be deterministic, preferably by oldest live
order timestamp, then exchange order id.

Replace policy is simpler:

1. Scope all live orders by `actor_id`, exchange, and symbol.
2. Emit cancels for every scoped live order.
3. Emit places for every target order.
4. Dispatch places only after cancel acknowledgements, unless the adapter exposes
   a trusted atomic cancel-replace command.

Risk and rules boundaries:

- Net-delta risk runs before the diff and approves the final target book state.
- The diff decides which existing orders can be kept and which commands are
  needed to reach the target state.
- Exchange rules run after the diff and validate `PlaceCommand` objects only.
- Cancel commands should not fail venue min-size/min-notional rules.
- Router/adapter failures should not mutate ledger state directly; the ledger
  changes only from private exchange events or trusted command acknowledgements
  that are modeled as ledger events.

### Exchange Rules Gateway

Create one gateway for hard venue formatting and limits:

```text
ExchangeRulesGateway.validate(plan) -> validated_plan + rejected_commands
```

This should own tick size, lot size, min qty, min notional, and command
formatting preconditions. Rejections should have one stable reason taxonomy:

- `rules_min_qty`
- `rules_min_notional`
- `rules_tick_size`
- `rules_qty_step`
- `rules_price_band`

Long term, order construction should produce economic intent first, and this
gateway should be the only hard venue-rule rejection point before the adapter.

### Order Router

The router should be deliberately simple:

- Receives validated primitive commands.
- Calls exchange adapters.
- Publishes command requested, accepted, rejected, and failed events.
- Does not mutate live-order state directly.

During migration, `OrderManager` can wrap the new planner/router interfaces, but
new code should move toward stateless routing.

## Migration Phases

### Phase 0: Lock Current Behavior

Goal: protect today's runtime while refactoring.

Work:

1. Add regression tests for current single-venue and primary-plus-hedge config
   loading.
2. Add tests for `StrategyRunner` observe-only, warmup, and live gates.
3. Add tests documenting current `OrderManager` reconcile counts and rejection
   labels.
4. Mark `HedgeExecutionService` and `hedge_rollout_enabled` as legacy in docs,
   but do not remove them yet.

Acceptance:

- Existing unit tests pass.
- The example config still loads.
- No runtime topic is removed.

### Phase 1: Introduce Neutral Config and Domain Models

Goal: make the new architecture expressible without changing behavior.

Work:

1. Add neutral role aliases: `local` maps to current `primary`, `global` maps to
   current `hedge`.
2. Add `liquidity_sourcing` config:

   ```json
   "liquidity_sourcing": {
     "enabled": false,
     "mode": "single_venue",
     "reference_venue": "global",
     "require_global_private": false
   }
   ```

3. Add target-book models and ledger snapshot models.
4. Add compatibility adapters:
   - `StrategyInputFrame -> JointStrategyFrame`
   - `Action -> VenueTargetBook`
   - `VenueTargetBook -> Action`
5. Update docs and config contracts to explain legacy aliases.

Acceptance:

- Existing strategy/execution behavior is unchanged with default config.
- New models serialize cleanly through persistence.
- Import-boundary tests remain lightweight.

### Phase 2: Build the Portfolio Ledger

Goal: create a deterministic state source beside existing manager state.

Work:

1. Add `mm.ledger` package with:
   - `models.py`
   - `state_machine.py`
   - `service.py`
   - `replay.py`
2. Subscribe the ledger service to existing private topics:
   - `private:state:{exchange}:snapshot`
   - `private:state:{exchange}:balances`
   - `private:state:{exchange}:open_orders`
   - `private:state:{exchange}:order_updates`
   - `private:state:{exchange}:fills`
3. Publish `ledger:snapshot` and `ledger:snapshot:{exchange}` topics.
4. Add consistency diagnostics comparing ledger live orders to current
   `OrderManager.live_orders`.
5. Add replay tests that rebuild a ledger from event sequences.

Acceptance:

- Ledger state matches `PrivateStateSnapshot` for balances and open orders.
- Ledger can reconstruct fills and terminal order states from event replay.
- Strategy still uses old state until parity is proven.

### Phase 3: Refactor Frame Assembly Around Ledger

Goal: provide strategy decisions with market state plus ledger state.

Work:

1. Add `JointStrategyFrameAssembler` or extend the existing assembler behind a
   new interface.
2. Consume `ledger:snapshot` instead of private snapshots for strategy-facing
   wallet and open-order state.
3. Add explicit synchronization status:
   - `local_market_ready`
   - `global_market_ready`
   - `local_private_ready`
   - `global_private_ready`
   - `ledger_fresh`
   - `latency_skew_ms`
4. Keep current `StrategyInputFrame` publication until the strategy runner is
   migrated.

Acceptance:

- Frames expose a single coherent state object.
- Local-only frames do not fail because global private state is missing when
  liquidity sourcing is disabled.
- Diagnostics report latency skew separately from risk rejection.

### Phase 4: Target-Book Strategy Output

Goal: separate strategy intent from execution mechanics.

Work:

1. Update `MpcStrategyEngine` to emit `JointTargetBook` internally, then convert
   to `Action` for the old execution path.
2. Move ladder construction into a target-book builder that can emit venue-scoped
   target books.
3. Rename diagnostics stages:
   - `target_book`
   - `risk_approved_target`
   - `planned_cancel`
   - `planned_place`
   - `rules_valid`
   - `router_accepted`
4. Keep old action pipeline fields as deprecated aliases until dashboards are
   migrated.

Acceptance:

- Strategy decisions still publish `Action` for compatibility.
- New target-book payloads are persisted.
- Grafana can show target, planned, valid, accepted without the old `placed`
  ambiguity.

### Phase 5: Net-Delta Risk Engine

Goal: approve or reject the desired final book state, not isolated orders.

Work:

1. Add `NetDeltaRiskEngine`.
2. Compute collateral from `LedgerSnapshot.live_orders` and proposed target
   books per venue.
3. Evaluate final locked cash/base per venue.
4. Split risk outcomes into:
   - frame gates
   - target-book gates
   - command gates
5. Add mode-aware freshness rules:
   - single venue: global market/private state optional
   - liquidity sourcing observe-only: global market required, global private
     optional if no global orders are emitted
   - liquidity sourcing live: global market and global private required when
     global target book is non-empty

Acceptance:

- Additive quote stacking no longer causes false cash/inventory failures.
- Risk diagnostics explain the transition that was rejected.
- Existing `RuntimeRiskEngine` remains available behind config during migration.

### Phase 6: Execution Planner and Router Split

Goal: make execution stateless with respect to live-order truth.

Work:

1. Add `mm.execution.target_book`, `mm.execution.planner`,
   `mm.execution.rules_gateway`, and `mm.execution.router`.
2. Implement `ToleranceReconcilePlanner` using the existing reconcile algorithm.
3. Implement `AtomicReplacePlanner`.
4. Move venue limit checks from `OrderManager` into `ExchangeRulesGateway`.
5. Introduce `OrderRouter` that publishes command lifecycle events and calls
   adapters.
6. Keep `OrderManager` as a compatibility facade while `StrategyRunner` is
   migrated.

Acceptance:

- Planner output is deterministic from target book plus ledger snapshot.
- Router does not update live orders directly.
- Cancel-before-place behavior is test-covered for replace mode.
- Reconcile mode preserves queue-position behavior where tolerances match.

### Phase 7: Joint-Venue MPC Rollout

Goal: turn liquidity sourcing into an optimizer-native behavior.

Work:

1. Add a joint state object for MPC:
   - local orderbook features
   - global orderbook features
   - local wallet
   - global wallet
   - reference price
   - mode mask
2. Generalize `soft_fill_batched` or add a venue-vectorized equivalent.
3. Extend action parameterization to four curves.
4. Add global flow and price models. Start with shared model classes and separate
   parameters per venue.
5. Replace `hedge_reward` with consolidated wealth objective.
6. Add single-venue masks so global quote depth is exactly zero when disabled.
7. Add optimizer diagnostics for local and global expected fills, inventory, and
   PnL contributions.

Acceptance:

- With liquidity sourcing disabled, optimizer output matches current local-only
  behavior within expected numerical tolerance.
- With liquidity sourcing enabled in simulation, the optimizer can place local
  and global target books independently.
- No reactive hedge trade is required for objective value.

### Phase 8: Live Liquidity Sourcing Mode

Goal: route approved joint target books to both exchanges.

Work:

1. Enable per-venue execution policies:
   - local: reconcile
   - global: replace or reconcile by config
2. Add source-venue order size caps and quote-distance caps independent from
   local venue caps.
3. Add config to disable global live placement while still using global market
   data as a reference.
4. Add emergency controls:
   - pause local venue
   - pause global venue
   - cancel local actor orders
   - cancel global actor orders
   - cancel all actor orders
5. Keep legacy `HedgeExecutionService` disabled by default and mutually exclusive
   with live liquidity sourcing unless explicitly allowed in an emergency mode.

Acceptance:

- A single strategy decision can produce two venue target books.
- Each venue can be independently observe-only or live.
- Global state staleness blocks global target placement without necessarily
  blocking local-only placement when configured that way.

### Phase 9: Sweeper and Capital Operations

Goal: handle physical capital drift outside the high-frequency decision loop.

Work:

1. Add a `CapitalSweeperService` in observe-only mode first.
2. Monitor ledger imbalances and publish rebalance recommendations.
3. Add manual approval hooks before any real transfer.
4. Track transfer state separately from trading state.

Acceptance:

- The strategy reports inventory drift and rebalance needs.
- No automated transfer is possible without explicit config and operator
  controls.

## Configuration Migration

Suggested future config shape:

```json
{
  "venues": {
    "local": {
      "exchange": "nobitex",
      "symbol": "USDT_IRT",
      "public": {"mode": "rest"},
      "private": {"mode": "rest", "api_token_env": "NOBITEX_API_TOKEN"}
    },
    "global": {
      "exchange": "wallex",
      "symbol": "USDT_IRT",
      "public": {"mode": "rest"},
      "private": {"mode": "rest", "api_token_env": "WALLEX_API_TOKEN"}
    }
  },
  "liquidity_sourcing": {
    "enabled": false,
    "reference_venue": "global",
    "global_live_orders": false
  },
  "execution": {
    "local": {"planner": "reconcile"},
    "global": {"planner": "replace"}
  }
}
```

Migration rules:

- Continue accepting `primary` as an alias for `venues.local`.
- Continue accepting `hedge` as an alias for `venues.global`.
- Continue accepting `hedging` only for the legacy fill-driven hedge service.
- Emit config warnings when legacy keys are used, but do not fail initially.

## Observability Plan

Add or update dashboards around the new concepts:

- Ledger freshness and event lag per venue.
- Ledger balances, available balances, locked collateral, and live order counts.
- Target-book risk: requested locked collateral vs approved locked collateral.
- Execution planning: kept, canceled, planned place, rules-valid, router-accepted.
- Joint MPC diagnostics: local/global expected fills, wealth contribution,
  capacity penalty, and inventory penalty.
- Liquidity sourcing mode status: disabled, observe-only, global-reference-only,
  live joint quoting.

The rejection taxonomy should separate:

- frame gates: stale data, missing private state, kill switch
- target-book gates: insufficient collateral, inventory capacity, notional limits
- rules gates: min notional, tick size, lot size
- router/adapter failures: exchange rejection, timeout, network/API failure

## Test Strategy

Add focused tests as each phase lands:

- Ledger event replay and snapshot parity.
- Frame assembler synchronization and latency skew behavior.
- Target-book serialization and compatibility conversion to/from `Action`.
- Net-delta collateral accounting with existing live orders.
- Reconcile planner keeps matching orders and cancels orphans.
- Replace planner cancels before placing.
- Rules gateway rejects invalid orders with stable reason codes.
- Joint MPC single-venue mask produces zero global depth.
- Joint MPC liquidity-sourcing toy scenario prefers buying cheaper venue and
  selling richer venue when inventory/cash constraints allow it.
- Runtime config migration from current example JSON to new aliases.

## Rollout Order

Recommended order for real implementation:

1. Add neutral models and config aliases.
2. Add ledger beside current state and compare it in diagnostics.
3. Move frame/risk reads to ledger once parity is established.
4. Introduce target books while preserving `Action` compatibility.
5. Replace order-level risk with net-delta risk.
6. Split planner/router and keep `OrderManager` as facade.
7. Upgrade MPC to joint-venue simulation.
8. Enable observe-only liquidity sourcing.
9. Enable live global target placement with strict caps.
10. Add capital sweeper recommendations.

## Non-Goals For The First Upgrade

- Do not remove existing `primary`/`hedge` config keys immediately.
- Do not automate cross-exchange transfers in the first liquidity-sourcing
  release.
- Do not enable live global quoting by default.
- Do not remove legacy `HedgeExecutionService` until the new joint execution path
  has live-history confidence.
- Do not rewrite all exchange adapters unless the planner/router interface
  requires a small compatibility shim.

## Main Risks

- Inventory accounting errors are the highest-risk area. The ledger should be
  introduced in shadow mode first.
- Quote-currency conversion must be explicit before consolidated wealth can be
  trusted for venues with different quote assets.
- Latency skew can create false arbitrage. The frame should expose skew and the
  risk engine should gate or shrink target books when state is stale.
- Replacing reactive hedge behavior with maker-style global quoting changes fill
  risk. The first rollout should be observe-only and simulation-backed.
- A broader action space increases optimizer instability. Keep single-venue masks
  and diagnostics strong before optimizing four curves live.

## Definition Of Done

The upgrade is complete when:

- The code can run in single-venue mode without requiring global market or
  private state.
- The code can run in liquidity-sourcing observe-only mode and emit local plus
  global target books.
- The strategy objective evaluates consolidated joint wealth without reactive
  hedge reward.
- Risk approves target-book transitions using ledger state and net locked
  collateral.
- Execution planning is deterministic from approved target book plus ledger
  snapshot.
- The router does not own live-order truth.
- Operators can explain every dropped quote from one rejection code and one
  pipeline stage.
