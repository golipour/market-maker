# Target order ladder execution

This document explains how an MPC target order ladder becomes live exchange
orders, and exactly where orders are kept, canceled, placed, rejected, or only
previewed.

The most important distinction is:

```text
ladder orders -> target book -> approved target book -> execution plan -> adapter commands -> private exchange truth
```

These are different objects. A target order is desired state. A place command is
an execution diff. A placed order is an adapter response with an exchange order
id. A live order is exchange-confirmed private state.

## Code map

| Step | Main code |
|------|-----------|
| Build ladder from optimizer output | `mm/core/order_constraints.py` |
| Wrap ladder as local/global target books | `mm/strategies/target_book_builder.py` |
| Run per-frame strategy, risk, gates, and routing | `mm/application/strategy_runner.py` |
| Apply execution caps and per-venue gates | `mm/application/venue_execution.py` |
| Approve/reject target-book transition | `mm/risk/net_delta.py` |
| Diff target book against live orders | `mm/execution/planner.py`, `mm/execution/reconcile.py` |
| Validate place commands against venue limits | `mm/execution/rules_gateway.py` |
| Send cancel/place calls to exchange adapter | `mm/execution/router.py` |
| Publish execution reports and snapshots | `mm/execution/order_manager.py`, `mm/execution/reporting.py` |

## Data objects

### Ladder `Order`

The first executable-looking object is `mm.core.models.execution.Order`:

```text
exchange, symbol, side, price, qty, order_type, client_order_id
```

At ladder-building time this is still only a strategy candidate. It has not
passed venue constraints, risk, execution gates, or planner diffing.

### `TargetOrder`

`TargetOrder` is the desired resting order stored in a venue target book:

```text
exchange, symbol, side, price, qty, order_type, tags
```

Current conversion from ladder `Order` to `TargetOrder` does not preserve the
ladder `client_order_id`. Later reconciliation matches by exchange, symbol,
side, price, and remaining quantity tolerance, not by the ladder id.

### `VenueTargetBook`

One venue target book means:

```text
for actor_id on exchange/symbol, this is the desired resting book
```

It also carries a planner policy:

| Policy | Meaning |
|--------|---------|
| `reconcile` | Keep live orders that already match target orders; cancel unmatched live orders; place unmatched targets. |
| `replace` | Cancel all scoped live orders, then place the full target book. Empty `replace` means cancel scoped live orders. |

### `JointTargetBook`

`JointTargetBook` contains:

- `local`: always present
- `global`: optional
- `mode`: `single_venue` or `liquidity_sourcing`
- `diagnostics`: currently includes local/global target counts

Local and global execution routing are per venue, but they are not perfectly
symmetric. Local is the primary leg. Global failures should not block local when
local passes risk, but a hard local risk failure can prevent the whole approved
book unless replace-mode empty books are approved.

## Ladder construction

`MpcStrategyEngine` asks `build_joint_target_book_from_sigmoid` to turn optimizer
output into target books.

For each venue role, `build_orders_from_sigmoid_with_report` performs only
strategy-owned steps:

1. Build a raw price/quantity ladder.
2. Cap the number of orders per side.

Venue tick/lot rounding, minimum quantity, and minimum notional are no longer
strategy ladder concerns. They are applied at the execution boundary by
`OrderManager` / `ExchangeRulesGateway`, after risk has approved the target
book and before place commands reach the exchange adapter.

### Raw ladder

`ladder_orders_from_sigmoid_with_report` does this:

1. Compute `mid` from the best bid/ask of the selected market event.
2. Read spread levels from `orderbook_grid.spread_ladder()`.
3. For each side, call the optimizer depth function at each spread.
4. Convert cumulative depth to marginal order quantity:

```text
marginal_qty = max(depth(spread_i) - depth(spread_{i-1}), 0)
```

5. Set price relative to mid:

```text
buy price  = mid - spread
sell price = mid + spread
```

6. Drop invalid candidates:

| Rejection | Meaning |
|-----------|---------|
| `zero_marginal_qty` | The marginal quantity at this spread is zero or negative. |
| `invalid_price` | The resulting price is non-positive or not finite. |

The stage report is `action_pipeline.ladder`.

### Per-side strategy cap

`cap_orders_per_side_with_report` sorts buy and sell orders by descending
quantity and keeps only the largest `max_orders_per_side` orders per side.

| Rejection | Meaning |
|-----------|---------|
| `max_orders_per_side_zero` | Cap is zero, so all orders are dropped. |
| `capped_buy` | Extra buy orders were dropped. |
| `capped_sell` | Extra sell orders were dropped. |

The stage report is `action_pipeline.capped`. Venue tick/lot rounding and
minimum-size validation are reported later by the execution rules stage.

### Target book output

After capping:

- local orders become `JointTargetBook.local.orders`
- global orders become `JointTargetBook.global.orders` only when liquidity
  sourcing is active and global inputs/constraints/action are available
- `action_pipeline.target_book` summarizes local plus global target orders

Current detail to know: the local action pipeline has detailed ladder,
quantized, and capped stages. The global target book is produced, but the
global ladder-stage report is not preserved in `ControllerDiagnostics` today.

## Risk approval

The strategy runner validates the target book before execution.

For `NetDeltaRiskEngine`, risk evaluates the desired transition:

```text
current ledger live orders + balances -> proposed target book
```

### Local risk

Local target-book risk is evaluated against the local ledger and local limits.
If local passes, the local book can be approved. If local fails:

- `replace` may approve an empty local book so existing scoped orders can be
  canceled
- otherwise the approved joint target book can become `None`

### Global risk

Global risk is evaluated only when a non-empty global target book exists.
Possible global rejection reasons include:

- `missing_global_market_data`
- `stale_global_market_data`
- `missing_global_ledger`
- `global_*` prefixed collateral or transition reasons

If local passes and global fails, local can still be approved. In that case the
global side may be omitted or replaced with an empty global book, depending on
the target and policy. Global rejects should not be treated as local blockers.

### Risk output

Risk returns:

- `approved`: approved `JointTargetBook` or `None`
- `rejection_counts`
- `exposure_before`
- `exposure_after`
- transition diagnostics

The runner maps this into:

- `risk` diagnostics
- `action_pipeline.risk_approved_target`
- `strategy:decisions`

## Execution preparation

After risk, `prepare_execution_joint_book` adjusts the approved book for live
execution.

It applies:

1. Global live-order filter:
   - if `liquidity_sourcing.global_live_orders` is false, global orders are
     emptied before live execution.
2. Planner policy:
   - local defaults to `execution.local.planner`
   - global defaults to `execution.global.planner`
   - config defaults are local `reconcile`, global `replace`
3. Venue execution caps:
   - `max_order_qty`
   - `max_quote_distance` from the current venue mid

The result is the `execution_book`. This is the book sent to per-venue gates and
then to order managers.

## Execution gates

`compute_venue_execution_gates` builds one gate per venue role.

Shared frame/controller blockers are copied into both venues:

- `dry_run_config`
- `dry_run_control`
- `warmup`
- `paused_actor`

Venue-specific blockers are then added:

| Blocker | Scope |
|---------|-------|
| `venue_observe_only` | That venue config |
| `venue_paused` | That role in the control plane |
| `no_manager` | That exchange |
| `global_live_orders_disabled` | Global only |
| `stale_global_market_data` | Global only |
| `no_orders` | That venue when there is no actionable book |

The gate fields are:

| Field | Meaning |
|-------|---------|
| `has_orders` | Target book has at least one order. |
| `can_sync` | There is a venue book and either it has orders or its policy is `replace`. |
| `live` | No blocked reasons are present. |

Important consequences:

- Empty `replace` can sync, because it means cancel scoped live orders.
- Empty `reconcile` does not sync through target-book routing.
- Global venue blocks do not make the local gate blocked.
- A frame executes live if any venue has `live and can_sync` and risk allows execution.

## Routing to order managers

`route_joint_target_book_with_report` iterates over `book.local` and
`book.global` independently.

For each venue book:

1. Find the gate by exchange.
2. Skip the venue if there is no gate or `gate.can_sync` is false.
3. Find the `OrderManager` for that exchange.
4. If `execute=True` and `gate.live=True`, call
   `manager.sync_venue_target_book(venue_book)`.
5. Otherwise call `manager.preview_venue_target_book(venue_book)` when
   available.

The returned `ExecutionSyncReport`s are merged. This means local and global
planner reports are combined in `action_pipeline.planned_*`, `rules_valid`,
`router_accepted`, and `kept` diagnostics unless a dashboard/query splits them
by venue source.

## Order manager sync

`OrderManager.sync_venue_target_book` owns the venue-level transaction:

1. Publish `ExecutionSyncRequested`.
2. Build an execution plan from the target book and current `OrderManager.live_orders`.
3. Validate place commands with exchange rules.
4. If the plan has cancels or valid places, call the router.
5. Apply optimistic local updates from router results.
6. Convert the plan/result into `ExecutionSyncReport`.
7. Publish `ExecutionSyncCompleted`.
8. Publish an `execution:state:{exchange}` snapshot.

The order manager has a lock, so one sync per exchange runs at a time.

## Planning policies

The planner input is:

```text
approved VenueTargetBook + current venue live_orders
```

The planner output is:

```text
keep[], cancel[], place[], rejected[]
```

### `reconcile`

`ToleranceReconcilePlanner` calls `reconcile_target_orders`.

It scopes live orders by:

- same actor id
- same exchange
- same symbol when a symbol is set

Unattributed live orders are not considered scoped for `reconcile`.

The matching algorithm:

1. Split desired and live orders by side.
2. Sort desired buys high-to-low price; sells low-to-high price.
3. For each target, find the unmatched live order on the same side that is
   within configured price and quantity tolerance.
4. Keep the best matching live order.
5. Place targets that did not match any live order.
6. Cancel scoped live orders that were not matched.

If there are no scoped desired orders, reconcile planning would cancel scoped
live orders. In the target-book routing path, however, empty `reconcile` books
normally do not pass `can_sync`, so they are not sent as cancel-only syncs.

### `replace`

`AtomicReplacePlanner` treats the target book as authoritative.

It scopes live orders by:

- same exchange
- same symbol
- same actor id, plus unattributed orders for that exchange/symbol

Then it:

1. Keeps nothing.
2. Cancels every scoped live order.
3. Places every target order.

Empty `replace` is actionable. It cancels scoped live orders and places nothing.

## Rules gateway

The raw planner output may contain place commands that are not yet in venue
wire shape. `ExchangeRulesGateway` normalizes and validates each `PlaceCommand`
against the latest venue limits:

- price is rounded down to the venue tick size when known
- quantity is rounded down to the venue lot size when known
- the rounded order is checked against min quantity and min notional

Invalid place commands are moved from `plan.place` to `plan.rejected` with
stable rule reasons such as:

- `rules_min_qty`
- `rules_min_notional`
- `rules_invalid_price_or_qty`

`ExecutionSyncReport.to_place` records the raw planner place candidates.
`ExecutionSyncReport.limit_valid` records only normalized commands that survived
rules validation. Its `rejections` array carries rejected order price/qty,
side, and reason for per-stage ladder dashboards.

Current implementation detail: `OrderValidationRejected` events are published
by the router. If a plan has only rejections and no cancels or valid places, the
order manager does not call the router, so the sync report still records the
rejection count but no per-order `OrderValidationRejected` event is emitted.

## Router execution

`OrderRouter.execute` sends validated primitive commands to the adapter.

### Cancel phase

Cancellations are attempted before placements.

If there are cancel commands:

1. Publish `CancelOrdersRequested`.
2. Call `command_adapter.cancel_orders(order_ids)`.
3. On success, publish `CancelOrdersAccepted`.
4. On failure, publish `CancelOrdersFailed`.

For `replace`, cancel failure stops placement for that plan. This avoids placing
a full replacement book while stale scoped orders are still live. For
`reconcile`, the router continues to the place phase after a cancel failure.

### Place phase

If there are valid place commands:

1. Publish `PlaceOrdersRequested`.
2. Call `command_adapter.place_orders(orders)`.
3. On success, receive exchange order ids.
4. Publish `PlaceOrdersAccepted` with the accepted orders and ids.
5. On failure, publish `PlaceOrdersFailed` and record router-stage rejections.

After cancel/place attempts, the router publishes `OrderValidationRejected` for
each `plan.rejected` command when the router was invoked.

## Optimistic state and exchange truth

After a successful router call, `OrderManager` applies optimistic updates:

- canceled ids are removed from `OrderManager.live_orders`
- placed ids are inserted as open orders with `remaining_qty = qty`

This prevents immediate duplicate work on the next frame. It is not the final
source of truth.

Exchange-private data remains authoritative:

1. Private snapshot/order/fill events arrive from the exchange adapter.
2. Private state updates the ledger.
3. Execution state reconciler applies private events back into `OrderManager`.
4. `execution:state:{exchange}` snapshots converge to exchange-confirmed open
   orders.

Fills are not created by `PlaceOrdersAccepted`. Fills come from
`private:state:{exchange}:fills` and update remaining quantity or close the live
order when remaining quantity reaches zero.

## Execution report fields

`execution_plan_to_sync_report` maps the plan/result into diagnostics:

| Field | Meaning |
|-------|---------|
| `target` | Desired target book for that venue. |
| `kept` | Live orders matched and retained by the planner. |
| `canceled` | Live orders selected for cancellation. |
| `to_place` | Raw planner place commands before rules validation. |
| `limit_valid` | Place commands that passed rules validation. |
| `accepted` | Orders accepted by the adapter in this sync. Empty in preview mode. |
| `venue_rejection_counts` | Counts of rejected place commands by reason. |
| `cancel_count` | Number of cancel commands. |
| `placed` | Whether any order was accepted by the adapter. |

`apply_execution_stages_to_pipeline` writes these into controller diagnostics:

| Diagnostic field | Source |
|------------------|--------|
| `action_pipeline.target_book` | `ExecutionSyncReport.target` |
| `action_pipeline.kept` | `ExecutionSyncReport.kept` |
| `action_pipeline.planned_cancel` | `ExecutionSyncReport.canceled` |
| `action_pipeline.planned_place` | `ExecutionSyncReport.to_place` |
| `action_pipeline.rules_valid` | `ExecutionSyncReport.limit_valid` plus rule rejection counts |
| `action_pipeline.router_accepted` | `ExecutionSyncReport.accepted` when live execution ran; otherwise `rules_valid` in preview |

## Event timeline

A successful live replace or reconcile sync usually emits:

```text
ExecutionSyncRequested
CancelOrdersRequested        optional
CancelOrdersAccepted         optional
PlaceOrdersRequested         optional
PlaceOrdersAccepted          optional
OrderValidationRejected      optional
ExecutionSyncCompleted
execution:state snapshot
private:state fills/snapshot   later, exchange-confirmed
ledger:snapshot                later, exchange-confirmed
```

An execution sync can be meaningful even when no orders are placed:

- all desired orders already matched live orders
- replace target was empty and only cancellations were required
- all place commands failed rules validation
- execution was only previewed because gates were blocked
- venue had no `can_sync` gate and was skipped

## Common debugging questions

### Why did the target book have orders but no `PlaceOrdersAccepted` event?

Check these in order:

1. Did risk approve the target book?
2. Did `execution_gates.venue_gates[exchange].live` pass?
3. Did `execution_gates.venue_gates[exchange].can_sync` pass?
4. Did the planner keep matching live orders instead of placing new ones?
5. Did rules validation reject the place commands?
6. Did a replace cancel fail before placement?
7. Did the adapter raise `PlaceOrdersFailed`?

### Why were orders canceled when target orders existed?

In `reconcile`, unmatched scoped live orders are orphans and are canceled.
Matching requires same side, same symbol, same actor, and price/quantity within
tolerance.

In `replace`, all scoped live orders are canceled first, even if an equivalent
target order will be placed afterward.

### Why did an empty target book cancel orders?

Because the venue policy was `replace`. Empty `replace` means:

```text
desired state = no scoped live orders
```

Therefore scoped live orders should be canceled.

### Why did global rejects not stop local orders?

Global target-book risk is evaluated separately after local risk. If local
passes and global fails, the approved book can keep local orders and empty or
omit the global book. Shared controller gates can still block both venues, but
global-specific risk and venue gate reasons should not be interpreted as local
blockers.

### Why did live-order charts differ from placed-order charts?

Placed/canceled/order-validation events are historical events. Live-order
charts should come from private state, ledger state, or `execution:state`
snapshots. A single order can appear in many state snapshots, but that does not
mean it was placed many times.

## Operational invariant

After private state catches up, these should converge:

```text
exchange open orders
  == private snapshot open_orders
  == ledger live_orders
  == OrderManager.live_orders
  == execution:state live_orders
```

Short differences immediately after REST commands are expected. Persistent
differences mean the private adapter, ledger reconciliation, or execution state
reconciler needs investigation.
