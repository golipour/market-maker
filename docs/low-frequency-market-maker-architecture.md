# Low-Frequency Market Maker Architecture

This document describes the implemented low-frequency market maker after the
liquidity-sourcing migration. The runtime is still a single-process async system
with an event-bus spine, but the strategy surface is now neutral local/global
venue orchestration.

The production controller path is:

```text
local/global public data
  -> SymbolEngine
  -> MarketEventBuilder on IntervalScheduler
  -> market:events:{exchange}:{symbol}:{interval_ms}
  -> StrategyInputFrameAssembler
  -> JointStrategyFrameAssembler + PortfolioLedgerService
  -> strategy:joint:frames:{local_exchange}:{symbol}:{interval_ms}
  -> StrategyRunner
  -> clean_strategy_frame
  -> MpcStrategyEngine.decide
  -> JointTargetBook + Action
  -> NetDeltaRiskEngine.validate_frame_with_result
  -> per-venue execution gates
  -> route_joint_target_book_with_report
  -> OrderManager.sync_venue_target_book / preview_venue_target_book
  -> exchange order adapter
```

The main strategy intent is `JointTargetBook`, published to
`strategy:target_books:{actor_id}`. `StrategyDecision`, `Action`, and
`ControllerDiagnostics` provide the operational review surface on
`strategy:diagnostics:{actor_id}`.

## Implemented Model

Neutral venue names are first-class in new runtime code:

| Role | Meaning |
| --- | --- |
| `local` | Venue where the strategy primarily distributes liquidity |
| `global` | Optional liquidity source and reference venue |
| `single_venue` | Local-only mode; global curves/orders are inactive |
| `liquidity_sourcing` | Joint local/global MPC mode with optional global live orders |

There is no live reactive secondary-venue service in the current runtime wiring.
Cross venue behavior is represented by joint target books plus per-venue
execution gates.

## Runtime Composition

Main wiring is in `mm.runtime.engine.build_runtime_components`.

`RuntimeComponents` owns:

| Component | Module | Responsibility |
| --- | --- | --- |
| Event bus | `mm.bus.factory` | In-memory bus or Redis Streams fan-out |
| Scheduler | `mm.aggregation.scheduler.IntervalScheduler` | Periodic interval callbacks |
| Symbol engines | `mm.ingestion.symbol_engine.SymbolEngine` | Public trade/book ingestion for each source |
| Input frame assembler | `mm.application.frame_assembler.StrategyInputFrameAssembler` | Align local/global market and private snapshots |
| Joint frame assembler | `mm.application.joint_frame_assembler.JointStrategyFrameAssembler` | Attach ledger/freshness and publish `JointStrategyFrame` |
| Portfolio ledger | `mm.ledger.service.PortfolioLedgerService` | Shadow ledger from private state topics |
| Control plane | `mm.runtime.control.RuntimeControlPlane` | Kill switch, observe-only, paused actors/venues |
| Strategy runner | `mm.application.strategy_runner.StrategyRunner` | Decision, risk, diagnostics, execution routing |
| Private reconcilers | `mm.application.reconciliation_service.PrivateStateReconciler` | Normalize exchange private state |
| Execution reconcilers | `mm.application.reconciliation_service.ExecutionStateReconciler` | Mirror private state into order managers |
| Order managers | `mm.application.execution_service.OrderManager` | Venue target-book reconciliation and adapter commands |
| Persistence daemons | `mm.persistence.postgres`, `mm.persistence.parquet_writer` | Durable event envelopes |
| Observability daemons | `mm.observability.subscribers` | Metrics and logs |

Startup order:

1. Persistence and observability subscribers start.
2. The joint frame assembler starts the input-frame assembler listeners.
3. The strategy runner starts.
4. The portfolio ledger starts, if private state is configured.
5. The control plane starts.
6. Public symbol engines start.
7. Execution and private reconcilers start.
8. The interval scheduler publishes market events.

Shutdown stops the scheduler, ledger, strategy runner, control plane,
reconcilers, daemons, frame assembler, engines, and bus.

## Configuration

Runtime JSON is loaded by `mm.runtime.config.runtime_config_from_mapping`.

Top-level structure:

| Key | Meaning |
| --- | --- |
| `venues.local` | Required public source and instrument |
| `venues.global` | Optional reference/liquidity-source venue |
| `interval_ms` | Strategy decision interval |
| `max_global_lag_ms` | Maximum aligned global market event lag |
| `global_grace_ms` | Short wait for same-interval global event |
| `local_private`, `global_private` | Private state and execution credentials |
| `liquidity_sourcing` | `single_venue` vs `liquidity_sourcing` mode and global-order switch |
| `execution` | Per-venue planner, observe-only, and placement caps |
| `controller` | MPC, risk, order-grid, and diagnostics settings |
| `bus` | Memory or Redis Streams |
| `persistence` | Postgres and/or Parquet event writers |
| `observability` | Metrics, JSON logs, run id, OTEL settings |

`liquidity_sourcing.global_live_orders` is the live switch for global resting
orders. When it is false, global target books may be produced for diagnostics,
but `prepare_execution_joint_book()` strips global orders before routing.

`execution.local.planner` defaults to `reconcile`; `execution.global.planner`
defaults to `replace`. Both venues also support `observe_only`,
`max_order_qty`, and `max_quote_distance`.

## Event Bus And Topics

Core topics:

| Topic | Producer | Main consumers |
| --- | --- | --- |
| `market:events:{ex}:{symbol}:{interval_ms}` | `MarketEventBuilder` | Frame assembler, persistence |
| `strategy:joint:frames:{local_ex}:{symbol}:{interval_ms}` | `JointStrategyFrameAssembler` | Strategy runner, observability |
| `strategy:decisions` | Strategy runner | Persistence, dashboards |
| `strategy:actions` | Strategy runner | Compatibility consumers, persistence |
| `strategy:target_books:{actor_id}` | Strategy runner | Persistence, liquidity-sourcing dashboards |
| `strategy:diagnostics:{actor_id}` | Strategy runner | Persistence, Grafana |
| `strategy:journal:{actor_id}` | Strategy runner | Operational summaries |
| `risk:events` | Strategy runner | Audit/debugging |
| `execution:orders:{exchange}` | OrderManager | Observability, persistence |
| `execution:state:{exchange}` | OrderManager / reconciler | Observability, persistence |
| `execution:errors` | Strategy runner | Operational alerting |
| `private:state:{exchange}:snapshot` | Private reconciler | Ledger, frame assembler, execution reconciler |
| `private:state:{exchange}:balances` | Private reconciler | Ledger |
| `private:state:{exchange}:open_orders` | Private reconciler | Ledger, execution reconciler |
| `private:state:{exchange}:order_updates` | Private reconciler | Ledger, execution reconciler |
| `private:state:{exchange}:fills` | Private reconciler | Ledger, execution reconciler |
| `ledger:snapshot:{exchange}` | Portfolio ledger | Persistence/debugging |
| `ledger:snapshot` | Portfolio ledger | Persistence, dashboards |
| `ledger:diagnostics:consistency:{exchange}` | Portfolio ledger | Ledger/order-manager comparison |

Persistence wraps events in `EventEnvelope` and writes `mm_events`. Latest
private/execution snapshots also upsert into `mm_latest_state`.

## Public And Private State

`SymbolEngine` owns one exchange/symbol public adapter. Trades are held in
`RollingTradeBuffer`, orderbooks in `OrderbookManager`, and normalized market
events are built on scheduler intervals by `MarketEventBuilder`.

Private state is optional. If private credentials resolve, a
`PrivateStateReconciler` publishes snapshots plus balance, open-order,
order-update, fill, and error events. REST adapters use a unified polling loop so
balances and open orders come from one snapshot call.

`PortfolioLedgerService` subscribes to private topics for configured venues and
updates a `VenueLedgerStateMachine` per venue. It publishes:

- `VenueLedger` snapshots on `ledger:snapshot:{exchange}`
- `JointLedgerSnapshot` on `ledger:snapshot`
- order-manager consistency diagnostics when an order manager exists

The ledger is a shadow source of strategy/risk truth. `ExecutionStateReconciler`
still mirrors private state into `OrderManager.live_orders`, so the current
system intentionally keeps both views and compares them. `OrderManager` also
records accepted orders locally after adapter placement.

## Joint Frame Assembly

`StrategyInputFrameAssembler` still performs local/global market alignment:

1. Drain latest global/private queues.
2. Wait up to `global_grace_ms` for a same-interval global event.
3. Pick the latest global event with `t_end_ms <= local.t_end_ms`.
4. Mark global status as `disabled`, `missing`, `stale`, or `ready`.
5. Publish a `StrategyInputFrame` internally.

`JointStrategyFrameAssembler` wraps that input frame into `JointStrategyFrame`:

- chooses a reference price using `liquidity_sourcing.reference_venue`
- asks the ledger service for a `JointLedgerSnapshot`
- adds local/global private and market freshness
- sets mode to `single_venue` or `liquidity_sourcing`
- publishes to `config.joint_frame_topic`

The strategy runner consumes `JointStrategyFrame` and passes the embedded
`input_frame` through the MPC bridge after ledger enrichment.

## Strategy Runner

`StrategyRunner.on_joint_frame()` requires a ledger-backed joint frame and then
processes the embedded input frame.

Per frame:

1. Increment `frame_step`.
2. Sync control-plane kill switch into risk.
3. Collect live orders from all order managers.
4. Subtract own orders from local/global books with `clean_strategy_frame`.
5. Build `TradingState` context for diagnostics.
6. Run `MpcStrategyEngine.decide`.
7. Prepare an execution target book with per-venue caps and global live-order filtering.
8. Compute intent execution gates for risk context.
9. Run `NetDeltaRiskEngine` against the joint target book and ledger adapter.
10. Publish risk events if actions were filtered.
11. Compute final venue execution gates.
12. Publish `StrategyDecision`, `JointTargetBook`, and approved `Action`s.
13. Execute or preview each venue target book through `route_joint_target_book_with_report`.
14. Publish `strategy:journal:{actor_id}`.
15. Publish `ControllerDiagnostics` in `finally`, including error frames.

If MPC raises, the runner publishes a `ControllerError`, records diagnostics, and
returns no actions.

## MPC Strategy Engine

`MpcStrategyEngine` is the bridge between runtime frames and MPC internals.

It builds `mm.mpc.mpc_state.State` with:

| Field | Source |
| --- | --- |
| `mid`, `ob_rel` | Cleaned local book |
| `buy_vol`, `sell_vol` | Local realized market flow |
| `inventory`, `cash` | Local private balances |
| `buy_vwap`, `sell_vwap` | Local reference VWAPs |
| `global_ob_rel` | Cleaned global book, if present |
| `global_buy_vol`, `global_sell_vol` | Global realized market flow |
| `global_inventory`, `global_cash` | Global private balances |
| `global_buy_vwap`, `global_sell_vwap` | Global reference VWAPs |
| `reference_price`, `global_active` | Liquidity-sourcing reference context |

When `liquidity_sourcing.enabled` is true, state is enriched for joint MPC and
`MonteCarloMPCTensorBatched` dispatches to `mm.mpc.joint_optimizer`. The engine
then discretizes local and optional global sigmoid actions into a
`JointTargetBook`. The `Action` is derived from that book for the execution
reconcile path.

Model scoring remains prequential: observations are scored before the market
flow, local residual price, and global price/flow models are updated.

## Joint Target Books

The strategy output contract is in `mm.core.models.target_book`:

| Model | Meaning |
| --- | --- |
| `TargetOrder` | Desired resting order at one venue |
| `VenueTargetBook` | Desired resting book for one actor/exchange/symbol |
| `JointTargetBook` | Local book plus optional global book |

`build_joint_target_book_from_sigmoid()`:

1. Converts the local sigmoid curve to ladder orders.
2. Applies local instrument precision/minimums/order-count caps.
3. If liquidity sourcing is active and a global sigmoid exists, repeats the
   process for the global venue.
4. Updates `ActionPipelineReport.target_book`.
5. Returns the joint book plus the local pipeline report.

## Risk

`NetDeltaRiskEngine` is the implemented hard risk engine. It evaluates target
book transitions against `JointLedgerSnapshot` rather than only checking
individual proposed orders.

Frame-level gates include:

| Gate | Reason |
| --- | --- |
| Control-plane kill switch | `kill_switch` |
| Stale local market | `stale_market_data` |
| Missing/stale local private ledger | `missing_local_private`, `stale_local_private` |
| Missing/stale global market when global live orders are active | `missing_global_market_data`, `stale_global_market_data` |
| Missing/stale global private ledger when required | `missing_global_private`, `stale_global_private` |

Per-venue target transition checks include:

| Check | Local reason | Global reason prefix |
| --- | --- | --- |
| Quantity positive | `invalid_qty` | `global_invalid_qty` |
| Per-order notional | `max_order_notional` | `global_max_order_notional` |
| Total open notional | `max_open_notional` | `global_max_open_notional` |
| Sell collateral | `insufficient_base_for_ask_orders` | `global_insufficient_base_for_ask_orders` |
| Buy collateral | `insufficient_quote_for_bid_orders` | `global_insufficient_quote_for_bid_orders` |
| Minimum cash | `min_cash` | `global_min_cash` |
| Inventory min/max | `min_inventory`, `max_inventory` | prefixed globally |

If local is valid but global orders fail, the risk engine may approve a local-only
book and strip the global orders. If local fails, the whole action is rejected.

## Execution

`prepare_execution_joint_book()` applies live execution filters:

- removes global orders unless `liquidity_sourcing.global_live_orders` is true
- applies local/global planner policies
- drops venue orders exceeding `max_order_qty` or `max_quote_distance`

`compute_venue_execution_gates()` computes independent gates per venue. Blocked
reasons include:

| Reason | Meaning |
| --- | --- |
| `observe_only_config` | Controller observe-only |
| `observe_only_control` | Runtime control-plane observe-only |
| `warmup` | `frame_step <= action_start_after_steps` |
| `paused_actor` | Actor paused by control plane |
| `venue_observe_only` | Venue-specific observe-only |
| `venue_paused` | Venue role paused by control plane |
| `no_manager` | No order manager for exchange |
| `global_live_orders_disabled` | Global live switch is false |
| `stale_global_market_data` | Global book is missing or not ready |
| `no_orders` | Venue book is empty |

`route_joint_target_book_with_report()` calls
`sync_venue_target_book()` when executing and `preview_venue_target_book()` when
live execution is blocked. The returned `ExecutionSyncReport` is folded into
controller diagnostics.

`OrderManager` reconciles desired target books against live open orders using
`ReconcileConfig.price_tol_bps` and `qty_tol_rel`. The supported planner policies
are `reconcile` and `replace`.

## Diagnostics And Observability

`ControllerDiagnostics` is emitted once per processed frame. It includes:

| Field | Meaning |
| --- | --- |
| `frame` | Global status, lag, frame age, private-state ages |
| `state` | Local/global balances, VWAPs, volumes |
| `models` | Pre-update scores and before/after params |
| `optimizer` | Objective/action/timing/rollout diagnostics |
| `action_pipeline` | Stage-by-stage order funnel |
| `risk` | Approval, rejections, exposure, limits |
| `execution_gates` | Per-venue live/blocked gate detail |
| `warnings` | Controller and execution errors |

The liquidity-sourcing dashboard reads joint target books, diagnostics, and
ledger snapshots. Existing dashboards also cover market data, orderbook depth,
private state, controller diagnostics, model params, portfolio, runtime metrics,
and execution state.

## Current Caveats

- `StrategyRunner` consumes `JointStrategyFrame`; MPC still receives the cleaned
  embedded `StrategyInputFrame`.
- The ledger is shadow state and is compared with `OrderManager.live_orders`;
  order managers still keep local live-order state after adapter acknowledgements.
- Global target-book diagnostics are produced by joint MPC, but global live
  placement is disabled unless `liquidity_sourcing.global_live_orders=true`.
- Global execution defaults to the `replace` planner, while local defaults to
  `reconcile`.
