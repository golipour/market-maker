# Low-Frequency Market Maker Architecture

This document describes the implemented architecture of the low-frequency market maker after the diagnostics, MPC, risk, execution, and hedging upgrades. It is written for two audiences:

- software engineers who need to reason about boundaries, runtime behavior, observability, and failure modes
- market-making reviewers who need to inspect the controller assumptions, quote construction, risk gates, and hedge behavior

The codebase is a single-process async runtime with an event-bus spine. Public market data is normalized into interval market events, primary and hedge data are aligned into strategy frames, the MPC controller produces primary limit-order ladders, risk filters them, and the execution layer optionally syncs them to venue adapters. Rich controller diagnostics are emitted on a separate event contract so a bad quote can be reconstructed from one frame-level payload.

## Current Implementation Summary

The implemented system has the following production path:

```text
exchange public data
  -> SymbolEngine
  -> MarketEventBuilder on IntervalScheduler
  -> market:events:{exchange}:{symbol}:{interval_ms}
  -> StrategyInputFrameAssembler
  -> strategy:frames:{primary_exchange}:{symbol}:{interval_ms}
  -> StrategyRunner
  -> clean_strategy_frame
  -> MpcStrategyEngine.decide
  -> RuntimeRiskEngine.validate_frame_with_result
  -> StrategyDecision + Action + ControllerDiagnostics
  -> optional route_actions_to_managers_with_report
  -> OrderManager.sync_with_report
  -> exchange order adapter
```

The controller still emits `StrategyDecision` for backward compatibility, but the richer review surface is `ControllerDiagnostics` on `strategy:diagnostics:{actor_id}`.

Major implemented upgrades:

- `ControllerDiagnostics` is a typed event emitted once per processed frame, including error frames.
- Model observations are scored before online model updates.
- The optimizer exposes initial, final, and best objective/action diagnostics.
- The order path reports stage-by-stage counts and rejection reasons.
- Hard risk configuration is separated from optimizer soft preferences.
- Fill-driven hedge execution exists as an optional service, separate from MPC hedge rollout simulation.

Important live-trading caveat:

- MPC hedge rollouts and live hedge execution are separate. `optimizer.hedge_rollout_enabled` changes the simulated objective. `hedging.enabled` starts a fill-driven hedge executor. Enabling one does not imply the other.

## Runtime Composition

Main wiring is in `mm.runtime.engine.build_runtime_components`.

`RuntimeComponents` owns:

| Component | Module | Responsibility |
| --- | --- | --- |
| Event bus | `mm.bus.factory` | In-memory bus or Redis Streams fan-out |
| Scheduler | `mm.aggregation.scheduler.IntervalScheduler` | Periodic interval callbacks |
| Symbol engines | `mm.ingestion.symbol_engine.SymbolEngine` | Public trade/book ingestion per source |
| Frame assembler | `mm.application.frame_assembler.StrategyInputFrameAssembler` | Primary/hedge/private alignment |
| Control plane | `mm.runtime.control.RuntimeControlPlane` | Admin kill switch and observe-only state |
| Strategy runner | `mm.application.strategy_runner.StrategyRunner` | Main controller pipeline |
| Hedge service | `mm.hedging.service.HedgeExecutionService` | Optional fill-driven hedge orders |
| Private reconcilers | `mm.private_state.reconciler.PrivateStateReconciler` | Wallet/open-order/fill state |
| Execution reconcilers | `mm.execution.reconciler.ExecutionStateReconciler` | Mirror private state into `OrderManager` |
| Order managers | `mm.execution.order_manager.OrderManager` | Target-order reconciliation, validation, cancel/place commands |
| Persistence daemons | `mm.persistence.postgres`, `mm.persistence.parquet_writer` | Durable event envelopes |
| Observability daemons | `mm.observability.subscribers` | Metrics, logs, traces |

Startup order:

1. Persistence and observability subscribers start first.
2. The frame assembler subscribes to market/private queues.
3. The strategy runner and hedge service start if enabled.
4. The control plane starts.
5. Public symbol engines connect and start feed tasks.
6. Private and execution reconcilers start.
7. The interval scheduler starts publishing market events.

Shutdown reverses the operational services and closes the bus.

## Configuration Model

Runtime config is loaded from JSON by `mm.runtime.config.runtime_config_from_mapping`.

Top-level structure:

| Key | Meaning |
| --- | --- |
| `primary` | Required public source and instrument for quote venue |
| `hedge` | Optional secondary source used for reference data and live hedge venue |
| `interval_ms` | Strategy decision interval |
| `max_hedge_lag_ms` | Maximum aligned hedge event lag |
| `hedge_grace_ms` | Short wait for matching hedge event |
| `primary_private`, `hedge_private` | Private state and execution credentials |
| `controller` | MPC controller, hard risk, and live quote settings |
| `hedging` | Optional fill-driven hedge executor |
| `bus` | Memory or Redis Streams |
| `persistence` | Postgres and/or Parquet event writers |
| `observability` | Metrics, JSON logs, run id, OTEL settings |

The strategy uses a `SymbolBinding` built from primary and optional hedge `InstrumentSpec`s. Venue precision and limits are loaded from exchange trade stats and can be overridden per source.

Units:

| Quantity | Unit |
| --- | --- |
| Prices, mid, VWAP | quote per base |
| Spread grid | absolute quote-per-base distance from mid |
| Quantity/depth | base units |
| Cash/notional | quote units |
| Inventory | base units |

## Event Bus And Topics

The bus is the runtime contract. Producers do not call persistence, Grafana, or admin code directly.

Core topics:

| Topic | Producer | Main consumers |
| --- | --- | --- |
| `market:raw:{ex}:{symbol}:trades` | `SymbolEngine` | Persistence/debugging |
| `market:raw:{ex}:{symbol}:orderbook` | `SymbolEngine` | Persistence/debugging |
| `market:errors:{ex}:{symbol}` | `SymbolEngine` / adapters | Observability |
| `market:events:{ex}:{symbol}:{interval_ms}` | `MarketEventBuilder` | Frame assembler, persistence |
| `strategy:frames:{primary_ex}:{symbol}:{interval_ms}` | Frame assembler | Strategy runner, observability |
| `strategy:decisions` | Strategy runner | Persistence, dashboards |
| `strategy:actions` | Strategy runner | Persistence |
| `strategy:diagnostics:{actor_id}` | Strategy runner | Persistence, Grafana |
| `risk:events` | Strategy runner | Audit/debugging |
| `execution:orders:{exchange}` | OrderManager | Observability, persistence |
| `execution:state:{exchange}` | OrderManager / reconciler | Observability, persistence |
| `private:state:{exchange}:snapshot` | PrivateStateReconciler | Frame assembler, execution reconciler, hedge service |
| `private:state:{exchange}:fills` | PrivateStateReconciler | Execution reconciler, hedge service |
| `hedge:intents:{actor_id}` | HedgeExecutionService | Persistence |
| `hedge:diagnostics:{actor_id}` | HedgeExecutionService | Persistence, dashboards |

Persistence wraps each event in an `EventEnvelope` (`mm.core.events`) and writes `mm_events`. Latest private/execution snapshots also upsert into `mm_latest_state`.

## Public Market Data Ingestion

`SymbolEngine` owns one exchange/symbol public adapter.

It starts three async loops when supported:

- trade stream
- orderbook stream
- adapter error stream

Trades are stored in `RollingTradeBuffer`. Orderbooks are stored in `OrderbookManager`, which handles snapshots and deltas. Every normalized trade/book is also published on raw market topics.

On each scheduler interval, `MarketEventBuilder.build()` reads:

- orderbook snapshot at `t_end_ms`
- trades over `[t_start_ms, t_end_ms]`
- buy and sell volume
- trade VWAP
- top-of-book imbalance
- microprice

The resulting `MarketEvent` is published to `market:events:{exchange}:{symbol}:{interval_ms}`.

## Private State And Execution State

Private state is optional and depends on API token resolution. If configured, `PrivateStateReconciler` connects a private adapter and publishes:

- full snapshots
- balance events
- open-order events
- order-update events
- fill events
- private-state errors

For REST polling adapters, the reconciler uses a unified polling loop so balances and open orders come from one snapshot call. The store then applies fills/order updates and publishes updated snapshots.

`ExecutionStateReconciler` listens to private snapshots, open orders, order updates, and fills and applies them to the `OrderManager`. This gives the order manager an exchange-truth live-order view used for:

- own-order subtraction before MPC
- target-order reconciliation before placement
- execution state telemetry
- cancel-all admin commands

## Strategy Frame Assembly

`StrategyInputFrameAssembler` consumes:

- primary market events
- optional hedge market events
- optional primary private snapshots
- optional hedge private snapshots

For every primary market event:

1. Drain latest hedge and private queues.
2. Wait up to `hedge_grace_ms` for a hedge event with the same `t_end_ms`.
3. Drain again.
4. Select the latest hedge event whose `t_end_ms <= primary.t_end_ms`.
5. Mark hedge status:
   - `disabled`: no hedge queue configured
   - `missing`: no eligible hedge event
   - `stale`: newest eligible hedge event exceeds `max_hedge_lag_ms`
   - `ready`: hedge event selected and within lag limit
6. Publish `StrategyInputFrame`.

The frame still contains the exchange-visible orderbook, including the bot's own resting orders. The strategy runner cleans it before calling MPC.

## Main Controller Pipeline

`StrategyRunner.on_frame()` is the single production path for controller decisions.

Per frame:

1. Increment `frame_step`.
2. Sync `RuntimeControlPlane.kill_switch` into `RuntimeRiskEngine`.
3. Collect live orders from all order managers.
4. Build a cleaned frame with `clean_strategy_frame`.
5. Build `TradingState` for risk.
6. Create `DecisionContext` for diagnostics.
7. Run `MpcStrategyEngine.decide(clean_frame, frame_step=...)`.
8. Risk-filter the proposed action using `validate_frame_with_result`.
9. Compute live execution gates and blocked reasons.
10. Publish `StrategyDecision`.
11. Publish approved `Action`s.
12. If live execution is allowed, route approved actions to order managers.
13. If live execution is blocked but an order manager exists, run placement preview for diagnostics.
14. Publish `ControllerDiagnostics`.
15. Publish a compact `strategy:journal:{actor_id}` event.

If MPC raises an exception, the runner publishes a `ControllerError`, records the error in diagnostics warnings, and returns no actions.

### Live Execution Gates

Orders reach `OrderManager.sync_with_report()` only if all gates pass:

| Gate | Blocked reason |
| --- | --- |
| Approved action list is non-empty | `no_approved_actions` |
| `controller.observe_only` is false | `observe_only_config` |
| control-plane observe-only is false | `observe_only_control` |
| `frame_step > action_start_after_steps` | `warmup` |
| Primary exchange order manager exists | `no_manager` |

The `RuntimeControlPlane` also tracks paused actors, but the current strategy runner does not check paused actors.

## Own-Order Subtraction

Before MPC, `clean_strategy_frame(frame, open_orders)` subtracts the bot's live open orders from the primary and hedge orderbooks.

This matters because the optimizer should see market liquidity, not liquidity created by its own previous quotes. The live orders come from `OrderManager.live_orders`, which is updated by private state reconciliation and optimistic placement records.

The subtraction is applied only to the strategy's input frame. Raw market events remain unchanged for auditability.

## MPC Strategy Engine

`MpcStrategyEngine` is a synchronous bridge between runtime frames and legacy/batched MPC internals.

It owns:

- market-flow model
- residual primary price model
- optional hedge price model
- `MonteCarloMPCTensorBatched` optimizer
- previous frame pointer for prequential scoring

Per decision:

1. Convert clean frame into MPC `State`.
2. Score current observations using model parameters from before seeing the frame.
3. Update hedge VWAP model.
4. Update market-flow model.
5. Update residual mid-price model.
6. Run the tensor MPC optimizer.
7. Convert continuous quote-depth curves into marginal ladder orders.
8. Apply instrument precision and minimums.
9. Cap orders per side.
10. Build `StrategyDecision`.
11. Store the current frame as `_previous_frame`.

### State Construction

`_state_from_frame()` computes:

| State field | Source |
| --- | --- |
| `mid` | top-of-book midpoint from cleaned primary book |
| `ob_rel.bids/asks` | cleaned book levels converted to spread from mid and clipped to `spread_max` |
| `buy_vol`, `sell_vol` | primary interval trade volumes |
| `inventory`, `cash` | primary private balances, base and quote assets |
| `buy_vwap`, `sell_vwap` | primary orderbook reference VWAPs at `vwap_depth` |
| `hedge_buy_vwap`, `hedge_sell_vwap` | hedge orderbook reference VWAPs when hedge is available |
| `hedge_inventory`, `hedge_cash` | hedge private balances when configured |

If private state is missing, inventory and cash default to zero. This is operationally important because hard risk gates can reject missing or stale private state when configured.

### Model Scoring

`mm.models.scoring.score_frame_models()` computes pre-update scores:

| Model | Observation | Score |
| --- | --- | --- |
| Market flow | realized primary buy/sell volume | negative log likelihood under hurdle/lognormal, hurdle/gamma, hurdle/weibull, gamma, or weibull |
| Primary price | current mid minus previous mid | Gaussian NLL and standardized residual |
| Hedge price | current hedge VWAP deltas | Gaussian NLL and standardized residual |

Skip reasons include:

- `missing_previous_frame`
- `missing_hedge`
- `invalid_sigma`
- `unsupported_model`

After model updates, `refresh_score_params_after()` writes post-update parameters into the same score record. This makes diagnostics explicitly prequential: the score measures prediction quality before the model learned the event.

### Model Updates

The implemented update order is:

1. Hedge model update from current hedge buy/sell VWAPs.
2. Market-flow model update from current buy/sell volumes.
3. Residual primary price model update from previous and current primary mid.

The residual price update is an EWMA update on primary mid changes, not an orderbook VWAP update.

## Tensor-Batched MPC Optimizer

`MonteCarloMPCTensorBatched` chooses continuous cumulative bid and ask depth curves over the spread grid.

Public methods:

- `optimize(state) -> SigmoidMixtureAction`
- `optimize_with_diagnostics(state) -> (SigmoidMixtureAction, OptimizerRunDiagnostics)`
- `last_diagnostics`

### Action Parameterization

Each side is a mixture of sigmoids over spread:

```text
D_side(spread) = sum_i L_i * sigmoid(k_i * (spread - x0_i))
```

The executable depth is baseline-shifted:

```text
effective_depth(spread) = raw_depth(spread) - raw_depth(0)
```

Every diagnostic and ladder conversion uses effective depth. This prevents non-zero raw sigmoid depth at spread zero from becoming phantom quote size.

### Optimizer Inputs

The optimizer receives:

- current state mid, inventory, and cash
- cleaned relative market orderbook
- primary buy/sell VWAPs
- optional hedge buy/sell VWAPs and hedge balances
- online market-flow, price, and hedge-price models
- execution model fees
- optimizer config and reward weights

### Monte Carlo Rollout

For each optimizer call, the optimizer simulates `n_samples` paths for `horizon` steps.

It samples:

- future buy market volume
- future sell market volume
- primary price residual shocks
- hedge buy VWAP deltas
- hedge sell VWAP deltas

If `fixed_mc_samples` is true, samples are drawn once before the Adam loop and reused throughout the call. This makes the objective deterministic within one optimization run and improves diagnostics comparability.

### Soft Fill Model

`soft_fill_batched()` simulates how market order flow consumes the combined book:

1. Convert cumulative quote depth into marginal quote levels.
2. Convert market book cumulative depth into marginal levels.
3. Combine market and own quote liquidity.
4. Use a sigmoid reach function to soft-consume depth.
5. Allocate consumed volume between market liquidity and our quote liquidity proportionally.
6. Return next market-only book, cash delta, and inventory delta.

This differentiable fill model is an approximation. It is useful for gradient optimization, but it is not a venue queue-position model.

### Objective

The optimizer maximizes terminal objective:

```text
J =
  expected_pnl_weight * reward_term
  - inventory_target_penalty_weight * inventory_target_term
  - pnl_volatility_penalty_weight * volatility_term
  - pnl_cvar_penalty_weight * soft_cvar_term
  - quote_shape_concentration_penalty_weight * concentration_term
  - quote_size_target_penalty_weight * size_term
  - quote_spread_penalty_weight * quoted_spread_term
```

Primary wealth is marked using sell-side VWAP:

```text
wealth = cash + inventory * sell_vwap
```

If hedge rollout is enabled, simulated primary fills are assumed to be immediately offset on the hedge venue. This is a model assumption inside the objective; live execution requires the separate hedge service.

When `objective_normalize_components=true`, reward, volatility, and CVaR are scaled by current absolute wealth, inventory target miss is scaled by an inventory scale, quote-size and quoted-spread penalties are relative terms, and concentration is normalized into `[0, 1]`. This makes objective weights more portable across quote currencies.

The quote-shape term is now concentration-based rather than L2-to-uniformity-based. It uses a Herfindahl-style share concentration over marginal quote depth, where lower values mean depth is more evenly distributed across levels.

The quoted-spread term computes bid/ask depth VWAPs from the controller curve at `quote_depth_target_for_spread`, converts the quoted width to bps, and penalizes excess over `quote_spread_target_bps`. It is a controller-curve spread, not the market BBO.

### Optimizer Diagnostics

`OptimizerRunDiagnostics` includes:

| Field | Meaning |
| --- | --- |
| `config` | kind, sample count, horizon, Adam epochs, grid |
| `timing` | sample draw, train, total seconds, best epoch |
| `objective.initial/final/best` | objective value, loss, weighted components, reward means |
| `objective.improvement` | best/final minus initial |
| `objective.contribution_fractions` | final-stage absolute component contribution shares |
| `action.initial/final/best` | effective depth curves and raw flat params |
| `gradients` | nonfinite loss/grad counts |
| `rollout` | sample count, horizon, hedge enabled, redirected log |

The strategy embeds this payload in both `StrategyDecision.optimizer` and `ControllerDiagnostics.optimizer`.

## Quote Discretization And Action Pipeline

The controller does not send the continuous optimizer action directly to the venue.

`build_orders_from_sigmoid_with_report()` performs:

1. `ladder_orders_from_sigmoid_with_report`
   - evaluate effective cumulative bid/ask depth at every spread-grid level
   - convert cumulative depth to marginal order quantities
   - create limit orders at `mid - spread` and `mid + spread`
2. `apply_venue_constraints_with_report`
   - quantize price and quantity by instrument tick/lot settings
   - drop invalid, below-min-qty, and below-min-notional orders
3. `cap_orders_per_side_with_report`
   - keep largest-quantity orders per side up to `max_orders_per_side`

The resulting `ActionPipelineReport` contains:

- continuous depth contract
- ladder totals
- venue-quantized totals
- capped totals
- later risk-approved totals
- later execution target totals
- later kept-live totals
- later canceled-live totals
- later placed/venue-limit-valid totals
- later venue-accepted totals

Every stage records order count, buy/sell quantity, buy/sell notional, and rejection counts.

## Hard Risk Engine

`RuntimeRiskEngine` applies hard gates after strategy proposal and before live execution.

It receives:

- proposed actions
- `TradingState` with inventory, cash, and open orders
- the cleaned strategy frame

Frame-level hard gates:

| Gate | Rejection reason |
| --- | --- |
| Control-plane kill switch | `kill_switch` |
| Primary market data too old | `stale_market_data` |
| Missing/stale primary private state | `missing_primary_private`, `stale_primary_private` |
| Hedge market data too old when hedge is present | `stale_hedge_market_data` |
| Missing/stale hedge private state when hedge is present | `missing_hedge_private`, `stale_hedge_private` |

Order-level hard gates:

| Gate | Rejection reason |
| --- | --- |
| non-positive quantity | `invalid_qty` |
| per-order notional too large | `max_order_notional` |
| open notional would exceed limit | `max_open_notional` |
| buy would breach min cash | `min_cash` |
| buy would exceed max inventory | `max_inventory` |
| sell quantity exceeds inventory | `insufficient_inventory` |
| sell would breach min inventory | `min_inventory` |
| unknown side | `invalid_side` |

`RiskResult` reports:

- approved actions
- rejection counts
- exposure before
- exposure after

If `controller.risk.enabled` is false, `PassThroughRiskEngine` approves all actions and only reports exposure snapshots.

## Execution Layer

`route_actions_to_managers_with_report()` groups declarative target orders by exchange and calls each exchange `OrderManager`.

`OrderManager.sync_with_report()`:

1. Publishes `ExecutionSyncRequested`.
2. Scopes each `Action` by exchange and symbol.
3. Reconciles the action's target orders against live open orders for the same actor.
4. Keeps live orders within price/quantity tolerance.
5. Cancels unmatched live orders in scope.
6. Places unmatched target orders after venue limit validation.
7. Publishes `OrderValidationRejected` events for invalid orders.
8. Sends valid new orders to the exchange command adapter.
9. Records accepted orders in local `live_orders`.
10. Publishes cancel/place and `execution:state:{exchange}` events as applicable.
11. Returns `ExecutionSyncReport`.

In observe-only or warmup mode, the runner may call `preview_placement()` to dry-run the same reconcile plan and fill target/kept/canceled/placed diagnostics without exchange cancel/place API calls.

The old `cancel_missing`/explicit cancel action model has been replaced by declarative target-state reconciliation. An empty target for an actor/symbol scope cancels that actor's live orders in scope; otherwise unmatched live orders are canceled and unmatched targets are placed. Matching tolerance is configured by `ReconcileConfig.price_tol_bps` and `ReconcileConfig.qty_tol_rel`.

## Controller Diagnostics Contract

`ControllerDiagnostics` is emitted once per processed frame to `strategy:diagnostics:{actor_id}`.

Top-level fields:

| Field | Meaning |
| --- | --- |
| `type`, `version` | Stable event identity |
| `actor_id`, `exchange`, `symbol` | Controller identity and primary market |
| `t_end_ms`, `frame_step`, `created_at_ms`, `run_id` | Timing and run identity |
| `frame` | Hedge status, lag, frame age, private-state ages |
| `state` | Mid, primary/hedge inventory/cash, VWAPs |
| `models` | Pre-update scores and before/after params |
| `optimizer` | Objective/action/timing/rollout diagnostics |
| `action_pipeline` | Stage-by-stage order funnel |
| `risk` | Approval, rejections, exposure, limits |
| `execution_gates` | Executed flag and blocked reasons |
| `warnings` | Controller errors and other warnings |

This is the best single event to inspect when asking:

- Why did the controller quote this size and spread?
- Did the optimizer improve the objective?
- Which objective component dominated?
- Did model likelihood deteriorate?
- Where did orders disappear between MPC and venue?
- Was execution blocked by config, warmup, control plane, manager absence, or risk?

Known diagnostics limitation:

- `DecisionContext._state_diagnostics()` currently sets hedge inventory/cash to zero when a decision exists, even though `_state_from_frame()` computes hedge balances. If hedge portfolio diagnostics matter, this should be corrected so diagnostics expose actual hedge balances.

## Observability And Persistence

`RuntimeObservabilitySubscriber` watches frames, decisions, actions, execution orders, execution state, and errors. It emits Prometheus metrics and JSON logs.

Important metrics include:

- strategy frame count by hedge status
- controller latency
- controller order count
- inventory and cash
- strategy action order count
- execution sync counts
- order validation rejection counts
- order sync latency
- frame-to-execution latency
- live order count
- runtime errors

Persistence supports:

- Postgres `mm_events`
- Postgres `mm_latest_state`
- optional Parquet writer

Default persistence topics include market events, frames, actions, decisions, diagnostics, execution events/state, private snapshots, and hedge topics when configured.

Grafana dashboards already exist for market data, orderbook depth, private state, controller diagnostics, model params, portfolio, runtime metrics, and hedge execution.

## Fill-Driven Hedge Execution

`HedgeExecutionService` is optional and starts only when:

- `hedging.enabled` is true
- a top-level hedge source is configured

It subscribes to:

- `private:state:{primary_exchange}:fills`
- `private:state:{hedge_exchange}:snapshot`

For each primary fill:

1. Ignore fills for other symbols/exchanges or non-positive quantities.
2. Dedupe by `exchange_order_id:timestamp_ms:qty:side`.
3. Convert primary fill side to hedge side:
   - primary buy -> hedge sell
   - primary sell -> hedge buy
4. Publish `HedgeIntent`.
5. Increase residual unhedged quantity by signed primary fill quantity.
6. Apply observe-only/control-plane/manager/config checks.
7. Build a market `Order` on the hedge venue.
8. Validate hedge hard risk.
9. Route the hedge order through the hedge `OrderManager`.
10. Publish `HedgeExecutionDiagnostic`.
11. If submitted, reduce residual unhedged quantity by the hedge order side/quantity.

Hedge hard-risk checks currently include:

- exchange supports market orders
- positive quantity
- max market-order notional
- sufficient hedge inventory for sells
- min/max hedge inventory
- min hedge cash for buys

Current hedge limitations for reviewers:

- It uses the primary fill price as the reference price for the market order notional check.
- `max_slippage_bps` exists in config but is not enforced in `validate_hedge_market_order`.
- It does not yet read a live hedge top-of-book/VWAP at execution time for slippage protection.
- Market-order adapter behavior depends on venue implementation.
- Partial hedge fills and retry policy are not modeled beyond order-manager acceptance.

## Exchange Adapter Boundaries

Exchange-specific code lives under `mm.exchanges`.

Public adapters must provide normalized:

- trade stream
- orderbook stream
- optional error stream

Private adapters must provide:

- snapshot
- balances
- open orders
- order updates
- fills
- optional errors

Order command adapters receive domain `Order`s and own all wire formatting, including market order handling. Capabilities are declared in `mm.exchanges.capabilities`; both Nobitex and Wallex are currently marked as supporting REST market orders.

Venue min base quantity and min notional checks are applied before placement through `order_passes_limits`.

## Failure Modes And Review Questions

The following are important areas for an engineer or market-making expert to review.

### Market Data And Frame Quality

- Are `interval_ms`, `depth_limit`, and `trades_limit` appropriate for low-frequency quoting on the target venues?
- Is the hedge alignment policy too permissive or too strict for the chosen pair?
- Should stale/missing hedge data block all primary quotes, or should quotes continue with wider spreads/reduced size?
- Are orderbook snapshots sufficiently deep for `vwap_depth` and `spread_max`?

### MPC And Modeling

- Does the soft-fill model approximate actual queue priority and adverse selection well enough?
- Should the market-flow model use integer rounded volume, or should it model continuous volume directly?
- Is marking inventory to sell-side VWAP appropriate for both long and short inventory?
- Are the reward weights dimensionally stable across symbols and quote currencies?
- Should inventory target and hard inventory limits be tied together or kept intentionally separate?
- Should price model scoring and update use mid, reference VWAP, or microprice?
- Should hedge rollout be enabled only when live hedge execution is enabled and healthy?

### Quote Construction

- Does keeping largest-quantity orders after capping preserve the intended spread shape?
- Should orders that would cross the top of book or sit too close/far from reference be blocked?
- Are reconcile tolerances (`price_tol_bps`, `qty_tol_rel`) strict enough to replace stale quotes without excessive churn?
- Should empty target actions be allowed to cancel all actor/symbol live orders, or should cancellation require an explicit safety gate?
- Are client order ids stable enough for idempotency, and do they fit venue length constraints?

### Risk

- Current hard risk covers inventory, cash, open notional, order notional, stale data, and private state freshness. It does not yet cover daily loss, drawdown, price bands, self-trade prevention, venue health, API rate limits, or max order age.
- The primary `min_inventory` parser clamps negative values to zero. That prevents intentional short inventory limits in current config parsing.
- Missing private state can be configured as a hard rejection. This is safer for live trading but can block observe-only/backtest-like runs unless configured carefully.
- Hedge slippage config exists but is not enforced.

### Execution

- Declarative target reconciliation should prevent stale quote accumulation, but aggressive tolerances can create cancel/replace churn and loose tolerances can leave stale quotes live.
- Venue adapter failures are surfaced as exceptions/events, but retry/backoff policy is limited.
- `preview_placement()` improves diagnostics under observe-only, but it cannot reveal adapter-side rejects.
- OrderManager records accepted REST order ids optimistically; private reconciliation should be treated as exchange truth.

### Diagnostics And Operations

- `ControllerDiagnostics` is the main review artifact; dashboards should prefer it over older compact `StrategyDecision` fields.
- Model NLL panels should be watched for regime shifts and broken data normalization.
- Optimizer `best_minus_initial` near zero over many frames may indicate poor initialization, insufficient epochs, bad reward scaling, or a flat objective.
- Nonfinite loss/grad counts should alert.
- Frame age and market-window-to-execution latency should be monitored against the low-frequency strategy assumptions.

## Suggested Next Improvements

High-priority engineering improvements:

1. Fix hedge inventory/cash fields in `ControllerDiagnostics.state`.
2. Tune reconcile tolerances and add guardrails for empty-target mass cancellations.
3. Enforce hedge `max_slippage_bps` using fresh hedge book/VWAP before market hedge submission.
4. Add price-band and no-crossing checks before venue submission.
5. Add max open-order age and stale-quote cancellation.
6. Add daily loss, step loss, and drawdown gates.
7. Add venue health/rate-limit gates.
8. Make paused actors in `RuntimeControlPlane` affect `StrategyRunner`.
9. Add stronger adapter-level market order capability contracts.
10. Persist and dashboard hedge residual inventory from `HedgeExecutionService`.

High-priority market-making review topics:

1. Validate whether the objective's hedge assumption matches the intended live hedge policy.
2. Calibrate reward and penalty weights in quote-currency units.
3. Compare simulated fill probability against realized fill data.
4. Review whether inventory is marked to the correct side/reference.
5. Evaluate whether the quote-size target should be per side, per level, or inventory-dependent.
6. Add inventory-skewed quoting if desired behavior is not fully captured by the MPC objective.
7. Add adverse-selection features such as recent imbalance, spread, microprice drift, and cross-venue basis.
8. Backtest controller diagnostics over replayed days and inspect NLL/objective/order-funnel distributions.

## File Map

| Concern | Main files |
| --- | --- |
| Runtime wiring | `mm/runtime/engine.py`, `mm/runtime/config.py`, `mm/runtime/pipeline.py` |
| Event topics/envelopes | `mm/core/topics.py`, `mm/core/events.py` |
| Public ingestion | `mm/ingestion/symbol_engine.py`, `mm/aggregation/builder.py` |
| Frame alignment | `mm/application/frame_assembler.py` |
| Own-order cleaning | `mm/application/frame_preparation.py`, `mm/application/orderbook_view.py` |
| Strategy orchestration | `mm/application/strategy_runner.py`, `mm/application/decision_service.py` |
| MPC bridge | `mm/strategies/mpc_engine.py` |
| MPC optimizer | `mm/mpc/mpc_tensor_batched.py`, `mm/mpc/optimizer_diagnostics.py` |
| Model scoring | `mm/models/scoring.py` |
| Quote constraints | `mm/core/order_constraints.py`, `mm/core/models/instrument.py` |
| Hard risk | `mm/risk/engine.py`, `mm/risk/factory.py`, `mm/core/risk_limits.py` |
| Action/execution reports | `mm/execution/plan.py`, `mm/application/action_routing.py` |
| Order management | `mm/execution/order_manager.py`, `mm/execution/reconciler.py` |
| Private state | `mm/private_state/reconciler.py`, `mm/private_state/store.py` |
| Hedge execution | `mm/hedging/service.py`, `mm/hedging/models.py`, `mm/hedging/risk.py` |
| Diagnostics | `mm/diagnostics/models.py`, `mm/diagnostics/publisher.py`, `mm/controller/context.py` |
| Persistence | `mm/persistence/postgres.py`, `mm/persistence/parquet_writer.py` |
| Observability | `mm/observability/subscribers.py`, `mm/observability/schema.py` |
| Exchange adapters | `mm/exchanges/*`, `mm/exchanges/capabilities.py` |

## Reviewer Checklist

Use this checklist when asking for outside review.

- Confirm exchange payload normalization and units for primary and hedge venues.
- Confirm instrument precision, min quantity, and min notional for the target symbol.
- Inspect `ControllerDiagnostics` for several live/replay frames.
- Check model NLL and standardized residual distributions.
- Check optimizer objective components and improvement.
- Check action-pipeline funnel counts from continuous depth to accepted venue orders.
- Check risk rejections and execution blocked reasons.
- Check live-order accumulation and quote cancellation behavior.
- Check hedge residual inventory after primary fills.
- Confirm observe-only, warmup, kill switch, and admin cancel-all behavior before live trading.
