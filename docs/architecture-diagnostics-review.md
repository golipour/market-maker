# Architecture, Data Flow, and Improvement Review

This note reviews the current low-frequency market maker architecture and proposes improvements for code structure, market-making design, controller diagnostics, observability, and debugging. It is intentionally practical: each recommendation points to where it belongs in the current codebase.

## Executive Summary

The runtime is a single-process async, event-bus-driven market maker. Public exchange feeds are normalized into interval `MarketEvent`s, aligned into `StrategyInputFrame`s, cleaned by subtracting own resting orders, passed into the MPC strategy engine, filtered by risk, and optionally synced to the venue through `OrderManager`.

The strongest parts of the architecture are the bus-first design, typed domain models, explicit primary/hedge frame alignment, dry-run default, and the separation between strategy, risk, execution, persistence, and observability subscribers.

The biggest improvement opportunity is diagnostics. Important facts exist, but they are scattered across `StrategyDecision`, logs, metrics, execution events, model internals, and controller private attributes. The system needs a first-class diagnostics contract that records what the controller believed before updating models, why actions changed between stages, and what every gate/risk/venue rule did.

## Current Architecture

### Runtime Wiring

Main wiring is in `mm.runtime.engine`.

`build_runtime_components` builds:

- public market adapters per source through `select_public_market_data_adapter`
- one `SymbolEngine` per primary/hedge source
- optional private-state adapters and reconcilers
- optional `OrderManager`s when private config has credentials and execution enabled
- a `StrategyInputFrameAssembler`
- a `RuntimeControlPlane`
- a `StrategyRunner` when `controller.enabled` is true
- persistence and observability daemons

`RuntimeComponents.start()` then starts daemons, frame assembly, strategy runner, control plane, symbol engines, reconcilers, and scheduler.

### Layer Map


| Layer               | Main files                                                                     | Responsibility                                                        |
| ------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Runtime composition | `mm/runtime/engine.py`, `mm/runtime/config.py`                                 | Build and start the graph from JSON config                            |
| Market ingestion    | `mm/ingestion/symbol_engine.py`                                                | Connect adapters, consume trades/books, publish raw data and errors   |
| Aggregation         | `mm/aggregation/builder.py`, `mm/aggregation/scheduler.py`                     | Convert rolling book/trades into interval `MarketEvent`s              |
| Frame assembly      | `mm/application/frame_assembler.py`                                            | Align primary, hedge, and private snapshots into `StrategyInputFrame` |
| Frame cleaning      | `mm/application/frame_preparation.py`                                          | Subtract own open orders before strategy decisions                    |
| Strategy/controller | `mm/application/strategy_runner.py`, `mm/strategies/mpc_engine.py`, `mm/mpc/*` | Build MPC state, update models, optimize, discretize ladder orders    |
| Risk                | `mm/risk/engine.py`                                                            | Kill switch, stale-data gate, inventory filtering                     |
| Execution           | `mm/execution/order_manager.py`, `mm/application/action_routing.py`            | Route actions, cancel/place orders, maintain local live-order view    |
| Venue constraints   | `mm/core/order_constraints.py`, `mm/execution/order_validation.py`             | Quantize strategy orders and reject venue-invalid orders              |
| Persistence         | `mm/persistence/*`                                                             | Store event envelopes in Postgres or Parquet                          |
| Observability       | `mm/observability/*`                                                           | Convert bus events into metrics, logs, traces                         |


## Data Flow

### 1. Public Market Data

`SymbolEngine.start()` connects the venue adapter and starts feed tasks:

- `_feed_trades()` consumes normalized trades, updates `RollingTradeBuffer`, and publishes `market:raw:{exchange}:{symbol}:trades`
- `_feed_orderbook()` consumes snapshots/deltas, updates `OrderbookManager`, and publishes `market:raw:{exchange}:{symbol}:orderbook`
- feed/normalization errors publish to `market:errors:{exchange}:{symbol}`

### 2. Interval Market Events

`SymbolEngine.publish_on_intervals()` registers callbacks on `IntervalScheduler`. For each interval, `MarketEventBuilder.build()` reads:

- orderbook snapshot at `t_end_ms`
- trades sliced over `[t_start_ms, t_end_ms]`
- buy/sell volume, VWAP, top imbalance, microprice

It publishes `market:events:{exchange}:{symbol}:{interval_ms}`.

### 3. Strategy Frames

`StrategyInputFrameAssembler` consumes the primary market queue, optional hedge queue, and optional private-state queues. For each primary event:

1. Drain latest hedge/private inputs.
2. Wait up to `hedge_grace_ms` for a hedge event with matching `t_end_ms`.
3. Drain again.
4. Select latest hedge event at or before primary `t_end_ms`.
5. Mark hedge as `ready`, `missing`, `stale`, or `disabled`.
6. Publish `StrategyInputFrame` on `strategy:frames:{primary_exchange}:{symbol}:{interval_ms}`.

The frame still contains the exchange-visible orderbook, including the bot's own resting orders.

### 4. Controller Pipeline

`StrategyRunner.on_frame()` is the main production path:

1. Increment frame step.
2. Sync control-plane kill switch into the risk engine.
3. Read `OrderManager.live_orders`.
4. Clean frame with `clean_strategy_frame`, subtracting own open orders from primary/hedge books.
5. Build `TradingState` for risk.
6. Call `MpcStrategyEngine.decide(clean_frame, frame_step=...)`.
7. Risk-filter the proposed `Action`.
8. Publish `StrategyDecision` and `Action`.
9. Check live-execution gates.
10. Route approved actions to `OrderManager.sync()`.
11. Publish a small `strategy:journal:{actor_id}` event.

Current live gates are:

- approved actions are non-empty
- `controller.dry_run` is false
- control-plane `dry_run` is false
- warmup has passed: `frame_step > execution_after_steps`
- an order manager exists for the primary exchange

The current code wires control-plane `kill_switch` and `dry_run` into the runner. `paused_actors` exists in `RuntimeControlPlane`, but is not currently checked by `StrategyRunner`.

### 5. MPC Strategy Engine

`MpcStrategyEngine.decide()` performs:

1. Convert clean frame to MPC `State`.
2. Update hedge VWAP model from current hedge VWAPs.
3. Update market-flow model from current buy/sell volumes.
4. Update residual mid-price model from previous and current primary mid.
5. Run `MonteCarloMPCTensorBatched.optimize(state)`.
6. Convert continuous sigmoid depth into ladder orders on `orderbook_grid`.
7. Apply instrument constraints from `InstrumentSpec`.
8. Cap max orders per side.
9. Build `StrategyDecision`.

Important current diagnostics in `StrategyDecision`:

- model parameter snapshots: `local_price_model`, `local_market_flow_model`,
  `global_market_flow_model`, `global_price_model`
- optimizer metadata: kind, last expected delta wealth, reward components, redirected optimizer log
- continuous pre-discretization depth
- orderbook grid contract
- ladder orders before final cap/venue filtering and final orders
- decision hash

### 6. Risk and Execution

`RuntimeRiskEngine.validate()` rejects or filters actions by:

- kill switch
- stale primary data
- max inventory after each proposed order

`OrderManager.sync()` then:

1. Publishes `ExecutionSyncRequested`.
2. Cancels explicit order ids.
3. Optionally cancels missing desired orders when `cancel_missing=True`.
4. Deduplicates already-live orders.
5. Validates venue limits using `order_passes_limits`.
6. Publishes `OrderValidationRejected` for invalid orders.
7. Sends place requests through the venue command adapter.
8. Updates local `live_orders`.
9. Publishes `execution:state:{exchange}`.

## Main Pain Points

### Diagnostics Are Not First-Class

The controller has many internal decisions, but only a small subset becomes structured output. When debugging a bad quote, you need to reconstruct state from multiple places:

- `StrategyDecision` payload
- `strategy:journal:{actor_id}`
- `risk:events`
- `execution:orders:{exchange}`
- logs
- Grafana panels
- private model fields

That makes "why did this quote happen?" unnecessarily hard.

### Model Evaluation Is Mixed With Model Update

`MpcStrategyEngine.decide()` updates models before it builds the decision. For market-flow diagnostics, the useful online score is the prequential score:

> Evaluate likelihood of the realized event using model parameters from before seeing that event, then update the model.

Right now, there is no explicit pre-update score for:

- realized buy/sell market volumes under the market-flow model
- realized primary price/mid change under the price model
- realized hedge VWAP deltas under the hedge model

This is exactly the measure needed to know if the price and flow models are behaving live.

### Stage Transitions Lose Detail

The action changes through several stages:

1. continuous optimizer action
2. ladder orders
3. venue-quantized orders
4. max-per-side capped orders
5. risk-approved orders
6. execution-deduped orders
7. venue-limit-valid orders
8. venue-accepted orders

Some stages are visible, but not all reasons and counts are emitted in one coherent contract.

### Controller Has Too Many Roles

`MpcStrategyEngine` is both a bridge to legacy MPC internals and an online model updater and a decision payload builder. `StrategyRunner` owns orchestration, risk, publishing, live gating, and journaling. This works, but it makes targeted diagnostics difficult because there is no explicit "decision context" object crossing each stage.

### Risk Is Too Coarse

Risk currently handles kill switch, stale data, and max inventory. A real market maker usually needs more explicit controls:

- max gross open notional
- max order count and open-order age
- max spread from reference price
- max quote skew from inventory
- min hedge readiness or stale hedge behavior policy
- price-band checks against top of book and reference VWAP
- self-trade/crossing prevention
- daily loss / step loss / drawdown gates
- venue health and private-state freshness gates

### Observability Has Useful Metrics But Limited Cardinality-Aware Diagnostics

Current metrics cover frames, controller latency/order count, inventory/cash, execution events, sync latency, and validation rejects. They do not yet expose model quality, action-stage counts, gating reasons, queue lag, frame staleness, or risk exposure in a systematic way.

## Recommended Architecture Improvements

### 1. Add A Diagnostics Event Contract

Create a typed domain event, for example:

- `ControllerDiagnostics`
- topic: `strategy:diagnostics:{actor_id}` or `strategy:diagnostics`
- persisted by default
- consumed by observability subscriber

Suggested top-level fields:

```text
type = ControllerDiagnostics
actor_id
exchange
symbol
t_end_ms
frame_step
run_id
trace_id
timing
frame_health
model_scores
model_params_before
model_params_after
state_summary
action_pipeline
risk
execution_gates
controller_internal
warnings
```

This should be separate from `StrategyDecision`. A decision should stay concise and stable for downstream execution; diagnostics can be richer, optional, and versioned.

### 2. Split Model Score From Model Update

Add a protocol for online probabilistic models:

```python
class OnlineModel(Protocol):
    def get_params(self) -> dict: ...
    def score(self, observation: object) -> dict: ...
    def update(self, observation: object) -> None: ...
```

Then enforce this order inside `MpcStrategyEngine.decide()`:

1. Capture `params_before`.
2. Compute `score_before_update` for realized observation.
3. Update model.
4. Capture `params_after`.
5. Optimize with the intended parameter snapshot.

For market-flow models, score the current realized `(buy_volume, sell_volume)` using the previous parameters:

```text
flow_prob = P(V_buy = observed_buy, V_sell = observed_sell | params_before)
flow_nll = -log(max(flow_prob, eps))
```

For the residual price model, score the realized primary mid delta before updating:

```text
delta_mid = current_mid - previous_mid
price_nll = -log Normal(delta_mid; mu_mid_before, sigma_mid_before)
```

For the hedge model, score realized hedge VWAP deltas:

```text
delta_hedge_buy = current_hedge_buy_vwap - previous_hedge_buy_vwap
delta_hedge_sell = current_hedge_sell_vwap - previous_hedge_sell_vwap
hedge_nll = -log P(delta_buy, delta_sell | params_before)
```

The model score payload should include:

- `nll`
- `log_likelihood`
- `probability` when meaningful
- `observation`
- `params_before`
- `params_after`
- `updated`
- `skip_reason` when not enough history exists

### 3. Introduce A Decision Context Object

Add an internal object such as `DecisionContext` in `mm/application/controller_diagnostics.py` or `mm/application/decision_context.py`.

It should accumulate:

- input frame metadata
- clean-frame deltas, such as own-order quantities subtracted from each side
- state built for MPC
- model scores and params
- optimizer result and timing
- action-stage counts and rejections
- risk decisions
- execution gate reasons

`StrategyRunner` can publish this as diagnostics after the decision path completes, even when strategy raises an error.

### 4. Make The Action Pipeline Explicit

Refactor action creation into named stages:

```text
continuous_optimizer_depth
ladder_orders
strategy_capped_orders
risk_approved_orders
execution_deduped_orders
venue_limit_valid_orders
venue_accepted_orders
```

Each stage should report:

- input count
- output count
- per-side count
- total qty per side
- total notional per side
- rejected count by reason
- sample rejected orders, capped to a small number

Current stage locations:

- continuous and ladder: `mm/strategies/mpc_engine.py`
- strategy caps: `mm/core/order_constraints.py`
- risk: `mm/risk/engine.py`
- execution dedupe and venue validation: `mm/execution/order_manager.py`

### 5. Add Structured Risk Results

Change risk from returning only `list[Action]` to returning an object:

```python
@dataclass(frozen=True)
class RiskResult:
    approved: list[Action]
    rejected: list[RiskRejection]
    exposure_before: dict
    exposure_after: dict
    limits: dict
```

Keep a compatibility adapter if needed. This gives you a direct answer to "what did the risk engine do?"

Useful rejection reasons:

- `kill_switch`
- `stale_data`
- `max_inventory`
- `max_notional`
- `private_state_stale`
- `hedge_unavailable`
- `price_band`
- `would_cross`
- `daily_loss`

### 6. Separate Controller, Strategy, Models, And Execution Plan

Recommended boundaries:

- `mm.application.strategy_runner`: async orchestration only
- `mm.strategies.mpc_engine`: converts frame/state to proposed strategy plan
- `mm.models.online`: scoring/updating market-flow, price, hedge models
- `mm.execution.plan`: action-stage pipeline and order transformations
- `mm.diagnostics`: typed diagnostic events and serializers

The goal is not a large rewrite. Start by extracting pure helper functions from `MpcStrategyEngine.decide()` and returning diagnostics alongside the `StrategyDecision`.

## Observability Recommendations

### Model Quality Metrics

Add Prometheus metrics:


| Metric                         | Type            | Labels                                            | Meaning                                                    |
| ------------------------------ | --------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| `mm_model_nll`                 | histogram/gauge | `actor_id`, `exchange`, `symbol`, `model`, `side` | Pre-update negative log likelihood                         |
| `mm_model_log_likelihood`      | gauge           | same                                              | Raw log likelihood                                         |
| `mm_model_observation`         | gauge           | `model`, `field`                                  | Realized volume/delta being scored                         |
| `mm_model_param`               | gauge           | `model`, `param`                                  | Selected bounded parameter values                          |
| `mm_model_score_skipped_total` | counter         | `model`, `reason`                                 | Missing previous frame, missing hedge, invalid sigma, etc. |


For Grafana, plot:

- NLL over time with rolling mean and percentile bands
- observed vs expected buy/sell volume
- realized mid delta vs predicted `mu_mid +/- 2*sigma_mid`
- hedge VWAP delta vs predicted bands
- model skip reasons

### Controller Health Metrics

Add:

- `mm_frame_age_ms`: wall clock minus `primary.t_end_ms`
- `mm_frame_queue_depth`
- `mm_hedge_status_total`
- `mm_cleaning_subtracted_qty`
- `mm_controller_stage_latency_ms` with stage labels: `clean`, `model_score`, `model_update`, `optimize`, `discretize`, `risk`, `publish`, `execute`
- `mm_controller_gate_block_total` with reason labels: `dry_run_config`, `dry_run_control`, `warmup`, `no_manager`, `no_approved_actions`

### Action Pipeline Metrics

Add:

- `mm_action_stage_orders`
- `mm_action_stage_qty`
- `mm_action_stage_notional`
- `mm_action_rejected_total`

Labels:

- bounded labels only: `actor_id`, `exchange`, `symbol`, `stage`, `side`, `reason`
- do not use order ids as metric labels

### Risk And Order Constraints

Add:

- `mm_risk_exposure_inventory`
- `mm_risk_exposure_open_notional`
- `mm_risk_limit`
- `mm_risk_reject_total`
- `mm_order_constraint_reject_total`
- `mm_order_constraint_rounding_delta`

For order constraints, diagnostics should distinguish:

- dropped for `min_qty`
- dropped for `min_notional`
- rounded to zero quantity
- invalid non-finite price/qty
- capped by `max_orders_per_side`
- execution-layer `min_base_qty`
- execution-layer `min_notional_quote`

### Debug Snapshots

Add an optional, sampled debug event:

```text
strategy:debug:{actor_id}
```

This can include heavier payloads:

- top N cleaned book levels
- top N raw book levels
- own orders subtracted
- continuous depth series
- proposed and rejected orders
- model params before/after
- optimizer reward components

Config should control:

- enabled/disabled
- sample rate
- max payload size
- whether to include books
- whether to persist

## Debugging Workflows To Support

### Is The Price Model Working?

Required live fields:

- `delta_mid`
- `mu_mid_before`
- `sigma_mid_before`
- `price_log_likelihood`
- `price_nll`
- `price_params_after`
- `score_skip_reason`

Grafana panels:

- `price_nll` over time
- rolling average `price_nll`
- `delta_mid` with `mu_mid_before +/- 2*sigma_mid_before`
- histogram of standardized residuals: `(delta_mid - mu_mid_before) / sigma_mid_before`

Alert examples:

- rolling NLL above threshold
- sigma collapsed near `sigma_min`
- standardized residual absolute value above threshold repeatedly

### Why Did The Controller Place This Quote?

One diagnostics row should answer:

- what was the raw mid, clean mid, inventory, cash, hedge status?
- what did each model predict before update?
- what did the optimizer output continuously?
- what ladder orders were generated?
- what was removed by venue constraints?
- what was removed by max orders per side?
- what did risk remove?
- did execution run, and if not, which gate blocked it?
- if execution ran, what did the venue accept/reject?

### Why Are No Orders Reaching The Venue?

Check a single action-pipeline view:

- strategy generated zero continuous depth
- ladder generated zero marginal orders
- venue constraints dropped orders
- cap set to zero or too low
- risk rejected all orders
- dry-run/warmup/control gate blocked execution
- no order manager was built due to missing token or `enable_execution=false`
- execution dedupe found identical live orders
- order validation rejected by trade stats
- venue adapter failed

## Project Structure Suggestions

Current structure is workable, but several concepts are split by technical layer instead of domain lifecycle. Suggested additions:

```text
mm/
  diagnostics/
    models.py              # ControllerDiagnostics, ModelScore, ActionStageReport
    publisher.py           # payload shaping, sampling, redaction
  models/
    online.py              # score/update protocols
    scoring.py             # NLL helpers
  controller/
    context.py             # DecisionContext
    pipeline.py            # pure decision pipeline stages
  execution/
    plan.py                # action transformation/rejection reports
```

Keep existing public imports stable while gradually moving implementation details.

Recommended code hygiene:

- Avoid reading private attributes like `_last_E_dW`, `_last_J_components`, `_mu_buy` in many places; wrap them in explicit snapshot methods.
- Prefer typed diagnostic dataclasses over ad hoc dict payloads.
- Keep `StrategyDecision` small and execution-oriented; put large model/controller internals into `ControllerDiagnostics`.
- Add unit tests for every diagnostic event shape.
- Add replay tests that assert diagnostics are deterministic for a fixture stream.

## Market-Making Architecture Suggestions

### Quote Policy

The MPC already optimizes continuous depth, then discretizes. Add explicit quote-policy checks after optimization:

- do not cross best bid/ask unless intentionally taking liquidity
- enforce minimum spread from mid/reference
- enforce maximum quote distance
- inventory-skew controls
- hedge-readiness behavior: quote smaller or wider when hedge is stale/missing
- stale private state behavior: dry-run or no quoting

### Inventory And PnL Controls

Add risk controls for:

- gross inventory
- net inventory including hedge venue
- open notional
- mark-to-market PnL
- realized PnL
- daily drawdown
- per-step expected loss
- fee-adjusted spread profitability

### Execution Control

Improve `OrderManager` reporting:

- classify dedupe skips
- emit cancel/place latencies separately
- capture adapter exceptions as structured events
- record live-order age
- support explicit quote replace policy: cancel-before-place, place-before-cancel, or atomic when venue supports it
- make `cancel_missing` configurable per venue and mode

### Hedge Handling

Hedge is currently an input to the MPC state and optional private state. Make hedge policy explicit:

- `hedge_required_for_quoting`
- `hedge_stale_action`: `quote_zero`, `widen`, `size_down`, `dry_run`
- `hedge_inventory_limit`
- `hedge_basis_limit`
- hedge VWAP model NLL and skip reasons

## Implementation Roadmap

### Phase 1: Low-Risk Diagnostics

Add `ControllerDiagnostics` payload without changing trading behavior.

- Add model pre-update NLL calculations.
- Publish diagnostics on a new topic.
- Persist diagnostics by default.
- Add Prometheus metrics for model NLL, frame age, stage latency, and execution gate blocks.
- Add tests with fixed frames and deterministic expected diagnostic fields.

### Phase 2: Structured Stage Results

Refactor action transformations to return stage reports.

- `apply_venue_constraints` should optionally return rejections.
- `cap_orders_per_side` should report capped orders.
- `RuntimeRiskEngine` should return `RiskResult`.
- `OrderManager` should report dedupe and validation stage summaries.

### Phase 3: Controller Boundary Cleanup

Extract:

- model scoring/updating
- state building
- optimizer invocation
- order pipeline
- diagnostics publication

Keep current runtime behavior and event names stable while improving internals.

### Phase 4: Dashboards And Alerts

Add Grafana panels for:

- model NLL
- standardized residuals
- observed vs expected flow
- action pipeline funnel
- risk rejections
- execution gates
- frame/queue lag
- private-state freshness

## Concrete First Patch Recommendation

The first engineering patch should be small:

1. Add `ModelScore` and `ControllerDiagnostics` dataclasses.
2. In `MpcStrategyEngine.decide()`, compute scores before calling update methods.
3. Add diagnostics to `StrategyDecision.diagnostics` temporarily or publish a separate event from `StrategyRunner`.
4. Extend `RuntimeObservabilitySubscriber` to emit `mm_model_nll`.
5. Add unit tests for market-flow NLL and price NLL using fixed model params.

This gives immediate visibility into whether the price and flow models are working without changing order behavior.
