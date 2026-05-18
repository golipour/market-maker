# Codebase Upgrade Plan

This plan turns the architecture, diagnostics, MPC optimizer, and hedging recommendations into an implementation roadmap. The goal is cleaner code, clearer responsibility boundaries, reliable diagnostics, and Grafana dashboards that make the controller understandable while it is running.

## Goals

- Make the controller pipeline readable as named stages.
- Separate hard risk/execution gates from soft optimizer preferences.
- Make model scoring pre-update and observable.
- Make optimizer diagnostics first-class: initial, final, and best objective/action snapshots.
- Keep `StrategyDecision` compact and move heavy internals into typed diagnostics events.
- Add Grafana dashboards for controller diagnostics, model scores, and portfolio state.
- Prepare the codebase for fill-driven hedge execution without pretending simulated hedging is live hedging.

## Current Constraints

The existing runtime already has a good event-bus foundation:

- market data becomes `MarketEvent`
- frames become `StrategyInputFrame`
- `StrategyRunner` performs clean frame -> MPC -> risk -> execution
- `StrategyDecision` is persisted to Postgres through `mm_events`
- dashboards already query JSON payloads from persisted bus events

The main limitation is that diagnostic facts are not a stable contract. They are distributed across `StrategyDecision`, optimizer private fields, logs, risk events, execution events, and dashboards.

## Target Architecture

### New Packages

Add small packages rather than making existing files larger:

```text
mm/
  diagnostics/
    __init__.py
    models.py              # typed ControllerDiagnostics, ModelScore, ObjectiveSnapshot, ActionStageReport
    publisher.py           # bus publishing, sampling, payload-size guardrails
    serializers.py         # JSON-safe conversion helpers
  controller/
    __init__.py
    context.py             # DecisionContext accumulated across one frame
    pipeline.py            # named pure-ish controller stages
  models/
    __init__.py
    scoring.py             # pre-update NLL helpers
    protocols.py           # OnlineScoredModel protocol
  execution/
    plan.py                # action stage reports and order transformation reports
  hedging/
    __init__.py
    models.py              # HedgeIntent, HedgeExecutionDiagnostic
    service.py             # future fill-driven hedge execution
```

Keep public imports stable at first. Move behavior behind compatibility wrappers and migrate gradually.

### Responsibility Boundaries

| Area | Owns | Should not own |
| --- | --- | --- |
| `StrategyRunner` | async orchestration, live gates, publishing | model update details, optimizer internals |
| `MpcStrategyEngine` | frame-to-strategy proposal bridge | risk, live execution, dashboard payload shaping |
| `MonteCarloMPCTensorBatched` | optimizing continuous effective action | order constraints, risk, persistence |
| `RuntimeRiskEngine` | hard action/portfolio gates | soft objective shaping |
| `OrderManager` | venue sync, dedupe, adapter calls | strategy intent, model assumptions |
| `RuntimeObservabilitySubscriber` | metrics/logs from events | business logic |
| `HedgeExecutionService` | future fill-driven hedge orders | primary quote optimization |

## Diagnostics Event Contract

Add `ControllerDiagnostics` as a separate event from `StrategyDecision`.

Suggested topic:

```text
strategy:diagnostics:{actor_id}
```

Persist it by default when persistence is enabled.

### Payload Shape

```text
type = ControllerDiagnostics
version = 1
actor_id
exchange
symbol
t_end_ms
frame_step
created_at_ms
run_id

frame:
  hedge_status
  hedge_lag_ms
  frame_age_ms
  primary_private_age_ms
  hedge_private_age_ms

state:
  mid
  primary_inventory
  primary_cash
  hedge_inventory
  hedge_cash
  primary_buy_vwap
  primary_sell_vwap
  hedge_buy_vwap
  hedge_sell_vwap

models:
  price: ModelScore
  market_flow: ModelScore
  hedge_price: ModelScore

optimizer:
  config
  timing
  objective
  action
  gradients
  rollout

action_pipeline:
  continuous
  ladder
  venue_quantized
  capped
  risk_approved
  execution_deduped
  venue_limit_valid
  venue_accepted

risk:
  approved
  rejection_counts
  exposure_before
  exposure_after
  limits

execution_gates:
  executed
  blocked_reasons

warnings
```

### ModelScore

Score model observations before updating the model.

```text
model
kind
updated
skip_reason
observation
params_before
params_after
log_likelihood
nll
standardized_residual
```

Required scores:

- market-flow NLL for realized `(buy_volume, sell_volume)`
- primary price NLL for realized mid delta
- hedge price NLL for realized hedge VWAP deltas

### Optimizer Diagnostics

The optimizer should expose a public `OptimizerDiagnostics` object instead of private fields.

```text
objective:
  initial:
    value
    loss
    components
    expected_total_reward
    expected_next_reward
    main_reward_mean
    hedge_reward_mean
  final:
    value
    loss
    components
  best:
    epoch
    value
    loss
    components
    expected_total_reward
    expected_next_reward
    main_reward_mean
    hedge_reward_mean
  improvement:
    best_minus_initial
    final_minus_initial
```

Action snapshots:

```text
action:
  initial:
    effective_depth
    raw_params
  final:
    effective_depth
    raw_params
  best:
    effective_depth
    raw_params
```

Use effective cumulative depth `D(spread) - D(0)` everywhere in diagnostics because that is what becomes executable ladder size.

## Implementation Phases

### Phase 1: Stabilize Diagnostics Without Changing Trading Behavior

Files:

- `mm/diagnostics/models.py`
- `mm/diagnostics/publisher.py`
- `mm/application/strategy_runner.py`
- `mm/strategies/mpc_engine.py`
- `mm/core/models/strategy.py`
- `mm/runtime/engine.py`

Work:

1. Add diagnostic dataclasses.
2. Add `strategy:diagnostics:{actor_id}` to default persistence topics.
3. Build `DecisionContext` inside `StrategyRunner.on_frame()`.
4. Publish diagnostics even when MPC raises `ControllerError`.
5. Add payload versioning.
6. Add tests for JSON shape and persistence envelope compatibility.

Acceptance criteria:

- Existing `StrategyDecision` consumers still work.
- One diagnostics event is emitted per processed frame.
- Diagnostics event can be replayed and decoded.

### Phase 2: Model Pre-Update Scoring

Files:

- `mm/models/scoring.py`
- `mm/mpc/market_order_flow_torch.py`
- `mm/mpc/price_torch.py`
- `mm/mpc/hedge_price_torch.py`
- `mm/strategies/mpc_engine.py`

Work:

1. Add scoring helpers for hurdle/gamma/weibull/lognormal flow models.
2. Add Gaussian residual score for primary price model.
3. Add hedge VWAP delta score.
4. Change model update order:
   - read params before
   - score current realized event
   - update
   - read params after
5. Add skip reasons:
   - `missing_previous_frame`
   - `missing_hedge`
   - `invalid_sigma`
   - `unsupported_model`

Acceptance criteria:

- Diagnostics contain `nll` for each available model.
- NLL is computed from pre-update params.
- Unit tests use deterministic observations and fixed params.

### Phase 3: Optimizer Diagnostics And Cleanup

Files:

- `mm/mpc/mpc_tensor_batched.py`
- `mm/mpc/reward_tensor.py`
- `mm/mpc/mpc_state.py`
- `mm/mpc/state_tensor.py`

Work:

1. Add `OptimizerDiagnostics`, `ObjectiveSnapshot`, and `ActionSnapshot`.
2. Add `optimize_with_diagnostics(state)` or `last_diagnostics`.
3. Track `initial`, `final`, and `best` objective snapshots explicitly.
4. Track `best_epoch`.
5. Track initial/final/best effective action curves.
6. Replace private-field scraping in `MpcStrategyEngine`.
7. Keep `optimize(state)` as compatibility wrapper.
8. Replace unconditional `print()` timing with diagnostics/logging.
9. Vectorize MC sample drawing where model APIs support batched samples.
10. Replace `.data` mutation with `torch.no_grad()` and `copy_()`.

Acceptance criteria:

- Initial/final/best objective components are visible in diagnostics.
- Initial/final/best effective action curves are visible.
- Tests prove `sum(components) == objective.value`.
- Tests prove `best.value >= initial.value` for fixed samples in a simple fixture.

### Phase 4: Action Pipeline Reports

Files:

- `mm/execution/plan.py`
- `mm/core/order_constraints.py`
- `mm/risk/engine.py`
- `mm/execution/order_manager.py`
- `mm/application/action_routing.py`

Work:

1. Return stage reports from ladder/constraint/cap functions.
2. Change risk to return `RiskResult`.
3. Report execution dedupe skips.
4. Report venue limit rejections by reason.
5. Add per-side qty/notional totals at every stage.

Acceptance criteria:

- Diagnostics can answer why no orders reached venue.
- Risk and order-constraint rejection reasons are counted.
- Backward-compatible methods still return plain actions where needed.

### Phase 5: Clean Risk And Config

Files:

- `mm/runtime/config.py`
- `mm/risk/engine.py`
- `mm/risk/factory.py`
- `mm/configs/market_maker.runtime.example.json`
- `docs/configuration.md`

Work:

1. Split hard risk from optimizer soft preferences.
2. Add primary and hedge hard limits:
   - min/max inventory
   - min cash
   - max open notional
   - max order notional
   - private state freshness
   - market data freshness
3. Keep optimizer preferences under `controller.optimizer`.
4. Add migration aliases for older config keys.

Suggested config:

```json
"risk": {
  "primary": {
    "min_inventory": -1000.0,
    "max_inventory": 1000.0,
    "min_cash": 10000000.0,
    "max_open_notional": 200000000.0,
    "max_order_notional": 50000000.0
  },
  "hedge": {
    "min_inventory": -1000.0,
    "max_inventory": 1000.0,
    "min_cash": 10000000.0,
    "max_market_order_notional": 50000000.0,
    "max_slippage_bps": 20.0
  }
}
```

Acceptance criteria:

- Hard risk limits are visible in diagnostics.
- Runtime rejects unsafe actions regardless of optimizer preference.
- Example config uses clear names only.

### Phase 6: Fill-Driven Hedge Execution

Files:

- `mm/hedging/models.py`
- `mm/hedging/service.py`
- `mm/core/models/execution.py`
- `mm/exchanges/*/order_adapter.py`
- `mm/exchanges/capabilities.py`
- `mm/runtime/engine.py`

Work:

1. Add `HedgingRuntimeConfig`.
2. Add `HedgeIntent`.
3. Add `HedgeExecutionDiagnostic`.
4. Subscribe hedge service to primary fills/private updates.
5. Convert primary fills to hedge market orders.
6. Add adapter capability checks for market orders.
7. Route approved hedge orders through hedge `OrderManager`.
8. Track residual unhedged inventory.

Acceptance criteria:

- Primary fill produces hedge intent.
- Hedge intent produces hedge market order when enabled and safe.
- Failure/rejection creates clear diagnostic event.
- Optimizer `hedge_rollout_enabled` remains separate from live `hedging.enabled`.

## Grafana Plan

The dashboards should use persisted Postgres `mm_events` for rich JSON diagnostics and Prometheus for lightweight live counters. Existing dashboards already use Postgres JSON queries, so the new diagnostics should be JSON-friendly and stable.

### Dashboard 1: Controller Diagnostics

New file:

```text
grafana/dashboards/mm_controller_diagnostics_pg.json
```

Variables:

- `actor_id`
- `exchange`
- `symbol`
- `diagnostics_topic`
- `interval_ms`

Panels:

1. Latest controller status table:
   - frame step
   - hedge status
   - executed
   - blocked reasons
   - risk rejection counts
   - latency/timing summary

2. Objective total time series:
   - `optimizer.objective.initial.value`
   - `optimizer.objective.final.value`
   - `optimizer.objective.best.value`
   - `optimizer.objective.improvement.best_minus_initial`

3. Objective components time series:
   - one panel per stage: `initial`, `final`, `best`
   - series from `components` keys:
     - `expected_pnl`
     - `inventory_target_penalty`
     - `pnl_volatility_penalty`
     - `quote_shape_uniformity_penalty`
     - `quote_size_target_penalty`
     - `quote_spread_penalty`

4. Initial vs final vs best action depth:
   - latest frame only
   - XY chart with spread on X, effective cumulative qty on Y
   - bid and ask shown separately or mirrored around mid
   - plot `optimizer.action.initial.effective_depth`
   - plot `optimizer.action.final.effective_depth`
   - plot `optimizer.action.best.effective_depth`

5. Action pipeline funnel:
   - continuous -> ladder -> venue quantized -> capped -> risk approved -> execution deduped -> venue valid -> accepted
   - counts, qty, notional by side

6. Execution gate blocks:
   - stacked time series by reason:
     - `observe_only_config`
     - `observe_only_control`
     - `warmup`
     - `no_manager`
     - `no_approved_actions`
     - `risk_rejected`

7. Optimizer timing:
   - sample draw time
   - train time
   - total time
   - best epoch
   - nonfinite loss/grad count

Example SQL pattern for objective totals:

```sql
SELECT
  e.ts AS "time",
  stage,
  (payload #>> ARRAY['optimizer','objective',stage,'value'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND $__timeFilter(e.ts)
ORDER BY 1;
```

Example SQL pattern for objective components:

```sql
SELECT
  e.ts AS "time",
  stage || '.' || kv.key AS metric,
  (kv.value #>> '{}')::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload #> ARRAY['optimizer','objective',stage,'components'], '{}'::jsonb)
) AS kv(key, value)
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND $__timeFilter(e.ts)
  AND jsonb_typeof(kv.value) = 'number'
ORDER BY 1;
```

### Dashboard 2: Extend Model Parameters With Model Scores

Existing file:

```text
grafana/dashboards/mm_strategy_model_params_pg.json
```

Add panels:

1. Model NLL time series:
   - `models.price.nll`
   - `models.market_flow.nll`
   - `models.hedge_price.nll`

2. Rolling NLL:
   - moving average by model
   - optional p95 if using SQL window functions

3. Standardized residuals:
   - primary price standardized residual
   - hedge buy/sell standardized residuals

4. Observed vs predicted:
   - observed buy/sell flow vs expected buy/sell flow
   - observed mid delta vs `mu_mid_before`
   - observed hedge VWAP delta vs `mu_delta_*`

5. Score skip reasons:
   - count by model and skip reason

Example SQL for NLL:

```sql
SELECT
  e.ts AS "time",
  model AS metric,
  (payload #>> ARRAY['models', model, 'nll'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','hedge_price']) AS model
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND $__timeFilter(e.ts)
  AND payload #>> ARRAY['models', model, 'nll'] IS NOT NULL
ORDER BY 1;
```

### Dashboard 3: Portfolio State

New file:

```text
grafana/dashboards/mm_portfolio_pg.json
```

Variables:

- `primary_exchange`
- `hedge_exchange`
- `symbol`
- `actor_id`

Panels:

1. Primary and hedge inventory time series:
   - from `ControllerDiagnostics.state.primary_inventory`
   - from `ControllerDiagnostics.state.hedge_inventory`
   - optionally from private snapshots for exchange truth

2. Primary and hedge cash time series:
   - `state.primary_cash`
   - `state.hedge_cash`

3. Marked portfolio value:
   - primary cash + primary inventory * primary sell VWAP
   - hedge cash + hedge inventory * hedge sell VWAP
   - combined value

4. Inventory limits overlay:
   - min/max primary inventory
   - min/max hedge inventory

5. Cash limits overlay:
   - primary min cash
   - hedge min cash

6. Residual hedge exposure:
   - `primary_inventory + hedge_inventory` when symbols are comparable
   - future `hedge.residual.unhedged_qty`

Example SQL for inventory:

```sql
SELECT e.ts AS "time",
       'primary_inventory' AS metric,
       (payload #>> '{state,primary_inventory}')::double precision AS value
FROM mm_events e
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND $__timeFilter(e.ts)
UNION ALL
SELECT e.ts AS "time",
       'hedge_inventory' AS metric,
       (payload #>> '{state,hedge_inventory}')::double precision AS value
FROM mm_events e
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND $__timeFilter(e.ts)
ORDER BY 1;
```

### Dashboard 4: Hedge Execution

Add after `HedgeExecutionService` exists:

```text
grafana/dashboards/mm_hedge_execution_pg.json
```

Panels:

- hedge intents total
- hedge submitted/accepted/rejected
- hedge latency
- hedge slippage
- residual unhedged inventory
- rejection reasons
- source fill to hedge order trace table

## Metrics Plan

Prometheus metrics should mirror dashboards but stay low cardinality:

```text
mm_model_nll{actor_id,exchange,symbol,model}
mm_optimizer_objective_value{actor_id,exchange,symbol,stage}
mm_optimizer_objective_component{actor_id,exchange,symbol,stage,component}
mm_optimizer_objective_improvement{actor_id,exchange,symbol}
mm_optimizer_best_epoch{actor_id,exchange,symbol}
mm_action_stage_orders{actor_id,exchange,symbol,stage,side}
mm_action_stage_qty{actor_id,exchange,symbol,stage,side}
mm_controller_gate_block_total{actor_id,exchange,symbol,reason}
mm_portfolio_inventory{actor_id,exchange,symbol,role}
mm_portfolio_cash{actor_id,exchange,symbol,role}
```

Do not include order ids, trace ids, or client ids as Prometheus labels.

## Testing Plan

### Unit Tests

- model scoring math
- score-before-update ordering
- optimizer objective snapshots
- effective action consistency
- action pipeline stage reports
- risk result reasons
- config aliases and migrations

### Integration Tests

- replay fixture -> diagnostics events
- diagnostics event -> Postgres envelope
- primary fill fixture -> hedge intent
- hedge risk reject fixture
- dashboard SQL smoke tests against sample `mm_events`

### Regression Tests

- existing `StrategyDecision` schema remains compatible
- old optimizer config names still parse during migration period
- observe-only never routes live orders
- diagnostics publish even when optimizer raises

## Migration Strategy

1. Add new diagnostics events while keeping existing decision payload.
2. Add dashboards against new diagnostics topic.
3. Keep old optimizer config names as aliases for at least one release.
4. Update example config and docs to use only clear names.
5. After dashboards are stable, reduce heavy fields in `StrategyDecision`.
6. Only then add fill-driven hedge execution.

## Suggested Delivery Order

1. `ControllerDiagnostics` event skeleton and persistence topic.
2. Optimizer initial/final/best diagnostics.
3. Grafana controller diagnostics dashboard.
4. Model pre-update NLL scoring.
5. Add model-score panels to existing model dashboard.
6. Portfolio diagnostics fields and portfolio dashboard.
7. Action pipeline stage reports.
8. Expanded hard risk config.
9. Hedge execution service.
10. Hedge execution dashboard.

This order gets visibility first, then structural cleanup, then live hedge execution. That is safer than changing trading behavior before the system can explain itself.

