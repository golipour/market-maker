# Observability and operations

This runtime should be observable through four complementary telemetry layers:

| Layer | Role | Storage / UI | Rule of thumb |
|-------|------|--------------|---------------|
| Events | Durable domain facts and decisions | PostgreSQL `mm_events`, Parquet, replay tools, Grafana SQL dashboards | Use for "what happened and why?" |
| Metrics | Low-cardinality health, latency, counts, and alerts | Prometheus, Grafana | Use for "is it healthy right now?" |
| Logs | Human-readable incident trail | JSON lines, Loki | Use for "what did the process say around this incident?" |
| Traces | Causal timing across async stages | OpenTelemetry, Tempo | Use for "where did this frame spend time?" |

The current codebase already has the right shape: business logic publishes bus
events, observability is a subscriber, and persistence stores event envelopes.
The recommendations below make that coverage systematic and clean.

## Current stack

`configure_observability(config.observability)` runs during runtime build in
`mm.observability.setup`.

| Config field | Effect |
|--------------|--------|
| `json_logs` | Enables structured JSON stdout / file logs through `MM_JSON_LOG` |
| `log_path` | Sets `MM_LOG_PATH`; relative paths resolve against the repo root so Alloy can mount `./logs` |
| `metrics_enabled` + `metrics_port` | Starts HTTP `/metrics` on the configured port, default `8000` |
| `run_id`, `service_name` | Runtime identity for logs, events, traces, and optional metric labels |
| `otel_enabled` + `otlp_endpoint` | Configures OpenTelemetry traces to Tempo through OTLP |

Local Docker services:

| Service | Port | Role |
|---------|------|------|
| Postgres | 5432 | Durable `mm_events`, `mm_latest_state`, `step_economics` |
| Redis | 6379 | Optional durable `redis_streams` bus |
| Grafana | 3000 | Dashboards from `grafana/dashboards` |
| Prometheus | 9090 | Scrapes runtime `/metrics` through `host.docker.internal:8000` |
| Loki | 3100 | Log aggregation |
| Tempo | 3200 / 4317 / 4318 | Distributed traces |
| Alloy | 12345 | Ships `logs/*.jsonl` and Docker logs to Loki |
| pgAdmin | 5050 | Database UI |
| RedisInsight | 5540 | Redis UI |

Provisioning:

- Datasources: `grafana/provisioning/datasources/`
- Dashboard loader: `grafana/provisioning/dashboards/dashboards.yml`
- Dashboard JSON: `grafana/dashboards/`
- Observability v2 dashboard JSON: `grafana/observability-v2/`
- Prometheus config: `docker/prometheus.yml`
- Prometheus alert rules: `docker/prometheus/market_maker_alerts.yml`
- Loki / Tempo / Alloy config: `docker/loki.yaml`, `docker/tempo.yaml`, `docker/alloy.alloy`

Grafana folders:

| Folder | Provider | Filesystem path |
|--------|----------|-----------------|
| `Market Maker` | `market_maker` | `/var/lib/grafana/dashboards` |
| `Market Maker - Observability v2` | `market_maker_observability_v2` | `/var/lib/grafana/observability-v2` |

## Telemetry policy

Use each layer deliberately.

Events:

- Persist every externally meaningful state transition, decision, rejection, and execution result.
- Keep payloads rich and replayable.
- Include `type`, `version` where appropriate, `run_id`, `actor_id`, `exchange`, `symbol`, `t_end_ms`, `frame_step`, and `timestamp_ms`.
- Do not rely on logs as the primary business record.

Metrics:

- Keep labels bounded: `actor_id`, `exchange`, `symbol`, `stage`, `side`, `reason`, `model`, `operation`, `writer`.
- Do not use order ids, client order ids, trace ids, exception messages, raw topics with unbounded values, or payload blobs as labels.
- Prefer histograms for latency and distributions, counters for events, gauges for current state.
- Expose only the fields needed for dashboards and alerts; leave rich detail in events.

Alert rules:

- Prometheus loads source-controlled rules from `docker/prometheus/market_maker_alerts.yml`.
- The local rule set covers target reachability, stale market data, ingest lag, frame age, stale private/ledger readiness, execution sync failures, router failures, stale pending order confirmations, validation rejection spikes, persistence errors, kill switch, and observe-only state.
- The local Compose stack evaluates alerts but does not provision Alertmanager or Grafana contact points. Production deployments should add notification routing and tune thresholds to the configured strategy interval and venue behavior.

Logs:

- Emit one structured log for important lifecycle events and unusual paths.
- Logs should be concise, correlated with events through `run_id`, `actor_id`, `exchange`, `symbol`, `frame_step`, `t_end_ms`, `decision_hash`, and `trace_id` when available.
- Log exceptions with class, operation, retryability, and a short message.

Traces:

- Use spans around frame assembly, controller stages, risk, execution routing, adapter calls, and persistence flushes.
- Use trace ids in logs and event payloads when possible; never put trace ids on Prometheus labels.

## Current telemetry coverage

### Runtime subscriber

`RuntimeObservabilitySubscriber` consumes:

- `strategy:joint:frames:...`
- `ledger:snapshot`
- `strategy:decisions`
- `strategy:actions`
- `execution:orders:{exchange}`
- `execution:state:{exchange}`
- `market:errors:{exchange}:{symbol}`

It currently emits metrics and logs for:

| Metric / log | Meaning |
|--------------|---------|
| `mm_strategy_frames_total` | Legacy strategy frame count |
| `mm_joint_strategy_frames_total` | Ledger-backed joint frame count |
| `mm_latency_skew_ms` | Local/global market skew |
| `mm_frame_local_market_ready` | Local market readiness flag |
| `mm_frame_global_market_ready` | Global market readiness flag |
| `mm_frame_ledger_fresh` | Ledger freshness flag |
| `mm_global_lag_ms` | Selected global event lag |
| `mm_controller_latency_ms` | Controller decision latency |
| `mm_controller_order_count` | Orders in decision |
| `mm_inventory`, `mm_cash` | Strategy inventory and cash from decisions |
| `mm_frame_to_exec_sync_latency_ms` | Frame creation to execution sync request |
| `mm_market_window_end_to_exec_sync_latency_ms` | Market window end to execution sync request |
| `mm_order_sync_latency_ms` | Completed order-manager sync latency; falls back to request-to-snapshot for legacy paths |
| `mm_execution_live_orders` | Current private-confirmed live orders by exchange |
| `mm_execution_pending_orders` | Router-accepted place/cancel changes awaiting private confirmation |
| `mm_execution_pending_order_max_age_ms` | Oldest pending order confirmation age by exchange |
| `mm_execution_events_total` | Execution events by type |
| `mm_order_validation_rejected_total` | Venue limit rejections |
| `mm_persistence_*` | Persistence write/error counters and batch gauges |
| `strategy_frame`, `joint_strategy_frame`, `strategy_decision`, `runtime_error` | Structured log events |

### Persistence

`PostgresEventWriter` stores envelopes in:

- `mm_events`: append-only event truth
- `mm_market_facts`: typed market interval facts
- `mm_action_stage_facts`: typed controller action-stage facts
- `mm_execution_facts`: typed order execution lifecycle facts
- `mm_target_order_facts`: typed target-book intent orders
- `mm_model_score_facts`: typed model score and skip facts
- `mm_optimizer_facts`: typed optimizer timing and objective facts
- `mm_optimizer_component_facts`: typed optimizer objective component facts
- `mm_latest_state`: latest state snapshots by key
- `admin_audit_log`: administrative audit records

Before writing `mm_events`, Postgres persistence compacts market-event trade lists after interval summaries are available, replaces persisted joint-frame embedded ledgers with summaries plus `ledger:snapshot` references, and stores controller-diagnostics action-pipeline order/rejection arrays as counts, side counts, digests, and small samples. It also writes each `MarketEvent` to `mm_market_facts` with typed columns for best bid/ask, depth, mid, spread, microprice, VWAP, interval flow, imbalance, and trade count. Each `ControllerDiagnostics.action_pipeline` stage is written to `mm_action_stage_facts` with typed columns for order counts, side counts, quantities, notionals, rejection counts, compacted counts, digests, and lineage IDs. Execution sync, place, cancel, validation rejection, fill, and execution snapshot events are written to `mm_execution_facts` with typed order lifecycle fields, including confirmed versus pending state kind and lineage IDs. Each `JointTargetBook` order is written to `mm_target_order_facts` with typed intent fields for actor, mode, venue role, exchange, symbol, policy, side, price, quantity, notional, order type, tags, and lineage IDs. Model score diagnostics are written to `mm_model_score_facts`, and optimizer timing/objective/component diagnostics are written to `mm_optimizer_facts` and `mm_optimizer_component_facts`; these diagnostic fact tables also carry lineage IDs. Private snapshots stay full in Postgres so `mm_latest_state` can preserve current-state truth. The writer reuses one psycopg2 connection across batches, reconnects after failures, retries failed batches, and spills exhausted batches to `persistence_spool/postgres`.

`ParquetEventWriter` stores compact archival envelopes in immutable part files. It omits the redundant `payload_json` copy and keeps one compact `envelope_json` for replay. It also writes typed nullable columns for common market, controller, optimizer, target-book, and execution fields so offline scans do not need to parse JSON for the usual analytics paths. The archival writer compacts market-event trade lists after interval summaries are available, compacts joint-frame embedded ledgers, compacts controller-diagnostics action-pipeline order/rejection arrays, and removes zero private balances while preserving counts. Live writes do not rewrite existing Parquet files; large deployments should compact small part files offline if needed.

Default persistence topics include market events, joint frames, ledger snapshots,
strategy actions, strategy decisions, controller diagnostics, target books,
execution events, execution state, and private state for local/global venues.
For REST private polling, the durable private-state path is the full snapshot;
per-balance and per-open-order topics are reserved for incremental adapters and
real deltas.

At larger scale, keep Postgres for event truth/latest state and consider a time-series database or typed columnar store for high-volume market data, execution latency, and dashboard facts.

### Controller diagnostics

`ControllerDiagnostics` is the rich diagnostic contract for one controller frame.
It includes:

- frame health and sync status
- state and wallet summaries
- model scores
- optimizer diagnostics
- action pipeline reports
- risk results
- execution gates
- warnings
- cleared orderbook used by MPC

This is the primary event for SQL-backed Grafana dashboards.

## Known gaps and first fixes

These should be handled before relying on production alerts.

| Priority | Item | Why |
|----------|------|-----|
| P0 | Set a non-empty `observability.run_id` for every real run | Incident review, dashboard filtering, replay, and Parquet organization need a stable run key |
| P1 | Emit Prometheus metrics from `ControllerDiagnostics` | The richest health facts are currently mostly SQL-only |
| P1 | Add persistence queue depth and flush latency | Backpressure and slow database writes should be visible before data loss or lag |
| P1 | Add structured `observability_setup_failed` logs when metrics, logs, or OTEL setup fails | Current setup is intentionally best effort, but silent setup failures are painful operationally |
| P1 | Add `ExecutionSyncCompleted` events | Dashboards should not infer successful sync only from snapshots and accepted order events |
| P2 | Add targeted Postgres indexes for dashboard filters | Most panels filter by `topic`, `event_type`, `exchange`, `symbol`, and `ts` |
| P2 | Add retention / partitioning policy for `mm_events` | Market events and diagnostics will grow quickly |

Suggested additional indexes:

```sql
CREATE INDEX IF NOT EXISTS mm_events_topic_type_ts_idx
ON mm_events (topic, event_type, ts DESC);

CREATE INDEX IF NOT EXISTS mm_events_actor_ts_idx
ON mm_events (actor_id, ts DESC)
WHERE actor_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS mm_events_payload_gin_idx
ON mm_events USING GIN (payload jsonb_path_ops);
```

The GIN index is optional. Use it only if JSON path filters become slow; it can
increase write cost.

## Event contract

Events are the durable source of truth. They should answer:

- What did the runtime observe?
- What did the controller believe?
- What action did it want?
- What did risk change?
- What did execution attempt?
- What did the venue accept or reject?
- What state resulted?

### Existing important events

| Event type | Topic | Purpose |
|------------|-------|---------|
| `MarketEvent` | `market:events:{exchange}:{symbol}:{interval_ms}` | Interval market snapshot and volume |
| `JointStrategyFrame` | `strategy:joint:frames:{exchange}:{symbol}:{interval_ms}` | Local/global/ledger-aligned controller input |
| `StrategyDecision` | `strategy:decisions` | Concise strategy output |
| `ControllerDiagnostics` | `strategy:diagnostics:{actor_id}` | Rich frame-level diagnostics |
| `JointTargetBook` | `strategy:target_books:{actor_id}` | Desired local/global resting book |
| `Action` | `strategy:actions` | Execution-compatible action |
| `RiskActionsFiltered` | `risk:events` | Risk filtered action summary |
| `ExecutionSyncRequested` | `execution:orders:{exchange}` | Order manager sync started |
| `PlaceOrdersRequested` / `PlaceOrdersAccepted` / `PlaceOrdersFailed` | `execution:orders:{exchange}` | Placement lifecycle |
| `CancelOrdersRequested` / `CancelOrdersAccepted` / `CancelOrdersFailed` | `execution:orders:{exchange}` | Cancel lifecycle |
| `OrderValidationRejected` | `execution:orders:{exchange}` | Venue rules rejected an order before REST |
| `ExecutionSnapshot` | `execution:state:{exchange}` | Current private-confirmed live orders plus pending place/cancel counts |
| private snapshots/fills/balances | `private:state:{exchange}:...` | Exchange-confirmed private state |
| `JointLedgerSnapshot` | `ledger:snapshot` | Consolidated portfolio state |
| `RuntimeControlState` | `admin:events` | Kill switch, observe-only, pause state |

### Proposed new or expanded events

| Event type | Topic | Producer | Required fields | Purpose |
|------------|-------|----------|-----------------|---------|
| `FrameAssemblyHealth` | `strategy:frames:health` | `JointStrategyFrameAssembler` | `exchange`, `symbol`, `t_end_ms`, `local_ready`, `global_ready`, `ledger_fresh`, `latency_skew_ms`, `local_private_age_ms`, `global_private_age_ms` | Explain frame readiness independent of strategy |
| `ControllerStageTiming` | `strategy:diagnostics:{actor_id}` or embedded in `ControllerDiagnostics.optimizer.timing` | `StrategyRunner` / `DecisionContext` | `actor_id`, `exchange`, `symbol`, `frame_step`, `stage`, `latency_ms` | Persist exact stage timings for SQL dashboards |
| `RiskDecision` | `risk:events` | risk engine | `actor_id`, `exchange`, `symbol`, `frame_step`, `approved`, `rejection_counts`, `exposure_before`, `exposure_after`, `limits` | Durable explanation of risk output |
| `ExecutionGateDecision` | `strategy:diagnostics:{actor_id}` or `execution:gates` | `StrategyRunner` | `actor_id`, `frame_step`, `venue_gates`, `blocked_reasons`, `executed` | Explain why live sync did or did not run |
| `ExecutionSyncCompleted` | `execution:orders:{exchange}` | `OrderManager` / router path | `exchange`, `actor_id`, `symbol`, `frame_step`, `target_count`, `keep_count`, `cancel_count`, `place_count`, `accepted_count`, `pending_place_count`, `pending_cancel_count`, `pending_max_age_ms`, `latency_ms` | Clear success boundary for sync rounds and confirmation backlog |
| `ExecutionSyncFailed` | `execution:errors` | `StrategyRunner` / `OrderManager` | `exchange`, `actor_id`, `symbol`, `frame_step`, `error`, `message`, `timeout_ms`, `retryable` | Clear failure boundary for sync rounds |
| `AdapterRequestFailed` | `execution:orders:{exchange}` and structured log | `OrderRouter` / private REST adapters | `exchange`, `operation`, `error`, `message`, `retryable`, `attempt` | Separate exchange/API failures from strategy errors |
| `PersistenceBatchWritten` | structured log, optional `persistence:events` | `PostgresEventWriter`, `ParquetEventWriter` | `writer`, `batch_size`, `latency_ms`, `queue_depth` | Persistence health and capacity planning |
| `PersistenceBatchFailed` | structured log, optional `persistence:events` | persistence writers | `writer`, `batch_size`, `error`, `message`, `retryable` | Persistence failure analysis |
| `ObservabilitySetupFailed` | structured log | `configure_observability` | `component`, `error`, `message` | Make best-effort setup failures visible |
| `ModelScoreSkipped` | `strategy:diagnostics:{actor_id}` or embedded model score | model scoring path | `model`, `reason`, `observation` | Explain missing NLL or residuals |

Prefer embedding per-frame facts inside `ControllerDiagnostics` when the fact is
about one controller frame. Use separate events when there is an independent
lifecycle, such as persistence flushes, adapter requests, or admin commands.

## Admin and safety telemetry

`RuntimeControlPlane` listens on `admin:commands` and publishes
`RuntimeControlState` on `admin:events`.

| Command | Current telemetry / behavior |
|---------|------------------------------|
| `kill_switch` | Sets control-plane state, emits `mm_kill_switch_state`, logs `admin_command_applied`, and is synchronized into the risk engine by `StrategyRunner` |
| `observe_only` | Sets control-plane state, emits `mm_observe_only_state`, logs `admin_command_applied`, and blocks live execution gates |
| `cancel_all` | Publishes cancel request events on execution cancel topics |
| `cancel_actor_orders` | Publishes actor-scoped cancel request events for the requested exchange |
| `pause` / `resume` | Updates actor pause state and publishes it; consumers must explicitly enforce it |
| `pause_local` / `resume_local` | Updates local venue role pause state and publishes it |
| `pause_global` / `resume_global` | Updates global venue role pause state and publishes it |

Recommended additions:

- Persist admin command requests and resulting `RuntimeControlState` in `admin_audit_log`.
- Add `mm_admin_commands_total{command,status}` rather than only `command`.
- Add structured logs for rejected or malformed admin commands.
- Show kill switch, observe-only, and pause state on the runtime overview dashboard.
- Alert on kill switch enabled and on observe-only enabled during an intended live run.

## Metrics plan

Prometheus metrics should mirror the dashboards and alerts but stay
low-cardinality.

### Runtime and ingestion

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_market_events_total` | counter | `exchange`, `symbol` | `MarketEvent` observer | Market intervals produced |
| `mm_market_event_age_ms` | gauge | `exchange`, `symbol` | `MarketEvent` observer | Wall clock minus event `t_end_ms` |
| `mm_market_spread` | gauge | `exchange`, `symbol` | `MarketEvent` observer | Best ask minus best bid |
| `mm_market_mid` | gauge | `exchange`, `symbol` | `MarketEvent` observer | Best bid/ask midpoint |
| `mm_market_trade_volume` | gauge | `exchange`, `symbol`, `side` | `MarketEvent` observer | Interval buy/sell volume |
| `mm_market_errors_total` | counter | `exchange`, `symbol`, `error` | market error topics | Ingestion errors |
| `mm_bus_queue_depth` | gauge | `topic` or bounded `consumer` | bus implementations | Subscriber lag / local queue pressure |

Use bounded topic labels if exposing `topic`; otherwise split into
`component`, `role`, `exchange`, and `symbol`.

### Frame and controller health

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_joint_strategy_frames_total` | counter | `exchange`, `symbol`, `mode`, `global_status` | existing subscriber | Joint frame count |
| `mm_frame_age_ms` | gauge or histogram | `exchange`, `symbol` | `ControllerDiagnostics.frame.frame_age_ms` | Frame staleness |
| `mm_frame_sync_ready` | gauge | `exchange`, `symbol`, `field` | `FrameSyncStatus` | Readiness flags: local market, global market, local private, global private, ledger |
| `mm_latency_skew_ms` | gauge | `exchange`, `symbol`, `mode`, `global_status` | existing subscriber | Local/global market skew |
| `mm_controller_decisions_total` | counter | `actor_id`, `exchange`, `symbol` | existing subscriber | Decision count |
| `mm_controller_latency_ms` | histogram | `actor_id`, `exchange`, `symbol` | decision or diagnostics | End-to-end controller latency |
| `mm_controller_stage_latency_ms` | histogram | `actor_id`, `exchange`, `symbol`, `stage` | `DecisionContext` | Stage latency for `clean`, `model_score`, `model_update`, `optimize`, `discretize`, `risk`, `publish`, `execute` |
| `mm_controller_warnings_total` | counter | `actor_id`, `exchange`, `symbol`, `reason` | `ControllerDiagnostics.warnings` | Warning rate |
| `mm_controller_errors_total` | counter | `actor_id`, `exchange`, `symbol`, `error` | `ControllerError` | Controller failures |

### Model quality

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_model_nll` | histogram or gauge | `actor_id`, `exchange`, `symbol`, `model` | `ControllerDiagnostics.models` | Pre-update negative log likelihood |
| `mm_model_log_likelihood` | gauge | `actor_id`, `exchange`, `symbol`, `model` | model scores | Raw log likelihood |
| `mm_model_standardized_residual` | gauge | `actor_id`, `exchange`, `symbol`, `model` | model scores | Residual divided by predicted sigma |
| `mm_model_observation` | gauge | `actor_id`, `exchange`, `symbol`, `model`, `field` | model scores | Realized volume/delta used for scoring |
| `mm_model_param` | gauge | `actor_id`, `exchange`, `symbol`, `model`, `param` | model params | Selected bounded model parameters |
| `mm_model_score_skipped_total` | counter | `actor_id`, `exchange`, `symbol`, `model`, `reason` | model scores | Missing history, invalid sigma, missing global, etc. |

If `param` becomes too broad, export only selected parameters or keep full
params in Postgres and dashboards.

### Action pipeline

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_action_stage_orders` | gauge | `actor_id`, `exchange`, `symbol`, `stage`, `side` | `ActionPipelineDiagnostics` | Number of orders per stage and side |
| `mm_action_stage_qty` | gauge | `actor_id`, `exchange`, `symbol`, `stage`, `side` | action stage reports | Base quantity per stage |
| `mm_action_stage_notional` | gauge | `actor_id`, `exchange`, `symbol`, `stage`, `side` | action stage reports | Quote notional per stage |
| `mm_action_rejected_total` | counter | `actor_id`, `exchange`, `symbol`, `stage`, `reason` | action stage reports | Rejection counts |

Recommended stage values:

- `continuous`
- `ladder`
- `capped`
- `target_book`
- `risk_approved_target`
- `planned_cancel`
- `planned_place`
- `rules_valid`
- `router_accepted`
- `kept`

### Risk and execution gates

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_risk_reject_total` | counter | `actor_id`, `exchange`, `symbol`, `reason` | risk engine / diagnostics | Risk rejection counts |
| `mm_risk_approved` | gauge | `actor_id`, `exchange`, `symbol` | diagnostics | 1 when risk approved frame |
| `mm_risk_exposure` | gauge | `actor_id`, `exchange`, `symbol`, `field` | diagnostics | Exposure before/after fields |
| `mm_risk_limit` | gauge | `actor_id`, `exchange`, `symbol`, `field` | risk config | Configured limits |
| `mm_execution_gate_block_total` | counter | `actor_id`, `exchange`, `symbol`, `reason` | execution gates | Live execution blocks |
| `mm_kill_switch_state` | gauge | none or `actor_id` | control plane / risk | Kill switch state |
| `mm_observe_only_state` | gauge | none or `actor_id` | control plane | Admin observe-only state |

Recommended gate reasons:

- `observe_only_config`
- `observe_only_control`
- `warmup`
- `no_order_manager`
- `no_approved_actions`
- `no_orders`
- `stale_market_data`
- `missing_private_state`
- `stale_private_state`
- `missing_global_market_data`
- `stale_global_market_data`
- `global_live_disabled`
- `kill_switch`

### Execution and adapter health

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_order_sync_total` | counter | `exchange` | existing subscriber | Sync requests |
| `mm_execution_sync_completed_total` | counter | `exchange` | proposed `ExecutionSyncCompleted` | Successful sync rounds |
| `mm_execution_sync_failed_total` | counter | `exchange`, `error` | `ExecutionSyncFailed` | Failed sync rounds |
| `mm_order_sync_latency_ms` | histogram | `exchange` | existing subscriber | Completed sync latency; legacy fallback uses request-to-snapshot latency |
| `mm_frame_to_exec_sync_latency_ms` | histogram | `exchange` | existing subscriber after typo fix | Frame creation to sync request |
| `mm_order_place_accepted_total` | counter | `exchange` | existing subscriber | Accepted places |
| `mm_order_cancel_accepted_total` | counter | `exchange` | existing subscriber | Accepted cancels |
| `mm_order_router_failures_total` | counter | `exchange`, `operation`, `error` | router | Adapter cancel/place failures |
| `mm_execution_placed_orders_total` | counter | `exchange`, `symbol`, `side` | `PlaceOrdersAccepted` | Venue-accepted new orders by bid/ask side |
| `mm_execution_placed_qty_total` | counter | `exchange`, `symbol`, `side` | `PlaceOrdersAccepted` | Venue-accepted new order base quantity by bid/ask side |
| `mm_execution_failed_orders_total` | counter | `exchange`, `symbol`, `side`, `reason` | `PlaceOrdersFailed`, `OrderValidationRejected` | Failed or rejected order placements by bid/ask side |
| `mm_execution_failed_qty_total` | counter | `exchange`, `symbol`, `side`, `reason` | `PlaceOrdersFailed`, `OrderValidationRejected` | Failed or rejected order base quantity by bid/ask side |
| `mm_execution_filled_orders_total` | counter | `exchange`, `symbol`, `side` | private fill stream | Private fill events by bid/ask side |
| `mm_execution_filled_qty_total` | counter | `exchange`, `symbol`, `side` | private fill stream | Private filled base quantity by bid/ask side |
| `mm_order_validation_rejected_total` | counter | `exchange`, `reason` | existing subscriber | Venue rules rejected order |
| `mm_execution_live_orders` | gauge | `exchange` | existing subscriber | Current live order count |
| `mm_execution_live_orders_by_side` | gauge | `exchange`, `symbol`, `side` | execution snapshots | Current live order count by bid/ask side |
| `mm_execution_live_order_qty` | gauge | `exchange`, `symbol`, `side` | execution snapshots | Current live remaining base quantity by bid/ask side |
| `mm_order_sync_create_count` | gauge | `exchange` | existing subscriber | Last requested create count |
| `mm_order_sync_cancel_count` | gauge | `exchange` | existing subscriber | Last requested cancel count |

### Persistence

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_persistence_events_written_total` | counter | `writer` | existing writer | Written event count |
| `mm_persistence_errors_total` | counter | `writer`, `error` | existing writer | Persistence errors |
| `mm_persistence_batch_size` | gauge | `writer` | existing writer | Last batch size |
| `mm_persistence_queue_depth` | gauge | `writer` | proposed writer metric | Queue backlog |
| `mm_persistence_write_latency_ms` | histogram | `writer` | proposed writer metric | Flush latency |
| `mm_persistence_dropped_events_total` | counter | `writer`, `reason` | proposed if queues can drop | Dropped persistence events |

### Event Payload Size

| Metric | Type | Labels | Source | Meaning |
|--------|------|--------|--------|---------|
| `mm_event_payloads_total` | counter | `topic`, `event_type`, `stage` | bus and persistence enqueue paths | Event count measured for byte sizing |
| `mm_event_payload_bytes_total` | counter | `topic`, `event_type`, `stage` | bus and persistence enqueue paths | Total normalized JSON payload bytes |
| `mm_event_payload_bytes` | histogram | `topic`, `event_type`, `stage` | bus and persistence enqueue paths | Payload byte distribution |
| `mm_event_envelope_bytes_total` | counter | `topic`, `event_type`, `stage` | Redis stream and persistence enqueue paths | Total durable envelope bytes |
| `mm_event_envelope_bytes` | histogram | `topic`, `event_type`, `stage` | Redis stream and persistence enqueue paths | Envelope byte distribution |

## Logs and exceptions

Use structured JSON logs for anything a human might need during an incident.

Recommended common log fields:

```json
{
  "ts_ms": 1710000000000,
  "level": "error",
  "event": "execution_sync_failed",
  "service": "market-maker",
  "env": "local",
  "run_id": "live-20260524-001",
  "actor_id": "mpc-controller",
  "exchange": "nobitex",
  "symbol": "USDT_IRT",
  "frame_step": 123,
  "t_end_ms": 1710000000000,
  "component": "strategy_runner",
  "operation": "route_joint_target_book",
  "error": "TimeoutError",
  "message": "ExecutionSyncTimeout: exceeded 3000ms placement_timeout_ms",
  "retryable": true,
  "decision_hash": "optional",
  "trace_id": "optional"
}
```

Recommended log events:

| Event | Level | Where | Meaning |
|-------|-------|-------|---------|
| `runtime_started` | info | runtime entrypoint | Config contract and run id |
| `runtime_stopped` | info | runtime entrypoint | Clean shutdown |
| `observability_setup_failed` | warning/error | `configure_observability` | Metrics/logs/OTEL setup failed |
| `persistence_writers_configured` | info | runtime engine | Persistence sink summary |
| `postgres_event_writer_ready` | info | Postgres writer | Database ready |
| `postgres_event_writer_error` | error | Postgres writer | Batch write or DDL error |
| `strategy_frame` | info/debug | observability subscriber | Frame observed |
| `joint_strategy_frame` | info/debug | observability subscriber | Joint frame observed |
| `strategy_decision` | info | observability subscriber | Decision observed |
| `controller_error` | error | strategy runner | MPC/controller failed |
| `controller_warning` | warning | strategy runner | Nonfatal warning |
| `risk_decision_rejected` | warning | risk engine / subscriber | Risk blocked action |
| `execution_gate_blocked` | info/warning | strategy runner | Live execution did not run |
| `execution_sync_completed` | info | order manager | Sync completed |
| `execution_sync_failed` | error | strategy runner / order manager | Sync failed or timed out |
| `adapter_request_failed` | error | router / adapters | Exchange API failure |
| `order_validation_rejected` | warning | router / subscriber | Local venue rule rejection |
| `admin_command_applied` | warning | control plane | Runtime control changed |

Exception handling policy:

| Exception location | Required telemetry | Runtime behavior |
|--------------------|--------------------|------------------|
| Market ingestion adapter | counter, structured log, `market:errors:{exchange}:{symbol}` event | Continue polling/streaming when safe; reconnect/backoff if applicable |
| Frame assembly | structured log, runtime error counter, optional `FrameAssemblyHealth` | Continue next frame unless state is corrupt |
| MPC/controller decision | `ControllerError` event, `ControllerDiagnostics` with warning, counter, log | Do not execute orders for that frame; continue next frame |
| Risk engine | counter, log, diagnostics warning | Prefer fail-closed: reject actions for that frame |
| Execution gate | metric and diagnostics blocked reason | No exception; explain the block |
| Execution sync timeout | `ExecutionSyncFailed`, counter, log, diagnostics warning | Mark frame not executed; continue |
| Adapter place/cancel failure | `PlaceOrdersFailed` or `CancelOrdersFailed`, counter, log | Continue according to planner policy; fail closed on uncertain state |
| Persistence write failure | `mm_persistence_errors_total`, structured log | Do not crash trading path; alert if sustained |
| Observability setup failure | structured log | Continue runtime, but make missing telemetry visible |

## Tracing plan

Use spans around the end-to-end frame path:

```text
mm.frame_assembler.build
mm.strategy_runner.frame
mm.strategy_runner.clean_frame
mm.controller.model_score
mm.controller.model_update
mm.controller.optimize
mm.controller.discretize
mm.controller.risk
mm.controller.publish
mm.execution.route
mm.execution.cancel_orders
mm.execution.place_orders
mm.persistence.flush
```

Span attributes should be bounded:

- `run_id`
- `actor_id`
- `exchange`
- `symbol`
- `frame_step`
- `t_end_ms`
- `stage`
- `operation`

Do not attach full order payloads, API tokens, or raw exchange responses to
span attributes.

## Grafana dashboard design

The existing dashboards are useful and should remain. The suggested clean
layout below gives operators a standard path from overview to drilldown.

Common variables:

| Variable | Example | Source |
|----------|---------|--------|
| `run_id` | `live-20260524-001` | `SELECT DISTINCT run_id FROM mm_events ORDER BY 1 DESC` |
| `actor_id` | `mpc-controller` | `SELECT DISTINCT actor_id FROM mm_events WHERE actor_id IS NOT NULL ORDER BY 1` |
| `local_exchange` | discovered | `mm_events` ledger payload |
| `global_exchange` | discovered | `mm_events` ledger payload |
| `exchange` | discovered | `SELECT DISTINCT exchange FROM mm_events` |
| `symbol` | discovered | `SELECT DISTINCT symbol FROM mm_events` |
| `interval_ms` | discovered | event payload/topic suffix |
| `diagnostics_topic` | discovered | `strategy:diagnostics:%` topics in `mm_events` |
| `target_books_topic` | discovered | `strategy:target_books:%` topics in `mm_events` |
| `ledger_topic` | `ledger:snapshot` | runtime config |

### Existing dashboard inventory

These dashboards already exist and should be kept source-controlled:

| File / UID | Purpose |
|------------|---------|
| `mm_runtime_metrics.json` / `mm-runtime-metrics` | Prometheus scrape health, frame/decision rates, execution rates, latency, errors, joint frame readiness |
| `mm_runtime_logs_loki.json` / `mm-runtime-logs` | Loki runtime logs and structured JSON log filtering |
| `mm_market_data_pg.json` / `mm-market-data-pg` | Local/global market volumes and best bid/ask |
| `mm_controller_diagnostics_pg.json` / `mm-controller-diagnostics-pg` | Optimizer objective, reward/return, inventory penalties, expected fills |
| `mm_local_action_pg.json` / `mm-local-action-pg` | Target book, action pipeline, risk, execution plan |
| `mm_local_action_rejections_pg.json` / `mm-local-action-rejections-pg` | Discretization, risk, rules gateway, live gate rejection reasons |
| `mm_liquidity_sourcing_pg.json` / `mm-liquidity-sourcing-pg` | Local/global frame sync, venue execution gates, joint target books |
| `mm_ledger_pg.json` / `mm-ledger-pg` | Ledger balances, live orders, freshness, consolidated inventory/wealth |
| `mm_portfolio_pg.json` / `mm-portfolio-pg` | Inventory, cash, wallets, limits, collateral exposure |
| `mm_orderbook_depth_pg.json` / `mm-orderbook-depth-pg` | Latest book, cleared book, MPC continuous depth, executable orders |
| `mm_reference_vwaps_pg.json` / `mm-reference-vwaps-pg` | Local/global reference VWAPs from decisions |
| `mm_strategy_model_params_pg.json` / `mm-strategy-model-params-pg` | Model params, conditional price gate/component params, NLL, skip reasons |
| `private_state_postgres.json` / `private_state_postgres` | Private snapshots, balances, fills, recent private events |

The proposed 00-10 dashboard structure below can be implemented by reorganizing
or extending these existing dashboards rather than replacing everything at once.

### 00 Runtime overview

Purpose: one screen to answer "is the bot alive and safe?"

Panels:

| Panel | Type | Datasource | Query idea |
|-------|------|------------|------------|
| Prometheus scrape | stat | Prometheus | `max(up{job="market_maker"}) or vector(0)` |
| Frame rate | timeseries | PostgreSQL | `JointStrategyFrame` rows in `mm_events`, rate-normalized |
| Decision rate | timeseries | PostgreSQL | `StrategyDecision` rows in `mm_events`, rate-normalized |
| Controller stage p95 latency | timeseries | PostgreSQL | p95 over `ControllerDiagnostics.optimizer.timing` values |
| Order sync p95 latency | timeseries | PostgreSQL | p95 over `ExecutionSyncCompleted` / `ExecutionSyncFailed` `latency_ms` |
| Sync requests | timeseries | PostgreSQL | `ExecutionSyncRequested` rows in `mm_events`, rate-normalized |
| Placed orders | timeseries | PostgreSQL | `PlaceOrdersAccepted.payload.orders`, split by exchange/symbol/side |
| Failed orders | timeseries | PostgreSQL | `PlaceOrdersFailed.payload.orders`, split by exchange/symbol/side/reason |
| Fill events | timeseries | PostgreSQL | `private:state:%:fills` rows, split by exchange/symbol/side |
| Filled quantity | timeseries | PostgreSQL | `private:state:%:fills.payload.qty`, rate-normalized |
| Live orders by side | timeseries | PostgreSQL | `execution:state:%` snapshot `live_orders`, split by side |
| Live order quantity by side | timeseries | PostgreSQL | `execution:state:%` snapshot `remaining_qty`, split by side |
| Persistence errors | timeseries | Loki | `persistence_batch_failed` log events by writer/error |
| Latest controller warnings | table | PostgreSQL | Latest warnings from `ControllerDiagnostics.warnings` |

The runtime overview intentionally uses persisted events for most panels. If the
Python runtime cannot open the `/metrics` socket, PostgreSQL-backed panels still
show frames, decisions, diagnostics, sync attempts, and zero baselines for event
types that did not happen in the selected time range. The metrics server emits
`observability_metrics_ready` when `/metrics` is bound, and
`observability_setup_failed` includes `host`, `port`, and a socket-permission hint
when binding fails.

### 01 Market data quality

Purpose: validate exchange feed quality before debugging strategy behavior.

Panels:

- local and global market event rate
- market event age by venue
- best bid / best ask / mid
- spread and spread percent
- buy/sell interval volume
- top-of-book depth
- local/global latency skew
- global status: `ready`, `missing`, `stale`, `disabled`
- market ingestion errors by exchange and symbol
- raw latest `MarketEvent` table for current variables

Preferred Postgres table: `mm_market_facts`.

Useful typed columns:

- `buy_volume`, `sell_volume`, `total_trade_volume`
- `best_bid`, `best_bid_qty`, `best_ask`, `best_ask_qty`
- `mid`, `spread`, `microprice`, `vwap`, `imbalance`
- `trade_count`, `bid_depth_levels`, `ask_depth_levels`

`mm_events` remains the source for full replay/debug payloads.

### 02 Frame and controller health

Purpose: explain whether inputs were fresh and where controller time was spent.

Panels:

- frame age
- local/private/global/private age
- frame readiness flags
- ledger fresh flag
- local/global latency skew
- controller end-to-end latency
- stage latency stacked or separate charts:
  - `clean`
  - `model_score`
  - `model_update`
  - `optimize`
  - `discretize`
  - `risk`
  - `publish`
  - `execute`
- controller warnings table
- controller errors table

Primary source: `ControllerDiagnostics`.

### 03 Strategy model quality

Purpose: tell whether price/flow/global models are predictive or drifting.

Panels:

- NLL by model
- rolling NLL mean
- standardized residuals with threshold lines at `-3`, `0`, `3`
- observed vs predicted buy volume
- observed vs predicted sell volume
- realized mid delta vs predicted `mu +/- 2 sigma`
- global VWAP delta vs predicted band
- model parameter evolution
- model skip reasons table

Recommended event source:

- `mm_model_score_facts`
- `ControllerDiagnostics.models.{model}.nll`
- `ControllerDiagnostics.models.{model}.standardized_residual`
- `params_before`
- `params_after`
- `observation`
- `skip_reason`

### 04 Optimizer diagnostics

Purpose: inspect MPC objective quality and optimization stability.

Panels:

- objective value: initial, final, best
- objective improvement: final minus initial
- objective contribution fractions
- expected total reward
- expected total return
- next-step reward
- inventory penalties
- global reward and main reward means
- rollout expected fills
- optimizer timing
- gradient norm, nonfinite loss count, nonfinite gradient count when available

Preferred Postgres tables: `mm_optimizer_facts` and `mm_optimizer_component_facts`.

Existing dashboard:

- `grafana/dashboards/mm_controller_diagnostics_pg.json`

### 05 Action pipeline funnel

Purpose: answer "did the joint target book reach execution?"

Panels:

- desired target orders by venue from `JointTargetBook`
- desired target quantity by venue and side
- high-level execution contract funnel:
  - desired target
  - risk approved
  - router accepted
- blocked/rejected reasons, grouped into readable domains such as risk,
  exchange rules, router, order shaping, and venue limits
- latest execution summary with friendly command counts for new orders,
  cancels, unchanged existing orders, and blocked reasons

Low-level planner fields such as `planned_place`, `planned_cancel`, `kept`, and
`rules_valid` are still present in live `ControllerDiagnostics.action_pipeline`
for debugging, but the v2 dashboard intentionally does not plot them as the
primary funnel. Persisted diagnostics keep the stage counts, side counts,
digests, and small samples instead of storing full repeated order/rejection
arrays.

Preferred Postgres table: `mm_action_stage_facts`.

Useful typed columns:

- `stage`, `order_count`, `buy_count`, `sell_count`
- `buy_qty`, `sell_qty`, `buy_notional`, `sell_notional`
- `rejection_count`, `rejection_counts`
- `orders_compacted_count`, `rejections_compacted_count`
- `orders_digest`, `rejections_digest`

Existing dashboards:

- `grafana/dashboards/mm_local_action_pg.json`
- `grafana/dashboards/mm_local_action_rejections_pg.json`

### 06 Risk and execution

Purpose: explain risk decisions and venue execution health.

Panels:

- risk approved flag
- risk rejection counts
- exposure before and after
- exposure vs configured limits
- execution gate live flags
- gate blocked reasons
- sync requested/completed/failed rates
- place requested/accepted/failed rates
- cancel requested/accepted/failed rates
- order validation rejection rate
- sync latency p50/p95/p99
- adapter failures table
- current live orders

Sources:

- `ControllerDiagnostics.risk`
- `ControllerDiagnostics.execution_gates`
- `mm_execution_facts`
- `execution:orders:{exchange}` and `execution:state:{exchange}` for full payload replay
- Prometheus execution metrics
- Loki adapter failure logs

### 07 Portfolio and PnL

Purpose: understand inventory, collateral, and realized/estimated economics.

Panels:

- local and global inventory
- local and global cash
- consolidated base inventory
- consolidated quote wealth
- available vs locked base/quote
- locked collateral before risk and after target book
- open notional
- inventory limits
- step economics:
  - `spread_pnl`
  - `inventory_pnl`
  - `cross_pnl`
  - `fee_pnl`
  - `wealth_pnl`
- rolling cumulative wealth PnL
- drawdown from session high

Existing dashboards:

- `grafana/dashboards/mm_portfolio_pg.json`
- `grafana/dashboards/mm_ledger_pg.json`

### 08 Liquidity sourcing

Purpose: inspect local/global coordination.

Panels:

- local/global market readiness
- local/global private readiness
- latency skew
- local and global target book count
- local and global target book buy/sell qty
- global live gate status
- global blocked reasons
- basis or reference VWAP spread
- global cap rejections
- global adapter failures

Existing dashboard:

- `grafana/dashboards/mm_liquidity_sourcing_pg.json`

### 09 Runtime logs

Purpose: operational text trail with structured filtering.

Panels:

- log volume by event
- error and warning volume
- all structured JSON logs
- error logs filtered by current `run_id`, `exchange`, `symbol`, `actor_id`
- admin command logs
- adapter failure logs
- persistence failure logs

Existing dashboard:

- `grafana/dashboards/mm_runtime_logs_loki.json`

Recommended Loki filters:

```logql
{job="market-maker-runtime"} | json | event="execution_sync_failed"
{job="market-maker-runtime"} | json | exchange="$exchange" | symbol="$symbol"
{job="market-maker-runtime"} | json | level=~"warning|error"
```

### 10 Incident drilldown

Purpose: reconstruct one bad or surprising frame.

Variables:

- `run_id`
- `actor_id`
- `exchange`
- `symbol`
- `frame_step`
- `t_end_ms`
- `decision_hash`

Panels:

- latest `ControllerDiagnostics` JSON for the frame
- matching `StrategyDecision`
- matching `JointTargetBook`
- matching `ExecutionSyncRequested` / `ExecutionSyncCompleted` / `ExecutionSyncFailed`
- matching execution order events
- matching `execution:state`
- matching `ledger:snapshot`
- Loki logs filtered by frame identifiers
- Tempo trace link when trace id is available

This dashboard should be slower and more detailed than the overview dashboards.

## Alert plan

Thresholds should be tuned to the trading interval and venue behavior. These
defaults assume `interval_ms = 10000`.

| Alert | Example expression | Severity | Action |
|-------|--------------------|----------|--------|
| Runtime scrape down | `up{job="market_maker"} == 0` for 1m | critical | Runtime may be down or metrics unreachable |
| No frames | `sum(rate(mm_joint_strategy_frames_total[1m])) == 0` for 2m | critical | Check ingestion, bus, frame assembler |
| Frames but no decisions | frame rate > 0 and decision rate == 0 for 2m | critical | Check strategy runner and controller errors |
| Market data stale | `mm_market_event_age_ms > 2 * interval_ms` | critical | Check exchange feed/polling |
| Private state stale | `local_private_age_ms` or `global_private_age_ms` above limit | critical for live | Stop live execution until private state recovers |
| Global skew high | `mm_latency_skew_ms > max_global_lag_ms` | warning/critical | Check global venue feed |
| Controller p95 slow | p95 `mm_controller_latency_ms > interval_ms * 0.5` | warning | Optimize or reduce workload |
| Controller timeout risk | p95 stage `execute` near `placement_timeout_ms` | warning | Check venue latency |
| Execution sync failed | `rate(mm_execution_sync_failed_total[5m]) > 0` | critical | Inspect adapter and exchange status |
| Adapter failures | `rate(mm_order_router_failures_total[5m]) > 0` | critical | Check API, tokens, limits |
| Order validation rejects spike | `rate(mm_order_validation_rejected_total[5m]) > threshold` | warning | Check min qty/notional, rounding, config |
| Risk reject spike | `rate(mm_risk_reject_total[5m]) > threshold` | warning | Inspect risk dashboard |
| Kill switch enabled | `mm_kill_switch_state == 1` | critical/info depending intent | Confirm operator action |
| Observe-only enabled while expected live | `mm_observe_only_state == 1` or config observe-only true | warning | Confirm deployment mode |
| Persistence errors | `rate(mm_persistence_errors_total[5m]) > 0` | warning/critical | Check Postgres/Parquet sink |
| Persistence backlog | `mm_persistence_queue_depth > batch_size * 5` | warning | Increase DB capacity or batch settings |
| Drawdown | session drawdown exceeds configured threshold | critical | Trigger safety review / kill switch |
| Exposure high | exposure > 80 percent of risk limit | warning | Watch inventory and target book |

## Code locations for additions

| Module | Additions |
|--------|-----------|
| `mm/observability/subscribers.py` | Fix frame bookkeeping typo; observe `ControllerDiagnostics`; emit model, action pipeline, risk, gate, and frame age metrics |
| `mm/observability/schema.py` | Add stable metric and log event constants |
| `mm/observability/setup.py` | Log setup failures for metrics, OTEL, and log path initialization |
| `mm/observability/logs.py` | Add optional `level`, `component`, and exception helper if desired |
| `mm/application/strategy_runner.py` | Add stage timing spans/metrics; emit execution gate logs; publish `ExecutionSyncCompleted` / failure context |
| `mm/controller/context.py` | Store stage timings, warning reasons, and normalized blocked reasons in diagnostics |
| `mm/diagnostics/models.py` | Extend diagnostics version when adding new top-level fields; keep backward-compatible SQL where possible |
| `mm/risk/net_delta.py` | Emit structured `RiskDecision`; add actor/exchange/symbol labels to risk metrics where available |
| `mm/execution/order_manager.py` | Publish `ExecutionSyncCompleted`; include latency and counts |
| `mm/execution/router.py` | Count/log adapter failures by operation and exception class |
| `mm/persistence/postgres.py` | Add queue depth gauge and write latency histogram |
| `mm/persistence/parquet_writer.py` | Add matching writer metrics/logs |
| `mm/ingestion/symbol_engine.py` | Add market event age/spread metrics and ingestion error events if not already emitted upstream |
| `mm/bus/redis_streams.py` | Add stream lag / queue depth metrics if needed for multi-process operation |
| `mm/tools/generate_grafana_dashboards.py` | Generate new dashboard panels from source-controlled SQL |

## Dashboard implementation notes

Regenerate dashboard JSON after changing generated panels:

```bash
python3 mm/tools/generate_grafana_dashboards.py
python3 mm/tools/generate_observability_v2_dashboards.py
```

Use PostgreSQL dashboards for rich event payloads and Prometheus dashboards for
fast health. If a PostgreSQL panel becomes slow:

1. Add a targeted index.
2. Create a materialized view for repeated JSON extraction.
3. Move only the low-cardinality value to Prometheus if it is alert-worthy.

Typed tables already available for common dashboard paths:

- `mm_market_facts`
- `mm_action_stage_facts`
- `mm_execution_facts`
- `mm_target_order_facts`
- `mm_model_score_facts`
- `mm_optimizer_facts`
- `mm_optimizer_component_facts`

Recommended materialized views if remaining dashboard latency becomes an issue:

- `mm_controller_diagnostics_flat`
- `mm_risk_decisions_flat`
- Debug-only model/optimizer views for fields not covered by typed facts, such as full parameter dictionaries, gradients, and rollout internals.

## Example SQL snippets

Latest diagnostics for one actor:

```sql
SELECT ts, payload
FROM mm_events
WHERE topic = '${diagnostics_topic}'
  AND event_type = 'ControllerDiagnostics'
  AND actor_id = '${actor_id}'
  AND exchange = '${exchange}'
  AND symbol = '${symbol}'
ORDER BY ts DESC
LIMIT 1;
```

Controller warning counts:

```sql
SELECT
  $__timeGroupAlias(e.ts, $__interval),
  warning AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'warnings', '[]'::jsonb)) AS warning
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND $__timeFilter(e.ts)
GROUP BY 1, warning
ORDER BY 1;
```

Execution blocked reasons:

```sql
SELECT
  $__timeGroupAlias(e.ts, $__interval),
  reason AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(
  COALESCE(e.payload #> ARRAY['execution_gates','blocked_reasons'], '[]'::jsonb)
) AS reason
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND $__timeFilter(e.ts)
GROUP BY 1, reason
ORDER BY 1;
```

Action stage order count:

```sql
SELECT e.ts AS "time",
       stage AS metric,
       COALESCE((e.payload #>> ARRAY['action_pipeline', stage, 'order_count'])::double precision, 0) AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY[
  'ladder',
  'capped',
  'target_book',
  'risk_approved_target',
  'planned_place',
  'rules_valid',
  'router_accepted'
]) AS stage
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND $__timeFilter(e.ts)
ORDER BY 1;
```

## Example PromQL snippets

Frame rate:

```promql
sum by (exchange, symbol) (
  rate(mm_joint_strategy_frames_total[$__rate_interval])
)
```

Decision rate:

```promql
sum by (actor_id, exchange, symbol) (
  rate(mm_controller_decisions_total[$__rate_interval])
)
```

Order sync p95:

```promql
histogram_quantile(
  0.95,
  sum by (le, exchange) (
    rate(mm_order_sync_latency_ms_bucket[$__rate_interval])
  )
)
```

Runtime error rate:

```promql
sum by (topic, error) (
  rate(mm_runtime_errors_total[$__rate_interval])
)
```

Persistence error rate:

```promql
sum by (writer, error) (
  rate(mm_persistence_errors_total[$__rate_interval])
)
```

## Implementation roadmap

### Phase 1: correctness and missing health signals

- Fix the frame bookkeeping typo in `RuntimeObservabilitySubscriber`.
- Set a non-empty `run_id` in runtime configs used for real runs.
- Add `mm_frame_age_ms`.
- Add `mm_controller_stage_latency_ms`.
- Add `mm_execution_gate_block_total`.
- Add `mm_persistence_queue_depth`.
- Add `mm_persistence_write_latency_ms`.
- Log `observability_setup_failed` instead of silently swallowing setup failures.

### Phase 2: diagnostics-to-metrics bridge

- Teach `RuntimeObservabilitySubscriber` to consume `ControllerDiagnostics`.
- Emit metrics for model NLL, standardized residuals, model skip reasons.
- Emit metrics for action pipeline stages.
- Emit metrics for risk rejection counts and execution gate blocked reasons.
- Add tests that feed a fixed `ControllerDiagnostics` payload into the subscriber and assert metrics text output.

### Phase 3: execution lifecycle clarity

- Publish `ExecutionSyncCompleted` after successful sync rounds.
- Add execution sync latency and counts to that event.
- Count/log adapter failures by operation.
- Add dashboard panels for requested vs accepted vs failed place/cancel operations.

### Phase 4: Grafana cleanup

- Add or regenerate the 00-10 dashboard set above.
- Keep current Postgres dashboards but make names and variables consistent.
- Add a dedicated incident drilldown dashboard.
- Add dashboard notes explaining source event, labels, and expected variables.

### Phase 5: retention and scale

- Add Postgres indexes for common filters.
- Add retention or partitioning for `mm_events`.
- Consider materialized views for slow JSON-heavy panels.
- Keep Parquet for replay/offline analysis.

## Verification checklist

Before live operation:

- `docker compose up -d` starts Postgres, Prometheus, Loki, Tempo, Alloy, Grafana.
- `http://localhost:9090/targets` shows `market_maker` up.
- `http://localhost:3000` loads provisioned dashboards.
- Runtime `/metrics` exposes frame, decision, execution, and persistence metrics.
- `logs/mm-runtime.jsonl` receives JSON logs.
- Loki shows `job="market-maker-runtime"` logs.
- Postgres `mm_events` receives market, diagnostics, target book, execution, private state, and ledger events.
- Dashboard variables match runtime config:
  - `exchange`
  - `symbol`
  - `interval_ms`
  - `actor_id`
  - `diagnostics_topic`
  - `target_books_topic`
  - `ledger_topic`
- `ControllerDiagnostics` exists for every controller frame, including controller errors.
- Observe-only and kill-switch states are visible.
- No telemetry setup failures appear in logs.

## Inspecting without Grafana

Print the runtime bus:

```bash
python -m mm.tools.print_events --config mm/configs/market_maker.runtime.example.json
```

Use `--raw` for full payloads.

Check Prometheus output directly:

```bash
curl http://localhost:8000/metrics
```

If Prometheus shows `up{job="market_maker"} == 0`, query its target state:

```bash
curl http://localhost:9090/api/v1/targets
```

`connection refused` for `host.docker.internal:8000` means the runtime is not
currently serving `/metrics` on port 8000, or it failed to bind that socket.

Inspect recent persisted events:

```sql
SELECT ts, topic, event_type, exchange, symbol, actor_id
FROM mm_events
ORDER BY ts DESC
LIMIT 50;
```

## Related documents

- [Architecture](architecture.md)
- [Data flow](data-flow.md)
- [Controller and execution](controller-and-execution.md)
- [Configuration](configuration.md)
