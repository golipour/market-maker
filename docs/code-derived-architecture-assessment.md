# Code-Derived Architecture Assessment

This assessment is based on code analysis only. Existing repository documentation was not used as source of truth. After the initial review, `docs/architecture_assessment.md` was also read as a secondary expert critique; its points were incorporated only where they were supported by inspected code or sharpened the code-derived assessment.

## Executive Summary

The system has a strong domain-oriented shape for a low-frequency market maker. Its best architectural choices are:

- clear runtime composition in one builder,
- event-based component interaction,
- exchange adapters isolated behind interfaces,
- ledger-backed strategy input,
- declarative target books instead of imperative order commands,
- pre-execution target-book risk,
- rich controller diagnostics,
- a practical local observability stack.

The main architectural risks are:

- the runtime is still effectively a single in-process bot orchestration model,
- private state, ledger, execution, and persistence duplicate payloads heavily,
- Postgres dashboards depend on expensive JSONB queries over large event payloads,
- Parquet persistence stores JSON strings rather than typed columnar facts,
- the MPC path can become CPU/GPU expensive as optimizer settings scale,
- execution has a visible pending-confirmation window between router acceptance and private exchange truth,
- Prometheus alert rules exist, but notification routing is not provisioned,
- some state wiring has edge-case gaps.

## Strengths

### Strong Separation Of Domain Concerns

Severity: Positive

The code separates public market data, private state, ledger, strategy, risk, execution, persistence, and observability into distinct packages. Exchange-specific behavior is mostly constrained to `mm/exchanges/*`, while domain models live under `mm/core/models`.

Impact: This makes it easier to add an exchange or change execution planning without rewriting the MPC engine.

Recommendation: Preserve this structure. Future features should enter through existing interfaces: market data adapter, private state adapter, order command adapter, risk engine, execution planner, and persistence writer.

### Declarative Target-Book Execution

Severity: Positive

The strategy emits target books and the execution manager diffs target state against live state. This is much safer than emitting raw create/cancel commands from the optimizer.

Impact: The system can recover from partial fills, missed cancels, and restarts by reconciling toward desired state.

Recommendation: Continue to treat target books as the central execution contract.

### Ledger-Centered Risk

Severity: Positive

Risk runs against ledger state and validates whole target-book transitions. It considers available balances, locked collateral, open notional, min/max inventory, and private/market freshness.

Impact: This is the correct layer for hard pre-trade controls because it avoids the common mistake of validating each order in isolation.

Recommendation: Extend the risk engine with more explicit per-venue and consolidated portfolio limits rather than moving risk into the optimizer.

### Rich Diagnostics

Severity: Positive

The controller publishes stage timings, model scores, optimizer objective snapshots, action pipeline stages, risk outputs, venue gates, warnings, and execution funnel state.

Impact: The system is far more debuggable than a black-box strategy loop.

Recommendation: Convert the most important diagnostics into typed, versioned schemas and materialized metrics/tables.

## Significant Issues And Recommendations

### 1. Event Payload Duplication Will Grow Quickly

Severity: High

A single private REST poll can publish:

- a full private snapshot,
- a ledger venue snapshot,
- a joint ledger snapshot,
- execution state snapshots,
- diagnostics that embed action/risk/execution stage orders.

REST private polling now publishes one authoritative private snapshot per poll instead of fanning the same snapshot out again as per-balance and per-open-order events. Incremental balance/open-order topics remain for stream-style adapters and real deltas. Market events on the live bus embed the full orderbook snapshot and all interval trades. Persistence now compacts some already-summarized content before saving: market trades are replaced by trade count/timestamps while retaining interval features, and persisted joint frames replace embedded ledger detail with a summary/reference because `ledger:snapshot` is persisted separately. Strategy decisions, model parameters, optimizer action snapshots, and stage reports still repeat related decision context in multiple durable events.

Persisted controller diagnostics now also compact repeated `action_pipeline` order and rejection arrays into counts, side counts, stable digests, and small samples. This keeps live in-process diagnostics rich for runtime debugging while preventing every durable `ControllerDiagnostics` row from storing the same stage-order payloads already represented by target-book, decision, risk, and execution events. Postgres persistence materializes these action-pipeline summaries into `mm_action_stage_facts`, so common stage dashboards no longer need to walk diagnostics JSONB. Execution lifecycle events are materialized into `mm_execution_facts`, including confirmed versus pending state kind. Target-book intent orders are materialized into `mm_target_order_facts`. Model score and optimizer objective/timing/component facts are also materialized into typed tables. Remaining durable duplication is concentrated in strategy decisions, full optimizer debug payloads, full orderbook snapshots, and opaque JSON payload storage.

Event serialization also recursively converts rich Python objects through `to_jsonable()`, including dataclasses, msgspec structs, dicts, lists, NumPy scalars, and arrays. That keeps publication flexible, but it means deep object graphs are repeatedly normalized into JSON-compatible structures on bus and persistence paths.

Impact: Postgres and Parquet storage growth is lower after persistence compaction, but will still be dominated by repeated JSON payloads as volume grows. Dashboard queries over `mm_events.payload` will slow down as data grows. Bandwidth and memory pressure increase with every new subscriber because in-memory bus fan-out copies large event objects into queues.

Recommendation:

- Split event classes into facts, snapshots, and diagnostics.
- Persist large snapshots at lower frequency or only by explicit topic selection.
- Store orderbooks and raw trades in normalized append tables, compressed columnar partitions, or a dedicated time-series store when raw market history is needed.
- Store diagnostics in typed tables for common panels.
- Use references for repeated structures. The runtime now carries `frame_id`, `decision_id`, `target_book_id`, and per-venue `execution_plan_id`, and replay contract tests verify those links end to end.
- Use the existing `mm_event_payload_*` and `mm_event_envelope_*` metrics to rank topics by bus and persistence byte cost before choosing compaction targets.
- Add retention policies per topic.

### 2. Postgres Persistence Spool Has Recovery Tooling But Needs Operations Policy

Severity: Medium

`PostgresEventWriter` now keeps a reusable psycopg2 connection across batches, drops/reconnects after write or readiness failures, retries failed batches with bounded backoff, and spills exhausted batches to local JSONL files.

`mm.tools.reingest_postgres_spool` can now scan failed JSONL spool records, decode their original envelopes, write them through the same Postgres writer path, and archive successfully reingested files. Reingest is idempotent because persisted events and typed fact tables use conflict keys.

Impact: Connection churn and immediate batch loss are no longer the main bottlenecks. During sustained database outages, events are preserved locally and recoverable after Postgres returns. The remaining risk is operational: retention, cleanup, and runbook ownership need to be explicit so spool directories do not grow silently.

Recommendation:

- Add retention and cleanup rules for failed and successfully reingested spool files.
- Document when operators should run spool reingest and how to verify row counts.
- Add metrics for reconnect count, connection errors, write rows/sec, and commit latency.

### 3. Parquet Storage Still Needs Columnar Lifecycle Policies

Severity: Medium

`ParquetEventWriter` now writes immutable part files and no longer reads, concatenates, and rewrites the existing hourly Parquet file during live writes.

The row schema now stores one compact `envelope_json` instead of both `payload_json` and `envelope_json`. For archival Parquet rows it compacts data that has already been summarized or integrated: `MarketEvent.trades` are replaced by trade counts/timestamps while retaining interval features such as buy/sell volume and VWAP, and zero private balances are removed while retaining balance counts. Parquet rows also include typed nullable columns for high-value market, controller, optimizer, target-book, and execution fields, so common offline scans no longer need to deserialize JSON.

Impact: Live archival writes are now append-like and object-store friendly, archival byte volume is lower than before, and common analytics can use typed columns. Remaining scaling risks are small-file growth, raw orderbook/debug payload storage, and the fact that long-form facts such as per-objective components are still better represented by separate datasets when scan volume grows.

Recommendation:

- Compact offline if needed.
- Partition by date/hour/topic/event_type/exchange/symbol.
- Add separate columnar datasets for very large raw snapshots or long-form facts when typed envelope columns are not enough.

### 4. Strategy Frame Assembly Can Lose Or Delay Global Alignment

Severity: Medium

The global frame assembler keeps a bounded deque and waits up to `global_grace_ms` for exact matching `t_end_ms`. It selects the latest global event with `t_end_ms <= local.t_end_ms`.

Impact: At low frequency this is simple and reasonable. At scale, jitter, missed intervals, or scheduler delays can create persistent stale/missing global status. The bounded history can also drop relevant global events when event rates or delays increase.

Recommendation:

- Add explicit sequence/window ids for interval events.
- Track global alignment hit rate and lag distribution.
- Make history size and grace policy configurable.
- Add backpressure or skip policies for slow global feeds.

### 5. Scheduler Awaits Callbacks Sequentially

Severity: Medium

`IntervalScheduler.run()` awaits each registered callback before advancing. If one callback stalls, all interval tasks behind it stall.

Impact: More symbols, venues, or expensive callbacks will increase timing skew and missed intervals.

Recommendation:

- Run interval callbacks concurrently with bounded concurrency.
- Record callback execution time and missed-deadline count.
- Consider one scheduler per symbol/venue or a timing wheel with independent tasks.

### 6. Optimizer Runtime Cost Can Dominate The Decision Loop

Severity: High for larger configs, Medium for current defaults

The MPC path scales with `adam_epochs * horizon * n_samples * grid_levels`, and joint optimization roughly doubles the state/action work. Decision compute now has a configured timeout and hold-current-orders fallback, but the underlying optimizer work can still continue in a background thread after a timeout.

The inspected strategy code also captures optimizer stdout through `contextlib.redirect_stdout()` while calling the optimizer. That is useful for diagnostics, but it is another sign that optimizer execution is treated as an inline blocking step rather than an isolated compute service.

Impact: Larger configs can exceed the decision interval. If optimization stalls, decisions become stale and downstream execution/risk may reject. A CPU-heavy optimizer can also starve other asyncio tasks.

Recommendation:

- Export optimizer total time vs interval budget as a first-class SLO.
- Monitor `ControllerFallback` timeout rate and tune `decision_timeout_ms` per strategy interval.
- Move from thread-based isolation to a dedicated worker process if CPU-bound optimizer work still starves the runtime.
- Cache or warm-start carefully, with diagnostics for stale warm starts.
- Add load tests for representative `n_samples`, `horizon`, `adam_epochs`, and symbol counts.

### 7. Alert Notification Routing Is Not Provisioned

Severity: Medium to High operational risk

Prometheus alert rules are now provisioned in `docker/prometheus/market_maker_alerts.yml` and mounted by the local Compose stack. They cover target down, stale market data, high ingest/frame lag, stale private/ledger readiness, execution sync failures, router failures, stale pending order confirmations, validation rejection spikes, persistence errors, kill switch, and observe-only state.

Impact: Prometheus can evaluate and display firing alerts, but no Alertmanager, Grafana contact point, notification policy, escalation route, or on-call receiver is provisioned. Operators still need to watch Prometheus/Grafana manually unless they add local notification routing.

Recommendation:

- Add Alertmanager or Grafana-managed contact points and notification policies.
- Route critical execution, market-data, persistence, and kill-switch alerts to an operator channel.
- Add log-derived Loki alerts for events that are not yet Prometheus metrics, such as sustained `LedgerOrderManagerMismatch` events and Loki ingestion failures.
- Tune thresholds per deployment interval and venue.

### 8. Observability Depends Too Heavily On JSONB Event Queries

Severity: Medium

Most Grafana panels query `mm_events.payload` with JSONB extraction, lateral JSON expansion, regex numeric casts, and generated time buckets.

Postgres persistence now creates typed side tables for common dashboard facts. `mm_market_facts` materializes `MarketEvent` interval data into normal columns for top-of-book, mid, spread, microprice, VWAP, interval flow, imbalance, trade count, depth level counts, and timestamps. `mm_action_stage_facts` materializes `ControllerDiagnostics.action_pipeline` stage summaries into normal columns for counts, side counts, quantities, notionals, rejection counts, compacted counts, order/rejection digests, and lineage IDs. `mm_execution_facts` materializes sync, place, cancel, validation rejection, fill, and execution snapshot events into normal columns for lifecycle, success, confirmed/pending state, order ids, quantities, fees, status, error, latency, and lineage IDs. `mm_target_order_facts` materializes `JointTargetBook` order intent into normal columns for actor, mode, venue role, exchange, symbol, policy, side, price, quantity, notional, order type, tags, and lineage IDs. `mm_model_score_facts` materializes model quality and update-skip facts. `mm_optimizer_facts` and `mm_optimizer_component_facts` materialize optimizer timing, objective values, improvement, best expected reward/return fields, objective components, and lineage IDs. `mm_events` remains the replay/debug event log.

There are also two dashboard generations: older `Market Maker` dashboards and a newer `MM Observability v2` set. That breadth is helpful for local debugging, but can become dashboard sprawl without a stronger fleet/drilldown hierarchy and shared template variables.

Impact: The market-data, action-pipeline, execution-lifecycle, target-order, model-quality, and optimizer-objective dashboard paths can now avoid repeated JSONB extraction for common panels. The remaining expensive paths are less common debugging panels that inspect full parameter dictionaries, gradients, rollout internals, and raw payload details. Operators may also have too many narrowly focused dashboards to navigate during incidents.

Recommendation:

- Materialize common metrics to typed tables or hypertables.
- Add generated columns for common JSON paths.
- Add indexes for event_type plus common fields.
- Use Prometheus for high-cardinality operational counters only where labels are bounded.
- Use Postgres for lower-cardinality business analytics and latest state.
- Keep Grafana organized around fleet overview, drilldown dashboards, and incident dashboards with shared `$run_id`, `$exchange`, and `$symbol` variables.

### 9. Custom Metrics Registry Has Limited Production Features

Severity: Medium

The custom Prometheus registry supports gauges, counters, histograms, and text rendering. It lacks standard client features such as process metrics, multiprocess support, default collectors, exemplars, and stale series lifecycle controls.

Impact: It is sufficient for a single local process, but it may underperform for production multi-process deployments and makes integration less standard.

Recommendation:

- Consider migrating to `prometheus_client`.
- Add process CPU, memory, open file descriptors, thread count, and event-loop lag.
- Add explicit stale-series zeroing for label sets that disappear.

### 10. Event Bus And Persistence Backpressure Are Underdeveloped

Severity: Medium

In-memory bus publishes by awaiting each subscriber queue put. The default queue size is unbounded unless configured. Persistence queues are bounded, but consumers block publishing when queues fill.

Impact: A slow subscriber can increase latency or memory use. High event volume can block core runtime paths.

Recommendation:

- Define per-topic backpressure policy: block, drop, sample, or spill.
- Add subscriber lag and queue-depth metrics per topic.
- Separate critical trading paths from observability/persistence paths.
- Use bounded queues by default in production.

### 11. Runtime Scaling Is One Bot Per Process

Severity: Medium

Config resolution explicitly requires one bot entry per config file. The composition root builds a local/global pair for that bot.

Impact: Scaling to many symbols or strategies means many processes/configs. That can be operationally fine, but shared resources, admin control, persistence, and observability need process-level coordination.

Recommendation:

- Make one-bot-per-process an explicit deployment model.
- Add run_id, bot_id, env, and instance labels consistently.
- Provide process supervisor templates.
- If multi-bot in one process is desired, refactor runtime components around bot instances and shared adapter pools.
- If many symbols share a venue and model family, evaluate multi-symbol batching only after measuring process-level memory, adapter connection, and optimizer costs.

### 12. Exchange Capabilities And Registration Still Split

Severity: Low

Adapter construction now goes through `mm/exchanges/registry.py` with one `registration.py` per exchange package. `mm/exchanges/factory.py` only resolves capabilities, looks up the registration, and builds the adapter.

Impact: Adding an exchange still requires a new package registration module, capability metadata in `capabilities.py`, wire files, and adapters. The factory if/elif chains are gone.

Recommendation:

- Keep capability metadata and adapter registration co-located per exchange when new venues are added.
- Consider merging capability declarations into each exchange `registration.py` if the split becomes annoying.

### 13. Remaining Event Payloads Still Mix Dicts And Dataclasses

Severity: Low to Medium

Execution lifecycle/snapshot events, joint target books, `StrategyDecision`, and `ControllerDiagnostics` now publish through versioned builders in `mm/core/payload_schema.py` (`payload_schema` + `payload_version` on every payload). Replay fidelity checks cover execution lifecycle coherence.

Impact: The highest-volume controller and decision events now share the same schema contract as execution and target-book events. Market events, private snapshots, and ledger snapshots still use mixed dataclass/dict shapes without dedicated builders.

Recommendation:

- Add typed payload builders for private snapshots and ledger snapshots next.
- Add replay/persistence contract tests as each remaining event family is migrated.

### 14. Secrets And Local Compose Defaults Need Production Hardening

Severity: Medium for production, Low for local development

Docker Compose includes fixed Postgres credentials for local use, and `.env` loading is repository-root based.

Impact: Fine for local development, unsafe if copied into production. Accidental log or config exposure could reveal credentials.

Recommendation:

- Document that Compose credentials are local-only.
- Use secret managers or environment injection in production.
- Ensure config contract output never includes credential values.
- Add CI checks to avoid committing real `.env`.

## Observability Assessment

### Current Coverage

Trading and business metrics:

- controller decisions,
- order counts,
- inventory and cash,
- local/global VWAPs in persisted diagnostics,
- action pipeline stages,
- risk approval/rejection,
- target books and ladder depth,
- fills and execution failures,
- portfolio and PnL dashboards.

System/runtime metrics:

- raw market data event lag and gaps,
- frame sync readiness,
- controller/optimizer timing,
- order sync latency,
- persistence queue depth and write latency,
- runtime errors from observed event topics.

Logs and traces:

- structured JSON event logs,
- Loki ingestion through Alloy,
- optional OTLP tracing to Tempo,
- limited span usage around controller decision observation.

### Gaps

- No Alertmanager/contact-point notification routing.
- No Loki alert rules for log-only failure modes.
- No event-loop lag metric.
- No process CPU/memory/thread/file-descriptor metrics.
- No external exchange API request latency/success metrics per endpoint.
- No websocket reconnect counters per exchange.
- No REST polling success/failure latency histograms by adapter endpoint.
- No order adapter rate-limit metrics.
- No subscriber queue lag by topic beyond writer queue depth.
- No database table size, write throughput, or dashboard query latency metrics.
- No clear SLO dashboards for "decision freshness", "execution freshness", or "data freshness".
- No explicit trace spans for full frame -> decision -> risk -> execution lifecycle.

### Recommended Dashboard Organization

Use an operational hierarchy:

1. Fleet overview: process health, bot state, data freshness, live/observe state, kill switch.
2. Market data: raw event gaps, ingest lag, market-event cadence, book health.
3. Controller: frame age, optimizer latency, objective health, model quality.
4. Risk: reject reasons, exposures, freshness gates.
5. Execution: target -> planned -> rules -> accepted funnel, sync latency, adapter failures.
6. Portfolio: balances, inventory, PnL, fills.
7. Persistence: queue depth, write latency, event volume, DB health.
8. Incident: recent warnings/errors, latest target books, latest live/private state, control events.

## Event And Data Efficiency Assessment

### Ownership

Current event ownership is mostly clear:

- market events are owned by ingestion,
- private events by private reconcilers,
- ledger snapshots by ledger service,
- strategy decisions and diagnostics by strategy runner,
- execution lifecycle by order manager/router,
- telemetry by observability subscriber,
- durable envelopes by persistence writers.

### Redundancy

The main redundancy patterns are:

- private snapshots plus incremental balance/open-order event topics for stream-style adapters,
- ledger snapshots derived from private snapshots,
- execution snapshots derived from private/open-order state,
- strategy decisions, target-book events, and remaining diagnostics repeating target/order intent,
- market events embedding full books and trades every interval,
- dashboards recomputing derived metrics from JSON payloads.

### Growth Risks

At one bot and a low interval, this is workable. At many symbols, venues, and strategies:

- Postgres JSONB rows will become large and numerous,
- dashboards will perform repeated JSON extraction over large tables,
- in-memory subscribers will hold large Python objects,
- persistence writers will block or lag,
- Parquet appends will become increasingly expensive,
- Redis Streams max length may drop historical events before consumers read them.

### Efficiency Improvements

Priority recommendations:

1. Define topic retention and persistence tiering.
2. Use payload and envelope byte metrics to rank high-cost topics and event types.
3. Store large orderbook/trade data separately from controller events.
4. Replace repeated snapshots with deltas plus periodic snapshots.
5. Continue materializing dashboard facts into typed tables; current coverage includes market intervals, action-pipeline stages, order lifecycle, target-book intent, model scores, optimizer objectives, and optimizer objective components.
6. Move high-frequency operational metrics and market-data time series to Prometheus, a dedicated time-series database, or typed columnar storage instead of Postgres JSONB.
7. Keep full diagnostics configurable and sampled in production.

## Prioritized Roadmap

### Priority 1: Scalability

- Add per-topic subscriber queue depth and lag metrics.
- Evaluate a time-series database alongside SQL for high-volume market data, execution latency, and low-cardinality business metrics.
- Add retention and compaction policies.

### Priority 2: Engine And Execution Quality

- Add load tests for optimizer settings and multi-symbol runtime.
- Improve global frame alignment metrics and policy.

### Priority 3: Maintainability And DX

- Add typed payload builders for private snapshots and ledger snapshots.
- Generate dashboard queries from stable schema helpers where possible.
- Document one-bot-per-process deployment explicitly.
- Extend replay fidelity into full frame re-simulation through strategy/risk/execution when a deterministic offline harness is needed.
