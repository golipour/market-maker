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

REST private polling now publishes one authoritative private snapshot per poll instead of fanning the same snapshot out again as per-balance and per-open-order events. Incremental balance/open-order topics remain for stream-style adapters and real deltas. Market events on the live bus embed the full orderbook snapshot and all interval trades. Persistence now compacts some already-summarized content before saving: market trades are replaced by trade count/timestamps while retaining interval features, and persisted joint frames replace embedded ledger detail with a summary/reference because `ledger:snapshot` is persisted separately. Strategy decisions and diagnostics still repeat orders, target books, model parameters, optimizer action snapshots, and stage reports.

Event serialization also recursively converts rich Python objects through `to_jsonable()`, including dataclasses, msgspec structs, dicts, lists, NumPy scalars, and arrays. That keeps publication flexible, but it means deep object graphs are repeatedly normalized into JSON-compatible structures on bus and persistence paths.

Impact: Postgres and Parquet storage growth is lower after persistence compaction, but will still be dominated by repeated JSON payloads as volume grows. Dashboard queries over `mm_events.payload` will slow down as data grows. Bandwidth and memory pressure increase with every new subscriber because in-memory bus fan-out copies large event objects into queues.

Recommendation:

- Split event classes into facts, snapshots, and diagnostics.
- Persist large snapshots at lower frequency or only by explicit topic selection.
- Store orderbooks and raw trades in normalized append tables, compressed columnar partitions, or a dedicated time-series store when raw market history is needed.
- Store diagnostics in typed tables for common panels.
- Use references for repeated structures, for example `frame_id`, `decision_id`, `target_book_id`.
- Use the existing `mm_event_payload_*` and `mm_event_envelope_*` metrics to rank topics by bus and persistence byte cost before choosing compaction targets.
- Add retention policies per topic.

### 2. Postgres Persistence Still Has Best-Effort Retry Semantics

Severity: Medium

`PostgresEventWriter` now keeps a reusable psycopg2 connection across batches and drops/reconnects after write or readiness failures.

Impact: Connection churn is no longer the main bottleneck. The remaining issue is that failed write batches are logged as retryable but are not requeued with a durable retry policy. During sustained database outages, persistence can still lose archived events even though trading continues.

Recommendation:

- Track failed batch retry behavior explicitly.
- Add bounded retry/backoff or a local durable spill queue for failed batches.
- Add metrics for reconnect count, connection errors, write rows/sec, and commit latency.

### 3. Parquet Storage Still Uses Opaque JSON Envelopes

Severity: Medium

`ParquetEventWriter` now writes immutable part files and no longer reads, concatenates, and rewrites the existing hourly Parquet file during live writes.

The row schema now stores one compact `envelope_json` instead of both `payload_json` and `envelope_json`. For archival Parquet rows it compacts data that has already been summarized or integrated: `MarketEvent.trades` are replaced by trade counts/timestamps while retaining interval features such as buy/sell volume and VWAP, and zero private balances are removed while retaining balance counts. The remaining issue is that the retained payload is still an opaque JSON string, which gives up most of Parquet's typed columnar benefits for analytics.

Impact: Live archival writes are now append-like and object-store friendly, and archival byte volume is much lower than before. Analysts still have to deserialize JSON at query time instead of filtering typed columns, and many small part files may need periodic offline compaction.

Recommendation:

- Compact offline if needed.
- Partition by date/hour/topic/event_type/exchange/symbol.
- For high-value event types, write typed Parquet columns instead of opaque JSON strings.

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

There are also two dashboard generations: older `Market Maker` dashboards and a newer `MM Observability v2` set. That breadth is helpful for local debugging, but can become dashboard sprawl without a stronger fleet/drilldown hierarchy and shared template variables.

Impact: This is flexible but expensive. As event volume grows, dashboards will become slow unless retention is short or indexes/materialized views are added. Operators may also have too many narrowly focused dashboards to navigate during incidents.

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

### 12. Exchange Capability And Factory Code Is Centralized

Severity: Low to Medium

Adding an exchange requires edits in `mm/exchanges/factory.py`, `capabilities.py`, wire files, adapters, and likely trade-stats sources.

Impact: This is manageable for four exchanges but becomes a maintenance bottleneck as exchanges grow.

Recommendation:

- Introduce a registry pattern where exchange packages self-register adapters and capabilities.
- Keep factory validation, but avoid long if/elif chains.

### 13. No Clear Schema Versioning Beyond Envelope Version

Severity: Medium

The envelope has `schema` and `version`, but individual payloads are mostly dataclasses or dicts without independent schema versions.

Impact: Dashboard SQL and replay readers can break when payload fields change. JSONB queries already include defensive casts, indicating payload instability.

Recommendation:

- Add event-specific schema versions.
- Define typed payload schemas for high-value events.
- Add migration tests for dashboards and replay.
- Keep old fields until dashboards are migrated.

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

- private snapshots plus balance/open-order event fan-out,
- ledger snapshots derived from private snapshots,
- execution snapshots derived from private/open-order state,
- strategy decisions plus controller diagnostics plus target-book events repeating order information,
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
5. Materialize latest state and dashboard facts into typed tables.
6. Use immutable Parquet part files.
7. Move high-frequency operational metrics and market-data time series to Prometheus, a dedicated time-series database, or typed columnar storage instead of Postgres JSONB.
8. Keep full diagnostics configurable and sampled in production.
9. Add event ids and references between frame, decision, target book, risk result, and execution report.
10. Replace opaque Parquet JSON payloads with typed columns for high-value event types.
11. Add replay contracts that verify the reduced event model can reconstruct required state.

## Prioritized Roadmap

### Priority 1: Scalability

- Add per-topic subscriber queue depth and lag metrics.
- Convert Parquet writer to immutable part files.
- Replace opaque Parquet JSON payloads with typed columns for high-value event types.
- Evaluate a time-series database alongside SQL for high-volume market data, execution latency, and low-cardinality business metrics.
- Reuse Postgres connections or use a pool.
- Materialize common dashboard facts out of JSONB.
- Add retention and compaction policies.

### Priority 2: Engine And Execution Quality

- Add load tests for optimizer settings and multi-symbol runtime.
- Improve global frame alignment metrics and policy.

### Priority 3: Maintainability And DX

- Add event-specific schema versions.
- Introduce an exchange adapter registry.
- Generate dashboard queries from stable schema helpers where possible.
- Document one-bot-per-process deployment explicitly.
- Add end-to-end replay/simulation tests around frame -> decision -> risk -> execution.
