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
- controller diagnostics and execution state are gathered through a broad mutable context object,
- execution has an unavoidable window between optimistic order-manager state and private exchange truth,
- alerting is not provisioned,
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
- each balance,
- each open order,
- a ledger venue snapshot,
- a joint ledger snapshot,
- execution state snapshots,
- diagnostics that embed action/risk/execution stage orders.

Market events embed the full orderbook snapshot and all interval trades. Strategy decisions and diagnostics also repeat orders, target books, model parameters, optimizer action snapshots, and stage reports.

Event serialization also recursively converts rich Python objects through `to_jsonable()`, including dataclasses, msgspec structs, dicts, lists, NumPy scalars, and arrays. That keeps publication flexible, but it means deep object graphs are repeatedly normalized into JSON-compatible structures on bus and persistence paths.

Impact: Postgres and Parquet storage growth will be dominated by repeated JSON payloads. Dashboard queries over `mm_events.payload` will slow down as data grows. Bandwidth and memory pressure increase with every new subscriber because in-memory bus fan-out copies large event objects into queues.

Recommendation:

- Split event classes into facts, snapshots, and diagnostics.
- Persist large snapshots at lower frequency or only by explicit topic selection.
- Store orderbooks and trades in normalized append tables or compressed columnar partitions.
- Store diagnostics in typed tables for common panels.
- Use references for repeated structures, for example `frame_id`, `decision_id`, `target_book_id`.
- Add payload-size metrics per event type and topic.
- Add retention policies per topic.

### 2. Postgres Persistence Opens A New Connection Per Batch

Severity: Medium

`PostgresEventWriter._write_batch()` calls `psycopg2.connect()` for every batch.

Impact: At low frequency this works, but at larger event volume it adds connection overhead and can exhaust database connection limits. Connection churn also makes transient database failures more disruptive.

Recommendation:

- Use a persistent connection with reconnect logic or a small connection pool.
- Track failed batch retry behavior explicitly.
- Add metrics for reconnect count, connection errors, write rows/sec, and commit latency.

### 3. Parquet Writer Rewrites Existing Files

Severity: Medium

`ParquetEventWriter` reads an existing Parquet file, concatenates it with the new batch, and writes a replacement file.

The row schema also stores both `payload_json` and `envelope_json`. Since the envelope includes the payload, this duplicates the largest field. Both are stored as strings, which gives up most of Parquet's typed columnar benefits for analytics.

Impact: This is acceptable for small local files, but it is not scalable for high event volume. Appending to the same hourly event-type file causes repeated read/modify/write cycles and makes object-store deployment impractical. Analysts also have to deserialize JSON at query time instead of filtering typed columns.

Recommendation:

- Write immutable chunk files, for example `part-<timestamp>-<uuid>.parquet`.
- Compact offline if needed.
- Partition by date/hour/topic/event_type/exchange/symbol.
- Avoid reading existing Parquet during live writes.
- Drop redundant `payload_json` if full envelopes are retained, or retain only metadata columns plus a typed payload table.
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

The MPC path scales with `adam_epochs * horizon * n_samples * grid_levels`, and joint optimization roughly doubles the state/action work. The controller runs synchronously inside the async strategy loop.

The inspected strategy code also captures optimizer stdout through `contextlib.redirect_stdout()` while calling the optimizer. That is useful for diagnostics, but it is another sign that optimizer execution is treated as an inline blocking step rather than an isolated compute service.

Impact: Larger configs can exceed the decision interval. If optimization stalls, decisions become stale and downstream execution/risk may reject. A CPU-heavy optimizer can also starve other asyncio tasks.

Recommendation:

- Export optimizer total time vs interval budget as a first-class SLO.
- Add a hard decision timeout and safe fallback action.
- Run the optimizer in a worker process or thread pool if CPU-bound.
- Cache or warm-start carefully, with diagnostics for stale warm starts.
- Add load tests for representative `n_samples`, `horizon`, `adam_epochs`, and symbol counts.

### 7. MPC And Controller Hot Path Are Tightly Coupled

Severity: Medium

`MpcStrategyEngine` directly constructs and calls the PyTorch-backed optimizer implementation. `DecisionContext` gathers frame state, clean frame state, trading state, ledger, timings, warnings, proposed/approved actions, execution gates, risk counters, and action-pipeline diagnostics across one strategy invocation.

Impact: For the current low-frequency, single-bot runtime this is understandable and practical. As the controller grows, the broad mutable context can become hard to reason about and easy to mutate incorrectly. Direct strategy-to-PyTorch coupling also makes it harder to swap optimizer implementations, run the optimizer out of process, or unit-test the decision lifecycle separately from the math backend.

Recommendation:

- Introduce a small optimizer backend interface around `optimize(state)` and diagnostics retrieval.
- Split `DecisionContext` into narrower records, for example frame diagnostics, decision diagnostics, risk diagnostics, and execution diagnostics.
- Keep a final assembler for `ControllerDiagnostics`, but avoid using one mutable object as the main lifecycle carrier.
- Add tests that exercise the strategy runner with a fake optimizer backend.

### 8. Threshold Penalty Config Appears Ineffective In Single-Venue Optimizer Path

Severity: Medium

The optimizer parses `inventory_threshold_penalty_weight`, but `terminal_objective()` is called with `inventory_threshold_penalty_weight=0.0` in the inspected single-venue path.

Impact: Operators may configure threshold penalties expecting optimizer behavior changes that do not occur. Hard risk still applies, but soft inventory/cash shaping may be absent.

Recommendation:

- Either wire the configured weight into `terminal_objective()` or remove/mark the config as no-op.
- Add tests proving objective components change when the configured penalty changes.
- Surface no-op config fields in runtime validation.

### 9. Strategy-Side Venue Quantization Stage Is Mostly Diagnostic Dead Weight

Severity: Low to Medium

`apply_venue_constraints_with_report()` exists, but `build_orders_from_sigmoid_with_report()` currently builds ladder orders, caps them, and leaves `venue_quantized` empty. Actual rounding and validation happens later in `ExchangeRulesGateway`.

Impact: The action pipeline has a `venue_quantized` stage that can mislead dashboard users. Operators may expect strategy-side min-size drops, while drops happen at execution rules stage.

Recommendation:

- Rename the diagnostics stage or remove it from the strategy pipeline.
- If pre-execution quantization should be visible earlier, populate it consistently.
- Keep final validation at the execution boundary regardless.

### 10. Own-Order Subtraction Uses Exact Float Price Matching

Severity: Medium

`subtract_own_orders()` subtracts open orders by exact float price keys in the book. Order adapters may round prices at execution, while strategy frames use float target prices.

Impact: Own orders may fail to subtract if live order price differs from public book level by rounding or representation. That can bias the clean book and cause self-referential quote decisions.

Recommendation:

- Subtract using venue tick-aware price normalization or a tolerance.
- Use exchange-confirmed live order prices after adapter normalization.
- Add tests for rounded prices and cross-venue unit scaling.

### 11. Optimistic OrderManager State Can Diverge From Ledger Truth

Severity: Medium

`OrderManager` updates `_live_orders` optimistically after router results. Private state reconciliation later applies exchange-confirmed state. The ledger also maintains its own state and emits consistency diagnostics.

Impact: During the gap, own-order subtraction, planning, and diagnostics can see inconsistent live orders. Partial failures in cancel/place calls can widen the gap.

Recommendation:

- Make the ledger the single authoritative live-order source for planning where feasible.
- Track optimistic vs confirmed state separately.
- Add explicit pending order/cancel states with timeouts.
- Alert on ledger/order-manager mismatch.

### 12. Alerting Is Not Provisioned

Severity: High operational risk

Grafana dashboards exist, but no Grafana alert rules, Prometheus alert rules, or Alertmanager configuration were found.

Impact: Operators must watch dashboards manually. Critical issues such as stale market data, stale private state, execution failures, high reject rates, persistence failures, and kill-switch state may go unnoticed.

Recommendation:

Add alert rules for:

- no market events for a configured venue/symbol,
- raw market ingest lag above threshold,
- frame age above threshold,
- stale or missing private state,
- controller decisions missing,
- optimizer total time near interval budget,
- risk rejects above baseline,
- execution sync failures,
- router adapter failures,
- order validation rejection spikes,
- ledger/order-manager mismatch,
- persistence write failures or queue growth,
- Prometheus target down,
- Loki log ingestion stopped,
- Postgres write latency high,
- kill switch enabled,
- unexpected observe-only/live state.

### 13. Observability Depends Too Heavily On JSONB Event Queries

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

### 14. Custom Metrics Registry Has Limited Production Features

Severity: Medium

The custom Prometheus registry supports gauges, counters, histograms, and text rendering. It lacks standard client features such as process metrics, multiprocess support, default collectors, exemplars, and stale series lifecycle controls.

Impact: It is sufficient for a single local process, but it may underperform for production multi-process deployments and makes integration less standard.

Recommendation:

- Consider migrating to `prometheus_client`.
- Add process CPU, memory, open file descriptors, thread count, and event-loop lag.
- Add explicit stale-series zeroing for label sets that disappear.

### 15. Event Bus And Persistence Backpressure Are Underdeveloped

Severity: Medium

In-memory bus publishes by awaiting each subscriber queue put. The default queue size is unbounded unless configured. Persistence queues are bounded, but consumers block publishing when queues fill.

Impact: A slow subscriber can increase latency or memory use. High event volume can block core runtime paths.

Recommendation:

- Define per-topic backpressure policy: block, drop, sample, or spill.
- Add subscriber lag and queue-depth metrics per topic.
- Separate critical trading paths from observability/persistence paths.
- Use bounded queues by default in production.

### 16. Runtime Scaling Is One Bot Per Process

Severity: Medium

Config resolution explicitly requires one bot entry per config file. The composition root builds a local/global pair for that bot.

Impact: Scaling to many symbols or strategies means many processes/configs. That can be operationally fine, but shared resources, admin control, persistence, and observability need process-level coordination.

Recommendation:

- Make one-bot-per-process an explicit deployment model.
- Add run_id, bot_id, env, and instance labels consistently.
- Provide process supervisor templates.
- If multi-bot in one process is desired, refactor runtime components around bot instances and shared adapter pools.
- If many symbols share a venue and model family, evaluate multi-symbol batching only after measuring process-level memory, adapter connection, and optimizer costs.

### 17. Exchange Capability And Factory Code Is Centralized

Severity: Low to Medium

Adding an exchange requires edits in `mm/exchanges/factory.py`, `capabilities.py`, wire files, adapters, and likely trade-stats sources.

Impact: This is manageable for four exchanges but becomes a maintenance bottleneck as exchanges grow.

Recommendation:

- Introduce a registry pattern where exchange packages self-register adapters and capabilities.
- Keep factory validation, but avoid long if/elif chains.

### 18. No Clear Schema Versioning Beyond Envelope Version

Severity: Medium

The envelope has `schema` and `version`, but individual payloads are mostly dataclasses or dicts without independent schema versions.

Impact: Dashboard SQL and replay readers can break when payload fields change. JSONB queries already include defensive casts, indicating payload instability.

Recommendation:

- Add event-specific schema versions.
- Define typed payload schemas for high-value events.
- Add migration tests for dashboards and replay.
- Keep old fields until dashboards are migrated.

### 19. Secrets And Local Compose Defaults Need Production Hardening

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

- No alert rules.
- No event-loop lag metric.
- No process CPU/memory/thread/file-descriptor metrics.
- No external exchange API request latency/success metrics per endpoint.
- No websocket reconnect counters per exchange.
- No REST polling success/failure latency histograms by adapter endpoint.
- No order adapter rate-limit metrics.
- No subscriber queue lag by topic.
- No persisted payload size or event volume metrics by topic/event type.
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
2. Measure payload bytes per topic and event type.
3. Store large orderbook/trade data separately from controller events.
4. Replace repeated snapshots with deltas plus periodic snapshots.
5. Materialize latest state and dashboard facts into typed tables.
6. Use immutable Parquet part files.
7. Move high-frequency operational metrics to Prometheus or a dedicated metrics backend instead of Postgres JSONB.
8. Keep full diagnostics configurable and sampled in production.
9. Add event ids and references between frame, decision, target book, risk result, and execution report.
10. Replace opaque Parquet JSON payloads with typed columns for high-value event types.
11. Add replay contracts that verify the reduced event model can reconstruct required state.

## Prioritized Roadmap

### Priority 0: Safety And Correctness

- Add alert rules for stale data, stale private state, execution failures, persistence failures, and kill switch.
- Add optimizer decision timeout and safe fallback behavior.
- Add ledger/order-manager mismatch alerting.

### Priority 1: Scalability

- Add per-topic queue depth, lag, and payload-size metrics.
- Convert Parquet writer to immutable part files.
- Replace opaque Parquet JSON payloads with typed columns for high-value event types.
- Reuse Postgres connections or use a pool.
- Materialize common dashboard facts out of JSONB.
- Add retention and compaction policies.

### Priority 2: Engine And Execution Quality

- Wire or remove no-op optimizer threshold penalty config.
- Add an optimizer backend interface and fake backend tests.
- Split `DecisionContext` into narrower diagnostic records.
- Make own-order subtraction tick/tolerance-aware.
- Make execution state pending/confirmed rather than purely optimistic.
- Add load tests for optimizer settings and multi-symbol runtime.
- Improve global frame alignment metrics and policy.

### Priority 3: Maintainability And DX

- Add event-specific schema versions.
- Introduce an exchange adapter registry.
- Generate dashboard queries from stable schema helpers where possible.
- Document one-bot-per-process deployment explicitly.
- Add end-to-end replay/simulation tests around frame -> decision -> risk -> execution.
