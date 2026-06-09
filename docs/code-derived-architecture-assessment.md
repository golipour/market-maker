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
- large market-event and optimizer-debug payloads still dominate storage and bus fan-out,
- execution has a visible pending-confirmation window between router acceptance and private exchange truth,
- the interval scheduler and in-memory bus still lack production-grade backpressure and concurrency controls,
- the MPC path can become CPU/GPU expensive as optimizer settings scale,
- some state wiring has edge-case gaps around global frame alignment under jitter.

Most of the earlier persistence, replay, schema, and observability hardening work is already in code: typed Postgres fact tables, Parquet typed columns, lineage IDs, offline frame replay, versioned payload builders for execution/target-book/decision/diagnostics/private/ledger events, exchange adapter registry, Postgres spool recovery, and Observability v2 dashboards.

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

Recommendation: Convert the most important diagnostics into typed, versioned schemas and materialized metrics/tables. Common controller, execution, target-book, model-score, and optimizer facts are already materialized; remaining work is mainly full optimizer debug payloads and market-event orderbook/trade blobs.

## Significant Issues And Recommendations

### 1. Event Payload Duplication Will Grow Quickly

Severity: Medium

A single private REST poll can still publish a full private snapshot, ledger venue snapshot, joint ledger snapshot, execution state snapshots, and diagnostics that embed action/risk/execution stage orders. Market events on the live bus still embed the full orderbook snapshot and all interval trades.

Much of the earlier duplication problem is already mitigated in durable storage:

- REST private polling publishes one authoritative private snapshot per poll.
- Persistence compacts market trades, joint-frame ledgers, and controller action-pipeline arrays before saving.
- Typed Postgres fact tables cover market intervals, action-pipeline stages, execution lifecycle, target-book intent, model scores, and optimizer objectives/components.
- Lineage IDs (`frame_id`, `decision_id`, `target_book_id`, `execution_plan_id`) link durable events, with replay fidelity checks and offline frame replay in `mm/persistence/replay.py` and `mm/persistence/frame_replay.py`.

Remaining durable duplication is concentrated in strategy decisions, full optimizer debug payloads, full orderbook snapshots inside `MarketEvent`, and opaque JSON in `mm_events.payload` for replay/debug paths.

Impact: Postgres and Parquet storage growth is lower after persistence compaction, but will still be dominated by repeated JSON payloads as volume grows. Dashboard queries over `mm_events.payload` will slow down as data grows. Bandwidth and memory pressure increase with every new subscriber because in-memory bus fan-out copies large event objects into queues.

Recommendation:

- Add versioned payload builders for incremental private balance/open-order deltas.
- Persist large orderbook/trade snapshots at lower frequency or in a dedicated time-series store when raw market history is needed at scale.
- Keep using payload/envelope byte metrics to rank remaining high-cost topics before further compaction.

### 2. Postgres Persistence Spool Has Recovery Tooling But Needs Operations Policy

Severity: Medium

`PostgresEventWriter` now keeps a reusable psycopg2 connection across batches, drops/reconnects after write or readiness failures, retries failed batches with bounded backoff, and spills exhausted batches to local JSONL files.

`mm.tools.reingest_postgres_spool` can now scan failed JSONL spool records, decode their original envelopes, write them through the same Postgres writer path, and archive successfully reingested files. Reingest is idempotent because persisted events and typed fact tables use conflict keys.

Impact: Connection churn and immediate batch loss are no longer the main bottlenecks. During sustained database outages, events are preserved locally and recoverable after Postgres returns. Spool retention is now configurable through `persistence.retention` and applied by `python -m mm.tools.apply_retention`.

Recommendation:

- Run `apply_retention` on a schedule in production.
- Document when operators should run spool reingest before archived spool files expire.
- Add metrics for reconnect count, connection errors, write rows/sec, and commit latency.

### 3. Parquet Storage Still Needs Small-File Compaction Beyond Date Retention

Severity: Low

`ParquetEventWriter` writes immutable part files with one compact `envelope_json` plus typed nullable columns for high-value market, controller, optimizer, target-book, and execution fields. Archival compaction already drops redundant trade lists and zero balances while keeping interval features. `apply_retention` can delete whole `event_date=YYYY-MM-DD` partitions older than `persistence.retention.parquet_days`.

Impact: Live archival writes are append-like and object-store friendly. Remaining scaling risks are many small part files within retained dates and raw orderbook/debug payload storage inside `envelope_json`.

Recommendation:

- Consider offline compaction of small part files inside retained date partitions when scan volume grows.
- Consider separate columnar datasets for very large raw snapshots when typed envelope columns are not enough.

### 4. Strategy Frame Assembly Can Lose Or Delay Global Alignment

Severity: Medium

The global frame assembler keeps a bounded deque and waits up to `global_grace_ms` for exact matching `t_end_ms`. It selects the latest global event with `t_end_ms <= local.t_end_ms`.

Impact: At low frequency this is simple and reasonable. At scale, jitter, missed intervals, or scheduler delays can create persistent stale/missing global status. The bounded history can also drop relevant global events when event rates or delays increase.

Recommendation:

- Add explicit sequence/window ids for interval events.
- Track global alignment hit rate and lag distribution.
- Make history size and grace policy configurable.
- Add backpressure or skip policies for slow global feeds.

### 5. Scheduler Missed Deadlines Under Heavy Workloads

Severity: Low to Medium

`IntervalScheduler` now runs due callbacks with bounded concurrency (`scheduler.max_concurrency`, default 8) and records callback latency plus missed-deadline counters when a callback exceeds its interval budget. Different intervals that fire on the same tick also run in parallel.

Impact: Slow callbacks no longer block unrelated callbacks on the same tick, but a single hot callback can still miss its own interval budget and leave work unfinished before the next fire.

Recommendation:

- Tune `scheduler.max_concurrency` per deployment.
- Consider one scheduler per symbol/venue only if missed-deadline rates remain high after tuning.

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

Severity: Low for local development, Medium for unattended production

Prometheus alert rules are provisioned in the local Compose stack, but no Alertmanager or Grafana contact-point routing is configured. This is an operations gap, not a runtime architecture blocker.

Recommendation: Add notification routing only when moving to unattended production operation.

### 8. Observability Still Mixes Typed Facts With JSONB Debug Paths

Severity: Low to Medium

Typed Postgres fact tables and Observability v2 dashboards now cover the common market, action-pipeline, execution-lifecycle, target-order, model-quality, and optimizer-objective paths. `mm_events` remains the replay/debug event log, and some deep debugging panels still inspect full parameter dictionaries, gradients, rollout internals, and raw payload details.

Impact: Common operational dashboards no longer need repeated JSONB extraction. Remaining cost is concentrated in debug/incident panels and very large optimizer payloads.

Recommendation:

- Keep Grafana organized around fleet overview, drilldown dashboards, and incident dashboards with shared `$run_id`, `$exchange`, and `$symbol` variables.
- Add generated columns or materialized views only for debug panels that are still slow in practice.

### 9. Custom Metrics Registry Has Limited Production Features

Severity: Medium

The custom Prometheus registry supports gauges, counters, histograms, and text rendering. It lacks standard client features such as process metrics, multiprocess support, default collectors, exemplars, and stale series lifecycle controls.

Impact: It is sufficient for a single local process, but it may underperform for production multi-process deployments and makes integration less standard.

Recommendation:

- Consider migrating to `prometheus_client`.
- Add process CPU, memory, open file descriptors, thread count, and event-loop lag.
- Add explicit stale-series zeroing for label sets that disappear.

### 10. Event Bus Backpressure Is Local-Process Only

Severity: Low to Medium

`InMemoryBus` and local Redis Streams fan-out now use bounded subscriber queues by default (`bus.max_queue_size=1024`) with explicit backpressure policies (`block`, `drop_oldest`, `drop_newest`). They publish per-topic subscriber queue depth, lag, blocked-publish latency, and dropped-event counters.

Impact: Slow in-process subscribers are now observable and bounded, but remote Redis consumers and persistence writers still need their own queue/backpressure policies.

Recommendation:

- Tune `bus.max_queue_size` and `bus.backpressure_policy` per deployment.
- Keep observability/persistence consumers fast enough that `block` remains safe for trading paths.

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

### 12. Exchange Adapter Registry

Severity: Positive

Adapter construction goes through `mm/exchanges/registry.py` with one `registration.py` per exchange package. `mm/exchanges/factory.py` resolves capabilities, looks up the registration, and builds the adapter. Tests in `tests/unit/test_exchange_registry.py` verify all supported exchanges register market-data, private-state, and order-command builders.

Recommendation: Keep capability metadata and adapter registration co-located per exchange when new venues are added.

### 13. Remaining Event Payloads Still Mix Structs And Unversioned Dicts

Severity: Low

Execution, target-book, strategy-decision, controller-diagnostics, private-snapshot, ledger-snapshot, and market-event bus payloads publish through versioned builders in `mm/core/payload_schema.py`. `SymbolEngine` publishes `mm.market_event` v1 payloads; frame assembly parses them back into domain structs at the queue boundary. Incremental private balance/open-order deltas still use typed structs on the bus without versioned payload builders.

Recommendation:

- Add versioned payload builders for incremental private balance/open-order deltas if those topics become durable at scale.
- Extend replay fidelity checks to cover private/ledger snapshot schema validation where useful.

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

1. Add versioned payload builders for incremental private balance/open-order deltas.
2. Store large orderbook/trade data separately from controller events when raw market history is needed at scale.
4. Keep full diagnostics configurable and sampled in production.
5. Use payload and envelope byte metrics to rank remaining high-cost topics.

## Prioritized Roadmap

### Already Implemented

These items are in code and covered by focused tests; they are no longer open roadmap work:

- Exchange adapter registry (`mm/exchanges/registry.py`, per-exchange `registration.py`)
- Versioned payload builders for execution lifecycle, execution snapshots, target books, strategy decisions, controller diagnostics, private snapshots, ledger snapshots, and market events (`mm.market_event` v1).
- Lineage IDs and replay reconstruction (`mm/persistence/replay.py`)
- Replay lifecycle fidelity checks (`verify_lineage_fidelity()`)
- Offline frame replay through strategy/risk/execution planning (`mm/application/frame_simulator.py`, `mm/persistence/frame_replay.py`)
- Persisted market-data replay adapter (`mm/ingestion/replay_adapter.py`)
- Typed Postgres fact tables and Parquet typed columns
- Persistence compaction, Postgres connection reuse, failed-batch spool, and spool reingest
- Retention/tiering policies and `python -m mm.tools.apply_retention` (`mm/persistence/retention.py`)
- Observability v2 dashboards and common Prometheus metrics (`mm_frame_age_ms`, persistence queue depth/write latency, controller diagnostics bridge)
- Bounded bus backpressure with subscriber queue depth/lag metrics (`mm/bus/delivery.py`, `bus.max_queue_size`, `bus.backpressure_policy`)
- Concurrent interval scheduler with callback latency and missed-deadline metrics (`scheduler.max_concurrency`)

### Priority 1: Scalability And Runtime Pressure

- Add versioned payload builders for incremental private balance/open-order deltas.
- Wire retention execution into normal runtime/ops workflows and validate tier settings under real data volume.
- Tune and validate bus backpressure defaults under real subscriber load.

### Priority 2: Engine And Execution Quality

- Add load tests for representative optimizer settings (`n_samples`, `horizon`, `adam_epochs`) and multi-symbol frame execution.
- Improve global frame alignment observability and policy: explicit hit/miss/stale metrics, configurable history/grace, and clearer skip/backpressure behavior.
- Extend offline frame replay to full MPC re-simulation checks, not just execution-only replay from persisted target books.

### Priority 3: Maintainability And Deployment Clarity

- Document one-bot-per-process as the explicit deployment model.
- Consider merging exchange capability metadata into each exchange `registration.py` if the split becomes annoying.
- Trim or retire legacy Grafana dashboards once Observability v2 fully covers incident/debug workflows.
