This is a substantial, well-structured architecture question. Let me build a comprehensive response across all 13 dimensions — I'll produce a written answer and a visual architecture diagram.This is a comprehensive, well-formed architecture document. Let me give you a thorough written response across all dimensions, anchored by a target architecture diagram.

---

This is a substantial platform redesign. I'll address it in two parts: the recommended target architecture (all 13 questions integrated), then a concrete migration roadmap.

---

## Part 1: Target Architecture

### Overall Design Philosophy

The guiding principle is: **shared infrastructure, isolated strategy execution**. Most of the complexity comes from deciding exactly where the isolation boundary sits and what gets promoted to a shared service. The answer depends on three axes: data fan-out (who needs this data?), fault blast radius (what breaks if this dies?), and compute profile (I/O-bound vs CPU-bound).

---

### Q1 & Q7 — Service Decomposition

Here's the recommended service boundary map:

**Shared platform services (run once, scale as needed):**

- `market-data-service` — normalizes feeds from all exchanges, publishes to event bus
- `execution-service` — owns order submission, amendment, cancellation; handles exchange-specific API quirks
- `ledger-service` — source of truth for balances, fills, positions at the exchange level
- `portfolio-service` — capital allocation, PnL attribution, cross-strategy risk aggregation
- `control-plane-api` — REST/gRPC admin API for lifecycle management
- `config-service` — stores exchange credentials, strategy configs, parameters; distributes to workers
- `persistence-service` — write-path for events, facts, diagnostics to durable storage
- `dashboard-backend` — aggregates metrics, serves WebSocket feeds to the frontend

**Per-strategy workers (one process per strategy instance):**

- MPC optimizer
- Frame assembly and market data consumption (subscribes to shared event bus)
- Private state reconciliation (calls ledger-service)
- Risk engine (local risk limits; delegates portfolio risk to portfolio-service)
- Execution management (calls execution-service)
- Local event emission

The key insight: strategy workers become stateless-ish event consumers. They don't own connectivity to exchanges, they don't own persistence, and they don't own account truth. They own the strategy logic and local risk state.

---

### Q2 — Strategy Execution Model

The optimal model is **one process per strategy instance**, deployed as Kubernetes pods.

Each strategy worker pod:
- Runs a single asyncio event loop for I/O (consuming from event bus, calling execution-service)
- Runs MPC optimization in a subprocess via a `ProcessPoolExecutor` or a dedicated sidecar
- Has its own resource limits (`requests`/`limits` in K8s)
- Can be restarted independently without affecting other strategies

Avoid hosting multiple strategies in one process. The blast radius of a Python exception, memory leak, or optimizer hang is then bounded to one strategy. Thread-based isolation within a process is insufficient for CPU workloads because of the GIL; subprocess-based isolation within a pod adds OS-process complexity that K8s already handles better at the pod level.

Lifecycle management (`start`/`stop`/`pause`/`resume`) maps cleanly to K8s Deployment replicas and custom annotations, managed via the control-plane-api issuing `kubectl apply` or calling the K8s API directly.

---

### Q3 — Market Data Architecture

Yes, decouple market data entirely. The target shape is:

```
Exchange WebSocket → Market Data Service → Event Bus (Redis Streams or Kafka)
                                             ↓
                                    Strategy Workers (consumers)
```

The market data service per exchange:
- Maintains the WebSocket connection
- Normalizes orderbook snapshots and deltas into a canonical schema
- Publishes `OrderbookUpdate`, `TradeEvent`, `TickerUpdate` events to the bus
- Maintains a current-state cache (Redis sorted sets or an in-memory store with Redis fallback)

Strategies subscribe to the relevant streams. Orderbook reconstruction happens once in the market data service, not N times per strategy. This eliminates N×WebSocket connections to each exchange and means you can add strategies without adding exchange connections.

For the frame assembly step that currently lives in the strategy runtime, move the raw data consumption here but keep the strategy-specific feature extraction (VWAP windows, spread metrics, conditional model inputs) inside the worker, since those are strategy-specific transforms.

---

### Q4 — MPC Performance

In rough order of payoff:

**Largest gains — algorithmic:**

The MPC rollout is fundamentally a constrained optimization repeated at each timestep. The biggest wins come from:

- **Warm starting**: initialize the optimizer at each tick from the previous solution shifted by one step. This dramatically reduces iterations for a well-conditioned problem.
- **Reducing horizon**: empirically tune the MPC horizon to the minimum that maintains acceptable policy quality. Halving the horizon roughly quarters the computation.
- **Vectorizing the rollout**: express the horizon simulation as batched matrix operations rather than a Python loop over timesteps. With NumPy or PyTorch this can be 10–100× faster depending on current implementation.
- **Approximate value functions**: replace the full MPC rollout with a learned terminal value function for shorter horizons. Train offline, evaluate online as a neural net forward pass.

**Large gains — implementation:**

- Move the inner optimizer loop to NumPy + `scipy.optimize.minimize` with `method='SLSQP'` if you're doing constrained optimization — it's significantly faster than Python-level gradient descent.
- If using PyTorch already, ensure the rollout is a single `.backward()` call on a batched computation graph, not a Python loop.
- **Numba `@jit`** on hot numerical loops (non-matrix operations, index math) can give 10–50× for tight loops.
- GPU is only worthwhile if batch size > ~100 concurrent strategies sharing a single optimizer call. For single-strategy optimization, GPU transfer overhead dominates.

**System level:**

- Run optimization in a `ProcessPoolExecutor` with worker count matching CPU cores
- If multiple strategies trade the same symbol on the same exchange, consider a shared optimizer that batches their rollouts and decomposes the solution — saves repeated market model calls

---

### Q5 — Concurrency Model

**I/O workloads → asyncio throughout.** WebSocket connections, REST polling, event bus consumption, calls to internal services are all naturally async. A single asyncio event loop per process handles hundreds of concurrent I/O operations efficiently. Use `aioredis` for Redis Streams, `aiohttp` or `httpx` for REST. No threads needed for I/O.

**CPU workloads → `ProcessPoolExecutor` or a dedicated worker process.** The MPC optimizer is the primary CPU workload. Run it in a separate OS process. The asyncio event loop `await`s a `loop.run_in_executor(process_pool, optimize, state)` call. This sidesteps the GIL entirely and keeps the event loop unblocked for I/O.

**The hybrid model in practice:**

```
asyncio event loop (main process)
  ├── market data consumer (asyncio)
  ├── execution manager (asyncio)
  ├── private state reconciler (asyncio)
  └── optimizer call → run_in_executor → ProcessPoolExecutor
                                              └── optimizer worker process
```

The risk calculations are usually fast enough to run inline in the asyncio loop if implemented as vectorized NumPy. If they're slow, they go in the process pool too.

---

### Q6 — Kubernetes Deployment

**Horizontally scalable (Deployments with HPA):**
- `strategy-worker` — scale to thousands of pods
- `market-data-service` — one per exchange, or sharded by symbol group
- `dashboard-backend` — stateless, scale behind a load balancer

**Singleton or low-replica (StatefulSets or controlled Deployments):**
- `ledger-service` — needs leader election or a single writer pattern to avoid double-booking
- `execution-service` — should be 2–3 replicas for HA but needs idempotent order submission logic
- `portfolio-service` — similar to ledger; consistency over availability
- `control-plane-api` — 2–3 replicas, stateless, fronted by an ingress

**Configuration distribution:** Store exchange credentials in K8s Secrets (or Vault with the vault-agent sidecar). Strategy parameters in ConfigMaps, with a config-service that watches for changes and pushes updates to running workers via a control channel (Redis pub/sub or a gRPC stream). Avoid requiring pod restarts for parameter changes.

**Rolling upgrades:** Use `RollingUpdate` strategy for stateless services. For strategy workers, the cleanest upgrade path is: drain (pause the strategy gracefully, flush in-flight orders), terminate old pod, start new pod, resume. Implement a pre-stop lifecycle hook that calls the control-plane-api to pause before shutdown.

**Autoscaling:** HPA on CPU utilization for strategy workers works but lags for bursty workloads. Consider KEDA (Kubernetes Event-Driven Autoscaling) with Redis Streams consumer lag as the scaling metric — this lets you scale based on how backed-up the event pipeline is rather than CPU.

**Secrets:** K8s Secrets for API keys at minimum; HashiCorp Vault with dynamic secrets for production (avoids long-lived credentials in etcd).

---

### Q8 — Event and Messaging Architecture

**Keep Redis Streams as the backbone for now; plan for Kafka when you exceed ~100 strategies.**

Redis Streams are operationally simpler, support consumer groups with acknowledgment, have sub-millisecond latency, and integrate well with your existing stack. They're appropriate up to moderate throughput (tens of thousands of events/second).

**Partitioning:** Partition by exchange+symbol (e.g. `md:wallex:btcusdt`). Each strategy worker is a consumer in a consumer group on its relevant streams. This gives natural fan-out without broadcasting everything to everyone.

**When to introduce Kafka:** When you need replay of days/weeks of market data, true multi-datacenter replication, or throughput beyond Redis Streams' practical limits. Kafka's compacted topics are also better for the "latest orderbook state" use case (strategy workers that restart can catch up from the compacted topic without needing a full snapshot call).

**NATS** is worth considering as a lightweight alternative to Kafka for pub/sub if you don't need the replay/retention guarantees. NATS JetStream gives persistent streams with lower operational overhead than Kafka.

**Ordering:** Within a Redis Stream for a given symbol, ordering is guaranteed. Across symbols it isn't, but cross-symbol ordering is rarely semantically meaningful for a market maker.

**Backpressure:** Consumer group lag is your primary backpressure signal. Alert when consumer lag exceeds N events (configurable per stream). Strategy workers should have circuit-breaker logic: if they fall too far behind on market data, pause order submission until they've caught up.

---

### Q9 — Persistence Architecture

**PostgreSQL:** Strategy configuration, account/credential registry, strategy lifecycle events, fill history, PnL records, audit trail. Anything that needs relational queries, ACID guarantees, or is queried by the dashboard.

**TimescaleDB (extension on Postgres) or InfluxDB:** Market data snapshots, strategy metrics time series (spread, inventory, exposure over time), system health metrics. TimescaleDB keeps you on the Postgres stack, which simplifies operations.

**Object Storage (S3-compatible):** Full orderbook snapshots for replay, raw tick data archives, diagnostic dumps, large optimizer state objects. Cheap, durable, not query-optimized.

**Event sourcing:** Implement it for the strategy ledger specifically — store all events (fills, transfers, allocations, reconciliation results) as an append-only log, derive balance/position state by replaying. This gives you free replay, audit trail, and the ability to reconstruct state at any point in time. Don't event-source everything — it adds complexity where you don't need it.

**Retention:** Hot path (last 7 days) in TimescaleDB with automatic downsampling; warm path (30–90 days) in object storage; cold path (archive) in cheap object storage with Glacier-equivalent. Implement this with automated policies from day one — it's painful to retrofit.

---

### Q10 — Control Plane Architecture

The control plane is a gRPC/REST API server with these resource types:

- `Exchange` — config, credentials, connectivity status
- `Account` — per-exchange account, subaccount assignments
- `Portfolio` — capital allocation container
- `Strategy` — configuration, parameter set, lifecycle state
- `Command` — async commands (start, stop, pause, kill)

**Implementation:** FastAPI or gRPC-gateway on Python (consistent with your stack), backed by Postgres for state. The control plane pushes configuration changes to running workers via a Redis pub/sub channel keyed by `strategy:{id}:commands`. Workers subscribe to their own command channel and implement a state machine: `initializing → running → paused → stopping → stopped`.

**Global kill switch:** A Redis key `platform:emergency_halt` that all workers poll (or subscribe to via pub/sub). When set, all workers immediately cancel open orders and halt. This must be testable and rehearsed regularly.

**Risk controls:** Implement a risk gate in the control plane that blocks strategy commands when portfolio-level limits are breached (max drawdown, max open exposure). The risk engine in `portfolio-service` writes risk state to Redis; the control plane reads it before processing commands.

---

### Q11 — Dashboard

**Stack:** React frontend, FastAPI backend serving WebSocket streams for live data, Postgres for historical queries.

**What operators see first (the "flight deck" view):**

- Strategy health matrix: each strategy as a row, columns for status, PnL (today/MTD), inventory, spread, last fill time, alerts
- Exchange connectivity: green/amber/red per exchange, latency indicators
- Portfolio-level risk: aggregate exposure, drawdown from peak, capital utilization

**Detail views (drill down from the matrix):**

- Per-strategy: MPC state, open orders, recent fills, optimizer performance, event lag
- Per-exchange: WebSocket status, API rate limit headroom, data freshness
- System: CPU/memory per pod, event bus lag, persistence queue depth

**Technology:** For the live matrix, WebSocket subscriptions from the backend to each strategy's metrics stream, aggregated and fanned out to connected clients. Use server-sent events for lower-complexity scenarios. Avoid polling the Postgres DB at dashboard refresh rate — maintain a Redis-backed metrics cache that workers write to continuously.

---

### Q12 — Observability

**Metric cardinality:** This is the main scaling challenge. With thousands of strategy instances, naively labeling every metric with `{strategy_id, exchange, symbol}` creates millions of time series. Mitigate by:

- Aggregating at source: workers emit rollup metrics (last-N-minutes PnL, current inventory) not raw tick data
- Using exemplars (OpenTelemetry) to link aggregated metrics to specific trace IDs for drill-down
- Storing high-cardinality raw data in TimescaleDB/ClickHouse, not Prometheus

**Distributed tracing:** An order placement trace should span: strategy worker → execution service → exchange API → fill event → ledger update. This is invaluable for debugging latency and correctness issues. OpenTelemetry with Tempo or Jaeger as the backend.

**Log aggregation:** Structured JSON logs from all services → Fluent Bit → Elasticsearch or Loki. The key structured fields are `strategy_id`, `exchange`, `symbol`, `event_type`. This lets you query "all events for strategy MM-3 in the last 5 minutes" across all services.

**Alerting tiers:**

- P0 (immediate page): global halt triggered, ledger reconciliation failure, exchange connectivity loss
- P1 (prompt response): strategy error rate >X%, optimizer timeout, large drawdown
- P2 (next business day): event bus lag growing, disk utilization, API rate limit pressure

---

### Q13 — Account, Portfolio, Ledger, Capital Allocation

This is the most important architectural question because it determines the ownership model for everything else. Here's my recommendation:

**Adopt the 4-level hierarchy:**

```
Platform
→ Exchange
→ Account (real exchange account)
→ Portfolio (logical capital container)
→ Strategy Instance
```

**Why not 3-level (Exchange → Account → Strategy)?** Because multiple strategies sharing an account with no portfolio abstraction means the risk engine has to reason about raw exchange balances minus estimated in-flight orders, which is fragile and creates race conditions. The portfolio layer provides a clean accounting abstraction that decouples strategy execution from exchange truth.

**The portfolio layer is a software construct.** It doesn't correspond to anything on the exchange. It's maintained by `portfolio-service` and enforced by the risk gate. A portfolio holds a capital allocation (e.g. 100,000 USDT) carved from the account's actual balance. Strategies draw against their portfolio allocation, not directly against the account balance.

**Ledger architecture (three layers):**

- `exchange-ledger` — tracks actual exchange state: real balances, real open orders, real fills. Reconciled continuously against exchange REST snapshots. Source of truth for what the exchange believes.
- `portfolio-ledger` — tracks allocated capital per portfolio. Derives from exchange-ledger minus unallocated buffer. Source of truth for what strategies can spend.
- `strategy-ledger` — tracks attributed inventory, attributed PnL, net position per strategy. Derived from fill events attributed to each strategy. Source of truth for performance accounting.

**Reconciliation flow:**

```
Exchange fill event
  → exchange-ledger (update actual position)
    → portfolio-ledger (update available capital)
      → strategy-ledger (attribute fill to strategy, update PnL)
```

Discrepancies between exchange-ledger and the sum of portfolio-ledger allocations trigger a reconciliation alert. Discrepancies between portfolio-ledger and strategy-ledger are a bug in attribution logic.

**Capital allocation enforcement:** Each strategy worker's risk engine checks against its portfolio allocation before submitting an order. The check is: `current_inventory_value + pending_order_value < allocation_limit`. The portfolio-service maintains the current allocated/used state in Redis for low-latency reads. The strategy worker reads this before computing the MPC rollout's order size.

**On Model C (API keys):** API key separation gives audit trail isolation and rate limit isolation. It doesn't give balance isolation if the account is unified. Worth using for the audit/rate-limit benefits, but don't rely on it for capital isolation — the portfolio layer handles that.

---

## Part 2: Migration Roadmap

The anti-pattern to avoid is a big-bang rewrite. The current system works. Migrate incrementally while keeping the existing system running.

**Phase 1 — Extract shared infrastructure (2–3 months)**

Extract market data ingestion into a standalone `market-data-service` alongside the existing bot. Route the existing bot to consume from it rather than its own WebSocket connection. This is purely additive — the bot's behavior doesn't change. Deploy this as a single service initially, not yet in K8s.

In parallel: introduce the portfolio/ledger abstraction in the data model. Add the new tables to Postgres without removing the existing ones. Run the new ledger logic in shadow mode (compute but don't act on it) to validate correctness.

**Phase 2 — Strategy worker isolation (2–3 months)**

Refactor the existing bot's strategy core into a standalone process that consumes from the market-data-service event bus and calls a stub execution-service (initially just a thin wrapper around the existing exchange API calls). Run the existing and new processes in parallel, comparing outputs.

Once validated, cut over. Now you have a strategy worker that is decoupled from market data ingestion and has a clean execution boundary.

**Phase 3 — Control plane and multi-strategy (2–3 months)**

Build the `control-plane-api` and configuration model. Launch a second strategy instance (different symbol, same exchange) using the new worker architecture. Validate that the two workers share market data correctly and don't interfere.

Introduce the portfolio-service and activate capital allocation enforcement. Run the portfolio-service's reconciliation against the existing ledger to validate.

**Phase 4 — Kubernetes and operational hardening (2–3 months)**

Containerize all services. Deploy to K8s. Implement the global kill switch, rolling upgrade hooks, autoscaling. Add the dashboard. Validate the full stack with 5–10 concurrent strategy instances.

**Phase 5 — Scale testing and optimization (ongoing)**

Load test with 50–100 strategies. Profile the MPC optimizer and apply the algorithmic improvements. Instrument distributed tracing end-to-end. Tune event bus partitioning and consumer scaling. Introduce Kafka if Redis Streams throughput becomes a bottleneck.

---

Now let me render the target architecture as a visual:---

A few things worth calling out explicitly that didn't fit neatly into the numbered sections:

**On the ledger hierarchy decision**: The 4-level model (Platform → Exchange → Account → Portfolio → Strategy) may feel like over-engineering for 4 strategies. The key is to implement the data model now even if the enforcement logic is thin. Adding portfolio-level risk gates later is much easier than retrofitting the ownership model after you have 50 strategies and tangled balances.

**On the execution service boundary**: One subtle question is whether the execution service should be shared across strategies or per-strategy. A shared execution service is cleaner architecturally but introduces a coordination point for rate limit management (exchanges have per-account API rate limits). The cleanest answer: one execution service instance per exchange account, not per strategy. This naturally pools rate limit budget across strategies sharing an account while keeping per-account state coherent.

**On MPC warm starting**: This is genuinely the single highest-ROI optimization you can make and costs perhaps a day of implementation. The optimizer's previous solution is already available at each tick — passing it as the initial guess to the next call is trivial and typically cuts iterations by 60–80% for stable market conditions.

**On the migration sequencing**: Resist the temptation to build the dashboard and control plane first. They're visible and feel like "the platform," but they're useless without the service decomposition underneath. The order in the roadmap above is load-bearing: market data → strategy isolation → control plane → K8s → observability.