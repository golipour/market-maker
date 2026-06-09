Now I have a thorough understanding of both files. Let me compose a comprehensive architectural review.This is a detailed and substantive review request. Let me write a thorough response grounded in the actual code documentation.

---

# Scaling the Market-Making Platform: Architecture Review

*Based on the code-derived system documentation.*

---

## Framing

The current system is not a toy. It already has a robust event bus with Redis Streams support, versioned payload schemas, lineage IDs across frame → decision → execution, a layered Postgres persistence model with typed fact tables, a full observability stack, a control plane with kill switches, and a FastAPI admin API. The gap between where you are and a production multi-strategy platform is narrower than it might seem — but the shape of that gap matters a lot.

The key insight: **the system is already almost entirely message-driven**. Most components communicate via topics on `AbstractEventBus`. The one significant exception is the strategy pipeline, which directly owns the `MpcStrategyEngine`, risk engine, and order-manager map. That exception is also the most computationally expensive part of the system. This is the architectural center of gravity for the scaling work.

---

## 1. Recommended Target Architecture

The right target for your scale assumptions (10–200 active strategy instances, 5–20 exchanges, single operator) is a **modestly decomposed, event-driven monorepo deployment**. Not a microservice fleet. The decomposition should follow natural isolation boundaries that are already present in the codebase.

### The core split

```
┌─────────────────────────────────────────────────┐
│              Control Plane Service               │
│  (Admin API, Strategy Lifecycle, Config Store)  │
└──────────────────┬──────────────────────────────┘
                   │  admin:commands / admin:events
          ┌────────┴────────┐
          │   Redis Streams  │  (event spine)
          └────────┬────────┘
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼────┐   ┌─────▼──────┐
│Market │    │Strategy │   │Persistence │
│Data   │    │Workers  │   │& Ledger    │
│Pods   │    │(N pods) │   │Service     │
└───────┘    └─────────┘   └────────────┘
```

**Market Data Pods**: one pod per exchange (or per exchange+symbol cluster). These run `SymbolEngine` + `IntervalScheduler` and publish `market:events:*` to Redis Streams. No strategy logic. Horizontally scalable, but in practice you'll rarely run more than one per exchange.

**Strategy Worker Pods**: each pod hosts one or more strategy instances, each owning an `MpcStrategyEngine` + `NetDeltaRiskEngine` + `OrderManager`. They subscribe to `market:events` and `ledger:snapshot` from Redis, run decisions, and publish back to Redis. This is where the GIL-bound CPU work lives.

**Persistence & Ledger Service**: subscribes to all configured topics and writes to Postgres. Also hosts `PortfolioLedgerService` when strategies share exchange accounts (discussed in Q13).

**Control Plane Service**: the FastAPI admin app, promoted to a proper service with its own Postgres schema. Owns strategy lifecycle state (deployed, running, paused, stopped), exchange/credential registration, and operator commands.

This is essentially the architecture you already have when running multiple configs/processes — but made intentional, containerized, and wired through a single Redis Streams spine.

---

## 2. Migration Roadmap

The migration is evolutionary, not a rewrite. There are four stages.

### Stage 1: Centralize configuration and strategy lifecycle (2–4 weeks)

The biggest missing piece is a **strategy registry** — a Postgres-backed store that knows about exchanges, accounts, strategy definitions, and their current lifecycle state. Right now this lives in JSON files per process.

What to build:
- A `strategies` table: `(id, exchange, symbol, actor_id, preset, config_json, status, created_at, updated_at)`
- An `exchanges` table: `(id, name, credentials_ref, capabilities)`
- Extend the admin API to CRUD these tables
- A strategy supervisor process that reads from this table and starts/stops strategy worker processes accordingly

The existing `RuntimeControlPlane` + `admin:commands` topic already handles runtime lifecycle signals (pause/resume/kill). You're extending this upward to include process-level lifecycle.

Keep running one process per strategy at this stage. Don't change the strategy runtime at all.

### Stage 2: Extract Market Data as a shared service (1–2 weeks)

Right now each strategy process ingests its own market data. When three strategies trade BTCUSDT on Wallex, three processes connect to the same feed.

The fix is straightforward: the codebase already has `RedisStreamsBus`. Move `SymbolEngine` into a dedicated Market Data process per exchange. Strategy workers subscribe to `market:events:wallex:BTCUSDT:*` from Redis instead of running their own adapter. The `StrategyInputFrameAssembler` already parses versioned market-event payloads — it doesn't care where they came from.

One Market Data pod per exchange. Strategies decouple from feed management entirely.

### Stage 3: Strategy Worker consolidation and process model (2–4 weeks)

Once market data is shared, the per-strategy overhead of a full Python process becomes the bottleneck. The right model for your scale (10–200 strategies):

**2–8 strategy workers per pod, running as asyncio tasks within one process.** Not threads, not subprocesses. Here's why:

- The I/O path (Redis Streams subscribe, REST/websocket private state, order commands) is already fully async. Multiple strategies sharing one asyncio event loop is cheap.
- The CPU path (MPC optimization) is already dispatched to a worker thread via `run_in_executor`. That pattern scales: N strategies in one process each submit their optimizer call to a shared thread pool. The GIL serializes CPython threads, but PyTorch releases it during most tensor operations — you get real parallelism on the optimizer's inner loops.
- Process-level isolation is preserved by keeping exchange accounts separated (see Q13 below) and by the existing kill-switch and pause mechanisms.
- Fault isolation: an optimizer exception in one strategy should not kill others. The existing `decision_timeout_ms` + `ControllerFallback` pattern already handles optimizer hangs. Unhandled exceptions in the strategy coroutine need a try/catch-and-restart wrapper at the `StrategyRunner` level.

**One pod per exchange account** is a natural grouping for fault isolation and resource allocation. If Wallex goes offline, only that pod is affected.

### Stage 4: Kubernetes deployment (1–2 weeks once Stage 3 is stable)

By Stage 3 you have well-defined containers with clear input/output contracts over Redis Streams. Kubernetes adds nothing architecturally — it's operational packaging. Use:
- `Deployment` for Market Data pods (one replica per exchange, singleton by design)
- `Deployment` for Strategy Worker pods (scalable, but you'll rarely need more than one per exchange account)
- `Deployment` for Control Plane (singleton + `RollingUpdate`)
- `Deployment` for Persistence Service (singleton)
- Postgres and Redis as managed services or Helm charts

HPA (Horizontal Pod Autoscaler) is not worth wiring up for this scale. Manual replica control via the Control Plane API is simpler and more appropriate.

---

## 3. What Should Stay Monolithic

- **MPC engine + risk + execution within a strategy worker**: these are tightly coupled by design. The `StrategyRunner` owns all three because they share `JointStrategyFrame` and `TradingState` in a single decision cycle. Pulling them apart would add latency and complexity for no benefit.
- **Frame assembly**: `StrategyInputFrameAssembler` + `JointStrategyFrameAssembler` are cheap, fast, and tightly coupled to the strategy's view of the world. Keep them inside the strategy worker.
- **Postgres persistence**: the `PostgresEventWriter` subscriber model already works well. One persistence service subscribing to all topics is simpler than per-strategy persistence.
- **The event envelope and topic schema**: the versioned payload contract (`mm/core/payload_schema.py`) and topic naming convention are load-bearing. Don't fragment these across services.

---

## 4. What Should Become Independently Scalable

- **Market data ingestion** (Stage 2 above): one pod per exchange, multiple strategies consuming from Redis Streams.
- **The control plane / admin API**: the FastAPI app already exists. Promoting it to a standalone service enables independent deployment and access control.
- **Strategy workers**: horizontally scalable as a group, with the grain of scaling being "exchange account" not "individual strategy."

That's it. Three independently scalable concerns. The temptation to further decompose (separate Risk Service, separate Ledger Service, separate Execution Service) should be resisted firmly. These components are decision-cycle co-located by necessity.

---

## 5. Optimizations Worth Implementing

### MPC / optimizer (highest leverage)

The default optimizer config (`n_samples=8, horizon=2, adam_epochs=10`) is conservative. The example config (`n_samples=64, horizon=1, adam_epochs=30`) is more realistic. The primary levers, roughly in order of impact:

**Reduce horizon to 1.** For market-making at low frequency, a 1-step lookahead is usually sufficient and cuts simulation cost linearly. The system already supports `horizon=1`.

**Batch across strategies on one device.** If multiple strategies run in the same process and use the same device (CPU or GPU), you can batch their optimizer calls together. The optimizer is `MonteCarloMPCTensorBatched` — the batching dimension is currently `n_samples`. You could extend this to batch across strategies that have the same model structure. This is a meaningful optimization if you have 10+ strategies per worker.

**GPU for larger n_samples.** The optimizer uses PyTorch and already supports device configuration. If `n_samples` needs to be >64 for model quality, a single cheap GPU (e.g. RTX 3080 or equivalent cloud instance) can run the tensor operations 10–30x faster than CPU. The `asyncio.run_in_executor` dispatch pattern already supports this — just move the tensor work to CUDA.

**Numba for the non-PyTorch paths.** `soft_fill_batched()` and the fill simulation inner loops may have pure-Python or NumPy hotspots that Numba can accelerate with minimal code change. Profile before committing to this.

**Rust extension for `OrderbookManager` delta application.** If you profile and find orderbook delta processing is a bottleneck at high symbol counts, this is the right boundary for a Rust extension. But it's only worth it if you're running hundreds of tracked symbols with fast websocket feeds. At 10–50 symbols with interval-based `MarketEvent` publishing, Python is fine.

**Do not rewrite the optimizer in Cython.** PyTorch's autodiff and GPU path is the right long-term compute substrate. Cython adds maintenance burden without the GPU payoff.

### I/O / concurrency

The existing asyncio + `run_in_executor` for CPU work is the correct model. No fundamental changes needed. One refinement: when multiple strategies in the same worker submit optimizer calls, use a `ProcessPoolExecutor` instead of `ThreadPoolExecutor` for the optimizer thread pool. This breaks the GIL entirely for the PyTorch C++ kernel calls and allows true parallel CPU execution across strategies. The tradeoff is inter-process serialization overhead for tensors, which PyTorch handles efficiently via shared memory (`fork`-safe on Linux).

Actually: if using GPU, stay with `ThreadPoolExecutor` since CUDA contexts are process-local. If CPU-only with many strategies per worker, `ProcessPoolExecutor` wins.

---

## 6. Infrastructure Additions That Are Justified

**Redis** — already used. Keep it. Redis Streams as the event spine for market data fanout and cross-process event delivery is the right call.

**A strategy registry in Postgres** — just a few tables in the existing database. This enables the control plane described in Q10.

**A proper secrets manager** — Kubernetes Secrets or HashiCorp Vault for exchange API keys. The current `PrivateRuntimeConfig` takes token/API-key credentials from config. At multi-exchange scale, rotating credentials and auditing access becomes important. Kubernetes Secrets + environment injection is the minimum viable approach; Vault is worth it if you have many exchange accounts.

**TimescaleDB (as a Postgres extension)** — if and when you want to query market data time-series efficiently without moving to a separate DB. TimescaleDB is just Postgres with hypertables; zero new operational complexity. Useful for `mm_market_facts` and `mm_execution_facts` as volume grows.

---

## 7. Infrastructure That Would Be Over-Engineering

**Kafka** — Redis Streams handles your event throughput with room to spare. Kafka introduces broker clusters, consumer group management, and operational overhead that provides no benefit at this scale. The existing `RedisStreamsBus` already provides ordered delivery, consumer groups, and replay. Keep it.

**Separate Risk Service** — risk validation is sub-millisecond and tightly coupled to the frame/decision cycle. An RPC hop to a separate risk service adds latency and failure modes for no gain.

**Separate Execution Service** — same reasoning. `OrderManager` is already a facade. The execution path touches private ledger state directly and needs to be co-located with the strategy for consistency.

**Separate Ledger Service** (for single-account-per-strategy cases) — `PortfolioLedgerService` subscribes to private topics over Redis. It can stay in the strategy worker or move to the persistence service. It does not need to be its own independently deployed service.

**NATS / RabbitMQ** — no. You already have Redis.

**MongoDB or Couchbase** — no. Postgres JSONB is sufficient for strategy configuration schema flexibility. The `config_json` column in the strategy registry table handles dynamic config without a document database.

**Multiple Kubernetes clusters** — single cluster. You're a single-operator system.

**Service mesh (Istio, Linkerd)** — completely unnecessary at this scale and operational model.

---

## 8. Event and Messaging Architecture

**Keep Redis Streams as the backbone.** The `RedisStreamsBus` is already well-implemented: compact envelope writing, consumer groups, local subscriber fanout, skip-own-messages logic, and backpressure. This is the right choice for your scale.

**Topic partitioning**: the existing topic scheme (`market:events:<exchange>:<symbol>:<interval>`) already provides natural partitioning. Strategy workers subscribe only to the topics they need. No changes required.

**Ordering guarantees**: Redis Streams provides per-stream ordering. Since market events for a given exchange/symbol/interval are a single stream, ordering is preserved. Cross-stream ordering (e.g. between market events and private state) is handled by `StrategyInputFrameAssembler`'s existing grace-period logic.

**Replay**: Postgres persistence + `mm/persistence/replay.py` is the replay path. Redis Streams provides short-term replay (via stream offset); Postgres provides long-term replay. This is the right division.

**Backpressure**: the `InMemoryBus` already supports `block`/`drop_oldest`/`drop_newest`. For cross-process Redis delivery, the strategy worker's subscribe loop should use bounded queues and monitor `bus_subscriber_queue_depth` metrics (already exported).

---

## 9. Persistence Architecture

The current Postgres-centric architecture is appropriate and should largely stay. The data classification in Q14 maps cleanly to a modest expansion:

**Keep in Postgres:**
- Operational metadata (exchanges, accounts, API key references, strategy definitions, deployment state)
- Execution records (`mm_execution_facts`, `mm_target_order_facts`)
- Ledger state and audit trail
- `mm_latest_state` current-state table
- Strategy configuration

**Add TimescaleDB hypertables (on the same Postgres instance) for:**
- `mm_market_facts` — already written by `PostgresEventWriter`. Hypertables with time-based partitioning and retention policies make time-range queries on market data fast without a separate DB.
- `mm_optimizer_facts`, `mm_model_score_facts` — same pattern; good candidates for 90-day retention with monthly aggregation into a summary table.

**Add MinIO (self-hosted S3) or AWS S3 for:**
- Parquet archives — the `ParquetEventWriter` already writes partitioned Parquet. Configuring its root to an S3-compatible path (via `s3fs` or `pyarrow`'s S3 filesystem) moves long-term archival off local disk with minimal code change.
- Optimizer rollout datasets for offline research
- Model training datasets

**Observability data (metrics, logs, traces) stays in:**
- Prometheus (metrics) — already in place
- Loki (logs) — already in place via Grafana Alloy
- Tempo (traces) — already in place

Do **not** add ClickHouse, Druid, Pinot, or InfluxDB. TimescaleDB gives you 90% of the analytical query benefit at zero additional operational cost since it runs in the same Postgres instance. At hundreds of millions of rows, revisit. You're not there yet.

**Data lifecycle:**
```
Market Data
→ Redis Streams (hot, ~hours)
→ mm_market_facts / TimescaleDB (warm, 90 days)
→ Parquet on S3/MinIO (cold, unlimited)

Execution Data
→ mm_execution_facts / Postgres (hot, 90 days)
→ Parquet on S3/MinIO (cold, unlimited)

Optimizer Diagnostics
→ mm_optimizer_facts / TimescaleDB (warm, 90 days)
→ Parquet on S3/MinIO (cold, unlimited)

Operational Data (config, accounts, strategy state)
→ Postgres (permanent, low volume)
```

**Dashboard queries** should use:
- Prometheus for system/latency metrics (already the case)
- Typed Postgres/TimescaleDB fact tables for business metrics (already the case for v2 dashboards)
- `mm_events` JSONB only for ad-hoc debugging (already the design intent)

---

## 10. Control Plane Architecture

The existing `RuntimeControlPlane` + `mm/api/admin.py` is the right foundation. The gaps for a multi-strategy platform are:

**Strategy lifecycle management** — the admin API needs endpoints to:
- Register a new strategy instance (write to `strategies` table)
- Deploy (provision config, create worker assignment)
- Start/stop/pause/resume (emit to `admin:commands` topic for the target actor)
- Query status (read from `strategies` table + live Prometheus state)

**Exchange and credential management** — a simple CRUD API backed by Postgres, with credentials stored as references to secrets (Kubernetes Secrets or Vault), not inline JSON. The `PrivateRuntimeConfig` credential fields map directly to a `credentials` table.

**Global kill switch** — already exists in `RuntimeControlPlane`. At multi-process scale, the kill switch command must be emitted to the Redis `admin:commands` stream so all strategy workers pick it up. The existing `RedisStreamsBus` fanout handles this.

**Operational commands** — the existing `admin:commands` topic already supports pause/resume actor, kill switch, cancel orders. At multi-process scale, commands need actor routing (already present via `actor_id` in the control plane state).

The control plane service should be the **single source of truth for strategy lifecycle state**. Prometheus is for metrics, not state. The strategies table is the authoritative record.

---

## 11. Dashboard and Operational Monitoring

The v2 Grafana dashboards are already well-designed. For multi-strategy scale:

**Add a strategy overview panel** — a table showing all strategy instances with exchange, symbol, lifecycle status, last decision timestamp, PnL (from `mm_execution_facts`), current inventory, and health status. This is the operator's first view.

**Add exchange health panels** — connectivity status, last market event timestamp per exchange/symbol, websocket reconnect counts. `market:errors:*` topics already emit these events.

**Tag all metrics with `actor_id`** — the existing metrics are tagged with exchange and symbol. Adding `actor_id` labeling enables per-strategy filtering in Grafana without new instrumentation. Check which metrics already carry it; `RuntimeObservabilitySubscriber` should emit all strategy-level metrics with the actor label.

**Instrument the control plane** — deploy/start/stop/pause events should emit to `admin_audit_log` (already exists in the retention config) and to Prometheus counters. Operators need to know when and why a strategy was stopped.

**Technology stack**: keep Grafana + Prometheus + Loki + Postgres. Don't add a custom React dashboard unless Grafana is genuinely insufficient. It isn't for this use case.

---

## 12. Observability at Scale

The existing observability stack is solid. For multi-strategy deployment:

**Metric cardinality**: with 200 strategy instances, labels like `(exchange, symbol, actor_id, stage)` on high-cardinality metrics (action pipeline stages, execution facts) will produce significant cardinality. Use recording rules to pre-aggregate per-strategy metrics into per-exchange and fleet-level summaries. Grafana Alloy + Prometheus remote_write to a managed Prometheus (e.g. Grafana Cloud or Thanos) is appropriate if you need long retention without running your own Prometheus storage.

**Distributed tracing**: OpenTelemetry + Tempo is already in place. At multi-process scale, ensure `run_id` and `frame_id` propagate across process boundaries (market data → strategy worker → execution). The lineage ID scheme already does this logically; wire it into OTLP span attributes.

**Log aggregation**: Loki via Alloy is already in place. At scale, structure logs around `actor_id`, `exchange`, `symbol`, and `frame_id` so you can filter to a single strategy's decision trace.

**Alerting**: the critical alerts for a market-making system are:
- Kill switch active (already a Prometheus gauge)
- Strategy not producing decisions for >N intervals
- Private state stale (already tracked via frame freshness metrics)
- Ledger/order-manager mismatch (already emitted to `ledger:diagnostics:mismatch:*`)
- Execution failure rate above threshold
- Bus queue depth above threshold (already tracked)

Wire these into Prometheus alertmanager rules. Notification routing (PagerDuty, Slack) is your call as the single operator.

---

## 13. Account, Portfolio, Ledger, and Capital Allocation

This is the most architecturally consequential question because the answer constrains everything downstream.

### Recommended model

**Platform → Exchange → Account → Capital Allocation → Strategy**

Specifically:

```
Exchange (Wallex)
  └── Account (Main Account)
        ├── Capital Allocation: 100,000 USDT → BTC MM Strategy
        ├── Capital Allocation: 50,000 USDT  → ETH MM Strategy
        └── Capital Allocation: 200,000 USDT → DOGE MM Strategy
```

Strategies do **not** own balances. They consume allocated capital from the portfolio layer. The portfolio layer reconciles against the exchange account.

### Why not Model B (dedicated subaccount per strategy)?

Subaccounts are operationally clean but most Iranian exchanges (Wallex, Nobitex, Bitpin, Tabdeal) do not support subaccounts in the way that Binance or FTX-era exchanges did. Treat this as unavailable unless you have confirmed exchange support.

### Why not Model C (multiple API keys)?

API keys on the same account share the same balance. You get credential isolation (useful for security) but not balance isolation. Inventory cannot actually be isolated at the key level. This is useful for credential management but not a substitute for capital allocation.

### The three-layer ledger

The system already has the right primitives. Extend them:

**Exchange-Level Ledger** (`VenueLedgerStateMachine`, existing): tracks actual exchange truth — balances, open orders, fills. This is the source of truth. One per exchange account. Managed by `PortfolioLedgerService`.

**Portfolio-Level Ledger** (new, lightweight): tracks capital allocations — how much of the account balance is assigned to each strategy. Written to Postgres. Updated by:
- Administrative allocation changes (via control plane)
- Realized PnL attribution from fills

**Strategy-Level Ledger** (extension of existing): each strategy's `PortfolioLedgerService` view should report against its *allocated capital*, not the full account balance. The risk engine checks `min_cash`, `max_inventory` etc. against allocated limits, not account totals.

### Enforcement

Capital allocation enforcement is **soft** in the strategy worker (the risk engine rejects orders that would exceed allocated limits) and **hard** at the portfolio layer (the control plane can halt a strategy that has consumed its allocation). This is sufficient for a single-operator system. You are not building a multi-tenant exchange.

### PnL attribution

Fills carry `actor_id` and lineage IDs. `mm_execution_facts` already records per-fill attribution. Aggregate `(actor_id, exchange, symbol)` fills over time for PnL per strategy. The portfolio-level ledger tracks cumulative PnL per allocation.

### Reconciliation

The three-layer reconciliation challenge is:
1. Exchange account actual balance (ground truth from `PrivateStateReconciler`)
2. Sum of all strategy allocations (should not exceed account balance)
3. Sum of strategy-attributed inventory (should approximately equal exchange-held inventory)

The existing `LedgerOrderManagerMismatch` detection handles (3) within a strategy. A new portfolio reconciliation job (daily or on-demand via control plane) handles (1) vs (2).

### Impact on current architecture

The `StrategyRunner` today requires a `JointStrategyFrame` with a ledger. This continues to work — the `VenueLedgerStateMachine` still reflects exchange truth. What changes is that the **risk engine receives allocation limits** (from the capital allocation record) rather than reading raw account balance. This is a targeted change to how `NetDeltaRiskEngine` receives its limits — instead of deriving them from full account balance, they come from the strategy's allocation record.

This is a manageable addition, not a redesign. The `ControllerRuntimeConfig` already carries risk limits; you're making those limits allocation-aware.

---

## Summary: Recommended Path

| Priority | Action | Effort |
|---|---|---|
| 1 | Build strategy registry (Postgres tables + admin API CRUD) | 1–2 weeks |
| 2 | Extract Market Data as shared service per exchange | 1–2 weeks |
| 3 | Strategy supervisor process (start/stop workers from registry) | 1–2 weeks |
| 4 | Consolidate 2–8 strategies per worker process | 2–3 weeks |
| 5 | Capital allocation layer (portfolio ledger + risk limit wiring) | 2–3 weeks |
| 6 | Containerize and deploy on Kubernetes | 1–2 weeks |
| 7 | TimescaleDB hypertables for market/optimizer facts | 3–5 days |
| 8 | S3/MinIO Parquet archive target | 3–5 days |
| 9 | GPU optimizer path (if n_samples needs to grow) | 1–2 weeks |

Kafka, separate risk/execution/ledger services, ClickHouse, MongoDB, and service meshes do not appear anywhere on this list. The architecture you need is already largely present in the codebase. The work is integration, lifecycle management, and operational packaging — not a rewrite.