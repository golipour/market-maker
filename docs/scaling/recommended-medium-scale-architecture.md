# Recommended Medium-Scale Market-Making Architecture

This document synthesizes the expert proposals in `docs/scaling/` against the current code-derived architecture. The goal is a practical platform for roughly 10-200 active strategy instances, 5-20 exchanges, many symbols, one operator, and moderate infrastructure budget.

The best solution is not a large microservice fleet. The best solution is a modestly decomposed, event-driven platform that keeps the trading decision loop local, extracts shared market data and persistence, adds a real control plane, and introduces portfolio/capital allocation as the main new domain concept.

## Executive Recommendation

Build toward this target:

```text
                 Control Plane / Strategy Registry
                  Postgres metadata + admin API
                            |
                     admin:commands
                            |
                        Redis Streams
                            |
      +---------------------+----------------------+
      |                     |                      |
 Market Data Services   Strategy Worker Pods   Persistence Service
 per exchange/symbol    5-20 strategies each   Postgres/Parquet writer
      |                     |
 market:events:*       decisions, target books,
                       risk, execution, diagnostics
```

The important boundaries are:

- Extract market data ingestion into shared services.
- Run persistence writers as independent subscribers.
- Promote the admin API/control plane into a real service backed by Postgres.
- Run strategy workers as grouped pods/processes, usually by exchange account, not one pod per strategy.
- Keep frame assembly, MPC, local risk, target-book planning, and order-management tightly coupled inside the worker for now.
- Add a portfolio/capital allocation layer so strategies consume allocated capital, not raw exchange account balances.

## What To Keep Together

Keep these inside the strategy worker:

- `StrategyInputFrameAssembler` and `JointStrategyFrameAssembler`.
- `MpcStrategyEngine`.
- `NetDeltaRiskEngine`.
- Target-book preparation and execution gates.
- `OrderManager` and `ExecutionStateReconciler` for the account handled by that worker.
- Strategy diagnostics and lineage publication.

This avoids adding RPC calls inside the frame -> decision -> risk -> execution path. The current code is strongest when these pieces share one `JointStrategyFrame`, one ledger view, and one decision context.

Do not extract a separate risk service now. It adds latency and failure modes without solving the medium-scale problem.

Do not extract a shared execution service immediately. If multiple workers later need to trade through the same exchange account, then extract a per-exchange-account execution service. Until then, group those strategies in the same account worker and keep order state local.

## What To Extract

### 1. Market Data Service

Extract `SymbolEngine`, exchange public adapters, `OrderbookManager`, `RollingTradeBuffer`, `MarketEventBuilder`, and `IntervalScheduler` into a standalone market-data runtime.

Recommended deployment:

- One service per exchange at first.
- Shard by exchange+symbol group only if one exchange service becomes too large.
- Publish the existing versioned `market:events:<exchange>:<symbol>:<interval>` payloads to Redis Streams.
- Optionally publish raw trade/book topics for observability and debugging, but strategies should normally consume low-frequency `MarketEvent` streams.

This removes duplicated REST polls/websocket connections and ensures all strategies see the same normalized book/trade stream.

### 2. Persistence Service

Run `PostgresEventWriter` and `ParquetEventWriter` as standalone consumers of Redis Streams.

Keep the current split:

- Redis Streams for hot short-term delivery.
- Postgres typed fact tables for operational analytics and dashboards.
- Parquet for replay/archive.

Do not add ClickHouse, Druid, Pinot, MongoDB, RabbitMQ, NATS, or Kafka at this scale.

### 3. Control Plane Service

Promote `mm/api/admin.py` and `RuntimeControlPlane` into a central service that owns lifecycle state.

Add a Postgres-backed registry:

- `exchanges`: exchange name, capabilities, status.
- `accounts`: exchange account, credential reference, account role, status.
- `strategy_presets`: versioned preset/config templates.
- `strategies`: strategy id, actor id, exchange account, symbol, preset, config overrides, status.
- `worker_assignments`: worker id, assigned strategy ids, lease/heartbeat.
- `capital_allocations`: portfolio id, strategy id, asset, limit, current allocation, status.
- `admin_audit_log`: operator actions and automated safety actions.

Credentials should be stored as references to Kubernetes Secrets or another secrets manager, not in strategy config JSON.

The control plane should emit existing `admin:commands` over Redis Streams for pause/resume/kill/cancel. It should also own process/pod lifecycle through a supervisor or Kubernetes deployment controller.

## Strategy Execution Model

Use grouped strategy workers:

```text
Strategy Worker Pod
  exchange_account = wallex-main
  strategies = 5-20 strategy instances
  shared Redis bus
  shared private state / ledger for the account
  shared optimizer executor
```

Why this is the best middle ground:

- One pod per strategy is easy but wastes memory and makes Kubernetes management noisy at 100-200 strategies.
- A single giant worker with 50-200 strategies has poor fault isolation and harder resource control.
- 5-20 strategies per worker gives good density while preserving manageable failure domains.
- Grouping by exchange account keeps private state, rate limits, open-order ownership, and account-level risk coherent.

Use one dedicated worker for unusually heavy strategies. The control plane should allow a strategy to be pinned to its own worker when MPC cost, symbol volatility, or operational importance justify it.

Inside a worker:

- Run I/O with asyncio.
- Run each strategy as an isolated task with crash handling and restart/backoff.
- Use the existing decision timeout behavior to hold current orders when MPC overruns.
- Use a shared optimizer executor. Start with thread execution because PyTorch often releases the GIL in tensor kernels; profile before switching to `ProcessPoolExecutor`.
- If CPU contention becomes obvious, test a process pool for CPU-only workloads. If using GPU, keep one process per CUDA context and batch work instead.

## Account, Portfolio, And Allocation Model

This is the most important domain upgrade.

Use:

```text
Platform
  Exchange
    Account
      Portfolio
        Capital Allocation
          Strategy
```

Strategies should not own exchange balances. Strategies should own allocations.

Three ledger views are needed:

- Exchange ledger: actual balances, open orders, fills, and exchange truth. This is current `VenueLedgerStateMachine` territory.
- Portfolio ledger: virtual allocations, portfolio risk budgets, and available capital per strategy/group.
- Strategy ledger: attributed inventory, fills, PnL, and capital consumption per `actor_id`.

Implementation should be incremental:

1. Keep current exchange ledger as source of truth.
2. Add non-enforcing allocation records and dashboards.
3. Feed allocation limits into `NetDeltaRiskEngine`/risk config.
4. Enforce allocation caps in strategy risk.
5. Add portfolio-level halt rules in the control plane.

Dedicated subaccounts are useful when an exchange supports them, but do not assume they exist. Multiple API keys on one account improve access control but do not isolate balances. Virtual allocation is the portable model.

## Event And Messaging Design

Keep Redis Streams.

The current topic model is already close to ideal:

- `market:events:<exchange>:<symbol>:<interval>`
- `ledger:snapshot`
- `strategy:joint:frames:<exchange>:<symbol>:<interval>`
- `strategy:decisions`
- `strategy:target_books:<actor_id>`
- `execution:orders:<exchange>`
- `execution:state:<exchange>`
- `admin:commands`
- `admin:events`

Use Redis Streams for hours-to-days of hot delivery/replay depending on memory and retention settings. Use Postgres/Parquet for longer replay and analytics.

Add Kafka only if at least one of these becomes true:

- Redis is consistently above about 70-80% CPU or memory after sensible stream trimming.
- Event volume is above tens of thousands of events per second for sustained periods.
- You need multi-day market-data replay directly from the message bus.
- You need multi-region replication or many independent consumer applications.

For the stated 10-200 strategy target, Kafka is overkill.

## Persistence And Data Lifecycle

Keep Postgres as the operational database.

Use it for:

- strategy registry and config metadata,
- accounts and credential references,
- allocations and portfolio state,
- execution/order/fill facts,
- current latest state,
- dashboard fact tables,
- admin audit logs.

Use existing Parquet output for:

- cold event archive,
- offline replay,
- model research datasets,
- long retention beyond operational dashboards.

Consider TimescaleDB later as a Postgres extension, not a separate database, when `mm_market_facts`, `mm_optimizer_facts`, or `mm_execution_facts` time-range queries become slow. Start with partitioning, indexes, retention, and the existing typed fact model.

Recommended lifecycle:

```text
Redis Streams      hot delivery and short replay       hours/days
Postgres facts     operational analytics/dashboard     30-180 days
Postgres metadata  config/account/allocation state     permanent
Parquet archive    replay/research/cold storage        long-term
```

## Observability

Keep Grafana, Prometheus, Loki, Tempo, and Postgres dashboards.

Add:

- Fleet overview dashboard: strategy id, exchange, symbol, worker, lifecycle state, last frame, last decision, execution status, inventory, PnL, health.
- Exchange health dashboard: feed status, reconnects, last market event age, private state age, order API error rate.
- Allocation dashboard: allocated capital, used capital, inventory exposure, allocation breaches, per-strategy PnL.
- Control-plane audit dashboard: start/stop/pause/resume/kill/cancel events.

Watch metric cardinality. `actor_id`, `exchange`, and `symbol` labels are useful, but high-cardinality per-order labels should stay in logs/fact tables, not Prometheus metrics.

At multi-process scale, make `run_id`, `actor_id`, `frame_id`, `decision_id`, `target_book_id`, and `execution_plan_id` first-class trace/log fields.

## Optimizations Worth Doing

Do these first:

- Share market data ingestion so books/trades are normalized once.
- Group strategies by exchange account to reduce private-state and execution duplication.
- Tune MPC configs: lower horizon where acceptable, reduce samples where quality permits, use warm-start if available, add early stopping if not already present.
- Add per-worker CPU/memory metrics and decision-latency SLOs before changing compute architecture.
- Use bounded queues and pause live execution when market data or private state falls behind.

Do after profiling:

- Switch MPC executor from thread pool to process pool for CPU-only workloads if CPU contention is real.
- Batch optimizer calls across strategies with compatible model/config shape.
- Use GPU only if optimizer calls are the bottleneck and there is enough batched work to amortize GPU overhead.
- Add TimescaleDB hypertables when Postgres fact queries degrade.
- Add MinIO/S3 archive for Parquet if local disk management becomes annoying.

Avoid for now:

- Kafka.
- Service mesh.
- Separate global risk service.
- Broad execution-service extraction.
- ClickHouse/InfluxDB/Druid/Pinot.
- One Kubernetes pod per strategy as the default.
- Multi-cluster Kubernetes.
- Rust/C++ rewrites before profiling.

## Migration Roadmap

### Phase 0: Stabilize Contracts

- Keep current one-bot runtime working.
- Ensure all strategy-facing events use versioned payload helpers.
- Add tests for cross-process Redis market-event consumption if missing.
- Add or confirm metrics for bus lag, queue depth, decision latency, private-state freshness, and execution failure rate.

### Phase 1: Strategy Registry And Config Control

- Add Postgres tables for exchanges, accounts, strategies, presets, worker assignments, and audit log.
- Extend admin API to CRUD strategy definitions and render a runtime config from registry rows.
- Keep launching one existing runtime process per strategy.
- Store credentials as secret references.

This gives centralized management without changing trading behavior.

### Phase 2: Shared Market Data

- Build a market-data entrypoint that runs `SymbolEngine`/scheduler without strategy runner.
- Publish `market:events:*` to Redis Streams.
- Change strategy runtimes to consume those market events instead of owning public adapters.
- Run old direct-ingestion and new shared-ingestion paths side by side on a few symbols and compare `MarketEvent` outputs.

This is the highest-value scaling step.

### Phase 3: Standalone Persistence

- Move Postgres and Parquet writers out of strategy runtimes.
- Subscribe persistence service to configured topics.
- Confirm replay lineage works from persisted multi-strategy data.
- Keep typed fact tables as dashboard source.

### Phase 4: Multi-Strategy Worker

- Add a worker entrypoint that loads assigned strategy definitions from the registry.
- Run 5-20 strategy instances as isolated asyncio tasks.
- Group by exchange account.
- Add per-strategy restart/backoff.
- Share private reconciler, ledger, and order manager where strategies share one account.
- Keep a fallback mode for one strategy per worker.

### Phase 5: Portfolio Allocation

- Add allocation records and portfolio ledger view.
- Attribute fills and PnL by `actor_id`.
- Show allocation usage in Grafana.
- Feed allocation limits into risk.
- Add control-plane halt rules for allocation breaches.

### Phase 6: Kubernetes Packaging

- Containerize market data, strategy worker, persistence, and control plane.
- Start with one namespace and one cluster.
- Use Kubernetes Secrets first; use Vault later only if rotation/audit needs justify it.
- Use manual scaling initially. Add HPA only after worker metrics prove it is useful.

### Phase 7: Performance Scaling

- Profile MPC decision time under 50, 100, and 200 strategies.
- Tune samples/horizon/epochs before adding hardware.
- Test CPU process pool vs thread pool.
- Add batched optimizer or GPU only if profiling shows the optimizer is the bottleneck.

## Final Shape

The platform should feel like a small, well-run trading system, not a distributed systems laboratory:

- Redis Streams is the event spine.
- Postgres is the operational brain.
- Grafana is the operator cockpit.
- Market data is shared.
- Strategy workers are grouped and account-aware.
- Trading decisions remain local and fast.
- Portfolio allocation prevents strategies from fighting over shared balances.
- Kubernetes is packaging and recovery, not the architecture itself.

That is the highest-leverage path from the current runtime to a real medium-scale market-making platform without burying the system under unnecessary infrastructure.
