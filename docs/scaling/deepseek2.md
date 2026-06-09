# Scaling the Market-Making Platform: Medium-Scale Architecture Review

## Executive Summary

Based on your codebase analysis and scale assumptions (10–200 strategies, 5–20 exchanges, single operator), I recommend a **moderately distributed architecture** that prioritizes operational simplicity over maximum theoretical scalability. The current monolithic "one bot per process" model is actually a solid foundation—the key evolution is adding **control plane orchestration** and **shared market data distribution**, not fragmenting into many microservices.

Your existing use of PostgreSQL, Redis Streams, Prometheus, and containerization puts you in a good position. The migration should be incremental, with each step providing clear value.

---

## 1. Recommended Target Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Control Plane (FastAPI)                           │
│  - Exchange/account/strategy management  - Global kill switch               │
│  - API + WebSocket dashboard             - Command distribution             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                │ Redis Streams (admin:commands)
                                │
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                          Shared Market Data Layer                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Market Data Service (per exchange/symbol, horizontally scalable)   │    │
│  │  - One pod per (exchange, symbol) or per exchange with fanout       │    │
│  │  - Publishes normalized trades/orderbooks to Redis Streams          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                      Redis Streams (market:events:*)
                                │
        ┌───────────────────────┼───────────────────────────────────────────┐
        │                       │                                           │
        ▼                       ▼                                           ▼
┌───────────────┐       ┌───────────────┐                         ┌───────────────┐
│ Strategy Pod  │       │ Strategy Pod  │                         │ Strategy Pod  │
│ (K8s Deployment)      │ (K8s Deployment)                        │ (K8s Deployment)
│               │       │               │                         │               │
│ - Subscribes  │       │ - Subscribes  │                         │ - Subscribes  │
│   to market   │       │   to market   │                         │   to market   │
│   data        │       │   data        │                         │   data        │
│ - Runs 1-N    │       │ - Runs 1-N    │                         │ - Runs 1-N    │
│   strategies  │       │   strategies  │                         │   strategies  │
│   (async)     │       │   (async)     │                         │   (async)     │
│ - Own MPC     │       │ - Own MPC     │                         │ - Own MPC     │
│   optimizer   │       │   optimizer   │                         │   optimizer   │
│ - Sends orders│       │ - Sends orders│                         │ - Sends orders│
└───────┬───────┘       └───────┬───────┘                         └───────┬───────┘
        │                       │                                         │
        └───────────────────────┼─────────────────────────────────────────┘
                                │
                    Redis Streams (execution:orders:*)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Execution Gateway (optional)                        │
│  - Rate limiting per exchange/API key  - Order validation                  │
│  - Credential management               - Failover routing                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Market data is shared, not per-strategy** — This is the biggest win. One feed per (exchange, symbol) instead of N redundant feeds.

2. **Strategies are grouped into worker pods** — Not one pod per strategy (too heavy), not all in one pod (no isolation). Target: 5–20 strategies per worker.

3. **Control plane is centralized but lightweight** — Single FastAPI pod with persistence. No complex orchestration needed.

4. **Postgres remains the primary store** — With careful indexing and partitioning, it handles millions of events. Add TimescaleDB only if time-series queries become painful.

5. **Redis Streams stays as the backbone** — It's sufficient for 200 strategies. Kafka is overkill at this scale.

---

## 2. Which Parts Stay Monolithic vs. Become Scalable

### Remain Embedded (Keep Per-Strategy or Per-Worker)

| Component | Why Keep |
|-----------|----------|
| MPC optimizer | CPU-bound, strategy-specific state; moving it adds network latency |
| Risk engine (strategy-level) | Needs immediate access to live orders and inventory |
| Order manager | Tight coupling with private state reconciliation |
| Strategy-specific ledger views | Per-strategy attribution requires local state |

### Become Independent Services (Extract)

| Component | Scaling Justification |
|-----------|----------------------|
| **Market data ingestion** | Biggest waste today—N strategies each polling the same exchange. Extract to 1 per exchange/symbol. |
| **Control plane API** | Centralized management; low traffic, so single pod is fine. |
| **Credential/secret management** | Security boundary; all strategies read from one place. |
| **Execution gateway (optional)** | Rate limiting across strategies; useful if you hit exchange limits. |

### May Be Extracted Later (If Needed)

| Component | Trigger for Extraction |
|-----------|----------------------|
| Portfolio/ledger service | If you need cross-strategy capital allocation |
| Global risk service | If you need stop-loss across all strategies |
| Time-series DB | If Postgres query performance degrades |

---

## 3. Strategy Execution Model

### Recommended: Multi-Strategy Worker Pods

```python
# Each worker pod runs a StrategyWorker class
class StrategyWorker:
    def __init__(self, worker_id, assigned_strategies):
        self.strategies = assigned_strategies  # 5-20 strategies
        self.runners = {}  # strategy_id -> StrategyRunner
        
    async def run(self):
        # One asyncio task per strategy
        for strategy_config in self.strategies:
            runner = StrategyRunner(strategy_config)
            self.runners[strategy_config.id] = runner
            asyncio.create_task(runner.run_forever())
```

### Why Not One Pod Per Strategy?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **One pod per strategy** | Perfect isolation, easy scaling | Resource overhead (~50-100MB per pod), slow startup, K8s scheduler limits | ❌ Too heavy for 200 strategies |
| **One process per strategy** | Better isolation than threads | Still high overhead, hard to manage 200 processes | ❌ |
| **One thread per strategy** | Low overhead | GIL limits CPU-bound MPC, no fault isolation | ❌ For CPU work |
| **One asyncio task per strategy (recommended)** | Low overhead, shared event loop for I/O | MPC must be in thread pool, one crash could affect others | ✅ Best balance |
| **One worker pod with 5-20 strategies** | Moderate isolation, efficient resource use | Requires graceful handling of per-strategy crashes | ✅ **RECOMMENDED** |

### Fault Isolation Strategy

```python
async def run_strategy_safely(runner):
    try:
        await runner.run()
    except Exception as e:
        # Log, emit metric, but don't crash the worker
        logger.error(f"Strategy {runner.id} crashed: {e}")
        await self.restart_strategy(runner.id)  # Exponential backoff
        # Continue running other strategies
```

### Lifecycle Management

Use Kubernetes for coarse orchestration (start/stop worker pods), but implement fine-grained strategy control within the worker:

```python
# Admin command topics
admin:commands:
  - strategy:start:{id}
  - strategy:stop:{id}
  - strategy:pause:{id}
  - strategy:resume:{id}
  - strategy:reload:{id}
```

---

## 4. Market Data Architecture

### This is Your Biggest Opportunity

**Current waste**: If you run 20 strategies on Wallex BTCUSDT, you have 20 simultaneous WebSocket connections or REST polls to the same exchange.

**Target**: One market data pod per (exchange, symbol) or per exchange with fanout.

### Option A: Per-Exchange Multiplexer (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│  Market Data Service: Wallex                               │
│  - One WebSocket connection per symbol (or one for all)    │
│  - Normalizes trades/books to internal format              │
│  - Publishes to Redis Streams: market:wallex:BTCUSDT:trades│
│  - Publishes to Redis Streams: market:wallex:BTCUSDT:book  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Redis Streams (consumer groups)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Strategy Pod 1    Strategy Pod 2    Strategy Pod 3
```

### Data Freshness Guarantees

Your existing `IntervalScheduler` expects aligned frames. With shared market data, you need to ensure strategies see consistent snapshots:

```python
# Market data service publishes versioned snapshots
market:wallex:BTCUSDT:snapshot:
  sequence: 12345
  timestamp: 1703001234567
  orderbook: {...}
  trades: [...]

# Strategies consume and align locally (as they already do)
```

### Implementation Path

The market data service can reuse your existing `SymbolEngine` and `MarketEventBuilder`—just extract the ingestion loop into a standalone service that publishes events instead of passing them directly to a strategy.

---

## 5. MPC Optimizer Performance

### Where the Time Goes (Based on Your Code)

From `mm/mpc/mpc_tensor_batched.py` and `MpcStrategyEngine.decide()`:

| Operation | Approx Cost | Scaling Factor |
|-----------|-------------|----------------|
| Adam optimization (epochs × horizon) | 60-80% | O(strategies × samples × horizon × epochs) |
| Soft fill simulation | 10-15% | O(strategies × samples × horizon × book_depth) |
| State construction (VWAP, orderbook) | 5-10% | O(strategies × book_depth) |
| Model scoring | 5-10% | O(strategies × models) |

### Algorithm-Level Improvements (Highest ROI)

| Change | Expected Gain | Complexity | Priority |
|--------|---------------|------------|----------|
| **Warm-start optimizer** (previous solution as initial guess) | 40-60% reduction | Medium | ⭐ High |
| **Early stopping** when objective change < threshold | 20-40% reduction | Low | ⭐ High |
| **Reduce samples** (32→16 or 16→8) with better distribution | 50% reduction per halving | Low | ⭐ High |
| **Stochastic sampling** (not all paths every epoch) | 30-50% reduction | Medium | Medium |
| **Vectorized book operations** (already mostly vectorized) | 10-20% | High | Low |

### Implementation-Level Improvements

| Change | Gain | Complexity | Notes |
|--------|------|------------|-------|
| **Move to PyTorch 2.0+ compile** | 20-40% | Low | `torch.compile` on `MonteCarloMPCTensorBatched.forward()` |
| **Numba for fill simulation** | 10-30% | Medium | But PyTorch already efficient |
| **GPU acceleration** | 100-300% (for batches) | High | Only if running many strategies per GPU |
| **Cython/Rust extensions** | 10-50% | High | Not justified at this scale |
| **Native code (C++)** | 50-100% | Very high | Overkill |

**Recommendation**: Start with warm-start, early stopping, and reduced samples. These require minimal code changes and yield immediate gains. Then evaluate if you actually need GPU—unlikely for 200 strategies with moderate MPC settings.

### System-Level Parallelism

```python
# Strategy worker runs MPC in parallel across strategies
async def run_mpc_batch(strategies):
    loop = asyncio.get_event_loop()
    tasks = []
    for strategy in strategies:
        # Run CPU-bound MPC in thread pool (avoids GIL)
        task = loop.run_in_executor(executor, strategy.optimize)
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results
```

With `max_workers = min(CPU_cores * 2, 20)`.

---

## 6. Python Concurrency Model

### Recommended Hybrid Model

```python
# Strategy Worker Architecture
class StrategyWorker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=CPU_COUNT * 2)
        
    async def run_strategy(self, strategy):
        # I/O: asyncio (Redis, network, DB)
        market_data = await self.subscribe_market_data()
        private_state = await self.fetch_private_state()
        
        # CPU-bound MPC: thread pool
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(
            self.executor, 
            strategy.mpc_optimize, 
            market_data, private_state
        )
        
        # Execution: asyncio
        await self.send_orders(decision)
```

### GIL Impact Analysis

| Workload | GIL Impact | Mitigation |
|----------|------------|-------------|
| Market data ingestion (WebSocket) | None (I/O) | Already using asyncio |
| Order placement (REST) | None (I/O) | asyncio |
| Private state reconciliation | None (mostly I/O + dict ops) | Keep asyncio |
| **MPC optimization** | **High (PyTorch CPU)** | Use thread pool or multiprocessing |
| Risk calculation | Low (mostly numeric) | asyncio fine |

### Recommendation

- **Keep existing asyncio architecture** for I/O workloads (it's well-designed)
- **Run MPC in `run_in_executor()` thread pool**—PyTorch releases GIL for tensor operations
- **Don't use multiprocessing** for per-strategy MPC—overhead of IPC exceeds benefit
- **If one strategy dominates CPU**, move it to a dedicated worker pod

---

## 7. Containerization & Kubernetes Deployment

### Service Decomposition

| Service | Replicas | Resources | Horizontal Scaling |
|---------|----------|-----------|---------------------|
| **Market Data Service** (per exchange) | 1 per exchange or symbol | 0.5-2 CPU, 1-4GB | By exchange/symbol |
| **Strategy Worker** (pool) | 5-20 (depending on strategy count) | 1-4 CPU, 2-8GB | Add workers as strategies increase |
| **Control Plane API** | 1-2 | 0.5 CPU, 1GB | Limited (low traffic) |
| **PostgreSQL** | 1 primary (+ read replica optional) | 2-8 CPU, 8-32GB | Vertical scaling |
| **Redis** | 1-3 (sentinel for HA) | 1-2 CPU, 4-8GB | Vertical scaling |
| **Prometheus/Grafana** | 1 each | 1-2 CPU, 4-8GB | Not needed |
| **Dashboard Frontend** | 1 | 0.5 CPU, 512MB | Not needed |

### Kubernetes Configuration Sketch

```yaml
# Strategy Worker Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategy-worker
spec:
  replicas: 5  # Adjust based on strategy count
  template:
    spec:
      containers:
      - name: worker
        image: mm-strategy-worker:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: ASSIGNED_STRATEGIES
          valueFrom:
            configMapKeyRef:
              name: strategy-assignments
              key: strategies
---
# Market Data Service (per exchange)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-data-wallex
spec:
  replicas: 1  # One per exchange/symbol; use HPA if needed
  template:
    spec:
      containers:
      - name: market-data
        image: mm-market-data:latest
        args: ["--exchange", "wallex", "--symbols", "BTCUSDT,ETHUSDT"]
```

### Service Discovery

Keep it simple: use Kubernetes DNS and environment variables.

```python
# Strategy worker finds market data service
market_data_host = os.getenv("MARKET_DATA_WALLEX_HOST", "market-data-wallex.default.svc.cluster.local")
```

### Autoscaling

```yaml
# Horizontal Pod Autoscaler for strategy workers
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: strategy-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: strategy-worker
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: strategy_queue_depth
      target:
        type: AverageValue
        averageValue: 100
```

---

## 8. Event & Messaging Architecture

### Redis Streams is Sufficient—Don't Add Kafka

| Requirement | Your Scale (200 strategies) | Redis Streams Capability |
|-------------|----------------------------|--------------------------|
| Throughput | ~1,000-10,000 events/sec | 100,000+ events/sec ✅ |
| Consumer groups | 200 strategies | Unlimited ✅ |
| Persistence | 7-30 days | AOF/RDB ✅ |
| Ordering | Per-stream | Per-stream ✅ |
| Backpressure | Consumer lag monitoring | XINFO STREAM ✅ |

**Stay with Redis Streams.** Kafka would add significant operational complexity for minimal benefit at your scale.

### Stream Design

```
# Market data (per symbol)
market:events:wallex:BTCUSDT:        # Consumer group: strategy-workers
market:events:wallex:ETHUSDT:
market:events:nobitex:BTCUSDT:

# Private state (per exchange)
private:wallex:snapshot:
private:wallex:fills:

# Control plane (global)
admin:commands:                      # Consumer group: all workers
admin:events:                        # Notifications back to control plane

# Execution (per exchange)
execution:wallex:orders:

# Diagnostics (global)
strategy:diagnostics:
```

### Consumer Group Scaling

```python
# Each strategy worker joins the consumer group for its assigned streams
await bus.consumer_group(
    stream="market:events:wallex:BTCUSDT",
    group="strategy-workers",
    consumer=f"worker-{worker_id}"
)
```

### Backpressure Handling

Your existing `AbstractEventBus` with bounded queues and backpressure policies (`block`, `drop_oldest`, `drop_newest`) is well-designed. Keep it.

---

## 9. Persistence Architecture

### PostgreSQL Remains the Primary Store—Add TimescaleDB If Needed

| Data Type | Current Store | Recommended | Justification |
|-----------|---------------|-------------|---------------|
| Operational metadata (exchanges, accounts, strategies) | Postgres | **Postgres** | Relational, low volume |
| Execution records (orders, fills) | Postgres (mm_execution_facts) | **Postgres** | Transactional, audit-critical |
| Ledger snapshots | Postgres (mm_latest_state) | **Postgres** | Current state only |
| **Market events** (aggregated) | Postgres (mm_market_facts) | **TimescaleDB** or Postgres partitioning | High volume, time-series |
| **Optimizer diagnostics** | Postgres (mm_optimizer_facts) | **Postgres** (with partitioning) | Medium volume |
| Raw market data (trades, orderbooks) | Parquet (object storage) | **MinIO/S3** ✅ | You already have this pattern |
| Archived events | Parquet/MinIO | **MinIO/S3** | You already have this |
| Metrics | Prometheus | **Prometheus** | Keep as is |
| Logs | Loki | **Loki** | Keep as is |

### Partitioning Strategy for Postgres

```sql
-- Partition mm_market_facts by time
CREATE TABLE mm_market_facts (
    id BIGSERIAL,
    ts TIMESTAMPTZ NOT NULL,
    exchange TEXT,
    symbol TEXT,
    -- ... other columns
) PARTITION BY RANGE (ts);

-- Create monthly partitions
CREATE TABLE mm_market_facts_2024_01 PARTITION OF mm_market_facts
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Migration Path from Current PostgreSQL-centric Design

Your existing schema with typed fact tables (`mm_market_facts`, `mm_execution_facts`, etc.) is already well-designed for scale. The main limitations are:

1. **No automatic partitioning** — Add it when insert volume exceeds ~1M rows/day
2. **Single writer** — Fine for your scale; read replicas if dashboard queries impact performance

### When to Add TimescaleDB

| Trigger | Action |
|---------|--------|
| `mm_market_facts` > 100M rows | Migrate to TimescaleDB (hypertable) |
| Queries on > 30 days of market data take > 5 seconds | Add TimescaleDB compression |
| You need continuous aggregates (e.g., 1-min bars from ticks) | TimescaleDB justified |

### Object Storage (MinIO/S3) for Raw Data

Your `ParquetEventWriter` is excellent. Keep it and consider:

```yaml
# Retention policy
raw_market_data: 7 days    # Enough for replay
aggregated_facts: 90 days  # Postgres/TimescaleDB
parquet_archive: 1 year+   # MinIO/S3 with lifecycle rules
```

---

## 10. Account/Portfolio/Capital Allocation Architecture

This is a **critical design decision** that affects many components. Based on your scale and operational model, here's my recommendation:

### Recommended Model: Portfolio Layer with Virtual Allocation

```
Platform
  → Exchange (e.g., Wallex)
    → Account (e.g., "Main Trading Account")
      → Portfolio (e.g., "BTC Portfolio", "ETH Portfolio")
        → Capital Allocation (e.g., 100,000 USDT to BTC strategy)
          → Strategy Instance (e.g., MM-BTC-Wallex)
```

### Why Not Simpler Models?

| Model | Problem for Your Use Case |
|-------|--------------------------|
| **Shared Account (no allocation)** | No PnL attribution, strategies compete for funds, risk hard to manage |
| **Dedicated Subaccount Per Strategy** | Most exchanges don't support enough subaccounts (10-20 max) |
| **Multiple API Keys Per Account** | Balances still shared underneath; provides no real isolation |

### Virtual Allocation Design

```python
# Portfolio ledger tracks allocations on top of real exchange balances
class PortfolioLedger:
    def __init__(self, exchange, account_id):
        self.real_balance = {}  # From exchange
        self.allocations = {}    # strategy_id -> VirtualBalance
        
    def allocate(self, strategy_id, base_allocation, quote_allocation):
        """Virtual allocation - no exchange action"""
        self.allocations[strategy_id] = VirtualBalance(base_allocation, quote_allocation)
        
    def check_risk(self, strategy_id, required_base, required_quote):
        allocation = self.allocations[strategy_id]
        return (required_base <= allocation.available_base and
                required_quote <= allocation.available_quote)
```

### PnL Attribution

```python
# Track realized and unrealized PnL per strategy
class StrategyPnL:
    def __init__(self, strategy_id, allocation):
        self.allocation = allocation
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
    def on_fill(self, fill):
        # Attribute to this strategy
        self.realized_pnl += fill.pnl_contribution
```

### Reconciliation

The portfolio ledger reconciles against the exchange's real balance:

```python
async def reconcile():
    real = await exchange.get_balances()
    total_allocated = sum(a.base + a.quote for a in allocations.values())
    discrepancy = real.total - total_allocated
    # Either adjust allocations proportionally or flag for investigation
```

### Implementation Impact

This model requires changes to:

| Component | Change Required |
|-----------|-----------------|
| `PortfolioLedgerService` | Add allocation layer above venue ledger |
| `StrategyRunner` | Check allocation before executing orders |
| Risk engine | Validate against allocated balance, not real balance |
| `PrivateStateReconciler` | Continue tracking real balances |
| Admin API | Add allocation management endpoints |

### Migration Path

1. **Phase 1**: Keep current direct strategy→account model
2. **Phase 2**: Add virtual allocation tracking (non-enforcing, just monitoring)
3. **Phase 3**: Enforce allocations in risk engine
4. **Phase 4**: Add dynamic rebalancing

---

## 11. Administrative API & Control Plane

### Recommended Architecture

```python
# mm/control_plane/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# State stored in Postgres (strategies, exchanges, allocations)
# Commands published to Redis Streams

class StrategyCreate(BaseModel):
    name: str
    exchange: str
    symbol: str
    allocation_usdt: float
    config_overrides: dict = {}

@app.post("/api/v1/strategies")
async def create_strategy(req: StrategyCreate):
    # 1. Write to Postgres
    strategy_id = await db.insert_strategy(req)
    # 2. Publish command to Redis for worker assignment
    await bus.publish("admin:commands", {
        "type": "strategy.create",
        "strategy_id": strategy_id,
        **req.dict()
    })
    return {"strategy_id": strategy_id}

@app.post("/api/v1/strategies/{id}/start")
async def start_strategy(id: str):
    await bus.publish("admin:commands", {
        "type": "strategy.start",
        "strategy_id": id
    })
    return {"status": "start_pending"}

@app.post("/api/v1/kill-switch/enable")
async def enable_kill_switch():
    await bus.publish("admin:commands", {
        "type": "kill_switch.enable",
        "reason": "operator_request"
    })
    return {"status": "kill_switch_enabled"}
```

### Worker Side Command Handling

```python
class StrategyWorker:
    async def handle_admin_commands(self):
        async for command in bus.subscribe("admin:commands"):
            if command["type"] == "strategy.create":
                await self.create_strategy(command["strategy_id"], command)
            elif command["type"] == "strategy.start":
                await self.start_strategy(command["strategy_id"])
            elif command["type"] == "kill_switch.enable":
                self.kill_switch = True
                await self.cancel_all_orders()
```

### Required Endpoints

| Category | Endpoints |
|----------|-----------|
| **Exchange Management** | POST /exchanges, GET /exchanges, PUT /exchanges/{id}/credentials |
| **Account Management** | POST /accounts, GET /accounts, GET /accounts/{id}/balances |
| **Strategy Management** | POST /strategies, GET /strategies, PUT /strategies/{id}/config, POST /strategies/{id}/start/stop/pause |
| **Operations** | POST /kill-switch, GET /health, GET /metrics, POST /emergency-shutdown |

---

## 12. Dashboard & Operational Monitoring

### Recommended Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Dashboard Backend** | FastAPI + WebSocket | Reuse existing patterns |
| **Dashboard Frontend** | React + Vite | Lightweight, good ecosystem |
| **Charts** | Plotly or ECharts | Interactive, good for finance |
| **Real-time updates** | WebSocket + Redis Streams | Your bus already supports this |

### Dashboard Data Flow

```
Postgres (facts) ──┐
                    ├──► Dashboard API ──► Frontend
Redis (live) ──────┘         │
                             ▼
                        WebSocket push
```

### Key Metrics to Surface

**First screen (operator needs to see immediately)**:

| Metric | Source |
|--------|--------|
| Active strategies / total | Postgres + Redis |
| Total PnL (today) | `mm_execution_facts` + ledger |
| Strategies in risk violation | Risk engine events |
| Exchange connectivity status | Market data service health |
| Kill switch status | Redis + Prometheus |

**Strategy detail**:

| Metric | Source |
|--------|--------|
| Current inventory | Ledger snapshot |
| PnL (realized + unrealized) | Ledger + execution facts |
| Open orders | Live orders + pending |
| Last decision latency | `mm_optimizer_facts` |
| MPC objective value | `mm_optimizer_facts` |
| Fill rate | `mm_execution_facts` |

### Reuse Your Grafana Dashboards

Your existing v2 dashboards are comprehensive. Don't rebuild everything—extend them:

- Add cross-strategy comparison panels
- Add allocation utilization views
- Add strategy health scores

---

## 13. Migration Roadmap

### Phase 0: Preparation (Week 1-2) — No Breaking Changes

| Task | Success Criteria |
|------|------------------|
| Add strategy ID to all event payloads | Every event has `strategy_id` field |
| Add allocation fields to ledger schema | `mm_allocations` table |
| Implement graceful shutdown for `StrategyRunner` | Can pause/resume without process exit |
| Add `--worker-id` and `--assigned-strategies` CLI args | Multi-strategy mode possible |

### Phase 1: Shared Market Data (Week 3-4) — Biggest Win

```bash
# Extract market data ingestion
git mv mm/ingestion/symbol_engine.py services/market-data/
# Create standalone entrypoint
python -m services.market_data --exchange wallex --symbols BTCUSDT,ETHUSDT

# Strategies now consume from Redis instead of local SymbolEngine
```

**Validation**: Run 1 market data pod + 3 strategy workers, each running 5 strategies. Should use 1/3 of previous exchange connections.

### Phase 2: Strategy Worker Pool (Week 5-6)

```python
# mm/runtime/worker.py
class StrategyWorker:
    def __init__(self, assigned_strategies: List[StrategyConfig]):
        self.strategies = assigned_strategies
        # Reuse existing StrategyRunner per strategy
        
    async def run(self):
        tasks = [self._run_strategy(s) for s in self.strategies]
        await asyncio.gather(*tasks)
```

**Validation**: 10 workers, 100 strategies running simultaneously.

### Phase 3: Control Plane (Week 7-8)

- Implement FastAPI admin API
- Add strategy CRUD endpoints
- Implement command distribution to workers

**Validation**: Create/start/stop strategy via API without restarting workers.

### Phase 4: Portfolio & Allocations (Week 9-10)

- Add `PortfolioLedger` with virtual allocations
- Modify risk engine to check allocations
- Add PnL attribution

**Validation**: Two strategies share one exchange account, each respects its allocation.

### Phase 5: Kubernetes Deployment (Week 11-12)

- Containerize all services
- Create Helm charts or kustomize configs
- Set up HPA for strategy workers
- Configure Prometheus/Grafana for K8s

**Validation**: Can scale workers from 3→10 replicas; strategies redistribute.

---

## 14. Infrastructure Additions: Justified vs. Over-Engineering

### Justified (Add)

| Infrastructure | Justification | Priority |
|----------------|---------------|----------|
| **PostgreSQL read replica** | Dashboard queries impacting strategy performance | Medium |
| **TimescaleDB** (if >100M market events) | Time-series queries getting slow | Medium |
| **MinIO/S3** (you already use Parquet) | Long-term archive, replay datasets | Low (already have) |
| **Redis Sentinel** | High availability for market data distribution | Low (can tolerate downtime) |
| **kubectl + Helm** | Manage K8s deployments | High |

### Unnecessary Over-Engineering (Avoid)

| Infrastructure | Why Avoid |
|----------------|-----------|
| **Kafka cluster** | Redis Streams is sufficient; Kafka adds Zookeeper/KRaft, topic management, operational burden |
| **Microservice fleet** | 200 strategies don't need 20 services; debugging becomes nightmare |
| **Service mesh (Istio, Linkerd)** | No multi-team, no fine-grained auth, no need |
| **Multiple databases** (ClickHouse + Postgres + MongoDB) | One operational DB + object storage is enough |
| **Distributed tracing (full Jaeger)** | You have OTLP; only enable sampling for debugging |
| **Separate data lake (Trino, Iceberg)** | Parquet + DuckDB for offline analysis is sufficient |
| **Airflow for ETL** | Retention scripts + cron/kubernetes jobs are fine |
| **Message queue (RabbitMQ, NATS)** | Redis Streams already handles your use case |

### The "Wait and See" List

| Infrastructure | Trigger to Add |
|----------------|----------------|
| **TimescaleDB** | Market event queries > 30 days take > 5 seconds |
| **Read replica** | Dashboard queries cause strategy frame latency > 10ms |
| **Redis Cluster** | Single Redis reaches 80% CPU/memory |
| **Kafka** | You exceed 50,000 events/sec or need multi-region |
| **Full Jaeger** | You can't debug a rare issue with logs + metrics |

---

## 15. Scaling Limits: Where Your Architecture Breaks

| Component | Limit | Beyond That |
|-----------|-------|--------------|
| **Single Redis** | ~50,000 events/sec or ~20GB memory | Add Redis Cluster |
| **Postgres (single primary)** | ~10,000 writes/sec, ~50M rows per fact table | Read replica + partitioning + TimescaleDB |
| **Strategy worker (CPU-bound MPC)** | ~10-20 strategies per worker (depending on MPC settings) | Add more workers |
| **Market data pod** | ~100 symbols per exchange (API rate limits first) | Multiple pods per exchange |
| **Kubernetes cluster** | ~1,000 pods (you'll have ~50) | Fine |

Your stated scale (200 strategies, 20 exchanges, hundreds of symbols) fits comfortably within these limits.

---

## Summary: The 80/20 Recommendation

If you only do **three things** from this analysis:

1. **Extract market data to shared services** — Reduces redundant connections, ensures data consistency, biggest operational win.

2. **Implement multi-strategy workers** (5-20 strategies per worker) — Efficient resource use without over-engineering.

3. **Add control plane API** — Centralized management is the core requirement you stated.

Everything else can be incrementally added when needed. Keep your existing architecture—it's well-designed for the problem. The current codebase's use of event sourcing, typed fact tables, Redis Streams, and asyncio is **already at a medium-scale production level**. You're closer than you think.