# Scaling the Market-Making Platform: Target Architecture & Migration Roadmap

## Executive Summary

Your current architecture is well-structured for a single bot but has tight coupling between market data, strategy execution, and state management that won't scale. The target should be a **control plane / data plane separation** where:

- **Control Plane**: Orchestration, configuration, admin API, dashboard (stateless, scales horizontally)
- **Data Plane**: Strategy execution workers (stateful, partitioned by strategy)

**Core recommendation**: One strategy instance per container/pod, with market data extracted as an independent service. Use Kubernetes for orchestration, Redis Streams for event distribution (evolve to Kafka for higher scale), and accept that the MPC optimizer will drive GPU acceleration needs.

---

## 1. Target System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL PLANE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Admin   │  │ Dashboard│  │  Config  │  │  Secret  │  │  Alert   │         │
│  │   API    │  │ Backend  │  │  Store   │  │   Store  │  │ Manager  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │             │             │             │               │
│  ┌────┴─────────────┴─────────────┴─────────────┴─────────────┴─────┐         │
│  │                    Orchestration Layer                            │         │
│  │          (Deployment, Scaling, Lifecycle Management)              │         │
│  └───────────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ gRPC/REST + Event Bus
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DATA PLANE                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    MARKET DATA SERVICE (Shared)                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │   │
│  │  │ WebSocket│  │   REST   │  │Normalize │  │  Fan-out │               │   │
│  │  │  Farm    │  │ Poller   │  │  & Cache │  │  Router  │               │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                      │
│                                          │ Redis Streams / Kafka                │
│                                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    STRATEGY WORKER POOL                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ Strategy Pod │  │ Strategy Pod │  │ Strategy Pod │  │ Strategy Pod │ │   │
│  │  │   (BTC-USDT) │  │   (ETH-USDT) │  │   (DOGE-IRT) │  │     ...      │ │   │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │ │   │
│  │  │  │Frame   │  │  │  │Frame   │  │  │  │Frame   │  │  │  │Frame   │  │ │   │
│  │  │  │Assemble│  │  │  │Assemble│  │  │  │Assemble│  │  │  │Assemble│  │ │   │
│  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │ │   │
│  │  │  │MPC     │  │  │  │MPC     │  │  │  │MPC     │  │  │  │MPC     │  │ │   │
│  │  │  │Optimize│  │  │  │Optimize│  │  │  │Optimize│  │  │  │Optimize│  │ │   │
│  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │ │   │
│  │  │  │Risk    │  │  │  │Risk    │  │  │  │Risk    │  │  │  │Risk    │  │ │   │
│  │  │  │Engine  │  │  │  │Engine  │  │  │  │Engine  │  │  │  │Engine  │  │ │   │
│  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │  │  ├────────┤  │ │   │
│  │  │  │Order   │  │  │  │Order   │  │  │  │Order   │  │  │  │Order   │  │ │   │
│  │  │  │Manager │  │  │  │Manager │  │  │  │Manager │  │  │  │Manager │  │ │   │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                      │
│                                          │ gRPC (state sync)                    │
│                                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    EXECUTION SERVICE (Optional)                          │   │
│  │         Centralized order routing & rate limiting                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PERSISTENCE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Postgres │  │  Parquet │  │  Redis   │  │  Influx  │  │    S3    │         │
│  │(Events/  │  │ (Archive/│  │ (State/  │  │ (Metrics/│  │ (Long-   │         │
│  │ State)   │  │  Replay) │  │  Cache)  │  │  Times-  │  │  term)   │         │
│  └──────────┘  └──────────┘  └──────────┘  │  series) │  └──────────┘         │
│                                            └──────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Boundaries

| Service | Scale | State | Deployed As | Communication |
|---------|-------|-------|-------------|---------------|
| **Market Data Service** | 1-3 replicas per exchange | In-memory cache + Redis | Deployment | Redis Streams |
| **Strategy Worker** | 1 per strategy (100s-1000s) | Internal (ledger/orders) + Redis snapshot | Pod (1 container) | gRPC + Bus |
| **Execution Service** | 1-3 replicas per exchange | Stateless (rate limit state in Redis) | Deployment | gRPC |
| **Admin API** | 2+ replicas | Stateless | Deployment | REST/gRPC |
| **Orchestrator** | 1 leader + standby | Kubernetes primitives | Deployment (leader election) | Kubernetes API |
| **Dashboard Backend** | 2+ replicas | Stateless (cache in Redis) | Deployment | WebSocket/REST |
| **Persistence Writers** | 1 per sink type | Stateless (buffers in Redis) | Deployment | Event Bus |

### Components That Remain Embedded Inside Strategy Workers

- **Frame assembly** (market-specific, low latency)
- **MPC optimizer** (compute-heavy, strategy-specific state)
- **Risk engine** (needs full strategy context)
- **Order manager** (needs pending state tracking)
- **Ledger for that strategy** (source of truth for attribution)

### Components That Become Independent Services

- **Market data normalization & distribution** (shared across strategies)
- **Exchange connectivity** (WebSocket farm consolidation)
- **Execution routing** (rate limiting, anti-spoofing across strategies)
- **Portfolio/account aggregation** (cross-strategy risk)
- **Persistence** (decoupled write path)
- **Admin & orchestration**

---

## 2. Strategy Execution Model

### Recommended: One Strategy Per Container (Pod)

```yaml
# Strategy pod spec
apiVersion: v1
kind: Pod
metadata:
  labels:
    strategy-id: "mm-wallex-btcusdt-001"
    exchange: "wallex"
    symbol: "BTCUSDT"
spec:
  containers:
  - name: strategy-worker
    image: mm-strategy-worker:v2
    env:
    - name: STRATEGY_ID
      value: "mm-wallex-btcusdt-001"
    - name: EXCHANGE
      value: "wallex"
    - name: SYMBOL
      value: "BTCUSDT"
    - name: ACCOUNT_ID
      valueFrom:
        secretKeyRef:
          name: exchange-credentials
          key: wallex-main-account
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
```

### Rationale

| Aspect | One pod per strategy | Many strategies per pod |
|--------|---------------------|------------------------|
| **Fault isolation** | ✅ Crash kills one strategy | ❌ Crash kills many |
| **Resource limits** | ✅ Per-strategy CPU/memory | ❌ Noisy neighbor |
| **Scaling** | ✅ Horizontal by strategy count | ❌ Uneven load |
| **Lifecycle management** | ✅ Start/stop independent | ❌ Complex |
| **Overhead** | ~50-100 MB per pod | ✅ Lower (10-20 MB per strategy) |

**Decision**: Start with one pod per strategy for isolation and simplicity. Accept the overhead. When you exceed ~500 strategies, consider batching low-volume strategies (e.g., illiquid pairs) 5-10 per pod.

### Strategy Lifecycle Management

```python
# Orchestrator manages strategy pods via Kubernetes API
class StrategyOrchestrator:
    async def deploy_strategy(self, config: StrategyConfig) -> str:
        # 1. Validate config
        # 2. Create Kubernetes deployment/pod
        # 3. Inject secrets via CSI driver or env from Secret
        # 4. Return strategy_id
        
    async def stop_strategy(self, strategy_id: str):
        # Scale deployment to 0 or delete pod
        # Preserve persistent state (checkpoints)
        
    async def resume_strategy(self, strategy_id: str):
        # Restore from last checkpoint
        # Scale deployment to 1
        
    async def update_strategy(self, strategy_id: str, new_config: StrategyConfig):
        # Rolling update: new pod, drain old
        # Preserve ledger state via handoff
```

### Fault Tolerance

- **Pod death**: Kubernetes restarts; strategy resumes from last persisted state (checkpoint every N frames)
- **Stale market data**: Pod health check fails → Kubernetes restarts
- **Exchange disconnect**: Exponential backoff within strategy worker; no pod restart needed
- **MPC optimizer hang**: Separate timeout + fallback; kill optimizer thread/process

---

## 3. Market Data Architecture

### Recommended: Independent Market Data Service

```
Exchange WebSocket → Market Data Service → Normalized Stream → Redis Streams → Strategy Workers
                                              ↓
                                        Parquet (raw archive)
```

### Service Design

```python
# Market Data Service responsibilities:
class MarketDataService:
    # Per exchange/symbol feed manager
    feeds: dict[str, FeedManager]
    
    # Normalization layer (reuse your existing adapters!)
    normalizers: dict[str, AbstractMarketDataAdapter]
    
    # Fan-out to Redis Streams
    publisher: EventPublisher
    
    async def run(self):
        # Single WebSocket connection per exchange/symbol
        # Reuse across all strategies subscribed to that symbol
        pass
```

### Benefits

| Current | Proposed |
|---------|----------|
| Each strategy maintains its own WebSocket | One connection per exchange/symbol across all strategies |
| 100 strategies × 2 exchanges = 200 connections | 2 connections total |
| Duplicate orderbook building per strategy | Single normalized book, distributed |
| Inconsistent snapshots across strategies | Consistent view for all strategies |

### Event Distribution Technology

| Scale | Recommended | Why |
|-------|-------------|-----|
| < 50 strategies | Redis Streams (you already have it) | Simple, low latency, works |
| 50-500 strategies | Redis Streams + consumer groups | Partitioned consumption |
| 500+ strategies | Kafka | Higher throughput, longer retention, replay |

**Start with Redis Streams** (you already have the code). Add Kafka as a configurable backend when you exceed 100 strategies.

### Data Partitioning

```
Topic pattern: market.{exchange}.{symbol}.{stream_type}
Example: market.wallex.BTCUSDT.orderbook

Partition key: hash(exchange, symbol) % num_partitions
```

This ensures all events for a given symbol go to the same partition, preserving order.

---

## 4. Performance and Compute Scaling (MPC Optimizer)

### Bottleneck Analysis

Your `MonteCarloMPCTensorBatched` with typical settings:
- `n_samples=64`, `horizon=1`, `adam_epochs=30` → ~30,000 forward passes per decision
- At 1-second intervals → ~30,000 tensor operations/second per strategy

**100 strategies** → 3M ops/sec → likely CPU saturation.

### Performance Gains Priority

| Priority | Approach | Expected Gain | Implementation Effort |
|----------|----------|---------------|----------------------|
| 1 | **Vectorize across strategies** (batch same model) | 5-10x | Medium |
| 2 | **GPU acceleration** (move PyTorch to CUDA) | 10-50x | Low (existing PyTorch) |
| 3 | **Reduce `n_samples` with better sampling** (antithetic, QMC) | 2-4x | Medium |
| 4 | **JIT compile with `torch.compile`** | 1.5-2x | Low |
| 5 | **Numba for non-PyTorch parts** | 1.2-1.5x | Medium |
| 6 | **Rust extension for simulation kernel** | 2-3x | High |

### Recommended Architecture

```python
# Batch optimizer across strategies with same model
class BatchedMPCOptimizer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    async def optimize_batch(self, states: list[State]) -> list[Action]:
        # Stack states into batch tensors
        batch_states = self._stack_states(states)
        batch_states = batch_states.to(self.device)
        
        # Single batched optimization call
        batch_actions = self._batched_optimize(batch_states)
        
        return batch_actions
```

### Deployment Model for Compute

| Scale | Deployment |
|-------|------------|
| < 20 strategies | CPU, in-process |
| 20-100 strategies | GPU-enabled nodes, 10 strategies per GPU (batched) |
| 100+ strategies | Multi-GPU with model parallelism |

**Specific GPU recommendation**: Start with NVIDIA T4 (cheap, 16GB) → move to A10G (24GB) for larger batches.

### Algorithm-Level Improvements

```python
# 1. Control variates for variance reduction
class ControlVariatesMPC(MonteCarloMPCTensorBatched):
    def sample_paths(self, n_samples):
        # Antithetic sampling: pairs of negatively correlated samples
        base_samples = torch.randn(n_samples // 2, ...)
        return torch.cat([base_samples, -base_samples])
        
# 2. Importance sampling focus on tail events
    def importance_sample(self, n_samples, risk_aversion):
        # Oversample paths with high inventory risk
        pass
        
# 3. Warm-start from previous action
    def warm_start(self, previous_action, learning_rate_reduction=0.5):
        # Initialize optimizer from previous optimum
        pass
```

---

## 5. Python Concurrency Model

### Recommended Hybrid Architecture

```python
class StrategyWorker:
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=1)  # For CPU-bound
    
    async def run(self):
        # Main event loop (asyncio) handles I/O
        asyncio.create_task(self._market_data_consumer())
        asyncio.create_task(self._private_state_consumer())
        asyncio.create_task(self._command_consumer())
        
        while True:
            frame = await self._next_frame()
            
            # Offload MPC to process pool (releases GIL)
            action = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._run_mpc, frame
            )
            
            # Risk and execution back in asyncio
            await self._apply_risk(action)
            await self._execute(action)
```

### I/O Workloads (asyncio ✅)

| Workload | Why asyncio |
|----------|-------------|
| Market data consumption | WebSocket, Redis streams, high concurrency |
| REST API calls | Many concurrent, mostly waiting |
| Event publishing | Non-blocking, high throughput |

### CPU Workloads (multiprocessing ✅)

| Workload | Why multiprocess |
|----------|------------------|
| MPC optimization | PyTorch releases GIL but still benefits from isolation |
| Simulation | NumPy/PyTorch operations, GIL-limited |
| Risk calculations | Mostly vectorized, but isolate for safety |

### Threads: Avoid for CPU work

Threads in Python are unsuitable for CPU-bound work due to GIL. Use only for:
- Legacy blocking I/O libraries
- Wrapped C extensions that release GIL

### Process Pool Strategy

```python
# Shared memory for model parameters (avoid serialization overhead)
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

# Pre-load model in each worker
class MPCWorker:
    def __init__(self):
        self.model = self._load_model()  # Load once per worker
        
    def optimize(self, state: State) -> Action:
        return self.model.optimize(state)
```

---

## 6. Kubernetes Deployment Model

### Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ingress-nginx (LoadBalancer)               │   │
│  └─────────────────┬───────────────┬───────────────────────┘   │
│                    │               │                           │
│         ┌──────────┴──────────┐ ┌──┴──────────────────────┐    │
│         │   Admin API         │ │   Dashboard Frontend    │    │
│         │   (2-3 pods)        │ │   (2-3 pods)            │    │
│         └─────────────────────┘ └─────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Strategy Namespace                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │Strategy │ │Strategy │ │Strategy │ │Strategy │       │   │
│  │  │Pod      │ │Pod      │ │Pod      │ │Pod      │ ...   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              System Namespace                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │Market    │ │Execution │ │Redis     │ │Postgres  │   │   │
│  │  │Data      │ │Service   │ │(Stateful)│ │(Stateful)│   │   │
│  │  │Service   │ │          │ │          │ │          │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Monitoring (kube-prometheus-stack)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Service Scaling Decisions

| Service | Scale Strategy | Max Replicas | HPA Enabled |
|---------|---------------|--------------|-------------|
| **Market Data Service** | Fixed (1-3) | 3 | ❌ |
| **Strategy Workers** | One per strategy | 500+ | ✅ (by strategy count) |
| **Execution Service** | CPU-based HPA | 10 | ✅ |
| **Admin API** | CPU-based HPA | 20 | ✅ |
| **Redis** | StatefulSet, fixed | 3 (cluster) | ❌ |
| **Postgres** | StatefulSet, fixed | 1 primary + replicas | ❌ |

### Autoscaling Configuration

```yaml
# Strategy worker autoscaling based on custom metric
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: strategy-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: strategy-worker
  minReplicas: 1
  maxReplicas: 500
  metrics:
  - type: External
    external:
      metric:
        name: strategy_count
        selector:
          matchLabels:
            status: "active"
      target:
        type: AverageValue
        averageValue: "1"  # 1 pod per active strategy
```

### Rolling Upgrade Strategy

```yaml
# Strategy worker deployment strategy
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0  # Don't kill running strategies until new ones are ready
  minReadySeconds: 30
```

### Secret Management

```yaml
# Use External Secrets Operator with HashiCorp Vault or AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: exchange-credentials
spec:
  secretStoreRef:
    name: vault-backend
  target:
    name: exchange-credentials
  data:
  - secretKey: wallex-api-key
    remoteRef:
      key: trading/wallex/main/api-key
  - secretKey: wallex-api-secret
    remoteRef:
      key: trading/wallex/main/api-secret
```

**Pod injection**:
```yaml
env:
- name: EXCHANGE_API_KEY
  valueFrom:
    secretKeyRef:
      name: exchange-credentials
      key: wallex-api-key
```

---

## 7. Containerization Strategy

### Docker Image Decomposition

```dockerfile
# Base image for all Python services
FROM python:3.11-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Shared library image (common code)
FROM base AS shared-lib
COPY mm/ /app/mm/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e /app

# Strategy Worker Image
FROM shared-lib AS strategy-worker
COPY mm/runtime/worker.py /app/
ENTRYPOINT ["python", "-m", "mm.runtime.worker"]

# Market Data Service Image
FROM shared-lib AS market-data
COPY mm/market_data/service.py /app/
ENTRYPOINT ["python", "-m", "mm.market_data.service"]

# Execution Service Image
FROM shared-lib AS execution
COPY mm/execution/service.py /app/
ENTRYPOINT ["python", "-m", "mm.execution.service"]

# Admin API Image
FROM shared-lib AS admin-api
COPY mm/api/admin.py /app/
ENTRYPOINT ["uvicorn", "mm.api.admin:app", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard Backend Image
FROM node:18 AS dashboard-backend
COPY dashboard/ /app/
RUN npm install && npm run build
CMD ["node", "dist/index.js"]

# Lightweight image for init/sidecar containers
FROM base AS sidecar
# No strategy code, just health checks and signal handling
```

### Image Sizes & Build Strategy

| Image | Size | Build Cache Strategy |
|-------|------|---------------------|
| base | ~500 MB | Weekly rebuild |
| shared-lib | ~600 MB | Daily or on code change |
| strategy-worker | ~620 MB | Per-commit |
| market-data | ~610 MB | Per-commit |
| execution | ~610 MB | Per-commit |
| admin-api | ~610 MB | Per-commit |

### Multi-stage Build Optimization

```dockerfile
# Leverage Docker layer caching
FROM python:3.11-slim AS builder
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Production image (no build tools)
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY mm/ /app/mm/
# Much smaller final image
```

---

## 8. Event and Messaging Architecture

### Evolution Path

```
Phase 1 (0-50 strategies): Redis Streams (already have)
    ↓
Phase 2 (50-500): Redis Streams + Consumer Groups + Persistence
    ↓
Phase 3 (500+): Kafka + Redis Streams (hot path)
```

### Phase 3 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kafka Cluster (3+ brokers)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  market.raw.wallex.BTCUSDT.trades     (partition 0)                │
│  market.raw.wallex.BTCUSDT.orderbook  (partition 1)                │
│  market.normalized.wallex.BTCUSDT     (partition 2)                │
│                                                                     │
│  strategy.decisions                    (partition by strategy_id)   │
│  strategy.actions                      (partition by strategy_id)   │
│  execution.orders                      (partition by exchange)      │
│  admin.commands                        (partition 0 - ordered)      │
│                                                                     │
│  persistence.events                    (partition 0 - all)          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Redis Streams (Hot Path Cache)                   │
├─────────────────────────────────────────────────────────────────────┤
│  • Last 5 minutes of market data (low-latency access)              │
│  • Strategy state snapshots                                         │
│  • Pending command responses                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Partitioning Strategy

| Event Type | Partition Key | Ordering Guarantee |
|------------|---------------|-------------------|
| Raw market data | `exchange:symbol` | Per-symbol sequential |
| Normalized market | `exchange:symbol` | Per-symbol sequential |
| Strategy decisions | `strategy_id` | Per-strategy sequential |
| Execution commands | `exchange:account` | Per-account sequential |
| Admin commands | `"global"` | Total order |

### Consumer Scaling

```python
# Kafka consumer group for market data
class MarketDataConsumer:
    def __init__(self, group_id="market-data-consumers", num_partitions=64):
        self.consumer = KafkaConsumer(
            'market.normalized.*',
            group_id=group_id,
            partition_assignment_strategy=RangeAssignor,
            max_poll_records=100,
            max_poll_interval_ms=300000
        )
        # One consumer per partition = parallel processing
```

### Backpressure Handling

```python
# Circuit breaker pattern
class EventConsumer:
    def __init__(self, max_queue_size=10000, backpressure_threshold=0.8):
        self.queue = asyncio.Queue(max_queue_size)
        self.circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
    async def consume(self, event):
        if self.queue.qsize() > backpressure_threshold * max_queue_size:
            # Apply backpressure: slow down consumer
            await asyncio.sleep(0.001)
            
        if self.circuit.is_open:
            # Persist to dead letter queue
            await self.dlq.send(event)
            return
            
        try:
            await self.queue.put(event)
        except asyncio.QueueFull:
            # Drop event with metric
            metrics.event_dropped.inc()
```

---

## 9. Persistence Architecture at Scale

### Layered Storage

| Tier | Technology | Retention | Access Pattern | Cost |
|------|------------|-----------|----------------|------|
| **Hot** | Redis | Minutes to hours | Real-time strategy state | High |
| **Warm** | Postgres | Days to weeks | Operational queries, admin UI | Medium |
| **Cold** | Parquet on S3 | Months to years | Replay, compliance, ML training | Low |
| **Analytics** | InfluxDB/TimescaleDB | Weeks | Time-series metrics, dashboards | Medium |

### Data Routing

```python
class TieredPersister:
    async def persist(self, event: EventEnvelope):
        # Route based on event type and age
        if event.topic.startswith('market.raw'):
            # Raw market data: direct to Parquet (no Postgres)
            await self.parquet_writer.write(event)
            
        elif event.topic.startswith('market.normalized'):
            # Normalized: Postgres (7 days) + Parquet
            await self.postgres_writer.write(event, ttl_days=7)
            await self.parquet_writer.write(event)
            
        elif event.topic.startswith('ledger.snapshot'):
            # Ledger: Postgres (30 days) + Parquet
            await self.postgres_writer.write(event, ttl_days=30, upsert=True)
            await self.parquet_writer.write(event)
            
        elif event.topic == 'strategy.decisions':
            # Decisions: Postgres (90 days) + S3 (indefinite)
            await self.postgres_writer.write(event, ttl_days=90)
            await self.s3_writer.write(event, prefix="decisions/")
```

### Event Sourcing Decision

**Do not implement full event sourcing** for strategy state. The overhead of replaying thousands of events per strategy per decision is prohibitive.

Instead:
- **State snapshots** every N frames (checkpointing)
- **Event log** for audit and replay (but not for state reconstruction)
- **Command-Query Responsibility Segregation (CQRS)** : Separate write and read paths

```python
# Checkpoint-based state persistence
class StrategyCheckpointer:
    async def checkpoint(self, strategy_id: str, state: StrategyState):
        # Write compressed checkpoint to S3
        await self.s3.put_object(
            Key=f"checkpoints/{strategy_id}/{state.frame_id}.msgpack",
            Body=msgpack.packb(state.to_dict(), use_bin_type=True)
        )
        
    async def restore(self, strategy_id: str) -> StrategyState:
        # Find latest checkpoint
        latest = await self._get_latest_checkpoint(strategy_id)
        return StrategyState.from_dict(msgpack.unpackb(latest))
```

### Postgres Schema Evolution

```sql
-- Partition by time for large tables
CREATE TABLE mm_events_partitioned (
    LIKE mm_events INCLUDING DEFAULTS
) PARTITION BY RANGE (ts);

-- Monthly partitions
CREATE TABLE mm_events_2025_01 PARTITION OF mm_events_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
    
-- Automatically drop old partitions (no DELETE)
DROP TABLE mm_events_2024_01;  -- After retention expires
```

---

## 10. Administrative API and Control Plane

### Control Plane Architecture

```python
# Central orchestration service
class ControlPlane:
    def __init__(self):
        self.k8s = KubernetesClient()
        self.registry = StrategyRegistry()
        self.exchange_manager = ExchangeManager()
        
    # Strategy lifecycle
    async def create_strategy(self, config: StrategyConfig) -> Strategy:
        # 1. Validate config
        # 2. Check resource availability
        # 3. Create Kubernetes deployment
        # 4. Register in etcd/Consul
        # 5. Return strategy object
        
    async def stop_strategy(self, strategy_id: str, timeout: int = 30):
        # 1. Send SIGTERM to pod
        # 2. Wait for graceful shutdown
        # 3. Scale deployment to 0
        # 4. Update registry
        
    async def migrate_strategy(self, strategy_id: str, target_node: str):
        # For maintenance: move strategy to different node
        pass
        
    # Exchange management
    async def add_exchange(self, exchange_config: ExchangeConfig):
        # 1. Test credentials
        # 2. Deploy exchange-specific services (market data, execution)
        # 3. Update routing tables
        
    # Risk controls
    async def global_kill_switch(self, reason: str):
        # 1. Publish kill command to all strategies
        # 2. Cancel all open orders across exchanges
        # 3. Set global_readonly flag
        # 4. Alert on-call
```

### API Endpoints

```yaml
# REST API (for operators, dashboards)
GET    /api/v1/strategies                 # List all strategies
POST   /api/v1/strategies                 # Create new strategy
GET    /api/v1/strategies/{id}            # Get strategy details
PUT    /api/v1/strategies/{id}            # Update configuration
DELETE /api/v1/strategies/{id}            # Delete strategy
POST   /api/v1/strategies/{id}/start      # Start strategy
POST   /api/v1/strategies/{id}/stop       # Stop strategy
POST   /api/v1/strategies/{id}/pause      # Pause execution
POST   /api/v1/strategies/{id}/resume     # Resume execution

GET    /api/v1/exchanges                   # List exchanges
POST   /api/v1/exchanges                   # Add exchange
PUT    /api/v1/exchanges/{id}/credentials  # Update credentials
POST   /api/v1/exchanges/{id}/test         # Test connectivity

GET    /api/v1/accounts                    # List trading accounts
POST   /api/v1/accounts                    # Create account

POST   /api/v1/control/kill                # Global kill switch
GET    /api/v1/control/status              # System status

GET    /api/v1/metrics                     # Prometheus metrics endpoint
```

### gRPC for Internal Services

```protobuf
// Internal service communication (lower latency)
service StrategyWorker {
    rpc GetStatus (Empty) returns (WorkerStatus);
    rpc UpdateConfig (ConfigUpdate) returns (ConfigAck);
    rpc PauseExecution (PauseRequest) returns (PauseResponse);
    rpc ForceCancelOrders (CancelRequest) returns (CancelResponse);
}

service ExecutionRouter {
    rpc PlaceOrder (PlaceOrderRequest) returns (PlaceOrderResponse);
    rpc CancelOrder (CancelOrderRequest) returns (CancelOrderResponse);
    rpc GetOpenOrders (Empty) returns (OpenOrdersResponse);
}

service MarketDataService {
    rpc Subscribe (SubscribeRequest) returns (stream MarketEvent);
    rpc GetSnapshot (SnapshotRequest) returns (OrderbookSnapshot);
}
```

---

## 11. Dashboard and Operational Monitoring

### Technology Stack Recommendation

| Layer | Technology | Why |
|-------|------------|-----|
| **Frontend** | React + TypeScript | Mature, large ecosystem |
| **UI Components** | MUI (Material-UI) or Ant Design | Professional trading interfaces |
| **Real-time** | GraphQL + Subscriptions (Apollo) | Flexible queries, real-time updates |
| **Charts** | Lightweight Charts (TradingView) or ECharts | Trading-specific, performant |
| **State** | Zustand or Redux Toolkit | Predictable state |
| **Build** | Vite | Fast dev experience |

### Dashboard Views

```yaml
# Operator Dashboard Layout

Overview:
  - Global health status (green/yellow/red)
  - Total strategies running / stopped / errored
  - Total PnL (today, week, month)
  - Total inventory exposure by asset
  - System resource usage (CPU/memory across cluster)

Strategy Table:
  - Name, exchange, symbol, status, uptime
  - Current inventory, cash, PnL
  - Last decision latency, market data age
  - Controls: start/stop/pause/delete

Strategy Detail View:
  - Real-time PnL chart
  - Orderbook depth visualization
  - Inventory vs. target over time
  - Open orders list
  - Recent fills
  - MPC optimizer diagnostics
  - Logs (recent, filtered)
  - Manual override controls

Exchange Health Dashboard:
  - Connectivity status per exchange
  - API latency (p50, p95, p99)
  - Rate limit usage
  - Error rates
  - Data freshness per symbol

Execution Dashboard:
  - Order placement rate
  - Fill rate
  - Rejection rate by reason
  - Average fill latency
  - Open order count by exchange

Risk Dashboard:
  - Current exposures by asset
  - VaR estimates
  - Risk limit utilization per strategy
  - Alert history
```

### Real-time Data Flow

```python
# Dashboard backend aggregates from multiple sources
class DashboardAggregator:
    async def get_strategy_status(self, strategy_id: str):
        # 1. Latest from Redis (sub-second)
        redis_data = await self.redis.hgetall(f"strategy:{strategy_id}:live")
        
        # 2. Recent history from Postgres (seconds to minutes)
        pg_data = await self.pg.fetch_one("""
            SELECT * FROM mm_latest_state 
            WHERE strategy_id = $1
        """, strategy_id)
        
        # 3. Time-series from InfluxDB (charts)
        influx_data = await self.influx.query("""
            SELECT mean(pnl) FROM strategy_pnl 
            WHERE strategy_id = '{}' 
            AND time > now() - 1h 
            GROUP BY time(1m)
        """.format(strategy_id))
        
        return DashboardResponse(
            realtime=redis_data,
            state=pg_data,
            history=influx_data
        )
```

### WebSocket Push

```python
# Server-sent events for real-time updates
@app.websocket("/ws/strategies/{strategy_id}")
async def strategy_websocket(websocket: WebSocket, strategy_id: str):
    await websocket.accept()
    
    # Subscribe to strategy's event stream
    async for event in event_bus.subscribe(f"strategy.{strategy_id}.*"):
        await websocket.send_json(event.to_dict())
```

---

## 12. Observability at Scale

### Metric Cardinality Management

```python
# BAD: High cardinality (explodes)
metrics.strategy_pnl.labels(
    strategy_id="mm-wallex-btcusdt-001",  # Unique per strategy
    exchange="wallex",
    symbol="BTCUSDT"
).set(pnl)

# GOOD: Low cardinality with aggregation
metrics.strategy_pnl_by_symbol.labels(
    exchange="wallex",
    symbol="BTCUSDT"
).set(avg_pnl_over_strategies)  # Aggregated

# For strategy-specific, use logs, not metrics
```

### Multi-Cluster Observability

```yaml
# Thanos for global Prometheus aggregation
apiVersion: v1
kind: ConfigMap
metadata:
  name: thanos-sidecar
data:
  thanos.yaml: |
    # Each cluster's Prometheus sidecar pushes to object storage
    objstore:
      type: S3
      config:
        bucket: mm-thanos-data
    # Query layer aggregates across clusters
```

### Distributed Tracing Configuration

```python
# Always sample errors, sample 1% of successful decisions
def should_sample(span_name: str, error: bool = False) -> bool:
    if error:
        return True
    if span_name == "mpc_optimization":
        return random.random() < 0.01  # 1% of decisions
    if span_name == "order_placement":
        return random.random() < 0.05  # 5% of orders
    return random.random() < 0.001  # 0.1% of everything
```

### Log Aggregation Architecture

```yaml
# Fluent-bit daemonset on each node
fluent-bit:
  input:
    tail:
      path: /var/log/containers/*.log
      multiline_parser: docker
  filter:
    - name: kubernetes
      match: "*"
      merge_log: true
      keep_log: false
    - name: modify
      match: "*"
      remove: ["timestamp", "stream", "kubernetes.annotations"]
  output:
    - name: loki
      match: "*"
      host: loki.monitoring.svc
      labels:
        job: fluentbit
```

### Alerting Architecture

```yaml
# Prometheus alert rules
groups:
  - name: strategy_alerts
    rules:
      - alert: StrategyStalled
        expr: strategy_last_decision_age_seconds > 30
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Strategy {{ $labels.strategy_id }} has not produced a decision for {{ $value }} seconds"
          
      - alert: MarketDataStale
        expr: market_data_age_seconds{stream="orderbook"} > 5
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Market data stale for {{ $labels.exchange }}/{{ $labels.symbol }}"
          
      - alert: OrderPlacementFailureRate
        expr: rate(execution_place_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
          
      - alert: InventoryImbalance
        expr: abs(strategy_inventory_norm) > 0.8
        for: 5m
        labels:
          severity: warning
```

### Capacity Planning Metrics

```python
# Track resource usage per strategy
class ResourceMonitor:
    async def record_strategy_resources(self, strategy_id: str):
        metrics.strategy_cpu_usage.labels(strategy_id=strategy_id).set(
            psutil.Process().cpu_percent()
        )
        metrics.strategy_memory_mb.labels(strategy_id=strategy_id).set(
            psutil.Process().memory_info().rss / 1024 / 1024
        )
        metrics.strategy_optimizer_duration_ms.labels(
            strategy_id=strategy_id,
            optimizer_type="mpc"
        ).observe(self.last_optimizer_duration)
        
    # Alert when approaching limits
    def check_capacity(self):
        total_cpu = sum(metrics.strategy_cpu_usage.values())
        if total_cpu > 0.8 * cluster_cpu_limit:
            alert("Approaching CPU capacity, consider scaling")
```

---

## 13. Account, Portfolio, Ledger, and Capital Allocation

This is the most critical architectural uncertainty. After deep analysis, here's the recommended model.

### Recommended Ownership Hierarchy

```
Platform
   └── Exchange
        └── Account (Exchange-level entity, 1:1 with credentials)
             └── Portfolio (Virtual construct, platform-level)
                  ├── Capital Allocation (Static or dynamic)
                  │    └── Strategy (Trading logic)
                  │         └── Strategy Ledger (Attributed balances)
                  └── Other Allocations...
```

### Detailed Model

```python
# Core entities
class ExchangeAccount:
    """Actual exchange account with real credentials and balances"""
    id: str
    exchange: str
    credentials: EncryptedCredentials
    real_balances: dict[str, float]  # From exchange API
    
class Portfolio:
    """Virtual container for capital allocation"""
    id: str
    exchange_account_id: str
    total_capital: float  # In quote currency
    allocated_capital: float  # Sum of allocations
    free_capital: float = total_capital - allocated_capital
    
class CapitalAllocation:
    """Allocation of portfolio capital to a strategy"""
    id: str
    portfolio_id: str
    strategy_id: str
    allocated_amount: float  # In quote currency
    allocation_type: Literal["static", "dynamic", "shared"]
    max_utilization: float = 1.0  # Can use up to allocated_amount * max_utilization
    
class StrategyLedger:
    """Strategy's attributed view of balances"""
    strategy_id: str
    allocation_id: str
    attributed_cash: float  # Virtual cash balance
    attributed_inventory: dict[str, float]  # Virtual inventory
    actual_cash: float  # Real cash used (for reconciliation)
    actual_inventory: dict[str, float]
    pnl_realized: float
    pnl_unrealized: float
```

### Operational Model Selection

**Recommendation: Model B (Dedicated Account Per Strategy) + Portfolio Abstraction**

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| **Primary model** | Dedicated exchange account/subaccount per strategy | Clean isolation, simple attribution |
| **Where not possible** | Portfolio with capital allocations | For exchanges with limited accounts |
| **Portfolio abstraction** | Always present (platform-level) | Enables cross-strategy risk management |

### Implementation for Dedicated Accounts (Preferred)

```python
# Each strategy gets its own exchange account
class DedicatedAccountStrategy:
    def __init__(self, strategy_id: str, exchange_account: ExchangeAccount):
        self.account = exchange_account  # 1:1 with strategy
        self.ledger = StrategyLedger(
            strategy_id=strategy_id,
            # Ledger tracks real balances directly
            actual_cash=self.account.balances['USDT'],
            actual_inventory=self.account.balances
        )
        
    def get_portfolio_view(self) -> Portfolio:
        # Strategy IS the portfolio (trivial mapping)
        return Portfolio(
            id=self.strategy_id,
            exchange_account_id=self.account.id,
            total_capital=self.ledger.actual_cash,
            allocated_capital=self.ledger.actual_cash,
            free_capital=0
        )
```

### Implementation for Shared Accounts (Fallback)

```python
# Multiple strategies share one exchange account
class SharedAccountStrategy:
    def __init__(self, strategy_id: str, allocation: CapitalAllocation):
        self.allocation = allocation
        self.ledger = StrategyLedger(
            strategy_id=strategy_id,
            attributed_cash=allocation.allocated_amount,
            attributed_inventory={},  # Starts empty
            actual_cash=0,  # Updated during reconciliation
            actual_inventory={}
        )
        
    async def reconcile(self, actual_account_state: ExchangeAccount):
        # Reconcile attributed vs actual
        # This is complex - see below
        pass
```

### Reconciliation for Shared Accounts

```python
class SharedAccountReconciler:
    """
    Complex reconciliation for shared accounts.
    This is the hardest part of shared accounts.
    """
    
    async def reconcile(self, 
                        allocations: list[CapitalAllocation],
                        actual_state: ExchangeAccount,
                        strategies_state: dict[str, StrategyLedger]):
        
        # 1. Attribute open orders to strategies (by order tags or inference)
        attributed_orders = self._attribute_orders(actual_state.open_orders)
        
        # 2. Attribute inventory (which strategy owns which inventory)
        #    This is ambiguous - use FIFO, pro-rata, or tag-based
        attributed_inventory = self._attribute_inventory(
            actual_state.balances,
            strategies_state,
            method="pro_rata"  # Allocate by capital allocation
        )
        
        # 3. Update each strategy's ledger
        for strategy_id, strategy in strategies_state.items():
            strategy.actual_cash = attributed_inventory[strategy_id]['cash']
            strategy.actual_inventory = attributed_inventory[strategy_id]['inventory']
            
            # 4. Check for violations
            if strategy.actual_cash < -self.allocation_tolerance * strategy.attributed_cash:
                # Violation - strategy used more than allocated
                self.alerts.send("Capital allocation exceeded", strategy_id)
                
        return strategies_state
```

### Capital Allocation Strategies

| Strategy Type | Allocation Strategy | Isolation Level |
|---------------|---------------------|-----------------|
| Market making (correlated assets) | Dedicated account | High |
| Market making (uncorrelated) | Shared with pro-rata | Medium |
| Arbitrage | Dedicated account | Very High |
| Liquidity provision (stable pairs) | Shared with tags | Low |

### Ledger Architecture Recommendation

**Single source of truth: Exchange Account**

```
Exchange API → PrivateStateReconciler → ExchangeAccount (source of truth)
                                              ↓
                                        Portfolio Layer
                                              ↓
                                    Strategy Ledger (attributed)
```

Do NOT maintain separate exchange-level ledger. The exchange is the source of truth. Your platform's role is attribution and allocation, not creating another ledger that can diverge.

### Impact on Existing Components

| Component | Required Changes | Complexity |
|-----------|-----------------|------------|
| Runtime config | Add `account_id`, `allocation_id`, `portfolio_id` | Medium |
| Private state adapters | Support multiple strategies per account | High |
| Portfolio ledger service | Extend to handle allocations | Medium |
| Risk engine | Add allocation-based limits | Medium |
| Inventory accounting | Attribution logic for shared accounts | High |
| Execution routing | Tag orders with strategy_id | Low |
| Persistence schema | Add allocation/portfolio tables | Medium |
| Admin API | Add account/portfolio/allocation CRUD | Low |

### Phased Approach

**Phase 1 (Immediate)**: Dedicated accounts only
- Simpler to implement
- Use existing architecture with minor changes
- Validate with exchanges that support subaccounts

**Phase 2 (3-6 months)**: Add portfolio abstraction
- Build portfolio and allocation entities
- No shared accounts yet - just for aggregation

**Phase 3 (6-12 months)**: Shared accounts if required
- Complex reconciliation
- Only if exchanges don't support enough subaccounts

---

## Migration Roadmap

### Phase 0: Preparation (Weeks 1-2)
**No downtime, no changes to running system**

- [ ] Add OpenTelemetry instrumentation to existing components
- [ ] Create Kubernetes development namespace
- [ ] Build container images for current codebase (test only)
- [ ] Set up monitoring stack (Prometheus/Grafana/Loki)
- [ ] Run load tests to establish baseline performance

### Phase 1: Extract Market Data Service (Weeks 3-6)
**Minimal downtime (hours)**

```python
# Step 1: Add market data service as optional consumer of exchange feeds
class MarketDataService:
    # Same adapters, just runs independently
    # Publish to Redis Streams (already supported!)
    pass

# Step 2: Modify strategy worker to consume from Redis Streams
# Fallback to direct connection if market data service unavailable

# Step 3: Gradual migration
for exchange in exchanges:
    # 1. Start market data service for exchange
    # 2. Migrate 10% of strategies to consume from service
    # 3. Monitor for issues
    # 4. Increase to 100%
```

**Success criteria**: Single WebSocket connection per exchange/symbol across all strategies

### Phase 2: Containerization & Kubernetes (Weeks 7-10)
**Rolling migration, no downtime**

- [ ] Package strategy worker as container
- [ ] Create Helm charts for deployment
- [ ] Set up Redis cluster (if not already)
- [ ] Deploy Postgres with replication (if not already)
- [ ] Migrate 1 strategy to Kubernetes as test
- [ ] Gradually migrate all strategies

```yaml
# Migration approach: side-by-side run
# Run both old (bare metal) and new (K8s) simultaneously
# Switch traffic using Redis Streams consumer groups
```

### Phase 3: Strategy Orchestration (Weeks 11-14)
**New capability, no existing changes**

- [ ] Build control plane service (Kubernetes operator)
- [ ] Implement strategy lifecycle API
- [ ] Add admin API endpoints
- [ ] Build basic dashboard (strategy list, status, start/stop)
- [ ] Migrate strategy configuration to ConfigMaps

### Phase 4: MPC Optimization Batching (Weeks 15-18)
**Performance improvement, no behavior change**

- [ ] Refactor MPC optimizer to support batched execution
- [ ] Add GPU node pool to Kubernetes
- [ ] Implement worker pool for batched optimization
- [ ] Benchmark and tune batch size

### Phase 5: Execution Service Extraction (Weeks 19-22)
**Optional, only if needed**

- [ ] Extract order placement/management to separate service
- [ ] Add rate limiting across strategies
- [ ] Add order pre-validation
- [ ] Gradual migration similar to market data

### Phase 6: Account & Portfolio Model (Weeks 23-30)
**Significant change, coordinate with operations**

- [ ] Design portfolio/allocation schema
- [ ] Extend private state reconciler for shared accounts
- [ ] Build attribution engine
- [ ] Migrate existing strategies to new model
- [ ] Update dashboard for portfolio view

### Phase 7: Production Hardening (Ongoing)

- [ ] Chaos engineering tests (pod failures, network partitions)
- [ ] Automated canary deployments
- [ ] Disaster recovery runbooks
- [ ] Capacity planning automation
- [ ] 24/7 on-call rotation

---

## Architecture Decision Record (ADR) Summary

| ADR | Decision | Rationale |
|-----|----------|-----------|
| **ADR-001** | One strategy per container | Fault isolation, simple lifecycle |
| **ADR-002** | Extract market data service | Reduce connections, consistent view |
| **ADR-003** | Redis Streams first, Kafka later | Already working, simpler operations |
| **ADR-004** | GPU for MPC when >20 strategies | Batch optimization = 10-50x speedup |
| **ADR-005** | asyncio for I/O, multiprocessing for CPU | Best of both worlds with Python |
| **ADR-006** | Control plane uses Kubernetes primitives | Declarative, battle-tested |
| **ADR-007** | Dedicated accounts per strategy (preferred) | Cleanest attribution, simplest code |
| **ADR-008** | Portfolio abstraction always present | Future flexibility, no downside |
| **ADR-009** | Exchange account is source of truth | Don't duplicate what exchange already has |
| **ADR-010** | CQRS with checkpoints, not event sourcing | Performance, simplicity |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Market data service single point of failure** | Run 3 replicas with leader election; strategies fall back to direct connection |
| **Kubernetes control plane failure** | Run multi-master; have emergency runbooks for bare-metal revert |
| **GPU node shortage** | Maintain CPU fallback; auto-scaling for GPU nodes |
| **Shared account reconciliation errors** | Extensive simulation testing; conservative allocation; real-time alerts |
| **Distributed tracing overhead** | Sampling as specified; always trace errors |
| **Database migration failures** | Blue-green deployment for Postgres; point-in-time recovery |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Strategies per node | 50 (CPU) / 200 (GPU) |
| Decision latency (p99) | < 2 seconds |
| Market data age (p99) | < 500 ms |
| Order placement latency (p99) | < 100 ms |
| System uptime | 99.95% |
| Time to deploy new strategy | < 30 seconds |
| Operator dashboard load time | < 2 seconds |

---

## Immediate Next Steps (Week 1)

1. **Set up observability baseline** on current system
2. **Containerize the existing strategy runner** (no behavior change)
3. **Deploy to a test Kubernetes namespace** (single strategy)
4. **Run load test** with 10 simulated strategies
5. **Identify first bottleneck** (likely market data connections)

The architecture described is production-ready for 100+ strategies. Start with Phase 1 (market data extraction) - it's the highest impact with lowest risk.


You raise a valid concern. Running one pod per strategy does introduce overhead—each container includes a Python runtime, OS libraries, and Kubernetes pod overhead (pause container, network namespace, etc.).

## Realistic Overhead Numbers

| Component | Memory | Notes |
|-----------|--------|-------|
| Base Python slim image | ~50-80 MB | Python 3.11 + minimal libs |
| Your strategy code + deps | ~30-50 MB | PyTorch, numpy, etc. (shared if using same image) |
| Kubernetes pod overhead | ~5-15 MB | Pause container, network, emptyDir |
| **Total per pod** | **~85-145 MB** | Without strategy-specific state |

For 100 strategies: **8.5–14.5 GB** of overhead just for containers.  
For 1,000 strategies: **85–145 GB** — prohibitive on most clusters.

## Better Solutions

### Option 1: Multi‑Strategy Pod (Recommended for 100+ strategies)

Run **multiple strategy instances inside one pod** using:

- **asyncio tasks** (I/O‑bound only – not suitable for CPU‑heavy MPC)
- **multiprocessing** (good for CPU isolation, but each process still has Python overhead)
- **subinterpreters** (Python 3.12+ `interpreters` module – low overhead, but experimental)

**Memory comparison**:

| Approach | Overhead per extra strategy | Total for 100 strategies |
|----------|----------------------------|--------------------------|
| One pod per strategy | 100× 85 MB = 8.5 GB | 8.5 GB + strategy state |
| 1 pod with 10 processes | Base pod 150 MB + (10× 40 MB) = 550 MB | 5.5 GB (10 pods) |
| 1 pod with 100 asyncio tasks | Base pod 150 MB + shared state | ~200 MB (only for I/O‑bound work) |

**But your MPC optimizer is CPU‑bound** – asyncio won’t help. Multiprocessing in one pod gives you:

```python
# One pod, multiple strategy processes
class MultiStrategyPod:
    def __init__(self, strategies: list[StrategyConfig]):
        self.processes = []
        for cfg in strategies:
            ctx = multiprocessing.get_context('spawn')
            p = ctx.Process(target=run_strategy, args=(cfg,))
            p.start()
            self.processes.append(p)
```

Memory: each process still has its own Python interpreter (~30-50 MB) plus strategy data. For 10 strategies per pod → ~500 MB for interpreters + shared base image (the Docker image is shared across all processes because they fork from the same binary). Overhead is much lower than separate pods.

### Option 2: Sidecar + Shared Python Runtime

Use an **init container** to pre‑warm a shared Python environment on a `emptyDir` volume, then all strategy containers in the pod mount and reuse it. This eliminates per‑container Python overhead.

```yaml
initContainers:
- name: python-env-builder
  image: python:3.11-slim
  command: ['cp', '-r', '/usr/local', '/shared-python']
  volumeMounts:
  - name: shared-python
    mountPath: /shared-python
containers:
- name: strategy-1
  image: alpine  # tiny image
  command: ['/shared-python/bin/python', '/app/strategy.py']
  volumeMounts:
  - name: shared-python
    mountPath: /shared-python
  - name: strategy-code
    mountPath: /app
- name: strategy-2  # second container, same pod
  ...
```

Result: Each strategy container adds only a few MB (Alpine + your code). Docker image for the strategy itself can be tiny (~10 MB) if it just executes Python from the shared volume.

### Option 3: Process per Strategy, but Isolated via cgroups (One Pod)

Use a single pod with **cgroups v2** to enforce CPU/memory limits per strategy process without Kubernetes pod overhead. This is what Kubernetes does internally anyway, but you can do it more densely.

```python
# Inside a single pod, run multiple strategy processes with cgroup constraints
import cgroup

for strategy in strategies:
    cgroup.create(f"/sys/fs/cgroup/strategy-{strategy.id}")
    cgroup.set_memory_limit("2G")
    cgroup.set_cpu_limit(2.0)
    pid = os.fork()
    if pid == 0:
        cgroup.attach(pid)
        run_strategy(strategy)
```

This gives you near‑bare‑metal density while still allowing per‑strategy resource limits and killing one strategy without affecting others.

### Option 4: Lightweight Container Runtimes

Replace Docker/containerd with **gVisor** or **Kata Containers**? No – they add more overhead. Stick with runc. Use **distroless** images (no shell, no package manager) to shave 20-30 MB per container.

## Recommended Strategy for Your Scale

| Number of Strategies | Recommended Model | Memory Overhead |
|---------------------|-------------------|-----------------|
| < 50 | One pod per strategy | Acceptable (4 GB overhead) |
| 50 – 500 | 5–10 strategies per pod (multiprocessing) | 150 MB base + 40 MB/strategy |
| 500 – 5000 | 1 pod per node with many processes (cgroups) | As low as 30 MB/strategy |
| > 5000 | Re‑architect: shared market data + batched MPC (GPU) | Marginal additional overhead |

## Concrete Implementation for Your Platform

Modify your strategy worker to support **multiple strategy instances inside one process** using asyncio for I/O and a process pool for CPU work shared across strategies:

```python
# multi_strategy_runner.py
class MultiStrategyRunner:
    def __init__(self):
        self.strategies = {}  # id -> StrategyContext
        self.cpu_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        
    async def add_strategy(self, config):
        ctx = StrategyContext(config)
        self.strategies[config.id] = ctx
        asyncio.create_task(self._run_strategy_loop(ctx))
        
    async def _run_strategy_loop(self, ctx):
        while True:
            frame = await self._get_frame(ctx)
            # Offload MPC to shared process pool
            action = await loop.run_in_executor(
                self.cpu_executor, 
                ctx.optimizer.optimize, 
                frame
            )
            await self._execute(ctx, action)
```

Then run **one pod** with this multi‑strategy runner, not one pod per strategy.

## Bottom Line

You are correct: one pod per strategy wastes memory. The sweet spot is **5–20 strategies per pod** using multiprocessing, with a shared base Python environment to cut per‑process overhead. At extreme scale (>500 strategies), move to a single pod per node with cgroup‑isolated processes.

Your existing code can be adapted to this model with moderate changes – the main work is making the `StrategyRunner` multi‑tenant (no global state, per‑strategy event bus subscriptions).