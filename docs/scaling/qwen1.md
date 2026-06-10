Based on the provided knowledge base, here is a comprehensive architectural review and migration roadmap for scaling your Market-Making Platform.

The core philosophy of this redesign is **Pragmatic Modularity**. We will avoid the "microservices trap" (Kafka, complex service meshes) and instead leverage your existing, robust event-driven foundation (`RedisStreamsBus`, `PostgresEventWriter`) to create a fleet of specialized, horizontally scalable workers.

---

### 1. High-Level Target Architecture

The current "one bot per process" model must evolve into a **Hub-and-Spoke Architecture** centered around Redis Streams as the central nervous system.

#### The Core Components (Service Boundaries)
1.  **Control Plane & Admin API (Singleton/HA)**:
    *   **Responsibility**: Manages exchange credentials, strategy configurations, capital allocations, and global risk parameters.
    *   **Tech**: FastAPI, PostgreSQL (Operational Data).
2.  **Market Data Ingestion Fleet (Scalable)**:
    *   **Responsibility**: Connects to Exchange WebSockets/REST, normalizes data, maintains local orderbook snapshots, and publishes to Redis Streams.
    *   **Deployment**: One Deployment per Exchange (or grouped by venue type).
3.  **Trading Node Fleet (The "Worker" - Scalable)**:
    *   **Responsibility**: Consumes market data, runs the Strategy/Risk/Execution loop, and reconciles private state.
    *   **Composition**: A single Trading Node Pod can host **multiple** strategy instances (e.g., 5-10 symbols) using Python `asyncio` tasks, sharing a single Redis connection and Event Loop.
    *   **Why bundle Strategy + Execution?**: Your `OrderManager` relies on tight, low-latency state reconciliation with `PrivateStateReconciler` and `TargetBook`. Splitting execution into a separate microservice introduces network latency and state-sync complexity.
4.  **MPC Optimizer Service (Optional but Recommended for Scale)**:
    *   **Responsibility**: Offloads the heavy PyTorch `MonteCarloMPCTensorBatched` computations from the Trading Nodes via gRPC.
5.  **Persistence & Analytics Fleet**:
    *   **Responsibility**: Subscribes to Redis Streams and writes to PostgreSQL, TimescaleDB, and S3/Parquet.

---

### 2. Deep Dive: Answering the Core Questions

#### A. Strategy Execution & Python Concurrency (Q2, Q5)
*   **The Concurrency Model**:
    *   **I/O Bound (Market Data, REST, Redis, WS)**: Handled by a single `asyncio` event loop per Pod.
    *   **CPU Bound (MPC, Risk)**: The GIL is the enemy here. While PyTorch releases the GIL during C++/CUDA operations, the Python overhead in Monte Carlo simulations can still block the event loop.
*   **The Solution**:
    *   **In-Process**: Use `ProcessPoolExecutor` for MPC optimization if keeping it local.
    *   **System-Level (Recommended)**: Extract `MpcStrategyEngine.decide()` into a standalone **Optimizer gRPC Service**. The Trading Node sends the `State` (tensors) over gRPC, and the Optimizer (running on GPU or high-CPU nodes) returns the `Action`. This prevents CPU spikes from crashing the I/O loop and allows independent scaling of compute vs. execution.
*   **Isolation Boundary**: The **Kubernetes Pod** is the isolation boundary. If a Pod crashes, K8s restarts it, and the `OrderManager` safely reconciles state upon reconnection.

#### B. Market Data Architecture (Q3)
*   **Decoupling is Mandatory**: Ingestion must be separated from Strategy logic. A WebSocket disconnect or a massive trade spike should not starve the strategy's decision loop.
*   **Flow**: `Exchange WS` $\rightarrow$ `MD Ingestion Pod` $\rightarrow$ `Redis Streams` (`market:events:<exchange>:<symbol>`).
*   **Orderbook Management**: The MD Ingestion Pod should maintain the authoritative normalized snapshot and publish *deltas* or *periodic snapshots* to Redis. Trading Nodes subscribe and maintain their own local `OrderbookManager` (as currently implemented) to ensure zero-latency access during the MPC simulation.

#### C. Event & Messaging Architecture (Q8)
*   **Verdict**: **Stick with Redis Streams.**
*   **Why not Kafka?**: For 10-200 strategies, Kafka is operational overkill. Redis Streams natively supports:
    *   **Consumer Groups**: Multiple Trading Node Pods can share the load of processing a high-volume symbol stream.
    *   **Replay**: Essential for your `ParquetEventWriter` and incident debugging.
    *   **Backpressure**: Handled via your existing `block` / `drop_oldest` logic.
*   **Topology**: Use topic patterns like `strategy:joint:frames:<exchange>:<symbol>:<interval>` for internal node communication if nodes need to share global liquidity states.

#### D. Account, Portfolio, and Ledger Architecture (Q13)
This is the most critical business logic shift. You must move from a 1:1 Strategy-to-Account model to a **Virtual Ledger Model**.

*   **The Hierarchy**:
    1.  **Exchange Account** (Physical): The actual API keys and real balances on Binance/Wallex.
    2.  **Portfolio** (Virtual): A logical grouping of capital (e.g., "Main Crypto Fund", "Stablecoin Arbitrage Fund").
    3.  **Capital Allocation**: The Control Plane allocates X USDT from the Exchange Account to Portfolio A, and Y USDT to Portfolio B.
    4.  **Strategy Instance**: Binds to a Portfolio.
*   **Ledger Implementation**:
    *   Upgrade `PortfolioLedgerService` to maintain **Sub-Ledgers**.
    *   **Pre-Trade Risk**: The `NetDeltaRiskEngine` must check against the *Virtual Sub-Ledger* (to prevent strategies from stealing each other's allocated capital).
    *   **Hard Risk**: A secondary check must ensure the *Physical Exchange Account* has enough total balance to cover the order (preventing API rejections).
*   **PnL Attribution**: Fills from the exchange are routed to the `ExecutionStateReconciler`, which tags the fill with the `actor_id` (Strategy) and updates the specific Virtual Sub-Ledger.

#### E. Database & Storage Architecture (Q9, Q14)
Your current PostgreSQL-centric approach will bottleneck on high-frequency market data and optimizer diagnostics.

| Data Class | Recommended Storage | Rationale |
| :--- | :--- | :--- |
| **Operational / Config** | **PostgreSQL** | Exchanges, Accounts, Strategies, API Keys. Relational integrity is key. |
| **Execution / Audit** | **PostgreSQL** | `mm_execution_facts`, `mm_target_order_facts`. ACID compliance required for financial audit trails. |
| **Market Data / Metrics** | **TimescaleDB** (PG Extension) | High-volume time-series (`mm_market_facts`, Prometheus metrics). Enables fast aggregation and retention policies without leaving the PG ecosystem. |
| **Optimizer / Diagnostics** | **TimescaleDB** or **ClickHouse** | `mm_optimizer_facts`. Analytical queries on MPC convergence and model scores. |
| **Replay / Archival** | **Object Storage (S3/MinIO)** | Your existing `ParquetEventWriter` is perfect here. Store raw events and full snapshots for offline backtesting. |
| **Hot State / Cache** | **Redis** | Latest Orderbook Snapshots, Latest Ledger State, Distributed Locks (e.g., preventing two nodes from trading the same sub-account). |

---

### 3. Kubernetes Deployment Model (Q6, Q7)

#### Containerization Strategy
Do not create 10 different microservice images. Create **3 Core Images**:
1.  `mm-platform-core`: Contains the FastAPI Control Plane, DB migrations, and Admin tools.
2.  `mm-ingestor`: Contains Exchange Adapters and `SymbolEngine`.
3.  `mm-trading-node`: Contains `StrategyRunner`, `MpcStrategyEngine`, `OrderManager`, and `PrivateStateReconciler`.

#### K8s Topology
*   **Control Plane**: `Deployment` (2 replicas for HA) + `PostgreSQL` (Managed RDS or StatefulSet).
*   **Ingestors**: `StatefulSet` or `Deployment` per Exchange. (e.g., `ingestor-wallex`, `ingestor-nobitex`).
*   **Trading Nodes**: `Deployment` with **Horizontal Pod Autoscaler (HPA)**.
    *   *Scaling Metric*: Scale based on **Redis Streams Consumer Group Lag** or **CPU Utilization**.
    *   *Assignment*: Use a "Claim" mechanism in Redis or Postgres where a Node boots up, claims a set of `Strategy IDs`, and subscribes only to those specific Redis topics.
*   **Optimizer (Optional)**: `Deployment` with GPU node pools (if using CUDA) or high-CPU nodes, exposed via internal gRPC Service.

---

### 4. Migration Roadmap

Transitioning a live trading system requires zero-downtime, incremental steps.

#### Phase 1: Decouple Market Data (The "Ingestion Fleet")
*   **Action**: Extract `SymbolEngine` and Exchange Adapters into a standalone service (`mm-ingestor`).
*   **Change**: The Ingestor publishes normalized `MarketEvent` objects to Redis Streams. The existing monolith is modified to *consume* from Redis Streams instead of connecting directly to the Exchange WS.
*   **Benefit**: Immediate stability improvement. Market data spikes no longer crash strategy logic.

#### Phase 2: Virtualize the Ledger (The "Portfolio Layer")
*   **Action**: Refactor `PortfolioLedgerService` to support `account_id` vs `portfolio_id`.
*   **Change**: Implement the Capital Allocation logic in the Control Plane. Update `NetDeltaRiskEngine` to enforce virtual limits.
*   **Benefit**: Enables running multiple strategies on a single exchange account safely.

#### Phase 3: Optimize the Compute (The "MPC Offload")
*   **Action**: Wrap `MonteCarloMPCTensorBatched` in a gRPC server.
*   **Change**: Modify `StrategyRunner` to call the gRPC service instead of the local worker thread.
*   **Benefit**: Solves the GIL/CPU bottleneck. Allows you to scale compute (GPU/CPU) independently of execution nodes.

#### Phase 4: Database Tiering & K8s Migration
*   **Action**: Enable TimescaleDB extension on your Postgres instance.
*   **Change**: Route `mm_market_facts` and `mm_optimizer_facts` to TimescaleDB hypertables. Configure retention policies (e.g., drop raw market facts after 30 days, keep Parquet in S3).
*   **Action**: Dockerize the components and deploy to Kubernetes using the topology defined above.

---

### 5. Summary of Recommendations

| Domain | Recommendation | Why? |
| :--- | :--- | :--- |
| **Messaging** | **Redis Streams** | Sufficient for scale, supports consumer groups, replay, and low latency. Avoids Kafka complexity. |
| **Database** | **Postgres + TimescaleDB** | Keeps stack simple (SQL) but handles time-series volume. |
| **Compute** | **gRPC Optimizer Service** | Bypasses Python GIL, enables GPU scaling, isolates I/O from CPU. |
| **Execution** | **Trading Node (Worker)** | Keeps `OrderManager` state reconciliation local and fast. |
| **Ledger** | **Virtual Sub-Ledgers** | Allows capital sharing while enforcing strict strategy-level risk limits. |
| **Observability** | **OpenTelemetry + Grafana** | Trace a decision from MD Ingestion $\rightarrow$ Redis $\rightarrow$ MPC $\rightarrow$ Execution. |

This architecture respects your preference for **simplicity and reliability** while providing a clear, engineered path to handle hundreds of strategies and millions of events without succumbing to enterprise bloat.