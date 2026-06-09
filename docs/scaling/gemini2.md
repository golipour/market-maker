It is great that you are approaching this platform evolution with a focus on simplicity and maintainability. Scaling a market-making platform can easily spiral into infrastructure bloat if you aren't careful. Based on the constraints you've shared and the code-derived system documentation, you already have a remarkably solid, cloud-native-ready foundation.

Here is an architectural review and roadmap for scaling your platform, grounded directly in the capabilities of your current codebase.

---

## 1. Target Architecture & Service Boundaries

Your system is currently built around an `AbstractEventBus` using topic names for communication, which inherently supports decoupling. However, because the main runtime builder (`mm/runtime/engine.py`) wires everything into a single process, the boundaries need to shift for scale.

**What Should Become Independently Scalable Services:**

* **Market Data Ingestion:** The `SymbolEngine` and `MarketEventBuilder` should be extracted into isolated pods per exchange. They already publish normalized events to `market:raw:<exchange>:<symbol>:...` topics via Redis Streams, meaning consumers don't need to live in the same process.
* **Persistence Writers:** The `PostgresEventWriter` and `ParquetEventWriter` should run as standalone background consumers.
* **Control Plane API:** The existing FastAPI admin app (`mm/api/admin.py`) should be broken out as the centralized gateway for operations.

**What Should Remain "Monolithic" (The Strategy Worker):**

* The core strategy pipeline must remain tightly coupled. The documentation notes that the pipeline "directly owns the MpcStrategyEngine, risk engine, and order-manager map".
* Splitting the `StrategyRunner`, `NetDeltaRiskEngine`, and `OrderManager` across a network would introduce catastrophic latency and split-brain risks. Keep these together inside a single **Strategy Worker** container.

---

## 2. Concurrency & Execution Model

The Python Global Interpreter Lock (GIL) is your primary compute bottleneck. Currently, the `MpcStrategyEngine.decide()` runs in a worker thread under a `decision_timeout_ms` within the asyncio loop. At scale, threading CPU-bound PyTorch optimizations alongside I/O-bound websocket handling in the same Python process will lead to missed intervals.

**Recommendations:**

* **Process Model:** Deploy **one strategy instance per Kubernetes Pod** (or a very small process pool per pod). The runtime is explicitly built as "one bot per process/config", making horizontal scaling via Kubernetes pods the most natural path.
* **Compute Optimization:** The `MonteCarloMPCTensorBatched` optimizer uses PyTorch but defaults to the CPU. Moving to GPU instances for your strategy workers is the single highest-impact optimization you can make without rewriting algorithms.
* **Algorithm Extensibility:** The strategy engine relies on an "optimizer backend protocol" rather than a hardcoded tensor optimizer, meaning you can easily swap in compiled Rust extensions or distributed workers later if needed.

---

## 3. Account, Portfolio & Ledger Model

This is the most critical operational pivot. The current system assumes a direct relationship: Strategy → Ledger → Exchange Account. The `PortfolioLedgerService` requires private state snapshots and order updates to build a `VenueLedgerStateMachine`, and the strategy runner will literally raise an error if no ledger is present.

**Recommended Model:** Dedicated Account/Subaccount or API Key Per Strategy (Models B & C).

* **Why:** The codebase is heavily optimized for localized risk. The `NetDeltaRiskEngine` enforces limits based on "wallet totals," "available base and quote," and "actor live-order collateral" tracked in the local ledger.
* Transitioning to a shared account (Model A) would require a massive rewrite of the `JointStrategyFrameAssembler`, the `OrderManager` (to prevent strategies from canceling each other's scoped orders), and the risk engine.
* Sticking to isolated API keys or subaccounts allows you to deploy 200 strategies horizontally without changing a single line of your core reconciliation logic.

---

## 4. Messaging & Persistence Architecture

You stated a preference to avoid Kafka unless necessary. Fortunately, your codebase is perfectly adapted for your preferred stack.

**Messaging:**

* Stick with **Redis Streams**. The `RedisStreamsBus` already compacts envelopes, handles remote stream decoding, and intelligently tags its own stream writes to avoid duplicate same-process delivery. It is entirely sufficient for the medium-scale target of 200 bots.

**Database Architecture:**

* **Operational & Fast Analytics:** Keep PostgreSQL. Your codebase already implements sophisticated typed fact tables (e.g., `mm_market_facts`, `mm_action_stage_facts`, `mm_execution_facts`) that materialize compacted data for fast Grafana querying.
* **Historical/Data Lake:** Do not add a complex time-series database like ClickHouse. The system already features a `ParquetEventWriter` that writes partitioned files (by date/exchange/symbol) containing typed columns for offline replay and analytics. Point this writer at an S3/MinIO bucket.

---

## 5. Observability & Control Plane

The platform is inherently cloud-native ready.

* **Metrics & Tracing:** The system already utilizes Prometheus metrics, OTLP traces, structured JSON logs, and a Docker Compose stack with Grafana, Loki, and Tempo.
* **Kubernetes Adaptation:** Instead of relying on the custom in-process Prometheus text registry on port 8000, simply configure Kubernetes ServiceMonitors to scrape these endpoints across all your strategy worker pods.
* **Control Plane:** The `RuntimeControlPlane` currently consumes `admin:commands` to handle kill switches, pausing, and observe-only modes, publishing state to `admin:events`. By decoupling the FastAPI server from the worker nodes, you can broadcast these commands via Redis Streams to manage your entire fleet centrally.

---

## 6. Migration Roadmap

Here is the step-by-step path from your single-process monolith to a horizontally scaled platform:

1. **Extract Infrastructure Services:** Move Redis, Postgres, and the observability stack (Grafana/Prometheus/Loki) into dedicated, highly available database/cache clusters outside of local Docker Compose.
2. **Decouple Market Data:** Create a lightweight deployment that only runs the `SymbolEngine` and `IntervalScheduler` components to push raw market events into Redis Streams.
3. **Decouple Persistence:** Run the `PostgresEventWriter` and `ParquetEventWriter` in their own background deployment, subscribing to the Redis Streams backbone.
4. **Containerize Strategy Workers:** Package the core pipeline (`StrategyRunner`, `PortfolioLedgerService`, `MpcStrategyEngine`) into a worker image. Use Kubernetes ConfigMaps to inject the strict JSON configs (bot IDs, venues, symbols) at pod startup.
5. **Centralize the Admin API:** Deploy the FastAPI admin app as a singleton deployment that interacts with the fleet via the `admin:commands` topic.

---

Scaling from a single runtime to a distributed fleet brings up configuration management challenges. How do you currently plan to generate, version, and distribute the JSON configuration files required by the `runtime_config_from_mapping()` schema across hundreds of Kubernetes pods?