# Market Maker System Documentation

## High-Level Architecture

The system is a sophisticated low-frequency, event-driven market-making runtime. It operates around an interval-based controller loop, leveraging Model Predictive Control (MPC) algorithms powered by PyTorch to optimize quotes. 

### Key Architectural Concepts
1. **Event-Driven Bus**: Components communicate through an `AbstractEventBus` (primarily implemented as `RedisStreamsBus`). Data flows as versioned `EventEnvelope` messages in JSON.
2. **Synchronous Controller Loop**: Market data and private state snapshots are digested at discrete intervals (frames), creating a synchronous `StrategyInputFrame` that drives the MPC engine.
3. **Multi-Venue (Liquidity Sourcing)**: The system naturally supports a local venue (where orders are placed) and a global reference venue (for liquidity sourcing and reference pricing).
4. **Float-First Representation**: Prices and quantities internally operate as pure floats. Quantization according to exchange tick sizes only occurs at the outer execution edges.

## Major Components and Responsibilities

- **Bus (`mm/bus`)**: Handles event routing via Redis Streams. It provides `publish`, `subscribe`, and persistent `subscribe_envelopes` methods.
- **Core Models (`mm/core`)**: Contains `msgspec` and `dataclass` primitives for the data model (`EventEnvelope`, `MarketEvent`, `PrivateStateSnapshot`, `JointLedgerSnapshot`). Defines event schemas, topic names, and constraints.
- **Controller (`mm/controller`)**: Controls the execution loop context. Packages market data, trading state, and risk metrics into a unified `DecisionContext` per frame.
- **Pricing & MPC Engine (`mm/strategies`, `mm/mpc`, `mm/pricing`)**: The heart of the algorithmic logic. Handles order book VWAP calculation, forecasting market flow, calculating optimal quote spreads/depths via Monte Carlo batched tensor optimization.
- **Execution & Routing (`mm/execution`)**: Transforms continuous target books (from the MPC engine) into discrete orders, manages in-flight plans, validates rules, handles deduplication, and coordinates actual limit order placement logic.
- **Risk (`mm/risk`)**: Operates as a gatekeeper over the proposed `ActionPipeline`. Evaluates proposed orders against cash/inventory limits and filters out unsafe actions.
- **Persistence & Observability (`mm/persistence`, `mm/observability`, `grafana`)**: Manages the ingestion of events into Postgres (for ledger and diagnostics) and Parquet files (for historical data). A robust observability stack exports metrics (Prometheus), logs (Loki), and traces (Tempo).

## Data and Event Flows

The primary flow follows a tick-to-trade lifecycle:
1. **Ingestion**: External market feeds are captured, parsed, and published as `MarketEvent` onto the bus.
2. **Aggregation**: State aggregators build combined views of orderbooks, portfolios, and execution ledgers.
3. **Strategy Frame**: The `Controller` receives a frame of data containing both `local` and `global` market conditions and corresponding `PrivateStateSnapshot`s.
4. **Decision Generation**: The `MpcStrategyEngine` evaluates the frame. It updates internal models (Flow, Price) and runs the MPC optimizer.
5. **Action Generation**: The optimizer outputs a continuous `sigmoid_action` representing desired order depth. `build_joint_target_book_from_sigmoid` discretizes this into a ladder of target orders.
6. **Risk Validation**: The execution plan runs through the risk engine. If the exposure exceeds limits, the actions are trimmed or blocked.
7. **Execution**: The validated `TargetBook` is processed by the execution router, which places/modifies limit orders on the exchange.
8. **Observability Sync**: Detailed `ControllerDiagnostics` are generated, serialized, and persisted (to Postgres) for dashboarding.

## Core Engine Workflows and Algorithms

The `MpcStrategyEngine` orchestrates several models during the `decide` loop:

- **State Enrichment**: The engine computes mid prices, VWAPs at specific depths, and aligns local vs global portfolios. 
- **Market Flow Models**: Estimates order flow intensity using hurdle distributions (e.g., `TorchHurdleLogNormalModel`, `TorchHurdleWeibullModel`). It attempts to forecast the probability and size of incoming flow.
- **Price Models**: Models price drift via mechanical mixtures (`TorchConditionalMechanicalMixturePriceModel`) or residual EWMA processes (`TorchResidualPriceModel`). The engine updates these models sequentially based on previous vs current frames.
- **Monte Carlo Optimizer**: `MonteCarloMPCTensorBatched` simulates hundreds/thousands of potential price/inventory paths using PyTorch tensors. It optimizes over a user-defined objective function considering constraints like `quote_spread_max`, `inventory_threshold`, and `return_risk_aversion_beta`.

## Pricing and Optimization Logic

Pricing relies fundamentally on the "VWAP Depth". The absolute inventory (plus global inventory if liquidity sourcing is enabled) dictates the target depth on the orderbook.
- **VWAP Calculation**: `vwap_from_raw_orderbook` looks into the bids/asks to determine the volume-weighted average price at the required depth. 
- **Continuous Action Space**: The MPC model outputs a soft probability or depth representation (sigmoid gate). This is mapped to a discrete ladder of limit orders matching exchange constraints (e.g., tick size, lot size).
- **Optimization Weighting**: The objective balances expected delta wealth (P&L) against penalties for queue position wait times (`soft_fill_queue_model`) and inventory deviations (`inventory_threshold_penalty_weight`).

## Operational Behavior and Configuration

- **Configuration Files**: Runtime profiles are strictly defined in JSON (e.g., `ControllerRuntimeConfig`, `ExecutionRuntimeConfig`).
- **Warmup**: The system has an `action_start_after_steps` configuration, enforcing a mandatory observation-only period to allow EWMA/price models to stabilize before firing live orders.
- **Observe Only**: A master kill-switch (`observe_only: true`) enables dry-run execution where decisions are fully simulated but no network requests to exchanges are made.
