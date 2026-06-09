# Code-Derived Market Maker System Documentation

This document describes the current low-frequency market-making system from code analysis only. Existing repository documentation was not used as source of truth.

## 1. High-Level Architecture

The system is a Python, asyncio-driven, config-wired market-maker runtime. A single runtime config currently starts one bot with:

- one local quoting venue,
- an optional global/reference or liquidity-source venue,
- public market data adapters,
- optional private state adapters,
- optional order command adapters,
- a ledger-backed strategy frame assembler,
- an MPC controller,
- a target-book risk and execution pipeline,
- optional persistence to Postgres and Parquet,
- runtime observability through Prometheus metrics, JSON logs, optional OTLP traces, and Grafana dashboards.

The main runtime builder is `mm/runtime/engine.py`. The command-line entrypoint is `python -m mm.runtime`, which delegates to `mm.tools.print_events`. With `--config`, that tool loads a JSON runtime config, builds runtime components, prints the resolved runtime contract, starts the runtime, and optionally prints live bus events.

The runtime is built around an `AbstractEventBus` interface:

- `InMemoryBus` provides in-process fan-out through `asyncio.Queue`.
- `RedisStreamsBus` writes compact event envelopes to Redis Streams, fans out original typed objects to local subscribers, and forwards new stream messages from other bus instances into normal subscribers.

Most runtime components communicate through topic names rather than direct calls. The one major exception is the strategy pipeline, which directly owns the `MpcStrategyEngine`, risk engine, and order-manager map.

## 2. Major Components

### Runtime Wiring

`build_runtime_components()` in `mm/runtime/engine.py` is the composition root. It:

1. Loads `.env` through `mm/env_loader.py`.
2. Configures metrics/logging/tracing with `configure_observability()`.
3. Builds an event bus.
4. Builds an `IntervalScheduler`.
5. Selects public market data adapters per source.
6. Builds one `SymbolEngine` per configured source.
7. Builds private state adapters and reconcilers if credentials are configured.
8. Builds order command adapters and `OrderManager` instances if execution is enabled.
9. Builds the local/global `StrategyInputFrameAssembler`.
10. Builds a `PortfolioLedgerService` when private state is available.
11. Builds a `JointStrategyFrameAssembler`.
12. Builds a `RuntimeControlPlane`.
13. Builds the `StrategyRunner` if the controller is enabled.
14. Builds persistence writers and a runtime observability subscriber.

`RuntimeComponents.start()` starts components in this order: persistence, observability, frame assembler, strategy runner, ledger, control plane, symbol engines, execution reconcilers, private reconcilers, then the interval scheduler. `stop()` cancels and closes them in roughly reverse order.

### Configuration

`mm/runtime/config.py` defines the full runtime config model:

- `MarketDataSourceConfig`: exchange, symbol, transport mode, polling limits, and public topic helpers.
- `PrivateRuntimeConfig`: private adapter mode, token or API-key credentials, polling interval, and fill source.
- `BusRuntimeConfig`: memory or Redis Streams settings.
- `PersistenceRuntimeConfig`: Postgres and Parquet sinks, batch size, flush interval, and topics.
- `ObservabilityRuntimeConfig`: metrics, JSON logs, log path, run id, and OTLP settings.
- `LiquiditySourcingRuntimeConfig`: single venue vs liquidity sourcing mode.
- `ExecutionRuntimeConfig`: per-venue planner, observe-only, max order count, max size, and max quote distance.
- `ControllerRuntimeConfig`: MPC model configs, optimizer config, orderbook grid, decision and placement timeouts, risk, reconciliation, observe-only, warmup, and topics.
- `MarketMakerRuntimeConfig`: the resolved bot runtime.

`runtime_config_from_mapping()` expects one bot per config file. It resolves:

- `bots.<id>` into quoting exchange, liquidity source exchange, and reference exchanges,
- a strategy preset,
- local/global sources,
- private exchange blocks,
- controller payload merged with exchange maker fees,
- derived liquidity-sourcing behavior,
- execution and persistence settings.

The example config in `mm/configs/market_maker.runtime.example.json` uses Redis Streams, Postgres or Parquet persistence, Prometheus metrics on port 8000, and an MPC controller.

### Exchange Adapters

`mm/exchanges/factory.py` selects adapters for Nobitex, Wallex, Tabdeal, and Bitpin:

- public market data: REST or websocket, resolved through exchange capabilities,
- private state: REST for Nobitex and Wallex; REST or websocket for Tabdeal and Bitpin,
- order commands: REST only,
- venue limits: pulled from trade-stats exchange info where supported.

The public market-data adapter contract is `AbstractMarketDataAdapter`:

- `connect()`,
- `disconnect()`,
- `trade_stream()`,
- `orderbook_stream()`.

Shared market-data adapter base classes live in `mm/exchanges/shared/provider_adapter.py`. REST polling adapters fetch trades and orderbooks periodically. Websocket adapters reconnect with exponential backoff and push normalized trade/book events to internal queues.

All strategy-facing prices are quote-per-base floats and all quantities are base-asset floats. Venue API unit conversions are handled by exchange-specific wire adapters and command adapters.

### Ingestion And Aggregation

`SymbolEngine` in `mm/ingestion/symbol_engine.py` owns one exchange/symbol public feed. It:

- connects an adapter,
- starts trade and orderbook feed tasks,
- adds deduplicated trades to `RollingTradeBuffer`,
- applies snapshots or deltas to `OrderbookManager`,
- publishes raw trade/orderbook events,
- publishes adapter and normalization errors,
- registers interval callbacks that build `MarketEvent` objects.

`RollingTradeBuffer` keeps sorted trades in a time-bounded deque and deduplicates by trade id or `(exchange, symbol, timestamp, price, qty, side)`.

`OrderbookManager` materializes snapshots from REST snapshots or websocket deltas. It keeps a bounded history of full snapshots and can return the latest snapshot at a requested timestamp.

`MarketEventBuilder` creates a low-frequency `MarketEvent` at each scheduler interval:

- latest orderbook snapshot at `t_end_ms`,
- trades in `[t_start_ms, t_end_ms)`,
- buy volume,
- sell volume,
- VWAP,
- top-level imbalance,
- microprice.

`IntervalScheduler` aligns callbacks to wall-clock interval boundaries. A slow callback blocks later callbacks in the same scheduler loop because callbacks are awaited sequentially.

### Frame Assembly

`StrategyInputFrameAssembler` consumes local `MarketEvent` objects and optionally global `MarketEvent` objects. For each local event, it:

- drains latest global/private inputs,
- optionally waits up to `global_grace_ms` for a matching global `t_end_ms`,
- selects the latest global event with `t_end_ms <= local.t_end_ms`,
- marks global status as `disabled`, `missing`, `stale`, or `ready`,
- attaches latest private snapshots if queues are configured,
- returns a `StrategyInputFrame`.

`JointStrategyFrameAssembler` wraps this with ledger state. It:

- computes a reference price from local or global microprice,
- asks `PortfolioLedgerService` for a joint ledger snapshot when available,
- computes market/private freshness,
- builds and publishes a `JointStrategyFrame` on `strategy:joint:frames:<exchange>:<symbol>:<interval>`.

### Private State And Ledger

`PrivateStateReconciler` owns authenticated private state ingestion. It:

- obtains an initial private snapshot,
- for REST polling adapters, runs a unified snapshot loop and publishes one authoritative snapshot per poll plus optional fill events,
- for incremental adapters, listens to balance, open-order, order-update, and fill streams,
- publishes all private events to topics under `private:state:<exchange>:...`.

`PrivateStateStore` keeps the normalized private state. It:

- stores balances, open orders, and fills,
- removes terminal or zero-remaining orders,
- preserves zero balances for the trading pair's base and quote assets,
- normalizes locked balances so live open-order collateral is reflected.

`PortfolioLedgerService` subscribes to private topics and maintains `VenueLedgerStateMachine` instances for the local and optional global venue. After each update, it publishes:

- `ledger:snapshot:<exchange>` with a venue payload,
- `ledger:snapshot` with a joint payload,
- optional ledger vs order-manager consistency diagnostics.

When ledger live orders and `OrderManager.live_orders` disagree with the same mismatch for three consecutive diagnostics, the service publishes `LedgerOrderManagerMismatch` on `ledger:diagnostics:mismatch:<exchange>`. This event includes mismatch reason, order id, ledger value, manager value, first-seen time, age, and consecutive count.

The strategy pipeline requires a `JointStrategyFrame` with a ledger. If no ledger exists, `StrategyRunner.on_joint_frame()` raises.

### Strategy Runner

`StrategyRunner` in `mm/application/strategy_runner.py` is the main decision loop. For each `JointStrategyFrame`, it:

1. Ensures an input frame and ledger are present.
2. Increments `frame_step`.
3. Syncs the runtime control-plane kill switch into the risk engine.
4. Extracts live orders from the ledger.
5. Produces a clean frame by subtracting own live orders from local and global orderbooks using tolerance-aware price matching.
6. Builds `TradingState` from the local ledger.
7. Creates a `DecisionContext` for diagnostics. The context is the final assembler for `ControllerDiagnostics`, with timing, risk, execution-gate, and action-pipeline facts held in narrower diagnostic state records. The runner records lifecycle facts through those state records rather than broad compatibility aliases on the context.
8. Runs `MpcStrategyEngine.decide()` under `decision_timeout_ms`.
9. Prepares the execution target book according to per-venue execution settings.
10. Computes venue execution gates.
11. Runs target-book risk validation.
12. Applies risk output to diagnostics.
13. Publishes `StrategyDecision`, target-book, and approved `Action` events.
14. Optionally syncs live orders through order managers with a timeout.
15. Publishes a compact journal event.
16. Always publishes `ControllerDiagnostics` in `finally`.

When decision compute exceeds `decision_timeout_ms`, or a previous timed-out optimizer call is still running, the runner publishes a `ControllerFallback` event with `fallback_policy="hold_current_orders"`. It does not publish a new target book or approved action for that frame, so existing exchange orders are left unchanged.

For each processed frame, the runner assigns deterministic lineage IDs: `frame_id`, `decision_id`, `target_book_id`, and per-venue `execution_plan_id`. `StrategyDecision`, `ControllerDiagnostics`, `JointTargetBook`, venue target-book payloads, execution sync events, and router request/accepted/failed events carry these IDs. This lets replay and incident analysis follow frame -> decision -> target book -> execution without inferring relationships from timestamp/topic proximity.

`MpcStrategyEngine` depends on a strategy-level optimizer backend protocol rather than the concrete tensor optimizer directly. The default backend wraps the existing `MonteCarloMPCTensorBatched` controller, while tests or future worker-process implementations can inject another backend that provides `optimize(state)`, model handles, optimizer diagnostics, and optional global sigmoid action.

Live execution is blocked by:

- controller observe-only,
- control-plane observe-only,
- warmup steps,
- paused actors,
- paused venue roles,
- missing order manager,
- global live-orders disabled,
- stale global market data,
- no orders unless a replace policy allows empty-book sync,
- risk rejection,
- execution timeout or sync error.

### Control Plane

`RuntimeControlPlane` consumes `admin:commands` and maintains runtime safety state:

- pause/resume actor,
- kill switch,
- observe-only mode,
- pause/resume local or global venue roles,
- request cancel actor orders,
- request cancel all.

It publishes state to `admin:events` and updates Prometheus gauges for kill switch and observe-only state.

`mm/api/admin.py` exposes a FastAPI app with health, status, config, metrics, and command endpoints. The admin app creates its own bus from environment unless injected.

## 3. Core Market-Making Engine

### State Construction

`MpcStrategyEngine.decide()` converts a `StrategyInputFrame` and optional ledger into an MPC `State`:

- local mid from best bid/ask,
- orderbook depth relative to mid, clipped by `spread_max`,
- local buy/sell volume,
- local inventory and cash,
- local buy/sell VWAP for a dynamic depth,
- optional global mid/orderbook/volume/VWAP,
- optional global inventory and cash,
- a reference price selected from local or global sell-side VWAP.

VWAP depth is derived from inventory:

- single venue: `abs(local_inventory)`,
- liquidity sourcing: `abs(local_inventory + global_inventory)`.

If no ledger snapshot is fresh, state construction falls back to private snapshots attached to the frame. In the normal runtime path, the strategy runner requires a ledger-backed frame.

### Online Models

The controller owns separate local and global flow/price models:

- Local market flow: gamma, Weibull, hurdle gamma, hurdle Weibull, or hurdle lognormal.
- Global market flow: same model family.
- Local price: residual VWAP model or conditional mechanical mixture model.
- Global price: EWMA Gaussian VWAP delta model, static Gaussian deltas, residual model, conditional mechanical mixture model, or disabled.

Model scoring happens before updates. `score_frame_models()` computes likelihood or NLL diagnostics against pre-update parameters. After scoring:

- global VWAP price model updates from observed global buy/sell VWAPs,
- local/global market-flow models update from interval volumes,
- local residual price model updates from the previous/current local frame,
- global residual price model updates from previous/current global frame when configured.

### MPC Optimization

The optimizer is `MonteCarloMPCTensorBatched` in `mm/mpc/mpc_tensor_batched.py`. It is a differentiable Monte Carlo MPC optimizer using PyTorch.

The action is a mixture of sigmoids per side. Each side has cumulative quoted depth as a function of spread:

```
D(spread) = sum_i L_i / (1 + exp(-k_i * (spread - x0_i)))
```

The optimizer searches over raw action parameters and maps them to constrained effective parameters through `LParamConfig` and sigmoid/softplus transforms. Per-venue optimizer settings include:

- number of components,
- risk aversion beta,
- objective weight,
- quote size min/max per side,
- quote capacity ratio,
- quote imbalance skew,
- quote spread max/min-depth/k-epsilon,
- soft fill temperature,
- soft fill activation,
- soft fill queue model.

For each optimize call, the optimizer:

1. Converts the current orderbook into spread-grid tensors.
2. Computes current inventory/cash capacity bounds.
3. Samples market order flow and price innovations for `n_samples * horizon`.
4. Runs Adam for `adam_epochs`.
5. For each epoch and horizon step, simulates soft fills against combined public depth and the strategy's quoted depth.
6. Updates simulated cash, inventory, book shape, mid, and VWAPs.
7. Computes per-path return.
8. Maximizes an entropic risk-adjusted return objective with optional cash/inventory threshold terms.
9. Stores optimizer diagnostics, including initial/final/best objective snapshots, action snapshots, gradient issues, and timing.

The default optimizer config in `_build_controller()` is modest:

- `n_samples = 8`,
- `horizon = 2`,
- `adam_epochs = 10`,
- CPU device,
- spread grid from `ControllerRuntimeConfig.orderbook_grid`.

The example runtime config uses larger values: `n_samples = 64`, `horizon = 1`, `adam_epochs = 30`.

If liquidity sourcing is enabled or global reference pricing is active, the optimizer delegates to `mm/mpc/joint_optimizer.py`, which optimizes local and global action vectors with joint objective terms.

### Objective

The single-venue terminal objective is implemented in `mm/mpc/reward_tensor.py`:

```
J = entropic_return + threshold_shortfall_penalty
```

The entropic return is a certainty equivalent over Monte Carlo path returns:

```
CE = min(R) - (1 / beta) * log(mean(exp(-beta * (R - min(R)))))
```

Returns are normalized by current wealth:

```
wealth = cash + inventory * sell_vwap
```

The threshold penalty is a one-sided normalized shortfall on simulated cash and inventory paths. `inventory_threshold_penalty_weight`, `cash_threshold`, and `inventory_threshold` are parsed into venue optimizer params and passed into the single-venue and joint terminal objectives.

### Fill Simulation

`soft_fill_batched()` simulates the combined market book plus the strategy's quote depth. It:

- converts current cumulative market depth into marginal levels,
- converts quoted cumulative sigmoid depth into marginal quote levels,
- combines public and own quote levels,
- applies market order flow through a differentiable fill fraction,
- splits consumed volume between public book and own quotes,
- supports `back_of_queue` and `pro_rata` queue models,
- returns the next market-only book plus cash and inventory deltas from own fills.

Maker fees are read from the real execution model config and applied in fill simulation.

### Quote Discretization

`MpcStrategyEngine.decide()` converts the best continuous action to a target book through `build_joint_target_book_from_sigmoid()`:

1. Build a diagnostic continuous depth series.
2. Evaluate cumulative depth at each configured spread ladder point.
3. Convert cumulative depth to marginal order sizes.
4. Build bid orders at `mid - spread` and ask orders at `mid + spread`.
5. Drop zero marginal quantities and invalid prices.
6. Cap to `max_orders_per_side` by keeping the largest-quantity orders per side.
7. Build `VenueTargetBook` and `JointTargetBook`.

The strategy-side order pipeline intentionally does not perform final venue rounding in the current path. The execution boundary performs venue limit normalization and validation through `ExchangeRulesGateway`.

## 4. Risk Management

The main runtime builds `NetDeltaRiskEngine` when risk is enabled. It validates the approved target-book transition against ledger state.

Frame-level gates reject all orders when:

- kill switch is enabled,
- local market data is stale,
- local private state is missing or stale,
- global market data is missing or stale when global live execution requires it,
- global private state is missing or stale when required.

Venue transition checks evaluate:

- actor live-order collateral,
- target locked base and quote,
- open notional,
- available base and quote,
- wallet totals,
- per-order max notional,
- max open notional,
- sufficient base for asks,
- sufficient quote for bids,
- min cash,
- min inventory,
- max inventory.

For replace policies, the engine assumes existing actor or unattributed scoped orders will be released. For reconcile policies, it uses available balances plus the actor's already locked collateral.

Risk may approve a partial target book by dropping only the rejected side when the rejection reason maps cleanly to buy-side or sell-side constraints.

## 5. Execution Workflow

Execution is target-state reconciliation, not fire-and-forget order placement.

`OrderManager` is a facade over:

- an execution planner,
- an exchange rules gateway,
- a stateless order router,
- private-confirmed live-order state,
- pending place/cancel changes accepted by the router but not yet confirmed by private state.

For each venue target book:

1. Select planner by policy:
   - `reconcile`: keep live orders that match side, symbol, price tolerance, and quantity tolerance; cancel unmatched live orders; place unmatched targets.
   - `replace`: cancel all scoped live orders, including unattributed orders, then place the full target book.
2. Normalize and validate place commands through venue limits.
3. Execute cancels before places.
4. Publish requested/accepted/failed lifecycle events.
5. Record successful router responses as pending places/cancels without mutating confirmed live orders.
6. Hold additional syncs for the same venue/symbol/actor while pending confirmations exist, avoiding duplicate replacement while private state catches up.
7. Publish execution snapshot with confirmed live-order count, pending place/cancel count, and oldest pending age.

The authoritative private-state path then reconciles exchange-confirmed state back into the order manager through `ExecutionStateReconciler` and into the ledger through `PortfolioLedgerService`. A private open-order event confirms a pending place and moves it into live orders; terminal private updates or snapshots remove pending cancels once the order disappears or becomes terminal.

## 6. Event Model And Data Lifecycle

### Event Envelope

Durable events use `EventEnvelope` from `mm/core/events.py`:

- schema and version,
- event id,
- run id,
- source,
- topic,
- event type,
- exchange,
- symbol,
- actor id,
- timestamp,
- JSON-compatible payload.

`envelope_event()` extracts top-level metadata from payload fields when possible.

### Core Topics

Representative runtime topics:

- `market:raw:<exchange>:<symbol>:trades`
- `market:raw:<exchange>:<symbol>:orderbook`
- `market:errors:<exchange>:<symbol>`
- `market:events:<exchange>:<symbol>:<interval_ms>`
- `private:state:<exchange>:snapshot`
- `private:state:<exchange>:balances` for incremental adapters
- `private:state:<exchange>:open_orders` for incremental adapters
- `private:state:<exchange>:order_updates`
- `private:state:<exchange>:fills`
- `private:state:<exchange>:errors`
- `ledger:snapshot:<exchange>`
- `ledger:snapshot`
- `strategy:joint:frames:<exchange>:<symbol>:<interval_ms>`
- `strategy:decisions`
- `strategy:actions`
- `strategy:diagnostics:<actor_id>`
- `strategy:target_books:<actor_id>`
- `risk:events`
- `execution:orders:<exchange>`
- `execution:state:<exchange>`
- `execution:errors`
- `admin:commands`
- `admin:events`

### Default Persistence Topics

`MarketMakerRuntimeConfig.default_persistence_topics` persists:

- local and global market events,
- joint frames,
- ledger snapshots,
- strategy actions,
- execution errors,
- execution orders and state per exchange,
- private snapshots/fills/errors per exchange,
- persistent ledger/order-manager mismatch diagnostics per exchange,
- decisions, diagnostics, and target books when the controller is configured.

Raw market-data topics are not in the default persistence topic list, but the observability subscriber watches raw topics for metrics.

### Persistence

`PostgresEventWriter` subscribes to configured topics and writes compacted event envelopes to:

- `mm_events`: append-only event table,
- `mm_market_facts`: typed market interval facts for fast dashboards and analytics,
- `mm_action_stage_facts`: typed controller action-pipeline stage facts,
- `mm_execution_facts`: typed order execution lifecycle facts,
- `mm_target_order_facts`: typed target-book intent orders,
- `mm_model_score_facts`: typed model score and update-skip facts,
- `mm_optimizer_facts`: typed optimizer timing and objective facts,
- `mm_optimizer_component_facts`: typed optimizer objective component facts,
- `mm_latest_state`: latest state table for private and execution state.

It compacts persisted market events by dropping raw trade lists after interval summaries are present, compacts persisted joint frames by replacing the embedded ledger with a summary plus `ledger:snapshot` reference, and compacts persisted controller-diagnostics action-pipeline order/rejection arrays into counts, side counts, digests, and small samples. Private snapshots remain full in Postgres so `mm_latest_state` can preserve current-state truth. It also materializes every `MarketEvent` into `mm_market_facts` with typed columns for top-of-book prices and quantities, mid, spread, microprice, VWAP, interval volumes, imbalance, trade count, level counts, and event timestamps. Every `ControllerDiagnostics.action_pipeline` stage is materialized into `mm_action_stage_facts` with typed columns for stage counts, side counts, quantities, notionals, rejection counts, compacted counts, and order/rejection digests. Execution sync, place, cancel, validation rejection, fill, and execution snapshot events are materialized into `mm_execution_facts` with typed columns for lifecycle, success, confirmed/pending state kind, order ids, side, price, quantity, remaining/filled quantity, fee, status, error, latency, count, and lineage IDs. Every `JointTargetBook` order is materialized into `mm_target_order_facts` with typed columns for actor, mode, venue role, exchange, symbol, policy, side, price, quantity, notional, order type, tags, and lineage IDs. Controller model scores are materialized into `mm_model_score_facts` with model, kind, update status, skip reason, likelihood/NLL/residual, and small shape counts for observation and parameter dictionaries, and lineage IDs. Optimizer diagnostics are materialized into `mm_optimizer_facts` and `mm_optimizer_component_facts` with timing, objective values, improvement, best expected reward/return fields, gradient/rollout counts, and per-snapshot objective components, and lineage IDs. It creates indexes on run/topic/timestamp, exchange/symbol/timestamp, topic/type/timestamp, actor/timestamp, market-fact exchange/symbol/timestamp, action-stage actor/exchange/symbol/stage/timestamp, execution action/order/timestamp, target-order actor/venue/exchange/symbol/timestamp, model actor/exchange/symbol/timestamp, and optimizer actor/exchange/symbol/timestamp. Batches are written in a worker thread through a reusable psycopg2 connection. Failed writes drop the connection, retry with bounded backoff, and spill exhausted batches to local JSONL files under `persistence_spool/postgres`.

`ParquetEventWriter` writes partitioned Parquet rows under:

```
event_date=<yyyy-mm-dd>/exchange=<exchange>/symbol=<symbol>/event_type=<event_type>/hour=<hh>/part-<timestamp>-<uuid>.parquet
```

Each flush writes immutable part files; live writes do not read or rewrite existing Parquet files. Parquet rows retain one compact `envelope_json` for replay plus typed nullable columns for high-value analytics fields. The writer does not store a separate `payload_json` copy. For archival storage, it compacts `MarketEvent.trades` after interval features are already present, compacts joint-frame embedded ledgers to summaries/references, compacts controller-diagnostics action-pipeline order/rejection arrays, and removes zero private balances while preserving counts and compaction flags.

Typed Parquet columns cover market top-of-book/depth/flow facts, controller state/model-count/optimizer-objective facts, target-book order counts and notionals, and execution lifecycle/order fields. The replay path still reads `envelope_json`, while offline scans can filter and aggregate common values without parsing JSON. This makes Parquet suitable for compact replay and offline analysis of summarized facts, not exact raw trade-list recovery.

If pyarrow is unavailable, it falls back to JSONL files in the same partition layout.

`mm/persistence/replay.py` can read Parquet or JSONL rows back into event envelopes and group replayed events by explicit lineage IDs. `read_lineage_chains()` returns one chain per `frame_id`, separating decisions, diagnostics, target books, execution events, and other linked events. Replay contract tests verify that persisted rows can reconstruct decision -> target book -> execution call chains and can distinguish pending optimistic execution changes from confirmed live orders.

`RedisStreamsBus` writes compact envelopes to Redis streams named from topics. It compacts zero balances and private snapshots before stream writing. Local subscribers receive the original in-process object, while remote stream messages are decoded to envelope payloads and forwarded through `subscribe()`/`subscribe_queue()`. Each bus instance tags its own stream writes and skips them in its stream forwarder to avoid duplicate same-process delivery.

## 7. Observability

### Metrics

`RuntimeObservabilitySubscriber` subscribes to selected topics and converts events into metrics:

- raw market event count, timestamps, ingest lag, and gaps,
- joint frame readiness and local/global latency skew,
- controller decision latency, order count, inventory, cash,
- controller diagnostics stage latency,
- model NLL/log-likelihood/residuals,
- action pipeline stage counts, quantities, notionals, and rejections,
- risk approval, risk rejections, and exposure fields,
- execution gate blocks,
- order sync counts and latency,
- placement/cancel accepted counts,
- live order counts and quantities by side,
- placement failures, fill counts and quantities,
- persistence queue depth and write latency,
- event payload and envelope bytes by topic, event type, and pipeline stage,
- runtime errors.

Metrics are stored by a custom in-process Prometheus text registry in `mm/observability/registry.py`. `configure_observability()` starts an HTTP server on the configured metrics port when metrics are enabled.

### Logs And Tracing

`log_event()` writes structured JSON to:

- stdout when `MM_JSON_LOG` is set,
- `MM_LOG_PATH` when configured.

The Docker Compose stack runs Grafana Alloy, which tails `./logs/*.jsonl` into Loki. `mm/observability/traces.py` provides optional OpenTelemetry spans; `configure_observability()` can configure an OTLP HTTP trace exporter.

### Grafana

Docker Compose provisions:

- Postgres,
- Prometheus,
- Loki,
- Tempo,
- Alloy,
- Grafana,
- Redis,
- RedisInsight.

Grafana datasources are provisioned with fixed UIDs:

- `postgres-mm`,
- `prometheus-mm`,
- `loki-mm`,
- `tempo-mm`.

There are two dashboard folders:

- `Market Maker`: older dashboards under `grafana/dashboards`.
- `Market Maker - Observability v2`: newer dashboards under `grafana/observability-v2`.

The v2 dashboards cover runtime overview, controller health, market data, optimizer diagnostics, model quality, action pipeline, risk/execution, target-order stems, orderbook depth, liquidity, portfolio/PnL, fills, execution errors, incidents, and runtime logs.

Most deep trading panels query `mm_events` JSONB in Postgres. Runtime liveness and latency panels use Prometheus metrics. Log dashboards use Loki.

No Grafana alert rule provisioning was found in code.

## 8. Testing Surface

The repository contains 56 unit test files under `tests/unit`. The tests cover:

- exchange payload normalization and contracts,
- runtime config and migration contracts,
- Redis Streams bus,
- orderbook grids and reference VWAPs,
- market data pipeline,
- frame assembly,
- MPC objective and optimizer diagnostics,
- strategy target books and runner gates,
- risk limits and net-delta risk,
- execution planning, reconciliation, validation, and order managers,
- private state reconciler,
- observability dashboards and subscriber behavior,
- Postgres latest-state deduplication,
- replay and parquet writing.

The test suite is broad for unit-level behavior. The codebase does not show a dedicated end-to-end exchange simulator or load/stress test harness in the inspected files.
