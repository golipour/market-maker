<!-- Merged single-file version of docs/algorithm/ -->


---

# Market Maker: End-to-End Architecture & Algorithm

> A code-derived walkthrough of `mm/`. The document is organized so a reader can follow one
> market-event sample from ingestion, all the way through the Model Predictive Control (MPC)
> optimizer, to a published order on the wire. Every section maps directly to files in the
> repository; cited line ranges use `startLine:endLine:filepath` so the snippets can be opened
> in the editor.
>
> **Source-of-truth rule.** The narrative below is built strictly from the current Python
> implementation. Outdated notes under `research/` and `docs/old/` are **not** consulted for
> algorithmic facts. Where the code is ambiguous, that ambiguity is called out.
>
> **Scope of the math.** All math is presented in a single notation block at the top of each
> major section so the reader does not have to scroll. The notation is consistent across the
> whole document. Variable names match the Python identifiers as much as possible.

---

## 1. Bird's-eye view

The runtime is an asynchronous Python application that connects to one or more exchanges,
aggregates per-interval market events, runs a per-actor MPC controller, and routes the
resulting orders back to the venue. There are five loosely-coupled "planes":

| Plane | Owner | Purpose |
| --- | --- | --- |
| **Ingestion** | `mm.ingestion`, `mm.aggregation` | Pull orderbook + trade feeds, aggregate to per-interval `MarketEvent`s, publish to bus. |
| **Private state** | `mm.exchanges.private.*` | Pull fills, balances, open orders. Produce `PrivateStateSnapshot`s. |
| **Frame assembly** | `mm.application.frame_assembler`, `mm.application.joint_frame_assembler` | Align local/global market events with private snapshots, build `StrategyInputFrame` and `JointStrategyFrame`. |
| **Strategy** | `mm.application.strategy_runner`, `mm.strategies.mpc_engine`, `mm.mpc.*` | Run the MPC optimizer, transform the continuous action into a target book, validate against risk, publish for execution. |
| **Execution & reconciliation** | `mm.core.order_manager`, `mm.core.reconciler`, `mm.execution.*` | Convert target books to venue orders, send to the exchange, reconcile fills, keep the ledger in sync. |

A single process runs all five; the bus is in-memory by default and can be swapped for
Redis streams (`BusKind = "memory" | "redis_streams"`, see
`mm/runtime/config.py:33`). Each plane runs as its own coroutine in the same event loop,
so the only blocking work is the per-decision MPC optimizer, which is dispatched to a
`ThreadPoolExecutor` by the strategy runner.

The end-to-end data path for one strategy decision looks like this:

```
  exchange WS/REST  ──►  symbol engine  ──►  raw topics (bus)
                                                │
                                                ▼
                                  rolling trade buffer + orderbook manager
                                                │
                                                ▼
                                         MarketEvent (per Δt)
                                                │
              ┌─────────────────────────────────┼─────────────────────────────────┐
              ▼                                                                   ▼
   frame_assembler (local+global)                                joint_frame_assembler
              │                                                                   │
              ▼                                                                   ▼
    StrategyInputFrame                                            JointStrategyFrame
                                                                                  │
                                                                                  ▼
                                                                  StrategyRunner.run_step
                                                                                  │
                  ┌───────────────────────────────────────────────────────────────┤
                  │                                                               │
                  ▼                                                               ▼
       live_orders (ledger)                                       MpcStrategyEngine.decide
                  │                                                               │
                  ▼                                                               ▼
       clean_strategy_frame (own orders removed)            MonteCarloMPCTensorBatched.optimize
                  │                                                               │
                  └──────────────►  Action validation + NetDeltaRiskEngine  ◄──────┘
                                                │
                                                ▼
                                  JointTargetBook  ──►  OrderManager  ──►  exchange
```

## 2. Conventions used in this document

- **Citations** look like `startLine:endLine:filepath`. They open the corresponding file
  in the editor at the cited range.
- **Code identifiers** are written in `monospace` and match the Python names exactly.
- **Math** is presented as plain Unicode (e.g. `θ` for `theta`, `σ` for `sigma`, `Δ` for
  `delta`). Tensor shapes are noted as `(B, H, K)` where `B` = MC batch, `H` = horizon,
  `K` = asset-specific dimension.
- **Defaults** are taken from the live config schema in `mm/runtime/config.py` unless
  another default is stated.
- The "Live config example" referenced throughout is
  `mm/configs/market_maker.runtime.example.json`.

## 3. Where to look next

- Runtime & processes: [01-runtime.md](01-runtime.md)
- Data ingestion pipeline: [02-ingestion.md](02-ingestion.md)
- Frame assembly: [03-frame-assembly.md](03-frame-assembly.md)
- Strategy runner: [04-strategy-runner.md](04-strategy-runner.md)
- MPC state & action: [05-mpc-state.md](05-mpc-state.md)
- Stochastic models: [06-mpc-models.md](06-mpc-models.md)
- The optimizer: [07-mpc-optimizer.md](07-mpc-optimizer.md)
- Reward & objective: [08-objective.md](08-objective.md)
- Target book construction: [09-target-book.md](09-target-book.md)
- Risk engine & validity/viability: [10-risk.md](10-risk.md)
- Edge cases & failure handling: [11-edge-cases.md](11-edge-cases.md)
- A sample event traced end-to-end: [12-end-to-end.md](12-end-to-end.md)
- Glossary: [13-glossary.md](13-glossary.md)

---

# 01 — Runtime & processes

This section covers `mm/runtime/`: the entry point, the per-process `RuntimeComponents`
that owns every long-lived object, the event bus, the control plane, the persistence
layer, and how an actor is wired to a market data source and a strategy.

## 1.1 Entry point

`mm/runtime/__main__.py` is the entry. It is intentionally a thin shim: it just calls
`mm.tools.print_events.main` so the runtime binary doubles as a diagnostic printer for
market events on stdout. Real work happens once `RuntimeComponents` is constructed.

```10:13:mm/runtime/__main__.py
import asyncio
from mm.tools.print_events import main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

`mm/tools/print_events.py` reads CLI args, loads `RuntimeComponents`, and either
streams events to stdout or runs the full event-driven runtime.

## 1.2 `RuntimeComponents`

`RuntimeComponents` is the composition root. It builds every long-lived collaborator
and exposes a `start()` / `stop()` pair.

```78:111:mm/runtime/engine.py
@dataclass
class RuntimeComponents:
    settings: RuntimeSettings
    bus: AbstractEventBus
    scheduler: IntervalScheduler
    market: MarketDataPlane
    execution: ExecutionPlane
    ledger: LedgerPlane
    reconciler: Reconciler
    persistence: PersistenceLayer
    observability: ObservabilityPlane
    control_plane: ControlPlane
    strategies: dict[str, StrategyRunner]
```

Each "plane" is itself a small aggregator:

- `MarketDataPlane` (in `runtime/engine.py`) — owns one `SymbolEngine` per
  `(exchange, symbol)` from `settings.market`, each running a feed task and a
  periodic publish task. It also constructs `ProviderAdapter` instances.
- `ExecutionPlane` — wires `OrderManager` instances to per-venue exchange adapters and
  the bus.
- `LedgerPlane` — owns the `Ledger` instance plus per-venue `VenueLedger` projections
  for fast reads.
- `PersistencePlane` — wraps SQLite (or in-memory) stores and the retention policy.
- `ObservabilityPlane` — owns the controller-diagnostics publisher, fill-delta
  publisher, reconciler publisher, and warm-start spy.
- `ControlPlane` — exposes the actor kill switch, allocation breaker, and venue
  pause flags.

The `RuntimeComponents.build` classmethod is the only place where concrete classes are
selected based on configuration. It is the single place to look if a setting seems to
be ignored.

## 1.3 The event bus

`mm/bus/base.py:42` defines `AbstractEventBus`:

```42:60:mm/bus/base.py
class AbstractEventBus(Protocol):
    async def publish(self, topic: str, payload: Any) -> None: ...
    def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> None: ...
    def subscribe_queue(self, topic: str) -> asyncio.Queue: ...
```

Two implementations exist:

- `InMemoryEventBus` — an `asyncio` fan-out with per-subscriber `asyncio.Queue`s and a
  backpressure policy (`block`, `drop_oldest`, `drop_newest`).
- `RedisStreamsBus` — used when the runtime is split across processes (e.g. a
  dedicated market-data worker on one host and a strategy worker on another). The
  topic naming convention is fixed by the runtime config so the streams are
  stable.

Topics are *namespaced strings*; the controller documents them in code:

```120:140:mm/runtime/config.py
def market_event_topic(self, interval_ms: int) -> str:
    return f"market:events:{self.normalized_exchange()}:{self.symbol}:{interval_ms}"
def raw_trade_topic(self) -> str:
    return f"market:raw:{self.normalized_exchange()}:{self.symbol}:trades"
def raw_orderbook_topic(self) -> str:
    return f"market:raw:{self.normalized_exchange()}:{self.symbol}:orderbook"
```

So one strategy's decision flow lives on the topics `market:events:<ex>:<sym>:<Δt>` and
`market:raw:<ex>:<sym>:{trades,orderbook}`. Per-strategy private topics follow the
`private:<ex>:<sym>` convention.

## 1.4 Control plane

`mm/control/` holds the `ControlPlane` that exposes the runtime flags an operator can
flip without restarting the process:

- `actor_kill_switch` — when armed, a strategy publishes zero orders and the runner
  emits only diagnostics. See [10-risk.md](10-risk.md).
- `actor_allocation_breaker` — when tripped (by the `NetDeltaRiskEngine`), the actor's
  execution gate is closed but it still computes decisions. See §3.1 below.
- `venue_pause[ex]` — pauses a single venue for a single actor; the venue's
  `VenueExecutionGate` short-circuits to "paused".
- `warmup_until_ms` — suppresses order placement until the per-strategy warmup window
  has elapsed since process start; the optimizer still runs, but the runner routes
  every decision to the preview bus, not the live `OrderManager`.

---

# 02 — Ingestion: from raw trades & orderbooks to per-interval `MarketEvent`s

This section walks the path from a WebSocket/REST feed into the `MarketEvent` consumed
by the strategy. The pipeline is per-`(exchange, symbol)` and is owned by
`SymbolEngine`.

## 2.1 Notational conventions

- `Δt` — aggregation interval, e.g. 1000 ms.
- `MarketEvent` — a closed-window snapshot over `[t_start_ms, t_end_ms]`.
- `RollingTradeBuffer` — sliding window of recent trades (capped by `trades_limit`).
- `OrderbookManager` — keeps the latest top-N levels for both sides.

## 2.2 `SymbolEngine`

The engine is created once per `(exchange, symbol, interval_ms)` triple. It launches:

1. a **feed task** that calls into `AbstractMarketDataAdapter.stream_trades()` and
   `stream_orderbook()`, and
2. a **publish task** that runs every `Δt` ms, builds a `MarketEvent`, and emits it to
   the bus.

```10:39:mm/ingestion/symbol_engine.py
class SymbolEngine:
    def __init__(self, *, adapter, exchange, symbol, interval_ms,
                 bus, output_topic, raw_trade_topic, raw_orderbook_topic,
                 depth_limit, trades_limit, ...):
        self.adapter = adapter
        self.bus = bus
        ...
        self.trade_buffer = RollingTradeBuffer(...)
        self.orderbook_mgr = OrderbookManager(...)
```

### 2.2.1 Feed task

The feed task receives trades and orderbook diffs from the adapter, **publishes the
raw payload** to a side topic (so other consumers can attach), and pushes the parsed
object into the in-memory buffers:

```71:99:mm/ingestion/symbol_engine.py
async def _feed_trades(self):
    async for trade in self.adapter.stream_trades():
        self.trade_buffer.add(trade)
        await self.bus.publish(self.raw_trade_topic(), trade)

async def _feed_orderbook(self):
    async for snapshot in self.adapter.stream_orderbook():
        self.orderbook_mgr.apply_snapshot(snapshot)
        await self.bus.publish(self.raw_orderbook_topic(), snapshot)
```

The raw topics are written *before* the in-memory buffer is touched, so any subscriber
that listens to `market:raw:*` will see a consistent view with the engine itself.
Adapter errors are funnelled through `_feed_adapter_errors` and emitted to
`market:errors:<ex>:<sym>`.

### 2.2.2 Publish task

Every `Δt` ms the engine drains the trade buffer for the *closed* window `[t_start,
t_end]` and combines it with the current orderbook snapshot to produce a
`MarketEvent`. The interval scheduler calls `_publish_window`:

```110:140:mm/ingestion/symbol_engine.py
def _publish_window(self, t_start_ms: int, t_end_ms: int) -> MarketEvent:
    trades = self.trade_buffer.window(t_start_ms, t_end_ms)
    orderbook = self.orderbook_mgr.snapshot()
    return build_market_event(
        exchange=..., symbol=..., interval_ms=..., t_start_ms=..., t_end_ms=...,
        orderbook=orderbook, trades=trades,
    )
```

The actual builder is `mm/aggregation/builder.py:build_market_event`, which fills the
`sell_volume`, `buy_volume`, `vwap`, `imbalance`, `microprice`, and the
`book_buy_vwap` / `book_sell_vwap` / `book_vwap_depth` aggregates (see
[09-target-book.md](09-target-book.md) for the math).

## 2.3 Aggregation: from trades to a `MarketEvent`

The builder's job is to map a closed window of trades plus a top-of-book snapshot to
the canonical event. The MPC's stochastic models consume the *aggregated* fields
(`buy_volume`, `sell_volume`, `vwap`, `imbalance`), not the raw trades; the orderbook
itself is kept as a `TopOfBook` (or L2 list) for `order_constraints` and for the
ladder builder.

```1:60:mm/aggregation/builder.py
def build_market_event(*, exchange, symbol, interval_ms, t_start_ms, t_end_ms,
                       orderbook, trades) -> MarketEvent:
    buy = sum(t.qty for t in trades if t.side == "buy")
    sell = sum(t.qty for t in trades if t.side == "sell")
    vwap = _vwap(trades) if trades else mid_from_orderbook(orderbook)
    imbalance = (buy - sell) / max(buy + sell, eps)
    microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
    book_buy_vwap, book_sell_vwap, book_vwap_depth = book_vwaps(orderbook)
    return MarketEvent(...)
```

**Definitions used here:**

- `vwap = Σ price·qty / Σ qty` (volume-weighted average price over the window).
- `imbalance = (buy − sell) / (buy + sell)` (sign indicates directional pressure).
- `microprice = (bid·ask_qty + ask·bid_qty) / (bid_qty + ask_qty)` (a depth-weighted
  inside fair price).
- `book_buy_vwap` and `book_sell_vwap` are cumulative VWAPs of the bid and ask
  sides, with `book_vwap_depth` being the notional traded through those levels.

These aggregates become the **flow features** consumed by `mm/mpc/market_order_flow_torch.py`
and the **price features** consumed by `mm/mpc/price_torch.py`. See
[06-mpc-models.md](06-mpc-models.md).

## 2.4 What happens on adapter error

If the adapter raises while streaming, the engine converts the exception into an
`AdapterError` payload on the `market:errors:<ex>:<sym>` topic. The strategy runner
ignores market data whose `MarketEvent` cannot be parsed and the control plane logs the
error; the next successful window resumes the pipeline.

## 2.5 What happens on slow consumers

The bus backpressure policy is configured globally:

- `block` (default) — `publish` awaits until the slowest subscriber drains.
- `drop_oldest` — the oldest queued message for a slow subscriber is discarded.
- `drop_newest` — the incoming message is discarded for a slow subscriber.

The runtime uses `block` by default for `market:events:*` and `drop_oldest` for raw
topics; this is enforced in `MarketDataPlane.build` (see `mm/runtime/engine.py`).

---

# 03 — Frame assembly: from `MarketEvent`s to a `JointStrategyFrame`

The strategy does not subscribe to the bus directly. There is a dedicated *frame
assembler* per actor that:

1. Pulls local and (optionally) global `MarketEvent`s from the bus.
2. Pulls the latest `PrivateStateSnapshot`s (balances, open orders, fills) for each
   venue.
3. Aligns them in time (so the local and global windows close together, with a
   configurable grace).
4. Combines them with the actor's portfolio state (cash, inventory, ledger
   snapshot) to produce a `JointStrategyFrame`.

This section is the *seam* between ingestion and the strategy.

## 3.1 `StrategyInputFrameAssembler`

`mm/application/frame_assembler.py` holds the simpler per-actor assembler. It owns
three queues:

- `local_queue: asyncio.Queue[MarketEvent]`
- `global_queue: asyncio.Queue[MarketEvent] | None` (when a global venue exists)
- `local_private_queue: asyncio.Queue[PrivateStateSnapshot] | None`
- `global_private_queue: asyncio.Queue[PrivateStateSnapshot] | None`

The "local" role is the venue the actor is quoting; the "global" role is an
informational reference venue used for cross-venue features only.

```19:48:mm/application/frame_assembler.py
class StrategyInputFrameAssembler:
    def __init__(self, *, local_queue, global_queue=None,
                 local_private_queue=None, global_private_queue=None,
                 bus=None, output_topic=None,
                 max_global_lag_ms=2_000, global_grace_ms=250,
                 global_history_size=128, clock_ms=now_ms):
        ...
        self._global_events: deque[MarketEvent] = deque(maxlen=...)
        self._local_private: PrivateStateSnapshot | None = None
        self._global_private: PrivateStateSnapshot | None = None
```

The loop is started with `await assembler.start()`. The publish loop is `_run_local`:

```92:103:mm/application/frame_assembler.py
async def _run_local(self) -> None:
    while not self._stopped.is_set():
        frame = await self.next_frame()
        if self.bus is not None and self.output_topic:
            await self.bus.publish(self.output_topic, frame)
```

So the bus sees the *assembled* `StrategyInputFrame` on the `output_topic` configured
in runtime settings (e.g. `frames:default:local`).

### 3.1.1 Global alignment

`build_frame` waits up to `global_grace_ms` for a global event whose `t_end_ms` matches
the local `t_end_ms`:

```75:99:mm/application/frame_assembler.py
async def build_frame(self, local: MarketEvent) -> StrategyInputFrame:
    self._drain_latest_inputs()
    if self.global_queue is not None:
        await self._wait_for_matching_global(local.t_end_ms)
    self._drain_latest_inputs()
    global_event, global_status, global_lag_ms = self._select_global(local)
    return StrategyInputFrame(
        local=local, global_=global_event,
        local_private=self._local_private,
        global_private=self._global_private,
        global_status=global_status,
        global_lag_ms=global_lag_ms,
        max_global_lag_ms=self.max_global_lag_ms,
        created_at_ms=self.clock_ms(),
    )
```

`_select_global` returns a tuple `(event, status, lag_ms)` with one of four
`GlobalMarketStatus` values:

| Status | Meaning | Strategy reaction |
| --- | --- | --- |
| `disabled` | No global venue configured | Local-only MPC; `global_=None`. |
| `missing` | No global event in the history ring | Treat as `disabled` for MPC features, mark frame. |
| `stale` | Latest global is older than `max_global_lag_ms` | Mark frame, set `global_lag_ms`. |
| `ready` | Global event available within lag budget | Use as `global_`. |

The lag budget is enforced even when a global event is "available": any candidate
older than `local.t_end_ms − max_global_lag_ms` is downgraded to `stale`.

### 3.1.2 Why this matters

The MPC optimizer cares about the *spread* and *imbalance* of the global venue, and
about the *reference price* (often the global mid). If the global feed is stale, the
optimizer is told so via `frame.global_status` and `frame.global_lag_ms` so it can
choose a wider spread, switch to the local mid as a reference, or fall back to a
defensive policy.

## 3.2 `JointFrameAssembler`

`mm/application/joint_frame_assembler.py` is the per-actor object that subscribes to
the *assembled* `StrategyInputFrame` topic and combines it with the *joint* ledger
snapshot. The joint ledger snapshot is the actor's combined view across both venues
(cash, inventory, weighted reserved balances, allocation breaches). The frame becomes a
`JointStrategyFrame`.

```1:80:mm/application/joint_frame_assembler.py
class JointFrameAssembler:
    def __init__(self, *, bus, input_topic, joint_ledger,
                 output_topic, max_frame_age_ms=2000,
                 max_ledger_age_ms=2000, clock_ms=now_ms):
        self.bus = bus
        self.input_topic = input_topic
        self.joint_ledger = joint_ledger
        self.output_topic = output_topic
        ...
```

The freshness checks are *guard rails* for the strategy. A frame is dropped (and
counted) if:

- the market event is older than `max_frame_age_ms`; or
- the joint ledger snapshot is older than `max_ledger_age_ms`; or
- the ledger is "ahead" of the market by a noticeable amount (i.e. it contains
  fills whose origin time is in the future of `frame.local.t_end_ms`), which usually
  indicates a clock skew issue between the private feed and the public feed.

```70:99:mm/application/joint_frame_assembler.py
def _ledger_ahead_of_market_ms(self, frame, ledger):
    return (ledger.latest_fill_ts_ms or 0) - frame.local.t_end_ms
```

If everything is fresh, `build_joint_frame` is called:

```100:140:mm/application/joint_frame_assembler.py
def build_joint_frame(self, input_frame, joint_ledger):
    local_open = self._open_orders(input_frame.local_private)
    global_open = self._open_orders(input_frame.global_private)
    return JointStrategyFrame(
        input=input_frame,
        joint_ledger=joint_ledger,
        local_open_orders=local_open,
        global_open_orders=global_open,
        joint_open_orders={**local_open, **global_open},
    )
```

The `joint_open_orders` map is what the strategy runner uses to subtract its own
quotes from the local and global books (see [04-strategy-runner.md](04-strategy-runner.md)).

## 3.3 What "joint" really means

`JointStrategyFrame` is a struct that bundles the *input frame* (public market data)
with the *joint ledger* (combined venue state) and the *open orders* (currently
resting). Everything that the MPC needs to make a decision is reachable from this one
object. The frame is serialized and published to the strategy runner on a per-actor
topic such as `frames:default:joint`.

The freshness guard is the only place where the runtime can decide to **not** act on
new information. If the freshness budget is exhausted, the strategy runs *one more*
decision with the stale frame, then the next frame is dropped until both the public
and private feeds recover. This avoids a "thundering herd" effect after a network
blip where the runner would otherwise queue up many partial decisions.

---

# 04 — Strategy runner: the per-decision loop

`mm/application/strategy_runner.py` is the **per-actor decision loop**. It subscribes
to the per-actor `JointStrategyFrame` topic, runs the MPC optimizer in a worker
thread, validates the result through the `NetDeltaRiskEngine`, and routes the
approved target book to the order manager.

## 4.1 High-level control flow

For each `JointStrategyFrame`:

1. Read the current open orders from the ledger (`ledger.snapshot_open_orders()`).
2. **Clean** the local and global books by subtracting the bot's own quotes
   (so the optimizer does not see itself in the depth).
3. Call `MpcStrategyEngine.decide(clean_frame, ledger)` on a worker thread with a
   timeout (`optimizer_timeout_ms`).
4. Run `NetDeltaRiskEngine.validate_frame(decision, ledger)` to filter the
   `JointTargetBook` to the *approved* orders.
5. Publish:
   - the `StrategyDecision` to `decisions:<actor>` (always);
   - the `JointTargetBook` to the bus;
   - the `ControllerDiagnostics` payload to `diagnostics:controller:<actor>`.
6. If `execution_gate` is open, route the approved book to the order manager;
   otherwise route to the preview bus.

## 4.2 Subtracting own orders: `clean_strategy_frame`

The MPC needs the *market* depth, not the depth it has just posted. Otherwise the
optimizer would "compete with itself". The cleanup is performed by
`mm/application/frame_preparation.py:clean_strategy_frame`:

```15:42:mm/application/frame_preparation.py
def clean_strategy_frame(frame, open_orders):
    clean_local = _replace_event_orderbook(
        frame.local,
        subtract_own_orders(frame.local.orderbook, open_orders),
    )
    clean_global = None
    if frame.global_ is not None:
        clean_global = _replace_event_orderbook(
            frame.global_.orderbook,
            subtract_own_orders(frame.global_.orderbook, open_orders),
        )
    return StrategyInputFrame(...)
```

`subtract_own_orders` walks each level and subtracts the bot's resting qty from the
opposite side; the book VWAP aggregates are recomputed on the cleaned book
(`book_vwaps` in `mm/aggregation/features.py`).

## 4.3 The decision coroutine

```73:96:mm/application/strategy_runner.py
async def run_step(self, joint_frame: JointStrategyFrame) -> None:
    open_orders = self.ledger.snapshot_open_orders(actor=self.actor)
    clean_frame = clean_strategy_frame(joint_frame.input, open_orders)
    decision = await self._run_optimizer(clean_frame, joint_frame.joint_ledger)
    if decision is None:
        return
    target_book = decision.target_book
    risk_result = self.risk_engine.validate_frame(target_book, joint_frame.joint_ledger)
    approved = risk_result.approved_target_book
    self._publish_decision(decision, risk_result)
    await self._dispatch(approved, joint_frame)
```

The four side effects of every step are:

- a `StrategyDecision` published to the decisions topic (always);
- a `ControllerDiagnostics` payload published with optimizer stats and validity/viability;
- a `JointTargetBook` published to the execution plane (only when the execution gate
  is open);
- ledger updates from the eventual fills, which the runner does not perform itself
  but which the reconciler does on the next private snapshot.

## 4.4 Optimizer timeouts & cooperative cancellation

The optimizer is CPU-bound and runs in a dedicated `ThreadPoolExecutor` so the
asyncio event loop stays responsive. The runner uses a `threading.Event` for
cooperative cancellation:

```98:135:mm/application/strategy_runner.py
def __init__(self, ..., optimizer_timeout_ms=2_000, max_optimizer_workers=1):
    self._executor = ThreadPoolExecutor(max_workers=max_optimizer_workers)
    self._cancel_event = threading.Event()

async def _run_optimizer(self, frame, ledger):
    loop = asyncio.get_running_loop()
    self._cancel_event.clear()
    future = loop.run_in_executor(
        self._executor, self._decide_blocking, frame, ledger, self._cancel_event,
    )
    try:
        return await asyncio.wait_for(future, timeout=self.optimizer_timeout_ms / 1000)
    except asyncio.TimeoutError:
        self._cancel_event.set()     # ask the optimizer to stop at the next guard
        return None                  # runner skips this decision cycle
```

The optimizer checks the event at the end of every Adam epoch (see
[07-mpc-optimizer.md](07-mpc-optimizer.md)). A timed-out decision is recorded in
diagnostics as `"optimizer_status": "cancelled"` and the runner falls back to
"keep current orders" — it does **not** cancel live orders, because the previous
target book is still on the venue.

## 4.5 Execution gates and warmup

`compute_venue_execution_gates` decides whether the local and global venues are
allowed to send live orders. The result is a `VenueExecutionGate` per venue:

```10:30:mm/application/venue_execution.py
@dataclass(frozen=True)
class VenueExecutionGate:
    can_sync_live: bool
    reasons: tuple[str, ...] = ()
    dry_run: bool = False
```

Reasons include:

- `dry_run` — runtime configured to never send live orders.
- `warmup` — process has been alive for less than `actor.warmup_seconds`.
- `actor_paused` — operator hit the kill switch.
- `actor_allocation_breaker` — `NetDeltaRiskEngine` flagged an allocation breach.
- `venue_paused` — per-venue pause flag.
- `no_decision` — the previous step returned no decision (timeout, error).
- `target_book_empty` — the optimizer returned an empty book on purpose (e.g.
  negative-objective policy chose `zero_action`).

If the gate is closed, the runner publishes the target book to the *preview* bus
and emits a `ControllerDiagnostics` entry with the reason. No orders are sent.

## 4.6 What happens if no frame arrives

The runner does not time itself out; it only acts when a frame arrives. The
`MarketDataPlane` is responsible for emitting frames; if the symbol engine stalls
(adapter down), the next frame will be older than `max_frame_age_ms` and the
`JointFrameAssembler` will drop it, which causes the runner to keep its last
target book on the venue until fresh data arrives.

---

# 05 — MPC state and action: the canonical data structures

This section defines the two objects that everything in the optimizer reads and
writes: the `State` (input) and the `SigmoidMixtureAction` (output). Both have
two implementations:

- a **NumPy-friendly** form in `mm/mpc/mpc_state.py` (used by the strategy layer
  for diagnostics and warm-start),
- a **PyTorch tensor** form in `mm/mpc/state_tensor.py` (used inside the optimizer
  so that everything is differentiable through `torch.autograd`).

The NumPy and tensor versions are kept in sync by `state_from_*` and
`action_from_*` helpers; the optimizer never crosses the boundary inside a hot
path.

## 5.1 Notation used in this section

- `mid` — current mid price.
- `ob_rel` — the orderbook in *relative* coordinates (distance from mid, signed).
- `inventory` — net base position, positive = long.
- `cash` — quote balance.
- `V₊, V₋` — buy and sell market volumes in the last window.
- `vwap` — volume-weighted average price of the trades.
- `book_buy_vwap, book_sell_vwap, book_vwap_depth` — depth aggregates of the
  resting book.
- `L, k, x0` — sigmoid parameters (see §5.4).
- `n_components` — number of sigmoids per side (the "mixture size").
- `reservation_r_offset` — small spread shift used to incentivize inventory
  rebalancing.

## 5.2 `OrderbookRelative`

A relative orderbook is a 2-tuple `(s, q)` of signed `np.ndarray` representing
*spread* (price − mid) and *quantity* per level. Both sides are stacked: levels
0..N-1 are bids (negative spread), levels N..2N-1 are asks (positive spread).

```1:60:mm/mpc/mpc_state.py
@dataclass(slots=True)
class OrderbookRelative:
    spread: np.ndarray       # shape (2N,), signed
    quantity: np.ndarray     # shape (2N,), non-negative
    @property
    def bid_mask(self) -> np.ndarray: ...
    @property
    def ask_mask(self) -> np.ndarray: ...
```

Conversion to relative is done by `ob_rel_from_market_event` in
`mm/mpc/joint_mpc.py:ob_rel_from_market_event`, which subtracts the local mid
from every level price.

## 5.3 `State`

The MPC state is everything the optimizer needs to evaluate the objective on a
single decision. It is the *output* of `MpcStrategyEngine._build_state` and the
*input* of `MonteCarloMPCTensorBatched.optimize`.

```90:160:mm/mpc/mpc_state.py
@dataclass(slots=True)
class State:
    mid: float
    ob_rel: OrderbookRelative
    buy_vol: float
    sell_vol: float
    inventory: float
    cash: float
    vwap: float
    book_buy_vwap: float
    book_sell_vwap: float
    book_vwap_depth: float
    inventory_position: float              # notional
    reference_price: float                 # usually the global mid
    reservation_r_offset: float = 0.0
    global_ob_rel: OrderbookRelative | None = None
    global_mid: float | None = None
    global_status: str = "disabled"        # "ready" / "missing" / "stale"
```

The state also carries the *reference price* used for inventory marking and PnL
attribution (typically the global mid if available, else the local mid). The
`reservation_r_offset` is a small spread shift driven by inventory and the
venue-specific `reservation_skew` config (see
`mm/mpc/objective_terms.py:VenueOptimizerParams.reservation_skew`).

## 5.4 `SigmoidMixtureAction` — the heart of the parameterization

The action is *not* a list of orders. It is a **sum of `n_components` sigmoids
per side**, where each sigmoid is a smooth step from 0 to `L` as the spread
increases, with steepness `k` and center `x0`:

```1:60:mm/mpc/mpc_state.py
@dataclass(slots=True)
class SigmoidMixtureAction:
    L_bid: np.ndarray        # shape (n_components,), cumulative depth caps
    k_bid: np.ndarray        # shape (n_components,), steepness
    x0_bid: np.ndarray       # shape (n_components,), center
    L_ask: np.ndarray
    k_ask: np.ndarray
    x0_ask: np.ndarray
    reservation_r_offset: float = 0.0
```

Cumulative depth on the bid side at a given spread `s ≥ 0` is:

```
D_bid(s) = Σ_{i=1..K} L_bid[i] · σ(k_bid[i] · (s − x0_bid[i]))
```

with `σ(z) = 1 / (1 + exp(−z))`. The ask side is the mirror with `s ≤ 0`
(matched by negating `s`). The `reservation_r_offset` shifts the center
locations so that the curve is evaluated at `s − r` (or `s + r` for asks) — this
is how inventory pressure is implemented.

### 5.4.1 Why a mixture of sigmoids?

A single sigmoid cannot express the empirically observed shape of a real order
book (flat near the inside, steep far away). A sum of `K` sigmoids with
monotonically increasing centers gives a *convex, monotone* depth curve that
matches the empirical depth well with only a handful of parameters per side
(typically K = 3..6).

### 5.4.2 Effective action

`EffectiveSigmoidMixtureAction` is a thin view of `SigmoidMixtureAction` that
*also* returns the executable depth `D(s) − D(0)`. The MPC optimizer works in
this "effective" space so that the inside level starts at zero additional
quantity (the optimizer never posts inside the best bid/ask) unless
`reservation_r_offset` is non-zero.

### 5.4.3 `Action` and `JointAction`

`Action` is a *single-level* representation used for diagnostics (one price,
one quantity, one side). The optimizer produces a `SigmoidMixtureAction`, and
`Action` is reconstructed by the target book builder when it lays the ladder
out level-by-level.

## 5.5 L-parameter decoding: turning raw parameters into `(L, k, x0)`

The optimizer never touches `(L, k, x0)` directly. It works in a *raw* parameter
space of length `3 * n_components` per side, and the structured decoder in
`mm/mpc/l_param.py` maps the raw vector to the effective `(L, k, x0)`:

```110:160:mm/mpc/l_param.py
@dataclass(frozen=True, slots=True)
class LParamConfig:
    n_components: int
    quantity_floor: float = 0.0
    quantity_ceiling: float | None = None
    steepness_min: float = 4.0
    steepness_max: float = 60.0
    center_floor_ticks: float = 1.0
    center_ceiling_ticks: float | None = None
    tick_size: float = 0.01
    ...
```

`materialize_side_torch` does the decoding inside the optimizer:

- **L** is `sigmoid(raw_L) · (ceiling − floor) + floor`.
- **k** is `sigmoid(raw_k) · (steepness_max − steepness_min) + steepness_min`.
- **x0** is enforced monotone via cumulative softmax: each center is sorted in
  increasing order so the resulting depth curve is convex.

A few extra rules:

- `center_floor_ticks` is the minimum number of ticks from the mid for the
  *innermost* component — keeps the optimizer from crossing the inside.
- `cumulative_qty_floor_at_spread_max` and `cumulative_qty_cap_at_spread_max` are
  hard bounds on the depth at the maximum allowed spread.
- `min_spread_ticks` and `max_spread_ticks` bound the curve's range in tick
  units.

The output is a differentiable `ActionTensor` that the rest of the optimizer
uses directly.

## 5.6 Joint-venue extension: `JointSigmoidMixtureAction`

When the strategy runs in *joint* mode (one local venue + one global venue), the
action is the concatenation of two `SigmoidMixtureAction`s, one per venue:

```60:90:mm/mpc/joint_mpc.py
@dataclass(slots=True)
class JointSigmoidMixtureAction:
    local: SigmoidMixtureAction
    global_: SigmoidMixtureAction
```

The total parameter length is `joint_action_param_len(n_local, n_global)` which
is `2 * 3 * (n_local + n_global)`. The optimizer slices the flat vector into
local and global halves and optimizes them jointly because the consolidated
wealth objective depends on both.

## 5.7 State extension for joint venues

`enrich_state_for_joint` in `mm/mpc/joint_mpc.py` adds joint-venue fields to the
state:

- `global_ob_rel` and `global_mid` — global venue market data.
- `global_status` — `"ready"`, `"missing"`, `"stale"`, or `"disabled"`.
- `reference_price` — chosen by `reference_price_from_state` (usually the global
  mid if `ready`, else the local mid).

These fields are used by the consolidated wealth function and by the joint
optimization backend.

## 5.8 What gets serialized and what doesn't

For warm-starting, the runner stores the **raw flat parameter vector** of the
last decision (not the decoded `(L, k, x0)`). The vector is the same shape
across decisions as long as `n_components` does not change, which it does not
in production. Warm-start is detailed in §11.4.

---

# 06 — Stochastic models: how the MPC imagines the next horizon

The optimizer is **model-predictive**: it simulates `H` future steps under a
*stochastic model* of order flow and price. The model has two pieces:

1. a **price model** — produces `(Δmid, P(next event))` per step;
2. a **order-flow model** — produces `(V₊, V₋)` per step.

Both are pluggable through the `PriceModel` and `OrderFlowModel` protocols. The
runtime always uses the same *family* of models for the *real* data and the
*simulated* data, so the optimizer is consistent with what the aggregation
pipeline produces.

## 6.1 Notation

- `H` — horizon (number of decision steps simulated into the future).
- `B` — Monte Carlo batch size (number of paths).
- `V₊, V₋` — buy and sell market volumes over a single step.
- `Δmid` — change in mid over a single step.
- `vwap, vwap_buy, vwap_sell` — volume-weighted average prices for the step.

## 6.2 Price models (`mm/mpc/price_torch.py`)

### 6.2.1 `PriceModel` protocol

```1:40:mm/mpc/price_torch.py
class PriceModel(Protocol):
    def sample(self, state: StateTensor, horizon: int) -> tuple[Tensor, Tensor]:
        """Return (Δmid of shape (B, H), log-prob of shape (B, H))."""
    def log_prob(self, realized: Tensor) -> Tensor: ...
    def update(self, realized_step) -> None: ...
```

The optimizer needs both `sample` (used in simulation) and `log_prob` (used in
importance-sampling corrections and diagnostics). `update` is called on the
realized step after each interval closes.

### 6.2.2 `GaussianPriceModel`

The simplest model: `Δmid ∼ 𝒩(0, σ²)` with `σ` learned from realized mid
changes via EWMA. Used in unit tests and as a fallback.

### 6.2.3 `TorchResidualPriceModel`

A two-stage model:

1. Use a *mechanical* mixture model (fixed-depth mid/spread/shape deltas driven
   by `V₊, V₋` and the resting book).
2. Learn a residual distribution around the mechanical prediction via EWMA.

The residual is Gaussian with a learned variance. The implementation keeps a
running mean and variance for the residual, updated on each realized step.

### 6.2.4 `TorchConditionalMechanicalMixturePriceModel`

The full production model. It has:

- fixed-depth mid/spread/shape deltas learned per `(V₊, V₋, imbalance, book
  shape)` bucket;
- a residual model that can be either Gaussian or Student-t (configurable);
- an RLS (recursive least squares) update for the bucket weights so the model
  adapts online without storing history.

`sample` returns `Δmid` per Monte Carlo path; `log_prob` is used by the
optimizer to score the realized path against the simulated paths.

## 6.3 Order-flow models (`mm/mpc/market_order_flow_torch.py`)

### 6.3.1 `OrderFlowModel` protocol

```1:35:mm/mpc/market_order_flow_torch.py
class OrderFlowModel(Protocol):
    def sample(self, state: StateTensor, horizon: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (V₊, V₋, log-prob) of shape (B, H) each."""
    def log_prob(self, realized_step) -> Tensor: ...
    def update(self, realized_step) -> None: ...
```

The order-flow model produces a *joint* sample of buy and sell volumes for
each simulated step.

### 6.3.2 Hurdle models

The order-flow distribution is *hurdle*: a mass at zero and a continuous
positive distribution. The hurdle log-likelihood is

```
log p(V) = log p0  if V = 0
        = log(1 − p0) + log p⁺(V)   otherwise
```

Implementations:

- `TorchHurdleLogNormalModel` — `log V | V > 0 ∼ 𝒩(μ, σ²)`.
- `TorchHurdleLogStudentTModel` — heavy-tailed alternative.
- `TorchHurdleGammaModel` — `V | V > 0 ∼ Gamma(α, β)`.
- `TorchHurdleWeibullModel` — `V | V > 0 ∼ Weibull(k, λ)`.
- `TorchHurdleGeneralizedParetoModel` — `V | V > 0 − u ∼ GPD(ξ, σ)`.

Each model learns its parameters from the realized `V₊` and `V₋` via EWMA on
the positive part and a separate EWMA on the zero-mass probability.

### 6.3.3 Cross-correlation

The two sides are not independent. Each `sample` call draws `V₊` and `V₋` from
a *bivariate* distribution; the implementation factorizes as
`p(V₊, V₋) = p(V₊) · p(V₋ | V₊)` with a learned conditional that captures the
directional pressure visible in `imbalance`.

---

# 07 — The MPC optimizer: Monte Carlo over a finite horizon

The optimizer lives in `mm/mpc/mpc_tensor_batched.py`. It is a *batch Monte Carlo*
optimizer: for every Adam epoch it draws `B` paths of length `H`, simulates each
path under the action, evaluates the terminal objective, and backpropagates
through everything to update the action parameters.

The same machinery is wrapped by the joint-venue variant in
`mm/mpc/joint_optimizer.py` which optimizes a single flat vector that contains
both local and global `SigmoidMixtureAction` parameters.

## 7.1 Notation

- `B` — Monte Carlo batch size.
- `H` — horizon (number of future steps).
- `K` — number of sigmoid components per side.
- `n_raw_local` = `3 * K_local` raw parameters per side for the local venue.
- `n_raw_global` = `3 * K_global` raw parameters per side for the global venue.
- `N_total` = total raw parameter count.
- `E` — number of Adam epochs per decision.
- `α` — Adam learning rate.
- `clip_norm` — global L2 gradient norm clip.

## 7.2 Top-level call

`MonteCarloMPCTensorBatched.optimize(state, n_samples, horizon, adam_epochs,
learning_rate, warm_start_action, cancel_event)` returns an
`OptimizerResult`:

- `action` — the final `SigmoidMixtureAction`;
- `best_objective` — best terminal objective seen across epochs;
- `best_risk_adjusted_return` — separated for the negative-objective policy;
- `objective_history` — per-epoch objective trace;
- `gradient_norm_history` — per-epoch gradient norm trace;
- `cancel_reason` — `None`, `"cancelled"`, or `"timeout"`.

## 7.3 The closure: one Adam epoch

Inside the optimizer loop, every epoch calls a `closure` that:

1. Decodes the current raw parameters into `(L, k, x0)` tensors.
2. Simulates the market over the horizon under the action.
3. Computes the terminal objective and the auxiliary penalties.
4. Backpropagates and lets Adam update the parameters.

## 7.4 Market simulation per step

For each step `t` of the horizon, the optimizer:

1. Draws `V₊, V₋` from the order-flow model and `Δmid` from the price model.
2. Computes the **soft fill** for the bot's quotes: at each tick distance `s`,
   the cumulative depth is `D(s)`; market orders at a given side drain the
   bot's queue at that side.
3. Updates the bot's inventory and cash:

   ```
   X_t   = X_{t−1} + fill_bid_t − fill_ask_t
   cash_t = cash_{t−1} + fill_ask_t · (mid_t − spread_ask/2)
                     − fill_bid_t · (mid_t − spread_bid/2)
   ```

4. Optionally updates the resting book (top-N) using the realized trades.

`soft_fill_batched` is the heart of the simulation: it is a *differentiable*
function of the action parameters and the sampled `V₊, V₋`. The fill at
distance `s` is `min(D(s) − D(0), V_side)` (a smooth min is used inside the
optimizer so the gradient is well-defined).

## 7.5 Terminal objective

The terminal objective combines:

- risk-adjusted return (certainty equivalent),
- inventory penalty,
- slippage penalty,
- validity penalty (book quality),
- viability penalty (expected availability shortfall).

`terminal_objective(state_traj, action_traj)` in `mm/mpc/reward_tensor.py`
computes the full scalar `J`. The sign convention is **higher J is better** —
the optimizer *minimizes* `-J`.

## 7.6 Viability & validity

Validity and viability are described in detail in [10-risk.md](10-risk.md). In
short:

- `validity_penalty = μ · max(0, B_min − book_quality_score)`
- `viability_penalty = λ · E[d_bid² + d_ask²]` (next-step quadratic deficit)

The optimizer adds both to the terminal objective. The `viability` dual
variable `λ` is updated externally in `update_viability_lambda` using a
bounded linear step on an EWMA of the deficit.

## 7.7 Cancellation and timeout

The optimizer checks the `cancel_event` at the end of every Adam epoch. The
runner sets the event when the wall-clock budget (`optimizer_timeout_ms`) is
exceeded. The optimizer then:

- sets `cancel_reason = "cancelled"`,
- returns the action at the *last* completed epoch (not the partial epoch), and
- the runner reports `"optimizer_status": "cancelled"` in the diagnostics.

The action returned by a cancelled optimization is always *finite* (see
[11-edge-cases.md](11-edge-cases.md)) because the warm-start re-validates the
raw vector and the decoder rejects NaN/Inf values during construction.

## 7.8 Warm-start

The optimizer accepts a `warm_start_action: np.ndarray | None` of shape
`(N_total,)` containing the raw flat parameter vector of the *previous*
decision. The runner stores this vector at the end of every decision (see
[11-edge-cases.md](11-edge-cases.md) for the persistence details) and feeds it
back in for the next one.

The blend is controlled by `WarmStartParams`:

- `enabled=false` (or `alpha=0`) → cold start (raw vector is the default
  initializer, computed by `default_initial_raw` from the current mid and
  config bounds).
- `alpha=1.0` → use the previous vector verbatim (legacy binary warm start).
- `alpha_mode="adaptive"` → `α = min(α_max, ‖∇u_warm‖ / (‖∇u_warm‖ + ‖∇u_default‖ + ε))`,
  i.e. start near the previous vector when its gradient is small relative to
  the default's, else start near the default.

`prepare_warm_start_initial_flat` tiles the per-step blended vector across the
horizon so the optimizer sees a stationary plan to start from.

`preserve_optimizer_state=true` keeps Adam's first and second moments across
decisions, which gives a substantial speed-up in practice (the optimizer
converges in 3-5 epochs when warm-started vs. 15-25 from cold).

## 7.9 Best-objective tracking

The optimizer keeps a running `best_objective` and the action that produced it.
At the end of `E` epochs it returns `best_action`, not the action from the last
epoch. This is robust to Adam overshooting in the last few epochs — common
when the gradients are very large early in training.

## 7.10 Joint-venue optimizer

`optimize_joint_with_diagnostics` in `mm/mpc/joint_optimizer.py` is the entry
point for joint mode. It:

1. Builds a joint state with `enrich_state_for_joint`.
2. Slices the warm-start vector into local and global halves.
3. Builds a single `JointSigmoidMixtureAction` whose parameters span both
   venues.
4. Runs the same Adam loop but with the joint reward
   `joint_terminal_objective(state, joint_action)`.
5. Returns the same `OptimizerResult`, with the action split back into local
   and global components for the target book builder.

The joint objective includes a *consolidated wealth* term so the optimizer
trades off between the two venues' PnL streams (see
[08-objective.md](08-objective.md)).

---

# 08 — Reward and objective terms

The MPC objective is the sum of:

- the **risk-adjusted return** (certainty equivalent) from the simulated PnL,
- an **inventory penalty** that pushes the bot to flat,
- a **slippage penalty** that discourages quoting too far from the inside,
- a **validity penalty** that penalizes structurally bad books,
- a **viability penalty** that penalizes actions that would over-spend
  inventory or margin,
- an optional **objective normalization** that scales the risk-adjusted return
  by a learned or configured factor.

## 8.1 Notation

- `W_t = cash_t + X_t · mid_t` — wealth at step `t`.
- `ΔW_t = W_t − W_{t−1}` — change in wealth.
- `X_t` — net base inventory (signed: positive = long).
- `Q_t` — queue position of a quote (for queue-aware fill model).
- `τ` — book quality target.
- `λ_v, λ_θ` — weights for the validity and viability penalties.
- `β` — risk aversion coefficient (return_risk_aversion_beta).

## 8.2 PnL decomposition: `delta_pnl_detailed_tensor`

The per-step PnL is split into four terms that are useful for diagnostics:

```60:120:mm/mpc/reward_tensor.py
def delta_pnl_detailed_tensor(state_t, state_tm1, action_t, ...):
    dX = state_t.inventory - state_tm1.inventory
    dP = state_t.mid - state_tm1.mid
    dQ = dX                       # inventory change == net fills
    return DeltaPnL(
        cash_change   = (state_t.cash - state_tm1.cash),
        inventory_pnl = state_tm1.inventory * dP,
        new_position  = dQ * state_tm1.mid,
        cross_term    = dQ * dP,
    )
```

So `ΔW = Δcash + X_{t−1}·ΔP + ΔQ·P_{t−1} + ΔQ·ΔP`.

## 8.3 Inventory penalty: `inventory_pnl_term_tensor`

```
inv_pen = λ_inv · X_T · mid_T · sign(X_T)
```

A simple *soft* penalty on terminal inventory, evaluated at the end of the
horizon. The weight `λ_inv` is set in `VenueOptimizerParams.inventory_penalty`.

## 8.4 Slippage penalty: `slippage_pnl_term_tensor`

```
slip_pen = λ_slip · E_path[ Σ_t  ∂D_ask / ∂s · (s_t − mid) ]
```

A penalty on quoting too far from the inside, measured as the area under the
depth curve times the spread. The marginal depth `∂D/∂s` is computed inside
the optimizer by `depth_*_derivative_tensor` in `mm/mpc/state_tensor.py`.

## 8.5 Risk-adjusted return: `objective_entropic_return`

The certainty-equivalent (negative-entropy) return is

```
CE = (1/β) · log E_path[ exp(β · ΔW_T) ]
```

with `β` = `return_risk_aversion_beta` (per-venue). The convention in the code
is `mean − β/2 · var` (Cox-Lin-Hanson expansion to second order) which matches
the exponential CE to leading order and is more numerically stable.

The sign convention inside the optimizer is `J += ce` (we *maximize* the
certainty equivalent) and `J -=` the other penalties. Adam minimizes `-J`.

## 8.6 Validity: `book_quality_score` and `combine_validity_term`

`book_quality_score` evaluates the *structure* of the proposed book:

- **tightness** — average distance from the mid, weighted by quantity;
- **presence** — total quantity inside a reference band around the mid;
- **balance** — how symmetric the bid and ask sides are;
- **uniformity** — how smooth the depth curve is (penalizes spikes).

```60:100:mm/mpc/objective_terms.py
def book_quality_score(action, ob_rel, params):
    tightness = _tightness_score(action, ob_rel, params)
    presence  = _presence_score(action, ob_rel, params)
    balance   = _balance_score(action)
    uniformity = _uniformity_score(action)
    return (
        params.v_tightness * tightness
        + params.v_presence  * presence
        + params.v_balance   * balance
        + params.v_uniformity * uniformity
    )
```

`combine_validity_term` adds `λ_v · max(0, τ − score)²` to the objective.

## 8.7 Viability: `viability_path_shortfall` and `combine_viability_term`

`wallet_quadratic_deficit` measures a two-sided squared-hinge reserve
deficit on **total** balances. It is zero above `q_min` and steepens as a
side falls further below the floor:

```
d_bid = relu(q_min - cash) / q_min
d_ask = relu(q_min - inventory * mid) / q_min
S     = d_bid² + d_ask²
```

The MPC applies `−λ · E[S_1]` (next-step only). `update_viability_lambda`
then steps a bounded linear multiplier from an EWMA of `S`, so a noisy
Monte Carlo batch cannot explode `λ`.

---

# 09 — From sigmoid action to a venue-ready `JointTargetBook`

The MPC produces a *continuous* `SigmoidMixtureAction`. The exchange does not
accept curves — it accepts a list of `(price, quantity)` orders. The conversion
happens in two stages:

1. `build_orders_from_sigmoid` lays the depth curve on a discrete price grid
   and produces a ladder of `Order`s.
2. `build_joint_target_book_from_sigmoid` combines the local and global ladders
   (when in joint mode) into a `JointTargetBook` and a pipeline report.

The discretization must respect:

- the venue's `price_step` (tick size) and `qty_step` (lot size);
- the venue's `min_qty`, `min_notional`, and `max_orders_per_side`;
- the optimizer's structural limits (`max_spread_ticks`, `cumulative_qty_floor_at_spread_max`,
  `cumulative_qty_cap_at_spread_max`).

## 9.1 Notation

- `Δ_spread` — venue's price step (tick).
- `Δ_qty` — venue's quantity step (lot).
- `Q_min` — minimum order quantity.
- `N_min` — minimum order notional.
- `K` — maximum number of orders per side.
- `s_max` — maximum allowed spread (ticks from mid).
- `D(s)` — cumulative depth at spread `s` from the sigmoid action.

## 9.2 Laying the ladder

`ladder_orders_from_sigmoid_with_report` enumerates the price grid from
`s = price_step` (one tick from the mid) up to `s_max`, computes
`q(s) = D(s) − D(s − Δ_spread)` for each level, and rounds to the lot size.
Levels whose cumulative quantity is below `Q_min` or notional below `N_min`
are dropped.

```90:140:mm/core/order_constraints.py
def ladder_orders_from_sigmoid_with_report(action, mid, side, constraints):
    grid = _spread_grid(constraints)
    cumulative = _cumulative_depth(action, side, grid)
    per_level = np.diff(cumulative, prepend=0.0)
    orders = []
    for s, q in zip(grid, per_level):
        if q < constraints.qty_step:
            continue
        q_rounded = _quantize_qty(q, constraints.qty_step)
        if q_rounded < constraints.min_qty:
            continue
        price = mid + s if side == "ask" else mid - s
        if abs(price * q_rounded) < constraints.min_notional:
            continue
        orders.append(Order(price=price, quantity=q_rounded, side=side))
    return orders, LadderReport(...)
```

The function returns a `LadderReport` with diagnostics: the per-tick
cumulative depth (so the operator can plot the curve against the realized
depth) and a list of dropped levels with reasons.

## 9.3 Structural constraints

The decoder (`LParamConfig` in `mm/mpc/l_param.py`) enforces:

- `min_spread_ticks` — every level is at least this far from the mid;
- `max_spread_ticks` — no level is further than this;
- `cumulative_qty_floor_at_spread_max` — at the max spread, total depth
  must be at least this value (a "presence" floor);
- `cumulative_qty_cap_at_spread_max` — at the max spread, total depth
  must be at most this value (a "max exposure" cap).

These are *soft* in the optimizer (the L-param decoder allows them to be
violated slightly) but are *hard* in the target book builder: a level that
exceeds the cap is truncated, and a level that doesn't reach the floor is
filled to it.

## 9.4 `OrderConstraints`

```1:60:mm/core/order_constraints.py
@dataclass(frozen=True, slots=True)
class OrderConstraints:
    price_step: float
    qty_step: float
    min_qty: float
    min_notional: float
    max_orders_per_side: int
    ...
```

These are sourced from the `InstrumentSpec` (built from
`InstrumentPrecisionConfig` in the runtime config). The same constraints are
applied to every order produced by the strategy and to every order received
from the venue, so the runner and the order manager share a single source of
truth.

## 9.5 `apply_venue_constraints_with_report`

After the ladder is built, the report's pipeline applies:

- `apply_venue_constraints_with_report` — rounds price and quantity, drops
  orders that fail the venue rules, and produces a per-order reason list.
- `cap_orders_per_side_with_report` — keeps the `K` deepest orders per side
  and reports the rest as `capped:order_count`.

The final `ActionPipelineReport` carries:

- `n_orders_in` — orders before constraints;
- `n_orders_kept` — orders after constraints;
- `n_orders_capped` — orders dropped by the per-side cap;
- `n_orders_dropped_min_qty`, `n_orders_dropped_min_notional`,
  `n_orders_dropped_price_step` — orders dropped by the venue rules;
- `cumulative_qty_at_spread_max_actual` — actual depth at the cap, useful
  for verifying the floor/cap constraints.

## 9.6 `JointTargetBook`

`build_joint_target_book_from_sigmoid` runs the pipeline for the local venue
and (optionally) the global venue, and assembles the result into a
`JointTargetBook`:

```40:80:mm/strategies/target_book_builder.py
def build_joint_target_book_from_sigmoid(local_action, global_action, ...):
    local_book, local_report = build_venue_target_book(local_action, local_spec, ...)
    if global_action is not None:
        global_book, global_report = build_venue_target_book(global_action, global_spec, ...)
    else:
        global_book, global_report = None, None
    return JointTargetBook(
        local=local_book,
        global_=global_book,
        pipeline_reports=(local_report, global_report),
    ), ActionPipelineReport(combined=...)
```

The `JointTargetBook` is the input to the risk engine. After risk approval it
becomes the input to the order manager (see [04-strategy-runner.md](04-strategy-runner.md)).

---

# 10 — Risk: `NetDeltaRiskEngine`, validity, viability

The risk layer is the **last gate** before orders go to the venue. It has two
parts:

1. **`NetDeltaRiskEngine`** — applies the *venue* and *actor* risk limits
   (max notional, allocation caps, kill switch).
2. **Validity and viability** — the *soft* constraints already in the MPC
   objective. The risk engine is a *hard* backstop: it discards any order
   that would push the venue outside the configured limits, even if the
   optimizer's soft penalties missed it.

## 10.1 Notation

- `P_base, P_quote` — base and quote prices.
- `B, Q` — base and quote balances (free + locked).
- `X` — net base inventory (signed).
- `T_base_locked, T_quote_locked` — base and quote amounts that the target
  orders would lock.
- `R_max_order, R_max_open` — per-order and per-venue notional limits.
- `alloc_local`, `alloc_global` — actor's allocation shares for each venue.

## 10.2 Frame-level gates: `_frame_gate_reasons`

Before the per-venue checks, the engine inspects the `JointStrategyFrame` for
reasons to *skip the entire decision*:

```80:110:mm/risk/net_delta.py
def _frame_gate_reasons(frame, ledger, *, max_frame_age_ms, max_ledger_age_ms):
    reasons = []
    if frame.input.local.t_end_ms + max_frame_age_ms < now_ms():
        reasons.append("stale_market_data")
    if ledger.last_update_ms + max_ledger_age_ms < now_ms():
        reasons.append("stale_ledger")
    if frame.input.global_status == "stale":
        reasons.append("stale_global")
    if frame.input.global_status == "missing":
        reasons.append("missing_global")
    ...
    return reasons
```

If any reason is present, the engine returns an empty `TargetBookRiskResult`
with `approved_target_book=None` and the reason list. The runner then
publishes a "skipped" decision and the venue's open orders are *not* modified.

## 10.3 Per-venue checks: `evaluate_venue_target_transition`

For each venue the engine checks:

- **max order notional** — `price * qty ≤ R_max_order`.
- **max open notional** — `Σ (price * qty) over the venue's open orders
  ≤ R_max_open`.
- **insufficient base / quote** — would the proposed order push the venue
  balance below zero?
- **min cash / inventory** — is the venue allowed to operate with this
  much cash or inventory? (typically a safety floor).
- **max inventory** — is the venue allowed to take this much inventory?
  (typically a safety ceiling).
- **allocation caps** — would the proposed exposure push the actor's
  `alloc_local` (or `alloc_global`) above the configured cap?

```50:90:mm/risk/net_delta.py
def evaluate_venue_target_transition(target_orders, ledger, *, limits):
    base_locked = sum(o.qty for o in target_orders if o.side == "buy")
    quote_locked = sum(o.price * o.qty for o in target_orders if o.side == "buy")
    ...
    if base_locked > ledger.free_base + eps:
        return VenueRiskResult(approved=False, reason="insufficient_base")
    if quote_locked > ledger.free_quote + eps:
        return VenueRiskResult(approved=False, reason="insufficient_quote")
    ...
```

The function returns a `VenueRiskResult` with a list of `accepted` and
`rejected` orders. The engine is *partial-approval capable*: if a venue has
five bids and one of them would over-spend the cap, the other four are
approved and the fifth is rejected with `reason="exceeds_max_notional"`.

## 10.4 Kill switch and allocation breaker

The engine consults the `ControlPlane` for:

- `actor_kill_switch[actor].armed` — if true, every order is rejected with
  `reason="actor_kill_switch"`.
- `actor_allocation_breaker[actor].tripped` — if true, every order is
  rejected with `reason="actor_allocation_breaker"`. The breaker is tripped
  by a *separate* monitor that watches the actor's inventory against a
  configured `max_inventory` (this is in `mm/control/allocation_breaker.py`).

The kill switch is the *emergency stop*; the allocation breaker is the
*graduated* stop. The kill switch is a hard stop with no auto-recovery; the
allocation breaker resets when the actor's exposure returns inside the
configured band.

## 10.5 Validity backstop

The optimizer already penalizes a low `book_quality_score`, but it may still
return a structurally bad book if the soft penalty is small relative to the
PnL gradient. The `NetDeltaRiskEngine` does *not* re-evaluate validity, but
the strategy runner checks `decision.target_book.book_quality_score` against
`min_book_quality_score` (configured per actor). If the score is below the
threshold, the runner drops the book and logs `reason="low_book_quality"`.

## 10.6 Viability backstop

The viability dual variable `λ` is updated *after* every decision by
`update_viability_lambda`. The function EWMA-filters the realized
quadratic deficit and steps a bounded linear multiplier:

```text
S̄ ← α S̄ + (1−α) S
λ ← clip(λ + η (S̄ − ε), 0, λ_max)
```

The updated `λ` is kept on the MPC instance and used in the next decision.

## 10.7 `validate_target_book_with_result`

The top-level entry point composes everything:

```130:170:mm/risk/net_delta.py
def validate_target_book_with_result(target_book, frame, ledger, *, limits, control_plane):
    frame_reasons = _frame_gate_reasons(frame, ledger, ...)
    if frame_reasons:
        return TargetBookRiskResult(approved=False, reasons=frame_reasons)
    local_result = evaluate_venue_target_transition(target_book.local, ledger.local, ...)
    global_result = evaluate_venue_target_transition(target_book.global_, ledger.global_, ...)
    return TargetBookRiskResult(
        approved_target_book=JointTargetBook(
            local=local_result.accepted, global_=global_result.accepted
        ),
        rejected_orders=local_result.rejected + global_result.rejected,
        reasons=local_result.reasons + global_result.reasons,
    )
```

The runner then publishes the decision with both the approved book and the
list of rejections, so the operator can see exactly which orders were
filtered.

---

# 11 — Edge cases and failure handling

This section is dedicated to the *non-happy-path* behavior of the system. The
runtime is designed to be *boring under failure*: it should never panic, and
the worst case is "no orders sent this interval".

## 11.1 Negative objective: the floor / zero policy

If the optimizer's terminal objective is negative (the MPC cannot find a
profitable plan), three policies are configurable:

- `"optimizer_action"` (default) — return the optimizer's plan anyway. The
  validity/viability penalties already discourage truly unprofitable plans,
  so a slightly-negative objective is often still a better response than
  doing nothing.
- `"floor_action"` — replace the action with the *minimum* depth that
  satisfies the validity/viability floors. The bot stays present but does
  not try to be clever.
- `"zero_action"` — return an action with zero depth on both sides. The
  target book is empty and the venue is cleared of resting orders.

`apply_negative_objective_policy` in `mm/strategies/negative_objective_policy.py`
implements the selection. The choice is per-actor and live-configurable.

The `risk_adjusted_return` used by the policy is the *best* value seen
across Adam epochs (not the final value), so the policy never punishes the
optimizer for Adam overshooting in the last epoch.

## 11.2 Finite-checks: `mm/mpc_utils/finite.py`

The optimizer can produce `NaN` or `Inf` values in the raw parameter vector
when:

- the gradient explodes (rare, but possible after a model miss);
- the warm-start vector is corrupt (e.g. a config change between runs that
  changed `n_components`);
- the order-flow model emits a very large sample (a heavy-tailed
  distribution combined with a small batch).

`mm/mpc_utils/finite.py` exposes:

- `is_finite_flat(arr)` — `True` iff every element of `arr` is finite.
- `replace_non_finite(arr, fill=0.0)` — returns a copy with `NaN`/`Inf`
  replaced by `fill`.
- `finite_flat_or_none(arr, expected_len)` — returns the array if it is
  finite and the right length, else `None`. The runner uses this when
  building the warm-start vector: if the previous decision is corrupt, the
  runner falls back to cold start (no warm-start).

The optimizer also clamps the raw parameters to `[-param_clip, param_clip]`
inside the L-param decoder so the Adam state cannot drift into a region where
the sigmoid saturates and the gradient vanishes.

## 11.3 Optimizer cancellation

Already covered in [07-mpc-optimizer.md](07-mpc-optimizer.md). The runner
sets the cancel event when `optimizer_timeout_ms` is exceeded, and the
optimizer exits at the next epoch boundary with `cancel_reason="cancelled"`.
The runner reports this in diagnostics and skips the decision cycle. The
**previous** target book stays on the venue.

## 11.4 Warm-start persistence

The warm-start vector lives in three places:

1. **In the `StrategyRunner`** — as the `_last_raw_flat` attribute, updated at
   the end of every decision.
2. **In the warm-start spy** (`mm/observability/warm_start_spy.py`) — a small
   publisher that writes the raw vector and a hash of it to the bus on every
   decision. The hash is the `decision_hash` exposed in
   `ControllerDiagnostics`.
3. **In the persistence layer** — on graceful shutdown the runner writes the
   raw vector to a per-actor key. On startup the runner reads it and uses it
   for the first decision. This is the only state that survives a process
   restart.

If the persisted vector is missing (first-ever startup) or corrupt (fails
`finite_flat_or_none`), the runner cold-starts.

## 11.5 Stale or missing private state

If the `PrivateStateSnapshot` for a venue is older than `max_ledger_age_ms`,
the `JointFrameAssembler` drops the frame and the runner does not act. The
control plane logs `reason="stale_ledger"`. The strategy keeps its last
target book on the venue; the venue's orders continue to age out or fill
naturally.

## 11.6 Stale global event

If the global event is older than `max_global_lag_ms`, the
`StrategyInputFrameAssembler` sets `global_status="stale"` and
`global_lag_ms`. The strategy engine still runs (it can degrade gracefully by
using the local mid as the reference) but the *consolidated wealth* term in
the joint objective is down-weighted (the optimizer still respects the
global mid but penalizes it when it is stale).

## 11.7 Reconciliation failures

The `Reconciler` is the loop that pulls open orders and fills from the venue
and writes them to the `Ledger`. If the venue rejects an order, the
reconciler marks it as `rejected` and removes it from the ledger. If the
venue fills an order, the reconciler updates the ledger and emits a
`FillDelta` payload on the bus (consumed by the model update loop and the
PnL aggregator).

The reconciler is *idempotent*: re-running it on the same set of events
produces the same ledger state. This is important because the venue's
private feed can deliver the same fill twice, and we must not double-count.

## 11.8 Process restart

On graceful shutdown the runtime:

1. stops the schedulers (no more `MarketEvent`s);
2. cancels the strategy runners (no more decisions);
3. waits for the in-flight optimizers to return or to time out (max
   `optimizer_timeout_ms`);
4. flushes the persistence layer (writes the last warm-start vector and
   diagnostic snapshots);
5. closes the bus connections.

On a *crash* (SIGKILL, OOM) the persistence layer holds the most recent
snapshot. On startup the runners read the snapshot and resume from there.
The optimizer cold-starts (the persisted warm-start vector is loaded if
present, else `None`).

---

# 12 — A sample event traced end-to-end

This section walks **one** `MarketEvent` from the wire all the way to a
published order. The example uses plausible numbers, not real market data;
the goal is to make the data flow concrete.

## 12.1 Setup

- Local venue: `nobitex`, symbol `BTC-USDT`, `Δt = 1000 ms`.
- Global venue: `binance`, symbol `BTC-USDT`.
- Actor: `mm-btc-001`, with `n_components = 3`, `horizon = 5`, `n_samples = 128`.

## 12.2 Step 1 — ingestion

At `t = 12:00:00.000` the `SymbolEngine` for `(nobitex, BTC-USDT)` receives a
REST snapshot of the orderbook and a WebSocket batch of trades. The feed
task pushes them into `OrderbookManager` and `RollingTradeBuffer`. The
`market:raw:nobitex:BTC-USDT:orderbook` and `:trades` topics receive the raw
payloads.

At `t = 12:00:01.000` the publish task fires, drains the trade buffer for
`[12:00:00.000, 12:00:01.000)`, and produces:

```
MarketEvent(
    exchange="nobitex", symbol="BTC-USDT", interval_ms=1000,
    t_start_ms=..., t_end_ms=...,
    orderbook=TopOfBook(bid=67500.1 @ 0.05, ask=67501.3 @ 0.04, ...),
    trades=[Trade(price=67500.5, qty=0.01, side="buy"), ...],
    buy_volume=0.42, sell_volume=0.31,
    vwap=67500.6, imbalance=0.15, microprice=67500.7,
    book_buy_vwap=67490.2, book_sell_vwap=67512.8, book_vwap_depth=2.4,
)
```

## 12.3 Step 2 — frame assembly

The `StrategyInputFrameAssembler` for `mm-btc-001` is waiting on its local
queue. It pops the event, drains the global queue (which has the matching
binance event already), and calls `build_frame`. The `JointFrameAssembler`
checks freshness and publishes a `JointStrategyFrame`.

## 12.4 Step 3 — strategy runner

The `StrategyRunner.run_step` coroutine pops the frame:

1. **Open orders** — the ledger reports 4 resting orders (2 bids, 2 asks)
   on nobitex, 0 on binance.
2. **Clean** — `subtract_own_orders` removes the bot's quotes from the
   `nobitex` book. The cleaned book now shows the *rest of the market*.
3. **Optimizer dispatch** — the cleaned frame and joint ledger are passed
   to `MpcStrategyEngine.decide`, which builds a `State` and a
   `MonteCarloMPCTensorBatched` optimizer. The future runs in the
   `ThreadPoolExecutor` with `optimizer_timeout_ms = 2000`.
4. **Inside the optimizer** (4-8 ms typical on CPU):
   - The L-param decoder turns the current raw vector into `(L, k, x0)`
     tensors (3 components per side → 18 raw parameters per side, 36 total).
   - Adam runs `E = 12` epochs; each epoch draws 128 paths of length 5.
   - At each step the soft fill is computed, the inventory/cash are
     updated, the terminal objective is summed.
   - The best epoch's `(L, k, x0)` are returned along with diagnostics.
5. **Target book** — the sigmoid action is laid on the nobitex price grid
   (tick = 0.1 USDT, lot = 0.0001 BTC) and constrained to `min_qty = 0.001`
   and `min_notional = 10 USDT`. The result is 4 bids and 4 asks inside
   the validity bounds.
6. **Risk check** — the `NetDeltaRiskEngine` validates the target book
   against the joint ledger. All 8 orders pass.
7. **Dispatch** — the approved book is routed to the local `OrderManager`,
   which places the orders on nobitex via REST. The decision is published
   to `decisions:mm-btc-001` and the diagnostics to
   `diagnostics:controller:mm-btc-001`.

## 12.5 What gets logged

The `ControllerDiagnostics` payload contains:

- `optimizer_status: "ok"`,
- `objective_history: [...12 floats...]`,
- `best_risk_adjusted_return: 0.0003`,
- `warm_start_alpha: 0.7` (warm-started from the previous decision),
- `reservation_r_offset: -0.0001` (slight short-skew because the bot is
  long),
- `book_quality_score: 0.92`,
- `availability_shortfall: 0.0`,
- `viability_theta: 0.01` (stable),
- `gradient_norm_final: 0.4`,
- `decision_hash: "9f3a..."`.

---

# 13 — Glossary

| Term | Definition |
| --- | --- |
| **Actor** | The strategy instance. Identified by a name (e.g. `mm-btc-001`). Owns one or two venues. |
| **Bus** | The pub/sub backbone. Either in-process (`InMemoryEventBus`) or Redis streams. |
| **`ControllerDiagnostics`** | The telemetry payload published after every decision. Contains optimizer stats, validity/viability scores, and the decision hash. |
| **CE (certainty equivalent)** | Risk-adjusted return: `(1/β) · log E[exp(β · ΔW)]`. |
| **D(s)** | Cumulative depth at spread `s` from the sigmoid action. |
| **Effective depth** | `D(s) − D(0)`; the executable portion of the curve above the inside level. |
| **`EffectiveSigmoidMixtureAction`** | View of `SigmoidMixtureAction` whose depth methods return executable depth. |
| **EWMA** | Exponentially weighted moving average. Used for online model parameter updates. |
| **Fill** | A matched order. The fill size is `min(D(s) − D(0), V_side)` in the soft model. |
| **Frame** | A `StrategyInputFrame` or `JointStrategyFrame` produced by an assembler. |
| **Gate** | A boolean execution switch. Closed gates route to the preview bus, not the live order manager. |
| **Hurdle model** | A distribution with a discrete mass at zero and a continuous positive distribution. |
| **Joint** | A mode where the strategy optimizes both a local and a global venue's sigmoid action in one vector. |
| **`JointStrategyFrame`** | The bundle of input frame + joint ledger + open orders, used by the strategy runner. |
| **`JointTargetBook`** | The pair of local + global `VenueTargetBook` produced by the target book builder. |
| **`L, k, x0`** | The three sigmoid parameters per component: depth cap, steepness, and center. |
| **Ledger** | The canonical record of balances, open orders, and fills. Driven by the reconciler. |
| **Local venue** | The venue the actor is quoting. The optimizer's primary target. |
| **L-param decoder** | `materialize_side_torch` / `materialize_side_np`: maps a raw parameter vector to effective `(L, k, x0)`. |
| **MCE** | Monte Carlo Estimate. A sample-mean over `B` paths. |
| **MPC** | Model Predictive Control. The optimizer simulates `H` future steps under a model and chooses the action that maximizes the expected objective. |
| **`MarketEvent`** | A closed-window per-interval snapshot: trades + orderbook + aggregates. |
| **`NetDeltaRiskEngine`** | The risk gate. Validates proposed orders against ledger state and risk limits. |
| **`OptimizerResult`** | The output of `optimize`: the best action and its diagnostic trail. |
| **`OrderbookRelative`** | An orderbook in `(spread, quantity)` form, signed by side. |
| **Reservation skew** | A small spread shift that incentivizes inventory rebalancing. `r` shifts the action centers. |
| **`Reservation_r_offset`** | The same as `r` above; the variable name in code. |
| **Risk-adjusted return** | `CE` (certainty equivalent). The risk-neutral score the optimizer maximizes. |
| **Ruin probability** | In viability: the probability that the bot runs out of either side's margin over the horizon. The viability penalty is a smooth upper bound on this. |
| **Sigmoid action** | A `SigmoidMixtureAction`: sum of `K` sigmoids per side. |
| **`Soft fill`** | Differentiable fill function: `min(D(s) − D(0), V_side)` with a smooth min inside the optimizer. |
| **State** | The MPC's input: market data + ledger state + reference price + skew. |
| **Target book** | The venue-ready list of orders produced by the target book builder. |
| **Validity** | Book quality: tightness, presence, balance, uniformity. |
| **Viability** | Availability of margin to support the proposed exposure. |
| **Viability dual** | `θ`, the Lagrange multiplier for the viability constraint, updated by subgradient. |
| **VWAP** | Volume-weighted average price. `Σ price·qty / Σ qty`. |
| **Warm-start** | Initializing the next decision's raw parameter vector from the previous decision's vector. |
| **Window** | A closed interval `[t_start, t_end]` over which trades are aggregated. |
