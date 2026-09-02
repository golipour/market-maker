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
