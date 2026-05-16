# Data flow

High-level path from exchange feeds to optional live orders. For **task queues, timing, double-drain hedge alignment, raw vs cleaned books**, and other internals, see **[Runtime mechanisms](mechanisms.md)**.

## End-to-end diagram

```mermaid
sequenceDiagram
  participant EX as Exchange
  participant SE as SymbolEngine
  participant SCH as IntervalScheduler
  participant BUS as Event bus
  participant FA as Frame assembler
  participant SR as StrategyRunner
  participant ENG as MpcStrategyEngine
  participant RISK as RiskEngine
  participant OM as OrderManager
  participant OBS as Observability
  participant PG as Postgres writer

  EX->>SE: orderbook_stream / trade_stream
  SE->>SE: OrderbookManager + RollingTradeBuffer
  SCH->>SE: tick (interval_ms)
  SE->>BUS: MarketEvent on market:events:…
  BUS->>FA: primary queue (+ hedge queue)
  FA->>FA: drain → hedge grace → drain → select hedge
  FA->>BUS: StrategyInputFrame on strategy:frames:…
  BUS->>SR: frame_queue
  SR->>SR: subtract own orders from books
  SR->>ENG: decide(clean_frame)
  ENG-->>SR: StrategyDecision + Action
  SR->>RISK: validate_frame
  RISK-->>SR: approved actions
  SR->>BUS: strategy:decisions, strategy:actions
  alt not observe_only and past warmup
    SR->>OM: route_actions(approved)
    OM->>EX: place_orders / cancel_orders
    OM->>BUS: execution:orders / execution:state
  end
  BUS->>OBS: metrics + logs
  BUS->>PG: EventEnvelope batches
```

## Phase 1 — Market ingestion

`SymbolEngine` maintains a local book + trade buffer, then publishes interval **`MarketEvent`** on `market:events:{exchange}:{symbol}:{interval_ms}`.

Details: [Mechanisms — market ingestion](mechanisms.md#phase-1--market-ingestion-symbolengine)

## Phase 2 — Strategy input frames

`StrategyInputFrameAssembler` runs async tasks on primary/hedge market queues and private snapshot queues.

Per primary `MarketEvent`:

1. **Drain** pending hedge + private inputs.
2. **Wait** up to `hedge_grace_ms` for hedge with same `t_end_ms`.
3. **Drain** again (catch arrivals during wait).
4. **Select** hedge → `hedge_status` (`ready` / `missing` / `stale` / `disabled`).
5. **Publish** `StrategyInputFrame` on `strategy:frames:…`.

The published frame still has the **exchange** orderbook (includes your resting orders). Cleaning happens in phase 2b.

Details: [Mechanisms — frame assembly](mechanisms.md#phase-2--frame-assembly-strategyinputframeassembler)

### Frame contents

| Field | Source |
|-------|--------|
| `primary` | Primary `MarketEvent` for this interval |
| `hedge` | Aligned hedge event or `None` |
| `primary_private` / `hedge_private` | Latest `private:state:{exchange}:snapshot` |
| `hedge_status`, `hedge_lag_ms` | Assembler |
| `created_at_ms` | Wall clock when frame was built |

## Phase 2b — Own-order subtraction

`StrategyRunner` loads `OrderManager.live_orders`, then `clean_strategy_frame` → `subtract_own_orders` on primary/hedge books before MPC.

Details: [Mechanisms — own-order subtraction](mechanisms.md#phase-2b--own-order-subtraction-strategyrunner)

## Phase 3 — Strategy runner

`StrategyRunner`: clean frame → `MpcStrategyEngine.decide` → risk → publish → optional `OrderManager.sync`.

Details: [Mechanisms — strategy decision](mechanisms.md#phase-3--strategy-decision-strategyrunner--mpcstrategyengine) · [Controller & execution](controller-and-execution.md)

## Phase 4 — Execution sync

`OrderManager.sync`: cancels → place with venue limits → update `live_orders` → `execution:state`.

Details: [Mechanisms — execution](mechanisms.md#phase-4--execution-ordermanager)

## Phase 5 — Side effects

| Subscriber | Role |
|------------|------|
| `RuntimeObservabilitySubscriber` | Metrics, logs, traces |
| `PostgresEventWriter` / `ParquetEventWriter` | Durable `mm_events` |
| `RuntimeControlPlane` | Admin kill_switch / observe_only |

Details: [Mechanisms — bus & persistence](mechanisms.md#event-bus-delivery) · [Observability](observability.md)

## Parallel flows (hedge venue)

Separate `SymbolEngine` per venue; assembler aligns hedge to primary window; MPC places on **primary**; hedge informs models/VWAP.

## Debugging breakpoints

| Goal | Location |
|------|----------|
| Frame assembled | `frame_assembler.build_frame` / `_run_primary` publish |
| Hedge drain / grace | `frame_assembler._drain_latest_inputs`, `_wait_for_matching_hedge` |
| Own-order subtraction | `clean_strategy_frame` |
| MPC | `mpc_engine.decide` |
| Risk | `risk_engine.validate_frame` |
| Venue limits | `order_manager._place_new` |

Full map: [Mechanisms — debugging](mechanisms.md#debugging-map)

## Replay

`mm.ingestion.replay_adapter` republishes persisted envelopes into the bus for offline runs.

## Related documents

- **[Runtime mechanisms](mechanisms.md)** — detailed internals (recommended)
- [Architecture overview](architecture.md)
- [Controller & execution](controller-and-execution.md)
- [Configuration](configuration.md)
