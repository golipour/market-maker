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
