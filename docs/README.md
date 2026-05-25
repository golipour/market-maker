# Market maker documentation

This folder describes the **event-bus runtime** in `mm/`: how market data becomes strategy frames, how the MPC controller decides quotes, how orders reach the exchange, and how telemetry and persistence fit in.

## Guides

| Document | Contents |
|----------|----------|
| [Architecture overview](architecture.md) | Layers, packages, event bus, envelopes, design principles |
| [Data flow](data-flow.md) | End-to-end phases with sequence diagram |
| **[Runtime mechanisms](mechanisms.md)** | **Detailed internals: tasks, queues, hedge drain/grace, raw vs cleaned books, execution** |
| [Controller & execution](controller-and-execution.md) | `StrategyRunner`, MPC, risk, `OrderManager` |
| [Observability & operations](observability.md) | Metrics, logs, traces, Postgres/Parquet, Grafana |
| [Configuration](configuration.md) | Runtime JSON, environment variables, topics |
| [Architecture diagnostics review](architecture-diagnostics-review.md) | Architecture review, data flow, improvement roadmap, diagnostics and observability suggestions |
| [Codebase upgrade plan](codebase-upgrade-plan.md) | Phased cleanup plan for diagnostics, optimizer refactors, risk config, hedging, and Grafana dashboards |
| [Joint local/global MPC](joint-local-global-mpc.md) | Current local/global models, joint rollout, global simulation, objective weights, and shared optimizer parameters |
| [Batched tensor MPC optimizer](mpc-tensor-batched-optimizer.md) | Detailed optimizer algorithm, objective components, diagnostics, performance and readability suggestions |
| [Hedging in MPC](hedging-in-mpc.md) | How hedge rollouts work today, why live hedge execution is incomplete, and an implementation roadmap |

Start with [Data flow](data-flow.md) for the big picture, then read **[Runtime mechanisms](mechanisms.md)** when you need to know *why* the code does something (e.g. double `_drain_latest_inputs`, where own orders are subtracted).

## Quick start

```bash
cp .env.example .env
docker compose up -d
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

## Repository layout (high level)

```
market_maker/
├── mm/
│   ├── runtime/           # build_runtime_components, CLI entry
│   ├── application/       # Frame assembler, StrategyRunner, frame prep
│   ├── strategies/      # MpcStrategyEngine
│   ├── ingestion/         # SymbolEngine
│   ├── bus/               # memory | redis_streams
│   ├── execution/         # OrderManager
│   ├── risk/              # RuntimeRiskEngine
│   └── configs/           # Example runtime JSON
├── grafana/
├── docker-compose.yml
└── docs/
```
