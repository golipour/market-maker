# Market maker documentation

This folder describes the **event-bus runtime** in `mm/`: how market data becomes strategy frames, how the MPC controller decides quotes, how orders reach the exchange, and how telemetry and persistence fit in.

## Guides

| Document | Contents |
|----------|----------|
| [Architecture overview](architecture.md) | Layers, packages, event bus, envelopes, design principles |
| [Data flow](data-flow.md) | Ingestion → frames → strategy → execution → persistence (with diagrams) |
| [Controller & execution](controller-and-execution.md) | `StrategyRunner`, MPC, risk, `OrderManager` |
| [Observability & operations](observability.md) | Metrics, logs, traces, Postgres/Parquet, Grafana, local stack |
| [Configuration](configuration.md) | Runtime JSON, environment variables, topics, example deploy |

## Quick start

```bash
# From repository root
cp .env.example .env   # add API tokens and MM_DATABASE_URL if needed
docker compose up -d     # Postgres, Redis, Grafana, Prometheus, Loki, …

python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
# equivalent: python -m mm.tools.print_events --config …
```

See [Configuration](configuration.md) for all flags and [Observability](observability.md) for Grafana dashboards under `grafana/dashboards/`.

## Repository layout (high level)

```
market_maker/
├── mm/                    # Application code
│   ├── runtime/           # Single production entry: build + start RuntimeComponents
│   ├── application/       # Frame assembler, execution service re-exports
│   ├── strategies/        # MpcStrategyEngine
│   ├── application/     # StrategyRunner, frame prep, execution routing
│   ├── ingestion/         # SymbolEngine, replay
│   ├── bus/               # In-memory or Redis Streams event bus
│   ├── execution/         # OrderManager, reconcilers
│   ├── persistence/       # Postgres + Parquet writers
│   ├── observability/     # Metrics, logs, traces, bus subscribers
│   ├── exchanges/         # Venue adapters (REST/WS market + orders)
│   ├── mpc/               # Tensor batched MPC optimizer
│   └── configs/           # Example runtime JSON
├── grafana/               # Provisioned dashboards + datasources
├── docker-compose.yml     # Local observability + Postgres + Redis
└── docs/                  # This documentation
```
