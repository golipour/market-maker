# Market maker

Event-bus market-making runtime: multi-venue ingestion, interval market events, MPC strategy frames, optional live limit-order execution, and Postgres/Grafana observability.

## Documentation

Full architecture and operations guides live in **[docs/](docs/README.md)**:

- [Architecture overview](docs/architecture.md)
- [Data flow](docs/data-flow.md)
- [Runtime mechanisms](docs/mechanisms.md) (detailed event flows & internals)
- [Controller & execution](docs/controller-and-execution.md)
- [Observability & operations](docs/observability.md)
- [Configuration](docs/configuration.md)
- [Local development (decomposed stack, admin, UI)](docs/local-development.md)

## Quick start

```bash
cp .env.example .env          # add API tokens, MM_DATABASE_URL, etc.
docker compose up -d          # optional: Postgres, Redis, Grafana stack
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

Use `controller.dry_run: true` in JSON until you intend to place live orders.
