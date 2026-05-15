# Configuration

Runtime behavior is driven by a **JSON config file** plus a **repository-root `.env`** for secrets and infrastructure URLs.

## Files

| File | Purpose |
|------|---------|
| `mm/configs/market_maker.runtime.example.json` | Reference runtime config (Nobitex + Wallex example) |
| `.env.example` | Template for tokens and `MM_*` env vars |
| `.env` | Local secrets (gitignored); loaded by `mm.env_loader.load_mm_env()` |

Env loading is **root-only**: `load_mm_env()` reads `<repo>/.env`, not `mm/.env`.

## Running the runtime

```bash
# Recommended entrypoints (equivalent)
python -m mm.runtime --config path/to/runtime.json
python -m mm.tools.print_events --config path/to/runtime.json
```

CLI flags on `print_events` can override exchange/symbol/interval when not using `--config`; with `--config`, JSON is the source of truth.

## Top-level runtime JSON

| Section | Type | Description |
|---------|------|-------------|
| `primary` | object | **Required.** Primary market data source (`role` implied primary) |
| `hedge` | object | Optional hedge/reference venue |
| `interval_ms` | int | Frame and market-event interval (e.g. `10000`) |
| `max_hedge_lag_ms` | int | Max allowed hedge `t_end_ms` lag vs primary |
| `hedge_grace_ms` | int | Wait time for hedge event alignment |
| `bus` | object | `memory` or `redis_streams` |
| `persistence` | object | Postgres and/or Parquet |
| `observability` | object | Metrics, logs, OTLP |
| `primary_private` | object | API token, poll, execution |
| `hedge_private` | object | Same for hedge venue |
| `controller` | object | MPC controller (optional) |

Loader: `mm.runtime.config.load_runtime_config(path)`.

## Market data source

```json
{
  "exchange": "nobitex",
  "symbol": "USDT_IRT",
  "mode": "rest",
  "depth_limit": 100,
  "trades_limit": 200,
  "poll_interval_ms": 1000
}
```

| Field | Notes |
|-------|--------|
| `mode` | `rest` or `websocket` (venue capabilities checked at adapter select) |
| Topics | Auto: `market:events:{exchange}:{symbol}:{interval_ms}` |

## Bus

```json
"bus": {
  "kind": "redis_streams",
  "redis_url": "redis://localhost:6379/0",
  "stream_prefix": "mm",
  "consumer_group": "mm-runtime",
  "consumer_name": "runtime",
  "max_stream_length": 100000,
  "read_block_ms": 1000,
  "max_queue_size": 0
}
```

| `kind` | Use case |
|--------|----------|
| `memory` | Tests, single process |
| `redis_streams` | Durable fan-out, multiple consumers |

## Persistence

```json
"persistence": {
  "enabled": true,
  "database_url_env": "MM_DATABASE_URL",
  "parquet_root_env": "MM_PARQUET_ROOT",
  "topics": ["market:events:...", "strategy:frames:...", "strategy:decisions", ...],
  "batch_size": 100,
  "flush_interval_ms": 1000
}
```

If `topics` is empty, defaults are computed in `engine._default_persistence_topics`.

Postgres URL example (Compose):

```text
MM_DATABASE_URL=postgresql://meysam:Welcome123@localhost:5432/postgres
```

## Private + execution

```json
"primary_private": {
  "mode": "rest",
  "api_token_env": "NOBITEX_API_TOKEN",
  "poll_interval_ms": 10000,
  "enable_execution": true
}
```

| Field | Effect |
|-------|--------|
| `api_token` / `api_token_env` | Credentials for private REST |
| `enable_execution` | If true and token present, creates `OrderManager` + execution reconciler |

Without token, private reconciler is skipped and **no live orders** on that exchange.

## Controller (MPC)

```json
"controller": {
  "enabled": true,
  "observe_only": false,
  "action_start_after_steps": 3,
  "actor_id": "mpc-controller",
  "decisions_topic": "strategy:decisions",
  "action_topic": "strategy:actions",
  "min_spread": 1000.0,
  "orderbook_grid": { "price_step": 1000.0, "max_spread": 100000.0 },
  "qty_step": 0.0001,
  "min_order_qty": 0.0001,
  "max_orders_per_side": 10,
  "risk": {
    "enabled": true,
    "max_inventory": 1000.0,
    "max_data_age_ms": 60000
  },
  "vwap_depth": 100.0,
  "price_model": { "kind": "residual", ... },
  "market_flow_model": { "kind": "hurdle_lognormal", ... },
  "hedge_price_model": { "kind": "legacy", ... },
  "execution_model": { "kind": "real", ... },
  "optimizer": { "kind": "monte_carlo_mpc_tensor_batched", ... }
}
```

| Field | Safety |
|-------|--------|
| `observe_only: true` | Decisions published; **no** `OrderManager.sync` |
| `action_start_after_steps` | MPC runs from frame 1; sync only after N frames |
| `enabled: false` | No strategy runner |
| `risk.enabled: false` | Pass-through risk (no inventory/staleness checks) |
| `risk.max_inventory` | Per-order cap vs wallet inventory from private snapshot |

Optimizer keys map to `mm.mpc.mpc_tensor_batched` (samples, horizon, coefs, inventory targets, device, etc.).

## Derived topics

From `MarketMakerRuntimeConfig`:

| Property | Pattern |
|----------|---------|
| `frame_topic` | `strategy:frames:{primary_exchange}:{primary_symbol}:{interval_ms}` |
| Primary market | `market:events:{primary_exchange}:{primary_symbol}:{interval_ms}` |
| Hedge market | `market:events:{hedge_exchange}:{hedge_symbol}:{interval_ms}` |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NOBITEX_API_TOKEN`, `WALLEX_API_TOKEN` | Exchange REST (names match `api_token_env`) |
| `MM_DATABASE_URL` | Postgres persistence |
| `MM_PARQUET_ROOT` | Parquet files directory |
| `MM_JSON_LOG` | `1` for JSON log lines |
| `MM_LOG_PATH` | e.g. `logs/mm-runtime.jsonl` |
| `MM_METRICS_PORT` | Prometheus scrape port (default 8000) |
| `MM_RUN_ID`, `MM_SERVICE` | Labeling |
| `MM_TORCH_DEVICE` | MPC device (`cpu` / `cuda`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Traces when `otel_enabled` |

Copy from `.env.example`; never commit `.env`.

## Contract introspection

At startup, `print_events` can print the resolved **topic contract** from `config.as_contract()` — useful to verify topic names before enabling persistence or Grafana variables.

## Related documents

- [Architecture](architecture.md) — package and bus overview
- [Data flow](data-flow.md) — how config shapes the pipeline
- [Controller & execution](controller-and-execution.md) — controller fields in depth
- [Observability](observability.md) — Grafana and ops stack
