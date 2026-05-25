# Configuration

Runtime behavior is driven by a **JSON config file** plus a **repository-root `.env`** for secrets and infrastructure URLs.

## Files

| File | Purpose |
|------|---------|
| `mm/configs/market_maker.runtime.example.json` | Reference runtime config (Nobitex local + Wallex global, live liquidity sourcing) |
| `.env.example` | Template for tokens and `MM_*` env vars |
| `.env` | Local secrets (gitignored); loaded by `mm.env_loader.load_mm_env()` |

Env loading is **root-only**: `load_mm_env()` reads `<repo>/.env`, not `mm/.env`.

## Running the runtime

```bash
python -m mm.runtime --config path/to/runtime.json
python -m mm.tools.print_events --config path/to/runtime.json
```

CLI flags on `print_events` can override exchange/symbol/interval when not using `--config`; with `--config`, JSON is the source of truth.

## Top-level runtime JSON

| Section | Type | Description |
|---------|------|-------------|
| `venues.local` | object | **Required.** Local distribution venue |
| `venues.global` | object | Optional global liquidity source / reference venue |
| `interval_ms` | int | Frame and market-event interval (e.g. `10000`) |
| `max_global_lag_ms` | int | Max allowed global `t_end_ms` lag vs local |
| `global_grace_ms` | int | Wait time for global event alignment |
| `liquidity_sourcing` | object | Single-venue vs joint-venue mode |
| `execution` | object | Per-venue planner policies and caps |
| `local_private` | object | API token, poll, execution for local venue |
| `global_private` | object | Same for global venue |
| `controller` | object | MPC controller (optional) |
| `bus` | object | `memory` or `redis_streams` |
| `persistence` | object | Postgres and/or Parquet |
| `observability` | object | Metrics, logs, OTLP |

Loader: `mm.runtime.config.load_runtime_config(path)`.

Use `venues.local` and optional `venues.global`; top-level venue aliases are
rejected.

### Venues and private credentials

```json
{
  "venues": {
    "local": {
      "exchange": "nobitex",
      "symbol": "USDT_IRT",
      "mode": "rest",
      "depth_limit": 100,
      "trades_limit": 200,
      "poll_interval_ms": 1000
    },
    "global": {
      "exchange": "wallex",
      "symbol": "USDT_IRT",
      "mode": "rest",
      "depth_limit": 100,
      "trades_limit": 200,
      "poll_interval_ms": 1000
    }
  },
  "local_private": {
    "mode": "rest",
    "api_token_env": "NOBITEX_API_TOKEN",
    "poll_interval_ms": 10000,
    "enable_execution": true
  },
  "global_private": {
    "mode": "rest",
    "api_token_env": "WALLEX_API_TOKEN",
    "poll_interval_ms": 10000,
    "enable_execution": true
  }
}
```

| Field | Effect |
|-------|--------|
| `api_token` / `api_token_env` | Credentials for private REST |
| `enable_execution` | If true and token present, creates `OrderManager` + execution reconciler |

Without token, private reconciler is skipped and **no live orders** on that exchange.

### Liquidity sourcing

```json
"liquidity_sourcing": {
  "enabled": true,
  "mode": "liquidity_sourcing",
  "reference_venue": "global",
  "require_global_private": false,
  "global_live_orders": false
}
```

| Field | Default | Notes |
|-------|---------|-------|
| `enabled` | `false` | Master switch for joint-venue architecture |
| `mode` | `single_venue` | `liquidity_sourcing` enables joint MPC and dual target books |
| `reference_venue` | `global` | Which venue supplies consolidated mark price |
| `require_global_private` | `false` | Gate global wallet freshness when global orders are emitted |
| `global_live_orders` | `false` | Allow live resting orders on the global venue |

### Execution planners

```json
"execution": {
  "local": {"planner": "reconcile"},
  "global": {
    "planner": "replace",
    "max_order_qty": 0.5,
    "max_quote_distance": 10.0
  }
}
```

| Field | Notes |
|-------|--------|
| `planner` | `reconcile` (keep within tolerance) or `replace` (cancel-all then place) |
| `observe_only` | Per-venue dry-run override |
| `max_order_qty` | Source-venue size cap (global venue) |
| `max_quote_distance` | Max quote spread from mid for global quotes |

## Controller (MPC)

```json
"controller": {
  "enabled": true,
  "observe_only": false,
  "action_start_after_steps": 10,
  "actor_id": "mpc-controller",
  "risk": {
    "enabled": true,
    "engine": "net_delta",
    "local": {
      "min_inventory": 0.0,
      "max_inventory": 1000.0,
      "min_cash": 0.0,
      "max_open_notional": 100000000.0,
      "max_order_notional": 100000000.0,
      "max_market_data_age_ms": 60000,
      "max_private_state_age_ms": 120000
    },
    "global": {
      "min_inventory": 0.0,
      "max_inventory": 1000.0,
      "min_cash": 0.0,
      "max_market_order_notional": 10000000.0,
      "max_slippage_bps": 100.0,
      "max_market_data_age_ms": 60000,
      "max_private_state_age_ms": 120000
    }
  },
  "global_price_model": {"kind": "ewma_gaussian", "decay": 0.95},
  "execution_model": {"kind": "real", "maker_fee": 0.0},
  "optimizer": {"kind": "monte_carlo_mpc_tensor_batched", ...}
}
```

| Field | Safety |
|-------|--------|
| `observe_only: true` | Decisions published; **no** live order sync |
| `action_start_after_steps` | MPC runs from frame 1; sync only after N frames |
| `risk.engine` | `net_delta` (default): target-book transition against ledger |
| `risk.local` / `risk.global` | Per-venue hard collateral and freshness gates |

### Hard risk (`controller.risk`)

Separate from optimizer soft preferences (`inventory_target_penalty_weight`, etc.).

| Field | Role |
|-------|------|
| `local.min_inventory` | Floor on base inventory after a sell |
| `local.max_inventory` | Ceiling on base inventory after a buy |
| `local.min_cash` | Minimum quote cash after a buy |
| `local.max_open_notional` | Cap on resting open order notional |
| `local.max_order_notional` | Per-order notional cap |
| `local.max_market_data_age_ms` | Reject if local market event is too old |
| `local.max_private_state_age_ms` | Reject if local private snapshot missing or stale |
| `global.*` | Same inventory/cash floors for global wallet; `max_market_order_notional` and `max_slippage_bps` for global venue |

## Derived topics

From `MarketMakerRuntimeConfig`:

| Property | Pattern |
|----------|---------|
| `joint_frame_topic` | `strategy:joint:frames:{local_exchange}:{local_symbol}:{interval_ms}` |
| Local market | `market:events:{local_exchange}:{local_symbol}:{interval_ms}` |
| Global market | `market:events:{global_exchange}:{global_symbol}:{interval_ms}` |
| Ledger | `ledger:snapshot` |
| Target books | `strategy:target_books:{actor_id}` |

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
- [Liquidity sourcing upgrade plan](liquidity-sourcing-architecture-upgrade-plan.md)
