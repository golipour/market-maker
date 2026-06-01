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
    "enable_execution": true,
    "fill_event_source": "exchange_fill_history"
  },
  "global_private": {
    "mode": "rest",
    "api_token_env": "WALLEX_API_TOKEN",
    "poll_interval_ms": 10000,
    "enable_execution": true,
    "fill_event_source": "exchange_fill_history"
  }
}
```

For Nobitex API-key auth, use `api_key_env` and `api_private_key_env`
instead of `api_token_env`. The public key is sent as `Nobitex-Key`; the
private key is used locally to sign requests with Ed25519.

| Field | Effect |
|-------|--------|
| `api_token` / `api_token_env` | Credentials for private REST |
| `enable_execution` | If true and token present, creates `OrderManager` + execution reconciler |
| `fill_event_source` | One of `exchange_fill_history` or `open_order_delta`; controls which source publishes `private:state:{exchange}:fills` |

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
  "local": {"planner": "replace"},
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
  "max_orders_per_side": 5,
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
      "max_open_notional": 100000000.0,
      "max_order_notional": 100000000.0,
      "max_market_data_age_ms": 60000,
      "max_private_state_age_ms": 120000
    }
  },
  "reference_price_source": "global",
  "local_price_model": {"kind": "residual", "residual_decay": 0.95},
  "local_market_flow_model": {"kind": "hurdle_lognormal", "decay": 0.95},
  "global_market_flow_model": {"kind": "hurdle_lognormal", "decay": 0.95},
  "global_price_model": {"kind": "ewma_gaussian", "decay": 0.95},
  "execution_model": {
    "kind": "real",
    "maker_fee": 0.0,
    "local_maker_fee": 0.0,
    "global_maker_fee": 0.0
  },
  "optimizer": {
    "kind": "monte_carlo_mpc_tensor_batched",
    "n_samples": 64,
    "horizon": 1,
    "local": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_threshold_penalty_weight": 0.000001,
      "cash_threshold": 0.0,
      "inventory_threshold": 0.0,
      "quote_size_min_per_side": 0.3,
      "quote_size_max_per_side": 10.0,
      "quote_capacity_ratio": 1.0,
      "soft_fill_temp": 0.1,
      "soft_fill_activation": "hardsigmoid",
      "soft_fill_queue_model": "back_of_queue"
    },
    "global": {
      "objective_weight": 1.0,
      "return_risk_aversion_beta": 0.1,
      "inventory_threshold_penalty_weight": 0.000001,
      "cash_threshold": 0.0,
      "inventory_threshold": 0.0,
      "quote_size_min_per_side": 0.3,
      "quote_size_max_per_side": 10.0,
      "quote_capacity_ratio": 1.0,
      "soft_fill_temp": 0.1,
      "soft_fill_activation": "hardsigmoid",
      "soft_fill_queue_model": "back_of_queue"
    }
  }
}
```

`controller.vwap_depth` is no longer a fixed config value. The controller
derives VWAP depth each frame from `abs(local_inventory)`, or from
`abs(local_inventory + global_inventory)` when liquidity sourcing is enabled.
If that depth is larger than the displayed book, VWAP consumes only available
book depth.

| Field | Safety |
|-------|--------|
| `observe_only: true` | Decisions published; **no** live order sync |
| `action_start_after_steps` | MPC runs from frame 1; sync only after N frames |
| `max_orders_per_side` | Strategy cap before target book: max bid levels and max ask levels kept |
| `risk.engine` | `net_delta` (default): target-book transition against ledger |
| `risk.local` / `risk.global` | Same per-venue hard collateral and freshness gates |

### Hard risk (`controller.risk`)

Separate from optimizer soft preferences (`optimizer.local.*` / `optimizer.global.*`).

| Field | Role |
|-------|------|
| `local.min_inventory` / `global.min_inventory` | Floor on base inventory after a sell |
| `local.max_inventory` / `global.max_inventory` | Ceiling on base inventory after a buy |
| `local.min_cash` / `global.min_cash` | Minimum quote cash after a buy |
| `local.max_open_notional` / `global.max_open_notional` | Cap on resting open order notional |
| `local.max_order_notional` / `global.max_order_notional` | Per-order notional cap |
| `local.max_market_data_age_ms` / `global.max_market_data_age_ms` | Reject if market event is too old |
| `local.max_private_state_age_ms` / `global.max_private_state_age_ms` | Reject if private snapshot missing or stale |

## Derived topics

From `MarketMakerRuntimeConfig`:

| Property | Pattern |
|----------|---------|
| `joint_frame_topic` | `strategy:joint:frames:{local_exchange}:{local_symbol}:{interval_ms}` |
| Local market | `market:events:{local_exchange}:{local_symbol}:{interval_ms}` |
| Global market | `market:events:{global_exchange}:{global_symbol}:{interval_ms}` |
| Ledger | `ledger:snapshot` |
| Target books | `strategy:target_books:{actor_id}` |

Normally, leave `persistence.topics` unset. Persistence then uses the default
topic set derived from the configured venues, controller actor, and
`interval_ms`. Set `persistence.topics` only when you intentionally want a
custom subset or extra durable topics.

When you do override `persistence.topics`, runtime placeholders are available so
interval or venue changes do not leave writers subscribed to old topic names:

| Placeholder | Resolves to |
|-------------|-------------|
| `{interval_ms}` | Configured frame interval |
| `{local_market_event_topic}` | Local market event topic for `interval_ms` |
| `{global_market_event_topic}` | Global market event topic for `interval_ms` |
| `{joint_frame_topic}` | Joint strategy frame topic for `interval_ms` |
| `{frame_topic}` | Local strategy frame topic for `interval_ms` |
| `{strategy_actions_topic}` | Controller action topic |
| `{strategy_decisions_topic}` | Controller decision topic |
| `{strategy_diagnostics_topic}` | Controller diagnostics topic |
| `{strategy_target_books_topic}` | Controller target-books topic |
| `{local_execution_orders_topic}` / `{global_execution_orders_topic}` | Execution order topics |
| `{local_execution_state_topic}` / `{global_execution_state_topic}` | Execution state topics |
| `{local_private_snapshot_topic}` / `{global_private_snapshot_topic}` | Private snapshot topics |
| `{ledger_topic}` | Ledger snapshot topic |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NOBITEX_API_TOKEN` | Legacy Nobitex account-token auth (`api_token_env`) |
| `NOBITEX_API_KEY`, `NOBITEX_API_PRIVATE_KEY` | Nobitex signed API-key auth (`api_key_env`, `api_private_key_env`) |
| `WALLEX_API_TOKEN` | Wallex REST API key (`api_token_env`) |
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
