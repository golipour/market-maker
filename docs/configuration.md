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
| `venues.local` | object | Preferred name for the primary distribution venue |
| `venues.global` | object | Optional liquidity source / reference venue |
| `primary` | object | **Legacy alias** for `venues.local` |
| `hedge` | object | **Legacy alias** for `venues.global` |
| `interval_ms` | int | Frame and market-event interval (e.g. `10000`) |
| `max_hedge_lag_ms` | int | Max allowed hedge `t_end_ms` lag vs primary |
| `hedge_grace_ms` | int | Wait time for hedge event alignment |
| `liquidity_sourcing` | object | Joint-venue mode (disabled by default; see below) |
| `bus` | object | `memory` or `redis_streams` |
| `persistence` | object | Postgres and/or Parquet |
| `observability` | object | Metrics, logs, OTLP |
| `local_private` / `primary_private` | object | API token, poll, execution for local venue |
| `global_private` / `hedge_private` | object | Same for global venue |
| `controller` | object | MPC controller (optional) |

Loader: `mm.runtime.config.load_runtime_config(path)`.

### Neutral venue aliases

New configs should use neutral roles:

```json
{
  "venues": {
    "local": {
      "exchange": "nobitex",
      "symbol": "USDT_IRT",
      "mode": "rest"
    },
    "global": {
      "exchange": "wallex",
      "symbol": "USDT_IRT",
      "mode": "rest"
    }
  },
  "local_private": {
    "mode": "rest",
    "api_token_env": "NOBITEX_API_TOKEN",
    "enable_execution": true
  },
  "global_private": {
    "mode": "rest",
    "api_token_env": "WALLEX_API_TOKEN",
    "enable_execution": true
  }
}
```

Legacy `primary` / `hedge` / `primary_private` / `hedge_private` keys still parse.
The loader records non-fatal warnings in `config.config_warnings` when legacy keys
are used. Runtime behavior is unchanged when only legacy keys are present.

Code accessors: `config.local` (alias for `primary`), `config.global_source` (alias
for `hedge`), `config.local_private`, `config.global_private`.

### Liquidity sourcing (Phase 1, disabled by default)

```json
"liquidity_sourcing": {
  "enabled": false,
  "mode": "single_venue",
  "reference_venue": "global",
  "require_global_private": false
}
```

| Field | Default | Notes |
|-------|---------|-------|
| `enabled` | `false` | When false, runtime uses today's single-venue path |
| `mode` | `single_venue` | `liquidity_sourcing` reserved for future joint quoting |
| `reference_venue` | `global` | Which venue supplies consolidated mark price |
| `require_global_private` | `false` | Future gate for global wallet freshness |

With defaults omitted, existing JSON configs behave exactly as before.

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
    "primary": {
      "min_inventory": 0.0,
      "max_inventory": 1000.0,
      "min_cash": 10000000.0,
      "max_open_notional": 200000000.0,
      "max_order_notional": 50000000.0,
      "max_market_data_age_ms": 60000,
      "max_private_state_age_ms": 120000
    },
    "hedge": {
      "min_inventory": 0.0,
      "max_inventory": 1000.0,
      "min_cash": 10000000.0,
      "max_market_order_notional": 50000000.0,
      "max_slippage_bps": 20.0,
      "max_market_data_age_ms": 60000,
      "max_private_state_age_ms": 120000
    }
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
| `risk.enabled: false` | Pass-through risk (no hard gates) |
| `risk.engine` | `order_level` (default, legacy per-order gate) or `net_delta` (target-book transition against ledger) |
| `risk.primary` | Hard gates for primary quote orders (see below) |
| `risk.hedge` | Hard gates for hedge venue / private freshness when hedge is active |

### Hard risk (`controller.risk`)

Separate from **optimizer soft preferences** (`inventory_target_penalty_weight`, `target_inventory`, etc. under `controller.optimizer`). The runtime risk engine enforces hard limits before any order reaches the venue.

| Field | Role |
|-------|------|
| `primary.min_inventory` | Floor on base inventory after a sell (default `0.0`; spot has no shorts) |
| `primary.max_inventory` | Ceiling on base inventory after a buy |
| `primary.min_cash` | Minimum quote cash after a buy |
| `primary.max_open_notional` | Cap on resting open order notional (same exchange/symbol) |
| `primary.max_order_notional` | Per-order notional cap |
| `primary.max_market_data_age_ms` | Reject if primary market event is too old |
| `primary.max_private_state_age_ms` | Reject if primary private snapshot missing or stale |
| `hedge.*` | Same inventory/cash floors for hedge wallet; `max_market_order_notional` and `max_slippage_bps` for live hedge orders (Phase 6) |

Legacy flat keys still parse for one release: top-level `max_inventory` and `max_data_age_ms` map to `risk.primary`.

Optimizer keys map to `mm.mpc.mpc_tensor_batched` (samples, horizon, penalty weights, inventory targets, device, etc.).

## Legacy hedge components

The following remain supported for current single-venue and primary-plus-hedge
deployments. They are **not** the foundation for the liquidity-sourcing upgrade
(see `docs/liquidity-sourcing-architecture-upgrade-plan.md`). Do not build new
features on them unless you are maintaining today's reactive hedge path.

| Component | Config / code | Role today | Migration direction |
|-----------|---------------|------------|---------------------|
| MPC hedge rollout | `controller.optimizer.hedge_rollout_enabled` | Simulates immediate offsetting hedge trades inside the optimizer objective | Replaced by joint-venue consolidated wealth when liquidity sourcing is enabled |
| Fill-driven hedge executor | `hedging.enabled`, `HedgeExecutionService` | Places reactive market hedge orders after primary fills | Remains as optional emergency/legacy tool; mutually exclusive with live joint-venue quoting |

Both are independent: enabling one does not imply the other.

## Live hedging (fill-driven, legacy)

Separate from MPC simulation (`controller.optimizer.hedge_rollout_enabled`, also legacy):

```json
"hedging": {
  "enabled": false,
  "observe_only": true,
  "actor_id": "hedge-executor",
  "dedupe_window_ms": 300000
}
```

| Field | Effect |
|-------|--------|
| `enabled` | Subscribe to `private:state:{primary}:fills` and emit hedge intents |
| `observe_only` | Publish intents/diagnostics but do not place hedge market orders |
| `dedupe_window_ms` | Ignore duplicate fill events within the window |

Primary fill **buy** → hedge **sell**; primary fill **sell** → hedge **buy**. Hedge orders use `order_type: market` and pass through `controller.risk.hedge` limits. Topics: `hedge:intents:{actor_id}`, `hedge:diagnostics:{actor_id}`.

Requires `hedge`, `hedge_private.enable_execution`, and a hedge `OrderManager`.

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
