# Observability and operations

Observability is **decoupled from strategy code**: the runtime publishes domain events on the bus; **`RuntimeObservabilitySubscriber`** and optional **OTLP / Prometheus / JSON logs** consume them.

## Setup

`configure_observability(config.observability)` runs at runtime build time (`mm.observability.setup`):

| Config field | Effect |
|--------------|--------|
| `json_logs` | Sets `MM_JSON_LOG=1` — structured stdout / file lines |
| `log_path` | `MM_LOG_PATH` — resolved relative to **repo root** (for Docker Alloy mount) |
| `metrics_enabled` + `metrics_port` | HTTP `/metrics` on port (default 8000) |
| `run_id`, `service_name` | `MM_RUN_ID`, `MM_SERVICE` labels |
| `otel_enabled` + `otlp_endpoint` | OpenTelemetry `TracerProvider` → Tempo |

## Bus subscriber

**`RuntimeObservabilitySubscriber`** (`mm.observability.subscribers`) subscribes to:

- `strategy:joint:frames:…` (when liquidity sourcing enabled)
- `strategy:decisions`
- `strategy:actions`
- `execution:orders:{local}` and `execution:state:{local}`
- `market:errors:{local}`  
- Same execution/error topics for **global** when configured

### Metrics (examples)

| Metric | When |
|--------|------|
| `mm_strategy_frames_total` | Each legacy `StrategyInputFrame` |
| `mm_joint_strategy_frames_total` | Each ledger-backed `JointStrategyFrame` |
| `mm_latency_skew_ms` | Joint frame local/global market skew |
| `mm_frame_local_market_ready` / `mm_frame_global_market_ready` | Joint frame sync readiness |
| `mm_frame_ledger_fresh` | Joint ledger freshness gate |
| `mm_global_lag_ms` | Frame includes global market lag |
| `mm_controller_latency_ms` | `StrategyDecision` payload |
| `mm_controller_orders_per_decision` | Order count in decision |
| `mm_execution_sync_latency_ms` | Time from frame to `ExecutionSyncRequested` |
| `mm_live_orders` | After execution snapshot |
| `mm_order_validation_rejected_total` | Venue limit rejection |
| `mm_persistence_*` | Postgres/Parquet writer health |

Definitions and help strings: `mm.observability.schema`.

### Logs

`log_event(name, payload)` — JSON lines when `MM_JSON_LOG` is set; optional append to `MM_LOG_PATH`.

### Traces

`span(...)` context manager around frame assembly and other hot paths when OTLP is configured.

## Persistence

Enabled when `persistence.enabled` is true in runtime JSON.

### PostgreSQL

- **Writer:** `mm.persistence.postgres.PostgresEventWriter`
- **URL:** `database_url` or env `MM_DATABASE_URL`
- **Table:** `mm_events` (see DDL in writer module)
- **Topics:** explicit `persistence.topics` or `_default_persistence_topics()` in engine

Default topics include market events, joint strategy frames, ledger snapshots, strategy actions, decisions (if controller on), execution and private state for local (and global).

### Parquet

- **Writer:** `mm.persistence.parquet_writer.ParquetEventWriter`
- **Root:** `parquet_root` or `MM_PARQUET_ROOT`

### Replay

`mm.persistence.replay.read_envelopes` + `mm.ingestion.replay_adapter` republish into the bus for offline runs.

## Local stack (Docker Compose)

From repository root:

```bash
docker compose up -d
```

| Service | Port | Role |
|---------|------|------|
| Postgres | 5432 | `mm_events`, economics SQL init script |
| Redis | 6379 | Optional `redis_streams` bus |
| Grafana | 3000 | Dashboards from `grafana/dashboards` |
| Prometheus | 9090 | Scrapes runtime `/metrics` (host gateway) |
| Loki | 3100 | Log aggregation |
| Tempo | 3200 / 4317–4318 | Traces |
| Alloy | 12345 | Ships logs to Loki |
| pgAdmin | 5050 | DB UI |
| RedisInsight | 5540 | Redis UI |

Grafana provisioning:

- `grafana/provisioning/datasources/` — Postgres, Prometheus (Mimir file), Loki, Tempo
- `grafana/provisioning/dashboards/dashboards.yml` — loads JSON from `grafana/dashboards`

### Key dashboards

| UID / file | Purpose |
|------------|---------|
| `mm-controller-diagnostics-pg` | Optimizer objective, joint inventory penalties, MC return/reward, joint expected fills (`ControllerDiagnostics`) |
| `mm-local-action-pg` | Target-book execution funnel order counts by stage (`ControllerDiagnostics.action_pipeline`) |
| `mm-local-action-rejections-pg` | Per-stage rejection reason counts (frame / target-book / rules / router taxonomy) |
| `mm-liquidity-sourcing-pg` | Frame sync readiness, venue execution gates, joint target books |
| `mm-ledger-pg` | Ledger balances, locked/available collateral, live order counts, freshness |
| `mm-strategy-model-params-pg` | Model params from decisions + NLL / skip reasons from diagnostics |
| `mm-portfolio-pg` | Inventory, cash, wallets, net-delta locked collateral exposure |
| `mm-orderbook-depth-pg` | Latest book + MPC continuous depth + executable orders (Postgres) |
| `mm-market-data-pg` | Local/global market trade volume and best bid/ask |
| `mm-reference-vwaps-pg` | Local/global reference VWAPs from decisions |
| `mm-runtime-metrics` | Prometheus runtime metrics including joint frame counters |
| `private_state_postgres` | Private snapshots |

**Dashboard variables** must match runtime: e.g. `exchange=nobitex`, `symbol=USDT_IRT`, `interval_ms=10000`, `diagnostics_topic=strategy:diagnostics:mpc-controller`, `ledger_topic=ledger:snapshot`.

Regenerate JSON from `python3 mm/tools/generate_grafana_dashboards.py` after changing panel SQL.

## Admin control plane

**`RuntimeControlPlane`** listens on `admin:commands`:

| Command | Effect |
|---------|--------|
| `pause` / `resume` | Per-actor pause set (not read by MPC runner today) |
| `kill_switch` | Sets internal flag + metric (not wired to MPC) |
| `observe_only` | Sets internal flag (not wired to MPC; use JSON config) |
| `cancel_all` | Publishes cancel request on bus |

Publishes state on `admin:events`. Schema: `mm.api.admin`.

## Inspecting the bus without Grafana

```bash
python -m mm.tools.print_events --config mm/configs/market_maker.runtime.example.json
```

Summarizes contracts, subscribes to configured topics, prints human-readable event summaries (`summarize_event`).

Use `--raw` for full payloads. Useful for verifying frames and decisions before enabling live execution.

## Environment variables (observability-related)

See [Configuration](configuration.md) and `.env.example`:

- `MM_JSON_LOG`, `MM_LOG_PATH`
- `MM_METRICS_PORT`, `MM_RUN_ID`, `MM_SERVICE`
- `MM_DATABASE_URL`, `MM_PARQUET_ROOT`

## Related documents

- [Architecture](architecture.md)
- [Data flow](data-flow.md)
- [Configuration](configuration.md)
