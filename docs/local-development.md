# Local development guide

This document describes the **decomposed local stack**: architecture, config files, environment variables, Docker Compose, the control-plane admin API, and the React platform UI.

For runtime JSON field reference, see [Configuration](configuration.md). For bus internals, see [Architecture overview](architecture.md) and [Runtime mechanisms](mechanisms.md).

---

## Architecture at a glance

The production direction is an **event-driven, multi-process** layout connected by **Redis Streams** (local dev) or Kafka (at scale). Postgres holds the **strategy registry** and event persistence; optional Parquet files land on disk.

```text
                         ┌─────────────────────────────────────┐
                         │  Control plane (mm.api.admin :8080) │
                         │  Registry CRUD · admin commands · UI  │
                         └──────────────┬──────────────────────┘
                                        │ admin:commands
┌──────────────┐   market:events:*     │                    strategy:* / execution:*
│   Ingestor   │ ──────────────────────┼──────────────────────────────────┐
│  (exchange   │                       │                                  │
│   REST/WS)   │                       ▼                                  ▼
└──────────────┘              ┌─────────────────┐              ┌─────────────────┐
                              │  Redis Streams  │              │   Persistence   │
                              │   (bus spine)   │─────────────▶│ Postgres/Parquet│
                              └────────┬────────┘              └─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
             ┌──────▼──────┐                       ┌───────▼───────┐
             │   Worker    │  ← recommended        │   Consumer    │  ← legacy / debug
             │ mm.worker   │    strategy runtime   │ print_events  │
             │ (registry)  │                       │ (JSON config) │
             └─────────────┘                       └───────────────┘
```

| Process | Module | Role |
|---------|--------|------|
| **Ingestor** | `mm.ingestor` | Connects to exchanges; publishes `market:events:{exchange}:{symbol}:{interval_ms}` |
| **Persistence** | `mm.persistence` | Subscribes to bus topics; writes Postgres + Parquet |
| **Worker** | `mm.worker` | Loads strategies from Postgres registry; runs MPC pipeline per assignment |
| **Consumer** | `mm.tools.print_events` | Same strategy pipeline from **static JSON**; also useful as a bus debug tap |
| **Admin** | `mm.api.admin` | FastAPI control plane, registry API, serves built platform UI at `/ui/` |
| **Monolith** | `mm.runtime` | Single-process runtime (all stages in one Python process); still supported |

**Important:** Do not run **worker** and **consumer** for the same strategy (`mpc-controller` on the same joint-frame topic). They duplicate work and can interfere. Prefer **worker + skip consumer** for registry-based dev.

---

## Prerequisites

| Dependency | Purpose | Local default |
|------------|---------|---------------|
| Python 3.10+ | All runtime processes | system / venv |
| `pip install -r requirements.txt` | Python deps | repo root |
| Redis | Event bus (`redis_streams`) | `localhost:6379` via Compose |
| Postgres | Registry + event persistence | `localhost:5432` via Compose |
| Node 20+ | Platform UI dev/build | only for UI work |
| Exchange API creds | Private state / optional live orders | `.env` |

Optional observability stack (Compose): Prometheus `:9090`, Grafana `:3000`, Loki, Tempo, Alloy (log tailing).

---

## One-time setup

```bash
# From repository root
cp .env.example .env
# Edit .env: MM_DATABASE_URL, NOBITEX_API_TOKEN (or key pair), etc.

pip install -r requirements.txt

docker compose up -d    # Redis, Postgres, Grafana, Prometheus, …
```

### `.env` (repository root)

Loaded automatically by `mm.env_loader.load_mm_env()` when any service starts. **Only** `<repo>/.env` is read (not `mm/.env`).

| Variable | Required for | Description |
|----------|--------------|-------------|
| `MM_DATABASE_URL` | Worker, persistence, registry API | Postgres URL, e.g. `postgresql://meysam:Welcome123@localhost:5432/postgres` (matches Compose defaults) |
| `NOBITEX_API_TOKEN` or `NOBITEX_API_KEY` + `NOBITEX_API_PRIVATE_KEY` | Private state on Nobitex | Exchange credentials (see `.env.example`) |
| `WALLEX_API_TOKEN` | Wallex private state | Optional |
| `MM_PARQUET_ROOT` | Parquet persistence | e.g. `./data/parquet` (default if unset in writer) |
| `MM_SERVICE` | Observability labels | Default `market_maker` |
| `MM_ENV` | Observability labels | Default `local` |
| `MM_ADMIN_TOKEN` | Admin API auth | If set, UI/API require `Authorization: Bearer <token>`. If unset, auth is open (`local` user). |
| `MM_ADMIN_HOST` / `MM_ADMIN_PORT` | Admin bind | Default `0.0.0.0:8080` |
| `MM_BUS_KIND` | Admin bus | Set to `redis_streams` when stack uses Redis (default admin bus is `memory`) |
| `MM_BUS_REDIS_URL` | Admin bus | Default `redis://localhost:6379/0` |
| `MM_GRAFANA_URL` | Platform UI embeds | Default `http://localhost:3000` |
| `MM_PLATFORM_CORS_ORIGINS` | Vite dev → admin | Default `http://localhost:5173,http://127.0.0.1:5173` |
| `MM_PLATFORM_SERVICE_TARGETS` | Platform health cards | Optional JSON list overriding default probe URLs |
| `MM_TORCH_DEVICE` | MPC optimizer | e.g. `cpu` |

Per-process metrics/log paths are usually set in each service's JSON config (`observability.metrics_port`, `observability.log_path`), not globally in `.env`.

---

## Docker Compose

From repo root:

```bash
docker compose up -d      # start
docker compose ps         # status
docker compose down       # stop
```

| Service | Port | Purpose |
|---------|------|---------|
| `redis` | 6379 | Event bus |
| `db` (Postgres) | 5432 | Registry + persistence (`POSTGRES_USER=meysam`, `POSTGRES_PASSWORD=Welcome123`, db `postgres`) |
| `grafana` | 3000 | Dashboards (anonymous Viewer enabled for local iframe embed) |
| `prometheus` | 9090 | Scrapes host metrics (see below) |
| `loki` | 3100 | Log aggregation |
| `tempo` | 3200, 4317, 4318 | Tracing |
| `alloy` | 12345 | Tails `./logs/*.jsonl` into Loki |
| `redisinsight` | 5540 | Redis UI |

**Parquet** has no container. The Python persistence writer creates files under `MM_PARQUET_ROOT` on the host.

**Prometheus** scrapes processes on the **host** via `host.docker.internal` (see `docker/prometheus.yml`):

| Job | Host port | Service |
|-----|-----------|---------|
| `market_maker_admin` | 8080 | Admin API |
| `market_maker_consumer` | 8000 | Consumer (`print_events`) |
| `market_maker_ingestor` | 8001 | Ingestor |
| `market_maker_persistence` | 8002 | Persistence |
| `market_maker_worker` | 8003 | Worker |

Start the Python stack **before** expecting Prometheus targets to be up.

---

## Config files

All example configs live in `mm/configs/`.

| File | Used by | What it configures |
|------|---------|-------------------|
| `market_maker.ingestor.example.json` | `mm.ingestor` | Feeds (exchange/symbol/interval), exchange REST/WS settings, Redis bus, metrics `:8001`. Optional `"load_feeds_from_registry": true` to read feeds from Postgres. |
| `market_maker.worker.example.json` | `mm.worker` | `worker_id`, registry (`MM_DATABASE_URL`), Redis bus, supervisor backoff, metrics `:8003`. Strategy runtime comes from registry rows, not this file. |
| `market_maker.persistence.example.json` | `mm.persistence` | Topics/feeds to persist, Postgres/Parquet env refs, metrics `:8002` |
| `market_maker.runtime.consumer.example.json` | `mm.tools.print_events` | Full strategy runtime with `market_data_source: "bus"`, bot `mpc-controller`, metrics `:8000` |
| `market_maker.runtime.example.json` | Monolith / reference | Complete single-process config (ingestion + strategy + persistence in one file) |

### Shared bus settings (decomposed configs)

All decomposed services use the same pattern:

```json
"bus": {
  "kind": "redis_streams",
  "redis_url": "redis://localhost:6379/0",
  "stream_prefix": "mm",
  "consumer_group": "<unique per service>",
  "consumer_name": "<instance id>"
}
```

Each service must have a **distinct** `consumer_group` (ingestor, persistence, worker, consumer, admin each use their own).

### Worker vs consumer config

| | **Worker** | **Consumer** (`print_events`) |
|--|------------|----------------------------------|
| Strategy source | Postgres registry (`worker_id` → assignment → strategy preset) | JSON `bots` + `strategy_presets` |
| Market data | `market_data_source: "bus"` (from registry seed) | `"bus"` in JSON |
| Entry | `python3 -m mm.worker --config …` | `python3 -m mm.tools.print_events --config …` |
| When to use | Normal decomposed dev / production direction | Debug tap, JSON-only experiments, legacy |

### Registry auto-seed

When `run_local_stack` includes the worker and `MM_DATABASE_URL` is set, it calls `ensure_local_worker_registry()` which idempotently seeds:

- Worker `worker-1` / assignment `worker-1-assignment`
- Exchange `nobitex` (minimal `config`; see [Registry configuration vs monolith JSON](#registry-configuration-vs-monolith-json))
- Account `account-nobitex-main` → `credential_ref: NOBITEX_API_TOKEN`
- Market feed `nobitex` / `USDT_IRT` / `10000` ms (ingestor)
- Strategy `mpc-usdt-local`, actor `mpc-controller`, symbol `USDT_IRT` on Nobitex
- Preset `thin_local_liquidity_weighted_mpc` (minimal MPC preset)
- Allocation / virtual wallet for `mpc-usdt-local` (initial USDT / IRT balances)

Manual seed:

```bash
python3 -m mm.tools.seed_local_registry
```

---

## Registry configuration vs monolith JSON

In **Mode A** (worker + registry), strategy runtime is assembled from Postgres rows, not from a single `market_maker.runtime.example.json`. The monolith file remains the **field reference** and is still used by **Mode C/D** (`mm.runtime` / `print_events`).

### Where to define what

| Concept | Postgres registry | JSON files | `.env` |
|---------|-------------------|------------|--------|
| Exchange venue defaults (fees, private polling, bus vs local market) | `registry_exchanges.config` | `market_maker.runtime.example.json` → `exchanges.*` (monolith only) | — |
| API credentials | `registry_accounts.credential_ref` (env names or JSON with `api_token_env` / `api_key_env` / `api_private_key_env`) | monolith `exchanges.*.private.api_*_env` | actual secret values in `.env` |
| Public market polling (depth, trades, poll interval) | `registry_market_feeds` (+ ingestor JSON fallback) | `market_maker.ingestor.example.json` → `feeds[]` | — |
| Strategy logic (MPC, risk, execution) | `registry_strategy_presets` | monolith `strategy_presets.*` | — |
| Strategy instance (symbol, account, preset, actor) | `registry_strategies` | monolith `bots.*` | — |
| Virtual wallet (initial base/quote pool) | `registry_allocations` | — | — |
| Worker assignment | `registry_worker_assignments` | `market_maker.worker.example.json` → `worker_id` | — |
| Bus / metrics / scheduler defaults | `registry_platform_settings` | per-service JSON (`worker`, `ingestor`, …) | `MM_BUS_*`, `MM_ADMIN_*` |

Control-plane UI path (recommended order): **Exchanges → Accounts → Market feeds → Presets → Strategies → Wallets (allocations) → Workers**.

### UI edit patterns

| Page | Edit style | Why |
|------|------------|-----|
| Accounts, Market feeds, Wallets, Workers | **Form** | Scalar binding fields |
| Presets, Platform settings | **JSON** | Large nested runtime blobs |
| Exchanges, Strategies | **Hybrid** | Form for identity/status; JSON for nested `config` / `config_overrides` |

**Accounts credentials:** use *API token + private key* for Nobitex/Tabdeal/Wallex (maps to `api_token_env` + `api_private_key_env` in stored `credential_ref` JSON). Single-token mode maps one env var to `api_token_env`. Advanced JSON supports `api_key_env` where the venue uses API key + secret instead of token.

### Mapping `exchanges.nobitex` from the monolith example

`mm/configs/market_maker.runtime.example.json` packs everything under one exchange block. In the decomposed registry path those responsibilities split:

| Monolith field (`exchanges.nobitex`) | Registry / runtime location | Notes |
|--------------------------------------|----------------------------|--------|
| `market_data.mode` | Exchange `config.market_data.mode` | e.g. `rest` |
| `market_data.depth_limit`, `trades_limit`, `poll_interval_ms` | **Market feed** row (ingestor) | Ingestor polls the exchange and publishes `market:events:…`. Not repeated on the worker exchange row when using the bus. |
| `market_data` (worker view) | Exchange `config.market_data.market_data_source: "bus"` | Worker subscribes to Redis; it does not poll public books itself. |
| `private.api_token_env` | Account `credential_ref` | Plain env name (`NOBITEX_API_TOKEN`) or JSON `{"api_token_env":"…","api_private_key_env":"…"}`. Merged at resolve time into `exchanges.*.private`. |
| `private.api_private_key_env` | Account `credential_ref` (JSON form) | Same merge as above. |
| `private.poll_interval_ms`, `fill_event_source` | Exchange `config.private` | Still venue defaults for private-state polling. |
| `fees.maker_fee` | Exchange `config.fees` | Used when building worker runtime JSON. |
| `_comment`, `_options` | — | Documentation only; runtime ignores them. |

**Local seed is intentionally minimal** (`mm/tools/seed_local_registry.py`): it sets `market_data_source: "bus"`, basic `private` + `fees`, and relies on defaults for anything omitted. That is why the **Exchanges** UI shows fewer keys than the monolith example — not because the schema dropped fields.

Omitted exchange fields fall back to runtime defaults when the worker resolves config, for example:

| Field | Example JSON | Default if missing |
|-------|--------------|-------------------|
| `depth_limit` | 100 | 10 |
| `trades_limit` | 200 | 100 |
| `poll_interval_ms` (market) | 2000 | 1000 (mostly irrelevant when `market_data_source` is `bus`) |
| `private.poll_interval_ms` | 10000 | 5000 |

To align local dev with the example file:

1. **Exchanges → nobitex → Edit** — paste `private` and `fees` from the example (and any exchange-level `market_data` you still want on the worker).
2. **Market feeds** — set `depth_limit`, `trades_limit`, `poll_interval_ms` for the ingestor feed (seed creates `nobitex` / `USDT_IRT` / `10000` ms).
3. **Accounts** — ensure `credential_ref` matches your `.env` vars.
4. **Strategies → select strategy → runtime-config** — inspect the **fully merged** JSON the worker actually loads.

### Ingestor feeds from registry

Ingestor JSON can load `feeds[]` from Postgres instead of inline JSON:

```json
"load_feeds_from_registry": true
```

in `market_maker.ingestor.example.json` (requires `MM_DATABASE_URL`). When false, `feeds[]` in the ingestor file is used as today.

### Allocations = virtual wallets

`registry_allocations` defines an **initial base/quote wallet** per strategy instance. The MPC and risk engine rebalance **inside that pool**, not against the full exchange account balance. Fills update wallet holdings; breaches halt the actor when the wallet goes negative. Physical exchange ledger remains for reconciliation.

### Exchange status in the UI

`active` / `disabled` on an exchange row is an operator flag in the registry. **`active`** means the venue is registered and its `config` is merged into strategy runtime. **`disabled`** is reserved for future enforcement; the worker does not yet refuse strategies on disabled venues.

---

## Running locally — modes

### Mode A — Full decomposed stack (recommended)

Registry-based worker, ingestor, persistence. **Skip consumer** to avoid duplicate strategies.

```bash
docker compose up -d

python3 -m mm.tools.run_local_stack --skip consumer --with-admin
```

| Flag | Effect |
|------|--------|
| `--with-admin` | Also start admin API on `:8080` |
| `--skip consumer` | Do not start `print_events` consumer |
| `--skip worker` | Consumer-only (no registry worker) |
| `--skip ingestor` / `persistence` | Partial stack |
| `--ingestor-warmup-s 2` | Delay before non-ingestor services (default 2s) |
| `--*-config PATH` | Override config path per service |

Open:

- Control plane UI: [http://localhost:8080/ui/](http://localhost:8080/ui/) (requires `platform/ui/dist` built — see UI section)
- Admin health: [http://localhost:8080/api/admin/health](http://localhost:8080/api/admin/health)
- Grafana: [http://localhost:3000](http://localhost:3000)

### Mode B — Stack without admin

```bash
python3 -m mm.tools.run_local_stack --skip consumer
```

Run admin separately when needed:

```bash
MM_BUS_KIND=redis_streams python3 -m mm.api.admin
```

Set `MM_BUS_KIND=redis_streams` so admin commands reach the running stack on Redis.

### Mode C — Monolith (single process)

Everything in one Python process — ingestion, strategy, persistence (if enabled in JSON). No ingestor/worker split.

```bash
docker compose up -d   # still need Redis/Postgres if config uses them

python3 -m mm.runtime --config mm/configs/market_maker.runtime.example.json
# equivalent:
python3 -m mm.tools.print_events --config mm/configs/market_maker.runtime.example.json
```

Use `market_maker.runtime.example.json` as the reference; copy and trim for your bot.

### Mode D — Consumer only (JSON strategy, bus market data)

Ingestor publishes market events; consumer runs strategy from JSON (no worker/registry).

```bash
docker compose up -d

python3 -m mm.ingestor --config mm/configs/market_maker.ingestor.example.json
# wait for first market events, then:
python3 -m mm.tools.print_events --config mm/configs/market_maker.runtime.consumer.example.json
```

Or use `run_local_stack --skip worker`.

### Mode E — Bus debug tap (print events, no strategy)

Subscribe to topics and print JSON summaries. Use a config with `"controller": {"enabled": false}` or run with `--build-mode` that disables strategy, OR use `--events N` with a minimal config.

Typical debug run (with strategy enabled, prints joint frames + market events):

```bash
python3 -m mm.tools.print_events --config mm/configs/market_maker.runtime.consumer.example.json --events 20
```

### Mode F — Individual services (manual)

```bash
load_mm_env  # happens automatically on startup

python3 -m mm.ingestor      --config mm/configs/market_maker.ingestor.example.json
python3 -m mm.persistence   --config mm/configs/market_maker.persistence.example.json
python3 -m mm.worker        --config mm/configs/market_maker.worker.example.json
MM_BUS_KIND=redis_streams python3 -m mm.api.admin
```

Start **ingestor first**; worker needs registry seed + Postgres.

---

## Control plane (admin API)

Entrypoint:

```bash
python3 -m mm.api.admin
# or: python3 -m mm.api
```

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/admin/health` | No | Liveness |
| `GET /api/admin/status` | Yes* | Env/config summary |
| `GET /metrics` | No | Prometheus metrics |
| `POST /api/admin/commands` | Yes* | Publish pause/kill_switch/dry_run/etc. to `admin:commands` |
| `GET/POST /api/admin/registry/...` | Yes* | Exchanges, accounts, presets, market feeds, strategies, allocations, workers, platform settings |
| `GET /api/admin/platform/services` | Yes* | Health probes for stack services |
| `GET /api/admin/platform/grafana` | Yes* | Grafana embed URLs for UI |
| `GET /ui/` | No | Static platform UI (if `platform/ui/dist` exists) |

\*If `MM_ADMIN_TOKEN` is unset, auth is bypassed (local dev). If set, pass `Authorization: Bearer <token>` (the React UI stores this in `localStorage`).

**Admin bus:** By default the admin process uses an in-memory bus. For commands to reach running services, set:

```bash
export MM_BUS_KIND=redis_streams
export MM_BUS_REDIS_URL=redis://localhost:6379/0
```

`run_local_stack --with-admin` starts admin in the same environment as other services (load `.env` first).

---

## Platform UI (React / Vite)

Source: `platform/ui/`.

### Production-style (served by admin)

Build static assets once; admin serves them at `/ui/`:

```bash
cd platform/ui
npm ci
npm run build          # output: platform/ui/dist/
cd ../..

python3 -m mm.api.admin
# → http://localhost:8080/ui/
```

The Docker image (`Dockerfile`) runs `npm run build` in a multi-stage build and copies `dist/` into the Python image.

### Local UI development (hot reload)

Run Vite dev server; it proxies `/api` to admin on `:8080`:

```bash
# Terminal 1
MM_BUS_KIND=redis_streams python3 -m mm.api.admin

# Terminal 2
cd platform/ui
npm ci
npm run dev            # → http://localhost:5173/ui/
```

Set `MM_PLATFORM_CORS_ORIGINS` if you use a non-default Vite port.

### UI pages

| Page | Purpose |
|------|---------|
| Overview | Service health summary |
| Exchanges | Venue `config` (private polling, fees, bus vs local); view/edit JSON |
| Accounts | `credential_ref` → env var names (secrets stay in `.env`) |
| Presets | Reusable MPC / risk / execution templates |
| Market feeds | Ingestor subscriptions (`depth_limit`, `poll_interval_ms`, …) |
| Strategies | Strategy instances; runtime-config preview; start/pause |
| Wallets (allocations) | Virtual wallet initial base/quote per strategy |
| Workers | Worker assignments and lease heartbeats |
| Settings | Platform-wide bus / observability / scheduler defaults |
| Commands | Admin bus commands (pause, kill switch, …) |
| Observability | Grafana embeds + service probes |

Local dev: build `platform/ui/dist` or use Vite dev server (see above). Set **Admin token** in the sidebar if `MM_ADMIN_TOKEN` is configured.

---

## Logs and data paths

| Path | Writer |
|------|--------|
| `logs/mm-ingestor.jsonl` | Ingestor config |
| `logs/mm-worker.jsonl` | Worker config |
| `logs/mm-persistence.jsonl` | Persistence config |
| `logs/mm-runtime.jsonl` | Consumer config |
| `./data/parquet/` (typical) | Persistence when `MM_PARQUET_ROOT` set |

Alloy (Compose) tails `./logs` into Loki for Grafana.

---

## Quick reference — commands

```bash
# Infrastructure
docker compose up -d

# Recommended local stack
python3 -m mm.tools.run_local_stack --skip consumer --with-admin

# Monolith
python3 -m mm.runtime --config mm/configs/market_maker.runtime.example.json

# Registry seed (manual)
python3 -m mm.tools.seed_local_registry

# UI build
cd platform/ui && npm ci && npm run build

# Admin only
MM_BUS_KIND=redis_streams python3 -m mm.api.admin
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RegistryNotFoundError: no active worker assignment` | Postgres registry empty | Set `MM_DATABASE_URL`, run stack (auto-seed) or `seed_local_registry` |
| `strategy_supervisor_slot_error` / dict frame errors | Worker + consumer both running same topic | `--skip consumer` |
| Admin commands have no effect | Admin on memory bus | `MM_BUS_KIND=redis_streams` |
| `/ui/` 404 | UI not built | `cd platform/ui && npm run build` |
| Prometheus targets down | Python processes not running | Start stack; check ports 8000–8003, 8080 |
| Worker starts but no decisions | Missing private API token | Set `NOBITEX_*` in `.env` |
| Exchange UI shows sparse config vs `runtime.example.json` | Seed is minimal; fields split across registry | See [Registry configuration vs monolith JSON](#registry-configuration-vs-monolith-json); use Strategies → runtime-config to verify merge |
| Duplicate `--skip` ignored | Old CLI bug | Use `--skip a b` or repeated `--skip` (fixed in current tree) |

---

## Related docs

- [Configuration](configuration.md) — JSON schema details
- [Observability & operations](observability.md) — Grafana, Postgres queries
- [Recommended medium-scale architecture](scaling/recommended-medium-scale-architecture.md) — target production shape
