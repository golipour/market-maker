# System architecture (market_maker)

This document describes how the repository’s components fit together at runtime: entrypoints, the trading loop, diagnostics, observability, and supporting infrastructure. For step-by-step trading and learning mechanics (MPC, steps, execution), see [PIPELINE_AND_DATA_FLOW.md](./PIPELINE_AND_DATA_FLOW.md).

---

## 1. High-level picture

The system is a **closed loop**: exchange market data is bucketed into discrete **steps**; learned models and an **MPC controller** choose **quotes**; an **execution model** applies the previous action to the current step to update **inventory, cash, and wealth**; optional **diagnostics** serialize one JSON **frame per step** to Redis (and optionally stdout); a separate **mm_viz** web UI reads those frames and plots them in the browser.

```mermaid
flowchart TB
  subgraph exchange["Exchange"]
    REST["REST / WS feeds"]
  end

  subgraph ingest["Market data"]
    TS["trade_stats / providers"]
    ST["OrderFlowStepper"]
  end

  subgraph brain["Models & control"]
    MOF["Torch MOF / price models"]
    MPC["MonteCarloMPCTensorBatched + Controller"]
    EX["Execution models"]
  end

  subgraph loop["Orchestration"]
    RUN["OrderFlowRunner.advance"]
    ENG["mm_core.app.engine.advance_order_flow_runner"]
  end

  subgraph diag["Diagnostics"]
    FV1["build_diagnostics_frame_v1"]
    EM["emit → Redis SET + PUBLISH"]
  end

  subgraph obs["Observability"]
    LOG["jsonlog / context"]
    MET["Prometheus /metrics"]
  end

  subgraph ui["Browser UI"]
    VIZ["mm_viz: FastAPI + WebSocket + static Plotly/Chart.js"]
  end

  REST --> TS
  TS --> ST
  ST --> RUN
  MOF --> RUN
  MPC --> RUN
  EX --> RUN
  RUN --> ENG
  ENG --> FV1
  FV1 --> EM
  EM -->|Pub/Sub + key| RS[(Redis)]
  RS --> VIZ
  ENG --> LOG
  ENG --> MET
```

**Process boundaries:** The market maker usually runs as a **Python process on the host** (or a custom container not defined in the default `docker-compose.yml`). **Redis, mm_viz, Prometheus, Grafana, Mimir, Postgres** are separate services, typically started with Docker Compose.

---

## 2. Repository layout (logical packages)

| Layer / area | Role | Notable paths |
|----------------|------|----------------|
| **Runtime shell** | CLI and future composition root | `mm_runtime/` — `cli.py`, `run.py` (delegates to `order_flow_pipeline.main`), `wiring.py` (`create_stepper` → `OrderFlowStepper`) |
| **Core (refactor target)** | Orchestration extract, domain types, diagnostics schema | `mm_core/app/engine.py` — one step of `OrderFlowRunner`; `mm_core/diagnostics/frame_v1.py` — `build_diagnostics_frame_v1`; `mm_core/domain/`, `mm_core/ports/`, `mm_core/units/`, `mm_core/legacy/step_v1.py` |
| **Adapters** | I/O and side effects | `mm_adapters/diagnostics/emitter.py` — stdout + Redis; `mm_adapters/observability/` — metrics, JSON log, request context |
| **Controller / models** | PyTorch models, MPC, state, rewards | `market_making_controller/` — `controller.py`, `mpc_tensor_batched.py`, `models/`, `state.py`, `reward_tensor.py` |
| **Legacy loop (still canonical)** | Stepper, runner, pipeline, animation | `order_flow_stepper.py`, `order_flow_runner.py`, `order_flow_pipeline.py`, `order_flow_animation.py`, `trade_stats.py` |
| **Viz** | Realtime debug UI | `mm_viz/server.py` (FastAPI, WebSocket, serves static files), `mm_viz/static/` (HTML/JS/Plotly/Chart.js), `mm_viz/mpl_figures.py` (reference / optional PNG) |

The **active trading path** still runs through **legacy modules**; `mm_core.app.engine` is the **first cut** of moving orchestration into `mm_core` while preserving behavior.

---

## 3. Runtime entrypoints

| Entry | Command / module | Behavior |
|--------|------------------|----------|
| **Primary (documented in practice)** | `python -m mm_runtime` or `python3 -m mm_runtime` | `mm_runtime/cli.py` dispatches; pipeline mode calls `order_flow_pipeline.main()` via `mm_runtime.run.run_order_flow_pipeline()` |
| **Legacy direct** | `python order_flow_pipeline.py` | Same pipeline without the `mm_runtime` wrapper |
| **Package** | `mm_runtime/__main__.py` → `cli.main()` | Same as `-m mm_runtime` |

Configuration is typically loaded from **environment** (e.g. `.env` via `python-dotenv` where used) and/or **kwargs** passed into pipeline construction inside `order_flow_pipeline`.

---

## 4. Trading loop (single step)

The structural sequence is:

1. **`OrderFlowStepper.advance(t_end_ms)`** fetches order book and trades for a time bucket, outputs a **Step** (or `None` on failure).
2. **`OrderFlowRunner`** (driven from `order_flow_pipeline` / animation) calls into **`mm_core.app.engine.advance_order_flow_runner`** for the refactored path, which:
   - Parses the step to internal structures (`parse_step`, optional **step snapshot** for typed units).
   - Runs **execution** of the *previous* action against the *current* step (inventory, PnL, histories).
   - Runs **testers / diagnostics MC** (e.g. expected mid), **MPC** to choose the **next** action, updates histories (`delta_*`, rewards, J components, etc.).
3. At the end of a successful step, **`build_diagnostics_frame_v1(runner, exchange, symbol, spread_max)`** assembles a large JSON-serializable **DiagnosticsFrameV1** (runner series, depth, action_depth, price diagnostics, LOF, anchor, loss specs, etc.).
4. **`emit_diagnostics(frame)`** (adapters) may print JSON and/or write to **Redis** (see §6).

For **full narrative** of delayed execution, MPC vs diagnostic sampling, and model roles, see [PIPELINE_AND_DATA_FLOW.md](./PIPELINE_AND_DATA_FLOW.md).

---

## 5. Diagnostics pipeline

### 5.1 Frame builder

- **`mm_core.diagnostics.frame_v1.build_diagnostics_frame_v1`** reads **duck-typed `runner`** state: histories on the runner, model plot hooks (`get_plot_data`, etc.), and embeds everything needed to reproduce desktop-style figures (Figure 1–6 style data).

### 5.2 Emitter (`mm_adapters.diagnostics.emitter`)

Controlled by environment:

| Variable | Purpose |
|----------|---------|
| **`MM_DIAGNOSTICS`** | Must be non-empty (e.g. `1`) to emit anything. |
| **`MM_DIAGNOSTICS_JSON`** | If set (default-friendly), also prints one JSON line per frame to **stdout**. |
| **`MM_DIAGNOSTICS_REDIS_URL`** | Redis URL (e.g. `redis://127.0.0.1:6379/0` on host, `redis://redis:6379/0` inside Compose network). |
| **`MM_DIAGNOSTICS_CHANNEL`** | Pub/Sub channel (default `mm.diagnostics.v1`). |
| **`MM_DIAGNOSTICS_REDIS_LATEST_KEY`** | String key for **`SET`** of last payload (default `mm_diagnostics:latest`) so consumers need not rely on Pub/Sub timing. |

Wire format is **`mm.diag.v2`**: **`SET`** stores a **`kind: full`** envelope (merged frame in **`frame`**). **`PUBLISH`** sends **`kind: full`** or **`kind: delta`** (patch vs previous frame). On each emit: **`SET`** latest key, then **`PUBLISH`** channel. Failures are swallowed (best-effort).

### 5.3 Realtime UI (`mm_viz`)

- **`mm_viz/server.py`**: **FastAPI** app with:
  - **Lifespan**: background **async Redis** loop — periodic **`GET`** of latest key + **`SUBSCRIBE`** to Pub/Sub; deduplicates by **`point.step_counter`**; maintains an in-memory deque of frames (**`MM_VIZ_MAX_FRAMES`**).
  - **`WebSocket /ws`**: pushes **`{type:"frame", frame: {...}}`** or compact **`{type:"frame_delta", patch: ...}`** when the Pub/Sub message is a delta; initial snapshot if already buffered.
  - **REST**: `/api/latest`, `/api/config`, `/api/history_steps`; optional **`/api/figures/*.png`** (matplotlib) for API/debug — **main UI uses client-side Plotly/Chart.js**, not PNG refresh.
- **Static SPA** (`mm_viz/static/`): **Chart.js** for overview/price/QQ/anchor line charts; **Plotly.js** for Action & Depth and LOF panels (built from the same frame JSON in `plotly_action_lof.js`, aligned with `mpl_figures.py`).

**Important:** If **`mm_viz` runs in Docker** and the market maker on the **host**, both must use a **Redis URL that resolves from each process** (e.g. host → `127.0.0.1`, container → `redis` service name).

---

## 6. Observability (non-diagnostics)

| Mechanism | Location | Activation |
|-----------|----------|------------|
| **Prometheus metrics** | `mm_adapters/observability/metrics.py` | **`MM_METRICS_PORT`** set (e.g. `8000`) exposes **`/metrics`** HTTP in a background thread. |
| **Scrape config** | `prometheus.yml` | Scrapes **`host.docker.internal:8000`** for job `market_maker` (host-run process). |
| **Structured / JSON log** | `mm_adapters/observability/jsonlog.py` | Used from engine via **`log_emit`** where wired. |
| **Context** | `mm_adapters/observability/context.py` | Correlation / metadata for logs or metrics. |

Grafana/Mimir/Prometheus in **`docker-compose.yml`** form a **metrics stack**; the trading app is **not** required to run inside Compose for metrics if the host exposes `MM_METRICS_PORT`.

---

## 7. Docker Compose topology (repo default)

Services (from `docker-compose.yml`):

| Service | Port(s) | Role |
|---------|---------|------|
| **redis** | 6379 | Diagnostics transport (SET + Pub/Sub). |
| **db** | 5432 | Postgres (application DB if used by future features). |
| **prometheus** | 9090 | Scrapes `/metrics` from host (see `prometheus.yml`). |
| **mimir** | 9009 | Long-term metrics backend (configured via `mimir.yaml`). |
| **grafana** | 3000 | Dashboards; provisions from `./grafana/`. |
| **mm_viz** | 8501 | FastAPI + static diagnostics UI; **`depends_on: redis`**. |

**Volumes:** persistent data for Redis, Postgres, Prometheus, Mimir, Grafana.

To run **only Redis + `mm_viz`** (no DB/metrics containers):  
`docker compose up -d redis mm_viz`

**Rebuild note:** The **`mm_viz` image** must be **rebuilt** after UI/code changes (`docker compose build mm_viz`) or Compose will reuse an old image.

---

## 8. Data shapes (conceptual)

- **Step** (time bucket): order book + buy/sell volumes in lots; see pipeline doc.
- **Diagnostics frame**: single JSON object per step, including **`point`** (e.g. `step_counter`), **`runner_series`** (full histories as arrays), **`depth`**, **`action_depth`**, **`price_diag`**, **`lof_*`**, **`anchor_diag`**, etc. The UI **tails** long series (e.g. last 500 points) for performance.

---

## 9. Extension points and migration notes

- **`mm_core.ports.market_data.StepperPort`** / **`mm_runtime.wiring.create_stepper`**: intended to replace direct `OrderFlowStepper` construction with a cleaner factory (currently still returns **`OrderFlowStepper`**).
- **`mm_core.app.engine.advance_order_flow_runner`**: central place to add cross-cutting behavior (metrics, tracing) without editing `order_flow_runner.py` everywhere.
- **Desktop animation**: `order_flow_animation.py` remains the reference for figure layout; **`mm_viz/mpl_figures.py`** (matplotlib) and **`mm_viz/static/plotly_action_lof.js`** mirror that behavior for server PNG or browser Plotly, respectively.

---

## 10. Related documents

| Document | Content |
|----------|---------|
| [PIPELINE_AND_DATA_FLOW.md](./PIPELINE_AND_DATA_FLOW.md) | Steps, execution delay, MPC vs diagnostic MC, model roles. |
| [MPC_MARKET_MAKER_EXPLANATION.md](./MPC_MARKET_MAKER_EXPLANATION.md) | MPC mathematics (if present). |
| [LOF_CONDITIONAL_BSPLINE_MATH.md](./LOF_CONDITIONAL_BSPLINE_MATH.md) | LOF model math (if present). |

---

## 11. Environment quick reference (diagnostics + viz)

| Variable | Typical value | Notes |
|----------|-----------------|-------|
| `MM_DIAGNOSTICS` | `1` | Enables frame build + emit. |
| `MM_DIAGNOSTICS_REDIS_URL` | `redis://127.0.0.1:6379/0` (host) | Must match where Redis listens. |
| `MM_DIAGNOSTICS_CHANNEL` | `mm.diagnostics.v1` | Must match `mm_viz` / emitter. |
| `MM_DIAGNOSTICS_REDIS_LATEST_KEY` | `mm_diagnostics:latest` | GET key for last frame. |
| `MM_VIZ_MAX_FRAMES` | `2000` | Server-side ring buffer of frames (viz). |
| `MM_METRICS_PORT` | `8000` | Enables Prometheus exporter on that port. |

This architecture is **intentionally incremental**: legacy modules carry the full trading behavior; **`mm_core`** and **`mm_adapters`** carve out stable boundaries for orchestration, domain typing, and side effects.
