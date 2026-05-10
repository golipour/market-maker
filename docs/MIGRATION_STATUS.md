# Market Maker Migration Status

The production target for new work is the async runtime under `mm/`.
`mm.layers.core`, `mm.layers.adapters`, `mm.layers.runtime`, and `market_making_controller` remain
compatibility and library surfaces until equivalent `mm/` runtime features are
available.

## Runtime Boundary

| Capability | `mm/` async runtime | `mm.layers.core` / `mm.layers.runtime` path | Migration target |
| --- | --- | --- | --- |
| Public market data | Config-driven symbol engines and exchange adapters | Market data mux/providers | Keep in `mm/` |
| Private state | Private reconcilers and exchange private adapters | Legacy runner state | Keep in `mm/` |
| Strategy/controller | `MpcControllerRunner` over `StrategyInputFrame` | `OrderFlowRunner` / controller step loop | Keep in `mm/` |
| Event bus | `InMemoryBus`; Redis Streams added as optional durable bus | Diagnostics Redis Pub/Sub | `mm.bus` |
| Diagnostics | Controller decisions bridge to `mm.diagnostics_emitter` | `mm.diag.v2` legacy payload shape | Keep sinks in `mm/`; external UI optional |
| Metrics/logging | Wired into `mm/` runtime | Legacy adapters re-export `mm` helpers | Keep in `mm.observability_*` |
| Postgres | Event writer target | `step_economics` only | Add append-only event tables |
| Parquet/replay | New writer/reader target | Compact JSONL journal only | Add full replay source |
| Admin UI | New authenticated control plane | Read-only diagnostics UI | Add separate control APIs |

## Topic Contract

The following topics are treated as stable integration points for persistence,
observability, replay, and admin daemons:

- `market:events:<exchange>:<symbol>:<interval_ms>`
- `strategy:frames:<exchange>:<symbol>:<interval_ms>`
- `strategy:controller`
- `strategy:actions`
- `execution:orders:<exchange>`
- `execution:state:<exchange>`
- `private:state:<exchange>:snapshot`
- `admin:commands`
- `admin:events`
- `risk:events`

Durable buses and persistence writers wrap these payloads in the versioned
`mm.event` envelope defined in `mm.core.events`.

## Clean Runtime Event Flow

The `mm/` runtime now treats Redis Streams as a durable external log, not as the
in-process object transport. Runtime subscribers receive typed Python objects;
Redis receives compact `mm.event` envelopes suitable for persistence, replay,
and admin readers.

Important durable topics:

- raw public market data: `market:raw:*`
- aggregated market events: `market:events:*`
- strategy frames and controller decisions: `strategy:frames:*`, `strategy:controller`
- strategy actions and execution state: `strategy:actions`, `execution:*`
- private balances/orders/fills: `private:state:*`
- admin and risk events: `admin:*`, `risk:events`

Large private snapshots are compacted before they are written to Redis Streams:
zero balances are omitted, non-zero balances are retained, and open orders are
kept. The full typed snapshot remains available inside the process for
execution reconciliation.

Postgres persistence is enabled when `persistence.enabled` is true in runtime
config, or automatically when `MM_DATABASE_URL` is present and no persistence
block is supplied. The writer creates `mm_events`, `mm_latest_state`, and
`admin_audit_log` at startup.

Prometheus metrics use the `mm_` prefix and are exposed on `MM_METRICS_PORT`
from the runtime, and also from the admin API at `/metrics`. JSON runtime logs
are written to `MM_LOG_PATH` so Alloy can ship them to Loki.

## Migration Rule

New production features should be reachable from `mm/`. Legacy modules may keep
compatibility shims, dashboards, and proven implementation details, but new
storage, bus, observability, admin, and replay work should not be implemented
only against the old runner path.
