# Market Maker Migration Status

The production target for new work is the async runtime under `mm/`.
`mm_core`, `mm_adapters`, `mm_runtime`, and `market_making_controller` remain
compatibility and library surfaces until equivalent `mm/` runtime features are
available.

## Runtime Boundary

| Capability | `mm/` async runtime | `mm_core` / `mm_runtime` path | Migration target |
| --- | --- | --- | --- |
| Public market data | Config-driven symbol engines and exchange adapters | Market data mux/providers | Keep in `mm/` |
| Private state | Private reconcilers and exchange private adapters | Legacy runner state | Keep in `mm/` |
| Strategy/controller | `MpcControllerRunner` over `StrategyInputFrame` | `OrderFlowRunner` / controller step loop | Keep in `mm/` |
| Event bus | `InMemoryBus`; Redis Streams added as optional durable bus | Diagnostics Redis Pub/Sub | `mm.bus` |
| Diagnostics UI | Controller decisions need bridge/consumer support | `mm.diag.v2` used by `mm_viz` | Bridge first, then admin |
| Metrics/logging | Being wired into `mm/` runtime | Existing Prometheus text registry and JSON logs | Reuse `mm_adapters` |
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

## Migration Rule

New production features should be reachable from `mm/`. Legacy modules may keep
compatibility shims, dashboards, and proven implementation details, but new
storage, bus, observability, admin, and replay work should not be implemented
only against the old runner path.
