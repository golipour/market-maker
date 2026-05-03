# Architecture RFC — boundaries, naming, unit policy

This document locks the **target shape** of the system (hexagonal: **ports in `mm_core`**, **adapters in `mm_adapters`**, **composition in `mm_runtime`**). Implementation is incremental; see **Execution order** at the end.

## Layering

| Package | Owns | Must not |
|--------|------|----------|
| **`mm_core`** | Domain DTOs, engine orchestration, units, pricing *policies* (pure) | Direct HTTP/WS, exchange SDKs, env-specific config |
| **`mm_adapters`** | Exchanges, transports, persistence, observability exporters, diagnostics emit | Trading policy, MPC, domain rules |
| **`mm_runtime`** | CLI, wiring, `OrderFlowRunner` / `OrderFlowStepper` *integration* (current home for moved modules) | New domain types without a home in `mm_core` first |

**Root scripts** (`order_flow_*.py` at repo top): **compatibility shims** only; new code should import from **`mm_runtime`**, **`mm_core`**, **`mm_adapters`**.

## Naming

- **Runner / stepper**: `OrderFlowRunner` and `OrderFlowStepper` live under **`mm_runtime.order_flow_*`**. Root modules re-export.
- **Step tuple**: venue-native 15-field tuple until Phase 3; parsed view: `mm_core.app.parsed_step.ParsedStep` and `parse_step`.
- **Diagnostics JSON**: frame `schema: "DiagnosticsFrame"`; wire deltas: `mm.diag.v2` (see `mm_adapters.diagnostics`).

## Unit policy (Phase 2 — complete; tuple deferred)

- **`InstrumentSpec`** + **`UnitConverter`** (`mm_core.units.converter`): single entry for tick/lot ↔ float; **`unit_converter_for(exchange, symbol)`** for edge lookup via `trade_stats`.
- **Rounding** at the grid: `mm_core.units.rounding` (`round_price_to_tick`, `round_qty_to_lot`); `UnitConverter.price_ticks_from_quote` / `qty_lots_from_base` use these for reverse mapping.
- **Canonical order book**: `mm_core.domain.orderbook` (`CanonicalLevel`, `CanonicalOrderbook`); normalize tick/lot books with **`mm_adapters.exchanges.to_canonical_orderbook`**
  after you have a `trade_stats.Orderbook` + `InstrumentSpec`.
- **Assemblers + live orders** use `UnitConverter` so snapshot JSON and live placement share one conversion path.
- **Deferred:** retire the 15-field step tuple end-to-end (`OrderFlowStepper`, MOF/price `update`, execution simulators) in favor of canonical DTOs — out of scope for “complete” here.

## Contracts (Phase 3 — complete; tuple + MOF internals deferred)

- **DTOs** (`mm_core.domain`): **`InstrumentSnapshot`**, **`StepSnapshot`** (includes typed **`main_instrument`**), **`OrderIntent`**, **`ExecutionReport`**, **`MarketEvent`**.
- **`StateAssembler`** (`mm_core.app.state_assembler`): **ParsedStep → `StepSnapshot`**; **tuple → controller `State`** via **`controller_state()`** (engine uses this instead of calling `state_from_step` directly).
- **`RolloutAssembler`** + **`RolloutFeatures`**: exposed on diagnostics frames as **`point.rollout_features`** (`mm_core.diagnostics.diagnostics_frame`); MPC optimizer remains inside `market_making_controller`.
- **Deferred:** optional canonical views on MOF/price `update()`; richer **`ExecutionReport`** fields per venue; removal of tuple from **`state_after_step`** / MOF paths.
- **Live path**: `place_limit_order_lots_with_report` + JSON **`mm.execution`**; MOF updates via **`update_flow_models_after_step`**.

## Observability (Phase 7 — complete)

- Logs: structured JSON (`mm_adapters.observability.jsonlog.emit`) merges `service`, `env`, optional `trace_id` / `run_id` from `get_context()` (`MM_TRACE_ID`, `MM_RUN_ID`).
- Metrics: Prometheus registry (`REGISTRY`); `metric_labels()` adds `service`, `env`, optional `run_id`, optional `trace_id` only if `MM_METRICS_LABEL_TRACE=1`. OTLP/Mimir push remains optional.

## Execution order (minimal risk)

1. ~~Skeleton + wrappers~~ — **Phase 1**: `mm_runtime` owns runner/stepper; root shims.
2. ~~**Phase 2**~~ — **complete** (unit/orderbook/assembler path); **deferred:** tuple retirement.
3. ~~**Phase 3**~~ — **complete** (DTOs, assembler, engine, rollout features in diagnostics); **deferred:** tuple/MOF internals.
4. ~~**Phase 5**~~ — **complete**: depth/valuation policies + **`mm_core.pricing`** facade over observed VWAPs (`reference_prices_*` re-export `ob_reference_prices`); implementation stays in one module for now.
5. ~~**Phase 4**~~ — **complete**: `MarketDataMux` + `mm_runtime/market_data_providers.py`; ports in `mm_core/ports/market_data.py`.
6. ~~**Phase 6**~~ — **complete** for **single hedge venue** (`HedgePolicy`, MPC weight/fee); **deferred:** multi-leg MPC tensors.
7. ~~**Phase 7**~~ — **complete** (`ObsContext`, JSON + Prometheus correlation).
8. **Phase 8**: persistence, replay, safety.

---

*Related: [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md).*
