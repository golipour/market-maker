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

## Unit policy (Phase 2 — in progress)

- **`InstrumentSpec`** + **`UnitConverter`** (`mm_core.units.converter`): single entry for tick/lot ↔ float; **`unit_converter_for(exchange, symbol)`** for edge lookup via `trade_stats`.
- **Rounding** at the grid: `mm_core.units.rounding` (`round_price_to_tick`, `round_qty_to_lot`); `UnitConverter.price_ticks_from_quote` / `qty_lots_from_base` use these for reverse mapping.
- **Canonical order book**: `mm_core.domain.orderbook` (`CanonicalLevel`, `CanonicalOrderbook`); normalize tick/lot books with **`mm_adapters.exchanges.to_canonical_orderbook`**
  after you have a `trade_stats.Orderbook` + `InstrumentSpec`.
- **Assemblers + live orders** use `UnitConverter` so snapshot JSON and live placement share one conversion path.
- **Still tick/lot inside** the 15-field step tuple and much of `OrderFlowStepper` until Phase 3 retires the tuple.

## Contracts (Phase 3 target)

- **`InstrumentSnapshot`**, **`StepSnapshot`**, **`OrderIntent`**, **`ExecutionReport`** (`mm_core.domain`) — expand fields as tuples are retired.
- **`StateAssembler`**, **`RolloutAssembler`**: not yet implemented; engine stays duck-typed on `OrderFlowRunner` until then.

## Observability (Phase 7 target)

- Logs: structured JSON (existing `mm_adapters.observability.jsonlog`), correlation via context.
- Metrics: Prometheus registry (`METRICS`); OTLP/Mimir alignment optional later.

## Execution order (minimal risk)

1. ~~Skeleton + wrappers~~ — **Phase 1**: `mm_runtime` owns runner/stepper; root shims.
2. ~~**Phase 2**~~ **(partial / ongoing)**: `UnitConverter`, canonical orderbook DTOs, adapter `to_canonical_orderbook`, assemblers + `_execute_live_orders` on converter; step tuple unchanged.
3. **Phase 3**: DTOs + assemblers; engine consumes contracts.
4. **Phase 5**: unify reference / valuation policies in `mm_core/pricing`.
5. **Phase 4**: `MarketDataMux` after contracts exist.
6. **Phase 6**: multi-instrument / hedge policy.
7. **Phase 8**: persistence, replay, safety.

---

*Related: [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md).*
