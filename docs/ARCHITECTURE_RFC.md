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
- **Step**: canonical type is **`market_making_controller.models.flow_implied.Step`** — a 15-field **`NamedTuple`** (tuple-compatible for legacy unpack/index). Parsed view: **`mm_core.app.parsed_step.ParsedStep`** and **`parse_step`**.
- **Diagnostics JSON**: frame `schema: "DiagnosticsFrame"`; wire deltas: `mm.diag.v2` (see `mm_adapters.diagnostics`).

## Unit policy (Phase 2 — complete)

- **`InstrumentSpec`** + **`UnitConverter`** (`mm_core.units.converter`): single entry for tick/lot ↔ float; **`unit_converter_for(exchange, symbol)`** for edge lookup via `trade_stats`.
- **Rounding** at the grid: `mm_core.units.rounding` (`round_price_to_tick`, `round_qty_to_lot`); `UnitConverter.price_ticks_from_quote` / `qty_lots_from_base` use these for reverse mapping.
- **Canonical order book**: `mm_core.domain.orderbook` (`CanonicalLevel`, `CanonicalOrderbook`); normalize tick/lot books with **`mm_adapters.exchanges.to_canonical_orderbook`**
  after you have a `trade_stats.Orderbook` + `InstrumentSpec`.
- **Assemblers + live orders** use `UnitConverter` so snapshot JSON and live placement share one conversion path.

## Contracts (Phase 3 — complete)

- **DTOs** (`mm_core.domain`): **`InstrumentSnapshot`**, **`StepSnapshot`**, **`OrderIntent`**, **`ExecutionReport`** (includes optional **`venue_metadata`**), **`MarketEvent`**.
- **`StateAssembler`**: **ParsedStep → `StepSnapshot`**; **tuple/`Step` → controller `State`** via **`controller_state()`**.
- **`RolloutAssembler`**: **`RolloutFeatures`** on diagnostics **`point.rollout_features`**.
- **MOF / price `update`**: receive optional **`parsed_prev`** / **`parsed_curr`** (`ParsedStep`) from **`update_flow_models_after_step`** and engine price path.
- **Live path**: `place_limit_order_lots_with_report` + JSON **`mm.execution`**; MOF updates via **`update_flow_models_after_step`**.

## Pricing (Phase 5 — complete)

- Depth / valuation: **`mm_core.pricing`** (`policies`, `depth`, `valuation`); observed VWAP entry points **`reference_prices_*`** (implementation in **`ob_reference_prices`**, imported only via **`mm_core.pricing`** at call sites).

## Market data (Phase 4 — complete)

- **`MarketDataMux`** + **`mm_runtime/market_data_providers.py`**; ports **`TradeDataProvider`** / **`OrderbookDataProvider`** in **`mm_core/ports/market_data.py`**.

## Hedge (Phase 6 — complete)

- **`mm_core/hedge/policy.HedgePolicy`**: maps to MPC fees / **`anchor_hedge_reward_weight`**. **`MonteCarloMPCTensorBatched`** supports **one** simulated hedge leg; multiple hedge legs in policy would require tensor work later.

## Observability (Phase 7 — complete)

- Logs: **`jsonlog.emit`** + **`ObsContext`** (`MM_TRACE_ID`, `MM_RUN_ID`).
- Metrics: **`metric_labels()`** + **`REGISTRY`** (`MM_METRICS_LABEL_TRACE` optional).

## Persistence & replay (Phase 8 — partial)

- **Journal**: set **`MM_STEP_JOURNAL_PATH`** → **`mm_core.app.step_journal.append_step_journal`** writes compact JSON lines per step (scalars + ob mid; not full books).
- **Replay / safety**: full deterministic replay from journal alone is **not** implemented (needs stored books or venue replay APIs).

## Execution order (minimal risk)

1. ~~Skeleton + wrappers~~ — **Phase 1**: `mm_runtime` owns runner/stepper; root shims.
2. ~~**Phase 2**~~ — **complete**.
3. ~~**Phase 3**~~ — **complete**.
4. ~~**Phase 5**~~ — **complete**.
5. ~~**Phase 4**~~ — **complete**.
6. ~~**Phase 6**~~ — **complete** (single hedge leg in MPC).
7. ~~**Phase 7**~~ — **complete**.
8. **Phase 8** — **partial**: on-disk step journal; **deferred:** durable full replay + safety controls.

---

*Related: [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md).*
