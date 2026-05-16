# Controller and execution

## Single production path

All strategy frames flow through **`StrategyRunner`** (`mm.application.strategy_runner`).

```mermaid
flowchart TD
  F[StrategyInputFrame] --> O[live_orders_from_managers]
  O --> C[clean_strategy_frame<br/>subtract own orders]
  C --> D[MpcStrategyEngine.decide]
  D --> R[risk_engine.validate_frame]
  R --> P[Publish decision + approved actions]
  P --> G{should_execute_live?}
  G -->|yes| S[route_actions_to_managers]
  G -->|no| X[Done]
  S --> OM[OrderManager per exchange]
  OM --> V[order_passes_limits]
  V --> EX[Exchange adapter]
```

```text
StrategyInputFrame
  → clean book (subtract own live orders)
  → MPC decide
  → risk validate
  → publish
  → [gates] → OrderManager.sync → venue
```

## Layer responsibilities

| Layer | Module | Role |
|-------|--------|------|
| Frame prep | `frame_preparation.clean_strategy_frame` | Remove bot orders from primary/hedge books |
| Trading context | `trading_context` | Live orders from managers; `TradingState` for risk |
| Strategy | `MpcStrategyEngine.decide` | MPC optimize + ladder discretization |
| Risk | `RuntimeRiskEngine` / `PassThroughRiskEngine` | Kill switch, staleness, max inventory |
| Execution gates | `StrategyRunner._should_execute_live` | observe_only, warmup, control plane, manager present |
| Venue sync | `OrderManager` + `action_routing` | Cancels, dedupe, min qty/notional, REST place |

## MpcStrategyEngine

**File:** `mm/strategies/mpc_engine.py`

Builds `StrategyDecision` + `Action` from a **clean** frame (books already adjusted).

Constraints inside `decide`:

- MPC optimizer bounds and reward penalties
- Spread grid: `min_spread`, `orderbook_grid`, `qty_step`, `min_order_qty`, `max_orders_per_side`
- State book clipped to `orderbook_grid.max_spread`

## Risk engine

**Files:** `mm.risk.engine`, `mm.risk.factory`

Configured under `controller.risk` in runtime JSON:

```json
"risk": {
  "enabled": true,
  "max_inventory": 1000.0,
  "max_data_age_ms": 60000
}
```

| Rule | Effect |
|------|--------|
| `kill_switch` | Reject all actions (synced from `RuntimeControlPlane` each frame) |
| Stale data | Reject all if `now - primary.t_end_ms > max_data_age_ms` |
| `max_inventory` | Drop individual orders that would breach inventory |

Filtered actions publish `risk:events` (`RiskActionsFiltered`).

Set `"enabled": false` to use `PassThroughRiskEngine` (no risk checks).

## Execution gates (`StrategyRunner`)

After risk, live sync requires **all** of:

| Gate | Source |
|------|--------|
| Non-empty approved actions | Risk output |
| `not controller.observe_only` | JSON |
| `not control_plane.state.observe_only` | Admin command |
| `frame_step > action_start_after_steps` | JSON warm-up |
| `OrderManager` exists for primary exchange | Token + `enable_execution` |

## OrderManager (venue layer)

**File:** `mm.execution.order_manager`

Runs **after** risk. Additional checks:

- Per-exchange routing and dedupe vs live book
- `order_passes_limits` (min base / min notional from trade_stats)
- Adapter success/failure on REST

Rejected orders emit `OrderValidationRejected` on `execution:orders:{exchange}`.

## Runtime wiring

`build_runtime_components` (`mm.runtime.engine`):

1. Creates `RuntimeControlPlane` (admin commands).
2. Calls `build_strategy_runner(..., control_plane=control_plane)`.
3. Passes `order_managers` keyed by exchange.

## Debugging breakpoints

| Step | Location |
|------|----------|
| Own-order subtraction | `clean_strategy_frame` |
| MPC | `MpcStrategyEngine.decide` |
| Risk | `AbstractRiskEngine.validate_frame` |
| Live gate | `StrategyRunner._should_execute_live` |
| Venue limits | `OrderManager._place_new` |

## Related documents

- [Data flow](data-flow.md)
- [Runtime mechanisms](mechanisms.md)
- [Configuration](configuration.md)
- [Observability](observability.md)
