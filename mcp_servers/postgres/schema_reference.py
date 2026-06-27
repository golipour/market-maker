"""Static schema overview for market-maker Postgres tables."""

SCHEMA_OVERVIEW = """# Market maker Postgres schema

## Event persistence (`mm/persistence/postgres.py`)

| Table | Purpose |
|-------|---------|
| `mm_events` | Raw event envelopes (topic, event_type, JSONB payload) |
| `mm_market_facts` | Denormalized market intervals: mid, spread, vwap, volumes |
| `mm_action_stage_facts` | Strategy action pipeline stages (proposed/approved/rejected orders) |
| `mm_execution_facts` | Order placement lifecycle (place/cancel/reconcile outcomes) |
| `mm_target_order_facts` | Target resting orders from the optimizer |
| `mm_model_score_facts` | Per-frame model scoring diagnostics |
| `mm_optimizer_facts` | MPC optimizer summary per decision |
| `mm_optimizer_component_facts` | Per-component optimizer breakdown |
| `mm_latest_state` | Latest snapshot per actor/key for fast lookups |
| `admin_audit_log` | Admin API command audit trail |

## Strategy registry (`mm/registry/schema.py`)

| Table | Purpose |
|-------|---------|
| `registry_exchanges` | Registered venues and capabilities |
| `registry_accounts` | Trading accounts and credential refs |
| `registry_strategy_presets` | Reusable strategy/MPC config templates |
| `registry_strategies` | Live strategy definitions (symbol, preset, exchanges) |
| `registry_worker_assignments` | Worker → strategy slot leases |
| `registry_allocations` | Virtual wallet allocations per strategy |
| `registry_market_sources` | Shared market data source registry |
| `registry_market_feeds` | Feed bindings for ingestor |
| `registry_platform_settings` | Global bus/persistence/observability settings |

## Analytics

| Table | Purpose |
|-------|---------|
| `step_economics` | Per-step PnL decomposition (spread, inventory, fees) for Grafana |

## Useful starter queries

```sql
-- Recent market intervals for a symbol
SELECT ts, exchange, symbol, mid, spread, vwap, trade_count
FROM mm_market_facts
ORDER BY ts DESC
LIMIT 20;

-- Active strategies
SELECT strategy_id, actor_id, symbol, status, quoting_exchange, liquidity_exchange
FROM registry_strategies
WHERE status = 'active';

-- Latest optimizer decisions
SELECT ts, exchange, symbol, objective, dry_run
FROM mm_optimizer_facts
ORDER BY ts DESC
LIMIT 20;
```
"""
