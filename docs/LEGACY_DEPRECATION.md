# Legacy Surface Deprecation

The async `mm/` runtime is the target for new production features.

## Compatibility Surfaces

These modules remain available while callers migrate:

- Root `order_flow_*.py` modules
- `mm_runtime.order_flow_runner`
- `mm_runtime.order_flow_stepper`
- `market_making_controller` model/controller shims
- `mm_core` runner-oriented diagnostics and step economics hooks

## Deprecation Conditions

A legacy surface can be retired only after the `mm/` runtime has equivalent:

- market data ingestion
- private state reconciliation
- order command and execution reconciliation
- diagnostics bridge or replacement
- metrics/logs/traces
- Postgres and Parquet persistence
- replay validation
- admin safety controls

Until then, legacy modules should be treated as compatibility wrappers or
reusable implementation references, not as the destination for new production
features.
