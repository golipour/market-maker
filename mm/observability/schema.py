"""Stable names for metrics and structured log events.

Label policy: ``service``, ``env``, exchange/symbol/actor where bounded — never
``trace_id`` on Prometheus; ``run_id`` only when ``MM_METRICS_LABEL_RUN_ID`` is enabled.
"""

# Metrics
MM_STRATEGY_FRAMES_TOTAL = "mm_strategy_frames_total"
MM_CONTROLLER_DECISIONS_TOTAL = "mm_controller_decisions_total"
MM_CONTROLLER_LATENCY_MS = "mm_controller_latency_ms"
MM_CONTROLLER_ORDER_COUNT = "mm_controller_order_count"
MM_INVENTORY = "mm_inventory"
MM_CASH = "mm_cash"
MM_EXECUTION_EVENTS_TOTAL = "mm_execution_events_total"
MM_ORDER_SYNC_TOTAL = "mm_order_sync_total"
MM_ORDER_SYNC_LATENCY_MS = "mm_order_sync_latency_ms"
MM_EXECUTION_LIVE_ORDERS = "mm_execution_live_orders"
MM_RUNTIME_ERRORS_TOTAL = "mm_runtime_errors_total"
MM_ORDER_VALIDATION_REJECTED_TOTAL = "mm_order_validation_rejected_total"
MM_STRATEGY_ACTION_ORDER_COUNT = "mm_strategy_action_order_count"

# Log event keys (mm.observability.log_event)
EVENT_STRATEGY_FRAME = "strategy_frame"
EVENT_CONTROLLER_DECISION = "controller_decision"
EVENT_ORDER_VALIDATION_REJECTED = "order_validation_rejected"
EVENT_RUNTIME_ERROR = "runtime_error"
