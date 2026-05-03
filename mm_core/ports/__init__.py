"""
Ports (interfaces) for hexagonal architecture.

Core logic depends on these Protocols; adapters implement them.
"""

from mm_core.ports.observability import (
    DiagnosticsEmit,
    MetricLabelsFn,
    MetricsRegistryPort,
    StructuredLogEmit,
)

__all__ = [
    "DiagnosticsEmit",
    "MetricLabelsFn",
    "MetricsRegistryPort",
    "StructuredLogEmit",
]
