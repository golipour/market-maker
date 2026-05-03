"""
Observability adapters (logging, metrics, tracing).

These are adapters so mm_core can stay free of vendor-specific dependencies.
"""

from mm_adapters.observability.context import ObsContext, get_context, metric_labels

__all__ = ["ObsContext", "get_context", "metric_labels"]
