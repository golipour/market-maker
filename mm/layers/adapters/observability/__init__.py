"""
Observability adapters (logging, metrics, tracing).

These are adapters so mm.layers.core can stay free of vendor-specific dependencies.
"""

from mm.layers.adapters.observability.context import ObsContext, get_context, metric_labels

__all__ = ["ObsContext", "get_context", "metric_labels"]
