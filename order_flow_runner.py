"""
Compatibility shim: import ``OrderFlowRunner`` from ``mm_runtime.order_flow_runner``.

Canonical implementation lives in that module so ``mm_runtime`` owns wiring-facing code.
"""

from __future__ import annotations

from mm_runtime.order_flow_runner import OrderFlowRunner

__all__ = ["OrderFlowRunner"]
