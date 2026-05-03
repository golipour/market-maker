"""
Compatibility shim for imports like ``from order_flow_stepper import OrderFlowStepper``.

Canonical implementation: ``mm_runtime.order_flow_stepper`` (PEP 562 lazy re-export).
"""

from __future__ import annotations

import importlib

_impl = importlib.import_module("mm_runtime.order_flow_stepper")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)))
