from __future__ import annotations

"""
Runtime entrypoints.

The operational entrypoint is ``python -m mm.layers.runtime``. Pipeline composition is
``mm.layers.runtime.pipeline`` (``build_pipeline``, ``create_runner``, ``main``).
"""


def run_order_flow_pipeline() -> None:
    """Deprecated legacy runtime shim."""

    raise RuntimeError(
        "mm.layers.runtime is deprecated. Use the config-driven mm.runtime path instead."
    )

