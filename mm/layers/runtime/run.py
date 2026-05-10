from __future__ import annotations

"""
Runtime entrypoints.

The operational entrypoint is ``python -m mm.layers.runtime``. Pipeline composition is
``mm.layers.runtime.pipeline`` (``build_pipeline``, ``create_runner``, ``main``).
"""


def run_order_flow_pipeline() -> None:
    """Run the default headless pipeline (same as ``mm.layers.runtime.pipeline.main``)."""
    from mm.layers.runtime.pipeline import main

    main()

