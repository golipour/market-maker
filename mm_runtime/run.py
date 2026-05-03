from __future__ import annotations

"""
Runtime entrypoints.

The operational entrypoint is ``python -m mm_runtime``. Pipeline composition is
``mm_runtime.pipeline`` (``build_pipeline``, ``create_runner``, ``main``).
"""


def run_order_flow_pipeline() -> None:
    """Run the default headless pipeline (same as ``mm_runtime.pipeline.main``)."""
    from mm_runtime.pipeline import main

    main()

