from __future__ import annotations

"""
New runtime entrypoints.

The operational entrypoint is `mm_runtime` (``python -m mm_runtime``). Runner and
stepper implementations live under ``mm_runtime.order_flow_*``; ``order_flow_pipeline``
still provides composition until wiring migrates here.
"""


def run_order_flow_pipeline() -> None:
    """
    Delegates to ``order_flow_pipeline.main()`` until pipeline construction moves
    into this package.
    """
    import order_flow_pipeline

    order_flow_pipeline.main()

