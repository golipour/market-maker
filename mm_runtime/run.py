from __future__ import annotations

"""
New runtime entrypoints.

This module gradually replaces legacy scripts like `order_flow_pipeline.py` by
delegation, so the operational entrypoint becomes `mm_runtime`.
"""


def run_order_flow_pipeline() -> None:
    """
    Compatibility runner: delegates to legacy `order_flow_pipeline.main()`.

    Next migrations will move composition/wiring here and shrink the legacy module.
    """
    import order_flow_pipeline

    order_flow_pipeline.main()

