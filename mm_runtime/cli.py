from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """
    Temporary entrypoint.

    For now we call the existing script entrypoint so behavior stays identical
    while we migrate code into mm_core/mm_adapters/mm_runtime incrementally.
    """
    from mm_runtime.local_env import load_local_env

    load_local_env()

    # Import lazily so module import stays lightweight.
    # Start optional local /metrics endpoint (enable with MM_METRICS_PORT=8000).
    from mm_adapters.observability.metrics import start_server

    start_server()

    from mm_runtime.run import run_order_flow_pipeline

    run_order_flow_pipeline()
    return 0

