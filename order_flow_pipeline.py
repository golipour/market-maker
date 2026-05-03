"""
Compatibility shim: pipeline composition lives in ``mm_runtime.pipeline``.

Import from ``mm_runtime.pipeline`` in new code.
"""

from __future__ import annotations

from mm_runtime.pipeline import build_pipeline, create_runner, main

__all__ = ["build_pipeline", "create_runner", "main"]

if __name__ == "__main__":
    main()
