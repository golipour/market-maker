from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """
    Deprecated legacy entrypoint.

    Production runtime is ``mm.runtime`` / ``python -m mm.tools.print_events``.
    """
    raise SystemExit(
        "mm.layers.runtime is deprecated and no longer a production entrypoint. "
        "Use the config-driven runtime path, for example `python -m mm.tools.print_events --config <path>`."
    )

