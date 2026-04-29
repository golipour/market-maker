from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObsContext:
    service: str
    env: str


def get_context() -> ObsContext:
    """
    Shared observability context.

    - service: stable service name
    - env: environment (dev/prod/etc)
    """
    service = str(os.environ.get("MM_SERVICE", "market_maker")).strip() or "market_maker"
    env = str(os.environ.get("MM_ENV", "dev")).strip() or "dev"
    return ObsContext(service=service, env=env)

