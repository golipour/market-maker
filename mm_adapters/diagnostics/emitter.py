from __future__ import annotations

import json
import os
from typing import Any


def _enabled() -> bool:
    return bool(str(os.environ.get("MM_DIAGNOSTICS", "")).strip())


def _redis_url() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_URL", "redis://localhost:6379/0")).strip()


def _redis_channel() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_CHANNEL", "mm.diagnostics.v1")).strip()


def _json_enabled() -> bool:
    # If MM_DIAGNOSTICS=1 and no redis is configured, stdout JSON is still useful.
    return bool(str(os.environ.get("MM_DIAGNOSTICS_JSON", "1")).strip())


def _default(o: Any):
    # Numpy scalars/arrays → python types.
    try:
        import numpy as np  # local import: keep core clean

        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return str(o)


_redis_client = None


def emit(frame: dict[str, Any]) -> None:
    """
    Emit one diagnostics frame (already JSON-serializable) to:
    - stdout JSON (optional; MM_DIAGNOSTICS_JSON=1)
    - Redis PubSub (optional; MM_DIAGNOSTICS_REDIS_URL set; requires `redis` package)
    """
    if not _enabled():
        return

    payload = json.dumps(frame, ensure_ascii=False, separators=(",", ":"), default=_default)

    if _json_enabled():
        print(payload)

    # Redis publish is best-effort.
    url = _redis_url()
    if not url:
        return
    global _redis_client
    try:
        if _redis_client is None:
            import redis  # type: ignore

            _redis_client = redis.Redis.from_url(url)
        _redis_client.publish(_redis_channel(), payload)
    except Exception:
        return

