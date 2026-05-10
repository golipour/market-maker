from __future__ import annotations

import copy
import json
import os
from typing import Any

from mm.diagnostics_delta_codec import compute_patch, envelope_delta, envelope_full


def _enabled() -> bool:
    return bool(str(os.environ.get("MM_DIAGNOSTICS", "")).strip())


def _redis_url() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_URL", "redis://localhost:6379/0")).strip()


def _redis_channel() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_CHANNEL", "mm.diagnostics.v1")).strip()


def _redis_latest_key() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_LATEST_KEY", "mm_diagnostics:latest")).strip() or "mm_diagnostics:latest"


def _json_enabled() -> bool:
    return bool(str(os.environ.get("MM_DIAGNOSTICS_JSON", "1")).strip())


def _default(o: Any):
    try:
        import numpy as np

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
_last_full_frame: dict[str, Any] | None = None


def emit(frame: dict[str, Any]) -> None:
    """Emit one diagnostics frame to stdout and optional Redis latest/PubSub."""

    if not _enabled():
        return

    merged = copy.deepcopy(frame)
    step = (merged.get("point") or {}).get("step_counter")

    global _last_full_frame
    store_obj = envelope_full(step, merged)
    if _last_full_frame is not None:
        patch = compute_patch(_last_full_frame, merged)
        pub_obj = store_obj if patch is None else envelope_delta(step, patch)
    else:
        pub_obj = store_obj

    _last_full_frame = merged

    payload_pub = json.dumps(pub_obj, ensure_ascii=False, separators=(",", ":"), default=_default)
    payload_store = json.dumps(store_obj, ensure_ascii=False, separators=(",", ":"), default=_default)

    if _json_enabled():
        print(payload_pub)

    url = _redis_url()
    if not url:
        return
    global _redis_client
    try:
        if _redis_client is None:
            import redis  # type: ignore

            _redis_client = redis.Redis.from_url(url)
        _redis_client.set(_redis_latest_key(), payload_store)
        _redis_client.publish(_redis_channel(), payload_pub)
    except Exception:
        return
