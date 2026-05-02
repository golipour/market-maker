from __future__ import annotations

import copy
import json
import os
from typing import Any

from mm_adapters.diagnostics.delta_codec import (
    compute_patch,
    envelope_delta,
    envelope_full,
)


def _enabled() -> bool:
    return bool(str(os.environ.get("MM_DIAGNOSTICS", "")).strip())


def _redis_url() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_URL", "redis://localhost:6379/0")).strip()


def _redis_channel() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_CHANNEL", "mm.diagnostics.v1")).strip()


def _redis_latest_key() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_LATEST_KEY", "mm_diagnostics:latest")).strip() or "mm_diagnostics:latest"


def _json_enabled() -> bool:
    # If MM_DIAGNOSTICS=1 and no redis is configured, stdout JSON is still useful.
    return bool(str(os.environ.get("MM_DIAGNOSTICS_JSON", "1")).strip())


def _delta_enabled() -> bool:
    """Smaller Pub/Sub (and stdout) payloads: v2 envelope with patch vs previous frame."""
    return bool(str(os.environ.get("MM_DIAGNOSTICS_DELTA", "")).strip())


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
_last_full_frame: dict[str, Any] | None = None


def emit(frame: dict[str, Any]) -> None:
    """
    Emit one diagnostics frame (already JSON-serializable) to:
    - stdout JSON (optional; MM_DIAGNOSTICS_JSON=1) — same payload as Redis Pub/Sub
    - Redis ``SET`` on a latest-frame key — **always** the merged **full** frame (legacy shape)
    - Redis PubSub — **full** or **delta** envelope (``MM_DIAGNOSTICS_DELTA=1``) for smaller fan-out
    """
    if not _enabled():
        return

    merged = copy.deepcopy(frame)
    step = (merged.get("point") or {}).get("step_counter")

    global _last_full_frame
    pub_obj: dict[str, Any]
    if _delta_enabled() and _last_full_frame is not None:
        patch = compute_patch(_last_full_frame, merged)
        if patch is None:
            pub_obj = envelope_full(step, merged)
        else:
            pub_obj = envelope_delta(step, patch)
    else:
        pub_obj = envelope_full(step, merged)

    _last_full_frame = merged

    payload_pub = json.dumps(pub_obj, ensure_ascii=False, separators=(",", ":"), default=_default)
    payload_full = json.dumps(merged, ensure_ascii=False, separators=(",", ":"), default=_default)

    if _json_enabled():
        print(payload_pub)

    # Redis publish is best-effort.
    url = _redis_url()
    if not url:
        return
    global _redis_client
    try:
        if _redis_client is None:
            import redis  # type: ignore

            _redis_client = redis.Redis.from_url(url)
        # Late subscribers / GET always see a complete merged frame (no schema envelope).
        _redis_client.set(_redis_latest_key(), payload_full)
        _redis_client.publish(_redis_channel(), payload_pub)
    except Exception:
        return
