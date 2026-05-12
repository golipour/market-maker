from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

from mm.observability.context import get_context


def _is_enabled() -> bool:
    v = str(os.environ.get("MM_JSON_LOG", "")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def emit(event: str, fields: dict[str, Any] | None = None) -> None:
    """Emit one structured JSON log line to stdout when ``MM_JSON_LOG`` is truthy."""

    if not _is_enabled():
        return
    ctx = get_context()
    payload: dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "event": str(event),
        "host": socket.gethostname(),
        "service": ctx.service,
        "env": ctx.env,
    }
    if ctx.trace_id:
        payload["trace_id"] = ctx.trace_id
    if ctx.run_id:
        payload["run_id"] = ctx.run_id
    if fields:
        payload.update(fields)
    try:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return
