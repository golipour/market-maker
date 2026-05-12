from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from mm.observability.context import get_context


def log_event(event: str, fields: dict[str, Any] | None = None) -> None:
    payload = {"ts_ms": int(time.time() * 1000), "event": event}
    if fields:
        payload.update(fields)
    _append_log(payload)
    _emit_json_stdout(event, fields)


def _json_stdout_enabled() -> bool:
    v = str(os.environ.get("MM_JSON_LOG", "")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def _emit_json_stdout(event: str, fields: dict[str, Any] | None = None) -> None:
    """One structured JSON line to stdout when ``MM_JSON_LOG`` is truthy."""

    if not _json_stdout_enabled():
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


def _append_log(payload: dict[str, Any]) -> None:
    path = os.environ.get("MM_LOG_PATH", "").strip()
    if not path:
        return
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        return
