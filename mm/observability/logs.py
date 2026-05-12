from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def log_event(event: str, fields: dict[str, Any] | None = None) -> None:
    payload = {"ts_ms": int(time.time() * 1000), "event": event}
    if fields:
        payload.update(fields)
    _append_log(payload)
    try:
        from mm.observability.jsonlog import emit

        emit(event, fields)
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
