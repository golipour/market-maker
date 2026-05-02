"""
FastAPI diagnostics UI: WebSocket push from Redis (GET + Pub/Sub), matplotlib PNG endpoints,
static SPA with Chart.js (no Streamlit).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mm_adapters.diagnostics.delta_codec import (
    KIND_DELTA,
    KIND_FULL,
    SCHEMA,
    apply_patch,
    is_v2_envelope,
)
from mm_viz.mpl_figures import figure_action_depth_window, figure_lof_observed_and_sample

logger = logging.getLogger("mm_viz.server")

logging.basicConfig(level=logging.INFO)


def _redis_url() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_URL", "redis://127.0.0.1:6379/0")).strip()


def _redis_channel() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_CHANNEL", "mm.diagnostics.v1")).strip()


def _redis_latest_key() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_LATEST_KEY", "mm_diagnostics:latest")).strip() or "mm_diagnostics:latest"


def _max_frames() -> int:
    try:
        return max(50, int(os.environ.get("MM_VIZ_MAX_FRAMES", "2000")))
    except Exception:
        return 2000


def _sanitize_for_json(obj: Any) -> Any:
    """
    Starlette's JSONResponse uses strict JSON (no NaN/Infinity). Frames from Redis may contain
    ``float('nan')`` after ``json.loads`` (non-standard JSON), which raises during serialization.
    """
    import math

    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return _sanitize_for_json(obj.item())
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def _merge_redis_message(obj: dict[str, Any], hub: "Hub") -> dict[str, Any] | None:
    """Turn Redis Pub/Sub JSON (legacy full or mm.diag.v2 envelope) into a merged full frame."""
    if not is_v2_envelope(obj):
        return obj
    kind = obj.get("kind")
    if kind == KIND_FULL:
        return obj.get("frame")
    if kind == KIND_DELTA:
        base = hub.latest_frame
        if base is None:
            logger.warning(
                "Skipping diagnostics delta (no base frame yet). Wait for Redis GET / full publish."
            )
            return None
        patch = obj.get("patch")
        return apply_patch(base, patch)
    return None


def _ws_payload_from_pub(raw: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    """WebSocket message: full frame, or compact delta for client-side merge."""
    if is_v2_envelope(raw) and raw.get("kind") == KIND_DELTA:
        return {
            "type": "frame_delta",
            "schema": SCHEMA,
            "step_counter": raw.get("step_counter"),
            "patch": raw.get("patch"),
        }
    return {"type": "frame", "frame": merged}


class Hub:
    """Shared state + WebSocket fan-out."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.latest_frame: dict[str, Any] | None = None
        self.last_step: Any = None
        self.frames: deque = deque(maxlen=_max_frames())

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def accept_frame(self, frame: dict[str, Any]) -> bool:
        """Return True if step advanced (or first frame)."""
        step = (frame.get("point") or {}).get("step_counter")
        if step == self.last_step and self.latest_frame is not None:
            return False
        self.last_step = step
        self.latest_frame = frame
        self.frames.append(frame)
        return True

    def frame_for_step(self, step: int | None) -> dict[str, Any] | None:
        if step is None:
            return self.latest_frame
        for f in reversed(self.frames):
            if (f.get("point") or {}).get("step_counter") == step:
                return f
        return None

    async def broadcast_ws_payload(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        out = _sanitize_for_json(payload)
        for ws in self.clients:
            try:
                await ws.send_json(out)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)


hub = Hub()
_listener_task: asyncio.Task | None = None
_stop = asyncio.Event()


async def _redis_loop() -> None:
    import redis.asyncio as aioredis

    url, channel, key = _redis_url(), _redis_channel(), _redis_latest_key()

    while not _stop.is_set():
        try:
            r = aioredis.from_url(url, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe(channel)

            async def pull_latest() -> bool:
                raw = await r.get(key)
                if not raw:
                    return False
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    return False
                merged = _merge_redis_message(obj, hub)
                if merged is None:
                    return False
                if hub.accept_frame(merged):
                    await hub.broadcast_ws_payload({"type": "frame", "frame": merged})
                    return True
                return False

            await pull_latest()

            last_backup = asyncio.get_event_loop().time()
            while not _stop.is_set():
                now = asyncio.get_event_loop().time()
                if now - last_backup >= 1.0:
                    last_backup = now
                    await pull_latest()

                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.35)
                if msg and msg.get("type") == "message":
                    data = msg.get("data")
                    if isinstance(data, str):
                        try:
                            raw_obj = json.loads(data)
                            merged = _merge_redis_message(raw_obj, hub)
                            if merged is None:
                                continue
                            if hub.accept_frame(merged):
                                await hub.broadcast_ws_payload(_ws_payload_from_pub(raw_obj, merged))
                        except json.JSONDecodeError:
                            pass

            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await r.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Redis listener error (%s), retry in 2s", e)
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _listener_task
    _stop.clear()
    _listener_task = asyncio.create_task(_redis_loop())
    yield
    _stop.set()
    if _listener_task:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="mm_viz", lifespan=lifespan)


@app.get("/api/latest")
async def api_latest():
    if not hub.latest_frame:
        raise HTTPException(status_code=404, detail="No diagnostics frame yet.")
    return _sanitize_for_json(hub.latest_frame)


@app.get("/api/config")
async def api_config():
    return {
        "ui_backend": "fastapi-websocket-v1",
        "redis_url": _redis_url(),
        "channel": _redis_channel(),
        "latest_key": _redis_latest_key(),
        "max_frames": _max_frames(),
    }


@app.get("/api/history_steps")
async def api_history_steps():
    """Steps currently buffered server-side (for figure URLs)."""
    out = []
    for f in hub.frames:
        out.append((f.get("point") or {}).get("step_counter"))
    return _sanitize_for_json({"steps": out})


def _png_response(fig) -> Response:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/api/figures/action_depth.png")
def figure_action_depth_png(step: int | None = None):
    frame = hub.frame_for_step(step)
    if not frame:
        raise HTTPException(status_code=404)
    return _png_response(figure_action_depth_window(frame))


@app.get("/api/figures/lof.png")
def figure_lof_png(step: int | None = None):
    frame = hub.frame_for_step(step)
    if not frame:
        raise HTTPException(status_code=404)
    fig = figure_lof_observed_and_sample(frame)
    if fig is None:
        raise HTTPException(status_code=404, detail="No LOF panels in this frame.")
    return _png_response(fig)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await hub.register(websocket)
    try:
        if hub.latest_frame:
            await websocket.send_json(
                {"type": "frame", "frame": _sanitize_for_json(hub.latest_frame)}
            )
        steps = [(f.get("point") or {}).get("step_counter") for f in hub.frames]
        await websocket.send_json(_sanitize_for_json({"type": "history_meta", "steps": steps}))

        # Client should send occasional pings (see static app.js) so we detect disconnects;
        # otherwise Starlette may not surface half-closed peers until a send fails.
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            raise
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(websocket)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
