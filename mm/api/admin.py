from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.runtime.config import BusRuntimeConfig, runtime_config_from_mapping


class AdminCommand(BaseModel):
    command: str = Field(..., pattern="^(pause|resume|kill_switch|cancel_all|observe_only|reload_risk)$")
    actor_id: str | None = None
    exchange: str | None = None
    enabled: bool | None = None
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


def create_admin_app(bus: AbstractEventBus | None = None) -> FastAPI:
    runtime_bus = bus or _bus_from_env()
    app = FastAPI(title="market-maker-admin")
    app.state.bus = runtime_bus
    app.state.started_ms = int(time.time() * 1000)

    @app.get("/api/admin/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "started_ms": app.state.started_ms}

    @app.get("/api/admin/config")
    async def config(_: None = Depends(_require_auth)) -> dict[str, Any]:
        return {"bus": _bus_config_from_env().kind, "admin_auth": bool(_admin_token())}

    @app.get("/api/admin/status")
    async def status(_: None = Depends(_require_auth)) -> dict[str, Any]:
        return {
            "status": "ok",
            "bus": _bus_config_from_env().kind,
            "database_url_configured": bool(os.environ.get("MM_DATABASE_URL", "").strip()),
            "parquet_root_configured": bool(os.environ.get("MM_PARQUET_ROOT", "").strip()),
            "metrics_port": os.environ.get("MM_METRICS_PORT", ""),
            "log_path": os.environ.get("MM_LOG_PATH", ""),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        from mm.observability_metrics import REGISTRY

        return Response(REGISTRY.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/api/admin/commands")
    async def command(command: AdminCommand, user: str = Depends(_require_auth)) -> dict[str, Any]:
        command_id = uuid.uuid4().hex
        event = {
            "type": "AdminCommand",
            "command_id": command_id,
            "user_id": user,
            "command": command.command,
            "actor_id": command.actor_id,
            "exchange": command.exchange,
            "enabled": command.enabled,
            "reason": command.reason,
            "payload": command.payload,
            "timestamp_ms": int(time.time() * 1000),
        }
        await app.state.bus.publish("admin:commands", event)
        await app.state.bus.publish(
            "admin:events",
            {
                "type": "AdminCommandAccepted",
                "command_id": command_id,
                "command": command.command,
                "user_id": user,
                "timestamp_ms": int(time.time() * 1000),
            },
        )
        return {"status": "accepted", "command_id": command_id}

    return app


def _require_auth(authorization: str | None = Header(default=None)) -> str:
    token = _admin_token()
    if not token:
        return "local"
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid admin token")
    return "admin"


def _admin_token() -> str:
    return os.environ.get("MM_ADMIN_TOKEN", "").strip()


def _bus_from_env() -> AbstractEventBus:
    return build_event_bus(_bus_config_from_env(), run_id=os.environ.get("MM_RUN_ID", ""))


def _bus_config_from_env() -> BusRuntimeConfig:
    kind = os.environ.get("MM_BUS_KIND", "memory").strip() or "memory"
    return runtime_config_from_mapping(
        {
            "primary": {"exchange": "admin", "symbol": "ADMIN"},
            "bus": {
                "kind": kind,
                "redis_url": os.environ.get("MM_BUS_REDIS_URL", "redis://localhost:6379/0"),
                "stream_prefix": os.environ.get("MM_BUS_STREAM_PREFIX", "mm"),
                "consumer_group": os.environ.get("MM_BUS_CONSUMER_GROUP", "mm-admin"),
                "consumer_name": os.environ.get("MM_BUS_CONSUMER_NAME", "admin"),
            },
        }
    ).bus


app = create_admin_app()
