from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from mm.api.platform_routes import create_platform_router
from mm.api.registry_routes import create_registry_router
from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.registry.audit import record_admin_audit
from mm.registry.store import registry_database_url
from mm.runtime.config import BusRuntimeConfig, runtime_config_from_mapping


class AdminCommand(BaseModel):
    command: str = Field(
        ...,
        pattern=(
            "^(pause|resume|kill_switch|cancel_all|dry_run|observe_only|reload_risk|"
            "pause_local|resume_local|pause_global|resume_global|cancel_actor_orders)$"
        ),
    )
    actor_id: str | None = None
    exchange: str | None = None
    enabled: bool | None = None
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


def create_admin_app(
    bus: AbstractEventBus | None = None,
    *,
    registry_store: object | None = None,
) -> FastAPI:
    runtime_bus = bus or _bus_from_env()
    app = FastAPI(title="market-maker-admin")
    app.state.bus = runtime_bus
    app.state.started_ms = int(time.time() * 1000)
    _register_admin_metrics(app.state.started_ms)
    _configure_cors(app)
    app.include_router(
        create_registry_router(
            require_auth=_require_auth,
            bus=runtime_bus,
            registry_store=registry_store,
        ),
    )
    app.include_router(create_platform_router(require_auth=_require_auth))
    _mount_platform_ui(app)

    @app.get("/api/admin/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "started_ms": app.state.started_ms}

    @app.get("/api/admin/config")
    async def config(_: str = Depends(_require_auth)) -> dict[str, Any]:
        return {
            "bus": _bus_config_from_env().kind,
            "admin_auth": bool(_admin_token()),
            "platform_ui": _platform_ui_dist().is_dir(),
        }

    @app.get("/api/admin/status")
    async def status(_: str = Depends(_require_auth)) -> dict[str, Any]:
        return {
            "status": "ok",
            "bus": _bus_config_from_env().kind,
            "database_url_configured": bool(os.environ.get("MM_DATABASE_URL", "").strip()),
            "registry_enabled": bool(os.environ.get("MM_DATABASE_URL", "").strip()),
            "parquet_root_configured": bool(os.environ.get("MM_PARQUET_ROOT", "").strip()),
            "metrics_port": os.environ.get("MM_METRICS_PORT", ""),
            "log_path": os.environ.get("MM_LOG_PATH", ""),
            "grafana_url": os.environ.get("MM_GRAFANA_URL", "http://localhost:3000"),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        from mm.observability.registry import REGISTRY

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
        _record_command_audit(
            user_id=user,
            command_id=command_id,
            command=command.command,
            status="accepted",
            payload=event,
        )
        return {"status": "accepted", "command_id": command_id}

    return app


def _register_admin_metrics(started_ms: int) -> None:
    from mm.observability.registry import REGISTRY

    REGISTRY.gauge(
        "mm_control_plane_up",
        1.0,
        help="Admin control plane process is running",
    )
    REGISTRY.gauge(
        "mm_control_plane_started_ms",
        float(started_ms),
        help="Unix epoch milliseconds when the admin API started",
    )


def _configure_cors(app: FastAPI) -> None:
    origins = os.environ.get(
        "MM_PLATFORM_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    allow = [origin.strip() for origin in origins.split(",") if origin.strip()]
    if allow:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def _platform_ui_dist() -> Path:
    return Path(__file__).resolve().parents[2] / "platform" / "ui" / "dist"


def _mount_platform_ui(app: FastAPI) -> None:
    dist = _platform_ui_dist()
    if not dist.is_dir():
        return

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    @app.get("/ui")
    async def ui_root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    @app.get("/ui/{resource_path:path}")
    async def ui_files(resource_path: str) -> FileResponse:
        if resource_path in ("", "/"):
            return FileResponse(dist / "index.html")
        candidate = dist / resource_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


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
            "exchanges": {"admin": {"market_data": {}}},
            "strategy_presets": {"admin": {"controller": {"enabled": False}}},
            "bots": {
                "admin": {
                    "preset": "admin",
                    "symbol": "ADMIN",
                    "quoting_exchange": "admin",
                    "liquidity_exchange": "admin",
                }
            },
            "bus": {
                "kind": kind,
                "redis_url": os.environ.get("MM_BUS_REDIS_URL", "redis://localhost:6379/0"),
                "stream_prefix": os.environ.get("MM_BUS_STREAM_PREFIX", "mm"),
                "consumer_group": os.environ.get("MM_BUS_CONSUMER_GROUP", "mm-admin"),
                "consumer_name": os.environ.get("MM_BUS_CONSUMER_NAME", "admin"),
            },
        }
    ).bus


def _record_command_audit(
    *,
    user_id: str,
    command_id: str,
    command: str,
    status: str,
    payload: dict[str, Any],
) -> None:
    database_url = registry_database_url()
    if not database_url:
        return
    try:
        record_admin_audit(
            database_url,
            command_id=command_id,
            user_id=user_id,
            command=command,
            status=status,
            payload=payload,
        )
    except Exception:
        return


app = create_admin_app()


if __name__ == "__main__":
    from mm.api.__main__ import main

    main()
