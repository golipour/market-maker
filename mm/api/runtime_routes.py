"""Runtime control and live status endpoints for the admin API."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from mm.api.runtime_state import RuntimeControlCache, build_runtime_overview
from mm.registry.store import RegistryStore, build_registry_store


def create_runtime_router(
    *,
    require_auth: Any,
    registry_store: RegistryStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/admin/runtime", tags=["runtime"])

    def _store() -> RegistryStore:
        if registry_store is not None:
            return registry_store
        url = os.environ.get("MM_DATABASE_URL", "").strip()
        if not url:
            raise HTTPException(status_code=503, detail="MM_DATABASE_URL is not configured")
        return build_registry_store(database_url=url)

    def _cache(request: Request) -> RuntimeControlCache:
        cache = getattr(request.app.state, "runtime_control", None)
        if cache is None:
            return RuntimeControlCache(bus=request.app.state.bus)
        return cache

    @router.get("/control")
    async def runtime_control(
        request: Request,
        _: str = Depends(require_auth),
    ) -> dict[str, Any]:
        cache = _cache(request)
        return {
            "connected": cache.connected,
            **cache.snapshot.as_dict(),
        }

    @router.get("/overview")
    async def runtime_overview(
        request: Request,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        return build_runtime_overview(store, _cache(request))

    return router
