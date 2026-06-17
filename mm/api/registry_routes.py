"""FastAPI routes for the strategy registry."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mm.core.events import to_jsonable

from mm.registry.market_data import market_data_settings_from_exchange, market_source_as_dict
from mm.registry.audit import record_admin_audit
from mm.registry.memory import RegistryConflictError, RegistryNotFoundError
from mm.registry.runtime_config import runtime_config_mapping_from_registry
from mm.registry.store import RegistryStore, build_registry_store
from mm.runtime.config import runtime_config_from_mapping


class ExchangeUpsertRequest(BaseModel):
    exchange: str
    status: str = "active"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class AccountUpsertRequest(BaseModel):
    account_id: str
    exchange: str
    credential_ref: str
    role: str = "trading"
    status: str = "active"


class PresetUpsertRequest(BaseModel):
    preset_id: str
    config: dict[str, Any]
    version: int | None = None
    status: str = "active"


class StrategyUpsertRequest(BaseModel):
    strategy_id: str
    actor_id: str
    account_id: str
    symbol: str
    preset_id: str
    quoting_exchange: str
    liquidity_exchange: str
    status: str = "draft"
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    wallet_id: str | None = None
    ledger_source: str = "virtual"


class StrategyPatchRequest(BaseModel):
    status: str | None = None
    config_overrides: dict[str, Any] | None = None
    wallet_id: str | None = None
    ledger_source: str | None = None


class PlatformSettingsRequest(BaseModel):
    bus: dict[str, Any] | None = None
    persistence: dict[str, Any] | None = None
    observability: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None


class MarketSourceUpsertRequest(BaseModel):
    source_id: str | None = None
    exchange: str
    symbol: str
    status: str = "active"


class MarketFeedUpsertRequest(BaseModel):
    feed_id: str | None = None
    source_id: str
    interval_ms: int
    status: str = "active"


class WorkerAssignmentUpsertRequest(BaseModel):
    assignment_id: str | None = None
    worker_id: str
    strategy_ids: list[str] = Field(default_factory=list)
    lease_ttl_seconds: int = 300
    status: str = "active"


class AllocationUpsertRequest(BaseModel):
    allocation_id: str | None = None
    strategy_id: str | None = None
    actor_id: str = ""
    account_id: str
    quote_asset: str = ""
    base_asset: str = ""
    allocated_quote: float = 0.0
    allocated_base: float = 0.0
    halt_on_breach: bool = True
    status: str = "active"


class WorkerHeartbeatRequest(BaseModel):
    lease_ttl_seconds: int = 300


def create_registry_router(
    *,
    require_auth: Any,
    registry_store: RegistryStore | None = None,
    bus: Any | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/admin/registry", tags=["registry"])

    def _store() -> RegistryStore:
        if registry_store is not None:
            return registry_store
        url = os.environ.get("MM_DATABASE_URL", "").strip()
        if not url:
            raise HTTPException(status_code=503, detail="MM_DATABASE_URL is not configured")
        return build_registry_store(database_url=url)

    def _audit(user: str, command: str, status: str, payload: dict[str, Any]) -> None:
        url = os.environ.get("MM_DATABASE_URL", "").strip()
        if not url:
            return
        record_admin_audit(
            url,
            command_id=uuid.uuid4().hex,
            user_id=user,
            command=command,
            status=status,
            payload=payload,
        )

    @router.get("/platform-settings")
    async def get_platform_settings(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        return store.get_platform_settings().as_dict()

    @router.put("/platform-settings")
    async def put_platform_settings(
        body: PlatformSettingsRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        record = store.upsert_platform_settings(
            bus=body.bus,
            persistence=body.persistence,
            observability=body.observability,
            scheduler=body.scheduler,
        )
        _audit(user, "registry.platform_settings.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/exchanges")
    async def list_exchanges(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_exchanges()
        return {"items": [row.as_dict() for row in rows]}

    @router.post("/exchanges")
    async def upsert_exchange(
        body: ExchangeUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        record = store.upsert_exchange(
            exchange=body.exchange,
            status=body.status,
            capabilities=body.capabilities,
            config=body.config,
        )
        _audit(user, "registry.exchange.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/exchanges/{exchange}")
    async def get_exchange(
        exchange: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            return store.get_exchange(exchange).as_dict()
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/accounts")
    async def list_accounts(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_accounts()
        return {"items": [row.as_dict() for row in rows]}

    @router.post("/accounts")
    async def upsert_account(
        body: AccountUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.upsert_account(
                account_id=body.account_id,
                exchange=body.exchange,
                credential_ref=body.credential_ref,
                role=body.role,
                status=body.status,
            )
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _audit(user, "registry.account.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/accounts/{account_id}")
    async def get_account(
        account_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            return store.get_account(account_id).as_dict()
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/presets")
    async def list_presets(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_presets()
        return {"items": [row.as_dict() for row in rows]}

    @router.post("/presets")
    async def upsert_preset(
        body: PresetUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        record = store.upsert_preset(
            preset_id=body.preset_id,
            config=body.config,
            version=body.version,
            status=body.status,
        )
        _audit(user, "registry.preset.upsert", "accepted", {"preset_id": record.preset_id})
        return record.as_dict()

    @router.get("/presets/{preset_id}")
    async def get_preset(
        preset_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            return store.get_preset(preset_id).as_dict()
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/strategies")
    async def list_strategies(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_strategies()
        return {"items": [row.as_dict() for row in rows]}

    @router.post("/strategies")
    async def upsert_strategy(
        body: StrategyUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.upsert_strategy(
                strategy_id=body.strategy_id,
                actor_id=body.actor_id,
                account_id=body.account_id,
                symbol=body.symbol,
                preset_id=body.preset_id,
                quoting_exchange=body.quoting_exchange,
                liquidity_exchange=body.liquidity_exchange,
                status=body.status,
                config_overrides=body.config_overrides,
                wallet_id=body.wallet_id,
                ledger_source=body.ledger_source,
            )
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RegistryConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _audit(user, "registry.strategy.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/strategies/{strategy_id}")
    async def get_strategy(
        strategy_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            return store.get_strategy(strategy_id).as_dict()
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.patch("/strategies/{strategy_id}")
    async def patch_strategy(
        strategy_id: str,
        body: StrategyPatchRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.patch_strategy(
                strategy_id,
                status=body.status,
                config_overrides=body.config_overrides,
                wallet_id=body.wallet_id,
                ledger_source=body.ledger_source,
            )
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _audit(user, "registry.strategy.patch", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/strategies/{strategy_id}/runtime-config")
    async def get_strategy_runtime_config(
        strategy_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            bundle = store.load_runtime_bundle(strategy_id)
            mapping = runtime_config_mapping_from_registry(**bundle)
            resolved = runtime_config_from_mapping(mapping)
        except (RegistryNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = {
            "strategy_id": strategy_id,
            "runtime_config": mapping,
            "resolved_contract": resolved.as_contract(),
        }
        return JSONResponse(content=to_jsonable(payload))

    async def _publish_strategy_control(
        *,
        strategy_id: str,
        command: str,
        audit_command: str,
        reason: str,
        user: str,
        store: RegistryStore,
    ) -> dict[str, Any]:
        try:
            strategy = store.get_strategy(strategy_id)
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        command_id = uuid.uuid4().hex
        event = {
            "type": "AdminCommand",
            "command_id": command_id,
            "user_id": user,
            "command": command,
            "actor_id": strategy.actor_id,
            "exchange": strategy.quoting_exchange,
            "enabled": command == "resume",
            "reason": reason,
            "payload": {"strategy_id": strategy_id},
            "timestamp_ms": int(time.time() * 1000),
        }
        if bus is None:
            raise HTTPException(status_code=503, detail="admin bus is not configured")
        await bus.publish("admin:commands", event)
        _audit(user, audit_command, "accepted", event)
        return {"status": "accepted", "command_id": command_id, "actor_id": strategy.actor_id}

    @router.post("/strategies/{strategy_id}/start")
    async def start_strategy(
        strategy_id: str,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        return await _publish_strategy_control(
            strategy_id=strategy_id,
            command="resume",
            audit_command="registry.strategy.start",
            reason=f"registry start for strategy {strategy_id}",
            user=user,
            store=store,
        )

    @router.post("/strategies/{strategy_id}/stop")
    async def stop_strategy(
        strategy_id: str,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        return await _publish_strategy_control(
            strategy_id=strategy_id,
            command="pause",
            audit_command="registry.strategy.stop",
            reason=f"registry stop for strategy {strategy_id}",
            user=user,
            store=store,
        )

    @router.get("/allocations")
    async def list_allocations(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_allocations()
        return {"items": [row.as_dict() for row in rows]}

    @router.get("/allocations/{allocation_id}")
    async def get_allocation(
        allocation_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.get_allocation(allocation_id)
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return record.as_dict()

    @router.get("/strategies/{strategy_id}/allocation")
    async def get_strategy_allocation(
        strategy_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.get_allocation_for_strategy(strategy_id)
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return record.as_dict()

    @router.post("/allocations")
    async def upsert_allocation(
        body: AllocationUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.upsert_allocation(
                allocation_id=body.allocation_id,
                strategy_id=body.strategy_id,
                actor_id=body.actor_id,
                account_id=body.account_id,
                quote_asset=body.quote_asset,
                base_asset=body.base_asset,
                allocated_quote=body.allocated_quote,
                allocated_base=body.allocated_base,
                halt_on_breach=body.halt_on_breach,
                status=body.status,
            )
        except (RegistryNotFoundError, RegistryConflictError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _audit(user, "registry.allocation.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/market-sources")
    async def list_market_sources(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> JSONResponse:
        rows = store.list_market_sources()
        items = []
        for row in rows:
            exchange = store.get_exchange(row.exchange)
            items.append(market_source_as_dict(exchange, row))
        return JSONResponse(content=to_jsonable({"items": items}))

    @router.get("/market-sources/{source_id}")
    async def get_market_source(
        source_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> JSONResponse:
        try:
            record = store.get_market_source(source_id)
            exchange = store.get_exchange(record.exchange)
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(content=to_jsonable(market_source_as_dict(exchange, record)))

    @router.post("/market-sources")
    async def upsert_market_source(
        body: MarketSourceUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.upsert_market_source(
                source_id=body.source_id,
                exchange=body.exchange,
                symbol=body.symbol,
                status=body.status,
            )
            exchange = store.get_exchange(record.exchange)
        except (RegistryNotFoundError, RegistryConflictError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = market_source_as_dict(exchange, record)
        _audit(user, "registry.market_source.upsert", "accepted", payload)
        return payload

    @router.get("/market-feeds")
    async def list_market_feeds(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> JSONResponse:
        rows = store.list_market_feeds()
        return JSONResponse(content=to_jsonable({"items": [row.as_dict() for row in rows]}))

    @router.get("/market-feeds/{feed_id}")
    async def get_market_feed(
        feed_id: str,
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> JSONResponse:
        try:
            record = store.get_market_feed(feed_id)
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(content=to_jsonable(record.as_dict()))

    @router.post("/market-feeds")
    async def upsert_market_feed(
        body: MarketFeedUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.upsert_market_feed(
                feed_id=body.feed_id,
                source_id=body.source_id,
                interval_ms=body.interval_ms,
                status=body.status,
            )
        except (RegistryNotFoundError, RegistryConflictError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _audit(user, "registry.market_feed.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.get("/worker-assignments")
    async def list_worker_assignments(
        _: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        rows = store.list_worker_assignments()
        return {"items": [row.as_dict() for row in rows]}

    @router.post("/worker-assignments")
    async def upsert_worker_assignment(
        body: WorkerAssignmentUpsertRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        record = store.upsert_worker_assignment(
            assignment_id=body.assignment_id,
            worker_id=body.worker_id,
            strategy_ids=body.strategy_ids,
            lease_ttl_seconds=body.lease_ttl_seconds,
            status=body.status,
        )
        _audit(user, "registry.worker_assignment.upsert", "accepted", record.as_dict())
        return record.as_dict()

    @router.post("/worker-assignments/{assignment_id}/heartbeat")
    async def heartbeat_worker_assignment(
        assignment_id: str,
        body: WorkerHeartbeatRequest,
        user: str = Depends(require_auth),
        store: RegistryStore = Depends(_store),
    ) -> dict[str, Any]:
        try:
            record = store.heartbeat_worker_assignment(
                assignment_id,
                lease_ttl_seconds=body.lease_ttl_seconds,
            )
        except RegistryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _audit(
            user,
            "registry.worker_assignment.heartbeat",
            "accepted",
            {"assignment_id": assignment_id},
        )
        return record.as_dict()

    return router
