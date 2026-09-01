"""Cached runtime control state and live strategy/worker status for the admin API."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from mm.bus.base import AbstractEventBus
from mm.observability.heartbeat import (
    ERRORS_TOPIC_PREFIX,
    HEARTBEAT_TOPIC_PREFIX,
    classify_data_status,
)
from mm.registry.models import StrategyRecord, WorkerAssignmentRecord
from mm.registry.store import RegistryStore

WorkerLiveStatus = Literal["alive", "stale", "offline"]
TradingStatus = Literal[
    "running",
    "paused",
    "breach",
    "kill_switch",
    "dry_run",
    "worker_offline",
    "unassigned",
    "registry_inactive",
]
DataStatus = Literal["fresh", "stale", "lost", "unknown", "unattached"]


@dataclass(slots=True)
class RuntimeControlSnapshot:
    paused_actors: tuple[str, ...] = ()
    allocation_breached_actors: tuple[str, ...] = ()
    paused_venue_roles: tuple[str, ...] = ()
    dry_run: bool = False
    kill_switch: bool = False
    updated_at_ms: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RuntimeControlSnapshot:
        return cls(
            paused_actors=tuple(str(item) for item in payload.get("paused_actors") or ()),
            allocation_breached_actors=tuple(
                str(item) for item in payload.get("allocation_breached_actors") or ()
            ),
            paused_venue_roles=tuple(str(item) for item in payload.get("paused_venue_roles") or ()),
            dry_run=bool(payload.get("dry_run")),
            kill_switch=bool(payload.get("kill_switch")),
            updated_at_ms=int(payload.get("timestamp_ms") or 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "paused_actors": list(self.paused_actors),
            "allocation_breached_actors": list(self.allocation_breached_actors),
            "paused_venue_roles": list(self.paused_venue_roles),
            "dry_run": self.dry_run,
            "kill_switch": self.kill_switch,
            "updated_at_ms": self.updated_at_ms,
        }


@dataclass(slots=True)
class RuntimeControlCache:
    bus: AbstractEventBus
    snapshot: RuntimeControlSnapshot = field(default_factory=RuntimeControlSnapshot)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def connected(self) -> bool:
        return self.snapshot.updated_at_ms > 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        await self._hydrate()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def _apply(self, event: dict[str, Any]) -> None:
        if str(event.get("type", "")).strip() != "RuntimeControlState":
            return
        snapshot = RuntimeControlSnapshot.from_payload(event)
        if snapshot.updated_at_ms >= self.snapshot.updated_at_ms:
            self.snapshot = snapshot

    async def _hydrate(self) -> None:
        latest = getattr(self.bus, "latest_event", None)
        if not callable(latest):
            return
        try:
            event = await latest("admin:events", event_type="RuntimeControlState")
        except Exception:
            return
        if isinstance(event, dict):
            self._apply(event)

    async def _run(self) -> None:
        try:
            async for event in self.bus.subscribe("admin:events"):
                if isinstance(event, dict):
                    self._apply(event)
        except asyncio.CancelledError:
            raise


@dataclass(slots=True)
class ActorHealth:
    actor_id: str
    data_status: DataStatus = "unknown"
    last_heartbeat_ms: int = 0
    last_heartbeat_age_ms: int | None = None
    interval_ms: int = 30_000
    capability_status: str = ""
    tradeable: bool = True
    last_error: dict[str, Any] | None = None


@dataclass(slots=True)
class DataHealthCache:
    bus: AbstractEventBus
    default_interval_ms: int = 30_000
    actors: dict[str, ActorHealth] = field(default_factory=dict)
    _poll_task: asyncio.Task | None = field(default=None, repr=False)
    _capability_task: asyncio.Task | None = field(default=None, repr=False)
    _capability: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def start(self) -> None:
        await self._hydrate()
        self._poll_task = asyncio.create_task(self._run_poll())
        self._capability_task = asyncio.create_task(self._run_capability())

    async def stop(self) -> None:
        for task in (self._poll_task, self._capability_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._poll_task = None
        self._capability_task = None

    def _ensure_actor(self, actor_id: str) -> ActorHealth:
        health = self.actors.get(actor_id)
        if health is None:
            health = ActorHealth(actor_id=actor_id, interval_ms=self.default_interval_ms)
            self.actors[actor_id] = health
        return health

    def _apply_heartbeat(self, event: dict[str, Any]) -> None:
        actor_id = str(event.get("actor_id") or "").strip()
        if not actor_id:
            return
        ts = int(event.get("ts_ms") or 0)
        if ts <= 0:
            return
        health = self._ensure_actor(actor_id)
        try:
            interval_ms = int(event.get("interval_ms") or 0) or self.default_interval_ms
        except (TypeError, ValueError):
            interval_ms = self.default_interval_ms
        health.interval_ms = interval_ms
        health.last_heartbeat_ms = ts
        health.last_heartbeat_age_ms = max(0, int(time.time() * 1000) - ts)
        health.data_status = classify_data_status(
            age_ms=health.last_heartbeat_age_ms,
            interval_ms=health.interval_ms,
        )

    def _apply_error(self, event: dict[str, Any]) -> None:
        actor_id = str(event.get("actor_id") or "").strip()
        if not actor_id:
            return
        health = self._ensure_actor(actor_id)
        health.last_error = {
            "kind": str(event.get("kind") or ""),
            "message": str(event.get("message") or ""),
            "frame_id": str(event.get("frame_id") or ""),
            "ts_ms": int(event.get("ts_ms") or 0),
        }

    def _apply_capability(self, event: dict[str, Any]) -> None:
        assignment_id = str(event.get("assignment_id") or "").strip()
        if assignment_id:
            self._capability[assignment_id] = event
        for actor in event.get("actors") or ():
            if not isinstance(actor, dict):
                continue
            actor_id = str(actor.get("actor_id") or "").strip()
            if not actor_id:
                continue
            health = self._ensure_actor(actor_id)
            health.capability_status = str(actor.get("status") or "")
            health.tradeable = bool(actor.get("tradeable"))

    async def _hydrate(self) -> None:
        latest = getattr(self.bus, "latest_event", None)
        if not callable(latest):
            return
        try:
            event = await latest("worker:capability")
        except Exception:
            event = None
        if isinstance(event, dict):
            self._apply_capability(event)
        await self._refresh_from_latest()

    async def _run_poll(self) -> None:
        try:
            while True:
                await self._refresh_from_latest()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _run_capability(self) -> None:
        try:
            async for event in self.bus.subscribe("worker:capability"):
                if isinstance(event, dict):
                    self._apply_capability(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _refresh_from_latest(self) -> None:
        latest = getattr(self.bus, "latest_event", None)
        if not callable(latest):
            return
        for actor_id in self._discover_actor_ids():
            try:
                event = await latest(f"{HEARTBEAT_TOPIC_PREFIX}{actor_id}")
            except Exception:
                event = None
            if isinstance(event, dict):
                self._apply_heartbeat(event)
            try:
                event = await latest(f"{ERRORS_TOPIC_PREFIX}{actor_id}")
            except Exception:
                event = None
            if isinstance(event, dict):
                self._apply_error(event)

    async def refresh_actors(self, actor_ids: tuple[str, ...] | list[str]) -> None:
        for actor_id in actor_ids:
            if actor_id:
                self._ensure_actor(actor_id)
        await self._refresh_from_latest()

    def _discover_actor_ids(self) -> tuple[str, ...]:
        ids = set(self.actors.keys())
        for payload in self._capability.values():
            for actor in payload.get("actors") or ():
                if isinstance(actor, dict) and actor.get("actor_id"):
                    ids.add(str(actor["actor_id"]))
        return tuple(sorted(ids))

    def register_actor(self, actor_id: str) -> None:
        self._ensure_actor(actor_id)

    def status_for(self, actor_id: str, *, now_ms: int | None = None) -> ActorHealth:
        health = self._ensure_actor(actor_id)
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        if health.last_heartbeat_ms > 0:
            health.last_heartbeat_age_ms = max(0, now - health.last_heartbeat_ms)
            health.data_status = classify_data_status(
                age_ms=health.last_heartbeat_age_ms,
                interval_ms=health.interval_ms,
            )
        elif not health.tradeable:
            health.data_status = "unattached"
        return health

    def capability_for_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        return self._capability.get(assignment_id)


def worker_live_status(
    assignment: WorkerAssignmentRecord,
    *,
    now_ms: int | None = None,
) -> tuple[WorkerLiveStatus, int]:
    """Return liveness and lease seconds remaining for a worker assignment."""

    if assignment.status != "active":
        return "offline", 0
    expires_ms = _timestamp_ms(assignment.lease_expires_at)
    if expires_ms is None:
        return "offline", 0
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    remaining_s = max(0, int((expires_ms - now) / 1000))
    if remaining_s > 0:
        return "alive", remaining_s
    heartbeat_ms = _timestamp_ms(assignment.heartbeat_at)
    if heartbeat_ms is not None and now - heartbeat_ms < 120_000:
        return "stale", 0
    return "offline", 0


def strategy_trading_status(
    strategy: StrategyRecord,
    *,
    control: RuntimeControlSnapshot,
    worker_live: WorkerLiveStatus | None,
) -> TradingStatus:
    actor = strategy.actor_id.strip()
    if strategy.status != "active":
        return "registry_inactive"
    if control.kill_switch:
        return "kill_switch"
    if control.dry_run:
        return "dry_run"
    if worker_live is None:
        return "unassigned"
    if worker_live != "alive":
        return "worker_offline"
    if actor in control.allocation_breached_actors:
        return "breach"
    if actor in control.paused_actors:
        return "paused"
    return "running"


def build_runtime_overview(
    store: RegistryStore,
    control: RuntimeControlCache,
    health: DataHealthCache | None = None,
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    strategies = store.list_strategies()
    assignments = [row for row in store.list_worker_assignments() if row.status == "active"]

    worker_by_strategy: dict[str, tuple[WorkerAssignmentRecord, WorkerLiveStatus, int]] = {}
    workers_payload: list[dict[str, Any]] = []
    for assignment in assignments:
        live, remaining_s = worker_live_status(assignment, now_ms=now_ms)
        cap = (
            health.capability_for_assignment(assignment.assignment_id)
            if health is not None
            else None
        )
        workers_payload.append(
            {
                "assignment_id": assignment.assignment_id,
                "worker_id": assignment.worker_id,
                "strategy_ids": list(assignment.strategy_ids),
                "registry_status": assignment.status,
                "live": live,
                "lease_remaining_s": remaining_s,
                "heartbeat_at": _iso(assignment.heartbeat_at),
                "lease_expires_at": _iso(assignment.lease_expires_at),
                "capability": cap,
            }
        )
        for strategy_id in assignment.strategy_ids:
            previous = worker_by_strategy.get(strategy_id)
            if previous is None or _live_rank(live) > _live_rank(previous[1]):
                worker_by_strategy[strategy_id] = (assignment, live, remaining_s)

    strategies_payload: list[dict[str, Any]] = []
    for strategy in strategies:
        mapped = worker_by_strategy.get(strategy.strategy_id)
        worker_live = mapped[1] if mapped is not None else None
        worker_id = mapped[0].worker_id if mapped is not None else None
        trading = strategy_trading_status(
            strategy,
            control=control.snapshot,
            worker_live=worker_live,
        )
        actor_health = None
        if health is not None and strategy.actor_id:
            actor = health.status_for(strategy.actor_id, now_ms=now_ms)
            actor_health = {
                "data_status": actor.data_status,
                "last_heartbeat_ms": actor.last_heartbeat_ms,
                "last_heartbeat_age_ms": actor.last_heartbeat_age_ms,
                "interval_ms": actor.interval_ms,
                "capability_status": actor.capability_status,
                "tradeable": actor.tradeable,
                "last_error": actor.last_error,
            }
        strategies_payload.append(
            {
                "strategy_id": strategy.strategy_id,
                "actor_id": strategy.actor_id,
                "registry_status": strategy.status,
                "worker_id": worker_id,
                "worker_live": worker_live,
                "trading_status": trading,
                "data_status": (actor_health or {}).get("data_status", "unknown"),
                "data_health": actor_health,
            }
        )

    return {
        "checked_at_ms": now_ms,
        "control": {
            "connected": control.connected,
            **control.snapshot.as_dict(),
        },
        "workers": workers_payload,
        "strategies": strategies_payload,
    }


def _live_rank(status: WorkerLiveStatus) -> int:
    return {"alive": 3, "stale": 2, "offline": 1}[status]


def _timestamp_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone().isoformat()
