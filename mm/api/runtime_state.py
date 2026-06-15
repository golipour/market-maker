"""Cached runtime control state and live strategy/worker status for the admin API."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from mm.bus.base import AbstractEventBus
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

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        try:
            async for event in self.bus.subscribe("admin:events"):
                if not isinstance(event, dict):
                    continue
                if str(event.get("type", "")).strip() != "RuntimeControlState":
                    continue
                self.snapshot = RuntimeControlSnapshot.from_payload(event)
        except asyncio.CancelledError:
            raise


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
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    strategies = store.list_strategies()
    assignments = [row for row in store.list_worker_assignments() if row.status == "active"]

    worker_by_strategy: dict[str, tuple[WorkerAssignmentRecord, WorkerLiveStatus, int]] = {}
    workers_payload: list[dict[str, Any]] = []
    for assignment in assignments:
        live, remaining_s = worker_live_status(assignment, now_ms=now_ms)
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
        strategies_payload.append(
            {
                "strategy_id": strategy.strategy_id,
                "actor_id": strategy.actor_id,
                "registry_status": strategy.status,
                "worker_id": worker_id,
                "worker_live": worker_live,
                "trading_status": trading,
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
