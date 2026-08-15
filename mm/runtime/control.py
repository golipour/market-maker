from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.observability import counter_inc, gauge, log_event


@dataclass(slots=True)
class RuntimeControlState:
    paused_actors: set[str] = field(default_factory=set)
    allocation_breached_actors: set[str] = field(default_factory=set)
    paused_venue_roles: set[str] = field(default_factory=set)
    dry_run: bool = False
    kill_switch: bool = False


class RuntimeControlPlane:
    """Consumes admin commands and exposes deterministic runtime safety state."""

    def __init__(
        self,
        *,
        bus: AbstractEventBus,
        state: RuntimeControlState | None = None,
        on_actor_start: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.bus = bus
        self.state = state or RuntimeControlState()
        self._on_actor_start = on_actor_start
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def publish_state(
        self,
        *,
        command: str = "bootstrap",
        command_id: str | None = None,
        observe: bool = True,
    ) -> None:
        await self._publish_state({"command": command, "command_id": command_id}, observe=observe)

    async def _run(self) -> None:
        try:
            async for command in self.bus.subscribe("admin:commands"):
                await self.apply(command)
        except asyncio.CancelledError:
            raise

    async def apply(self, command: dict[str, Any]) -> None:
        name = str(command.get("command", "")).strip()
        actor_id = command.get("actor_id")
        if name == "pause" and actor_id:
            self.state.paused_actors.add(str(actor_id))
        elif name == "resume" and actor_id:
            actor = str(actor_id)
            try:
                if self._on_actor_start is not None:
                    await self._on_actor_start(actor)
            except Exception as exc:
                log_event(
                    "actor_start_failed",
                    {
                        "actor_id": actor,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                # Stay paused if config reload / slot rebuild fails.
            else:
                self.state.paused_actors.discard(actor)
                self.state.allocation_breached_actors.discard(actor)
        elif name == "kill_switch":
            self.state.kill_switch = bool(command.get("enabled", True))
        elif name in ("dry_run", "observe_only"):
            self.state.dry_run = bool(command.get("enabled", True))
        elif name == "pause_local":
            self.state.paused_venue_roles.add("local")
        elif name == "resume_local":
            self.state.paused_venue_roles.discard("local")
        elif name == "pause_global":
            self.state.paused_venue_roles.add("global")
        elif name == "resume_global":
            self.state.paused_venue_roles.discard("global")
        elif name == "cancel_actor_orders":
            exchange = command.get("exchange")
            if exchange:
                await self.bus.publish(
                    f"execution:cancel_actor:{str(exchange).lower().strip()}",
                    {
                        "type": "CancelActorOrdersRequested",
                        "command_id": command.get("command_id"),
                        "exchange": str(exchange).lower().strip(),
                        "actor_id": actor_id,
                        "reason": command.get("reason", ""),
                        "timestamp_ms": int(time.time() * 1000),
                    },
                )
        elif name == "cancel_all":
            await self.bus.publish(
                f"execution:cancel_all:{command.get('exchange') or 'all'}",
                {
                    "type": "CancelAllRequested",
                    "command_id": command.get("command_id"),
                    "exchange": command.get("exchange"),
                    "actor_id": actor_id,
                    "reason": command.get("reason", ""),
                    "timestamp_ms": int(time.time() * 1000),
                },
            )
        await self._publish_state(command)

    def halt_actor_for_allocation_breach(
        self,
        actor_id: str,
        *,
        reasons: tuple[str, ...] | list[str] = (),
    ) -> None:
        actor = str(actor_id or "").strip()
        if not actor:
            return
        if actor in self.state.allocation_breached_actors:
            return
        self.state.allocation_breached_actors.add(actor)
        self.state.paused_actors.add(actor)
        log_event(
            "allocation_breach_halt",
            {
                "actor_id": actor,
                "reasons": list(reasons),
            },
        )

    async def _publish_state(self, command: dict[str, Any], *, observe: bool = True) -> None:
        payload = {
            "type": "RuntimeControlState",
            "command_id": command.get("command_id"),
            "paused_actors": sorted(self.state.paused_actors),
            "allocation_breached_actors": sorted(self.state.allocation_breached_actors),
            "paused_venue_roles": sorted(self.state.paused_venue_roles),
            "dry_run": self.state.dry_run,
            "kill_switch": self.state.kill_switch,
            "timestamp_ms": int(time.time() * 1000),
        }
        gauge("mm_kill_switch_state", 1.0 if self.state.kill_switch else 0.0, help="Runtime kill switch state")
        gauge("mm_dry_run_state", 1.0 if self.state.dry_run else 0.0, help="Runtime dry-run state")
        if observe:
            counter_inc("mm_admin_commands_total", labels={"command": str(command.get("command", ""))})
            log_event("admin_command_applied", payload)
        await self.bus.publish("admin:events", payload)
