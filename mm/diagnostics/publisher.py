"""Publish ``ControllerDiagnostics`` on the runtime event bus."""

from __future__ import annotations

from mm.bus.base import AbstractEventBus
from mm.diagnostics.models import ControllerDiagnostics
from mm.diagnostics.serializers import diagnostics_to_jsonable


class DiagnosticsPublisher:
    """Thin wrapper around bus publish with JSON-safe payloads."""

    def __init__(self, bus: AbstractEventBus | None, *, topic: str) -> None:
        self._bus = bus
        self._topic = topic.strip()

    @property
    def topic(self) -> str:
        return self._topic

    async def publish(self, diagnostics: ControllerDiagnostics) -> None:
        if self._bus is None or not self._topic:
            return
        payload = diagnostics_to_jsonable(diagnostics.as_payload())
        await self._bus.publish(self._topic, payload)
