from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StructuredLogEmit(Protocol):
    def __call__(self, event: str, fields: dict[str, Any] | None = None) -> None:
        ...


@runtime_checkable
class MetricLabelsFn(Protocol):
    def __call__(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        ...


@runtime_checkable
class MetricsRegistryPort(Protocol):
    def gauge(self, name: str, value: float, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        ...

    def counter_inc(self, name: str, inc: float = 1.0, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        ...


@runtime_checkable
class DiagnosticsEmit(Protocol):
    def __call__(self, frame: dict[str, Any]) -> None:
        ...
