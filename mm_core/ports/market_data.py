from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TradeDataProvider(Protocol):
    """Time-windowed trades for one venue symbol (tick/lot-normalized dict rows)."""

    def get_trades(self, t_start_ms: int, t_end_ms: int) -> list[Any]:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


@runtime_checkable
class OrderbookDataProvider(Protocol):
    """Point-in-time order book snapshots for one venue symbol."""

    def get_orderbook(self, t_end_ms: int) -> Any | None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


@runtime_checkable
class StepperPort(Protocol):
    """
    Port for the stepper-like component that produces windowed steps.

    This is intentionally minimal; we will refine types once we migrate away from
    tuple-based Step.
    """

    T_ms: int
    steps: list[tuple]

    def advance(self, t_end_ms: int | None = None) -> tuple | None:
        ...

