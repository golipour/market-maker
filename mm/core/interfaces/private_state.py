from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from mm.core.models.execution import Fill
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot


class AbstractPrivateStateAdapter(ABC):
    """Private, authenticated exchange state.

    This contract is intentionally separate from public market data and order
    commands. A venue may implement it with REST polling, websocket private
    streams, or both behind a reconciler, but consumers see normalized float
    balances, open orders, order updates, and fills.
    """

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def snapshot(self) -> PrivateStateSnapshot:
        ...

    @abstractmethod
    def balance_stream(self) -> AsyncIterator[Balance]:
        ...

    @abstractmethod
    def open_order_stream(self) -> AsyncIterator[OpenOrder]:
        ...

    @abstractmethod
    def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        ...

    @abstractmethod
    def fill_stream(self) -> AsyncIterator[Fill]:
        ...
