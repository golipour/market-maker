from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from mm.core.models.market import RawOrderbook, RawTrade


class AbstractMarketDataAdapter(ABC):
    """Provides RawTrade and RawOrderbook streams for one exchange symbol."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    def trade_stream(self) -> AsyncIterator[RawTrade]:
        ...

    @abstractmethod
    def orderbook_stream(self) -> AsyncIterator[RawOrderbook]:
        ...

