from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from mm.core.models.market import RawOrderbook, RawTrade


class AbstractMarketDataAdapter(ABC):
    """Public, read-only market data for one exchange symbol.

    Implementations must be single-transport from the engine's point of view:
    either REST polling or websocket streaming. Any exchange-specific bootstrap
    needed to make a websocket depth feed valid must stay internal to the
    adapter and still emit normalized RawTrade/RawOrderbook objects.
    """

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


PublicMarketDataAdapter = AbstractMarketDataAdapter

