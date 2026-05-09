from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class AbstractEventBus(ABC):
    @abstractmethod
    async def publish(self, topic: str, event: Any) -> None:
        ...

    @abstractmethod
    def subscribe(self, topic: str) -> AsyncIterator[Any]:
        ...

