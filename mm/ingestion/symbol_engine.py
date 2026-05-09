from __future__ import annotations

import asyncio

from mm.aggregation.builder import MarketEventBuilder
from mm.aggregation.scheduler import IntervalScheduler
from mm.bus.base import AbstractEventBus
from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.ingestion.orderbook import OrderbookManager
from mm.ingestion.trade_buffer import RollingTradeBuffer


class SymbolEngine:
    def __init__(
        self,
        exchange: str,
        symbol: str,
        adapter: AbstractMarketDataAdapter,
        trade_buffer: RollingTradeBuffer,
        orderbook_manager: OrderbookManager,
        bus: AbstractEventBus,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.adapter = adapter
        self.trade_buffer = trade_buffer
        self.orderbook_manager = orderbook_manager
        self.bus = bus
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.adapter.connect()
        self._tasks = [
            asyncio.create_task(self._feed_trades()),
            asyncio.create_task(self._feed_orderbook()),
        ]
        if hasattr(self.adapter, "error_stream"):
            self._tasks.append(asyncio.create_task(self._feed_adapter_errors()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.adapter.disconnect()

    def market_event_builder(self) -> MarketEventBuilder:
        return MarketEventBuilder(
            self.exchange,
            self.symbol,
            self.trade_buffer,
            self.orderbook_manager,
        )

    async def publish_on_intervals(self, scheduler: IntervalScheduler, intervals_ms: list[int]) -> None:
        builder = self.market_event_builder()

        async def publish_event(t_start_ms: int, t_end_ms: int, interval_ms: int) -> None:
            try:
                event = builder.build(interval_ms, t_start_ms, t_end_ms)
            except LookupError:
                return
            topic = f"market:events:{self.exchange}:{self.symbol}:{interval_ms}"
            await self.bus.publish(topic, event)

        for interval_ms in intervals_ms:
            scheduler.register(interval_ms, publish_event)

    async def _feed_trades(self) -> None:
        try:
            async for trade in self.adapter.trade_stream():
                if self.trade_buffer.add(trade):
                    await self.bus.publish(f"market:raw:{self.exchange}:{self.symbol}:trades", trade)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.bus.publish(
                f"market:errors:{self.exchange}:{self.symbol}",
                {"source": "trade_stream", "error": type(exc).__name__, "message": str(exc)},
            )

    async def _feed_orderbook(self) -> None:
        try:
            async for book in self.adapter.orderbook_stream():
                try:
                    if book.is_snapshot:
                        self.orderbook_manager.apply_snapshot(book)
                    else:
                        self.orderbook_manager.apply_delta(book)
                except Exception as exc:
                    await self.bus.publish(
                        f"market:errors:{self.exchange}:{self.symbol}",
                        {
                            "source": "orderbook_normalization",
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "is_snapshot": book.is_snapshot,
                            "timestamp_ms": book.timestamp_ms,
                        },
                    )
                    continue
                await self.bus.publish(f"market:raw:{self.exchange}:{self.symbol}:orderbook", book)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.bus.publish(
                f"market:errors:{self.exchange}:{self.symbol}",
                {"source": "orderbook_stream", "error": type(exc).__name__, "message": str(exc)},
            )

    async def _feed_adapter_errors(self) -> None:
        try:
            async for error in self.adapter.error_stream():  # type: ignore[attr-defined]
                await self.bus.publish(f"market:errors:{self.exchange}:{self.symbol}", error)
        except asyncio.CancelledError:
            raise
