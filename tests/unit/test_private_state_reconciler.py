from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.execution import Fill, Order
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.factory import select_private_state_adapter
from mm.exchanges.nobitex.private_state import NobitexRestPrivateStateAdapter
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.private_state.reconciler import PrivateStateReconciler


class FakePrivateStateAdapter(AbstractPrivateStateAdapter):
    def __init__(self):
        self._connected = False
        self.balances: asyncio.Queue[Balance | None] = asyncio.Queue()
        self.open_orders: asyncio.Queue[OpenOrder | None] = asyncio.Queue()
        self.order_updates: asyncio.Queue[OrderUpdate | None] = asyncio.Queue()
        self.fills: asyncio.Queue[Fill | None] = asyncio.Queue()

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        await self.balances.put(None)
        await self.open_orders.put(None)
        await self.order_updates.put(None)
        await self.fills.put(None)

    async def snapshot(self) -> PrivateStateSnapshot:
        return PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={
                "BTC": Balance(
                    exchange="nobitex",
                    asset="BTC",
                    free=1.0,
                    locked=0.25,
                    timestamp_ms=1000,
                )
            },
            open_orders={},
        )

    async def balance_stream(self) -> AsyncIterator[Balance]:
        async for item in self._stream(self.balances):
            yield item

    async def open_order_stream(self) -> AsyncIterator[OpenOrder]:
        async for item in self._stream(self.open_orders):
            yield item

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        async for item in self._stream(self.order_updates):
            yield item

    async def fill_stream(self) -> AsyncIterator[Fill]:
        async for item in self._stream(self.fills):
            yield item

    async def _stream(self, queue):
        while self._connected:
            item = await queue.get()
            if item is None:
                break
            yield item


@pytest.mark.asyncio
async def test_private_state_reconciler_publishes_snapshot_and_open_orders() -> None:
    adapter = FakePrivateStateAdapter()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    snapshots = bus.subscribe_queue("private:state:nobitex:snapshot")
    open_orders = bus.subscribe_queue("private:state:nobitex:open_orders")

    try:
        await reconciler.start()
        initial = await asyncio.wait_for(snapshots.get(), timeout=1.0)
        assert initial.balances["BTC"].free == 1.0

        order = OpenOrder(
            exchange_order_id="o1",
            actor_id="actor-1",
            order=Order(exchange="nobitex", symbol="BTC_IRT", side="buy", price=100.0, qty=0.5),
            remaining_qty=0.5,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1100,
        )
        await adapter.open_orders.put(order)
        seen = await asyncio.wait_for(open_orders.get(), timeout=1.0)
        updated = await asyncio.wait_for(snapshots.get(), timeout=1.0)

        assert seen.exchange_order_id == "o1"
        assert updated.open_orders["o1"].order.price == 100.0
    finally:
        await reconciler.stop()


def test_private_state_factory_selects_rest_adapter() -> None:
    selection = select_private_state_adapter(
        "nobitex",
        "BTC_IRT",
        mode="websocket",
        api_token="token",
    )

    assert selection.mode == "rest"
    assert isinstance(selection.adapter, NobitexRestPrivateStateAdapter)


def test_rest_private_state_normalizes_open_orders_to_float_domain() -> None:
    adapter = RestPollingPrivateStateAdapter(
        "nobitex",
        "BTC_IRT",
        api_token="token",
    )

    order = adapter.normalize_open_order(
        {
            "id": "o1",
            "type": "buy",
            "price": "1000000",
            "amount": "0.5",
            "filledAmount": "0.1",
            "status": "open",
        },
        timestamp_ms=1000,
    )

    assert order is not None
    assert order.order.price == 1_000_000.0
    assert order.remaining_qty == pytest.approx(0.4)
    assert order.order.order_type == "limit"
