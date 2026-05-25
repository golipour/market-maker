from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Action, Fill, Order
from mm.core.models.private import OpenOrder, PrivateStateSnapshot
from mm.execution.order_manager import ExecutionSnapshot, OrderManager
from mm.execution.reconciler import ExecutionStateReconciler


class FakeOrderCommandAdapter(AbstractOrderCommandAdapter):
    def __init__(self):
        self.placed: list[Order] = []
        self.cancelled: list[str] = []
        self._next = 1

    async def place_orders(self, orders: list[Order]) -> list[str]:
        self.placed.extend(orders)
        ids = [f"ex-{self._next + i}" for i in range(len(orders))]
        self._next += len(orders)
        return ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        self.cancelled.extend(order_ids)


@pytest.mark.asyncio
async def test_order_manager_places_and_publishes_execution_snapshot() -> None:
    bus = InMemoryBus()
    state_queue = bus.subscribe_queue("execution:state:nobitex")
    command_queue = bus.subscribe_queue("execution:orders:nobitex")
    commands = FakeOrderCommandAdapter()
    manager = OrderManager("nobitex", commands, bus=bus, clock_ms=lambda: 1000)

    await manager.sync(
        [
            Action(
                actor_id="actor-1",
                orders=(
                    Order("nobitex", "BTC_IRT", "buy", 100.0, 0.5, client_order_id="bid-1"),
                ),
            )
        ]
    )

    requested = await asyncio.wait_for(command_queue.get(), timeout=1.0)
    placed = await asyncio.wait_for(command_queue.get(), timeout=1.0)
    accepted = await asyncio.wait_for(command_queue.get(), timeout=1.0)
    completed = await asyncio.wait_for(command_queue.get(), timeout=1.0)
    snapshot = await asyncio.wait_for(state_queue.get(), timeout=1.0)

    assert requested["type"] == "ExecutionSyncRequested"
    assert placed["type"] == "PlaceOrdersRequested"
    assert accepted["type"] == "PlaceOrdersAccepted"
    assert completed["type"] == "ExecutionSyncCompleted"
    assert completed["accepted_count"] == 1
    assert completed["target_count"] == 1
    assert isinstance(snapshot, ExecutionSnapshot)
    assert commands.placed[0].price == 100.0
    assert snapshot.live_order_count == 1
    assert snapshot.live_orders["ex-1"].remaining_qty == 0.5


@pytest.mark.asyncio
async def test_execution_reconciler_mirrors_private_open_orders() -> None:
    bus = InMemoryBus()
    state_queue = bus.subscribe_queue("execution:state:nobitex")
    manager = OrderManager("nobitex", FakeOrderCommandAdapter(), bus=bus, clock_ms=lambda: 1000)
    reconciler = ExecutionStateReconciler("nobitex", manager, bus)

    try:
        await reconciler.start()
        await asyncio.wait_for(state_queue.get(), timeout=1.0)

        order = OpenOrder(
            exchange_order_id="real-1",
            actor_id="actor-1",
            order=Order("nobitex", "BTC_IRT", "sell", 101.0, 0.25),
            remaining_qty=0.25,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1200,
        )
        await bus.publish(
            "private:state:nobitex:snapshot",
            PrivateStateSnapshot("nobitex", 1200, balances={}, open_orders={"real-1": order}),
        )

        snapshot = await asyncio.wait_for(state_queue.get(), timeout=1.0)
        assert isinstance(snapshot, ExecutionSnapshot)
        assert snapshot.live_orders["real-1"].order.price == 101.0

        await bus.publish(
            "private:state:nobitex:fills",
            Fill(
                exchange_order_id="real-1",
                actor_id="actor-1",
                exchange="nobitex",
                symbol="BTC_IRT",
                side="sell",
                price=101.0,
                qty=0.25,
                fee=0.0,
                timestamp_ms=1300,
            ),
        )

        filled = await asyncio.wait_for(state_queue.get(), timeout=1.0)
        assert filled.live_order_count == 0
    finally:
        await reconciler.stop()
