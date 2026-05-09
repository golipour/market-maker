from __future__ import annotations

import asyncio

import pytest

from mm.actors.actor import Actor
from mm.actors.orderbook_view import subtract_own_orders
from mm.bus.memory import InMemoryBus
from mm.core.interfaces.actor import AbstractRiskEngine, AbstractStrategy
from mm.core.models.execution import Action, ActorState, Fill, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.execution.memory import RecordingOrderManager


def market_event() -> MarketEvent:
    book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 5.0), PriceLevel(99.0, 3.0)),
        asks=(PriceLevel(101.0, 4.0), PriceLevel(102.0, 2.0)),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )


def test_subtract_own_orders_removes_and_reduces_matching_levels():
    state_orders = {
        "bid-1": Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0),
        "ask-1": Order("nobitex", "BTC_IRT", "sell", 102.0, 2.0),
        "other-symbol": Order("nobitex", "ETH_IRT", "buy", 100.0, 10.0),
    }

    clean = subtract_own_orders(market_event().orderbook, state_orders)

    assert clean.bids == (PriceLevel(100.0, 3.0), PriceLevel(99.0, 3.0))
    assert clean.asks == (PriceLevel(101.0, 4.0),)


class CapturingStrategy(AbstractStrategy):
    def __init__(self):
        self.seen_events: list[MarketEvent] = []
        self.fills: list[Fill] = []

    async def on_market_event(self, event: MarketEvent, state: ActorState) -> list[Action]:
        self.seen_events.append(event)
        return [
            Action(
                actor_id=state.actor_id,
                orders_to_create=(
                    Order(
                        exchange="nobitex",
                        symbol="BTC_IRT",
                        side="buy",
                        price=99.0,
                        qty=1.0,
                        client_order_id="next-bid",
                    ),
                ),
            )
        ]

    async def on_fill(self, fill: Fill, state: ActorState) -> None:
        self.fills.append(fill)


class CapturingRisk(AbstractRiskEngine):
    def __init__(self):
        self.seen_events: list[MarketEvent] = []

    def validate(self, actions: list[Action], state: ActorState, event: MarketEvent) -> list[Action]:
        self.seen_events.append(event)
        return actions


@pytest.mark.asyncio
async def test_actor_cleans_orderbook_and_routes_actions():
    strategy = CapturingStrategy()
    risk = CapturingRisk()
    manager = RecordingOrderManager()
    bus = InMemoryBus()
    journal_queue = bus.subscribe_queue("actor:journal:actor-1")
    state = ActorState(
        actor_id="actor-1",
        open_orders={
            "own-bid": Order("nobitex", "BTC_IRT", "buy", 100.0, 4.0),
        },
    )
    actor = Actor(
        actor_id="actor-1",
        strategy=strategy,
        risk_engine=risk,
        order_managers={"nobitex": manager},
        state=state,
        event_queues=[],
        bus=bus,
    )

    approved = await actor.on_market_event(market_event())
    journal = await asyncio.wait_for(journal_queue.get(), timeout=1.0)

    assert approved
    assert strategy.seen_events[0].orderbook.bids[0] == PriceLevel(100.0, 1.0)
    assert risk.seen_events[0].orderbook.bids[0] == PriceLevel(100.0, 1.0)
    assert manager.synced[0][0].orders_to_create[0].client_order_id == "next-bid"
    assert journal["actor_id"] == "actor-1"
    assert journal["approved_actions"] == 1


@pytest.mark.asyncio
async def test_actor_updates_state_on_fill_and_notifies_strategy():
    strategy = CapturingStrategy()
    actor = Actor(
        actor_id="actor-1",
        strategy=strategy,
        risk_engine=CapturingRisk(),
        order_managers={},
        state=ActorState(
            actor_id="actor-1",
            cash=1_000.0,
            inventory=0.0,
            open_orders={"ex-1": Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0)},
        ),
        event_queues=[],
    )

    await actor.on_fill(
        Fill(
            exchange_order_id="ex-1",
            actor_id="actor-1",
            exchange="nobitex",
            symbol="BTC_IRT",
            side="buy",
            price=100.0,
            qty=0.5,
            fee=0.1,
            timestamp_ms=2_100,
        )
    )

    assert actor.state.cash == pytest.approx(949.9)
    assert actor.state.inventory == pytest.approx(0.5)
    assert actor.state.open_orders == {}
    assert strategy.fills[0].exchange_order_id == "ex-1"

