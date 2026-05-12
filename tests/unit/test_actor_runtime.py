from __future__ import annotations

import asyncio

import pytest

from mm.application.actor import Actor
from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.application.orderbook_view import subtract_own_orders
from mm.bus.memory import InMemoryBus
from mm.core.interfaces.actor import AbstractRiskEngine, AbstractStrategy
from mm.core.models.execution import Action, ActorState, Fill, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyInputFrame
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
        self.seen_frames: list[StrategyInputFrame] = []
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

    async def on_strategy_frame(self, frame: StrategyInputFrame, state: ActorState) -> list[Action]:
        self.seen_frames.append(frame)
        return await self.on_market_event(frame.primary, state)

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
    assert strategy.seen_frames[0].hedge_status == "disabled"
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


@pytest.mark.asyncio
async def test_frame_assembler_passes_none_when_hedge_is_not_configured():
    queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    await queue.put(market_event())
    assembler = StrategyInputFrameAssembler(primary_queue=queue, hedge_grace_ms=0)

    frame = await assembler.next_frame()

    assert frame.primary.exchange == "nobitex"
    assert frame.hedge is None
    assert frame.hedge_status == "disabled"
    assert frame.has_hedge is False


@pytest.mark.asyncio
async def test_frame_assembler_aligns_fresh_hedge_event_to_primary():
    primary_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    hedge_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    hedge_event = _event_for("wallex", "BTC_IRT", t_start_ms=1000, t_end_ms=2000)
    primary_event = market_event()
    await hedge_queue.put(hedge_event)
    await primary_queue.put(primary_event)
    assembler = StrategyInputFrameAssembler(
        primary_queue=primary_queue,
        hedge_queue=hedge_queue,
        max_hedge_lag_ms=1000,
        hedge_grace_ms=0,
    )
    frame = await assembler.next_frame()

    assert frame.hedge == hedge_event
    assert frame.hedge_status == "ready"
    assert frame.hedge_lag_ms == 0


@pytest.mark.asyncio
async def test_frame_assembler_hides_stale_hedge_event():
    primary_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    hedge_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    await hedge_queue.put(_event_for("wallex", "BTC_IRT", t_start_ms=0, t_end_ms=500))
    await primary_queue.put(market_event())
    assembler = StrategyInputFrameAssembler(
        primary_queue=primary_queue,
        hedge_queue=hedge_queue,
        max_hedge_lag_ms=1000,
        hedge_grace_ms=0,
    )
    frame = await assembler.next_frame()

    assert frame.hedge is None
    assert frame.hedge_status == "stale"
    assert frame.hedge_lag_ms == 1500


def _event_for(exchange: str, symbol: str, *, t_start_ms: int, t_end_ms: int) -> MarketEvent:
    book = RawOrderbook(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=t_end_ms,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange=exchange,
        symbol=symbol,
        interval_ms=t_end_ms - t_start_ms,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        orderbook=book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )
