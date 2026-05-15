from __future__ import annotations

import asyncio

import pytest

from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.application.frame_preparation import clean_strategy_frame
from mm.application.orderbook_view import subtract_own_orders
from mm.application.strategy_runner import StrategyRunner
from mm.bus.memory import InMemoryBus
from mm.core.interfaces.actor import AbstractRiskEngine
from mm.core.models.execution import Action, ActorState, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.private import OpenOrder
from mm.execution.memory import RecordingOrderManager
from mm.risk.engine import PassThroughRiskEngine
from mm.runtime.config import ControllerRuntimeConfig, RiskRuntimeConfig
from mm.runtime.config import MarketMakerRuntimeConfig, MarketDataSourceConfig


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


def test_clean_strategy_frame_adjusts_primary_book():
    frame = StrategyInputFrame(primary=market_event())
    open_orders = {"own-bid": Order("nobitex", "BTC_IRT", "buy", 100.0, 4.0)}
    clean = clean_strategy_frame(frame, open_orders)
    assert clean.primary.orderbook.bids[0] == PriceLevel(100.0, 1.0)


class _FakeOrderManager:
    """Minimal manager exposing ``live_orders`` for own-order subtraction tests."""

    def __init__(self, live_orders: dict[str, OpenOrder] | None = None):
        self._live_orders = live_orders or {}
        self.synced: list[list[Action]] = []

    @property
    def live_orders(self) -> dict[str, OpenOrder]:
        return dict(self._live_orders)

    async def sync(self, actions: list[Action]) -> None:
        self.synced.append(actions)


class RejectAllRisk(AbstractRiskEngine):
    def validate(self, actions: list[Action], state: ActorState, event: MarketEvent) -> list[Action]:
        return []


def _minimal_config() -> ControllerRuntimeConfig:
    return ControllerRuntimeConfig(
        enabled=True,
        actor_id="test-runner",
        observe_only=False,
        action_start_after_steps=0,
        risk=RiskRuntimeConfig(enabled=True),
    )


def _minimal_decision(event: MarketEvent) -> StrategyDecision:
    order = Order(
        exchange=event.exchange,
        symbol=event.symbol,
        side="buy",
        price=99.0,
        qty=1.0,
        client_order_id="test-bid",
    )
    return StrategyDecision(
        actor_id="test-runner",
        exchange=event.exchange,
        symbol=event.symbol,
        t_end_ms=event.t_end_ms,
        frame_step=1,
        action=Action(actor_id="test-runner", orders_to_create=(order,)),
    )


@pytest.mark.asyncio
async def test_strategy_runner_subtracts_own_orders_before_decide(monkeypatch: pytest.MonkeyPatch):
    seen_bid_qty: list[float] = []

    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        seen_bid_qty.append(frame.primary.orderbook.bids[0].qty)
        return _minimal_decision(frame.primary)

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    bus = InMemoryBus()
    manager = _FakeOrderManager(
        {
            "own-bid": OpenOrder(
                exchange_order_id="own-bid",
                actor_id="test-runner",
                order=Order("nobitex", "BTC_IRT", "buy", 100.0, 4.0),
                remaining_qty=4.0,
                filled_qty=0.0,
                status="open",
                timestamp_ms=2000,
            )
        }
    )

    runtime_config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
    )
    runner = StrategyRunner(
        config=_minimal_config(),
        binding=runtime_config.binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=PassThroughRiskEngine(),
        bus=bus,
    )
    await runner.on_frame(StrategyInputFrame(primary=market_event()))

    assert seen_bid_qty == [1.0]


@pytest.mark.asyncio
async def test_strategy_runner_risk_blocks_execution(monkeypatch: pytest.MonkeyPatch):
    def fake_decide(self, frame: StrategyInputFrame, *, frame_step: int) -> StrategyDecision:
        return _minimal_decision(frame.primary)

    monkeypatch.setattr("mm.application.strategy_runner.MpcStrategyEngine.decide", fake_decide)

    manager = RecordingOrderManager()
    runner = StrategyRunner(
        config=_minimal_config(),
        binding=MarketMakerRuntimeConfig(
            primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        ).binding,
        frame_queue=asyncio.Queue(),
        order_managers={"nobitex": manager},
        risk_engine=RejectAllRisk(),
        bus=None,
    )
    approved = await runner.on_frame(StrategyInputFrame(primary=market_event()))

    assert approved == []
    assert manager.synced == []


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
        asks=(PriceLevel(101.0, 1.0)),
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
