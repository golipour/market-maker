from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from mm.application.orderbook_view import subtract_own_orders
from mm.bus.base import AbstractEventBus
from mm.core.interfaces.actor import AbstractOrderManager, AbstractRiskEngine, AbstractStrategy
from mm.core.models.execution import Action, ActorState, Fill
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame
from mm.observability import counter_inc, gauge


class Actor:
    def __init__(
        self,
        actor_id: str,
        strategy: AbstractStrategy,
        risk_engine: AbstractRiskEngine,
        order_managers: dict[str, AbstractOrderManager],
        state: ActorState,
        event_queues: list[asyncio.Queue[MarketEvent]],
        bus: AbstractEventBus | None = None,
    ):
        self.actor_id = actor_id
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.order_managers = order_managers
        self.state = state
        self.event_queues = event_queues
        self.bus = bus
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._listen(queue)) for queue in self.event_queues]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def run(self) -> None:
        await self.start()
        await asyncio.gather(*self._tasks)

    async def _listen(self, queue: asyncio.Queue[MarketEvent]) -> None:
        while True:
            event = await queue.get()
            await self.on_market_event(event)

    async def on_market_event(self, event: MarketEvent) -> list[Action]:
        return await self.on_strategy_frame(StrategyInputFrame(primary=event))

    async def on_strategy_frame(self, frame: StrategyInputFrame) -> list[Action]:
        t0 = time.monotonic_ns()
        clean_primary = _replace_event_orderbook(
            frame.primary,
            subtract_own_orders(frame.primary.orderbook, self.state.open_orders),
        )
        clean_hedge = None
        if frame.hedge is not None:
            clean_hedge = _replace_event_orderbook(
                frame.hedge,
                subtract_own_orders(frame.hedge.orderbook, self.state.open_orders),
            )
        clean_frame = StrategyInputFrame(
            primary=clean_primary,
            hedge=clean_hedge,
            primary_private=frame.primary_private,
            hedge_private=frame.hedge_private,
            hedge_status=frame.hedge_status,
            hedge_lag_ms=frame.hedge_lag_ms,
            max_hedge_lag_ms=frame.max_hedge_lag_ms,
            created_at_ms=frame.created_at_ms,
        )
        proposed = await self.strategy.on_strategy_frame(clean_frame, self.state)
        approved = self.risk_engine.validate_frame(proposed, self.state, clean_frame)
        if len(approved) < len(proposed) and self.bus is not None:
            await self.bus.publish(
                "risk:events",
                {
                    "type": "RiskActionsFiltered",
                    "actor_id": self.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "proposed_actions": len(proposed),
                    "approved_actions": len(approved),
                },
            )
        counter_inc("mm_actor_frames_total", labels={"actor_id": self.actor_id})
        gauge("mm_actor_latency_ms", (time.monotonic_ns() - t0) / 1_000_000, labels={"actor_id": self.actor_id})
        await self._route_actions(approved)
        if self.bus is not None:
            await self.bus.publish(
                f"actor:journal:{self.actor_id}",
                {
                    "actor_id": self.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "hedge_status": frame.hedge_status,
                    "hedge_lag_ms": frame.hedge_lag_ms,
                    "proposed_actions": len(proposed),
                    "approved_actions": len(approved),
                    "latency_ms": (time.monotonic_ns() - t0) / 1_000_000,
                },
            )
        return approved

    async def on_fill(self, fill: Fill) -> None:
        if fill.side == "buy":
            self.state.cash -= fill.price * fill.qty + fill.fee
            self.state.inventory += fill.qty
        else:
            self.state.cash += fill.price * fill.qty - fill.fee
            self.state.inventory -= fill.qty
        self.state.open_orders.pop(fill.exchange_order_id, None)
        await self.strategy.on_fill(fill, self.state)

    async def _route_actions(self, actions: list[Action]) -> None:
        by_exchange: dict[str, list[Action]] = defaultdict(list)
        for action in actions:
            creates_by_exchange: dict[str, list] = defaultdict(list)
            for order in action.orders_to_create:
                creates_by_exchange[order.exchange].append(order)
            cancel_ids = action.orders_to_cancel
            for exchange, orders in creates_by_exchange.items():
                by_exchange[exchange].append(
                    Action(
                        actor_id=action.actor_id,
                        orders_to_create=tuple(orders),
                        orders_to_cancel=cancel_ids,
                    )
                )
            if cancel_ids and not creates_by_exchange:
                for exchange in self.order_managers:
                    by_exchange[exchange].append(
                        Action(actor_id=action.actor_id, orders_to_create=(), orders_to_cancel=cancel_ids)
                    )
        for exchange, routed in by_exchange.items():
            manager = self.order_managers.get(exchange)
            if manager is not None:
                await manager.sync(routed)


def _replace_event_orderbook(event: MarketEvent, orderbook) -> MarketEvent:
    return MarketEvent(
        exchange=event.exchange,
        symbol=event.symbol,
        interval_ms=event.interval_ms,
        t_start_ms=event.t_start_ms,
        t_end_ms=event.t_end_ms,
        orderbook=orderbook,
        trades=event.trades,
        buy_volume=event.buy_volume,
        sell_volume=event.sell_volume,
        vwap=event.vwap,
        imbalance=event.imbalance,
        microprice=event.microprice,
    )
