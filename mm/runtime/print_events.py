from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mm.aggregation.scheduler import IntervalScheduler
from mm.bus.memory import InMemoryBus
from mm.core.clock import WallClock
from mm.core.models.market import MarketEvent, RawOrderbook, RawTrade
from mm.exchanges.nobitex import NobitexMarketDataAdapter
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.wallex import WallexMarketDataAdapter
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class PrintStats:
    printed: int = 0


def build_adapter(exchange: str, symbol: str, mode: TransportMode, depth: int, trades: int, poll_ms: int):
    if exchange == "nobitex":
        return NobitexMarketDataAdapter(
            symbol,
            mode=mode,
            depth_limit=depth,
            trades_limit=trades,
            poll_interval_ms=poll_ms,
        )
    if exchange == "wallex":
        return WallexMarketDataAdapter(
            symbol,
            mode=mode,
            depth_limit=depth,
            trades_limit=trades,
            poll_interval_ms=poll_ms,
        )
    raise ValueError(f"unsupported exchange: {exchange}")


def summarize_event(event: Any) -> dict[str, Any]:
    if isinstance(event, RawTrade):
        return {
            "type": "RawTrade",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "timestamp_ms": event.timestamp_ms,
            "side": event.side,
            "price": event.price,
            "qty": event.qty,
        }
    if isinstance(event, RawOrderbook):
        return {
            "type": "RawOrderbook",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "timestamp_ms": event.timestamp_ms,
            "best_bid": event.bids[0].price if event.bids else None,
            "best_bid_qty": event.bids[0].qty if event.bids else None,
            "best_ask": event.asks[0].price if event.asks else None,
            "best_ask_qty": event.asks[0].qty if event.asks else None,
            "bid_levels": len(event.bids),
            "ask_levels": len(event.asks),
        }
    if isinstance(event, MarketEvent):
        return {
            "type": "MarketEvent",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "interval_ms": event.interval_ms,
            "t_start_ms": event.t_start_ms,
            "t_end_ms": event.t_end_ms,
            "trade_count": len(event.trades),
            "buy_volume": event.buy_volume,
            "sell_volume": event.sell_volume,
            "vwap": event.vwap,
            "imbalance": event.imbalance,
            "microprice": event.microprice,
            "best_bid": event.orderbook.bids[0].price if event.orderbook.bids else None,
            "best_ask": event.orderbook.asks[0].price if event.orderbook.asks else None,
        }
    if isinstance(event, dict) and "error" in event:
        return {"type": "MarketDataError", **event}
    return {"type": type(event).__name__, "repr": repr(event)}


async def print_events(queues: list[tuple[str, asyncio.Queue]], event_count: int, stats: PrintStats) -> None:
    pending = {
        asyncio.create_task(queue.get()): label
        for label, queue in queues
    }
    printed = 0
    try:
        while event_count <= 0 or printed < event_count:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                label = pending.pop(task)
                event = task.result()
                print(json.dumps({"topic": label, **summarize_event(event)}, ensure_ascii=False))
                printed += 1
                stats.printed = printed
                if event_count > 0 and printed >= event_count:
                    break
                queue = next(queue for candidate_label, queue in queues if candidate_label == label)
                pending[asyncio.create_task(queue.get())] = label
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def run(args: argparse.Namespace) -> None:
    exchange = args.exchange.lower().strip()
    mode = args.mode.lower().strip()
    adapter = build_adapter(
        exchange,
        args.symbol,
        mode,  # type: ignore[arg-type]
        args.depth,
        args.trades,
        args.poll_ms,
    )
    bus = InMemoryBus()
    engine = build_symbol_engine(exchange=exchange, symbol=args.symbol, adapter=adapter, bus=bus)
    scheduler = IntervalScheduler(WallClock())
    await engine.publish_on_intervals(scheduler, [args.interval_ms])

    event_topic = f"market:events:{exchange}:{args.symbol}:{args.interval_ms}"
    error_topic = f"market:errors:{exchange}:{args.symbol}"
    queues: list[tuple[str, asyncio.Queue]] = [
        (event_topic, bus.subscribe_queue(event_topic)),
        (error_topic, bus.subscribe_queue(error_topic)),
    ]
    if args.raw:
        trade_topic = f"market:raw:{exchange}:{args.symbol}:trades"
        book_topic = f"market:raw:{exchange}:{args.symbol}:orderbook"
        queues.extend([
            (trade_topic, bus.subscribe_queue(trade_topic)),
            (book_topic, bus.subscribe_queue(book_topic)),
        ])

    stats = PrintStats()
    print_task = asyncio.create_task(print_events(queues, args.events, stats))
    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await engine.start()
        try:
            await asyncio.wait_for(print_task, timeout=args.timeout_s)
        except asyncio.TimeoutError:
            print(json.dumps({
                "type": "Timeout",
                "printed": stats.printed,
                "target_events": args.events,
                "timeout_s": args.timeout_s,
                "message": (
                    f"printed {stats.printed} event(s) before timeout_s={args.timeout_s}; "
                    f"target_events={args.events}"
                ),
            }))
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        print_task.cancel()
        await asyncio.gather(print_task, return_exceptions=True)
        await engine.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live market-data bus events.")
    parser.add_argument("--exchange", choices=["nobitex", "wallex"], required=True)
    parser.add_argument("--symbol", default="BTC_IRT")
    parser.add_argument("--mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--interval-ms", type=int, default=5000)
    parser.add_argument("--poll-ms", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--trades", type=int, default=100)
    parser.add_argument("--events", type=int, default=10000, help="Stop after this many events; use 0 to run until timeout.")
    parser.add_argument("--timeout-s", type=float, default=60000.0)
    parser.add_argument("--raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
