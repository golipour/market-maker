from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from mm.aggregation.scheduler import IntervalScheduler
from mm.bus.memory import InMemoryBus
from mm.core.clock import WallClock
from mm.core.models.execution import Fill
from mm.core.models.market import MarketEvent, RawOrderbook, RawTrade
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.capabilities import get_exchange_capabilities
from mm.exchanges.factory import select_private_state_adapter, select_public_market_data_adapter
from mm.private_state.reconciler import PrivateStateReconciler
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class PrintStats:
    printed: int = 0


def build_adapter_selection(exchange: str, symbol: str, mode: str, depth: int, trades: int, poll_ms: int):
    return select_public_market_data_adapter(
        exchange,
        symbol,
        mode=mode,
        depth_limit=depth,
        trades_limit=trades,
        poll_interval_ms=poll_ms,
    )


def build_private_state_selection(
    exchange: str,
    symbol: str,
    mode: str,
    api_token: str | None,
    poll_ms: int,
):
    return select_private_state_adapter(
        exchange,
        symbol,
        mode=mode,
        api_token=api_token,
        poll_interval_ms=poll_ms,
    )


def summarize_contract(
    exchange: str,
    symbol: str,
    mode: str,
    adapter: Any,
    private_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = get_exchange_capabilities(exchange)
    public_caps = capabilities.public_market_data
    out = {
        "type": "ContractCheck",
        "exchange": exchange,
        "symbol": symbol,
        "requested_transport_mode": mode,
        "public_market_adapter": type(adapter).__name__,
        "market_data_contract": {
            "scope": "public_only",
            "single_transport": True,
            "prices": "float quote-per-base",
            "quantities": "float base units",
            "orderbook": "RawOrderbook snapshots/deltas normalized before aggregation",
            "trades": "RawTrade stream normalized before aggregation",
        },
        "complete_public_market_transports": {
            "rest": public_caps.rest,
            "websocket": public_caps.websocket,
        },
        "order_commands": {
            "requested_transport_mode": mode,
            "selected_transport": "rest",
            "reason": "private order commands are always REST",
        },
        "capabilities": asdict(capabilities),
    }
    if private_state is not None:
        out["private_state"] = private_state
    return out


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
            "is_snapshot": event.is_snapshot,
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
    if isinstance(event, PrivateStateSnapshot):
        return {
            "type": "PrivateStateSnapshot",
            "exchange": event.exchange,
            "timestamp_ms": event.timestamp_ms,
            "balance_assets": sorted(event.balances),
            "balance_count": len(event.balances),
            "open_order_count": len(event.open_orders),
            "open_order_ids": sorted(event.open_orders)[:10],
        }
    if isinstance(event, Balance):
        return {
            "type": "Balance",
            "exchange": event.exchange,
            "asset": event.asset,
            "free": event.free,
            "locked": event.locked,
            "timestamp_ms": event.timestamp_ms,
        }
    if isinstance(event, OpenOrder):
        return {
            "type": "OpenOrder",
            "exchange": event.order.exchange,
            "symbol": event.order.symbol,
            "exchange_order_id": event.exchange_order_id,
            "actor_id": event.actor_id,
            "side": event.order.side,
            "price": event.order.price,
            "remaining_qty": event.remaining_qty,
            "filled_qty": event.filled_qty,
            "status": event.status,
            "timestamp_ms": event.timestamp_ms,
        }
    if isinstance(event, OrderUpdate):
        return {
            "type": "OrderUpdate",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "exchange_order_id": event.exchange_order_id,
            "actor_id": event.actor_id,
            "status": event.status,
            "remaining_qty": event.remaining_qty,
            "filled_qty": event.filled_qty,
            "timestamp_ms": event.timestamp_ms,
        }
    if isinstance(event, Fill):
        return {
            "type": "Fill",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "exchange_order_id": event.exchange_order_id,
            "actor_id": event.actor_id,
            "side": event.side,
            "price": event.price,
            "qty": event.qty,
            "fee": event.fee,
            "timestamp_ms": event.timestamp_ms,
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
    selection = build_adapter_selection(
        exchange,
        args.symbol,
        mode,  # type: ignore[arg-type]
        args.depth,
        args.trades,
        args.poll_ms,
    )
    adapter = selection.adapter
    private_selection = None
    private_reconciler = None
    api_token = resolve_api_token(exchange, args.private_api_token, args.private_api_token_env)
    private_selection = build_private_state_selection(
        exchange,
        args.symbol,
        mode,
        api_token,
        args.private_poll_ms,
    )
    bus = InMemoryBus()
    engine = build_symbol_engine(exchange=exchange, symbol=args.symbol, adapter=adapter, bus=bus)
    if api_token:
        private_reconciler = PrivateStateReconciler(
            exchange=exchange,
            adapter=private_selection.adapter,
            bus=bus,
        )
    scheduler = IntervalScheduler(WallClock())
    await engine.publish_on_intervals(scheduler, [args.interval_ms])
    private_contract = {
        "requested_transport_mode": private_selection.requested_mode,
        "selected_transport": private_selection.mode,
        "adapter": type(private_selection.adapter).__name__,
        "scope": "private_balances_and_orders_only",
        "enabled": bool(api_token),
        "token_configured": bool(api_token),
        "balances": "Balance stream uses float asset units",
        "open_orders": "OpenOrder stream uses float base qty and quote-per-base price",
        "fallback": "uses REST when websocket private state is not fully supported",
    }
    contract = summarize_contract(exchange, args.symbol, selection.requested_mode, adapter, private_contract)
    contract["selected_public_market_transport"] = selection.transport
    print(json.dumps(contract, ensure_ascii=False))

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
    if private_reconciler is not None:
        private_topics = [
            f"private:state:{exchange}:snapshot",
            f"private:state:{exchange}:balances",
            f"private:state:{exchange}:open_orders",
            f"private:state:{exchange}:order_updates",
            f"private:state:{exchange}:fills",
            f"private:state:{exchange}:errors",
        ]
        queues.extend((topic, bus.subscribe_queue(topic)) for topic in private_topics)

    stats = PrintStats()
    print_task = asyncio.create_task(print_events(queues, args.events, stats))
    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await engine.start()
        if private_reconciler is not None:
            await private_reconciler.start()
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
        if private_reconciler is not None:
            await private_reconciler.stop()
        await engine.stop()


def resolve_api_token(exchange: str, explicit: str | None, env_name: str | None) -> str | None:
    if explicit:
        return explicit
    if env_name:
        return os.environ.get(env_name)
    return os.environ.get(f"{exchange.upper()}_API_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live market-data bus events.")
    parser.add_argument("--exchange", choices=["nobitex", "wallex"], required=True)
    parser.add_argument("--symbol", default="BTC_IRT")
    parser.add_argument("--mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--private-api-token", default=None)
    parser.add_argument("--private-api-token-env", default=None)
    parser.add_argument("--private-poll-ms", type=int, default=5000)
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
