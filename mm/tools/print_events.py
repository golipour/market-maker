"""Live bus event tap for debugging (canonical implementation).

Run: ``python -m mm.tools.print_events`` (or legacy ``python -m mm.runtime.print_events``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from mm.core.models.execution import Action, Fill
from mm.core.models.instrument import SymbolBinding, build_instrument_spec, instrument_contract
from mm.core.models.market import MarketEvent, RawOrderbook, RawTrade
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.exchanges.capabilities import get_exchange_capabilities
from mm.execution.order_manager import ExecutionSnapshot
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, PrivateRuntimeConfig, load_runtime_config
from mm.runtime.engine import RuntimeComponents, build_runtime_components


@dataclass(slots=True)
class PrintStats:
    printed: int = 0


def summarize_contract(
    exchange: str,
    symbol: str,
    mode: str,
    adapter: Any,
    binding: SymbolBinding,
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
            "orderbook": "REST snapshots and websocket snapshots/deltas are normalized into materialized snapshots before MarketEvent aggregation",
            "trades": "REST batches and websocket trade events are normalized, sorted, and deduplicated before MarketEvent aggregation",
        },
        "instrument_units": binding.as_contract(),
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
        instrument = build_instrument_spec(event.exchange, event.symbol)
        return {
            "type": "RawTrade",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "instrument": instrument_contract(instrument),
            "timestamp_ms": event.timestamp_ms,
            "side": event.side,
            "price": event.price,
            "qty": event.qty,
            "notional": event.price * event.qty,
            "unit_policy": "price=quote_per_base, qty=base, notional=quote",
        }
    if isinstance(event, RawOrderbook):
        book_health = summarize_book_health(event)
        instrument = build_instrument_spec(event.exchange, event.symbol)
        return {
            "type": "RawOrderbook",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "instrument": instrument_contract(instrument),
            "timestamp_ms": event.timestamp_ms,
            "is_snapshot": event.is_snapshot,
            "book_update_kind": "snapshot" if event.is_snapshot else "delta",
            "best_bid": event.bids[0].price if event.bids else None,
            "best_bid_qty": event.bids[0].qty if event.bids else None,
            "best_ask": event.asks[0].price if event.asks else None,
            "best_ask_qty": event.asks[0].qty if event.asks else None,
            "bid_levels": len(event.bids),
            "ask_levels": len(event.asks),
            "unit_policy": "price=quote_per_base, qty=base",
            **book_health,
        }
    if isinstance(event, MarketEvent):
        book_health = summarize_book_health(event.orderbook)
        instrument = build_instrument_spec(event.exchange, event.symbol)
        return {
            "type": "MarketEvent",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "instrument": instrument_contract(instrument),
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
            "orderbook_timestamp_ms": event.orderbook.timestamp_ms,
            "orderbook_lag_ms": event.t_end_ms - event.orderbook.timestamp_ms,
            "unit_policy": "prices/vwap/microprice=quote_per_base, volumes=base, notionals=quote",
            **book_health,
        }
    if isinstance(event, StrategyInputFrame):
        primary_book_health = summarize_book_health(event.primary.orderbook)
        hedge_book_health = summarize_book_health(event.hedge.orderbook) if event.hedge is not None else {}
        return {
            "type": "StrategyInputFrame",
            "primary_exchange": event.primary.exchange,
            "primary_symbol": event.primary.symbol,
            "interval_ms": event.primary.interval_ms,
            "t_start_ms": event.primary.t_start_ms,
            "t_end_ms": event.primary.t_end_ms,
            "primary_best_bid": event.primary.orderbook.bids[0].price if event.primary.orderbook.bids else None,
            "primary_best_ask": event.primary.orderbook.asks[0].price if event.primary.orderbook.asks else None,
            "primary_mid": primary_book_health["mid"],
            "primary_trade_count": len(event.primary.trades),
            "hedge_status": event.hedge_status,
            "has_hedge": event.has_hedge,
            "hedge_exchange": event.hedge.exchange if event.hedge is not None else None,
            "hedge_symbol": event.hedge.symbol if event.hedge is not None else None,
            "hedge_t_end_ms": event.hedge.t_end_ms if event.hedge is not None else None,
            "hedge_lag_ms": event.hedge_lag_ms,
            "max_hedge_lag_ms": event.max_hedge_lag_ms,
            "hedge_best_bid": event.hedge.orderbook.bids[0].price if event.hedge and event.hedge.orderbook.bids else None,
            "hedge_best_ask": event.hedge.orderbook.asks[0].price if event.hedge and event.hedge.orderbook.asks else None,
            "hedge_mid": hedge_book_health.get("mid"),
            "hedge_trade_count": len(event.hedge.trades) if event.hedge is not None else None,
            "primary_private_snapshot": event.primary_private is not None,
            "primary_balance_assets": sorted(event.primary_private.balances) if event.primary_private is not None else [],
            "primary_open_order_count": (
                len(event.primary_private.open_orders) if event.primary_private is not None else None
            ),
            "hedge_private_snapshot": event.hedge_private is not None,
            "created_at_ms": event.created_at_ms,
            "frame_contract": (
                "primary is always present; hedge is None unless a configured hedge exchange has a fresh aligned event"
            ),
            "unit_policy": "prices=quote_per_base, quantities/volumes=base, notionals=quote",
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
        instrument = build_instrument_spec(event.order.exchange, event.order.symbol)
        return {
            "type": "OpenOrder",
            "exchange": event.order.exchange,
            "symbol": event.order.symbol,
            "instrument": instrument_contract(instrument),
            "exchange_order_id": event.exchange_order_id,
            "actor_id": event.actor_id,
            "side": event.order.side,
            "price": event.order.price,
            "remaining_qty": event.remaining_qty,
            "filled_qty": event.filled_qty,
            "status": event.status,
            "timestamp_ms": event.timestamp_ms,
            "unit_policy": "price=quote_per_base, remaining_qty/filled_qty=base",
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
        instrument = build_instrument_spec(event.exchange, event.symbol)
        return {
            "type": "Fill",
            "exchange": event.exchange,
            "symbol": event.symbol,
            "instrument": instrument_contract(instrument),
            "exchange_order_id": event.exchange_order_id,
            "actor_id": event.actor_id,
            "side": event.side,
            "price": event.price,
            "qty": event.qty,
            "fee": event.fee,
            "timestamp_ms": event.timestamp_ms,
            "unit_policy": "price=quote_per_base, qty=base",
        }
    if isinstance(event, ExecutionSnapshot):
        live_orders = sorted(event.live_orders.values(), key=lambda item: item.exchange_order_id)
        return {
            "type": "ExecutionSnapshot",
            "exchange": event.exchange,
            "timestamp_ms": event.timestamp_ms,
            "live_order_count": event.live_order_count,
            "live_order_ids": [item.exchange_order_id for item in live_orders[:10]],
            "live_orders": [
                {
                    "exchange_order_id": item.exchange_order_id,
                    "actor_id": item.actor_id,
                    "symbol": item.order.symbol,
                    "side": item.order.side,
                    "price": item.order.price,
                    "remaining_qty": item.remaining_qty,
                    "filled_qty": item.filled_qty,
                    "status": item.status,
                }
                for item in live_orders[:5]
            ],
            "note": "execution state is mirrored from private exchange state; print_events does not place or cancel orders",
        }
    if isinstance(event, Action):
        return {
            "type": "StrategyAction",
            "actor_id": event.actor_id,
            "create_count": len(event.orders_to_create),
            "cancel_count": len(event.orders_to_cancel),
            "orders": [
                {
                    "exchange": order.exchange,
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": order.price,
                    "qty": order.qty,
                    "notional": order.price * order.qty,
                    "client_order_id": order.client_order_id,
                }
                for order in event.orders_to_create[:10]
            ],
            "unit_policy": "price=quote_per_base, qty=base, notional=quote",
        }
    if isinstance(event, dict) and str(event.get("type", "")).startswith(("Execution", "Place", "Cancel")):
        return event
    if isinstance(event, dict) and str(event.get("type", "")).startswith(("Controller", "Strategy")):
        return event
    if isinstance(event, dict) and "error" in event:
        return {"type": "MarketDataError", **event}
    return {"type": type(event).__name__, "repr": repr(event)}


def summarize_book_health(book: RawOrderbook) -> dict[str, Any]:
    best_bid = book.bids[0].price if book.bids else None
    best_ask = book.asks[0].price if book.asks else None
    spread = None
    mid = None
    is_crossed = False
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        is_crossed = spread <= 0
    return {
        "spread": spread,
        "mid": mid,
        "is_crossed": is_crossed,
        "is_one_sided": not book.bids or not book.asks,
    }


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
    if args.config:
        runtime_config = load_runtime_config(args.config)
        runtime = await build_runtime_components(runtime_config)
        contract = runtime_config.as_contract()
        contract["type"] = "RuntimeConfigCheck"
        contract["selected_public_market_transports"] = {
            role: selection.transport
            for role, selection in runtime.public_market_selections.items()
        }
        contract["selected_private_state_modes"] = {
            role: selection.mode
            for role, selection in runtime.private_state_selections.items()
        }
        contract["controller_runner_enabled"] = runtime.controller_runner is not None
        print(json.dumps(contract, ensure_ascii=False))
        await run_print_loop(runtime, args)
        return

    exchange = args.exchange.lower().strip()
    mode = args.mode.lower().strip()
    hedge_exchange = args.hedge_exchange.lower().strip() if args.hedge_exchange else None
    hedge_mode = (args.hedge_mode or mode).lower().strip()
    hedge_symbol = args.hedge_symbol or args.symbol
    api_token = resolve_api_token(exchange, args.private_api_token, args.private_api_token_env)
    hedge_api_token = (
        resolve_api_token(hedge_exchange, args.hedge_private_api_token, args.hedge_private_api_token_env)
        if hedge_exchange
        else None
    )
    runtime_config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(
            role="primary",
            exchange=exchange,
            symbol=args.symbol,
            mode=mode,
            depth_limit=args.depth,
            trades_limit=args.trades,
            poll_interval_ms=args.poll_ms,
        ),
        hedge=(
            MarketDataSourceConfig(
                role="hedge",
                exchange=hedge_exchange,
                symbol=hedge_symbol,
                mode=hedge_mode,
                depth_limit=args.depth,
                trades_limit=args.trades,
                poll_interval_ms=args.poll_ms,
            )
            if hedge_exchange
            else None
        ),
        interval_ms=args.interval_ms,
        max_hedge_lag_ms=args.max_hedge_lag_ms,
        hedge_grace_ms=args.hedge_grace_ms,
        primary_private=PrivateRuntimeConfig(
            mode=args.private_mode,
            api_token=api_token,
            poll_interval_ms=args.private_poll_ms,
        ),
        hedge_private=(
            PrivateRuntimeConfig(
                mode=args.hedge_private_mode,
                api_token=hedge_api_token,
                poll_interval_ms=args.hedge_private_poll_ms,
            )
            if hedge_exchange
            else None
        ),
    )
    runtime = await build_runtime_components(runtime_config)
    private_contract = {
        "requested_transport_mode": args.private_mode,
        "selected_transport": runtime.private_state_selections["primary"].mode
        if "primary" in runtime.private_state_selections
        else None,
        "adapter": type(runtime.private_state_selections["primary"].adapter).__name__
        if "primary" in runtime.private_state_selections
        else None,
        "scope": "private_balances_and_orders_only",
        "enabled": bool(api_token),
        "token_configured": bool(api_token),
        "balances": "Balance stream uses float asset units",
        "open_orders": "OpenOrder stream uses float base qty and quote-per-base price",
        "fallback": "uses REST when websocket private state is not fully supported",
    }
    if api_token:
        private_contract["execution"] = {
            "enabled": True,
            "order_command_transport": "rest",
            "print_events_mode": "observe_only",
            "safety": "does not place or cancel orders unless an actor calls OrderManager.sync",
            "live_orders": "ExecutionSnapshot mirrors private open orders for own-order subtraction and execution recovery",
        }
    selection = runtime.public_market_selections["primary"]
    contract = summarize_contract(
        exchange,
        args.symbol,
        selection.requested_mode,
        selection.adapter,
        runtime_config.binding,
        private_contract,
    )
    contract["runtime_config"] = runtime_config.as_contract()
    contract["selected_public_market_transport"] = selection.transport
    if hedge_exchange is not None:
        hedge_selection = runtime.public_market_selections["hedge"]
        contract["hedge_public_market"] = {
            "exchange": hedge_exchange,
            "symbol": hedge_symbol,
            "requested_transport_mode": hedge_selection.requested_mode,
            "selected_public_market_transport": hedge_selection.transport,
            "public_market_adapter": type(hedge_selection.adapter).__name__,
            "role": "hedge",
            "hedge_is_anchor": True,
            "symbol_policy": "same domain symbol as primary by default; no tick-size conversion is applied",
        }
        contract["hedge_private_state"] = {
            "requested_transport_mode": args.hedge_private_mode,
            "selected_transport": runtime.private_state_selections["hedge"].mode
            if "hedge" in runtime.private_state_selections
            else None,
            "adapter": type(runtime.private_state_selections["hedge"].adapter).__name__
            if "hedge" in runtime.private_state_selections
            else None,
            "enabled": bool(hedge_api_token),
            "token_configured": bool(hedge_api_token),
        }
    contract["strategy_input_frame"] = {
        "topic": f"strategy:frames:{exchange}:{args.symbol}:{args.interval_ms}",
        "primary": "always present",
        "hedge": "None when --hedge-exchange is omitted, missing, or stale",
        "hedge_is_anchor": True,
        "max_hedge_lag_ms": args.max_hedge_lag_ms,
        "hedge_grace_ms": args.hedge_grace_ms,
    }
    print(json.dumps(contract, ensure_ascii=False))

    await run_print_loop(runtime, args)


async def run_print_loop(runtime: RuntimeComponents, args: argparse.Namespace) -> None:
    queues = build_print_queues(runtime, include_raw=args.raw)

    stats = PrintStats()
    print_task = asyncio.create_task(print_events(queues, args.events, stats))
    try:
        await runtime.start()
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
        print_task.cancel()
        await asyncio.gather(print_task, return_exceptions=True)
        await runtime.stop()


def build_print_queues(runtime: RuntimeComponents, *, include_raw: bool) -> list[tuple[str, asyncio.Queue]]:
    config = runtime.config
    bus = runtime.bus
    primary_event_topic = config.primary.market_event_topic(config.interval_ms)
    primary_error_topic = config.primary.market_error_topic()
    queues: list[tuple[str, asyncio.Queue]] = [
        (config.frame_topic, bus.subscribe_queue(config.frame_topic)),
        (primary_event_topic, bus.subscribe_queue(primary_event_topic)),
        (primary_error_topic, bus.subscribe_queue(primary_error_topic)),
    ]
    if config.hedge is not None:
        hedge_event_topic = config.hedge.market_event_topic(config.interval_ms)
        hedge_error_topic = config.hedge.market_error_topic()
        queues.extend([
            (hedge_event_topic, bus.subscribe_queue(hedge_event_topic)),
            (hedge_error_topic, bus.subscribe_queue(hedge_error_topic)),
        ])
    if include_raw:
        trade_topic = config.primary.raw_trade_topic()
        book_topic = config.primary.raw_orderbook_topic()
        queues.extend([
            (trade_topic, bus.subscribe_queue(trade_topic)),
            (book_topic, bus.subscribe_queue(book_topic)),
        ])
        if config.hedge is not None:
            hedge_trade_topic = config.hedge.raw_trade_topic()
            hedge_book_topic = config.hedge.raw_orderbook_topic()
            queues.extend([
                (hedge_trade_topic, bus.subscribe_queue(hedge_trade_topic)),
                (hedge_book_topic, bus.subscribe_queue(hedge_book_topic)),
            ])
    for role, source in (("primary", config.primary), ("hedge", config.hedge)):
        if source is None or role not in runtime.private_state_selections:
            continue
        exchange = source.normalized_exchange()
        private_topics = [
            f"private:state:{exchange}:snapshot",
            f"private:state:{exchange}:balances",
            f"private:state:{exchange}:open_orders",
            f"private:state:{exchange}:order_updates",
            f"private:state:{exchange}:fills",
            f"private:state:{exchange}:errors",
        ]
        queues.extend((topic, bus.subscribe_queue(topic)) for topic in private_topics)
        execution_topics = [
            f"execution:state:{exchange}",
            f"execution:orders:{exchange}",
        ]
        queues.extend((topic, bus.subscribe_queue(topic)) for topic in execution_topics)
    if config.controller is not None and config.controller.enabled:
        queues.extend(
            [
                (
                    config.controller.normalized_diagnostics_topic,
                    bus.subscribe_queue(config.controller.normalized_diagnostics_topic),
                ),
                (
                    config.controller.normalized_action_topic,
                    bus.subscribe_queue(config.controller.normalized_action_topic),
                ),
            ]
        )
    return queues


def resolve_api_token(exchange: str, explicit: str | None, env_name: str | None) -> str | None:
    if explicit:
        return explicit
    if env_name:
        return os.environ.get(env_name)
    return os.environ.get(f"{exchange.upper()}_API_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live market-data bus events.")
    parser.add_argument("--config", default=None, help="Path to a JSON runtime config file.")
    parser.add_argument("--exchange", choices=["nobitex", "wallex"], default=None)
    parser.add_argument("--symbol", default="BTC_IRT")
    parser.add_argument("--hedge-exchange", choices=["nobitex", "wallex"], default=None)
    parser.add_argument("--hedge-symbol", default=None)
    parser.add_argument("--hedge-mode", choices=["rest", "websocket"], default=None)
    parser.add_argument("--mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--private-mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--private-api-token", default=None)
    parser.add_argument("--private-api-token-env", default=None)
    parser.add_argument("--private-poll-ms", type=int, default=5000)
    parser.add_argument("--hedge-private-mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--hedge-private-api-token", default=None)
    parser.add_argument("--hedge-private-api-token-env", default=None)
    parser.add_argument("--hedge-private-poll-ms", type=int, default=5000)
    parser.add_argument("--interval-ms", type=int, default=5000)
    parser.add_argument("--poll-ms", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--trades", type=int, default=100)
    parser.add_argument("--events", type=int, default=10000, help="Stop after this many events; use 0 to run until timeout.")
    parser.add_argument("--timeout-s", type=float, default=60000.0)
    parser.add_argument("--max-hedge-lag-ms", type=int, default=2000)
    parser.add_argument("--hedge-grace-ms", type=int, default=250)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if args.config is None and args.exchange is None:
        parser.error("--exchange is required unless --config is provided")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
