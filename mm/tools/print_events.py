"""Live bus event tap for debugging (canonical implementation).

Run: ``python -m mm.tools.print_events --config <path>`` (see ``--help``).
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
from mm.core.payload_schema import MARKET_EVENT_TYPE, market_event_event, market_event_from_payload
from mm.core.wallet import balance_event_payload
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.exchanges.capabilities import get_exchange_capabilities
from mm.execution.order_manager import ExecutionSnapshot
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, PrivateRuntimeConfig, load_runtime_config
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.engine import RuntimeComponents, build_runtime_components, resolve_build_options


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
    if isinstance(event, dict) and event.get("type") == MARKET_EVENT_TYPE:
        parsed = market_event_from_payload(event)
        if parsed is not None:
            return summarize_event(parsed)
    if isinstance(event, StrategyInputFrame):
        local_book_health = summarize_book_health(event.local.orderbook)
        global_book_health = (
            summarize_book_health(event.global_.orderbook) if event.global_ is not None else {}
        )
        return {
            "type": "StrategyInputFrame",
            "local_exchange": event.local.exchange,
            "local_symbol": event.local.symbol,
            "interval_ms": event.local.interval_ms,
            "t_start_ms": event.local.t_start_ms,
            "t_end_ms": event.local.t_end_ms,
            "local_best_bid": event.local.orderbook.bids[0].price if event.local.orderbook.bids else None,
            "local_best_ask": event.local.orderbook.asks[0].price if event.local.orderbook.asks else None,
            "local_mid": local_book_health["mid"],
            "local_trade_count": len(event.local.trades),
            "global_status": event.global_status,
            "has_global": event.has_global,
            "global_exchange": event.global_.exchange if event.global_ is not None else None,
            "global_symbol": event.global_.symbol if event.global_ is not None else None,
            "global_t_end_ms": event.global_.t_end_ms if event.global_ is not None else None,
            "global_lag_ms": event.global_lag_ms,
            "max_global_lag_ms": event.max_global_lag_ms,
            "global_best_bid": (
                event.global_.orderbook.bids[0].price
                if event.global_ and event.global_.orderbook.bids
                else None
            ),
            "global_best_ask": (
                event.global_.orderbook.asks[0].price
                if event.global_ and event.global_.orderbook.asks
                else None
            ),
            "global_mid": global_book_health.get("mid"),
            "global_trade_count": len(event.global_.trades) if event.global_ is not None else None,
            "local_private_snapshot": event.local_private is not None,
            "local_balance_assets": sorted(event.local_private.balances) if event.local_private is not None else [],
            "local_open_order_count": (
                len(event.local_private.open_orders) if event.local_private is not None else None
            ),
            "global_private_snapshot": event.global_private is not None,
            "created_at_ms": event.created_at_ms,
            "frame_contract": (
                "local is always present; global is None unless a configured global exchange has a fresh aligned event"
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
    if isinstance(event, dict) and event.get("type") == "PrivateStateSnapshot":
        balances = event.get("balances") or {}
        open_orders = event.get("open_orders") or {}
        return {
            "type": "PrivateStateSnapshot",
            "exchange": event.get("exchange"),
            "timestamp_ms": event.get("timestamp_ms"),
            "balance_assets": sorted(balances),
            "balance_count": event.get("balance_count", len(balances)),
            "open_order_count": event.get("open_order_count", len(open_orders)),
            "open_order_ids": sorted(open_orders)[:10],
        }
    if isinstance(event, Balance):
        return {
            "type": "Balance",
            **balance_event_payload(event),
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
    if isinstance(event, dict) and event.get("type") == "ExecutionSnapshot":
        live_orders = event.get("live_orders") or {}
        live_order_ids = list(live_orders)[:10]
        return {
            "type": "ExecutionSnapshot",
            "payload_schema": event.get("payload_schema"),
            "payload_version": event.get("payload_version"),
            "exchange": event.get("exchange"),
            "timestamp_ms": event.get("timestamp_ms"),
            "live_order_count": event.get("live_order_count"),
            "pending_place_count": event.get("pending_place_count"),
            "pending_cancel_count": event.get("pending_cancel_count"),
            "pending_max_age_ms": event.get("pending_max_age_ms"),
            "live_order_ids": live_order_ids,
            "note": "execution state is mirrored from private exchange state; print_events does not place or cancel orders",
        }
    if isinstance(event, ExecutionSnapshot):
        return _summarize_event(event.as_payload())
    if isinstance(event, Action):
        return {
            "type": "StrategyAction",
            "actor_id": event.actor_id,
            "create_count": len(event.orders),
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
                for order in event.orders[:10]
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
                print(json.dumps({"topic": label, **summarize_event(event)}, ensure_ascii=False), flush=True)
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
        build_options = _resolve_build_options(runtime_config, args.build_mode)
        runtime = await build_runtime_components(runtime_config, build_options=build_options)
        contract = runtime_config.as_contract()
        contract["type"] = "RuntimeConfigCheck"
        contract["runtime_build_mode"] = build_options.mode
        contract["selected_public_market_transports"] = {
            role: selection.transport
            for role, selection in runtime.public_market_selections.items()
        }
        contract["selected_private_state_modes"] = {
            role: selection.mode
            for role, selection in runtime.private_state_selections.items()
        }
        contract["strategy_runner_enabled"] = runtime.strategy_runner is not None
        print(json.dumps(contract, ensure_ascii=False), flush=True)
        await run_print_loop(runtime, args)
        return

    exchange = args.exchange.lower().strip()
    mode = args.mode.lower().strip()
    global_exchange = args.global_exchange.lower().strip() if args.global_exchange else None
    global_mode = (args.global_mode or mode).lower().strip()
    global_symbol = args.global_symbol or args.symbol
    api_token = resolve_api_token(exchange, args.private_api_token, args.private_api_token_env)
    global_api_token = (
        resolve_api_token(global_exchange, args.global_private_api_token, args.global_private_api_token_env)
        if global_exchange
        else None
    )
    runtime_config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange=exchange,
            symbol=args.symbol,
            mode=mode,
            depth_limit=args.depth,
            trades_limit=args.trades,
            poll_interval_ms=args.poll_ms,
        ),
        global_=(
            MarketDataSourceConfig(
                role="global",
                exchange=global_exchange,
                symbol=global_symbol,
                mode=global_mode,
                depth_limit=args.depth,
                trades_limit=args.trades,
                poll_interval_ms=args.poll_ms,
            )
            if global_exchange
            else None
        ),
        interval_ms=args.interval_ms,
        max_global_lag_ms=args.max_global_lag_ms,
        global_grace_ms=args.global_grace_ms,
        local_private=PrivateRuntimeConfig(
            mode=args.private_mode,
            api_token=api_token,
            poll_interval_ms=args.private_poll_ms,
        ),
        global_private=(
            PrivateRuntimeConfig(
                mode=args.global_private_mode,
                api_token=global_api_token,
                poll_interval_ms=args.global_private_poll_ms,
            )
            if global_exchange
            else None
        ),
    )
    runtime = await build_runtime_components(runtime_config)
    private_contract = {
        "requested_transport_mode": args.private_mode,
        "selected_transport": runtime.private_state_selections["local"].mode
        if "local" in runtime.private_state_selections
        else None,
        "adapter": type(runtime.private_state_selections["local"].adapter).__name__
        if "local" in runtime.private_state_selections
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
            "print_events_mode": "dry_run",
            "safety": "does not place or cancel orders unless an actor calls OrderManager.sync",
            "live_orders": "ExecutionSnapshot mirrors private open orders for own-order subtraction and execution recovery",
        }
    selection = runtime.public_market_selections["local"]
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
    if global_exchange is not None:
        global_selection = runtime.public_market_selections["global"]
        contract["global_public_market"] = {
            "exchange": global_exchange,
            "symbol": global_symbol,
            "requested_transport_mode": global_selection.requested_mode,
            "selected_public_market_transport": global_selection.transport,
            "public_market_adapter": type(global_selection.adapter).__name__,
            "role": "global",
            "symbol_policy": "same domain symbol as local by default; no tick-size conversion is applied",
        }
        contract["global_private_state"] = {
            "requested_transport_mode": args.global_private_mode,
            "selected_transport": runtime.private_state_selections["global"].mode
            if "global" in runtime.private_state_selections
            else None,
            "adapter": type(runtime.private_state_selections["global"].adapter).__name__
            if "global" in runtime.private_state_selections
            else None,
            "enabled": bool(global_api_token),
            "token_configured": bool(global_api_token),
        }
    contract["joint_strategy_frame"] = {
        "topic": runtime_config.joint_frame_topic,
        "local": "always present",
        "global": "None when --global-exchange is omitted, missing, or stale",
        "max_global_lag_ms": args.max_global_lag_ms,
        "global_grace_ms": args.global_grace_ms,
    }
    print(json.dumps(contract, ensure_ascii=False), flush=True)

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
            }), flush=True)
    finally:
        print_task.cancel()
        await asyncio.gather(print_task, return_exceptions=True)
        await runtime.stop()


def build_print_queues(runtime: RuntimeComponents, *, include_raw: bool) -> list[tuple[str, asyncio.Queue]]:
    config = runtime.config
    bus = runtime.bus
    local_event_topic = config.local.market_event_topic(config.interval_ms)
    local_error_topic = config.local.market_error_topic()
    queues: list[tuple[str, asyncio.Queue]] = [
        (config.joint_frame_topic, bus.subscribe_queue(config.joint_frame_topic)),
        (local_event_topic, bus.subscribe_queue(local_event_topic)),
        (local_error_topic, bus.subscribe_queue(local_error_topic)),
    ]
    if config.global_ is not None:
        global_event_topic = config.global_.market_event_topic(config.interval_ms)
        global_error_topic = config.global_.market_error_topic()
        queues.extend([
            (global_event_topic, bus.subscribe_queue(global_event_topic)),
            (global_error_topic, bus.subscribe_queue(global_error_topic)),
        ])
    if include_raw:
        trade_topic = config.local.raw_trade_topic()
        book_topic = config.local.raw_orderbook_topic()
        queues.extend([
            (trade_topic, bus.subscribe_queue(trade_topic)),
            (book_topic, bus.subscribe_queue(book_topic)),
        ])
        if config.global_ is not None:
            global_trade_topic = config.global_.raw_trade_topic()
            global_book_topic = config.global_.raw_orderbook_topic()
            queues.extend([
                (global_trade_topic, bus.subscribe_queue(global_trade_topic)),
                (global_book_topic, bus.subscribe_queue(global_book_topic)),
            ])
    for role, source in (("local", config.local), ("global", config.global_)):
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
                    config.controller.normalized_decisions_topic,
                    bus.subscribe_queue(config.controller.normalized_decisions_topic),
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


def _resolve_build_options(
    config: MarketMakerRuntimeConfig,
    build_mode: str,
) -> RuntimeBuildOptions:
    if build_mode == "auto":
        return resolve_build_options(config)
    return RuntimeBuildOptions.for_mode(build_mode)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live market-data bus events.")
    parser.add_argument("--config", default=None, help="Path to a JSON runtime config file.")
    parser.add_argument(
        "--build-mode",
        choices=["auto", "monolith", "consumer_only", "ingestor_only", "persistence_only"],
        default="auto",
        help="Runtime composition mode; auto picks consumer_only when all market sources use bus.",
    )
    parser.add_argument("--exchange", choices=["nobitex", "wallex"], default=None)
    parser.add_argument("--symbol", default="BTC_IRT")
    parser.add_argument("--global-exchange", choices=["nobitex", "wallex"], default=None)
    parser.add_argument("--global-symbol", default=None)
    parser.add_argument("--global-mode", choices=["rest", "websocket"], default=None)
    parser.add_argument("--mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--private-mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--private-api-token", default=None)
    parser.add_argument("--private-api-token-env", default=None)
    parser.add_argument("--private-poll-ms", type=int, default=5000)
    parser.add_argument("--global-private-mode", choices=["rest", "websocket"], default="rest")
    parser.add_argument("--global-private-api-token", default=None)
    parser.add_argument("--global-private-api-token-env", default=None)
    parser.add_argument("--global-private-poll-ms", type=int, default=5000)
    parser.add_argument("--interval-ms", type=int, default=5000)
    parser.add_argument("--poll-ms", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--trades", type=int, default=100)
    parser.add_argument("--events", type=int, default=10000, help="Stop after this many events; use 0 to run until timeout.")
    parser.add_argument("--timeout-s", type=float, default=60000.0)
    parser.add_argument("--max-global-lag-ms", type=int, default=2000)
    parser.add_argument("--global-grace-ms", type=int, default=250)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if args.config is None and args.exchange is None:
        parser.error("--exchange is required unless --config is provided")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
