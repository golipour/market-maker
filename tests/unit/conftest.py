"""Shared helpers for unit tests."""

from __future__ import annotations

from typing import Any

from mm.core.adapters.joint import build_frame_sync_status, joint_ledger_from_frame
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.target_book import JointTargetBook
from mm.runtime.config import LiquiditySourcingRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
from mm.strategies.target_book_builder import build_venue_target_book_from_orders


def venues_payload(
    *,
    local: dict[str, Any] | None = None,
    global_: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    local_cfg = dict(local or {"exchange": "nobitex", "symbol": "BTC_IRT"})
    local_exchange = str(local_cfg.get("exchange", "nobitex")).lower()
    symbol = str(local_cfg.get("symbol", "BTC_IRT"))
    global_cfg = None if global_ is False else dict(global_ or {})
    global_exchange = str(global_cfg.get("exchange", "wallex")).lower() if global_cfg is not None else local_exchange
    exchanges: dict[str, Any] = {
        local_exchange: {
            "market_data": {
                key: value
                for key, value in local_cfg.items()
                if key
                in {
                    "mode",
                    "depth_limit",
                    "trades_limit",
                    "poll_interval_ms",
                    "instrument",
                }
            }
        }
    }
    if global_cfg is not None:
        exchanges[global_exchange] = {
            "market_data": {
                key: value
                for key, value in global_cfg.items()
                if key
                in {
                    "mode",
                    "depth_limit",
                    "trades_limit",
                    "poll_interval_ms",
                    "instrument",
                }
            }
        }
    return {
        "exchanges": exchanges,
        "strategy_presets": {
            "test_preset": {
                "controller": {"enabled": False},
                "liquidity_sourcing": {"mode": "off"},
            }
        },
        "bots": {
            "test-bot": {
                "preset": "test_preset",
                "symbol": symbol,
                "quoting_exchange": local_exchange,
                "liquidity_source_exchange": global_exchange,
                "reference_exchanges": [global_exchange] if global_exchange != local_exchange else [local_exchange],
            }
        },
        **extra,
    }


def strategy_decision_with_joint(
    event: MarketEvent,
    *,
    frame_step: int,
    actor_id: str = "test-runner",
    order: Order | None = None,
) -> StrategyDecision:
    if order is None:
        order = Order(
            event.exchange,
            event.symbol,
            "buy",
            99.0,
            1.0,
            client_order_id="c1",
        )
    joint_book = JointTargetBook(
        actor_id=actor_id,
        local=build_venue_target_book_from_orders(actor_id=actor_id, orders=(order,)),
    )
    return StrategyDecision(
        actor_id=actor_id,
        exchange=event.exchange,
        symbol=event.symbol,
        t_end_ms=event.t_end_ms,
        frame_step=frame_step,
        action=Action(actor_id=actor_id, orders=(order,)),
        joint_target_book=joint_book,
    )


def joint_frame_for_input(
    frame: StrategyInputFrame,
    *,
    binding: SymbolBinding | None = None,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig | None = None,
) -> JointStrategyFrame:
    if binding is None:
        global_source = (
            MarketDataSourceConfig(
                role="global",
                exchange=frame.global_.exchange,
                symbol=frame.global_.symbol,
            )
            if frame.global_ is not None
            else None
        )
        binding = MarketMakerRuntimeConfig(
            local=MarketDataSourceConfig(
                role="local",
                exchange=frame.local.exchange,
                symbol=frame.local.symbol,
            ),
            global_=global_source,
        ).binding
    sourcing = liquidity_sourcing or LiquiditySourcingRuntimeConfig()
    created_at_ms = int(frame.created_at_ms or frame.local.t_end_ms)
    ledger = joint_ledger_from_frame(frame, binding, created_at_ms=created_at_ms)
    return JointStrategyFrame(
        local_market=frame.local,
        global_market=frame.global_,
        ledger=ledger,
        mode=(
            "liquidity_sourcing"
            if sourcing.enabled and sourcing.mode == "liquidity_sourcing"
            else "single_venue"
        ),
        sync_status=build_frame_sync_status(
            frame,
            ledger=ledger,
            binding=binding,
            liquidity_sourcing=sourcing,
            now_ms=created_at_ms,
        ),
        created_at_ms=created_at_ms,
        input_frame=frame,
    )
