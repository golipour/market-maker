from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Fill, Order
from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.core.risk_limits import HedgeVenueRiskLimits
from mm.execution.memory import RecordingOrderManager
from mm.exchanges.capabilities import supports_market_orders
from mm.hedging.models import HEDGE_EXECUTION_DIAGNOSTIC_TYPE, HEDGE_INTENT_TYPE
from mm.hedging.risk import validate_hedge_market_order
from mm.hedging.service import HedgeExecutionService, _hedge_side_for_primary_fill
from mm.runtime.config import (
    ControllerRuntimeConfig,
    HedgingRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    RiskRuntimeConfig,
)


def _runtime_config(*, hedging_enabled: bool = True, observe_only: bool = False) -> MarketMakerRuntimeConfig:
    return MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="USDT_IRT"),
        hedge=MarketDataSourceConfig(role="hedge", exchange="wallex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            risk=RiskRuntimeConfig(
                hedge=HedgeVenueRiskLimits(
                    min_inventory=0.0,
                    max_inventory=10_000.0,
                    min_cash=0.0,
                    max_market_order_notional=1_000_000.0,
                )
            ),
        ),
        hedging=HedgingRuntimeConfig(
            enabled=hedging_enabled,
            observe_only=observe_only,
            actor_id="hedge-executor",
        ),
    )


def test_hedge_side_offsets_primary_fill() -> None:
    assert _hedge_side_for_primary_fill("buy") == "sell"
    assert _hedge_side_for_primary_fill("sell") == "buy"


def test_market_order_capability_for_configured_venues() -> None:
    assert supports_market_orders("nobitex")
    assert supports_market_orders("wallex")


def test_hedge_risk_rejects_sell_without_inventory() -> None:
    order = Order("wallex", "USDT_IRT", "sell", 100.0, 5.0, order_type="market")
    ok, reason = validate_hedge_market_order(
        order,
        limits=HedgeVenueRiskLimits(min_inventory=0.0),
        hedge_inventory=1.0,
        hedge_cash=1000.0,
    )
    assert not ok
    assert reason == "insufficient_hedge_inventory"


@pytest.mark.asyncio
async def test_primary_fill_emits_intent_and_diagnostic_observe_only() -> None:
    bus = InMemoryBus()
    runtime = _runtime_config(hedging_enabled=True, observe_only=True)
    hedge_manager = RecordingOrderManager(exchange="wallex")
    service = HedgeExecutionService(
        config=runtime.hedging,  # type: ignore[arg-type]
        runtime_config=runtime,
        binding=runtime.binding,
        bus=bus,
        order_managers={"wallex": hedge_manager},
        hedge_limits=runtime.controller.risk.hedge,  # type: ignore[union-attr]
    )
    intents_q = bus.subscribe_queue("hedge:intents:hedge-executor")
    diag_q = bus.subscribe_queue("hedge:diagnostics:hedge-executor")

    fill = Fill(
        exchange_order_id="fill-1",
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="buy",
        price=100_000.0,
        qty=2.0,
        fee=0.0,
        timestamp_ms=1_000,
    )
    await service._on_primary_fill(fill)

    intent = await asyncio.wait_for(intents_q.get(), timeout=1.0)
    assert intent["type"] == HEDGE_INTENT_TYPE
    assert intent["primary_side"] == "buy"
    assert intent["hedge_side"] == "sell"
    assert intent["qty"] == 2.0

    diagnostic = await asyncio.wait_for(diag_q.get(), timeout=1.0)
    assert diagnostic["type"] == HEDGE_EXECUTION_DIAGNOSTIC_TYPE
    assert diagnostic["status"] == "skipped"
    assert diagnostic["reason"] == "observe_only_config"
    assert hedge_manager.synced == []


@pytest.mark.asyncio
async def test_primary_fill_places_hedge_market_order_when_enabled() -> None:
    bus = InMemoryBus()
    runtime = _runtime_config(hedging_enabled=True, observe_only=False)
    hedge_manager = RecordingOrderManager(exchange="wallex")
    service = HedgeExecutionService(
        config=runtime.hedging,  # type: ignore[arg-type]
        runtime_config=runtime,
        binding=runtime.binding,
        bus=bus,
        order_managers={"wallex": hedge_manager},
        hedge_limits=runtime.controller.risk.hedge,  # type: ignore[union-attr]
    )
    diag_q = bus.subscribe_queue("hedge:diagnostics:hedge-executor")

    service._latest_hedge_private = PrivateStateSnapshot(
        exchange="wallex",
        timestamp_ms=1_000,
        balances={
            "USDT": Balance("wallex", "USDT", 100.0, 0.0, 1_000),
            "IRT": Balance("wallex", "IRT", 500_000_000.0, 0.0, 1_000),
        },
    )

    fill = Fill(
        exchange_order_id="fill-2",
        actor_id="mpc-controller",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="sell",
        price=100_000.0,
        qty=1.5,
        fee=0.0,
        timestamp_ms=2_000,
    )
    await service._on_primary_fill(fill)

    diagnostic = await asyncio.wait_for(diag_q.get(), timeout=1.0)
    assert diagnostic["status"] == "submitted"
    assert len(hedge_manager.synced) == 1
    placed = hedge_manager.synced[0][0].orders[0]
    assert placed.exchange == "wallex"
    assert placed.side == "buy"
    assert placed.order_type == "market"
    assert placed.qty == 1.5
    assert abs(service.residual_unhedged_qty) < 1e-9


@pytest.mark.asyncio
async def test_duplicate_fill_is_deduped() -> None:
    bus = InMemoryBus()
    runtime = _runtime_config(hedging_enabled=True, observe_only=True)
    service = HedgeExecutionService(
        config=runtime.hedging,  # type: ignore[arg-type]
        runtime_config=runtime,
        binding=runtime.binding,
        bus=bus,
        order_managers={},
        hedge_limits=HedgeVenueRiskLimits(),
    )
    fill = Fill(
        exchange_order_id="fill-dup",
        actor_id="a",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="buy",
        price=1.0,
        qty=1.0,
        fee=0.0,
        timestamp_ms=3_000,
    )
    await service._on_primary_fill(fill)
    assert len(service._seen_fills) == 1
    await service._on_primary_fill(fill)
    assert len(service._seen_fills) == 1


@pytest.mark.asyncio
async def test_bus_subscription_receives_primary_fill() -> None:
    bus = InMemoryBus()
    runtime = _runtime_config(hedging_enabled=True, observe_only=True)
    service = HedgeExecutionService(
        config=runtime.hedging,  # type: ignore[arg-type]
        runtime_config=runtime,
        binding=runtime.binding,
        bus=bus,
        order_managers={},
        hedge_limits=HedgeVenueRiskLimits(),
    )
    intents_q = bus.subscribe_queue("hedge:intents:hedge-executor")
    await service.start()
    await asyncio.sleep(0.05)
    fill = Fill(
        exchange_order_id="fill-bus",
        actor_id="a",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="buy",
        price=1.0,
        qty=1.0,
        fee=0.0,
        timestamp_ms=4_000,
    )
    await bus.publish("private:state:nobitex:fills", fill)
    intent = await asyncio.wait_for(intents_q.get(), timeout=1.0)
    assert intent["source_fill_id"] == "fill-bus"
    await service.stop()
