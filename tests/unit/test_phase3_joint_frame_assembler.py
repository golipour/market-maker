"""Phase 3 tests: joint frame assembly around ledger snapshots."""

from __future__ import annotations

import asyncio

import pytest

from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.application.joint_frame_assembler import JointStrategyFrameAssembler
from mm.bus.memory import InMemoryBus
from mm.core.adapters.joint import build_frame_sync_status
from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.wallet import make_balance
from mm.ledger.service import PortfolioLedgerService
from mm.runtime.config import (
    LiquiditySourcingRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    PrivateRuntimeConfig,
)
from tests.unit.test_strategy_pipeline import _event_for, market_event
from tests.unit.conftest import market_event_payload


def _runtime_config(*, global_: bool = True) -> MarketMakerRuntimeConfig:
    global_source = (
        MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT")
        if global_
        else None
    )
    return MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        global_=global_source,
        local_private=PrivateRuntimeConfig(api_token="token-a"),
        global_private=PrivateRuntimeConfig(api_token="token-b") if global_ else None,
    )


def _private_snapshot(exchange: str) -> PrivateStateSnapshot:
    return PrivateStateSnapshot(
        exchange=exchange,
        timestamp_ms=1500,
        balances={
            "BTC": make_balance(exchange, "BTC", 1.0, 0.0, 1500),
            "IRT": make_balance(exchange, "IRT", 1000.0, 0.0, 1500),
        },
        open_orders={},
    )


@pytest.mark.asyncio
async def test_joint_frame_assembler_publishes_joint_topic() -> None:
    bus = InMemoryBus()
    local_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    global_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
    local_private_queue: asyncio.Queue[PrivateStateSnapshot] = asyncio.Queue()
    global_private_queue: asyncio.Queue[PrivateStateSnapshot] = asyncio.Queue()

    config = _runtime_config()
    input_assembler = StrategyInputFrameAssembler(
        local_queue=local_queue,
        global_queue=global_queue,
        local_private_queue=local_private_queue,
        global_private_queue=global_private_queue,
        max_global_lag_ms=1000,
        global_grace_ms=0,
        clock_ms=lambda: 2000,
    )
    ledger_service = PortfolioLedgerService(config, config.binding, bus)
    assembler = JointStrategyFrameAssembler(
        input_assembler=input_assembler,
        binding=config.binding,
        liquidity_sourcing=config.liquidity_sourcing,
        bus=bus,
        joint_output_topic=config.joint_frame_topic,
        ledger_service=ledger_service,
        clock_ms=lambda: 2000,
    )

    joint_queue = bus.subscribe_queue(config.joint_frame_topic)

    await assembler.start()
    try:
        await local_private_queue.put(_private_snapshot("nobitex"))
        await global_private_queue.put(_private_snapshot("wallex"))
        global_event = _event_for("wallex", "BTC_IRT", t_start_ms=1000, t_end_ms=2000)
        local_event = market_event()
        await global_queue.put(market_event_payload(global_event))
        await local_queue.put(market_event_payload(local_event))

        joint_frame = await asyncio.wait_for(joint_queue.get(), timeout=1.0)
    finally:
        await assembler.stop()

    assert joint_frame.input_frame is not None
    assert joint_frame.input_frame.local == local_event
    assert joint_frame.local_market == local_event
    assert joint_frame.global_market == global_event
    assert joint_frame.ledger is not None
    assert joint_frame.sync_status.local_market_ready is True
    assert joint_frame.sync_status.global_market_ready is True
    assert joint_frame.sync_status.latency_skew_ms == 0
    assert joint_frame.mode == "single_venue"


@pytest.mark.asyncio
async def test_single_venue_joint_frame_tolerates_missing_global_private() -> None:
    config = _runtime_config(global_=True)
    input_frame = StrategyInputFrame(local=market_event(),
        global_=_event_for("wallex", "BTC_IRT", t_start_ms=1000, t_end_ms=2000),
        global_status="ready",
        global_lag_ms=0,
        local_private=_private_snapshot("nobitex"),
        global_private=None,
        created_at_ms=2000,
    )

    sync_status = build_frame_sync_status(
        input_frame,
        ledger=None,
        binding=config.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=False),
        now_ms=2000,
    )

    assert sync_status.global_private_ready is True
    assert sync_status.local_private_ready is False


def test_liquidity_sourcing_mode_requires_ready_global_market() -> None:
    config = _runtime_config(global_=True)
    stale_frame = StrategyInputFrame(local=market_event(),
        global_=None,
        global_status="stale",
        global_lag_ms=1500,
        created_at_ms=2000,
    )

    sync_status = build_frame_sync_status(
        stale_frame,
        ledger=None,
        binding=config.binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(
            enabled=True,
            mode="liquidity_sourcing",
        ),
        now_ms=2000,
    )

    assert sync_status.global_market_ready is False
    assert sync_status.latency_skew_ms == 1500


@pytest.mark.asyncio
async def test_joint_frame_uses_ledger_service_for_wallet_state() -> None:
    bus = InMemoryBus()
    config = _runtime_config(global_=False)
    ledger_service = PortfolioLedgerService(config, config.binding, bus)
    snapshot = _private_snapshot("nobitex")
    ledger_service._local.apply_snapshot(snapshot)

    input_frame = StrategyInputFrame(local=market_event(),
        local_private=snapshot,
        created_at_ms=2000,
    )
    assembler = JointStrategyFrameAssembler(
        input_assembler=StrategyInputFrameAssembler(
            local_queue=asyncio.Queue(),
            clock_ms=lambda: 2000,
        ),
        binding=config.binding,
        liquidity_sourcing=config.liquidity_sourcing,
        ledger_service=ledger_service,
        clock_ms=lambda: 2000,
    )

    joint = assembler.build_joint_frame(input_frame)

    assert joint.ledger is not None
    assert joint.ledger.local.available_base == pytest.approx(1.0)
    assert joint.ledger.freshness.local_market_age_ms == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_joint_frame_assembler_skips_stale_market_frames() -> None:
    bus = InMemoryBus()
    local_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()

    config = _runtime_config(global_=False)
    input_assembler = StrategyInputFrameAssembler(
        local_queue=local_queue,
        max_global_lag_ms=1000,
        global_grace_ms=0,
        clock_ms=lambda: 1_000_000,
    )
    assembler = JointStrategyFrameAssembler(
        input_assembler=input_assembler,
        binding=config.binding,
        liquidity_sourcing=config.liquidity_sourcing,
        bus=bus,
        joint_output_topic=config.joint_frame_topic,
        clock_ms=lambda: 1_000_000,
        max_local_market_age_ms=5_000,
    )
    joint_queue = bus.subscribe_queue(config.joint_frame_topic)

    stale_event = _event_for("nobitex", "BTC_IRT", t_start_ms=990_000, t_end_ms=995_000)
    await local_queue.put(market_event_payload(stale_event))

    assert assembler.should_skip_input_frame(
        await input_assembler.build_frame(stale_event)
    ) == "stale_local_market_frame"

    await assembler.start()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(joint_queue.get(), timeout=0.2)
    finally:
        await assembler.stop()
    assert joint.sync_status.ledger_fresh is True


def test_build_runtime_components_wires_joint_frame_topic() -> None:
    from mm.runtime.engine import build_runtime_components

    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    )

    async def _build() -> None:
        runtime = await build_runtime_components(config)
        assert runtime.frame_assembler.joint_output_topic == config.joint_frame_topic

    asyncio.run(_build())
