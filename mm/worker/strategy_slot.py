"""Per-strategy runtime stack for multi-strategy workers."""

from __future__ import annotations

from dataclasses import dataclass

from mm.application.decision_service import StrategyRunner, build_strategy_runner
from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.application.joint_frame_assembler import JointStrategyFrameAssembler
from mm.bus.base import AbstractEventBus
from mm.core.topics import RuntimeTopics
from mm.ledger.service import PortfolioLedgerService, build_portfolio_ledger_service
from mm.runtime.config import MarketMakerRuntimeConfig
from mm.portfolio.virtual_ledger_service import VirtualLedgerService
from mm.runtime.control import RuntimeControlPlane

from mm.worker.account_pool import SharedAccountPool


@dataclass(slots=True)
class StrategySlot:
    strategy_id: str
    account_id: str
    config: MarketMakerRuntimeConfig
    frame_assembler: JointStrategyFrameAssembler | None = None
    ledger_service: PortfolioLedgerService | None = None
    strategy_runner: StrategyRunner | None = None

    async def start(self) -> None:
        if self.ledger_service is not None:
            await self.ledger_service.start()
        if self.frame_assembler is not None:
            await self.frame_assembler.start()
        if self.strategy_runner is not None:
            await self.strategy_runner.start()

    async def stop(self) -> None:
        if self.strategy_runner is not None:
            await self.strategy_runner.stop()
        if self.frame_assembler is not None:
            await self.frame_assembler.stop()
        if self.ledger_service is not None:
            await self.ledger_service.stop()

    def monitor_tasks(self) -> list[object]:
        tasks: list[object] = []
        if self.frame_assembler is not None:
            tasks.extend(self.frame_assembler._tasks)
        if self.strategy_runner is not None and self.strategy_runner._task is not None:
            tasks.append(self.strategy_runner._task)
        if self.ledger_service is not None:
            tasks.extend(self.ledger_service._tasks)
        return tasks


def build_strategy_slot(
    *,
    strategy_id: str,
    account_id: str,
    config: MarketMakerRuntimeConfig,
    bus: AbstractEventBus,
    account_pool: SharedAccountPool,
    control_plane: RuntimeControlPlane,
    virtual_ledger_service: VirtualLedgerService | None = None,
    ledger_source: str = "virtual",
) -> StrategySlot:
    if config.controller is None or not config.controller.enabled:
        return StrategySlot(strategy_id=strategy_id, account_id=account_id, config=config)

    topics = _runtime_topics(config)
    local_topic = topics.local_market_events
    global_topic = topics.global_market_events
    frame_assembler_inner = StrategyInputFrameAssembler(
        local_queue=bus.subscribe_queue(local_topic),
        global_queue=bus.subscribe_queue(global_topic) if global_topic is not None else None,
        max_global_lag_ms=config.max_global_lag_ms,
        global_grace_ms=config.global_grace_ms,
    )
    ledger_service = build_portfolio_ledger_service(
        config,
        bus=bus,
        order_managers=account_pool.order_managers,
    )
    max_local_private_age_ms = 120_000
    max_global_private_age_ms = 120_000
    if config.controller is not None:
        max_local_private_age_ms = config.controller.risk.local.max_private_state_age_ms
        max_global_private_age_ms = config.controller.risk.global_.max_private_state_age_ms
    actor_id = config.controller.actor_id
    use_virtual_ledger = ledger_source.strip().lower() == "virtual"
    allocation = (
        virtual_ledger_service.allocation_for(actor_id)
        if use_virtual_ledger and virtual_ledger_service is not None
        else None
    )
    strategy_wallet_provider = None
    if use_virtual_ledger and virtual_ledger_service is not None and allocation is not None:
        strategy_wallet_provider = lambda actor_id=actor_id: virtual_ledger_service.wallet_for(actor_id)
    frame_assembler = JointStrategyFrameAssembler(
        input_assembler=frame_assembler_inner,
        binding=config.binding,
        liquidity_sourcing=config.liquidity_sourcing,
        bus=bus,
        joint_output_topic=config.joint_frame_topic,
        ledger_service=ledger_service,
        max_local_private_age_ms=max_local_private_age_ms,
        max_global_private_age_ms=max_global_private_age_ms,
        strategy_wallet_provider=strategy_wallet_provider,
    )
    strategy_runner = build_strategy_runner(
        config,
        bus=bus,
        frame_topic=config.joint_frame_topic,
        order_managers=account_pool.order_managers,
        control_plane=control_plane,
        allocation=allocation,
        strategy_wallet_provider=strategy_wallet_provider,
    )
    return StrategySlot(
        strategy_id=strategy_id,
        account_id=account_id,
        config=config,
        frame_assembler=frame_assembler,
        ledger_service=ledger_service,
        strategy_runner=strategy_runner,
    )


def _runtime_topics(config: MarketMakerRuntimeConfig) -> RuntimeTopics:
    return RuntimeTopics(
        local=config.local,
        global_=config.global_,
        interval_ms=config.interval_ms,
        action_topic=config.controller.normalized_action_topic if config.controller is not None else "strategy:actions",
        decisions_topic=(
            config.controller.normalized_decisions_topic
            if config.controller is not None
            else "strategy:decisions"
        ),
        diagnostics_topic=(
            config.controller.normalized_diagnostics_topic
            if config.controller is not None
            else None
        ),
    )
