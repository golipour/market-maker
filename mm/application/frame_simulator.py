"""Synchronous offline replay of one joint strategy frame through strategy, risk, and execution planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from mm.application.frame_preparation import clean_strategy_frame
from mm.application.strategy_runner import (
    _lineage_ids,
    _live_orders_from_ledger,
    _with_decision_lineage,
    _with_target_book_lineage,
)
from mm.application.trading_context import trading_state_from_ledger
from mm.application.venue_execution import prepare_execution_joint_book
from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, VenueTargetBook
from mm.execution.order_validation import NoOpVenueOrderLimitSource
from mm.execution.planning import plan_venue_target_book
from mm.execution.reconcile import ReconcileConfig
from mm.execution.rules_gateway import ExchangeRulesGateway
from mm.execution.target_book import ExecutionPlan
from mm.risk.net_delta import NetDeltaRiskEngine, target_book_result_to_risk_result
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
)
from mm.strategies.mpc_engine import MpcStrategyEngine


@dataclass(frozen=True, slots=True)
class VenueExecutionContext:
    """Pre-frame execution state used when replaying one venue plan."""

    pending_place_count: int = 0
    pending_cancel_count: int = 0


@dataclass(frozen=True, slots=True)
class FrameSimulationResult:
    """Offline outputs for one replayed strategy frame."""

    frame_id: str
    decision_id: str
    target_book_id: str
    frame_step: int
    decision: StrategyDecision
    proposed_target_book: JointTargetBook
    approved_target_book: JointTargetBook | None
    risk_approved: bool
    risk_rejection_counts: dict[str, int]
    execution_plans: dict[str, ExecutionPlan]
    validation_rejections: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class FrameSimulator:
    """Replay strategy -> risk -> execution planning for one ``JointStrategyFrame``."""

    controller: ControllerRuntimeConfig
    binding: SymbolBinding
    liquidity_sourcing: LiquiditySourcingRuntimeConfig = field(
        default_factory=LiquiditySourcingRuntimeConfig
    )
    execution: ExecutionRuntimeConfig = field(default_factory=ExecutionRuntimeConfig)
    risk_engine: AbstractRiskEngine | None = None
    reconcile_config: ReconcileConfig | None = None
    venue_contexts: dict[str, VenueExecutionContext] = field(default_factory=dict)
    _engine: MpcStrategyEngine = field(init=False, repr=False)
    _rules_gateway: ExchangeRulesGateway = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = MpcStrategyEngine(
            config=self.controller,
            binding=self.binding,
            liquidity_sourcing=self.liquidity_sourcing,
            execution=self.execution,
        )
        self._rules_gateway = ExchangeRulesGateway(NoOpVenueOrderLimitSource())

    def simulate_from_target_book(
        self,
        joint_frame: JointStrategyFrame,
        target_book: JointTargetBook,
        *,
        frame_step: int = 1,
    ) -> FrameSimulationResult:
        """Replay risk and execution planning from a persisted target book."""

        frame = joint_frame.input_frame
        if frame is None:
            raise ValueError("JointStrategyFrame.input_frame is required")
        if joint_frame.ledger is None:
            raise ValueError("JointStrategyFrame.ledger is required")

        lineage = _lineage_ids(
            actor_id=self.controller.actor_id,
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            existing_frame_id=target_book.frame_id or frame.frame_id or joint_frame.frame_id,
        )
        frame = replace(frame, frame_id=lineage["frame_id"])
        ledger = joint_frame.ledger
        open_orders = _live_orders_from_ledger(ledger)
        clean_frame = clean_strategy_frame(frame, open_orders)
        proposed = _with_target_book_lineage(target_book, lineage)
        return self._simulate_risk_and_execution(
            frame=frame,
            clean_frame=clean_frame,
            ledger=ledger,
            lineage=lineage,
            frame_step=frame_step,
            proposed=proposed,
        )

    def simulate(
        self,
        joint_frame: JointStrategyFrame,
        *,
        frame_step: int = 1,
    ) -> FrameSimulationResult:
        frame = joint_frame.input_frame
        if frame is None:
            raise ValueError("JointStrategyFrame.input_frame is required")
        if joint_frame.ledger is None:
            raise ValueError("JointStrategyFrame.ledger is required")

        lineage = _lineage_ids(
            actor_id=self.controller.actor_id,
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=frame_step,
            existing_frame_id=frame.frame_id or joint_frame.frame_id,
        )
        frame = replace(frame, frame_id=lineage["frame_id"])
        ledger = joint_frame.ledger
        open_orders = _live_orders_from_ledger(ledger)
        clean_frame = clean_strategy_frame(frame, open_orders)

        self._engine.set_ledger_context(ledger)
        _seed_optimizer_if_configured(self.controller)
        decision = self._engine.decide(clean_frame, frame_step=frame_step)
        decision = _with_decision_lineage(decision, lineage)
        proposed = decision.joint_target_book
        if proposed is None:
            raise ValueError("StrategyDecision.joint_target_book is required")
        result = self._simulate_risk_and_execution(
            frame=frame,
            clean_frame=clean_frame,
            ledger=ledger,
            lineage=lineage,
            frame_step=frame_step,
            proposed=proposed,
            decision=decision,
        )
        return result

    def _simulate_risk_and_execution(
        self,
        *,
        frame: StrategyInputFrame,
        clean_frame: StrategyInputFrame,
        ledger: JointLedgerSnapshot,
        lineage: dict[str, str],
        frame_step: int,
        proposed: JointTargetBook,
        decision: StrategyDecision | None = None,
    ) -> FrameSimulationResult:
        if decision is None:
            from mm.core.adapters.joint import joint_target_book_to_action

            decision = StrategyDecision(
                actor_id=self.controller.actor_id,
                exchange=frame.local.exchange,
                symbol=frame.local.symbol,
                t_end_ms=frame.local.t_end_ms,
                frame_step=frame_step,
                action=joint_target_book_to_action(proposed),
                joint_target_book=proposed,
                frame_id=lineage["frame_id"],
                decision_id=lineage["decision_id"],
                target_book_id=lineage["target_book_id"],
            )
            decision = _with_decision_lineage(decision, lineage)

        execution_book = prepare_execution_joint_book(
            proposed,
            execution=self.execution,
            liquidity_sourcing=self.liquidity_sourcing,
            frame=clean_frame,
            binding=self.binding,
        )
        live_execution = _any_live_execution_enabled(self.execution, self.liquidity_sourcing)
        global_live_execution = _global_live_execution_enabled(
            self.execution,
            self.liquidity_sourcing,
            binding=self.binding,
        )

        risk_engine = self.risk_engine
        approved_book: JointTargetBook | None = None
        risk_approved = False
        rejection_counts: dict[str, int] = {}
        if isinstance(risk_engine, NetDeltaRiskEngine):
            target_risk_result = risk_engine.validate_target_book_with_result(
                proposed,
                ledger,
                clean_frame,
                live_execution=live_execution,
                global_live_execution=global_live_execution,
            )
            risk_result = target_book_result_to_risk_result(
                target_risk_result,
                proposed_actions=[decision.action],
            )
            rejection_counts = dict(risk_result.rejection_counts)
            if target_risk_result.approved is not None:
                approved_book = prepare_execution_joint_book(
                    target_risk_result.approved,
                    execution=self.execution,
                    liquidity_sourcing=self.liquidity_sourcing,
                    frame=clean_frame,
                    binding=self.binding,
                )
                risk_approved = True
        elif risk_engine is not None:
            trading_state = trading_state_from_ledger(
                ledger,
                actor_id=self.controller.actor_id,
                open_orders=_live_orders_from_ledger(ledger),
            )
            risk_result = risk_engine.validate_frame_with_result(
                [decision.action],
                trading_state,
                clean_frame,
            )
            rejection_counts = dict(risk_result.rejection_counts)
            risk_approved = bool(risk_result.approved_list)
            if risk_approved:
                approved_book = execution_book
        else:
            approved_book = execution_book
            risk_approved = True

        plan_book = approved_book if approved_book is not None else execution_book
        execution_plans = _plan_joint_target_book(
            plan_book,
            ledger=ledger,
            binding=self.binding,
            rules_gateway=self._rules_gateway,
            reconcile_config=self.reconcile_config,
            venue_contexts=self.venue_contexts,
        )
        validation_rejections = tuple(
            {
                "exchange": rejection.exchange,
                "symbol": rejection.symbol,
                "side": rejection.order.side,
                "price": rejection.order.price,
                "qty": rejection.order.qty,
                "reason": rejection.reason,
                "stage": "rules",
            }
            for plan in execution_plans.values()
            for rejection in plan.rejected
        )

        return FrameSimulationResult(
            frame_id=lineage["frame_id"],
            decision_id=lineage["decision_id"],
            target_book_id=lineage["target_book_id"],
            frame_step=frame_step,
            decision=decision,
            proposed_target_book=_with_target_book_lineage(proposed, lineage),
            approved_target_book=(
                _with_target_book_lineage(approved_book, lineage) if approved_book is not None else None
            ),
            risk_approved=risk_approved,
            risk_rejection_counts=rejection_counts,
            execution_plans=execution_plans,
            validation_rejections=validation_rejections,
        )


def _plan_joint_target_book(
    book: JointTargetBook,
    *,
    ledger: JointLedgerSnapshot,
    binding: SymbolBinding,
    rules_gateway: ExchangeRulesGateway,
    reconcile_config: ReconcileConfig | None,
    venue_contexts: dict[str, VenueExecutionContext],
) -> dict[str, ExecutionPlan]:
    plans: dict[str, ExecutionPlan] = {}
    for venue_book in (book.local, book.global_):
        if venue_book is None:
            continue
        plan_id = venue_book.execution_plan_id.strip()
        if not plan_id:
            continue
        venue_ledger = _venue_ledger_for_book(venue_book, ledger, binding)
        context = venue_contexts.get(plan_id, VenueExecutionContext())
        plans[plan_id] = plan_venue_target_book(
            venue_book,
            venue_ledger,
            rules_gateway=rules_gateway,
            reconcile_config=reconcile_config,
            pending_place_count=context.pending_place_count,
            pending_cancel_count=context.pending_cancel_count,
        )
    return plans


def _venue_ledger_for_book(
    venue_book: VenueTargetBook,
    ledger: JointLedgerSnapshot,
    binding: SymbolBinding,
) -> VenueLedger:
    exchange = venue_book.exchange.lower().strip()
    if exchange == ledger.local.exchange.lower().strip():
        return ledger.local
    if ledger.global_ is not None and exchange == ledger.global_.exchange.lower().strip():
        return ledger.global_
    spec = binding.local if exchange == binding.local.exchange.lower().strip() else binding.global_
    if spec is None:
        spec = binding.local
    return VenueLedger(
        exchange=venue_book.exchange,
        symbol=venue_book.symbol,
        base_asset=spec.base_asset,
        quote_asset=spec.quote_asset,
        live_orders={},
    )


def _any_live_execution_enabled(
    execution: ExecutionRuntimeConfig,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
) -> bool:
    if execution.local.enabled and not execution.local.observe_only:
        return True
    return bool(
        liquidity_sourcing.global_live_orders
        and execution.global_.enabled
        and not execution.global_.observe_only
    )


def _global_live_execution_enabled(
    execution: ExecutionRuntimeConfig,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
    *,
    binding: SymbolBinding,
) -> bool:
    return bool(
        binding.global_ is not None
        and liquidity_sourcing.global_live_orders
        and execution.global_.enabled
        and not execution.global_.observe_only
    )


def _seed_optimizer_if_configured(controller: ControllerRuntimeConfig) -> None:
    optimizer = controller.optimizer
    if not isinstance(optimizer, dict) or not optimizer.get("fixed_mc_samples"):
        return
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:
        return


def target_book_order_rows(book: JointTargetBook | None) -> tuple[dict[str, Any], ...]:
    if book is None:
        return ()
    rows: list[dict[str, Any]] = []
    for venue_book in (book.local, book.global_):
        if venue_book is None:
            continue
        for order in venue_book.orders:
            rows.append(
                {
                    "exchange": order.exchange,
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": float(order.price),
                    "qty": float(order.qty),
                    "order_type": order.order_type,
                }
            )
    return tuple(rows)
