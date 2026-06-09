"""Replay persisted frames through the offline simulator and compare to durable outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm.application.frame_simulator import FrameSimulationResult, FrameSimulator, target_book_order_rows
from mm.application.joint_frame_assembler import joint_strategy_frame_from_input
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.payload_schema import joint_ledger_snapshot_from_payload, joint_target_book_from_payload
from mm.core.payload_schema import market_event_from_payload
from mm.persistence.replay import (
    ReplayFidelityViolation,
    ReplayFrameState,
    ReplayLineageChain,
    read_envelopes,
    read_lineage_chains,
    verify_lineage_fidelity,
)


@dataclass(frozen=True, slots=True)
class PersistedFrameInputs:
    """Inputs required to replay one strategy frame offline."""

    frame_id: str
    joint_frame: JointStrategyFrame
    chain: ReplayLineageChain


@dataclass(frozen=True, slots=True)
class FrameReplayReport:
    """Comparison of offline simulation against persisted lineage outputs."""

    frame_id: str
    fidelity: tuple[ReplayFidelityViolation, ...]
    simulation: tuple[ReplayFidelityViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.fidelity and not self.simulation


def replay_and_compare_persisted_frame(
    path: str | Path,
    frame_id: str,
    *,
    simulator: FrameSimulator,
    frame_step: int = 1,
    execution_only: bool = False,
) -> FrameReplayReport:
    """Load one frame from disk, simulate offline, and compare to persisted outputs."""

    inputs = load_persisted_frame_inputs(path, frame_id)
    if execution_only:
        simulated = replay_execution_from_persisted(inputs, simulator=simulator, frame_step=frame_step)
    else:
        simulated = simulator.simulate(inputs.joint_frame, frame_step=frame_step)
    return compare_frame_simulation(simulated, inputs.chain)


def replay_execution_from_persisted(
    inputs: PersistedFrameInputs,
    *,
    simulator: FrameSimulator,
    frame_step: int = 1,
) -> FrameSimulationResult:
    """Replay risk and execution planning from the persisted target book."""

    persisted = inputs.chain.reconstruct_state()
    if persisted.latest_target_book is None:
        raise ValueError(f"no persisted target book for frame_id={inputs.frame_id!r}")
    target_book = joint_target_book_from_payload(persisted.latest_target_book)
    return simulator.simulate_from_target_book(
        inputs.joint_frame,
        target_book,
        frame_step=frame_step,
    )


def load_persisted_frame_inputs(path: str | Path, frame_id: str) -> PersistedFrameInputs:
    chains = read_lineage_chains(path)
    if frame_id not in chains:
        raise KeyError(f"no lineage chain found for frame_id={frame_id!r}")
    chain = chains[frame_id]
    joint_frame = _joint_frame_from_envelopes(read_envelopes(path), frame_id=frame_id)
    if joint_frame is None:
        raise ValueError(f"unable to rebuild JointStrategyFrame for frame_id={frame_id!r}")
    return PersistedFrameInputs(frame_id=frame_id, joint_frame=joint_frame, chain=chain)


def compare_frame_simulation(
    simulated: FrameSimulationResult,
    chain: ReplayLineageChain,
) -> FrameReplayReport:
    """Compare offline simulation outputs to a persisted lineage chain."""

    fidelity = verify_lineage_fidelity(chain).violations
    persisted = chain.reconstruct_state()
    violations: list[ReplayFidelityViolation] = []
    violations.extend(_compare_target_book(simulated, persisted))
    violations.extend(_compare_execution_plans(simulated, persisted))
    return FrameReplayReport(
        frame_id=simulated.frame_id,
        fidelity=fidelity,
        simulation=tuple(violations),
    )


def _compare_target_book(
    simulated: FrameSimulationResult,
    persisted: ReplayFrameState,
) -> list[ReplayFidelityViolation]:
    if persisted.latest_target_book is None:
        return []
    try:
        persisted_book = joint_target_book_from_payload(persisted.latest_target_book)
    except (TypeError, ValueError) as exc:
        return [
            ReplayFidelityViolation(
                code="persisted_target_book_invalid",
                message=str(exc),
            )
        ]
    simulated_book = simulated.approved_target_book or simulated.proposed_target_book
    simulated_keys = {_order_key(row) for row in target_book_order_rows(simulated_book)}
    persisted_keys = {
        _order_key(
            {
                "side": order.side,
                "price": order.price,
                "qty": order.qty,
            }
        )
        for venue in (persisted_book.local, persisted_book.global_)
        if venue is not None
        for order in venue.orders
    }
    if simulated_keys != persisted_keys:
        return [
            ReplayFidelityViolation(
                code="target_book_mismatch",
                message=(
                    f"simulated target orders {sorted(simulated_keys)} "
                    f"!= persisted {sorted(persisted_keys)}"
                ),
            )
        ]
    return []


def _compare_execution_plans(
    simulated: FrameSimulationResult,
    persisted: ReplayFrameState,
) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    for plan_id, plan_state in persisted.execution_plans.items():
        simulated_plan = simulated.execution_plans.get(plan_id)
        if simulated_plan is None:
            if plan_state.place_requested_orders or plan_state.cancel_requested_order_ids:
                violations.append(
                    ReplayFidelityViolation(
                        code="missing_simulated_execution_plan",
                        message=f"no simulated execution plan for {plan_id!r}",
                        execution_plan_id=plan_id,
                    )
                )
            continue
        if simulated_plan.diagnostics.get("planner") == "pending_confirmation_hold":
            continue
        simulated_place_keys = {
            _order_key(
                {
                    "side": command.order.side,
                    "price": command.order.price,
                    "qty": command.order.qty,
                }
            )
            for command in simulated_plan.place
        }
        persisted_place_keys = {_order_key(order) for order in plan_state.place_requested_orders}
        if simulated_place_keys != persisted_place_keys:
            violations.append(
                ReplayFidelityViolation(
                    code="place_request_mismatch",
                    message=(
                        f"simulated place orders {sorted(simulated_place_keys)} "
                        f"!= persisted {sorted(persisted_place_keys)}"
                    ),
                    execution_plan_id=plan_id,
                )
            )
        simulated_cancel_ids = tuple(sorted(command.exchange_order_id for command in simulated_plan.cancel))
        persisted_cancel_ids = tuple(sorted(plan_state.cancel_requested_order_ids))
        if simulated_cancel_ids != persisted_cancel_ids:
            violations.append(
                ReplayFidelityViolation(
                    code="cancel_request_mismatch",
                    message=(
                        f"simulated cancel ids {simulated_cancel_ids} "
                        f"!= persisted {persisted_cancel_ids}"
                    ),
                    execution_plan_id=plan_id,
                )
            )
    return violations


def _joint_frame_from_envelopes(
    envelopes: Any,
    *,
    frame_id: str,
) -> JointStrategyFrame | None:
    market_events: dict[tuple[str, str, int], Any] = {}
    ledgers: list[tuple[int, Any]] = []
    joint_payload: dict[str, Any] | None = None
    t_end_ms = _frame_t_end_ms(frame_id)

    for envelope in envelopes:
        if envelope.event_type == "MarketEvent":
            event = market_event_from_payload(envelope.payload)
            if event is not None:
                market_events[(event.exchange.lower(), event.symbol.lower(), event.t_end_ms)] = event
        elif envelope.event_type == "JointLedgerSnapshot":
            try:
                ledger = joint_ledger_snapshot_from_payload(envelope.payload)
            except (TypeError, ValueError):
                continue
            ledgers.append((ledger.created_at_ms, ledger))
        elif envelope.event_type == "JointStrategyFrame":
            payload = envelope.payload if isinstance(envelope.payload, dict) else {}
            if str(payload.get("frame_id") or "").strip() == frame_id:
                joint_payload = payload

    local_market = _select_market_event(market_events, joint_payload, role="local", t_end_ms=t_end_ms)
    if local_market is None:
        return None
    global_market = _select_market_event(market_events, joint_payload, role="global", t_end_ms=t_end_ms)
    ledger = _select_ledger(ledgers, joint_payload, t_end_ms=t_end_ms)
    if ledger is None:
        return None

    input_frame = StrategyInputFrame(
        local=local_market,
        global_=global_market,
        local_private=_private_snapshot_from_venue(ledger.local),
        global_private=(
            _private_snapshot_from_venue(ledger.global_)
            if ledger.global_ is not None
            else None
        ),
        frame_id=frame_id,
        created_at_ms=int(joint_payload.get("created_at_ms") if joint_payload else t_end_ms),
    )
    from mm.runtime.config import LiquiditySourcingRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig

    global_source = (
        MarketDataSourceConfig(
            role="global",
            exchange=global_market.exchange,
            symbol=global_market.symbol,
        )
        if global_market is not None
        else None
    )
    binding = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange=local_market.exchange,
            symbol=local_market.symbol,
        ),
        global_=global_source,
    ).binding
    sourcing = LiquiditySourcingRuntimeConfig(
        enabled=bool(joint_payload and joint_payload.get("mode") == "liquidity_sourcing")
    )
    return joint_strategy_frame_from_input(
        input_frame,
        binding,
        ledger=ledger,
        liquidity_sourcing=sourcing,
        now_ms=input_frame.created_at_ms,
        input_frame=input_frame,
    )


def _select_market_event(
    market_events: dict[tuple[str, str, int], Any],
    joint_payload: dict[str, Any] | None,
    *,
    role: str,
    t_end_ms: int,
) -> Any | None:
    if joint_payload is not None:
        summary = joint_payload.get(f"{role}_market")
        if isinstance(summary, dict):
            key = (
                str(summary.get("exchange") or "").lower(),
                str(summary.get("symbol") or "").lower(),
                int(summary.get("t_end_ms") or t_end_ms),
            )
            if key in market_events:
                return market_events[key]
    for event in market_events.values():
        if int(event.t_end_ms) == t_end_ms:
            return event
    return next(iter(market_events.values()), None)


def _select_ledger(
    ledgers: list[tuple[int, Any]],
    joint_payload: dict[str, Any] | None,
    *,
    t_end_ms: int,
) -> Any | None:
    if not ledgers:
        return None
    if joint_payload is not None and not joint_payload.get("ledger_compacted"):
        ledger_payload = joint_payload.get("ledger")
        if isinstance(ledger_payload, dict) and ledger_payload.get("payload_schema"):
            try:
                return joint_ledger_snapshot_from_payload(ledger_payload)
            except (TypeError, ValueError):
                pass
    target_ms = int(joint_payload.get("created_at_ms") if joint_payload else t_end_ms)
    return min(ledgers, key=lambda item: abs(item[0] - target_ms))[1]


def _frame_t_end_ms(frame_id: str) -> int:
    parts = frame_id.split(":")
    if len(parts) >= 5 and parts[-1].isdigit() and parts[-2].isdigit():
        return int(parts[-2])
    return 0


def _private_snapshot_from_venue(venue: VenueLedger) -> PrivateStateSnapshot:
    timestamp_ms = int(venue.last_snapshot_ms or venue.last_private_event_ms or 0)
    return PrivateStateSnapshot(
        exchange=venue.exchange,
        timestamp_ms=timestamp_ms,
        balances=dict(venue.balances),
        open_orders=dict(venue.live_orders),
    )


def _order_key(order: dict[str, Any]) -> tuple[str, float, float]:
    side = str(order.get("side") or "").strip().lower()
    price = round(float(order.get("price") or 0.0), 8)
    qty = round(float(order.get("qty") or 0.0), 8)
    return side, price, qty
