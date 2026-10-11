"""Per-venue live execution gates, caps, and joint target-book routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.instrument import SymbolBinding
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.execution.plan import ExecutionSyncReport
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
    VenueExecutionConfig,
)
from mm.runtime.control import RuntimeControlPlane


@dataclass(frozen=True, slots=True)
class VenueExecutionGate:
    """Whether one venue may sync live orders this frame."""

    exchange: str
    role: str
    live: bool
    blocked_reasons: tuple[str, ...]
    has_orders: bool
    can_sync: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "role": self.role,
            "live": self.live,
            "blocked_reasons": list(self.blocked_reasons),
            "has_orders": self.has_orders,
            "can_sync": self.can_sync,
        }


def apply_venue_execution_caps(
    book: VenueTargetBook,
    *,
    mid: float,
    config: VenueExecutionConfig,
) -> VenueTargetBook:
    """Drop target orders exceeding source-venue size or quote-distance caps."""

    filtered: list[TargetOrder] = []
    for order in book.orders:
        qty = float(order.qty)
        if config.max_order_qty is not None and qty > float(config.max_order_qty) + 1e-12:
            continue
        if config.max_quote_distance is not None:
            if abs(float(order.price) - float(mid)) > float(config.max_quote_distance) + 1e-12:
                continue
        filtered.append(order)
    return VenueTargetBook(
        exchange=book.exchange,
        symbol=book.symbol,
        actor_id=book.actor_id,
        orders=tuple(filtered),
        policy=book.policy,
        frame_id=book.frame_id,
        decision_id=book.decision_id,
        target_book_id=book.target_book_id,
        execution_plan_id=book.execution_plan_id,
    )


def execution_book_from_joint(
    book: JointTargetBook,
    *,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
) -> JointTargetBook:
    """Return the subset of a joint book that may be sent to live execution."""

    global_book = book.global_
    if global_book is not None and not liquidity_sourcing.global_live_orders:
        global_book = VenueTargetBook(
            exchange=global_book.exchange,
            symbol=global_book.symbol,
            actor_id=global_book.actor_id,
            orders=(),
            policy=global_book.policy,
            frame_id=global_book.frame_id,
            decision_id=global_book.decision_id,
            target_book_id=global_book.target_book_id,
            execution_plan_id=global_book.execution_plan_id,
        )
    return JointTargetBook(
        actor_id=book.actor_id,
        local=book.local,
        global_=global_book,
        mode=book.mode,
        diagnostics=dict(book.diagnostics),
        frame_id=book.frame_id,
        decision_id=book.decision_id,
        target_book_id=book.target_book_id,
    )


def _global_market_stale(frame: StrategyInputFrame | None) -> bool:
    if frame is None:
        return False
    return frame.global_ is None or frame.global_status not in {"ready"}


def _venue_book(
    book: JointTargetBook | None,
    role: str,
) -> VenueTargetBook | None:
    if book is None:
        return None
    if role == "local":
        return book.local
    return book.global_


def compute_venue_execution_gates(
    *,
    config: ControllerRuntimeConfig,
    execution: ExecutionRuntimeConfig,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
    binding: SymbolBinding,
    frame_step: int,
    joint_target_book: JointTargetBook | None,
    order_managers: dict[str, AbstractOrderManager],
    control_plane: RuntimeControlPlane | None,
    frame: StrategyInputFrame | None = None,
    ignore_empty_book: bool = False,
) -> dict[str, VenueExecutionGate]:
    """Compute independent live/observe gates for local and global venues."""

    global_blocked: list[str] = []
    if config.dry_run:
        global_blocked.append("dry_run_config")
    dry_run_control = (
        control_plane is not None and control_plane.state.dry_run
    )
    if dry_run_control:
        global_blocked.append("dry_run_control")
    if frame_step <= config.execution_after_steps:
        global_blocked.append("warmup")
    if control_plane is not None and control_plane.state.kill_switch:
        global_blocked.append("kill_switch")
    if control_plane is not None and (
        config.actor_id.strip() in {str(a).strip() for a in control_plane.state.paused_actors}
    ):
        global_blocked.append("paused_actor")
    if control_plane is not None and (
        config.actor_id.strip() in {str(a).strip() for a in control_plane.state.allocation_breached_actors}
    ):
        global_blocked.append("allocation_breach")

    gates: dict[str, VenueExecutionGate] = {}
    venue_specs: list[tuple[str, str, VenueExecutionConfig, VenueTargetBook | None]] = [
        (
            binding.local.exchange,
            "local",
            execution.local,
            _venue_book(joint_target_book, "local"),
        ),
    ]
    if binding.global_ is not None:
        venue_specs.append(
            (
                binding.global_.exchange,
                "global",
                execution.global_,
                _venue_book(joint_target_book, "global"),
            )
        )

    for exchange, role, venue_exec, venue_book in venue_specs:
        exchange_key = exchange.lower().strip()
        blocked = list(global_blocked)
        has_orders = venue_book is not None and bool(venue_book.orders)
        can_sync = venue_book is not None and (has_orders or venue_book.policy == "replace")

        if venue_exec.observe_only:
            blocked.append("venue_observe_only")
        if control_plane is not None and role in control_plane.state.paused_venue_roles:
            blocked.append("venue_paused")
        if order_managers.get(exchange_key) is None:
            blocked.append("no_manager")
        if role == "global" and not liquidity_sourcing.global_live_orders:
            blocked.append("global_live_orders_disabled")
        if role == "global" and _global_market_stale(frame):
            blocked.append("stale_global_market_data")
        if not can_sync and not ignore_empty_book:
            blocked.append("no_orders")

        live = not blocked
        gates[exchange_key] = VenueExecutionGate(
            exchange=exchange_key,
            role=role,
            live=live,
            blocked_reasons=tuple(blocked),
            has_orders=has_orders,
            can_sync=can_sync,
        )
    return gates


def prepare_execution_joint_book(
    book: JointTargetBook,
    *,
    execution: ExecutionRuntimeConfig,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
    frame: StrategyInputFrame,
    binding: SymbolBinding,
) -> JointTargetBook:
    """Apply live-execution filters, caps, and configured planner policies."""

    book = execution_book_from_joint(book, liquidity_sourcing=liquidity_sourcing)
    local = VenueTargetBook(
        exchange=book.local.exchange,
        symbol=book.local.symbol,
        actor_id=book.local.actor_id,
        orders=book.local.orders,
        policy=execution.local.planner,
        frame_id=book.local.frame_id,
        decision_id=book.local.decision_id,
        target_book_id=book.local.target_book_id,
        execution_plan_id=book.local.execution_plan_id,
    )
    local = apply_venue_execution_caps(
        local,
        mid=float(frame.local.microprice),
        config=execution.local,
    )

    global_book = book.global_
    if global_book is not None and binding.global_ is not None:
        global_mid = (
            float(frame.global_.microprice)
            if frame.global_ is not None
            else float(frame.local.microprice)
        )
        global_book = VenueTargetBook(
            exchange=global_book.exchange,
            symbol=global_book.symbol,
            actor_id=global_book.actor_id,
            orders=global_book.orders,
            policy=execution.global_.planner,
            frame_id=global_book.frame_id,
            decision_id=global_book.decision_id,
            target_book_id=global_book.target_book_id,
            execution_plan_id=global_book.execution_plan_id,
        )
        global_book = apply_venue_execution_caps(
            global_book,
            mid=global_mid,
            config=execution.global_,
        )

    return JointTargetBook(
        actor_id=book.actor_id,
        local=local,
        global_=global_book,
        mode=book.mode,
        diagnostics=dict(book.diagnostics),
        frame_id=book.frame_id,
        decision_id=book.decision_id,
        target_book_id=book.target_book_id,
    )


async def route_joint_target_book_with_report(
    book: JointTargetBook,
    *,
    order_managers: dict[str, AbstractOrderManager],
    venue_gates: dict[str, VenueExecutionGate],
    execute: bool,
) -> ExecutionSyncReport:
    """Route each venue book independently using its configured planner policy."""

    combined = ExecutionSyncReport()
    for venue_book in (book.local, book.global_):
        if venue_book is None:
            continue
        exchange = venue_book.exchange.lower().strip()
        gate = venue_gates.get(exchange)
        if gate is None or not gate.can_sync:
            continue

        manager = order_managers.get(exchange)
        if manager is None:
            continue

        if execute and gate.live:
            sync_venue = getattr(manager, "sync_venue_target_book", None)
            if sync_venue is not None:
                report = await sync_venue(venue_book)
                combined = combined.merge(report)
            continue

        preview_venue = getattr(manager, "preview_venue_target_book", None)
        if preview_venue is not None:
            combined = combined.merge(preview_venue(venue_book))
    return combined
