"""Subscribe to primary fills and place offsetting hedge market orders."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mm.application.action_routing import route_actions_to_managers_with_report
from mm.bus.base import AbstractEventBus
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Action, Fill, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.private import PrivateStateSnapshot
from mm.core.risk_limits import HedgeVenueRiskLimits
from mm.hedging.models import HedgeExecutionDiagnostic, HedgeIntent
from mm.hedging.risk import validate_hedge_market_order
from mm.runtime.config import HedgingRuntimeConfig, MarketMakerRuntimeConfig
from mm.runtime.control import RuntimeControlPlane


def _signed_qty(side: str, qty: float) -> float:
    return float(qty) if str(side).lower().strip() == "buy" else -float(qty)


def _hedge_side_for_primary_fill(primary_side: str) -> str:
    """Primary buy increases inventory → sell hedge; primary sell → buy hedge."""

    side = str(primary_side).lower().strip()
    if side == "buy":
        return "sell"
    if side == "sell":
        return "buy"
    raise ValueError(f"unsupported fill side: {primary_side}")


def _fill_dedupe_key(fill: Fill) -> str:
    return f"{fill.exchange_order_id}:{fill.timestamp_ms}:{fill.qty}:{fill.side}"


def _inventory_cash(snapshot: PrivateStateSnapshot | None, base_asset: str, quote_asset: str) -> tuple[float, float]:
    if snapshot is None:
        return 0.0, 0.0
    base = snapshot.balances.get(base_asset)
    quote = snapshot.balances.get(quote_asset)
    inventory = (base.free + base.locked) if base is not None else 0.0
    cash = (quote.free + quote.locked) if quote is not None else 0.0
    return float(inventory), float(cash)


@dataclass(slots=True)
class HedgeExecutionService:
    """Fill-driven hedge executor; independent of MPC ``hedge_rollout_enabled``."""

    config: HedgingRuntimeConfig
    runtime_config: MarketMakerRuntimeConfig
    binding: SymbolBinding
    bus: AbstractEventBus
    order_managers: dict[str, AbstractOrderManager]
    hedge_limits: HedgeVenueRiskLimits
    control_plane: RuntimeControlPlane | None = None
    run_id: str = ""
    _residual_unhedged_qty: float = field(default=0.0, init=False)
    _seen_fills: dict[str, int] = field(default_factory=dict, init=False)
    _latest_hedge_private: PrivateStateSnapshot | None = field(default=None, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _private_task: asyncio.Task | None = field(default=None, init=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    @property
    def residual_unhedged_qty(self) -> float:
        return self._residual_unhedged_qty

    async def start(self) -> None:
        if self.runtime_config.hedge is None:
            return
        self._stopped.clear()
        hedge_exchange = self.runtime_config.hedge.normalized_exchange()
        self._private_task = asyncio.create_task(self._listen_hedge_private(hedge_exchange))
        self._task = asyncio.create_task(self._listen_primary_fills())

    async def stop(self) -> None:
        self._stopped.set()
        for task in (self._task, self._private_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *[t for t in (self._task, self._private_task) if t is not None],
            return_exceptions=True,
        )
        self._task = None
        self._private_task = None

    async def _listen_hedge_private(self, hedge_exchange: str) -> None:
        topic = f"private:state:{hedge_exchange}:snapshot"
        try:
            async for event in self.bus.subscribe(topic):
                if isinstance(event, PrivateStateSnapshot):
                    self._latest_hedge_private = event
        except asyncio.CancelledError:
            raise

    async def _listen_primary_fills(self) -> None:
        primary_exchange = self.runtime_config.primary.normalized_exchange()
        topic = f"private:state:{primary_exchange}:fills"
        try:
            async for event in self.bus.subscribe(topic):
                if not isinstance(event, Fill):
                    continue
                await self._on_primary_fill(event)
        except asyncio.CancelledError:
            raise

    async def _on_primary_fill(self, fill: Fill) -> None:
        if not self.config.enabled:
            return
        if self._stopped.is_set():
            return
        if fill.exchange.lower().strip() != self.runtime_config.primary.normalized_exchange():
            return
        if fill.symbol.strip() != self.runtime_config.primary.symbol.strip():
            return
        if fill.qty <= 0.0:
            return

        now_ms = int(time.time() * 1000)
        dedupe_key = _fill_dedupe_key(fill)
        if dedupe_key in self._seen_fills:
            return
        self._prune_seen_fills(now_ms)
        self._seen_fills[dedupe_key] = now_ms

        hedge_exchange = self.runtime_config.hedge.normalized_exchange() if self.runtime_config.hedge else ""
        hedge_symbol = self.runtime_config.hedge.symbol if self.runtime_config.hedge else fill.symbol
        try:
            hedge_side = _hedge_side_for_primary_fill(fill.side)
        except ValueError as exc:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="skipped",
                    reason=str(exc),
                    hedge_exchange=hedge_exchange,
                    symbol=hedge_symbol,
                    timestamp_ms=now_ms,
                )
            )
            return

        intent = HedgeIntent(
            actor_id=self.config.actor_id,
            source_fill_id=fill.exchange_order_id,
            primary_exchange=fill.exchange,
            hedge_exchange=hedge_exchange,
            symbol=hedge_symbol,
            primary_side=fill.side,
            hedge_side=hedge_side,
            qty=float(fill.qty),
            reference_price=float(fill.price) if fill.price > 0.0 else None,
            timestamp_ms=now_ms,
            run_id=self.run_id,
        )
        await self.bus.publish(self.config.normalized_intents_topic, intent.as_payload())

        self._residual_unhedged_qty += _signed_qty(fill.side, fill.qty)
        await self._execute_intent(intent, fill=fill)

    async def _execute_intent(self, intent: HedgeIntent, *, fill: Fill) -> None:
        now_ms = intent.timestamp_ms
        if self.control_plane is not None and self.control_plane.state.observe_only:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="skipped",
                    reason="observe_only_control",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                )
            )
            return
        if self.config.observe_only:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="skipped",
                    reason="observe_only_config",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                )
            )
            return

        manager = self.order_managers.get(intent.hedge_exchange.lower().strip())
        if manager is None:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="rejected",
                    reason="no_hedge_order_manager",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                )
            )
            return

        hedge_spec = self.binding.hedge
        if hedge_spec is None:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="rejected",
                    reason="hedge_not_configured",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                )
            )
            return

        ref_price = intent.reference_price if intent.reference_price and intent.reference_price > 0.0 else 1.0
        hedge_order = Order(
            exchange=intent.hedge_exchange,
            symbol=intent.symbol,
            side=intent.hedge_side,
            price=ref_price,
            qty=intent.qty,
            order_type="market",
            client_order_id=f"hedge-{uuid.uuid4().hex[:16]}",
        )

        hedge_inventory, hedge_cash = _inventory_cash(
            self._latest_hedge_private,
            hedge_spec.base_asset,
            hedge_spec.quote_asset,
        )
        ok, reason = validate_hedge_market_order(
            hedge_order,
            limits=self.hedge_limits,
            hedge_inventory=hedge_inventory,
            hedge_cash=hedge_cash,
        )
        if not ok:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="rejected",
                    reason=reason,
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                    hedge_order=hedge_order,
                )
            )
            return

        action = Action(actor_id=self.config.actor_id, orders_to_create=(hedge_order,))
        try:
            report = await route_actions_to_managers_with_report([action], self.order_managers)
            placed = report.accepted.order_count > 0
        except Exception as exc:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                    hedge_order=hedge_order,
                )
            )
            return

        if not placed:
            await self._publish_diagnostic(
                self._diagnostic(
                    fill,
                    status="rejected",
                    reason="venue_rejected",
                    hedge_exchange=intent.hedge_exchange,
                    symbol=intent.symbol,
                    timestamp_ms=now_ms,
                    intent=intent,
                    hedge_order=hedge_order,
                )
            )
            return

        self._residual_unhedged_qty += _signed_qty(intent.hedge_side, intent.qty)
        await self._publish_diagnostic(
            self._diagnostic(
                fill,
                status="submitted",
                reason=None,
                hedge_exchange=intent.hedge_exchange,
                symbol=intent.symbol,
                timestamp_ms=now_ms,
                intent=intent,
                hedge_order=hedge_order,
            )
        )

    def _diagnostic(
        self,
        fill: Fill,
        *,
        status: str,
        reason: str | None,
        hedge_exchange: str,
        symbol: str,
        timestamp_ms: int,
        intent: HedgeIntent | None = None,
        hedge_order: Order | None = None,
    ) -> HedgeExecutionDiagnostic:
        intent_payload: dict[str, Any] = intent.as_payload() if intent is not None else {}
        order_payload: dict[str, Any] = {}
        if hedge_order is not None:
            order_payload = {
                "exchange": hedge_order.exchange,
                "symbol": hedge_order.symbol,
                "side": hedge_order.side,
                "qty": hedge_order.qty,
                "order_type": hedge_order.order_type,
                "client_order_id": hedge_order.client_order_id,
            }
        return HedgeExecutionDiagnostic(
            actor_id=self.config.actor_id,
            status=status,
            reason=reason,
            source_fill_id=fill.exchange_order_id,
            primary_exchange=fill.exchange,
            hedge_exchange=hedge_exchange,
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            run_id=self.run_id,
            intent=intent_payload,
            hedge_order=order_payload,
            residual_unhedged_qty=self._residual_unhedged_qty,
        )

    async def _publish_diagnostic(self, diagnostic: HedgeExecutionDiagnostic) -> None:
        await self.bus.publish(self.config.normalized_diagnostics_topic, diagnostic.as_payload())

    def _prune_seen_fills(self, now_ms: int) -> None:
        window = max(1000, int(self.config.dedupe_window_ms))
        cutoff = now_ms - window
        self._seen_fills = {key: ts for key, ts in self._seen_fills.items() if ts >= cutoff}


def build_hedge_execution_service(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus,
    order_managers: dict[str, AbstractOrderManager],
    control_plane: RuntimeControlPlane | None = None,
) -> HedgeExecutionService | None:
    hedging = config.hedging
    if hedging is None or not hedging.enabled:
        return None
    if config.hedge is None:
        return None
    hedge_limits = (
        config.controller.risk.hedge
        if config.controller is not None
        else HedgeVenueRiskLimits()
    )
    return HedgeExecutionService(
        config=hedging,
        runtime_config=config,
        binding=config.binding,
        bus=bus,
        order_managers=order_managers,
        hedge_limits=hedge_limits,
        control_plane=control_plane,
        run_id=config.observability.run_id,
    )
