from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mm.core.models.execution import Action
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.execution.plan import ActionPipelineReport

HedgeStatus = Literal["disabled", "ready", "missing", "stale"]


@dataclass(frozen=True, slots=True)
class StrategyInputFrame:
    """Aligned strategy input for one primary decision window.

    ``hedge`` is intentionally optional. When no hedge exchange is configured,
    or when the configured hedge stream is missing/stale for this primary
    window, strategies receive ``None`` plus an explicit ``hedge_status``.
    """

    primary: MarketEvent
    hedge: MarketEvent | None = None
    primary_private: PrivateStateSnapshot | None = None
    hedge_private: PrivateStateSnapshot | None = None
    hedge_status: HedgeStatus = "disabled"
    hedge_lag_ms: int | None = None
    max_hedge_lag_ms: int = 0
    created_at_ms: int = 0

    @property
    def has_hedge(self) -> bool:
        return self.hedge is not None and self.hedge_status == "ready"

    @property
    def t_end_ms(self) -> int:
        return self.primary.t_end_ms


@dataclass(frozen=True, slots=True)
class ModelUpdateReport:
    """Model state summary produced while evaluating one strategy frame."""

    market_flow: dict[str, Any] = field(default_factory=dict)
    price: dict[str, Any] = field(default_factory=dict)
    hedge_price: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionDiagnostics:
    """Exporter-neutral diagnostics facts for one strategy decision."""

    optimizer: dict[str, Any] = field(default_factory=dict)
    continuous_action_depth: dict[str, Any] = field(default_factory=dict)
    orderbook_grid: dict[str, Any] = field(default_factory=dict)
    unit_policy: str = "prices=quote_per_base, quantities=base, cash/notional=quote"


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Canonical strategy output consumed by execution, observability, and persistence."""

    actor_id: str
    exchange: str
    symbol: str
    t_end_ms: int
    frame_step: int
    action: Action
    observe_only: bool = True
    action_start_after_steps: int = 0
    live_execution_skipped_warmup: bool = False
    hedge_status: HedgeStatus = "disabled"
    has_hedge: bool = False
    mid: float = 0.0
    primary_buy_vwap: float = 0.0
    primary_sell_vwap: float = 0.0
    hedge_buy_vwap: float | None = None
    hedge_sell_vwap: float | None = None
    inventory: float = 0.0
    cash: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    latency_ms: float = 0.0
    decision_hash: str = ""
    orders: tuple[dict[str, Any], ...] = ()
    ladder_orders: tuple[dict[str, Any], ...] = ()
    model_updates: ModelUpdateReport = field(default_factory=ModelUpdateReport)
    model_scores: dict[str, Any] = field(default_factory=dict)
    action_pipeline_report: ActionPipelineReport | None = None
    diagnostics: DecisionDiagnostics = field(default_factory=DecisionDiagnostics)
    economics: dict[str, Any] | None = None

    @property
    def order_count(self) -> int:
        return len(self.action.orders_to_create)

    def as_payload(self, *, executed_orders: tuple[dict[str, Any], ...] | None = None) -> dict[str, Any]:
        """JSON-friendly event payload for the runtime bus and persistence."""

        return {
            "type": "StrategyDecision",
            "actor_id": self.actor_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "t_end_ms": self.t_end_ms,
            "observe_only": self.observe_only,
            "frame_step": self.frame_step,
            "action_start_after_steps": self.action_start_after_steps,
            "live_execution_skipped_warmup": self.live_execution_skipped_warmup,
            "hedge_status": self.hedge_status,
            "has_hedge": self.has_hedge,
            "mid": self.mid,
            "primary_buy_vwap": self.primary_buy_vwap,
            "primary_sell_vwap": self.primary_sell_vwap,
            "hedge_buy_vwap": self.hedge_buy_vwap,
            "hedge_sell_vwap": self.hedge_sell_vwap,
            "inventory": self.inventory,
            "cash": self.cash,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "order_count": self.order_count,
            "orders": list(self.orders),
            "ladder_orders": list(self.ladder_orders),
            "executed_orders": list(executed_orders if executed_orders is not None else ()),
            "price_model": self.model_updates.price,
            "market_flow_model": self.model_updates.market_flow,
            "hedge_price_model": self.model_updates.hedge_price,
            "optimizer": self.diagnostics.optimizer,
            "continuous_action_depth": self.diagnostics.continuous_action_depth,
            "orderbook_grid": self.diagnostics.orderbook_grid,
            "latency_ms": self.latency_ms,
            "unit_policy": self.diagnostics.unit_policy,
            "decision_hash": self.decision_hash,
            **({"economics": dict(self.economics)} if self.economics else {}),
        }
