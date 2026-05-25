from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mm.core.models.execution import Action
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.target_book import JointTargetBook
from mm.execution.plan import ActionPipelineReport

GlobalMarketStatus = Literal["disabled", "ready", "missing", "stale"]


@dataclass(frozen=True, slots=True)
class StrategyInputFrame:
    """Aligned strategy input for one local decision window.

    ``global_`` is intentionally optional. When no global exchange is configured,
    or when the configured global stream is missing/stale for this local
    window, strategies receive ``None`` plus an explicit ``global_status``.
    The live runtime carries account/order truth in ``JointLedgerSnapshot``.
    """

    local: MarketEvent
    global_: MarketEvent | None = None
    local_private: PrivateStateSnapshot | None = None
    global_private: PrivateStateSnapshot | None = None
    global_status: GlobalMarketStatus = "disabled"
    global_lag_ms: int | None = None
    max_global_lag_ms: int = 0
    created_at_ms: int = 0

    @property
    def has_global(self) -> bool:
        return self.global_ is not None and self.global_status == "ready"

    @property
    def t_end_ms(self) -> int:
        return self.local.t_end_ms


@dataclass(frozen=True, slots=True)
class ModelUpdateReport:
    """Model state summary produced while evaluating one strategy frame."""

    local_market_flow: dict[str, Any] = field(default_factory=dict)
    local_price: dict[str, Any] = field(default_factory=dict)
    global_market_flow: dict[str, Any] = field(default_factory=dict)
    global_price: dict[str, Any] = field(default_factory=dict)


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
    joint_target_book: JointTargetBook | None = None
    observe_only: bool = True
    action_start_after_steps: int = 0
    live_execution_skipped_warmup: bool = False
    global_status: GlobalMarketStatus = "disabled"
    has_global: bool = False
    mid: float = 0.0
    local_buy_vwap: float = 0.0
    local_sell_vwap: float = 0.0
    global_buy_vwap: float | None = None
    global_sell_vwap: float | None = None
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
        return len(self.action.orders)

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
            "global_status": self.global_status,
            "has_global": self.has_global,
            "mid": self.mid,
            "local_buy_vwap": self.local_buy_vwap,
            "local_sell_vwap": self.local_sell_vwap,
            "global_buy_vwap": self.global_buy_vwap,
            "global_sell_vwap": self.global_sell_vwap,
            "inventory": self.inventory,
            "cash": self.cash,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "order_count": self.order_count,
            "orders": list(self.orders),
            "joint_target_book": (
                self.joint_target_book.as_payload() if self.joint_target_book is not None else None
            ),
            "ladder_orders": list(self.ladder_orders),
            "executed_orders": list(executed_orders if executed_orders is not None else ()),
            "local_price_model": self.model_updates.local_price,
            "local_market_flow_model": self.model_updates.local_market_flow,
            "global_market_flow_model": self.model_updates.global_market_flow,
            "global_price_model": self.model_updates.global_price,
            "optimizer": self.diagnostics.optimizer,
            "continuous_action_depth": self.diagnostics.continuous_action_depth,
            "orderbook_grid": self.diagnostics.orderbook_grid,
            "latency_ms": self.latency_ms,
            "unit_policy": self.diagnostics.unit_policy,
            "decision_hash": self.decision_hash,
            **({"economics": dict(self.economics)} if self.economics else {}),
        }
