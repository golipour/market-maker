from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

STRATEGY_ACTIONS_TOPIC = "strategy:actions"
DEFAULT_STRATEGY_DECISIONS_TOPIC = "strategy:decisions"
DEFAULT_STRATEGY_DIAGNOSTICS_TOPIC_PREFIX = "strategy:diagnostics"


class MarketDataSourceLike(Protocol):
    def market_event_topic(self, interval_ms: int) -> str:
        ...


@dataclass(frozen=True, slots=True)
class RuntimeTopics:
    """Central topic contract for one market-maker runtime."""

    primary: MarketDataSourceLike
    interval_ms: int
    hedge: MarketDataSourceLike | None = None
    frame_topic: str | None = None
    action_topic: str = STRATEGY_ACTIONS_TOPIC
    decisions_topic: str = DEFAULT_STRATEGY_DECISIONS_TOPIC
    diagnostics_topic: str | None = None

    @property
    def strategy_frame(self) -> str:
        if self.frame_topic:
            return self.frame_topic
        return self.primary.market_event_topic(self.interval_ms).replace("market:events", "strategy:frames", 1)

    @property
    def strategy_actions(self) -> str:
        return self.action_topic

    @property
    def strategy_decisions(self) -> str:
        return self.decisions_topic

    @property
    def strategy_diagnostics(self) -> str | None:
        return self.diagnostics_topic

    @property
    def primary_market_events(self) -> str:
        return self.primary.market_event_topic(self.interval_ms)

    @property
    def hedge_market_events(self) -> str | None:
        return self.hedge.market_event_topic(self.interval_ms) if self.hedge is not None else None

    @staticmethod
    def execution_orders(exchange: str) -> str:
        return f"execution:orders:{exchange.lower().strip()}"

    @staticmethod
    def execution_state(exchange: str) -> str:
        return f"execution:state:{exchange.lower().strip()}"

    @staticmethod
    def private_snapshot(exchange: str) -> str:
        return f"private:state:{exchange.lower().strip()}:snapshot"

    @staticmethod
    def ledger_snapshot() -> str:
        return "ledger:snapshot"

    @staticmethod
    def ledger_snapshot_for(exchange: str) -> str:
        return f"ledger:snapshot:{exchange.lower().strip()}"

    @staticmethod
    def joint_strategy_frame(exchange: str, symbol: str, interval_ms: int) -> str:
        return f"strategy:joint:frames:{exchange.lower().strip()}:{symbol}:{interval_ms}"
