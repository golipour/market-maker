"""Joint strategy frame models for liquidity sourcing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.ledger import JointLedgerSnapshot
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import GlobalMarketStatus
from mm.core.models.target_book import LiquiditySourcingMode


@dataclass(frozen=True, slots=True)
class FrameSyncStatus:
    """Synchronization readiness for joint strategy decisions."""

    local_market_ready: bool = False
    global_market_ready: bool = False
    local_private_ready: bool = False
    global_private_ready: bool = False
    ledger_fresh: bool = True
    latency_skew_ms: int | None = None
    global_status: GlobalMarketStatus = "disabled"

    def as_payload(self) -> dict[str, Any]:
        return {
            "local_market_ready": self.local_market_ready,
            "global_market_ready": self.global_market_ready,
            "local_private_ready": self.local_private_ready,
            "global_private_ready": self.global_private_ready,
            "ledger_fresh": self.ledger_fresh,
            "latency_skew_ms": self.latency_skew_ms,
            "global_status": self.global_status,
        }


@dataclass(frozen=True, slots=True)
class JointStrategyFrame:
    """Strategy input with neutral local/global roles and ledger state."""

    local_market: MarketEvent
    global_market: MarketEvent | None = None
    ledger: JointLedgerSnapshot | None = None
    mode: LiquiditySourcingMode = "single_venue"
    sync_status: FrameSyncStatus = field(default_factory=FrameSyncStatus)
    created_at_ms: int = 0
    input_frame: "StrategyInputFrame | None" = None

    @property
    def t_end_ms(self) -> int:
        return self.local_market.t_end_ms

    def as_payload(self) -> dict[str, Any]:
        return {
            "local_market": {
                "exchange": self.local_market.exchange,
                "symbol": self.local_market.symbol,
                "t_end_ms": self.local_market.t_end_ms,
                "interval_ms": self.local_market.interval_ms,
            },
            "global_market": (
                {
                    "exchange": self.global_market.exchange,
                    "symbol": self.global_market.symbol,
                    "t_end_ms": self.global_market.t_end_ms,
                    "interval_ms": self.global_market.interval_ms,
                }
                if self.global_market is not None
                else None
            ),
            "ledger": self.ledger.as_payload() if self.ledger is not None else None,
            "mode": self.mode,
            "sync_status": self.sync_status.as_payload(),
            "created_at_ms": int(self.created_at_ms),
            "t_end_ms": self.t_end_ms,
        }
