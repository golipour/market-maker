"""Registry domain records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

ExchangeStatus = Literal["active", "disabled"]
AccountStatus = Literal["active", "disabled"]
PresetStatus = Literal["active", "disabled"]
MarketFeedStatus = Literal["active", "disabled"]
StrategyStatus = Literal["draft", "active", "paused", "retired"]
AllocationStatus = Literal["active", "paused", "retired"]
LedgerSource = Literal["virtual", "physical"]
WorkerAssignmentStatus = Literal["active", "released", "expired"]


@dataclass(frozen=True, slots=True)
class ExchangeRecord:
    exchange: str
    status: ExchangeStatus = "active"
    capabilities: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "status": self.status,
            "capabilities": dict(self.capabilities),
            "config": dict(self.config),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class AccountRecord:
    account_id: str
    exchange: str
    credential_ref: str
    role: str = "trading"
    status: AccountStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "exchange": self.exchange,
            "role": self.role,
            "status": self.status,
            "credential_ref": self.credential_ref,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class StrategyPresetRecord:
    preset_id: str
    config: dict[str, Any]
    version: int = 1
    status: PresetStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "version": self.version,
            "status": self.status,
            "config": dict(self.config),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class MarketSourceRecord:
    """Shared exchange polling config for one exchange/symbol pair."""

    source_id: str
    exchange: str
    symbol: str
    mode: str = "rest"
    depth_limit: int = 10
    trades_limit: int = 100
    poll_interval_ms: int = 1000
    status: MarketFeedStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "mode": self.mode,
            "depth_limit": self.depth_limit,
            "trades_limit": self.trades_limit,
            "poll_interval_ms": self.poll_interval_ms,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class MarketFeedRecord:
    """Aggregation interval published from a shared market source."""

    feed_id: str
    source_id: str
    interval_ms: int
    status: MarketFeedStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    exchange: str = ""
    symbol: str = ""
    mode: str = "rest"
    depth_limit: int = 10
    trades_limit: int = 100
    poll_interval_ms: int = 1000

    def as_dict(self) -> dict[str, Any]:
        return {
            "feed_id": self.feed_id,
            "source_id": self.source_id,
            "interval_ms": self.interval_ms,
            "status": self.status,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "mode": self.mode,
            "depth_limit": self.depth_limit,
            "trades_limit": self.trades_limit,
            "poll_interval_ms": self.poll_interval_ms,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    strategy_id: str
    actor_id: str
    account_id: str
    symbol: str
    preset_id: str
    quoting_exchange: str
    liquidity_exchange: str
    status: StrategyStatus = "draft"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    wallet_id: str | None = None
    ledger_source: LedgerSource = "virtual"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "actor_id": self.actor_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "preset_id": self.preset_id,
            "quoting_exchange": self.quoting_exchange,
            "liquidity_exchange": self.liquidity_exchange,
            "status": self.status,
            "config_overrides": dict(self.config_overrides),
            "wallet_id": self.wallet_id,
            "ledger_source": self.ledger_source,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class AllocationRecord:
    allocation_id: str
    strategy_id: str | None
    actor_id: str
    account_id: str
    quote_asset: str = ""
    base_asset: str = ""
    allocated_quote: float = 0.0
    allocated_base: float = 0.0
    halt_on_breach: bool = True
    status: AllocationStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "strategy_id": self.strategy_id,
            "actor_id": self.actor_id,
            "account_id": self.account_id,
            "quote_asset": self.quote_asset,
            "base_asset": self.base_asset,
            "allocated_quote": self.allocated_quote,
            "allocated_base": self.allocated_base,
            "halt_on_breach": self.halt_on_breach,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class WorkerAssignmentRecord:
    assignment_id: str
    worker_id: str
    strategy_ids: tuple[str, ...] = ()
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    status: WorkerAssignmentStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "worker_id": self.worker_id,
            "strategy_ids": list(self.strategy_ids),
            "lease_expires_at": _iso(self.lease_expires_at),
            "heartbeat_at": _iso(self.heartbeat_at),
            "status": self.status,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class PlatformSettingsRecord:
    settings_id: str
    bus: dict[str, Any] = field(default_factory=dict)
    persistence: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "settings_id": self.settings_id,
            "bus": dict(self.bus),
            "persistence": dict(self.persistence),
            "observability": dict(self.observability),
            "scheduler": dict(self.scheduler),
            "updated_at": _iso(self.updated_at),
        }


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone().isoformat()
