"""In-memory registry store for unit tests and local development."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from mm.registry.models import (
    AccountRecord,
    ExchangeRecord,
    PlatformSettingsRecord,
    StrategyPresetRecord,
    StrategyRecord,
    WorkerAssignmentRecord,
)
from mm.registry.schema import DEFAULT_PLATFORM_SETTINGS_ID


class RegistryNotFoundError(KeyError):
    """Raised when a registry row does not exist."""


class RegistryConflictError(ValueError):
    """Raised when a unique registry constraint would be violated."""


class MemoryRegistryStore:
    def __init__(self) -> None:
        self._exchanges: dict[str, ExchangeRecord] = {}
        self._accounts: dict[str, AccountRecord] = {}
        self._presets: dict[str, StrategyPresetRecord] = {}
        self._strategies: dict[str, StrategyRecord] = {}
        self._assignments: dict[str, WorkerAssignmentRecord] = {}
        self._platform = PlatformSettingsRecord(settings_id=DEFAULT_PLATFORM_SETTINGS_ID)

    def ensure_schema(self) -> None:
        return None

    def get_platform_settings(self) -> PlatformSettingsRecord:
        return self._platform

    def upsert_platform_settings(
        self,
        *,
        bus: dict[str, Any] | None = None,
        persistence: dict[str, Any] | None = None,
        observability: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> PlatformSettingsRecord:
        self._platform = PlatformSettingsRecord(
            settings_id=DEFAULT_PLATFORM_SETTINGS_ID,
            bus=dict(bus if bus is not None else self._platform.bus),
            persistence=dict(persistence if persistence is not None else self._platform.persistence),
            observability=dict(observability if observability is not None else self._platform.observability),
            scheduler=dict(scheduler if scheduler is not None else self._platform.scheduler),
            updated_at=_utcnow(),
        )
        return self._platform

    def list_exchanges(self) -> list[ExchangeRecord]:
        return sorted(self._exchanges.values(), key=lambda row: row.exchange)

    def get_exchange(self, exchange: str) -> ExchangeRecord:
        key = exchange.lower().strip()
        try:
            return self._exchanges[key]
        except KeyError as exc:
            raise RegistryNotFoundError(f"exchange {exchange!r} not found") from exc

    def upsert_exchange(
        self,
        *,
        exchange: str,
        status: str = "active",
        capabilities: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> ExchangeRecord:
        key = exchange.lower().strip()
        now = _utcnow()
        previous = self._exchanges.get(key)
        record = ExchangeRecord(
            exchange=key,
            status=status,  # type: ignore[arg-type]
            capabilities=dict(capabilities or (previous.capabilities if previous else {})),
            config=dict(config or (previous.config if previous else {})),
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._exchanges[key] = record
        return record

    def list_accounts(self) -> list[AccountRecord]:
        return sorted(self._accounts.values(), key=lambda row: row.account_id)

    def get_account(self, account_id: str) -> AccountRecord:
        try:
            return self._accounts[account_id]
        except KeyError as exc:
            raise RegistryNotFoundError(f"account {account_id!r} not found") from exc

    def upsert_account(
        self,
        *,
        account_id: str,
        exchange: str,
        credential_ref: str,
        role: str = "trading",
        status: str = "active",
    ) -> AccountRecord:
        exchange_key = exchange.lower().strip()
        if exchange_key not in self._exchanges:
            raise RegistryNotFoundError(f"exchange {exchange!r} not found")
        now = _utcnow()
        previous = self._accounts.get(account_id)
        record = AccountRecord(
            account_id=account_id,
            exchange=exchange_key,
            credential_ref=credential_ref,
            role=role,
            status=status,  # type: ignore[arg-type]
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._accounts[account_id] = record
        return record

    def list_presets(self) -> list[StrategyPresetRecord]:
        return sorted(self._presets.values(), key=lambda row: row.preset_id)

    def get_preset(self, preset_id: str) -> StrategyPresetRecord:
        try:
            return self._presets[preset_id]
        except KeyError as exc:
            raise RegistryNotFoundError(f"preset {preset_id!r} not found") from exc

    def upsert_preset(
        self,
        *,
        preset_id: str,
        config: dict[str, Any],
        version: int | None = None,
        status: str = "active",
    ) -> StrategyPresetRecord:
        now = _utcnow()
        previous = self._presets.get(preset_id)
        record = StrategyPresetRecord(
            preset_id=preset_id,
            config=dict(config),
            version=int(version if version is not None else (previous.version if previous else 1)),
            status=status,  # type: ignore[arg-type]
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._presets[preset_id] = record
        return record

    def list_strategies(self) -> list[StrategyRecord]:
        return sorted(self._strategies.values(), key=lambda row: row.strategy_id)

    def get_strategy(self, strategy_id: str) -> StrategyRecord:
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise RegistryNotFoundError(f"strategy {strategy_id!r} not found") from exc

    def upsert_strategy(
        self,
        *,
        strategy_id: str,
        actor_id: str,
        account_id: str,
        symbol: str,
        preset_id: str,
        quoting_exchange: str,
        liquidity_exchange: str,
        status: str = "draft",
        config_overrides: dict[str, Any] | None = None,
    ) -> StrategyRecord:
        if account_id not in self._accounts:
            raise RegistryNotFoundError(f"account {account_id!r} not found")
        if preset_id not in self._presets:
            raise RegistryNotFoundError(f"preset {preset_id!r} not found")
        for other in self._strategies.values():
            if other.actor_id == actor_id and other.strategy_id != strategy_id:
                raise RegistryConflictError(f"actor_id {actor_id!r} already assigned")
        now = _utcnow()
        previous = self._strategies.get(strategy_id)
        record = StrategyRecord(
            strategy_id=strategy_id,
            actor_id=actor_id,
            account_id=account_id,
            symbol=symbol,
            preset_id=preset_id,
            quoting_exchange=quoting_exchange.lower().strip(),
            liquidity_exchange=liquidity_exchange.lower().strip(),
            status=status,  # type: ignore[arg-type]
            config_overrides=dict(
                config_overrides
                if config_overrides is not None
                else (previous.config_overrides if previous else {})
            ),
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._strategies[strategy_id] = record
        return record

    def patch_strategy(
        self,
        strategy_id: str,
        *,
        status: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> StrategyRecord:
        current = self.get_strategy(strategy_id)
        return self.upsert_strategy(
            strategy_id=current.strategy_id,
            actor_id=current.actor_id,
            account_id=current.account_id,
            symbol=current.symbol,
            preset_id=current.preset_id,
            quoting_exchange=current.quoting_exchange,
            liquidity_exchange=current.liquidity_exchange,
            status=status if status is not None else current.status,
            config_overrides=(
                config_overrides if config_overrides is not None else current.config_overrides
            ),
        )

    def list_worker_assignments(self) -> list[WorkerAssignmentRecord]:
        return sorted(self._assignments.values(), key=lambda row: row.assignment_id)

    def get_worker_assignment(self, assignment_id: str) -> WorkerAssignmentRecord:
        try:
            return self._assignments[assignment_id]
        except KeyError as exc:
            raise RegistryNotFoundError(f"worker assignment {assignment_id!r} not found") from exc

    def upsert_worker_assignment(
        self,
        *,
        assignment_id: str | None = None,
        worker_id: str,
        strategy_ids: list[str] | tuple[str, ...] | None = None,
        lease_ttl_seconds: int | None = 300,
        status: str = "active",
    ) -> WorkerAssignmentRecord:
        key = assignment_id or uuid.uuid4().hex
        now = _utcnow()
        previous = self._assignments.get(key)
        lease_expires_at = None
        if lease_ttl_seconds is not None and lease_ttl_seconds > 0:
            lease_expires_at = now + timedelta(seconds=int(lease_ttl_seconds))
        record = WorkerAssignmentRecord(
            assignment_id=key,
            worker_id=worker_id,
            strategy_ids=tuple(strategy_ids or (previous.strategy_ids if previous else ())),
            lease_expires_at=lease_expires_at,
            heartbeat_at=now,
            status=status,  # type: ignore[arg-type]
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._assignments[key] = record
        return record

    def heartbeat_worker_assignment(
        self,
        assignment_id: str,
        *,
        lease_ttl_seconds: int = 300,
    ) -> WorkerAssignmentRecord:
        current = self.get_worker_assignment(assignment_id)
        now = _utcnow()
        record = WorkerAssignmentRecord(
            assignment_id=current.assignment_id,
            worker_id=current.worker_id,
            strategy_ids=current.strategy_ids,
            lease_expires_at=now + timedelta(seconds=max(1, int(lease_ttl_seconds))),
            heartbeat_at=now,
            status=current.status,
            created_at=current.created_at,
            updated_at=now,
        )
        self._assignments[assignment_id] = record
        return record

    def load_runtime_bundle(self, strategy_id: str) -> dict[str, Any]:
        strategy = self.get_strategy(strategy_id)
        account = self.get_account(strategy.account_id)
        preset = self.get_preset(strategy.preset_id)
        exchanges = {
            strategy.quoting_exchange: self.get_exchange(strategy.quoting_exchange),
        }
        if strategy.liquidity_exchange != strategy.quoting_exchange:
            exchanges[strategy.liquidity_exchange] = self.get_exchange(strategy.liquidity_exchange)
        return {
            "strategy": strategy,
            "account": account,
            "preset": preset,
            "exchanges": exchanges,
            "platform": self.get_platform_settings(),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
