"""Postgres-backed strategy registry store."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from mm.registry.memory import MemoryRegistryStore, RegistryConflictError, RegistryNotFoundError
from mm.registry.models import (
    AccountRecord,
    AllocationRecord,
    ExchangeRecord,
    MarketFeedRecord,
    MarketSourceRecord,
    PlatformSettingsRecord,
    StrategyPresetRecord,
    StrategyRecord,
    WorkerAssignmentRecord,
)
from mm.registry.schema import DEFAULT_PLATFORM_SETTINGS_ID, REGISTRY_DDL, REGISTRY_MIGRATIONS

_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False
_REGISTRY_SCHEMA_ADVISORY_LOCK_KEY = 89_423_712


class PostgresRegistryStore(MemoryRegistryStore):
    """Registry store backed by Postgres.

    Mutations delegate to SQL while reads can use the parent in-memory helpers
    only in tests. Production paths use SQL for every operation.
    """

    def __init__(self, database_url: str, *, connect_timeout_s: int = 5) -> None:
        super().__init__()
        self.database_url = database_url
        self.connect_timeout_s = max(1, int(connect_timeout_s))

    def ensure_schema(self) -> None:
        """Apply DDL/migrations once per process; serialize across workers via advisory lock."""

        global _SCHEMA_READY
        if _SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if _SCHEMA_READY:
                return
            self._ensure_schema_once()
            _SCHEMA_READY = True

    def _ensure_schema_once(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(REGISTRY_DDL)
                cur.execute(
                    "SELECT pg_advisory_lock(%s)",
                    (_REGISTRY_SCHEMA_ADVISORY_LOCK_KEY,),
                )
                try:
                    cur.execute(REGISTRY_MIGRATIONS)
                finally:
                    cur.execute(
                        "SELECT pg_advisory_unlock(%s)",
                        (_REGISTRY_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
                cur.execute(
                    """
                    INSERT INTO registry_platform_settings (settings_id)
                    VALUES (%s)
                    ON CONFLICT (settings_id) DO NOTHING
                    """,
                    (DEFAULT_PLATFORM_SETTINGS_ID,),
                )
            conn.commit()

    def get_platform_settings(self) -> PlatformSettingsRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT settings_id, bus, persistence, observability, scheduler, updated_at
                    FROM registry_platform_settings
                    WHERE settings_id = %s
                    """,
                    (DEFAULT_PLATFORM_SETTINGS_ID,),
                )
                row = cur.fetchone()
            if row is None:
                raise RegistryNotFoundError("platform settings not found")
        return _platform_from_row(row)

    def upsert_platform_settings(
        self,
        *,
        bus: dict[str, Any] | None = None,
        persistence: dict[str, Any] | None = None,
        observability: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> PlatformSettingsRecord:
        current = self.get_platform_settings()
        record = PlatformSettingsRecord(
            settings_id=DEFAULT_PLATFORM_SETTINGS_ID,
            bus=dict(bus if bus is not None else current.bus),
            persistence=dict(persistence if persistence is not None else current.persistence),
            observability=dict(observability if observability is not None else current.observability),
            scheduler=dict(scheduler if scheduler is not None else current.scheduler),
            updated_at=_utcnow(),
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_platform_settings (
                        settings_id, bus, persistence, observability, scheduler, updated_at
                    ) VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (settings_id) DO UPDATE SET
                        bus = EXCLUDED.bus,
                        persistence = EXCLUDED.persistence,
                        observability = EXCLUDED.observability,
                        scheduler = EXCLUDED.scheduler,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.settings_id,
                        _json(record.bus),
                        _json(record.persistence),
                        _json(record.observability),
                        _json(record.scheduler),
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_exchanges(self) -> list[ExchangeRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT exchange, status, capabilities, config, created_at, updated_at
                    FROM registry_exchanges
                    ORDER BY exchange
                    """
                )
                rows = cur.fetchall()
        return [_exchange_from_row(row) for row in rows]

    def get_exchange(self, exchange: str) -> ExchangeRecord:
        key = exchange.lower().strip()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT exchange, status, capabilities, config, created_at, updated_at
                    FROM registry_exchanges
                    WHERE exchange = %s
                    """,
                    (key,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"exchange {exchange!r} not found")
        return _exchange_from_row(row)

    def upsert_exchange(
        self,
        *,
        exchange: str,
        status: str = "active",
        capabilities: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> ExchangeRecord:
        key = exchange.lower().strip()
        previous = None
        try:
            previous = self.get_exchange(key)
        except RegistryNotFoundError:
            pass
        now = _utcnow()
        record = ExchangeRecord(
            exchange=key,
            status=status,  # type: ignore[arg-type]
            capabilities=dict(capabilities or (previous.capabilities if previous else {})),
            config=dict(config or (previous.config if previous else {})),
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_exchanges (
                        exchange, status, capabilities, config, created_at, updated_at
                    ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    ON CONFLICT (exchange) DO UPDATE SET
                        status = EXCLUDED.status,
                        capabilities = EXCLUDED.capabilities,
                        config = EXCLUDED.config,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.exchange,
                        record.status,
                        _json(record.capabilities),
                        _json(record.config),
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_accounts(self) -> list[AccountRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, exchange, role, status, credential_ref, created_at, updated_at
                    FROM registry_accounts
                    ORDER BY account_id
                    """
                )
                rows = cur.fetchall()
        return [_account_from_row(row) for row in rows]

    def get_account(self, account_id: str) -> AccountRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, exchange, role, status, credential_ref, created_at, updated_at
                    FROM registry_accounts
                    WHERE account_id = %s
                    """,
                    (account_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"account {account_id!r} not found")
        return _account_from_row(row)

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
        self.get_exchange(exchange_key)
        now = _utcnow()
        previous = None
        try:
            previous = self.get_account(account_id)
        except RegistryNotFoundError:
            pass
        record = AccountRecord(
            account_id=account_id,
            exchange=exchange_key,
            credential_ref=credential_ref,
            role=role,
            status=status,  # type: ignore[arg-type]
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_accounts (
                        account_id, exchange, role, status, credential_ref, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (account_id) DO UPDATE SET
                        exchange = EXCLUDED.exchange,
                        role = EXCLUDED.role,
                        status = EXCLUDED.status,
                        credential_ref = EXCLUDED.credential_ref,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.account_id,
                        record.exchange,
                        record.role,
                        record.status,
                        record.credential_ref,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_presets(self) -> list[StrategyPresetRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT preset_id, version, status, config, created_at, updated_at
                    FROM registry_strategy_presets
                    ORDER BY preset_id
                    """
                )
                rows = cur.fetchall()
        return [_preset_from_row(row) for row in rows]

    def get_preset(self, preset_id: str) -> StrategyPresetRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT preset_id, version, status, config, created_at, updated_at
                    FROM registry_strategy_presets
                    WHERE preset_id = %s
                    """,
                    (preset_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"preset {preset_id!r} not found")
        return _preset_from_row(row)

    def upsert_preset(
        self,
        *,
        preset_id: str,
        config: dict[str, Any],
        version: int | None = None,
        status: str = "active",
    ) -> StrategyPresetRecord:
        previous = None
        try:
            previous = self.get_preset(preset_id)
        except RegistryNotFoundError:
            pass
        now = _utcnow()
        record = StrategyPresetRecord(
            preset_id=preset_id,
            config=dict(config),
            version=int(version if version is not None else (previous.version if previous else 1)),
            status=status,  # type: ignore[arg-type]
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_strategy_presets (
                        preset_id, version, status, config, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (preset_id) DO UPDATE SET
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        config = EXCLUDED.config,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.preset_id,
                        record.version,
                        record.status,
                        _json(record.config),
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_strategies(self) -> list[StrategyRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT strategy_id, actor_id, account_id, symbol, preset_id,
                           quoting_exchange, liquidity_exchange, status, config_overrides,
                           wallet_id, ledger_source, created_at, updated_at
                    FROM registry_strategies
                    ORDER BY strategy_id
                    """
                )
                rows = cur.fetchall()
        return [_strategy_from_row(row) for row in rows]

    def get_strategy(self, strategy_id: str) -> StrategyRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT strategy_id, actor_id, account_id, symbol, preset_id,
                           quoting_exchange, liquidity_exchange, status, config_overrides,
                           wallet_id, ledger_source, created_at, updated_at
                    FROM registry_strategies
                    WHERE strategy_id = %s
                    """,
                    (strategy_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"strategy {strategy_id!r} not found")
        return _strategy_from_row(row)

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
        wallet_id: str | None = None,
        ledger_source: str = "virtual",
    ) -> StrategyRecord:
        record = super().upsert_strategy(
            strategy_id=strategy_id,
            actor_id=actor_id,
            account_id=account_id,
            symbol=symbol,
            preset_id=preset_id,
            quoting_exchange=quoting_exchange,
            liquidity_exchange=liquidity_exchange,
            status=status,
            config_overrides=config_overrides,
            wallet_id=wallet_id,
            ledger_source=ledger_source,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT strategy_id FROM registry_strategies WHERE actor_id = %s AND strategy_id <> %s",
                    (record.actor_id, record.strategy_id),
                )
                if cur.fetchone() is not None:
                    raise RegistryConflictError(f"actor_id {record.actor_id!r} already assigned")
                cur.execute(
                    """
                    INSERT INTO registry_strategies (
                        strategy_id, actor_id, account_id, symbol, preset_id,
                        quoting_exchange, liquidity_exchange, status, config_overrides,
                        wallet_id, ledger_source, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (strategy_id) DO UPDATE SET
                        actor_id = EXCLUDED.actor_id,
                        account_id = EXCLUDED.account_id,
                        symbol = EXCLUDED.symbol,
                        preset_id = EXCLUDED.preset_id,
                        quoting_exchange = EXCLUDED.quoting_exchange,
                        liquidity_exchange = EXCLUDED.liquidity_exchange,
                        status = EXCLUDED.status,
                        config_overrides = EXCLUDED.config_overrides,
                        wallet_id = EXCLUDED.wallet_id,
                        ledger_source = EXCLUDED.ledger_source,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.strategy_id,
                        record.actor_id,
                        record.account_id,
                        record.symbol,
                        record.preset_id,
                        record.quoting_exchange,
                        record.liquidity_exchange,
                        record.status,
                        _json(record.config_overrides),
                        record.wallet_id,
                        record.ledger_source,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        if record.wallet_id:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    wallet = self.get_allocation(record.wallet_id)
                    cur.execute(
                        """
                        UPDATE registry_allocations
                        SET actor_id = %s, strategy_id = NULL, updated_at = %s
                        WHERE allocation_id = %s
                        """,
                        (record.actor_id, record.updated_at, wallet.allocation_id),
                    )
                conn.commit()
        return record

    def patch_strategy(
        self,
        strategy_id: str,
        *,
        status: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        wallet_id: str | None = None,
        ledger_source: str | None = None,
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
            wallet_id=wallet_id if wallet_id is not None else current.wallet_id,
            ledger_source=ledger_source if ledger_source is not None else current.ledger_source,
        )

    def list_allocations(self) -> list[AllocationRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT allocation_id, strategy_id, actor_id, account_id, quote_asset,
                           base_asset, allocated_quote, allocated_base, halt_on_breach,
                           status, created_at, updated_at
                    FROM registry_allocations
                    ORDER BY allocation_id
                    """
                )
                rows = cur.fetchall()
        return [_allocation_from_row(row) for row in rows]

    def get_allocation(self, allocation_id: str) -> AllocationRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT allocation_id, strategy_id, actor_id, account_id, quote_asset,
                           base_asset, allocated_quote, allocated_base, halt_on_breach,
                           status, created_at, updated_at
                    FROM registry_allocations
                    WHERE allocation_id = %s
                    """,
                    (allocation_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"allocation {allocation_id!r} not found")
        return _allocation_from_row(row)

    def get_allocation_for_strategy(self, strategy_id: str) -> AllocationRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT allocation_id, strategy_id, actor_id, account_id, quote_asset,
                           base_asset, allocated_quote, allocated_base, halt_on_breach,
                           status, created_at, updated_at
                    FROM registry_allocations
                    WHERE strategy_id = %s
                    """,
                    (strategy_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"allocation for strategy {strategy_id!r} not found")
        return _allocation_from_row(row)

    def get_allocation_for_actor(self, actor_id: str) -> AllocationRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT allocation_id, strategy_id, actor_id, account_id, quote_asset,
                           base_asset, allocated_quote, allocated_base, halt_on_breach,
                           status, created_at, updated_at
                    FROM registry_allocations
                    WHERE actor_id = %s
                    """,
                    (actor_id.strip(),),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"allocation for actor {actor_id!r} not found")
        return _allocation_from_row(row)

    def upsert_allocation(
        self,
        *,
        allocation_id: str | None = None,
        strategy_id: str | None = None,
        actor_id: str = "",
        account_id: str,
        quote_asset: str = "",
        base_asset: str = "",
        allocated_quote: float = 0.0,
        allocated_base: float = 0.0,
        halt_on_breach: bool = True,
        status: str = "active",
    ) -> AllocationRecord:
        record = super().upsert_allocation(
            allocation_id=allocation_id,
            strategy_id=strategy_id,
            actor_id=actor_id,
            account_id=account_id,
            quote_asset=quote_asset,
            base_asset=base_asset,
            allocated_quote=allocated_quote,
            allocated_base=allocated_base,
            halt_on_breach=halt_on_breach,
            status=status,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_allocations (
                        allocation_id, strategy_id, actor_id, account_id, quote_asset,
                        base_asset, allocated_quote, allocated_base, halt_on_breach,
                        status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (allocation_id) DO UPDATE SET
                        strategy_id = EXCLUDED.strategy_id,
                        actor_id = EXCLUDED.actor_id,
                        account_id = EXCLUDED.account_id,
                        quote_asset = EXCLUDED.quote_asset,
                        base_asset = EXCLUDED.base_asset,
                        allocated_quote = EXCLUDED.allocated_quote,
                        allocated_base = EXCLUDED.allocated_base,
                        halt_on_breach = EXCLUDED.halt_on_breach,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.allocation_id,
                        record.strategy_id,
                        record.actor_id,
                        record.account_id,
                        record.quote_asset,
                        record.base_asset,
                        record.allocated_quote,
                        record.allocated_base,
                        record.halt_on_breach,
                        record.status,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_market_sources(self) -> list[MarketSourceRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_id, exchange, symbol, mode, depth_limit, trades_limit,
                           poll_interval_ms, status, created_at, updated_at
                    FROM registry_market_sources
                    ORDER BY source_id
                    """
                )
                rows = cur.fetchall()
        return [_market_source_from_row(row) for row in rows]

    def get_market_source(self, source_id: str) -> MarketSourceRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_id, exchange, symbol, mode, depth_limit, trades_limit,
                           poll_interval_ms, status, created_at, updated_at
                    FROM registry_market_sources
                    WHERE source_id = %s
                    """,
                    (source_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"market source {source_id!r} not found")
        return _market_source_from_row(row)

    def upsert_market_source(
        self,
        *,
        source_id: str | None = None,
        exchange: str,
        symbol: str,
        mode: str = "rest",
        depth_limit: int = 10,
        trades_limit: int = 100,
        poll_interval_ms: int = 1000,
        status: str = "active",
    ) -> MarketSourceRecord:
        record = super().upsert_market_source(
            source_id=source_id,
            exchange=exchange,
            symbol=symbol,
            mode=mode,
            depth_limit=depth_limit,
            trades_limit=trades_limit,
            poll_interval_ms=poll_interval_ms,
            status=status,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_market_sources (
                        source_id, exchange, symbol, mode, depth_limit, trades_limit,
                        poll_interval_ms, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id) DO UPDATE SET
                        exchange = EXCLUDED.exchange,
                        symbol = EXCLUDED.symbol,
                        mode = EXCLUDED.mode,
                        depth_limit = EXCLUDED.depth_limit,
                        trades_limit = EXCLUDED.trades_limit,
                        poll_interval_ms = EXCLUDED.poll_interval_ms,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.source_id,
                        record.exchange,
                        record.symbol,
                        record.mode,
                        record.depth_limit,
                        record.trades_limit,
                        record.poll_interval_ms,
                        record.status,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return record

    def list_market_feeds(self) -> list[MarketFeedRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT f.feed_id, f.source_id, f.interval_ms, f.status, f.created_at, f.updated_at,
                           s.exchange, s.symbol, s.mode, s.depth_limit, s.trades_limit, s.poll_interval_ms
                    FROM registry_market_feeds f
                    JOIN registry_market_sources s ON s.source_id = f.source_id
                    ORDER BY f.feed_id
                    """
                )
                rows = cur.fetchall()
        return [_market_feed_from_row(row) for row in rows]

    def get_market_feed(self, feed_id: str) -> MarketFeedRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT f.feed_id, f.source_id, f.interval_ms, f.status, f.created_at, f.updated_at,
                           s.exchange, s.symbol, s.mode, s.depth_limit, s.trades_limit, s.poll_interval_ms
                    FROM registry_market_feeds f
                    JOIN registry_market_sources s ON s.source_id = f.source_id
                    WHERE f.feed_id = %s
                    """,
                    (feed_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"market feed {feed_id!r} not found")
        return _market_feed_from_row(row)

    def upsert_market_feed(
        self,
        *,
        feed_id: str | None = None,
        source_id: str,
        interval_ms: int,
        status: str = "active",
    ) -> MarketFeedRecord:
        record = super().upsert_market_feed(
            feed_id=feed_id,
            source_id=source_id,
            interval_ms=interval_ms,
            status=status,
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_market_feeds (
                        feed_id, source_id, interval_ms, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (feed_id) DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        interval_ms = EXCLUDED.interval_ms,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.feed_id,
                        record.source_id,
                        record.interval_ms,
                        record.status,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
        return self.get_market_feed(record.feed_id)

    def list_worker_assignments(self) -> list[WorkerAssignmentRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assignment_id, worker_id, strategy_ids, lease_expires_at,
                           heartbeat_at, status, created_at, updated_at
                    FROM registry_worker_assignments
                    ORDER BY assignment_id
                    """
                )
                rows = cur.fetchall()
        return [_assignment_from_row(row) for row in rows]

    def get_worker_assignment(self, assignment_id: str) -> WorkerAssignmentRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assignment_id, worker_id, strategy_ids, lease_expires_at,
                           heartbeat_at, status, created_at, updated_at
                    FROM registry_worker_assignments
                    WHERE assignment_id = %s
                    """,
                    (assignment_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"worker assignment {assignment_id!r} not found")
        return _assignment_from_row(row)

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
        previous = None
        try:
            previous = self.get_worker_assignment(key)
        except RegistryNotFoundError:
            pass
        now = _utcnow()
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
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO registry_worker_assignments (
                        assignment_id, worker_id, strategy_ids, lease_expires_at,
                        heartbeat_at, status, created_at, updated_at
                    ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s)
                    ON CONFLICT (assignment_id) DO UPDATE SET
                        worker_id = EXCLUDED.worker_id,
                        strategy_ids = EXCLUDED.strategy_ids,
                        lease_expires_at = EXCLUDED.lease_expires_at,
                        heartbeat_at = EXCLUDED.heartbeat_at,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.assignment_id,
                        record.worker_id,
                        _json(list(record.strategy_ids)),
                        record.lease_expires_at,
                        record.heartbeat_at,
                        record.status,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            conn.commit()
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
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE registry_worker_assignments
                    SET lease_expires_at = %s, heartbeat_at = %s, updated_at = %s
                    WHERE assignment_id = %s
                    """,
                    (
                        record.lease_expires_at,
                        record.heartbeat_at,
                        record.updated_at,
                        record.assignment_id,
                    ),
                )
            conn.commit()
        return record

    def get_active_worker_assignment(self, worker_id: str) -> WorkerAssignmentRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assignment_id, worker_id, strategy_ids, lease_expires_at,
                           heartbeat_at, status, created_at, updated_at
                    FROM registry_worker_assignments
                    WHERE worker_id = %s AND status = 'active'
                    ORDER BY heartbeat_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (worker_id.strip(),),
                )
                row = cur.fetchone()
        if row is None:
            raise RegistryNotFoundError(f"no active worker assignment for worker_id {worker_id!r}")
        return _assignment_from_row(row)

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

    def _connection(self) -> Any:
        import psycopg2

        return psycopg2.connect(self.database_url, connect_timeout=self.connect_timeout_s)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _exchange_from_row(row: tuple[Any, ...]) -> ExchangeRecord:
    return ExchangeRecord(
        exchange=row[0],
        status=row[1],
        capabilities=dict(row[2] or {}),
        config=dict(row[3] or {}),
        created_at=row[4],
        updated_at=row[5],
    )


def _account_from_row(row: tuple[Any, ...]) -> AccountRecord:
    return AccountRecord(
        account_id=row[0],
        exchange=row[1],
        role=row[2],
        status=row[3],
        credential_ref=row[4],
        created_at=row[5],
        updated_at=row[6],
    )


def _preset_from_row(row: tuple[Any, ...]) -> StrategyPresetRecord:
    return StrategyPresetRecord(
        preset_id=row[0],
        version=int(row[1]),
        status=row[2],
        config=dict(row[3] or {}),
        created_at=row[4],
        updated_at=row[5],
    )


def _strategy_from_row(row: tuple[Any, ...]) -> StrategyRecord:
    return StrategyRecord(
        strategy_id=row[0],
        actor_id=row[1],
        account_id=row[2],
        symbol=row[3],
        preset_id=row[4],
        quoting_exchange=row[5],
        liquidity_exchange=row[6],
        status=row[7],
        config_overrides=dict(row[8] or {}),
        wallet_id=row[9],
        ledger_source=row[10] or "virtual",  # type: ignore[arg-type]
        created_at=row[11],
        updated_at=row[12],
    )


def _allocation_from_row(row: tuple[Any, ...]) -> AllocationRecord:
    return AllocationRecord(
        allocation_id=row[0],
        strategy_id=row[1],
        actor_id=row[2],
        account_id=row[3],
        quote_asset=str(row[4] or ""),
        base_asset=str(row[5] or ""),
        allocated_quote=float(row[6] or 0.0),
        allocated_base=float(row[7] or 0.0),
        halt_on_breach=bool(row[8]),
        status=row[9],
        created_at=row[10],
        updated_at=row[11],
    )


def _market_source_from_row(row: tuple[Any, ...]) -> MarketSourceRecord:
    return MarketSourceRecord(
        source_id=row[0],
        exchange=row[1],
        symbol=row[2],
        mode=str(row[3] or "rest"),
        depth_limit=int(row[4] or 10),
        trades_limit=int(row[5] or 100),
        poll_interval_ms=int(row[6] or 1000),
        status=row[7],
        created_at=row[8],
        updated_at=row[9],
    )


def _market_feed_from_row(row: tuple[Any, ...]) -> MarketFeedRecord:
    return MarketFeedRecord(
        feed_id=row[0],
        source_id=row[1],
        interval_ms=int(row[2]),
        status=row[3],
        created_at=row[4],
        updated_at=row[5],
        exchange=row[6],
        symbol=row[7],
        mode=str(row[8] or "rest"),
        depth_limit=int(row[9] or 10),
        trades_limit=int(row[10] or 100),
        poll_interval_ms=int(row[11] or 1000),
    )


def _assignment_from_row(row: tuple[Any, ...]) -> WorkerAssignmentRecord:
    strategy_ids = row[2] or []
    return WorkerAssignmentRecord(
        assignment_id=row[0],
        worker_id=row[1],
        strategy_ids=tuple(strategy_ids),
        lease_expires_at=row[3],
        heartbeat_at=row[4],
        status=row[5],
        created_at=row[6],
        updated_at=row[7],
    )


def _platform_from_row(row: tuple[Any, ...]) -> PlatformSettingsRecord:
    return PlatformSettingsRecord(
        settings_id=row[0],
        bus=dict(row[1] or {}),
        persistence=dict(row[2] or {}),
        observability=dict(row[3] or {}),
        scheduler=dict(row[4] or {}),
        updated_at=row[5],
    )
