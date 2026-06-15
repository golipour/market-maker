"""Postgres DDL for the strategy registry and platform settings."""

from __future__ import annotations

REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS registry_exchanges (
    exchange TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS registry_accounts (
    account_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL REFERENCES registry_exchanges(exchange),
    role TEXT NOT NULL DEFAULT 'trading',
    status TEXT NOT NULL DEFAULT 'active',
    credential_ref TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS registry_accounts_exchange_idx ON registry_accounts (exchange);

CREATE TABLE IF NOT EXISTS registry_strategy_presets (
    preset_id TEXT PRIMARY KEY,
    version INT NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS registry_strategies (
    strategy_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL UNIQUE,
    account_id TEXT NOT NULL REFERENCES registry_accounts(account_id),
    symbol TEXT NOT NULL,
    preset_id TEXT NOT NULL REFERENCES registry_strategy_presets(preset_id),
    quoting_exchange TEXT NOT NULL,
    liquidity_exchange TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    config_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
    wallet_id TEXT,
    ledger_source TEXT NOT NULL DEFAULT 'virtual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS registry_strategies_account_idx ON registry_strategies (account_id);
CREATE INDEX IF NOT EXISTS registry_strategies_preset_idx ON registry_strategies (preset_id);
CREATE INDEX IF NOT EXISTS registry_strategies_status_idx ON registry_strategies (status);

CREATE TABLE IF NOT EXISTS registry_worker_assignments (
    assignment_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    strategy_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    lease_expires_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS registry_worker_assignments_worker_idx
ON registry_worker_assignments (worker_id, status);

CREATE TABLE IF NOT EXISTS registry_allocations (
    allocation_id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES registry_strategies(strategy_id),
    actor_id TEXT NOT NULL,
    account_id TEXT NOT NULL REFERENCES registry_accounts(account_id),
    quote_asset TEXT NOT NULL DEFAULT '',
    base_asset TEXT NOT NULL DEFAULT '',
    allocated_quote DOUBLE PRECISION NOT NULL DEFAULT 0,
    allocated_base DOUBLE PRECISION NOT NULL DEFAULT 0,
    halt_on_breach BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS registry_allocations_account_idx ON registry_allocations (account_id);
CREATE INDEX IF NOT EXISTS registry_allocations_actor_idx ON registry_allocations (actor_id);
CREATE UNIQUE INDEX IF NOT EXISTS registry_allocations_strategy_id_unique
ON registry_allocations (strategy_id)
WHERE strategy_id IS NOT NULL AND strategy_id <> '';

CREATE TABLE IF NOT EXISTS registry_market_sources (
    source_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL REFERENCES registry_exchanges(exchange),
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'rest',
    depth_limit INT NOT NULL DEFAULT 10,
    trades_limit INT NOT NULL DEFAULT 100,
    poll_interval_ms INT NOT NULL DEFAULT 1000,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS registry_market_sources_unique_pair
ON registry_market_sources (exchange, symbol);
CREATE INDEX IF NOT EXISTS registry_market_sources_exchange_idx ON registry_market_sources (exchange);

CREATE TABLE IF NOT EXISTS registry_market_feeds (
    feed_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES registry_market_sources(source_id),
    interval_ms INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS registry_market_feeds_unique_interval
ON registry_market_feeds (source_id, interval_ms);
CREATE INDEX IF NOT EXISTS registry_market_feeds_source_idx ON registry_market_feeds (source_id);

CREATE TABLE IF NOT EXISTS registry_platform_settings (
    settings_id TEXT PRIMARY KEY,
    bus JSONB NOT NULL DEFAULT '{}'::jsonb,
    persistence JSONB NOT NULL DEFAULT '{}'::jsonb,
    observability JSONB NOT NULL DEFAULT '{}'::jsonb,
    scheduler JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

DEFAULT_PLATFORM_SETTINGS_ID = "default"

REGISTRY_MIGRATIONS = """
ALTER TABLE registry_strategies ADD COLUMN IF NOT EXISTS wallet_id TEXT;
ALTER TABLE registry_strategies ADD COLUMN IF NOT EXISTS ledger_source TEXT NOT NULL DEFAULT 'virtual';
ALTER TABLE registry_allocations ALTER COLUMN strategy_id DROP NOT NULL;
ALTER TABLE registry_allocations DROP CONSTRAINT IF EXISTS registry_allocations_strategy_id_key;
CREATE UNIQUE INDEX IF NOT EXISTS registry_allocations_strategy_id_unique
ON registry_allocations (strategy_id)
WHERE strategy_id IS NOT NULL AND strategy_id <> '';
CREATE INDEX IF NOT EXISTS registry_strategies_wallet_idx ON registry_strategies (wallet_id);

CREATE TABLE IF NOT EXISTS registry_market_sources (
    source_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL REFERENCES registry_exchanges(exchange),
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'rest',
    depth_limit INT NOT NULL DEFAULT 10,
    trades_limit INT NOT NULL DEFAULT 100,
    poll_interval_ms INT NOT NULL DEFAULT 1000,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS registry_market_sources_unique_pair
ON registry_market_sources (exchange, symbol);

DO $migrate_market_sources$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'registry_market_feeds'
      AND column_name = 'exchange'
  ) THEN
    INSERT INTO registry_market_sources (
      source_id, exchange, symbol, mode, depth_limit, trades_limit,
      poll_interval_ms, status, created_at, updated_at
    )
    SELECT DISTINCT ON (lower(exchange), symbol)
      lower(exchange) || '-' || symbol,
      lower(exchange),
      symbol,
      mode,
      depth_limit,
      trades_limit,
      poll_interval_ms,
      status,
      created_at,
      updated_at
    FROM registry_market_feeds
    ORDER BY lower(exchange), symbol, updated_at DESC
    ON CONFLICT (source_id) DO NOTHING;

    ALTER TABLE registry_market_feeds ADD COLUMN IF NOT EXISTS source_id TEXT;
    UPDATE registry_market_feeds AS f
    SET source_id = s.source_id
    FROM registry_market_sources AS s
    WHERE f.source_id IS NULL
      AND lower(f.exchange) = s.exchange
      AND f.symbol = s.symbol;

    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS exchange;
    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS symbol;
    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS mode;
    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS depth_limit;
    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS trades_limit;
    ALTER TABLE registry_market_feeds DROP COLUMN IF EXISTS poll_interval_ms;

    DROP INDEX IF EXISTS registry_market_feeds_exchange_idx;
    DROP INDEX IF EXISTS registry_market_feeds_unique_feed;
    CREATE UNIQUE INDEX IF NOT EXISTS registry_market_feeds_unique_interval
    ON registry_market_feeds (source_id, interval_ms);
    CREATE INDEX IF NOT EXISTS registry_market_feeds_source_idx
    ON registry_market_feeds (source_id);

    IF NOT EXISTS (
      SELECT 1
      FROM information_schema.table_constraints
      WHERE table_schema = current_schema()
        AND table_name = 'registry_market_feeds'
        AND constraint_name = 'registry_market_feeds_source_id_fkey'
    ) THEN
      ALTER TABLE registry_market_feeds
      ADD CONSTRAINT registry_market_feeds_source_id_fkey
      FOREIGN KEY (source_id) REFERENCES registry_market_sources(source_id);
    END IF;
  END IF;
END
$migrate_market_sources$;
"""
