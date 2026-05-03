-- Step economics table for Grafana (same schema as mm_core.app.step_economics_pg).
-- Applied automatically on first Postgres init via docker-entrypoint-initdb.d.
-- For an existing volume: docker exec -i postgres psql -U meysam -d postgres < docker/postgres/step_economics.sql

CREATE TABLE IF NOT EXISTS step_economics (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    t_end_ms BIGINT,
    step_counter INT NOT NULL,
    exchange TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL DEFAULT '',
    d_w DOUBLE PRECISION,
    spread_pnl DOUBLE PRECISION,
    inventory_pnl DOUBLE PRECISION,
    cross_pnl DOUBLE PRECISION,
    fee_pnl DOUBLE PRECISION,
    wealth_pnl DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS step_economics_ts_idx ON step_economics (ts DESC);
