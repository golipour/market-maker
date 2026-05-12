from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_table_ready: set[str] = set()


def _dsn() -> str:
    return str(os.environ.get("MM_STEP_ECONOMICS_DATABASE_URL", "")).strip()


def _ensure_table(cur: Any, dsn: str) -> None:
    if dsn in _table_ready:
        return
    cur.execute(
        """
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
        """
    )
    _table_ready.add(dsn)


def persist_step_economics_row(
    *,
    t_fill: datetime,
    t_end_ms: int,
    step_counter: int,
    exchange: str,
    symbol: str,
    d_w: float,
    spread_pnl: float,
    inventory_pnl: float,
    cross_pnl: float,
    fee_pnl: float,
    wealth_pnl: float,
) -> None:
    """
    Append one realized-PnL row (best-effort). Set ``MM_STEP_ECONOMICS_DATABASE_URL``, e.g.
    ``postgresql://user:pass@localhost:5432/postgres``. Requires ``psycopg2-binary``.
    """
    dsn = _dsn()
    if not dsn:
        return
    try:
        import psycopg2
    except ImportError:
        return
    if t_fill.tzinfo is None:
        t_fill = t_fill.replace(tzinfo=timezone.utc)
    with _lock:
        try:
            conn = psycopg2.connect(dsn)
            try:
                with conn.cursor() as cur:
                    _ensure_table(cur, dsn)
                    cur.execute(
                        """
                        INSERT INTO step_economics (
                            ts, t_end_ms, step_counter, exchange, symbol,
                            d_w, spread_pnl, inventory_pnl, cross_pnl, fee_pnl, wealth_pnl
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            t_fill,
                            int(t_end_ms),
                            int(step_counter),
                            str(exchange)[:64],
                            str(symbol)[:64],
                            float(d_w),
                            float(spread_pnl),
                            float(inventory_pnl),
                            float(cross_pnl),
                            float(fee_pnl),
                            float(wealth_pnl),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return


def keep_runner_pnl_series_memory() -> bool:
    """
    Keep ``dW_*`` / ``delta_*pnl`` lists on the runner for diagnostics and matplotlib.

    - Always on when ``MM_DIAGNOSTICS`` is set (``build_diagnostics_frame`` needs series).
    - Otherwise driven by ``MM_RUNNER_PNL_HISTORY`` (default ``1``); set ``0`` to disable when
      persisting to Postgres only.
    """
    if str(os.environ.get("MM_DIAGNOSTICS", "")).strip():
        return True
    v = str(os.environ.get("MM_RUNNER_PNL_HISTORY", "1")).strip().lower()
    return v not in ("0", "false", "no", "off")
