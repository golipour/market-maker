"""Durable held balances for virtual strategy wallets.

The allocation grant stays in the registry. This store only remembers held
base and quote so a restart does not put the wallet back to the original grant.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mm.portfolio.wallet import StrategyWallet


@dataclass(frozen=True, slots=True)
class WalletBalance:
    actor_id: str
    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    initial_base: float
    initial_quote: float
    held_base: float
    held_quote: float

    @classmethod
    def from_wallet(cls, wallet: StrategyWallet) -> WalletBalance:
        return cls(
            actor_id=wallet.actor_id,
            exchange=wallet.exchange,
            symbol=wallet.symbol,
            base_asset=wallet.base_asset,
            quote_asset=wallet.quote_asset,
            initial_base=wallet.initial_base,
            initial_quote=wallet.initial_quote,
            held_base=wallet.held_base,
            held_quote=wallet.held_quote,
        )


class WalletStore(Protocol):
    def load(self) -> dict[str, WalletBalance]: ...

    def save(self, rows: Sequence[WalletBalance]) -> None: ...


class MemoryWalletStore:
    """Process-local stand-in used when ``MM_DATABASE_URL`` is unset."""

    def __init__(self) -> None:
        self._rows: dict[str, WalletBalance] = {}

    def load(self) -> dict[str, WalletBalance]:
        return dict(self._rows)

    def save(self, rows: Sequence[WalletBalance]) -> None:
        for row in rows:
            self._rows[row.actor_id] = row


_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False

_DDL = """
CREATE TABLE IF NOT EXISTS mm_virtual_wallets (
    actor_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    base_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    initial_base DOUBLE PRECISION NOT NULL,
    initial_quote DOUBLE PRECISION NOT NULL,
    held_base DOUBLE PRECISION NOT NULL,
    held_quote DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_UPSERT_SQL = """
INSERT INTO mm_virtual_wallets (
    actor_id,
    exchange,
    symbol,
    base_asset,
    quote_asset,
    initial_base,
    initial_quote,
    held_base,
    held_quote,
    updated_at
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
)
ON CONFLICT (actor_id) DO UPDATE SET
    exchange = EXCLUDED.exchange,
    symbol = EXCLUDED.symbol,
    base_asset = EXCLUDED.base_asset,
    quote_asset = EXCLUDED.quote_asset,
    initial_base = EXCLUDED.initial_base,
    initial_quote = EXCLUDED.initial_quote,
    held_base = EXCLUDED.held_base,
    held_quote = EXCLUDED.held_quote,
    updated_at = now()
"""


class PostgresWalletStore:
    def __init__(self, database_url: str, *, connect_timeout_s: int = 5) -> None:
        self.database_url = database_url
        self.connect_timeout_s = connect_timeout_s

    def ensure_schema(self) -> None:
        global _SCHEMA_READY
        if _SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if _SCHEMA_READY:
                return
            conn = self._connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(_DDL)
                conn.commit()
            finally:
                conn.close()
            _SCHEMA_READY = True

    def load(self) -> dict[str, WalletBalance]:
        self.ensure_schema()
        conn = self._connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT actor_id, exchange, symbol, base_asset, quote_asset,
                           initial_base, initial_quote, held_base, held_quote
                    FROM mm_virtual_wallets
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        loaded: dict[str, WalletBalance] = {}
        for row in rows:
            balance = WalletBalance(
                actor_id=str(row[0]),
                exchange=str(row[1]),
                symbol=str(row[2]),
                base_asset=str(row[3]),
                quote_asset=str(row[4]),
                initial_base=float(row[5]),
                initial_quote=float(row[6]),
                held_base=float(row[7]),
                held_quote=float(row[8]),
            )
            loaded[balance.actor_id] = balance
        return loaded

    def save(self, rows: Sequence[WalletBalance]) -> None:
        if not rows:
            return
        self.ensure_schema()
        conn = self._connection()
        try:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        _UPSERT_SQL,
                        (
                            row.actor_id,
                            row.exchange,
                            row.symbol,
                            row.base_asset,
                            row.quote_asset,
                            row.initial_base,
                            row.initial_quote,
                            row.held_base,
                            row.held_quote,
                        ),
                    )
            conn.commit()
        finally:
            conn.close()

    def _connection(self):
        import psycopg2

        return psycopg2.connect(
            self.database_url,
            connect_timeout=self.connect_timeout_s,
        )


def build_wallet_store() -> WalletStore:
    """Postgres when ``MM_DATABASE_URL`` is set, otherwise an in-process store."""

    url = os.environ.get("MM_DATABASE_URL", "").strip()
    if url:
        return PostgresWalletStore(url)
    return MemoryWalletStore()
