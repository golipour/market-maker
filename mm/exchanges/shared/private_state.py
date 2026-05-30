from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from mm.core.asset_aliases import canonical_asset_code
from mm.core.wallet import make_balance
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.instrument import split_symbol
from mm.core.models.execution import Fill, Order
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.provider_adapter import AsyncRateLimiter, now_ms

# Drop dust and API rows where both legs are zero so snapshots and strategy frames stay small.
_BALANCE_TOTAL_EPS = 1e-10


class RestPollingPrivateStateAdapter(AbstractPrivateStateAdapter):
    """REST-polling private state adapter.

    This is intentionally private-state only: it never places orders and never
    consumes public market data.
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.api_token = api_token
        self.poll_interval_ms = max(500, int(poll_interval_ms))
        self._rate_limiter = AsyncRateLimiter(self.poll_interval_ms)
        self._connected = False
        self._error_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self._seen_fill_qty_by_order: dict[str, float] = {}
        self._fill_history_seeded = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def snapshot(self) -> PrivateStateSnapshot:
        await self._rate_limiter.wait()
        timestamp_ms = now_ms()
        raw_balances, raw_orders = await asyncio.gather(
            self.fetch_balances(),
            self.fetch_open_orders(),
        )
        if not isinstance(raw_balances, dict) or not raw_balances:
            raise RuntimeError(f"{self.exchange} private balance fetch returned no balances")
        if not isinstance(raw_orders, list):
            raise RuntimeError(f"{self.exchange} private open-order fetch returned invalid payload")
        balances = self.normalize_balances(raw_balances, timestamp_ms=timestamp_ms)
        open_orders = self.normalize_open_orders(raw_orders, timestamp_ms=timestamp_ms)
        return PrivateStateSnapshot(
            exchange=self.exchange,
            timestamp_ms=timestamp_ms,
            balances=balances,
            open_orders=open_orders,
        )

    async def balance_stream(self) -> AsyncIterator[Balance]:
        """Not used for REST polling: ``PrivateStateReconciler`` applies full snapshots on a single timer."""
        if False:  # pragma: no cover
            yield make_balance(self.exchange, "_", 0.0, 0.0, 0)

    async def open_order_stream(self) -> AsyncIterator[OpenOrder]:
        """Not used for REST polling: ``PrivateStateReconciler`` applies full snapshots on a single timer."""
        if False:  # pragma: no cover
            yield OpenOrder(
                exchange_order_id="_",
                actor_id=None,
                order=Order(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    side="buy",
                    price=1.0,
                    qty=1.0,
                ),
                remaining_qty=1.0,
                filled_qty=0.0,
                status="open",
                timestamp_ms=0,
            )

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        async for _ in self._empty_stream():
            yield _

    async def fill_stream(self) -> AsyncIterator[Fill]:
        while self._connected:
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            try:
                timestamp_ms = now_ms()
                rows = await self.fetch_fills()
                fills = self.normalize_fills(rows, timestamp_ms=timestamp_ms)
                if not self._fill_history_seeded:
                    self._fill_history_seeded = True
                    continue
                for fill in fills:
                    yield fill
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("fill_poll", exc)

    async def error_stream(self) -> AsyncIterator[dict[str, str]]:
        while self._connected:
            yield await self._error_queue.get()

    async def fetch_balances(self) -> dict:
        raise NotImplementedError

    async def fetch_open_orders(self) -> list[dict]:
        raise NotImplementedError

    async def fetch_fills(self) -> list[dict]:
        return []

    def normalize_balances(self, raw: dict, *, timestamp_ms: int) -> dict[str, Balance]:
        balances: dict[str, Balance] = {}
        if not isinstance(raw, dict):
            return balances
        economy = _economy_canonical_assets(self.symbol)
        for asset, bucket in raw.items():
            if not isinstance(bucket, dict):
                continue
            asset_key = canonical_asset_code(asset)
            if not asset_key:
                continue
            free = _safe_float(bucket.get("free", 0.0))
            locked = _safe_float(bucket.get("locked", 0.0))
            total = _safe_float(bucket.get("total", free + locked))
            locked = max(0.0, locked)
            if max(abs(free), abs(locked), abs(total)) <= _BALANCE_TOTAL_EPS:
                if asset_key not in economy:
                    continue
            balances[asset_key] = make_balance(
                self.exchange,
                asset_key,
                free,
                locked,
                timestamp_ms,
                total=total,
            )
        _inject_missing_economy_balances(self.exchange, balances, economy, timestamp_ms)
        return balances

    def normalize_open_orders(self, rows: list[dict], *, timestamp_ms: int) -> dict[str, OpenOrder]:
        orders: dict[str, OpenOrder] = {}
        if not isinstance(rows, list):
            return orders
        for row in rows:
            order = self.normalize_open_order(row, timestamp_ms=timestamp_ms)
            if order is not None:
                orders[order.exchange_order_id] = order
        return orders

    def normalize_open_order(self, row: dict, *, timestamp_ms: int) -> OpenOrder | None:
        if not isinstance(row, dict):
            return None
        order_id = _first_text(row, "clientOrderId", "client_order_id", "id", "orderId", "order_id")
        if not order_id:
            return None
        side = _normalize_side(_first_text(row, "side", "type"))
        if side not in {"buy", "sell"}:
            return None
        price = _safe_float(_first_value(row, "price", "averagePrice", "avgFilledPrice"))
        qty = _safe_float(_first_value(row, "amount", "quantity", "qty", "origQty"))
        filled = _safe_float(_first_value(row, "filledAmount", "filled", "executedQty", "tradeAmount"))
        remaining = _safe_float(_first_value(row, "remainingAmount", "remaining", "leavesQty"))
        if remaining <= 0.0:
            remaining = max(0.0, qty - filled)
        if price <= 0.0 or remaining <= 0.0:
            return None
        status = str(row.get("status", "open") or "open").lower()
        actor_id = _first_text(row, "actor_id", "actorId")
        client_order_id = _first_text(row, "clientOrderId", "client_order_id")
        return OpenOrder(
            exchange_order_id=order_id,
            actor_id=actor_id or None,
            order=Order(
                exchange=self.exchange,
                symbol=self.symbol,
                side=side,
                price=price,
                qty=remaining,
                order_type=str(row.get("execution", row.get("orderType", "limit")) or "limit"),
                client_order_id=client_order_id or None,
            ),
            remaining_qty=remaining,
            filled_qty=max(0.0, filled),
            status=status,
            timestamp_ms=_safe_int(_first_value(row, "time", "createdAt", "updatedAt"), timestamp_ms),
        )

    def normalize_fills(self, rows: list[dict], *, timestamp_ms: int) -> tuple[Fill, ...]:
        fills: list[Fill] = []
        if not isinstance(rows, list):
            return ()
        for row in rows:
            fill = self.normalize_fill(row, timestamp_ms=timestamp_ms)
            if fill is not None:
                fills.append(fill)
        return tuple(fills)

    def normalize_fill(self, row: dict, *, timestamp_ms: int) -> Fill | None:
        if not isinstance(row, dict):
            return None
        order_id = _first_text(row, "exchange_order_id", "clientOrderId", "client_order_id", "id", "orderId", "order_id")
        if not order_id:
            return None
        side = _normalize_side(_first_text(row, "side", "type"))
        if side not in {"buy", "sell"}:
            return None
        cumulative_qty = _safe_float(
            _first_value(row, "filled_qty", "filledAmount", "matchedAmount", "executedQty", "tradeAmount", "quantity")
        )
        if cumulative_qty <= 0.0:
            return None
        seen_qty = self._seen_fill_qty_by_order.get(order_id, 0.0)
        delta_qty = max(0.0, cumulative_qty - seen_qty)
        self._seen_fill_qty_by_order[order_id] = max(seen_qty, cumulative_qty)
        if delta_qty <= 1e-12:
            return None
        price = _safe_float(_first_value(row, "price", "averagePrice", "avgFilledPrice", "executedPrice"))
        fee = _safe_float(_first_value(row, "fee", "commission"), 0.0)
        actor_id = _first_text(row, "actor_id", "actorId")
        return Fill(
            exchange_order_id=order_id,
            actor_id=actor_id or "",
            exchange=self.exchange,
            symbol=self.symbol,
            side=side,
            price=price,
            qty=delta_qty,
            fee=fee,
            timestamp_ms=_safe_int(_first_value(row, "timestamp_ms", "timestamp", "time", "createdAt", "updatedAt"), timestamp_ms),
        )

    async def _publish_error(self, source: str, exc: Exception) -> None:
        await self._error_queue.put(
            {
                "source": source,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )

    async def _empty_stream(self) -> AsyncIterator:
        while self._connected:
            await asyncio.sleep(3600.0)
            if False:
                yield None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return int(default)


def _first_value(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _first_text(row: dict, *keys: str) -> str:
    value = _first_value(row, *keys)
    return str(value).strip() if value is not None else ""


def _normalize_side(raw: str) -> str:
    side = str(raw or "").strip().lower()
    if side.startswith("buy"):
        return "buy"
    if side.startswith("sell"):
        return "sell"
    return side


def _economy_canonical_assets(symbol: str) -> frozenset[str]:
    """Base and quote for the traded pair (canonical), used for explicit-zero balance policy."""
    try:
        base, quote = split_symbol(symbol)
    except ValueError:
        return frozenset()
    b = canonical_asset_code(base)
    q = canonical_asset_code(quote)
    if not b or not q:
        return frozenset()
    return frozenset({b, q})


def _inject_missing_economy_balances(
    exchange: str,
    balances: dict[str, Balance],
    economy: frozenset[str],
    timestamp_ms: int,
) -> None:
    for asset in economy:
        if asset not in balances:
            balances[asset] = make_balance(
                exchange=exchange,
                asset=asset,
                free=0.0,
                locked=0.0,
                timestamp_ms=timestamp_ms,
                total=0.0,
            )
