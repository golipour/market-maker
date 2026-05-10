from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.execution import Fill, Order
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.provider_adapter import AsyncRateLimiter, now_ms


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
        api_token: str | None,
        poll_interval_ms: int = 5000,
    ):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.api_token = api_token
        self.poll_interval_ms = max(500, int(poll_interval_ms))
        self._rate_limiter = AsyncRateLimiter(self.poll_interval_ms)
        self._connected = False
        self._last_balances: dict[str, Balance] = {}
        self._last_open_orders: dict[str, OpenOrder] = {}
        self._error_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()

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
        balances = self.normalize_balances(raw_balances, timestamp_ms=timestamp_ms)
        open_orders = self.normalize_open_orders(raw_orders, timestamp_ms=timestamp_ms)
        return PrivateStateSnapshot(
            exchange=self.exchange,
            timestamp_ms=timestamp_ms,
            balances=balances,
            open_orders=open_orders,
        )

    async def balance_stream(self) -> AsyncIterator[Balance]:
        while self._connected:
            try:
                snapshot = await self.snapshot()
                for asset, balance in snapshot.balances.items():
                    if self._last_balances.get(asset) != balance:
                        yield balance
                self._last_balances = dict(snapshot.balances)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("balance_stream", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def open_order_stream(self) -> AsyncIterator[OpenOrder]:
        while self._connected:
            try:
                snapshot = await self.snapshot()
                seen = set(snapshot.open_orders)
                for order_id, order in snapshot.open_orders.items():
                    if self._last_open_orders.get(order_id) != order:
                        yield order
                for missing_id, missing in self._last_open_orders.items():
                    if missing_id not in seen:
                        yield OpenOrder(
                            exchange_order_id=missing.exchange_order_id,
                            actor_id=missing.actor_id,
                            order=missing.order,
                            remaining_qty=0.0,
                            filled_qty=missing.filled_qty,
                            status="closed",
                            timestamp_ms=snapshot.timestamp_ms,
                        )
                self._last_open_orders = dict(snapshot.open_orders)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("open_order_stream", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        async for _ in self._empty_stream():
            yield _

    async def fill_stream(self) -> AsyncIterator[Fill]:
        async for _ in self._empty_stream():
            yield _

    async def error_stream(self) -> AsyncIterator[dict[str, str]]:
        while self._connected:
            yield await self._error_queue.get()

    async def fetch_balances(self) -> dict:
        raise NotImplementedError

    async def fetch_open_orders(self) -> list[dict]:
        raise NotImplementedError

    def normalize_balances(self, raw: dict, *, timestamp_ms: int) -> dict[str, Balance]:
        balances: dict[str, Balance] = {}
        if not isinstance(raw, dict):
            return balances
        for asset, bucket in raw.items():
            if not isinstance(bucket, dict):
                continue
            asset_key = str(asset or "").upper().strip()
            if not asset_key:
                continue
            free = _safe_float(bucket.get("free", 0.0))
            locked = _safe_float(bucket.get("locked", 0.0))
            total = _safe_float(bucket.get("total", free + locked))
            balances[asset_key] = Balance(
                exchange=self.exchange,
                asset=asset_key,
                free=free,
                locked=max(0.0, locked),
                timestamp_ms=timestamp_ms,
            )
            _ = total
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
