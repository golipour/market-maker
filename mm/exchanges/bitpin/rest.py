from __future__ import annotations

from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.shared.provider_adapter import (
    RestClient,
    RestPollingMarketDataAdapter,
    parse_timestamp_ms,
    rows_to_levels,
)


class BitpinRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = "https://api.bitpin.ir"

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("bitpin", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)
        self._wire = build_instrument_wire("bitpin", symbol)
        self._market_id: int | None = None

    async def fetch_trades(self) -> list[RawTrade]:
        market_id = await self._resolve_market_id()
        payload = await self._client.get_json(f"/v1/mth/matches/{market_id}/")
        rows = _result_rows(payload)[: self.trades_limit]
        trades = [
            self._wire.scale_trade_from_api(self._trade_from_item(item))
            for item in rows
            if isinstance(item, dict)
        ]
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        market_id = await self._resolve_market_id()
        bids_payload = await self._client.get_json(f"/v2/mth/actives/{market_id}/", {"type": "buy"})
        asks_payload = await self._client.get_json(f"/v2/mth/actives/{market_id}/", {"type": "sell"})
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=max(
                _payload_timestamp_ms(bids_payload),
                _payload_timestamp_ms(asks_payload),
            ),
            bids=rows_to_levels(_order_rows(bids_payload), limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(_order_rows(asks_payload), limit=self.depth_limit),
            is_snapshot=True,
        )
        return self._wire.scale_orderbook_from_api(book)

    async def _resolve_market_id(self) -> int:
        if self._market_id is not None:
            return self._market_id
        market_symbol = self._wire.api_market_symbol
        path = "/v1/mkt/markets/"
        page = 1
        while True:
            payload = await self._client.get_json(path, {"page": page})
            for row in _result_rows(payload):
                if not isinstance(row, dict):
                    continue
                if _same_market_symbol(_market_code(row), market_symbol):
                    market_id = row.get("id")
                    if market_id is None:
                        break
                    self._market_id = int(market_id)
                    return self._market_id
            if not isinstance(payload, dict) or not payload.get("next"):
                break
            page += 1
        raise ValueError(f"bitpin market symbol not found: {self.symbol}")

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        side = str(item.get("type") or item.get("side") or "").lower()
        return RawTrade(
            trade_id=str(item.get("id") or item.get("match_id") or item.get("trade_id"))
            if (item.get("id") or item.get("match_id") or item.get("trade_id")) is not None
            else None,
            timestamp_ms=parse_timestamp_ms(item.get("created_at") or item.get("time") or item.get("timestamp")),
            price=_safe_float(item.get("price") or item.get("price_limit") or item.get("average_price")),
            qty=_safe_float(
                item.get("match_amount")
                or item.get("amount1")
                or item.get("amount")
                or item.get("quantity")
                or item.get("exchanged1")
            ),
            side="buy" if side == "buy" else "sell",
            exchange=self.exchange,
            symbol=self.symbol,
        )


def _market_code(row: dict[str, Any]) -> str:
    code = str(row.get("code") or row.get("symbol") or "").upper().strip()
    if code:
        return code.replace("-", "_")
    base = row.get("currency1") if isinstance(row.get("currency1"), dict) else {}
    quote = row.get("currency2") if isinstance(row.get("currency2"), dict) else {}
    b = str(base.get("code") or "").upper().strip()
    q = str(quote.get("code") or "").upper().strip()
    return f"{b}_{q}" if b and q else ""


def _same_market_symbol(left: str, right: str) -> bool:
    return left.replace("_", "").replace("-", "").upper() == right.replace("_", "").replace("-", "").upper()


def _result_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results")
    if isinstance(rows, list):
        return rows
    if isinstance(rows, dict):
        return [rows]
    for key in ("orders", "matches", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return [payload] if payload else []


def _order_rows(payload: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in _result_rows(payload):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append([str(item[0]), str(item[1])])
            continue
        if not isinstance(item, dict):
            continue
        price = item.get("price") or item.get("price_limit") or item.get("active_limit")
        qty = (
            item.get("remain_amount")
            or item.get("remain")
            or item.get("amount1")
            or item.get("amount")
            or item.get("quantity")
            or item.get("value")
        )
        if price is not None and qty is not None:
            rows.append([str(price), str(qty)])
    return rows


def _payload_timestamp_ms(payload: Any) -> int:
    if isinstance(payload, dict):
        for key in ("timestamp_ms", "timestamp", "time", "created_at", "updated_at"):
            if payload.get(key) is not None:
                return parse_timestamp_ms(payload.get(key))
    return parse_timestamp_ms(None)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
