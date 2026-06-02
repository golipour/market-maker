from __future__ import annotations

import asyncio
from typing import Any

from mm.core.asset_aliases import canonical_asset_code
from mm.core.models.execution import Fill
from mm.core.models.instrument import split_symbol
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.exchanges.wallex import private_rest


class WallexRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: str | None,
        poll_interval_ms: int = 5000,
    ):
        super().__init__(
            "wallex",
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)

    async def fetch_fills(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_filled_orders, self.symbol, self.api_token)

    def normalize_fill(self, row: dict, *, timestamp_ms: int) -> Fill | None:
        normalized = self._row_with_quote_fee(row)
        return super().normalize_fill(normalized, timestamp_ms=timestamp_ms)

    def _row_with_quote_fee(self, row: dict) -> dict:
        fills = row.get("fills") if isinstance(row, dict) else None
        if not isinstance(fills, list) or not fills:
            return row

        total_qty = 0.0
        total_notional = 0.0
        total_fee_quote = 0.0
        last_timestamp: Any = None
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            qty = _safe_float(fill.get("quantity"))
            price = _safe_float(fill.get("price"))
            notional = _safe_float(fill.get("sum"), price * qty)
            fee = _safe_float(fill.get("fee"))
            fee_asset = canonical_asset_code(fill.get("feeAsset"))
            total_qty += max(0.0, qty)
            total_notional += max(0.0, notional)
            total_fee_quote += self._fee_to_quote(fee, fee_asset=fee_asset, price=price)
            last_timestamp = fill.get("timestamp") or last_timestamp

        if total_qty <= 0.0:
            return row

        out = dict(row)
        out["executedQty"] = str(total_qty)
        fallback_price = _safe_float(row.get("executedPrice", row.get("price")))
        out["executedPrice"] = str(total_notional / total_qty if total_notional > 0.0 else fallback_price)
        out["fee"] = str(total_fee_quote)
        if last_timestamp is not None:
            out["timestamp"] = last_timestamp
        return out

    def _fee_to_quote(self, fee: float, *, fee_asset: str, price: float) -> float:
        if fee <= 0.0:
            return 0.0
        base_asset, quote_asset = split_symbol(self.symbol)
        if fee_asset == canonical_asset_code(quote_asset):
            return float(fee)
        if fee_asset == canonical_asset_code(base_asset):
            return float(fee) * max(0.0, float(price))
        return float(fee)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
