from __future__ import annotations

from mm.core.models.execution import Order
from mm.exchanges.wallex.order_adapter import WallexRestOrderCommandAdapter


def test_wallex_order_adapter_trims_float_payloads(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_place_limit_order(symbol, side, amount, price, api_token, client_order_id=None):
        _ = symbol, side, api_token, client_order_id
        seen["amount"] = amount
        seen["price"] = price
        return {"success": True, "result": {"clientOrderId": "abc"}}

    monkeypatch.setattr(
        "mm.exchanges.wallex.order_adapter.private_rest.place_limit_order",
        fake_place_limit_order,
    )

    adapter = WallexRestOrderCommandAdapter("USDT_IRT", api_token="token")
    adapter._place_one(
        Order(
            exchange="wallex",
            symbol="USDT_IRT",
            side="sell",
            price=173_650.00000000003,
            qty=0.29000000000000004,
        )
    )

    assert seen == {"amount": "0.29", "price": "173650"}
