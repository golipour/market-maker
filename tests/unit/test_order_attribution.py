from __future__ import annotations

from mm.core.models.execution import Fill
from mm.execution.order_attribution import (
    OrderAttributionRegistry,
    wire_client_order_id_for_exchange,
)
from mm.exchanges.nobitex.private_rest import nobitex_wire_client_order_id
from mm.exchanges.nobitex.private_state import NobitexRestPrivateStateAdapter


def test_order_attribution_registry_resolves_exchange_and_client_ids() -> None:
    registry = OrderAttributionRegistry()
    registry.register(
        "nobitex",
        "12345",
        "mpc-controller",
        client_order_id="mpc-controller:buy:0",
        wire_client_order_id=nobitex_wire_client_order_id("mpc-controller:buy:0"),
    )

    assert registry.resolve_actor_id("nobitex", exchange_order_id="12345") == "mpc-controller"
    assert (
        registry.resolve_actor_id(
            "nobitex",
            client_order_id=nobitex_wire_client_order_id("mpc-controller:buy:0") or "",
        )
        == "mpc-controller"
    )


def test_nobitex_fill_normalization_uses_attribution_registry() -> None:
    registry = OrderAttributionRegistry()
    wire_id = nobitex_wire_client_order_id("mpc-controller:buy:0")
    registry.register(
        "nobitex",
        "999",
        "mpc-controller",
        client_order_id="mpc-controller:buy:0",
        wire_client_order_id=wire_id,
    )
    adapter = NobitexRestPrivateStateAdapter("USDT_IRT", api_token="token")
    adapter.attribution_registry = registry

    fill = adapter.normalize_fill(
        {
            "id": "999",
            "type": "buy",
            "clientOrderId": wire_id,
            "matchedAmount": "1.0",
            "averagePrice": "158000",
            "fee": "0",
        },
        timestamp_ms=1000,
    )

    assert fill is not None
    assert fill.actor_id == "mpc-controller"


def test_wire_client_order_id_for_exchange_uses_registration_hook() -> None:
    assert (
        wire_client_order_id_for_exchange("nobitex", "mpc-controller:buy:0")
        == nobitex_wire_client_order_id("mpc-controller:buy:0")
    )
    assert wire_client_order_id_for_exchange("wallex", "actor:buy:0") == "actor:buy:0"


def test_enrich_fill_preserves_existing_actor_id() -> None:
    registry = OrderAttributionRegistry()
    fill = Fill(
        exchange_order_id="1",
        actor_id="already-set",
        exchange="nobitex",
        symbol="USDT_IRT",
        side="buy",
        price=1.0,
        qty=1.0,
        fee=0.0,
        timestamp_ms=1,
    )
    enriched = registry.enrich_fill(fill)
    assert enriched.actor_id == "already-set"
