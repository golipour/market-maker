"""Map exchange order ids and client order ids back to strategy actor ids."""

from __future__ import annotations

from dataclasses import dataclass, field

from mm.core.models.execution import Fill


def _norm_exchange(exchange: str) -> str:
    return exchange.lower().strip()


def _norm_actor(actor_id: str) -> str:
    return actor_id.strip()


@dataclass(slots=True)
class OrderAttributionRegistry:
    """In-memory placement registry used to stamp ``actor_id`` on exchange fills."""

    _by_exchange_order_id: dict[str, str] = field(default_factory=dict)
    _by_client_order_id: dict[str, str] = field(default_factory=dict)

    def register(
        self,
        exchange: str,
        exchange_order_id: str,
        actor_id: str,
        *,
        client_order_id: str | None = None,
        wire_client_order_id: str | None = None,
    ) -> None:
        ex = _norm_exchange(exchange)
        actor = _norm_actor(actor_id)
        if not ex or not actor:
            return
        order_id = str(exchange_order_id or "").strip()
        if order_id:
            self._by_exchange_order_id[f"{ex}:{order_id}"] = actor
        for key in (client_order_id, wire_client_order_id):
            text = str(key or "").strip()
            if text:
                self._by_client_order_id[f"{ex}:{text}"] = actor

    def resolve_actor_id(
        self,
        exchange: str,
        *,
        exchange_order_id: str = "",
        client_order_id: str = "",
    ) -> str:
        ex = _norm_exchange(exchange)
        order_id = str(exchange_order_id or "").strip()
        if order_id:
            actor = self._by_exchange_order_id.get(f"{ex}:{order_id}")
            if actor:
                return actor
        client_id = str(client_order_id or "").strip()
        if client_id:
            actor = self._by_client_order_id.get(f"{ex}:{client_id}")
            if actor:
                return actor
        return ""

    def enrich_fill(self, fill: Fill, *, client_order_id: str = "") -> Fill:
        actor = (fill.actor_id or "").strip()
        if actor:
            return fill
        resolved = self.resolve_actor_id(
            fill.exchange,
            exchange_order_id=fill.exchange_order_id,
            client_order_id=client_order_id,
        )
        if not resolved:
            return fill
        return Fill(
            exchange_order_id=fill.exchange_order_id,
            actor_id=resolved,
            exchange=fill.exchange,
            symbol=fill.symbol,
            side=fill.side,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            timestamp_ms=fill.timestamp_ms,
        )


def nobitex_wire_client_order_id(client_order_id: str | None) -> str | None:
    from mm.exchanges.nobitex.private_rest import nobitex_wire_client_order_id as wire

    return wire(client_order_id)
