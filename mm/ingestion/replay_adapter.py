"""Replay persisted envelopes onto the bus (same topic names as live ``SymbolEngine``)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade
from mm.core.payload_schema import private_state_snapshot_from_payload
from mm.persistence.replay import read_envelopes


def market_event_from_payload(payload: Any) -> MarketEvent | None:
    """Rebuild a ``MarketEvent`` from JSON-like data (e.g. persisted replay rows)."""

    if isinstance(payload, MarketEvent):
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        ob = payload["orderbook"]
        if isinstance(ob, RawOrderbook):
            orderbook = ob
        elif isinstance(ob, dict):
            orderbook = RawOrderbook(
                exchange=str(ob["exchange"]),
                symbol=str(ob["symbol"]),
                timestamp_ms=int(ob["timestamp_ms"]),
                bids=_price_levels(ob.get("bids", ())),
                asks=_price_levels(ob.get("asks", ())),
                is_snapshot=bool(ob.get("is_snapshot", True)),
            )
        else:
            return None
        trades_raw = payload.get("trades") or ()
        trades = tuple(_raw_trade(t) for t in trades_raw if _raw_trade(t) is not None)
        return MarketEvent(
            exchange=str(payload["exchange"]),
            symbol=str(payload["symbol"]),
            interval_ms=int(payload["interval_ms"]),
            t_start_ms=int(payload["t_start_ms"]),
            t_end_ms=int(payload["t_end_ms"]),
            orderbook=orderbook,
            trades=trades,
            buy_volume=float(payload.get("buy_volume", 0.0)),
            sell_volume=float(payload.get("sell_volume", 0.0)),
            vwap=float(payload.get("vwap", 0.0)),
            imbalance=float(payload.get("imbalance", 0.0)),
            microprice=float(payload.get("microprice", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def private_snapshot_from_payload(payload: Any):
    try:
        return private_state_snapshot_from_payload(payload)
    except (TypeError, ValueError):
        return None


def _price_levels(seq: Any) -> tuple[PriceLevel, ...]:
    out: list[PriceLevel] = []
    if not seq:
        return ()
    for x in seq:
        if isinstance(x, PriceLevel):
            out.append(x)
        elif isinstance(x, dict):
            out.append(PriceLevel(price=float(x["price"]), qty=float(x["qty"])))
    return tuple(out)


def _raw_trade(x: Any) -> RawTrade | None:
    if isinstance(x, RawTrade):
        return x
    if not isinstance(x, dict):
        return None
    try:
        return RawTrade(
            trade_id=x.get("trade_id"),
            timestamp_ms=int(x["timestamp_ms"]),
            price=float(x["price"]),
            qty=float(x["qty"]),
            side=str(x["side"]),
            exchange=str(x["exchange"]),
            symbol=str(x["symbol"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class ReplayMarketDataAdapter:
    """Republish ``market:events:*`` and optional ``private:state:*`` topics from disk."""

    def __init__(
        self,
        bus: AbstractEventBus,
        source_path: str | Path,
        *,
        topic_prefix: str | None = None,
        pace_ms: int = 0,
    ):
        self.bus = bus
        self.source_path = Path(source_path)
        self.topic_prefix = topic_prefix
        self.pace_ms = max(0, int(pace_ms))

    async def publish_all(self) -> tuple[int, int]:
        """Returns ``(market_events_published, private_snapshots_published)``."""

        n_market = 0
        n_private = 0
        for envelope in read_envelopes(self.source_path):
            if self.topic_prefix is not None and not str(envelope.topic).startswith(self.topic_prefix):
                continue
            topic = str(envelope.topic)
            if topic.startswith("market:events:"):
                ev = market_event_from_payload(envelope.payload)
                if ev is not None:
                    await self.bus.publish(topic, ev)
                    n_market += 1
            elif topic.startswith("private:state:") and topic.endswith(":snapshot"):
                snap = private_snapshot_from_payload(envelope.payload)
                if snap is not None:
                    await self.bus.publish(topic, snap.as_payload())
                    n_private += 1
            if self.pace_ms:
                await asyncio.sleep(self.pace_ms / 1000.0)
        return n_market, n_private
