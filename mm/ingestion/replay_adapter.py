"""Replay persisted envelopes onto the bus (same topic names as live ``SymbolEngine``)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.payload_schema import (
    market_event_event,
    market_event_from_payload,
    private_state_snapshot_from_payload,
)
from mm.persistence.replay import read_envelopes


def private_snapshot_from_payload(payload: Any):
    try:
        return private_state_snapshot_from_payload(payload)
    except (TypeError, ValueError):
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
                event = market_event_from_payload(envelope.payload)
                if event is not None:
                    await self.bus.publish(topic, market_event_event(event))
                    n_market += 1
            elif topic.startswith("private:state:") and topic.endswith(":snapshot"):
                snap = private_snapshot_from_payload(envelope.payload)
                if snap is not None:
                    await self.bus.publish(topic, snap.as_payload())
                    n_private += 1
            if self.pace_ms:
                await asyncio.sleep(self.pace_ms / 1000.0)
        return n_market, n_private
