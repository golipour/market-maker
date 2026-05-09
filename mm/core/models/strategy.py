from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot

HedgeStatus = Literal["disabled", "ready", "missing", "stale"]


@dataclass(frozen=True, slots=True)
class StrategyInputFrame:
    """Aligned strategy input for one primary decision window.

    ``hedge`` is intentionally optional. When no hedge exchange is configured,
    or when the configured hedge stream is missing/stale for this primary
    window, strategies receive ``None`` plus an explicit ``hedge_status``.
    """

    primary: MarketEvent
    hedge: MarketEvent | None = None
    primary_private: PrivateStateSnapshot | None = None
    hedge_private: PrivateStateSnapshot | None = None
    hedge_status: HedgeStatus = "disabled"
    hedge_lag_ms: int | None = None
    max_hedge_lag_ms: int = 0
    created_at_ms: int = 0

    @property
    def has_hedge(self) -> bool:
        return self.hedge is not None and self.hedge_status == "ready"

    @property
    def t_end_ms(self) -> int:
        return self.primary.t_end_ms
