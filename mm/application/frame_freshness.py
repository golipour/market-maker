"""Market-frame freshness checks for strategy pipeline gating."""

from __future__ import annotations

from mm.core.models.strategy import StrategyInputFrame


def local_market_age_ms(input_frame: StrategyInputFrame, *, now_ms: int) -> int:
    return max(0, int(now_ms) - int(input_frame.local.t_end_ms))


def global_market_age_ms(input_frame: StrategyInputFrame, *, now_ms: int) -> int | None:
    if input_frame.global_ is None:
        return None
    return max(0, int(now_ms) - int(input_frame.global_.t_end_ms))


def stale_market_frame_reason(
    input_frame: StrategyInputFrame,
    *,
    now_ms: int,
    max_local_market_age_ms: int,
    max_global_market_age_ms: int,
) -> str | None:
    """Return a skip reason when the aligned market frame is too old to process."""

    local_age = local_market_age_ms(input_frame, now_ms=now_ms)
    if max_local_market_age_ms > 0 and local_age > max_local_market_age_ms:
        return "stale_local_market_frame"
    if max_global_market_age_ms > 0 and input_frame.global_ is not None:
        global_age = global_market_age_ms(input_frame, now_ms=now_ms)
        if global_age is not None and global_age > max_global_market_age_ms:
            return "stale_global_market_frame"
    return None
