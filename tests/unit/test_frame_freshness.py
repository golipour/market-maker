from __future__ import annotations

from mm.application.frame_freshness import stale_market_frame_reason
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig
from tests.unit.test_strategy_pipeline import market_event


def test_runtime_config_defaults_market_age_to_interval_ms() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        interval_ms=30_000,
    )
    assert config.resolved_max_local_market_age_ms() == 30_000
    assert config.resolved_max_global_market_age_ms() == 0


def test_stale_market_frame_reason_flags_backlog_frames() -> None:
    frame = market_event()
    reason = stale_market_frame_reason(
        frame,
        now_ms=frame.local.t_end_ms + 881_564,
        max_local_market_age_ms=30_000,
        max_global_market_age_ms=0,
    )
    assert reason == "stale_local_market_frame"


def test_fresh_market_frame_passes_within_interval() -> None:
    frame = market_event()
    reason = stale_market_frame_reason(
        frame,
        now_ms=frame.local.t_end_ms + 27,
        max_local_market_age_ms=30_000,
        max_global_market_age_ms=0,
    )
    assert reason is None
