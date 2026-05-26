from __future__ import annotations

import time

from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Fill, Order
from mm.core.models.market import RawTrade
from mm.core.models.private import OpenOrder
from mm.diagnostics.models import (
    ActionPipelineDiagnostics,
    ControllerDiagnostics,
    ExecutionGateDiagnostics,
    FrameDiagnostics,
    ModelScore,
    OptimizerDiagnostics,
    RiskDiagnostics,
)
from mm.execution.order_manager import ExecutionSnapshot
from mm.observability.registry import REGISTRY
from mm.observability.subscribers import RuntimeObservabilitySubscriber


def test_observability_subscriber_emits_controller_diagnostics_metrics() -> None:
    subscriber = RuntimeObservabilitySubscriber(bus=InMemoryBus(), topics=())
    payload = ControllerDiagnostics(
        actor_id="obs-test-controller",
        exchange="nobitex",
        symbol="BTC_IRT",
        t_end_ms=2_000,
        frame_step=3,
        created_at_ms=2_100,
        frame=FrameDiagnostics(
            frame_age_ms=100,
            sync_status={
                "local_market_ready": True,
                "global_market_ready": False,
                "ledger_fresh": True,
            },
        ),
        models={
            "price": ModelScore(
                model="price",
                nll=1.25,
                log_likelihood=-1.25,
                standardized_residual=0.5,
            ),
            "flow": ModelScore(model="flow", skip_reason="missing_previous_frame"),
        },
        optimizer=OptimizerDiagnostics(timing={"clean_ms": 2.0, "total_sec": 0.01}),
        action_pipeline=ActionPipelineDiagnostics(
            ladder={
                "order_count": 2,
                "buy_qty": 1.0,
                "sell_qty": 0.5,
                "buy_notional": 100.0,
                "sell_notional": 55.0,
                "rejection_counts": {"min_qty": 1},
            }
        ),
        risk=RiskDiagnostics(
            approved=False,
            rejection_counts={"max_inventory": 2},
            exposure_before={"inventory": 5.0},
            exposure_after={"inventory": 4.0},
        ),
        execution_gates=ExecutionGateDiagnostics(
            executed=False,
            blocked_reasons=("observe_only_config",),
        ),
    ).as_payload()

    subscriber.observe("strategy:diagnostics:obs-test-controller", payload)

    text = REGISTRY.render()
    assert 'mm_frame_age_ms{exchange="nobitex",symbol="BTC_IRT"} 100.0' in text
    assert "mm_controller_stage_latency_ms_bucket" in text
    assert 'stage="clean"' in text
    assert "mm_model_nll_bucket" in text
    assert 'model="price"' in text
    assert 'mm_model_score_skipped_total{actor_id="obs-test-controller",exchange="nobitex",model="flow",reason="missing_previous_frame",symbol="BTC_IRT"} 1.0' in text
    assert 'mm_action_stage_orders{actor_id="obs-test-controller",exchange="nobitex",side="all",stage="ladder",symbol="BTC_IRT"} 2.0' in text
    assert 'mm_risk_approved{actor_id="obs-test-controller",exchange="nobitex",symbol="BTC_IRT"} 0.0' in text
    assert 'reason="observe_only_config"' in text


def test_observability_subscriber_clears_sync_pending_on_completion_or_failure() -> None:
    subscriber = RuntimeObservabilitySubscriber(bus=InMemoryBus(), topics=())

    subscriber.observe(
        "execution:orders:nobitex",
        {"type": "ExecutionSyncRequested", "exchange": "nobitex", "timestamp_ms": 1000},
    )
    assert subscriber._sync_pending_ts_ms["nobitex"] == 1000

    subscriber.observe(
        "execution:errors",
        {
            "type": "ExecutionSyncFailed",
            "exchange": "nobitex",
            "error": "ExecutionSyncTimeout",
            "latency_ms": 50.0,
        },
    )
    assert "nobitex" not in subscriber._sync_pending_ts_ms
    assert "mm_order_sync_latency_ms_bucket" in REGISTRY.render()

    subscriber.observe(
        "execution:orders:nobitex",
        {"type": "ExecutionSyncRequested", "exchange": "nobitex", "timestamp_ms": 2000},
    )
    subscriber.observe(
        "execution:orders:nobitex",
        {
            "type": "ExecutionSyncCompleted",
            "exchange": "nobitex",
            "latency_ms": 42.0,
        },
    )
    assert "nobitex" not in subscriber._sync_pending_ts_ms


def test_observability_subscriber_emits_side_aware_execution_metrics() -> None:
    subscriber = RuntimeObservabilitySubscriber(bus=InMemoryBus(), topics=())

    subscriber.observe(
        "execution:orders:nobitex",
        {
            "type": "PlaceOrdersAccepted",
            "exchange": "nobitex",
            "order_ids": ["bid-1", "ask-1"],
            "orders": [
                {"exchange": "nobitex", "symbol": "USDT_IRT", "side": "buy", "qty": 0.5},
                {"exchange": "nobitex", "symbol": "USDT_IRT", "side": "sell", "qty": 0.25},
            ],
        },
    )
    subscriber.observe(
        "execution:orders:nobitex",
        {
            "type": "PlaceOrdersFailed",
            "exchange": "nobitex",
            "error": "HTTPError",
            "orders": [
                {"exchange": "nobitex", "symbol": "USDT_IRT", "side": "sell", "qty": 1.5},
            ],
        },
    )
    subscriber.observe(
        "private:state:nobitex:fills",
        Fill(
            exchange_order_id="bid-1",
            actor_id="actor-1",
            exchange="nobitex",
            symbol="USDT_IRT",
            side="buy",
            price=10.0,
            qty=0.1,
            fee=0.0,
            timestamp_ms=1234,
        ),
    )
    subscriber.observe(
        "execution:state:nobitex",
        ExecutionSnapshot(
            exchange="nobitex",
            timestamp_ms=1300,
            live_order_count=2,
            live_orders={
                "bid-1": OpenOrder(
                    exchange_order_id="bid-1",
                    actor_id="actor-1",
                    order=Order("nobitex", "USDT_IRT", "buy", 10.0, 0.5),
                    remaining_qty=0.4,
                    filled_qty=0.1,
                    status="open",
                    timestamp_ms=1300,
                ),
                "ask-1": OpenOrder(
                    exchange_order_id="ask-1",
                    actor_id="actor-1",
                    order=Order("nobitex", "USDT_IRT", "sell", 11.0, 0.25),
                    remaining_qty=0.25,
                    filled_qty=0.0,
                    status="open",
                    timestamp_ms=1300,
                ),
            },
        ),
    )

    text = REGISTRY.render()
    assert 'mm_execution_placed_orders_total{exchange="nobitex",side="bid",symbol="USDT_IRT"} 1.0' in text
    assert 'mm_execution_placed_qty_total{exchange="nobitex",side="ask",symbol="USDT_IRT"} 0.25' in text
    assert 'mm_execution_failed_orders_total{exchange="nobitex",reason="httperror",side="ask",symbol="USDT_IRT"} 1.0' in text
    assert 'mm_execution_filled_qty_total{exchange="nobitex",side="bid",symbol="USDT_IRT"} 0.1' in text
    assert 'mm_execution_live_orders_by_side{exchange="nobitex",side="bid",symbol="USDT_IRT"} 1.0' in text
    assert 'mm_execution_live_order_qty{exchange="nobitex",side="ask",symbol="USDT_IRT"} 0.25' in text


def test_observability_subscriber_emits_raw_market_feed_metrics() -> None:
    subscriber = RuntimeObservabilitySubscriber(bus=InMemoryBus(), topics=())
    ts_ms = int(time.time() * 1000) - 2_000
    first = RawTrade(
        trade_id="raw-metric-1",
        timestamp_ms=ts_ms,
        price=100.0,
        qty=1.0,
        side="buy",
        exchange="rawtest",
        symbol="RAW_TEST",
    )
    second = RawTrade(
        trade_id="raw-metric-2",
        timestamp_ms=ts_ms + 1_000,
        price=101.0,
        qty=1.0,
        side="sell",
        exchange="rawtest",
        symbol="RAW_TEST",
    )

    subscriber.observe("market:raw:rawtest:RAW_TEST:trades", first)
    subscriber.observe("market:raw:rawtest:RAW_TEST:trades", second)

    text = REGISTRY.render()
    assert 'mm_raw_market_events_total{exchange="rawtest",kind="trade",symbol="RAW_TEST"} 2.0' in text
    assert f'mm_raw_market_event_timestamp_ms{{exchange="rawtest",kind="trade",symbol="RAW_TEST"}} {float(second.timestamp_ms)}' in text
    assert 'mm_raw_market_ingest_lag_latest_ms{exchange="rawtest",kind="trade",symbol="RAW_TEST"}' in text
    assert 'mm_raw_market_event_gap_latest_ms{exchange="rawtest",kind="trade",symbol="RAW_TEST"} 1000.0' in text
    assert "mm_raw_market_ingest_lag_ms_bucket" in text
    assert "mm_raw_market_event_gap_ms_bucket" in text
