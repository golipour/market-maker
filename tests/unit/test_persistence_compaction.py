from __future__ import annotations

from mm.core.events import envelope_event, json_size_bytes
from mm.core.models.execution import Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.persistence.compaction import compact_envelope_for_persistence
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig
from tests.unit.conftest import joint_frame_for_input


def _market_event() -> MarketEvent:
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=2000,
            bids=(PriceLevel(100.0, 1.0),),
            asks=(PriceLevel(101.0, 2.0),),
            is_snapshot=True,
        ),
        trades=(
            RawTrade("t1", 1100, 100.0, 0.1, "buy", "nobitex", "BTC_IRT"),
            RawTrade("t2", 1200, 101.0, 0.2, "sell", "nobitex", "BTC_IRT"),
        ),
        buy_volume=0.1,
        sell_volume=0.2,
        vwap=100.66,
        imbalance=-0.1,
        microprice=100.5,
    )


def _private_snapshot() -> PrivateStateSnapshot:
    order = Order(
        "nobitex",
        "BTC_IRT",
        "buy",
        99.0,
        0.2,
        client_order_id="live-1",
    )
    return PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2000,
        balances={
            "BTC": Balance("nobitex", "BTC", 1.0, 0.0, 1.0, 2000),
            "ZERO": Balance("nobitex", "ZERO", 0.0, 0.0, 0.0, 2000),
        },
        open_orders={
            "live-1": OpenOrder(
                exchange_order_id="live-1",
                actor_id="mpc-controller",
                order=order,
                remaining_qty=0.2,
                filled_qty=0.0,
                status="open",
                timestamp_ms=2000,
            )
        },
    )


def test_postgres_compaction_drops_market_trades_but_keeps_interval_features() -> None:
    envelope = envelope_event("market:events:nobitex:BTC_IRT:1000", _market_event())

    compacted = compact_envelope_for_persistence(envelope, sink="postgres")

    assert compacted.payload["trades"] == []
    assert compacted.payload["trades_compacted"] is True
    assert compacted.payload["trade_count"] == 2
    assert compacted.payload["buy_volume"] == 0.1
    assert compacted.payload["sell_volume"] == 0.2
    assert compacted.payload["vwap"] == 100.66
    assert compacted.payload["orderbook"]["bids"]
    assert json_size_bytes(compacted.payload) < json_size_bytes(envelope.payload)


def test_postgres_compaction_preserves_private_snapshot_balances_for_latest_state() -> None:
    envelope = envelope_event("private:state:nobitex:snapshot", _private_snapshot())

    compacted = compact_envelope_for_persistence(envelope, sink="postgres")

    assert sorted(compacted.payload["balances"]) == ["BTC", "ZERO"]
    assert "balances_compacted" not in compacted.payload


def test_parquet_compaction_removes_zero_balances_from_archival_snapshot() -> None:
    envelope = envelope_event("private:state:nobitex:snapshot", _private_snapshot())

    compacted = compact_envelope_for_persistence(envelope, sink="parquet")

    assert sorted(compacted.payload["balances"]) == ["BTC"]
    assert compacted.payload["balance_count"] == 2
    assert compacted.payload["nonzero_balance_count"] == 1
    assert compacted.payload["balances_compacted"] is True


def test_compaction_summarizes_joint_frame_ledger_with_replay_reference() -> None:
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
    )
    frame = StrategyInputFrame(
        local=_market_event(),
        local_private=_private_snapshot(),
        created_at_ms=2000,
    )
    joint = joint_frame_for_input(frame, binding=runtime.binding)
    envelope = envelope_event(runtime.joint_frame_topic, joint)

    compacted = compact_envelope_for_persistence(envelope, sink="postgres")

    assert compacted.payload["local_market"]["exchange"] == "nobitex"
    assert compacted.payload["sync_status"]["ledger_fresh"] is True
    assert compacted.payload["ledger_compacted"] is True
    assert compacted.payload["ledger_ref"]["topic"] == "ledger:snapshot"
    assert compacted.payload["ledger"]["local"]["balance_count"] == 2
    assert compacted.payload["ledger"]["local"]["live_order_count"] == 1
    assert "balances" not in compacted.payload["ledger"]["local"]
    assert "live_orders" not in compacted.payload["ledger"]["local"]
    assert json_size_bytes(compacted.payload) < json_size_bytes(envelope.payload)
