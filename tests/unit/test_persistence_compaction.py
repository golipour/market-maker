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


def test_compaction_summarizes_controller_diagnostics_action_pipeline_orders() -> None:
    base_order = {
        "exchange": "nobitex",
        "symbol": "BTC_IRT",
        "side": "buy",
        "price": 100.0,
        "qty": 0.5,
        "notional": 50.0,
        "order_type": "limit",
        "client_order_id": "c1",
        "strategy_trace": {
            "level_index": 0,
            "inventory_skew": 0.125,
            "forecast_delta": 1.25,
            "optimizer_objective": "spread_capture_minus_inventory_penalty",
            "candidate_source": "joint_mpc_target_book",
        },
        "metadata": {
            "local_reference_price": 100.0,
            "global_reference_price": 100.2,
            "normalization": "quote_notional",
        },
    }
    orders = [
        {
            **base_order,
            "client_order_id": f"c{i}",
            "price": 100.0 + i,
            "qty": 0.5 + (i * 0.01),
            "notional": (100.0 + i) * (0.5 + (i * 0.01)),
            "strategy_trace": {
                **base_order["strategy_trace"],
                "level_index": i,
            },
        }
        for i in range(10)
    ]
    rejection = {
        **orders[0],
        "reason": "rules_min_notional",
        "details": {
            "min_notional": 100.0,
            "available_quote": 42.0,
            "rule_path": "exchange_rules.nobitex.BTC_IRT.min_notional",
        },
    }
    payload = {
        "type": "ControllerDiagnostics",
        "actor_id": "mpc-controller",
        "exchange": "nobitex",
        "symbol": "BTC_IRT",
        "t_end_ms": 2000,
        "created_at_ms": 2000,
        "action_pipeline": {
            "target_book": {
                "order_count": 10,
                "buy_qty": 5.0,
                "sell_qty": 0.0,
                "buy_notional": 500.0,
                "sell_notional": 0.0,
                "orders": orders,
                "rejection_counts": {},
                "rejections": [],
            },
            "rules_valid": {
                "order_count": 6,
                "buy_qty": 3.0,
                "sell_qty": 0.0,
                "buy_notional": 300.0,
                "sell_notional": 0.0,
                "orders": orders[:6],
                "rejection_counts": {"rules_min_notional": 4},
                "rejections": [
                    {
                        **rejection,
                        "client_order_id": f"rejected-{i}",
                        "details": {
                            **rejection["details"],
                            "candidate_index": i,
                        },
                    }
                    for i in range(4)
                ],
            },
        },
    }
    envelope = envelope_event("strategy:diagnostics:mpc-controller", payload)

    compacted = compact_envelope_for_persistence(envelope, sink="postgres")

    target = compacted.payload["action_pipeline"]["target_book"]
    rules = compacted.payload["action_pipeline"]["rules_valid"]
    assert compacted.payload["action_pipeline_compacted"] is True
    assert target["order_count"] == 10
    assert target["buy_qty"] == 5.0
    assert target["orders"] == []
    assert target["orders_compacted_count"] == 10
    assert target["orders_compacted"] is True
    assert target["buy_count"] == 10
    assert target["orders_digest"]
    assert target["orders_sample"][0]["price"] == 100.0
    assert "client_order_id" not in target["orders_sample"][0]
    assert rules["rejection_counts"] == {"rules_min_notional": 4}
    assert rules["rejections"] == []
    assert rules["rejections_compacted_count"] == 4
    assert rules["rejections_sample"][0]["reason"] == "rules_min_notional"
    assert json_size_bytes(compacted.payload) < json_size_bytes(envelope.payload)
