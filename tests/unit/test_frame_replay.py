from __future__ import annotations

import json

from mm.application.frame_simulator import FrameSimulator
from mm.core.events import envelope_event
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.payload_schema import (
    JOINT_LEDGER_SNAPSHOT_SCHEMA,
    JOINT_LEDGER_SNAPSHOT_VERSION,
    JOINT_TARGET_BOOK_SCHEMA,
    JOINT_TARGET_BOOK_VERSION,
    market_event_event,
    execution_lifecycle_event,
    joint_ledger_snapshot_event,
    stamped,
)
from mm.persistence.frame_replay import (
    compare_frame_simulation,
    load_persisted_frame_inputs,
    replay_and_compare_persisted_frame,
)
from mm.persistence.replay import lineage_chains_from_envelopes, read_envelopes
from mm.risk.factory import build_risk_engine
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    OrderbookGridConfig,
    RiskRuntimeConfig,
    VenueExecutionConfig,
)
from tests.unit.conftest import joint_frame_for_input


def _market_event() -> MarketEvent:
    book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1_000,
        t_start_ms=1_000,
        t_end_ms=2_000,
        orderbook=book,
        trades=(),
        buy_volume=1.0,
        sell_volume=2.0,
        vwap=100.5,
        imbalance=0.0,
        microprice=100.5,
    )


def _simulator() -> FrameSimulator:
    binding = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
    ).binding
    controller = ControllerRuntimeConfig(
        enabled=True,
        actor_id="test-runner",
        dry_run=False,
        execution_after_steps=0,
        decision_timeout_ms=0,
        risk=RiskRuntimeConfig(enabled=True),
        optimizer={
            "n_samples": 4,
            "horizon": 1,
            "adam_epochs": 2,
            "fixed_mc_samples": True,
            "local": {"cumulative_qty_floor_at_spread_max": 0.1, "cumulative_qty_cap_at_spread_max": 5.0},
            "global": {"cumulative_qty_floor_at_spread_max": 0.1, "cumulative_qty_cap_at_spread_max": 5.0},
        },
        orderbook_grid=OrderbookGridConfig(spread_max=50.0, spread_step=10.0),
        max_orders_per_side=3,
    )
    return FrameSimulator(
        controller=controller,
        binding=binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        execution=ExecutionRuntimeConfig(
            local=VenueExecutionConfig(enabled=True, planner="replace", observe_only=True),
        ),
        risk_engine=build_risk_engine(controller.risk, binding=binding),
    )


def test_compare_frame_simulation_matches_persisted_outputs_from_offline_run() -> None:
    market = _market_event()
    private = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2_000,
        balances={
            "BTC": Balance("nobitex", "BTC", 1.0, 0.0, 1.0, 2_000),
            "IRT": Balance("nobitex", "IRT", 1_000_000.0, 0.0, 1_000_000.0, 2_000),
        },
        open_orders={},
    )
    frame = StrategyInputFrame(local=market, local_private=private, created_at_ms=2_000)
    joint = joint_frame_for_input(frame)
    simulator = _simulator()
    simulated = simulator.simulate(joint, frame_step=1)

    rows = _persisted_rows_from_simulation(simulated, joint, market)
    chain = lineage_chains_from_envelopes(read_envelopes_from_rows(rows))[simulated.frame_id]

    report = compare_frame_simulation(simulated, chain)

    assert report.ok, [v.code for v in report.fidelity + report.simulation]


def test_replay_and_compare_persisted_frame_from_disk(tmp_path) -> None:
    market = _market_event()
    private = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2_000,
        balances={
            "BTC": Balance("nobitex", "BTC", 1.0, 0.0, 1.0, 2_000),
            "IRT": Balance("nobitex", "IRT", 1_000_000.0, 0.0, 1_000_000.0, 2_000),
        },
        open_orders={},
    )
    frame = StrategyInputFrame(local=market, local_private=private, created_at_ms=2_000)
    joint = joint_frame_for_input(frame)
    simulator = _simulator()
    simulated = simulator.simulate(joint, frame_step=1)
    rows = _persisted_rows_from_simulation(simulated, joint, market)
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_persisted_frame_inputs(path, simulated.frame_id)
    assert loaded.joint_frame.ledger is not None
    report = replay_and_compare_persisted_frame(
        path,
        simulated.frame_id,
        simulator=simulator,
        execution_only=True,
    )

    assert report.ok, [v.code for v in report.fidelity + report.simulation]


def _persisted_rows_from_simulation(simulated, joint, market) -> list[dict]:
    rows: list[dict] = []
    rows.append(
        _row(
            f"market:events:{market.exchange}:{market.symbol}:{market.interval_ms}",
            market_event_event(market),
        )
    )
    assert joint.ledger is not None
    rows.append(
        _row(
            "ledger:snapshot",
            joint_ledger_snapshot_event(joint.ledger),
        )
    )
    target_book = simulated.approved_target_book or simulated.proposed_target_book
    rows.append(
        _row(
            f"strategy:target_books:{simulated.decision.actor_id}",
            {"type": "JointTargetBook", **target_book.as_payload()},
        )
    )
    for plan_id, plan in simulated.execution_plans.items():
        if plan.diagnostics.get("planner") == "pending_confirmation_hold":
            continue
        if plan.place:
            rows.append(
                _row(
                    f"execution:orders:{plan.exchange}",
                    execution_lifecycle_event(
                        "PlaceOrdersRequested",
                        exchange=plan.exchange,
                        orders=[
                            {
                                "side": command.order.side,
                                "price": command.order.price,
                                "qty": command.order.qty,
                            }
                            for command in plan.place
                        ],
                        policy=plan.policy,
                        timestamp_ms=2_100,
                        frame_id=simulated.frame_id,
                        decision_id=simulated.decision_id,
                        target_book_id=simulated.target_book_id,
                        execution_plan_id=plan_id,
                    ),
                )
            )
        if plan.cancel:
            rows.append(
                _row(
                    f"execution:orders:{plan.exchange}",
                    execution_lifecycle_event(
                        "CancelOrdersRequested",
                        exchange=plan.exchange,
                        order_ids=[command.exchange_order_id for command in plan.cancel],
                        policy=plan.policy,
                        timestamp_ms=2_100,
                        frame_id=simulated.frame_id,
                        decision_id=simulated.decision_id,
                        target_book_id=simulated.target_book_id,
                        execution_plan_id=plan_id,
                    ),
                )
            )
    return rows


def _row(topic: str, payload: dict) -> dict:
    if payload.get("type") == "JointTargetBook":
        payload = stamped(JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION, **payload)
    envelope = envelope_event(topic, payload)
    return {"envelope_json": envelope.as_dict()}


def read_envelopes_from_rows(rows: list[dict]):
    from mm.core.events import decode_envelope

    return [decode_envelope(row["envelope_json"]) for row in rows]
