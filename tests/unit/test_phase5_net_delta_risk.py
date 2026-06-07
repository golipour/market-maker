"""Phase 5 tests: net-delta target-book risk engine."""

from __future__ import annotations

import pytest

from mm.core.adapters.joint import joint_ledger_from_frame
from mm.core.models.execution import Action, Order, TradingState
from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.risk_limits import GlobalVenueRiskLimits, VenueRiskLimits
from mm.risk.engine import RiskLimits, RuntimeRiskEngine
from mm.risk.factory import build_risk_engine
from mm.risk.net_delta import NetDeltaRiskEngine, evaluate_venue_target_transition
from tests.unit.conftest import venues_payload
from mm.runtime.config import LiquiditySourcingRuntimeConfig, RiskRuntimeConfig, runtime_config_from_mapping


def _binding() -> SymbolBinding:
    local = InstrumentSpec(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    return SymbolBinding(local=local, global_=None)


def _global_binding() -> SymbolBinding:
    local = InstrumentSpec(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    global_ = InstrumentSpec(
        exchange="wallex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    return SymbolBinding(local=local, global_=global_)


def _market_event(*, t_end_ms: int = 1000) -> MarketEvent:
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=t_end_ms,
        orderbook=RawOrderbook("nobitex", "BTC_IRT", t_end_ms, (), (), True),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.0,
    )


def _global_market_event(*, t_end_ms: int = 1000) -> MarketEvent:
    return MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=t_end_ms,
        orderbook=RawOrderbook("wallex", "BTC_IRT", t_end_ms, (), (), True),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.0,
    )


def _private_snapshot(
    *,
    timestamp_ms: int = 1000,
    base_free: float = 10.0,
    quote_free: float = 200.0,
    quote_locked: float = 0.0,
    open_orders: dict[str, OpenOrder] | None = None,
) -> PrivateStateSnapshot:
    return PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=timestamp_ms,
        balances={
            "BTC": Balance("nobitex", "BTC", base_free, 0.0, base_free, timestamp_ms),
            "IRT": Balance("nobitex", "IRT", quote_free, quote_locked, quote_free + quote_locked, timestamp_ms),
        },
        open_orders=open_orders or {},
    )


def _global_private_snapshot(
    *,
    timestamp_ms: int = 1000,
    base_free: float = 10.0,
    quote_free: float = 200.0,
    quote_locked: float = 0.0,
    open_orders: dict[str, OpenOrder] | None = None,
) -> PrivateStateSnapshot:
    return PrivateStateSnapshot(
        exchange="wallex",
        timestamp_ms=timestamp_ms,
        balances={
            "BTC": Balance("wallex", "BTC", base_free, 0.0, base_free, timestamp_ms),
            "IRT": Balance("wallex", "IRT", quote_free, quote_locked, quote_free + quote_locked, timestamp_ms),
        },
        open_orders=open_orders or {},
    )


def _live_buy(
    order_id: str,
    *,
    actor_id: str | None = "actor-1",
    price: float = 100.0,
    qty: float = 8.0,
) -> OpenOrder:
    return OpenOrder(
        exchange_order_id=order_id,
        actor_id=actor_id,
        order=Order("nobitex", "BTC_IRT", "buy", price, qty),
        remaining_qty=qty,
        filled_qty=0.0,
        status="open",
        timestamp_ms=1000,
    )


def _venue_ledger_from_private(snapshot: PrivateStateSnapshot):
    ledger = joint_ledger_from_frame(
        StrategyInputFrame(local=_market_event(), local_private=snapshot),
        _binding(),
        created_at_ms=1000,
    )
    return ledger.local


def _target_book(*orders: TargetOrder) -> JointTargetBook:
    return JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook(
            exchange="nobitex",
            symbol="BTC_IRT",
            actor_id="actor-1",
            orders=orders,
        ),
    )


def test_build_risk_engine_defaults_to_net_delta() -> None:
    engine = build_risk_engine(RiskRuntimeConfig(enabled=True), binding=_binding())
    assert isinstance(engine, NetDeltaRiskEngine)


def test_build_risk_engine_net_delta_requires_binding() -> None:
    with pytest.raises(ValueError, match="binding is required"):
        build_risk_engine(RiskRuntimeConfig(enabled=True, engine="net_delta"))


def test_build_risk_engine_selects_net_delta() -> None:
    engine = build_risk_engine(
        RiskRuntimeConfig(enabled=True, engine="net_delta"),
        binding=_binding(),
    )
    assert isinstance(engine, NetDeltaRiskEngine)


def test_additive_replacement_approves_under_net_delta_rejected_by_order_level() -> None:
    """Order-level risk double-counts live open notional when evaluating replacements."""

    live = _live_buy("live-1", price=100.0, qty=4.0)
    snapshot = _private_snapshot(
        quote_free=600.0,
        quote_locked=400.0,
        open_orders={"live-1": live},
    )
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot)
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)

    target = _target_book(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 5.0))
    limits = RiskLimits(local=VenueRiskLimits(
            min_cash=0.0,
            max_open_notional=550.0,
            max_market_data_age_ms=10_000,
            max_private_state_age_ms=10_000,
        )
    )

    net_delta = NetDeltaRiskEngine(
        limits=limits,
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 1000,
    )
    net_result = net_delta.validate_target_book_with_result(target, ledger, frame)
    assert net_result.approved is not None

    runtime = RuntimeRiskEngine(limits=limits, clock_ms=lambda: 1000)
    runtime_result = runtime.validate_frame_with_result(
        [Action(actor_id="actor-1", orders=(Order("nobitex", "BTC_IRT", "buy", 100.0, 5.0),))],
        TradingState(
            actor_id="actor-1",
            inventory=10.0,
            cash=1000.0,
            open_orders={"live-1": live.order},
        ),
        frame,
    )
    assert runtime_result.approved == ()
    assert runtime_result.rejection_counts.get("max_open_notional", 0) >= 1


def test_venue_transition_ignores_unattributed_live_orders() -> None:
    live = _live_buy("exchange-live-1", actor_id=None, price=100.0, qty=4.0)
    snapshot = _private_snapshot(
        quote_free=100.0,
        quote_locked=400.0,
        open_orders={"exchange-live-1": live},
    )
    ledger = _venue_ledger_from_private(snapshot)
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 5.0),),
    )

    ok, reasons, exposure = evaluate_venue_target_transition(
        ledger=ledger,
        target=target,
        limits=VenueRiskLimits(min_cash=0.0),
    )

    assert not ok
    assert "insufficient_quote_for_bid_orders" in reasons
    assert exposure.actor_locked_quote == 0.0


def test_evaluate_venue_transition_rejects_insufficient_quote_for_bid_orders() -> None:
    snapshot = _private_snapshot(quote_free=100.0, quote_locked=0.0)
    ledger = _venue_ledger_from_private(snapshot)
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0),),
    )
    ok, reasons, exposure = evaluate_venue_target_transition(
        ledger=ledger,
        target=target,
        limits=VenueRiskLimits(min_cash=0.0),
    )
    assert not ok
    assert "insufficient_quote_for_bid_orders" in reasons
    assert exposure.target_locked_quote == 200.0


def test_net_delta_partial_approval_preserves_asks_when_bid_collateral_fails() -> None:
    snapshot = _private_snapshot(base_free=10.0, quote_free=100.0, quote_locked=0.0)
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot)
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)
    target = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook(
            exchange="nobitex",
            symbol="BTC_IRT",
            actor_id="actor-1",
            orders=(
                TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0),
                TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 1.0),
            ),
            policy="replace",
        ),
    )
    engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=50.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 1000,
    )

    result = engine.validate_target_book_with_result(target, ledger, frame)

    assert result.approved is not None
    assert [order.side for order in result.approved.local.orders] == ["sell"]
    assert result.target_book_gate_reasons == (
        "insufficient_quote_for_bid_orders",
        "min_cash",
    )


def test_global_partial_approval_preserves_asks_when_bid_collateral_fails() -> None:
    binding = _global_binding()
    frame = StrategyInputFrame(
        local=_market_event(),
        global_=_global_market_event(),
        local_private=_private_snapshot(),
        global_private=_global_private_snapshot(base_free=10.0, quote_free=0.0),
        global_status="ready",
        created_at_ms=1000,
    )
    ledger = joint_ledger_from_frame(frame, binding, created_at_ms=1000)
    target = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook("nobitex", "BTC_IRT", "actor-1", ()),
        global_=VenueTargetBook(
            "wallex",
            "BTC_IRT",
            "actor-1",
            (
                TargetOrder("wallex", "BTC_IRT", "buy", 100.0, 1.0),
                TargetOrder("wallex", "BTC_IRT", "sell", 101.0, 1.0),
            ),
            policy="replace",
        ),
        mode="liquidity_sourcing",
    )
    engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            ),
            global_=GlobalVenueRiskLimits(
                min_cash=0.0,
                min_inventory=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            ),
        ),
        binding=binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        clock_ms=lambda: 1000,
    )

    result = engine.validate_target_book_with_result(
        target,
        ledger,
        frame,
        live_execution=True,
        global_live_execution=True,
    )

    assert result.approved is not None
    assert result.approved.global_ is not None
    assert [order.side for order in result.approved.global_.orders] == ["sell"]
    assert result.target_book_gate_reasons == (
        "global_insufficient_quote_for_bid_orders",
        "global_min_cash",
    )


def test_replace_transition_checks_total_quote_even_when_locked() -> None:
    snapshot = _private_snapshot(quote_free=50.0, quote_locked=150.0)
    ledger = _venue_ledger_from_private(snapshot)
    reconcile_target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0),),
    )
    replace_target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0),),
        policy="replace",
    )

    reconcile_ok, reconcile_reasons, _ = evaluate_venue_target_transition(
        ledger=ledger,
        target=reconcile_target,
        limits=VenueRiskLimits(min_cash=0.0),
    )
    replace_ok, replace_reasons, replace_exposure = evaluate_venue_target_transition(
        ledger=ledger,
        target=replace_target,
        limits=VenueRiskLimits(min_cash=0.0),
    )

    assert not reconcile_ok
    assert "insufficient_quote_for_bid_orders" in reconcile_reasons
    assert replace_ok
    assert replace_reasons == []
    assert replace_exposure.target_locked_quote == pytest.approx(200.0)


def test_evaluate_venue_transition_rejects_insufficient_base_for_ask_orders() -> None:
    snapshot = _private_snapshot(base_free=1.0)
    ledger = _venue_ledger_from_private(snapshot)
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "sell", 100.0, 2.0),),
    )
    ok, reasons, _ = evaluate_venue_target_transition(
        ledger=ledger,
        target=target,
        limits=VenueRiskLimits(min_inventory=0.0),
    )
    assert not ok
    assert "insufficient_base_for_ask_orders" in reasons


def test_frame_gate_kill_switch_rejects_entire_book() -> None:
    snapshot = _private_snapshot()
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot)
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)
    target = _target_book(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0))

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(local=VenueRiskLimits(max_market_data_age_ms=10_000, max_private_state_age_ms=10_000)),
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 1000,
    )
    engine.set_kill_switch(True)
    result = engine.validate_target_book_with_result(target, ledger, frame)
    assert result.approved is None
    assert result.frame_gate_reasons == ("kill_switch",)


def test_single_venue_does_not_require_global_private() -> None:
    snapshot = _private_snapshot()
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot, global_status="missing")
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)
    target = _target_book(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0))

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(local=VenueRiskLimits(max_market_data_age_ms=10_000, max_private_state_age_ms=10_000)),
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=False, mode="single_venue"),
        clock_ms=lambda: 1000,
    )
    result = engine.validate_target_book_with_result(target, ledger, frame, live_execution=True)
    assert result.approved is not None
    assert "missing_global_private" not in result.frame_gate_reasons


def test_liquidity_sourcing_live_requires_global_private_for_global_orders() -> None:
    local = InstrumentSpec(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    global_ = InstrumentSpec(
        exchange="wallex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    binding = SymbolBinding(local=local, global_=global_)

    snapshot = _private_snapshot()
    global_event = MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=RawOrderbook("wallex", "BTC_IRT", 1000, (), (), True),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.0,
    )
    frame = StrategyInputFrame(local=_market_event(),
        global_=global_event,
        local_private=snapshot,
        global_private=None,
        global_status="ready",
    )
    ledger = joint_ledger_from_frame(frame, binding, created_at_ms=1000)
    target = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook("nobitex", "BTC_IRT", "actor-1", ()),
        global_=VenueTargetBook(
            "wallex",
            "BTC_IRT",
            "actor-1",
            (TargetOrder("wallex", "BTC_IRT", "buy", 100.0, 1.0),),
        ),
        mode="liquidity_sourcing",
    )

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(local=VenueRiskLimits(max_market_data_age_ms=10_000, max_private_state_age_ms=10_000),
        ),
        binding=binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        clock_ms=lambda: 1000,
    )
    result = engine.validate_target_book_with_result(target, ledger, frame, live_execution=True)
    assert result.approved is None
    assert "missing_global_private" in result.frame_gate_reasons


def test_liquidity_sourcing_observe_only_allows_missing_global_private_without_orders() -> None:
    local = InstrumentSpec(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    global_ = InstrumentSpec(
        exchange="wallex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    binding = SymbolBinding(local=local, global_=global_)

    snapshot = _private_snapshot()
    global_event = MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=RawOrderbook("wallex", "BTC_IRT", 1000, (), (), True),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.0,
    )
    frame = StrategyInputFrame(local=_market_event(),
        global_=global_event,
        local_private=snapshot,
        global_private=None,
        global_status="ready",
    )
    ledger = joint_ledger_from_frame(frame, binding, created_at_ms=1000)
    target = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook(
            "nobitex",
            "BTC_IRT",
            "actor-1",
            (TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0),),
        ),
        global_=VenueTargetBook("wallex", "BTC_IRT", "actor-1", ()),
        mode="liquidity_sourcing",
    )

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(local=VenueRiskLimits(max_market_data_age_ms=10_000, max_private_state_age_ms=10_000),
        ),
        binding=binding,
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing"),
        clock_ms=lambda: 1000,
    )
    result = engine.validate_target_book_with_result(target, ledger, frame, live_execution=False)
    assert result.approved is not None


def test_empty_partial_approval_includes_transition_diagnostics() -> None:
    snapshot = _private_snapshot(quote_free=50.0)
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot)
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)
    target = _target_book(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0))

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 1000,
    )
    result = engine.validate_target_book_with_result(target, ledger, frame)
    assert result.approved is not None
    assert result.approved.local.orders == ()
    assert "insufficient_quote_for_bid_orders" in result.target_book_gate_reasons
    assert "local" in result.transition_diagnostics
    assert result.transition_diagnostics["local"]["target_locked_quote"] == 200.0


def test_replace_rejection_approves_empty_cancel_only_book() -> None:
    snapshot = _private_snapshot(quote_free=50.0)
    frame = StrategyInputFrame(local=_market_event(), local_private=snapshot)
    ledger = joint_ledger_from_frame(frame, _binding(), created_at_ms=1000)
    target = JointTargetBook(
        actor_id="actor-1",
        local=VenueTargetBook(
            "nobitex",
            "BTC_IRT",
            "actor-1",
            (TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 2.0),),
            policy="replace",
        ),
    )

    engine = NetDeltaRiskEngine(
        limits=RiskLimits(
            local=VenueRiskLimits(
                min_cash=0.0,
                max_market_data_age_ms=10_000,
                max_private_state_age_ms=10_000,
            )
        ),
        binding=_binding(),
        liquidity_sourcing=LiquiditySourcingRuntimeConfig(),
        clock_ms=lambda: 1000,
    )
    result = engine.validate_target_book_with_result(target, ledger, frame)

    assert result.approved is not None
    assert result.approved.local.policy == "replace"
    assert result.approved.local.orders == ()
    assert "insufficient_quote_for_bid_orders" in result.target_book_gate_reasons
    assert result.rejection_counts["insufficient_quote_for_bid_orders"] == 1


def test_risk_config_parses_engine_field() -> None:
    payload = venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"})
    payload["strategy_presets"]["test_preset"]["controller"] = {
        "enabled": True,
        "risk": {"enabled": True, "engine": "net_delta"},
    }
    config = runtime_config_from_mapping(
        payload
    )
    assert config.controller is not None
    assert config.controller.risk.engine == "net_delta"
