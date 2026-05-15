from __future__ import annotations

from mm.core.models.execution import Action, Order, TradingState
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.risk.engine import RiskLimits, RuntimeRiskEngine


def _event(t_end_ms: int = 1000) -> MarketEvent:
    return MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=t_end_ms,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=t_end_ms,
            bids=(),
            asks=(),
            is_snapshot=True,
        ),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=0.0,
    )


def test_runtime_risk_rejects_when_kill_switch_enabled() -> None:
    risk = RuntimeRiskEngine(clock_ms=lambda: 1000)
    risk.set_kill_switch(True)

    approved = risk.validate(
        [Action(actor_id="a1", orders_to_create=(Order("nobitex", "BTC_IRT", "buy", 1.0, 1.0),))],
        TradingState(actor_id="a1"),
        _event(),
    )

    assert approved == []


def test_runtime_risk_filters_inventory_breach() -> None:
    risk = RuntimeRiskEngine(RiskLimits(max_inventory=1.0), clock_ms=lambda: 1000)

    approved = risk.validate(
        [Action(actor_id="a1", orders_to_create=(Order("nobitex", "BTC_IRT", "buy", 1.0, 2.0),))],
        TradingState(actor_id="a1"),
        _event(),
    )

    assert approved == []


def test_runtime_risk_rejects_stale_data() -> None:
    risk = RuntimeRiskEngine(RiskLimits(max_data_age_ms=10), clock_ms=lambda: 2000)

    approved = risk.validate(
        [Action(actor_id="a1", orders_to_create=(Order("nobitex", "BTC_IRT", "buy", 1.0, 1.0),))],
        TradingState(actor_id="a1"),
        _event(1000),
    )

    assert approved == []
