from __future__ import annotations

from mm.core.models.execution import Action, Order, TradingState
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.controller.context import DecisionContext
from mm.risk.engine import RiskLimits, RuntimeRiskEngine
from mm.core.risk_limits import VenueRiskLimits
from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig, RiskRuntimeConfig, runtime_config_from_mapping


def _event(t_end_ms: int = 1000) -> MarketEvent:
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
        microprice=0.0,
    )


def _private(ts: int = 1000) -> PrivateStateSnapshot:
    return PrivateStateSnapshot(exchange="nobitex", timestamp_ms=ts, balances={})


def test_risk_config_migration_from_flat_keys() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"exchange": "nobitex", "symbol": "BTC_IRT"},
            "controller": {
                "enabled": True,
                "risk": {
                    "enabled": True,
                    "max_inventory": 500.0,
                    "max_data_age_ms": 30_000,
                    "min_inventory": 0.0,
                },
            },
        }
    )
    assert config.controller is not None
    assert config.controller.risk.primary.max_inventory == 500.0
    assert config.controller.risk.primary.max_market_data_age_ms == 30_000
    assert config.controller.risk.primary.min_inventory == 0.0


def test_spot_rejects_sell_exceeding_inventory() -> None:
    limits = RiskLimits(primary=VenueRiskLimits(min_inventory=0.0, max_inventory=100.0))
    risk = RuntimeRiskEngine(limits=limits, clock_ms=lambda: 1000)
    frame = StrategyInputFrame(
        primary=_event(),
        primary_private=_private(1000),
    )
    action = Action(
        actor_id="a",
        orders_to_create=(Order("nobitex", "BTC_IRT", "sell", 100.0, 5.0),),
    )
    result = risk.validate_frame_with_result(
        [action],
        TradingState(actor_id="a", inventory=2.0, cash=1000.0),
        frame,
    )
    assert result.approved == ()
    assert result.rejection_counts.get("insufficient_inventory", 0) >= 1


def test_rejects_buy_when_cash_too_low() -> None:
    limits = RiskLimits(primary=VenueRiskLimits(min_cash=500.0, max_market_data_age_ms=10_000, max_private_state_age_ms=10_000))
    risk = RuntimeRiskEngine(limits=limits, clock_ms=lambda: 1000)
    frame = StrategyInputFrame(primary=_event(), primary_private=_private(1000))
    action = Action(
        actor_id="a",
        orders_to_create=(Order("nobitex", "BTC_IRT", "buy", 100.0, 10.0),),
    )
    result = risk.validate_frame_with_result(
        [action],
        TradingState(actor_id="a", inventory=0.0, cash=600.0),
        frame,
    )
    assert result.approved == ()
    assert result.rejection_counts.get("min_cash", 0) >= 1


def test_diagnostics_expose_primary_and_hedge_limits() -> None:
    risk_cfg = RiskRuntimeConfig(
        enabled=True,
        primary=VenueRiskLimits(max_inventory=42.0, min_inventory=0.0),
    )
    runtime = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, risk=risk_cfg),
    )
    ctx = DecisionContext(
        config=runtime.controller,  # type: ignore[arg-type]
        frame=StrategyInputFrame(primary=_event()),
        clean_frame=StrategyInputFrame(primary=_event()),
        trading_state=TradingState(actor_id="a"),
        frame_step=1,
    )
    diag = ctx.to_diagnostics()
    payload = diag.as_payload()
    assert payload["risk"]["limits"]["primary"]["max_inventory"] == 42.0
    assert payload["risk"]["limits"]["primary"]["min_inventory"] == 0.0
    assert "hedge" in payload["risk"]["limits"]
