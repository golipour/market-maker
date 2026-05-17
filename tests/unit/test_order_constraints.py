from __future__ import annotations

import math

from mm.core.models.execution import Order
from mm.core.models.instrument import InstrumentPrecisionConfig, InstrumentSpec, build_instrument_spec
from mm.core.order_constraints import (
    OrderConstraints,
    apply_venue_constraints,
    cap_orders_per_side,
    ladder_orders_from_sigmoid,
)
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig


class _FakeSigmoid:
    def depth_bid(self, spread: float) -> float:
        return float(spread) * 0.1

    def depth_ask(self, spread: float) -> float:
        return float(spread) * 0.05


class _FakeGrid:
    def spread_ladder(self) -> tuple[float, ...]:
        return (1.0, 2.0, 3.0)


def _order(side: str, qty: float, price: float = 100.0) -> Order:
    return Order("nobitex", "USDT_IRT", side, price, qty)


def test_order_constraints_from_spec() -> None:
    spec = build_instrument_spec("nobitex", "USDT_IRT", qty_step=0.01, min_qty=0.02, min_notional=50_000.0)
    c = OrderConstraints.from_spec(spec, max_orders_per_side=10)
    assert c.qty_step == 0.01
    assert c.min_qty == 0.02
    assert c.min_notional == 50_000.0
    assert c.max_orders_per_side == 10


def test_apply_venue_constraints_quantizes_and_drops_small() -> None:
    constraints = OrderConstraints(
        price_step=None,
        qty_step=0.01,
        min_qty=0.02,
        min_notional=None,
        max_orders_per_side=99,
    )
    orders = [_order("buy", 0.015), _order("sell", 0.03)]
    out = apply_venue_constraints(orders, constraints)
    assert len(out) == 1
    assert out[0].side == "sell"
    assert math.isclose(out[0].qty, 0.03)


def test_cap_orders_per_side_keeps_largest_qty() -> None:
    orders = [
        _order("buy", 0.01),
        _order("buy", 0.5),
        _order("buy", 0.2),
        _order("sell", 0.4),
        _order("sell", 0.1),
    ]
    capped = cap_orders_per_side(orders, 2)
    buy_qtys = sorted((o.qty for o in capped if o.side == "buy"), reverse=True)
    sell_qtys = sorted((o.qty for o in capped if o.side == "sell"), reverse=True)
    assert buy_qtys == [0.5, 0.2]
    assert sell_qtys == [0.4, 0.1]


def test_binding_merges_api_venue_then_config_fallback(monkeypatch) -> None:
    from mm.core.models.instrument import build_instrument_spec
    from mm.exchanges.shared import instrument_spec_trade_stats as mod

    api_spec = build_instrument_spec("nobitex", "USDT_IRT", qty_step=0.001, min_qty=0.05)
    monkeypatch.setattr(mod, "instrument_spec_from_trade_stats", lambda ex, sym: api_spec)

    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(
            role="primary",
            exchange="nobitex",
            symbol="USDT_IRT",
            instrument=InstrumentPrecisionConfig(qty_step=0.01, min_qty=0.01),
        ),
    )
    spec = config.binding.primary
    assert spec.qty_step == 0.001
    assert spec.min_qty == 0.05


def test_binding_config_fills_when_api_omits_fields(monkeypatch) -> None:
    from mm.core.models.instrument import build_instrument_spec
    from mm.exchanges.shared import instrument_spec_trade_stats as mod

    monkeypatch.setattr(
        mod,
        "instrument_spec_from_trade_stats",
        lambda ex, sym: build_instrument_spec(ex, sym),
    )
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(
            role="primary",
            exchange="nobitex",
            symbol="USDT_IRT",
            instrument=InstrumentPrecisionConfig(qty_step=0.01, min_qty=0.01),
        ),
    )
    spec = config.binding.primary
    assert spec.qty_step == 0.01
    assert spec.min_qty == 0.01

