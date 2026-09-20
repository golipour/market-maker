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


class _RawSigmoidWithBaseline:
    def depth_bid(self, spread: float) -> float:
        return 10.0 + float(spread)

    def depth_ask(self, spread: float) -> float:
        return 20.0 + 2.0 * float(spread)


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


def test_ladder_orders_use_effective_depth_not_raw_depth_at_zero() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook

    event = MarketEvent(
        exchange="nobitex",
        symbol="USDT_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="USDT_IRT",
            timestamp_ms=1000,
            bids=(PriceLevel(price=99.0, qty=1.0),),
            asks=(PriceLevel(price=101.0, qty=1.0),),
            is_snapshot=True,
        ),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=0.0,
    )
    orders = ladder_orders_from_sigmoid(
        actor_id="a",
        event=event,
        sigmoid_action=_RawSigmoidWithBaseline(),
        orderbook_grid=_FakeGrid(),
    )

    buy_qtys = [order.qty for order in orders if order.side == "buy"]
    sell_qtys = [order.qty for order in orders if order.side == "sell"]
    assert buy_qtys == [1.0, 1.0, 1.0]
    assert sell_qtys == [2.0, 2.0, 2.0]


def test_ladder_orders_honor_explicit_execution_spread_ladder() -> None:
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook

    event = MarketEvent(
        exchange="nobitex",
        symbol="USDT_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="USDT_IRT",
            timestamp_ms=1000,
            bids=(PriceLevel(price=99.0, qty=1.0),),
            asks=(PriceLevel(price=101.0, qty=1.0),),
            is_snapshot=True,
        ),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=0.0,
    )
    orders = ladder_orders_from_sigmoid(
        actor_id="a",
        event=event,
        sigmoid_action=_RawSigmoidWithBaseline(),
        orderbook_grid=_FakeGrid(),
        spread_ladder=(2.0, 5.0),
    )
    buy = [o for o in orders if o.side == "buy"]
    # mid = 100; effective depth = raw(spread) - raw(0) => qty = Δspread
    assert [round(o.price, 6) for o in buy] == [98.0, 95.0]
    assert [o.qty for o in buy] == [2.0, 3.0]


def test_binding_merges_api_venue_then_config_fallback(monkeypatch) -> None:
    from mm.core.models.instrument import build_instrument_spec
    from mm.exchanges.shared import instrument_spec_trade_stats as mod

    api_spec = build_instrument_spec("nobitex", "USDT_IRT", qty_step=0.001, min_qty=0.05)
    monkeypatch.setattr(mod, "instrument_spec_from_trade_stats", lambda ex, sym: api_spec)

    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="USDT_IRT",
            instrument=InstrumentPrecisionConfig(qty_step=0.01, min_qty=0.01),
        ),
    )
    spec = config.binding.local
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
    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="USDT_IRT",
            instrument=InstrumentPrecisionConfig(qty_step=0.01, min_qty=0.01),
        ),
    )
    spec = config.binding.local
    assert spec.qty_step == 0.01
    assert spec.min_qty == 0.01
