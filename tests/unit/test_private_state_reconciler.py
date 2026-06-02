from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.execution import Fill, Order
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.factory import select_private_state_adapter
from mm.exchanges.nobitex.private_state import NobitexRestPrivateStateAdapter
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.exchanges.wallex import private_rest as wallex_private_rest
from mm.exchanges.wallex.private_state import WallexRestPrivateStateAdapter
from mm.private_state.reconciler import PrivateStateReconciler
from mm.private_state.store import PrivateStateStore


class FakePrivateStateAdapter(AbstractPrivateStateAdapter):
    def __init__(self):
        self.symbol = "BTC_IRT"
        self._connected = False
        self.balances: asyncio.Queue[Balance | None] = asyncio.Queue()
        self.open_orders: asyncio.Queue[OpenOrder | None] = asyncio.Queue()
        self.order_updates: asyncio.Queue[OrderUpdate | None] = asyncio.Queue()
        self.fills: asyncio.Queue[Fill | None] = asyncio.Queue()

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        await self.balances.put(None)
        await self.open_orders.put(None)
        await self.order_updates.put(None)
        await self.fills.put(None)

    async def snapshot(self) -> PrivateStateSnapshot:
        return PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={
                "BTC": Balance(
                    exchange="nobitex",
                    asset="BTC",
                    free=1.0,
                    locked=0.25,
                    total=1.25,
                    timestamp_ms=1000,
                )
            },
            open_orders={},
        )

    async def balance_stream(self) -> AsyncIterator[Balance]:
        async for item in self._stream(self.balances):
            yield item

    async def open_order_stream(self) -> AsyncIterator[OpenOrder]:
        async for item in self._stream(self.open_orders):
            yield item

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        async for item in self._stream(self.order_updates):
            yield item

    async def fill_stream(self) -> AsyncIterator[Fill]:
        async for item in self._stream(self.fills):
            yield item

    async def _stream(self, queue):
        while self._connected:
            item = await queue.get()
            if item is None:
                break
            yield item


@pytest.mark.asyncio
async def test_private_state_reconciler_publishes_snapshot_and_open_orders() -> None:
    adapter = FakePrivateStateAdapter()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    snapshots = bus.subscribe_queue("private:state:nobitex:snapshot")
    open_orders = bus.subscribe_queue("private:state:nobitex:open_orders")

    try:
        await reconciler.start()
        initial = await asyncio.wait_for(snapshots.get(), timeout=1.0)
        assert initial.balances["BTC"].free == 1.0
        assert "IRT" in initial.balances
        assert initial.balances["IRT"].free == 0.0
        assert initial.balances["IRT"].locked == 0.0

        order = OpenOrder(
            exchange_order_id="o1",
            actor_id="actor-1",
            order=Order(exchange="nobitex", symbol="BTC_IRT", side="buy", price=100.0, qty=0.5),
            remaining_qty=0.5,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1100,
        )
        await adapter.open_orders.put(order)
        seen = await asyncio.wait_for(open_orders.get(), timeout=1.0)
        updated = await asyncio.wait_for(snapshots.get(), timeout=1.0)

        assert seen.exchange_order_id == "o1"
        assert updated.open_orders["o1"].order.price == 100.0
    finally:
        await reconciler.stop()


def test_private_state_factory_selects_rest_adapter() -> None:
    selection = select_private_state_adapter(
        "nobitex",
        "BTC_IRT",
        mode="websocket",
        api_token="token",
    )

    assert selection.mode == "rest"
    assert isinstance(selection.adapter, NobitexRestPrivateStateAdapter)


def test_rest_private_state_normalizes_open_orders_to_float_domain() -> None:
    adapter = RestPollingPrivateStateAdapter(
        "nobitex",
        "BTC_IRT",
        api_token="token",
    )

    order = adapter.normalize_open_order(
        {
            "id": "o1",
            "type": "buy",
            "price": "1000000",
            "amount": "0.5",
            "filledAmount": "0.1",
            "status": "open",
        },
        timestamp_ms=1000,
    )

    assert order is not None
    assert order.order.price == 1_000_000.0
    assert order.remaining_qty == pytest.approx(0.4)
    assert order.order.order_type == "limit"


def test_nobitex_irt_open_order_normalizes_price_from_rial_to_toman() -> None:
    adapter = NobitexRestPrivateStateAdapter("BTC_IRT", api_token="token")

    order = adapter.normalize_open_order(
        {
            "id": "o1",
            "type": "buy",
            "price": "10000000",
            "amount": "0.5",
            "filledAmount": "0.1",
            "status": "open",
        },
        timestamp_ms=1000,
    )

    assert order is not None
    assert order.order.price == pytest.approx(1_000_000.0)
    assert order.remaining_qty == pytest.approx(0.4)


def test_rest_private_state_normalize_balances_skips_zero_rows() -> None:
    adapter = RestPollingPrivateStateAdapter(
        "wallex",
        "BTC_IRT",
        api_token="token",
    )
    out = adapter.normalize_balances(
        {
            "BTC": {"free": 0.01, "locked": 0.0},
            "ETH": {"free": 0.0, "locked": 0.0},
            "USDT": {"free": 0.0, "locked": 1e-11},
        },
        timestamp_ms=1,
    )
    assert set(out) == {"BTC", "IRT"}
    assert out["BTC"].free == pytest.approx(0.01)
    assert out["IRT"].free == pytest.approx(0.0)


def test_rest_private_state_normalize_balances_normalizes_irt_aliases() -> None:
    adapter = RestPollingPrivateStateAdapter(
        "nobitex",
        "BTC_IRT",
        api_token="token",
    )
    out = adapter.normalize_balances(
        {
            "RLS": {"free": 10.0, "locked": 1.0},
            "TMN": {"free": 3.0, "locked": 2.0},
            "IRT": {"free": 1.0, "locked": 4.0},
        },
        timestamp_ms=1,
    )
    assert set(out) == {"IRT", "BTC"}
    assert out["IRT"].free == pytest.approx(1.0)
    assert out["IRT"].locked == pytest.approx(4.0)
    assert out["BTC"].free == pytest.approx(0.0)
    assert out["BTC"].locked == pytest.approx(0.0)


def test_nobitex_irt_fill_normalizes_price_and_fee_from_rial_to_toman() -> None:
    adapter = NobitexRestPrivateStateAdapter("BTC_IRT", api_token="token")

    fill = adapter.normalize_fill(
        {
            "id": "o1",
            "type": "buy",
            "averagePrice": "10000000",
            "matchedAmount": "0.1",
            "fee": "5000",
        },
        timestamp_ms=1000,
    )

    assert fill is not None
    assert fill.price == pytest.approx(1_000_000.0)
    assert fill.fee == pytest.approx(500.0)


def test_wallex_fill_detail_normalizes_fee_to_quote_units() -> None:
    adapter = WallexRestPrivateStateAdapter("USDT_IRT", api_token="token")

    buy_fill = adapter.normalize_fill(
        {
            "clientOrderId": "buy-1",
            "side": "BUY",
            "fills": [
                {
                    "price": "173193",
                    "quantity": "0.64",
                    "fee": "0.00192",
                    "feeAsset": "USDT",
                    "sum": "110843.52",
                }
            ],
        },
        timestamp_ms=1000,
    )
    sell_fill = adapter.normalize_fill(
        {
            "clientOrderId": "sell-1",
            "side": "SELL",
            "fills": [
                {
                    "price": "173209",
                    "quantity": "0.6",
                    "fee": "259.8135",
                    "feeAsset": "TMN",
                    "sum": "103925.4",
                }
            ],
        },
        timestamp_ms=1000,
    )

    assert buy_fill is not None
    assert buy_fill.price == pytest.approx(173193.0)
    assert buy_fill.qty == pytest.approx(0.64)
    assert buy_fill.fee == pytest.approx(0.00192 * 173193.0)
    assert sell_fill is not None
    assert sell_fill.fee == pytest.approx(259.8135)


def test_rest_fill_normalizer_emits_fee_delta_once() -> None:
    adapter = RestPollingPrivateStateAdapter("nobitex", "BTC_IRT", api_token="token")

    first = adapter.normalize_fill(
        {"id": "o1", "side": "buy", "executedQty": "0.5", "executedPrice": "100.0", "fee": "2.0"},
        timestamp_ms=1000,
    )
    second = adapter.normalize_fill(
        {"id": "o1", "side": "buy", "executedQty": "0.8", "executedPrice": "100.0", "fee": "3.0"},
        timestamp_ms=1001,
    )
    duplicate = adapter.normalize_fill(
        {"id": "o1", "side": "buy", "executedQty": "0.8", "executedPrice": "100.0", "fee": "3.0"},
        timestamp_ms=1002,
    )

    assert first is not None
    assert first.qty == pytest.approx(0.5)
    assert first.fee == pytest.approx(2.0)
    assert second is not None
    assert second.qty == pytest.approx(0.3)
    assert second.fee == pytest.approx(1.0)
    assert duplicate is None


def test_wallex_filled_order_history_fetches_detail_when_fills_missing(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    def fake_get_auth(path, api_token, params=None):
        calls.append((path, params))
        if path == "/v1/account/orders":
            return {
                "success": True,
                "result": {
                    "orders": [
                        {
                            "clientOrderId": "order-1",
                            "status": "FILLED",
                            "executedQty": "0.6",
                            "fills": None,
                        }
                    ]
                },
            }
        if path == "/v1/account/orders/order-1":
            return {
                "success": True,
                "result": {
                    "clientOrderId": "order-1",
                    "status": "FILLED",
                    "executedQty": "0.6",
                    "fee": "259.8135",
                    "fills": [{"fee": "259.8135", "feeAsset": "TMN"}],
                },
            }
        raise AssertionError(path)

    monkeypatch.setattr(wallex_private_rest, "_get_auth", fake_get_auth)

    rows = wallex_private_rest.list_filled_orders("USDT_IRT", "token")

    assert rows == [
        {
            "clientOrderId": "order-1",
            "status": "FILLED",
            "executedQty": "0.6",
            "fee": "259.8135",
            "fills": [{"fee": "259.8135", "feeAsset": "TMN"}],
        }
    ]
    assert calls[0] == ("/v1/account/orders", {"market": "USDTTMN", "per_page": 100})
    assert calls[1] == ("/v1/account/orders/order-1", None)


def test_private_state_store_apply_balance_pops_zero() -> None:
    store = PrivateStateStore("wallex")
    store.apply_balance(Balance("wallex", "BTC", 1.0, 0.0, 1.0, 1))
    assert "BTC" in store.snapshot().balances
    store.apply_balance(Balance("wallex", "BTC", 0.0, 0.0, 0.0, 2))
    assert "BTC" not in store.snapshot().balances


class _StubRestPolling(RestPollingPrivateStateAdapter):
    """Minimal REST adapter to exercise unified reconciler polling."""

    def __init__(self) -> None:
        super().__init__("nobitex", "BTC_IRT", api_token="t", poll_interval_ms=25)
        self.snapshot_calls = 0

    async def fetch_balances(self) -> dict:
        return {"BTC": {"free": "0.25", "locked": "0.0"}}

    async def fetch_open_orders(self) -> list[dict]:
        return []

    async def snapshot(self) -> PrivateStateSnapshot:
        self.snapshot_calls += 1
        return await super().snapshot()


class _FillingRestPolling(RestPollingPrivateStateAdapter):
    def __init__(self) -> None:
        super().__init__("nobitex", "BTC_IRT", api_token="t", poll_interval_ms=25)
        self.open_order_calls = 0

    async def fetch_balances(self) -> dict:
        return {"BTC": {"free": "0.25", "locked": "0.0"}}

    async def fetch_open_orders(self) -> list[dict]:
        self.open_order_calls += 1
        filled = "0.0" if self.open_order_calls == 1 else "0.1"
        return [
            {
                "id": "o1",
                "type": "buy",
                "price": "1000000",
                "amount": "0.5",
                "filledAmount": filled,
                "status": "open",
            }
        ]


class _FillHistoryRestPolling(RestPollingPrivateStateAdapter):
    def __init__(self) -> None:
        super().__init__("nobitex", "BTC_IRT", api_token="t", poll_interval_ms=500)
        self.fill_calls = 0

    async def fetch_balances(self) -> dict:
        return {"BTC": {"free": "0.25", "locked": "0.0"}}

    async def fetch_open_orders(self) -> list[dict]:
        return []

    async def fetch_fills(self) -> list[dict]:
        self.fill_calls += 1
        matched = "0.1" if self.fill_calls == 1 else "0.2"
        return [
            {
                "id": "o1",
                "type": "buy",
                "averagePrice": "1000000",
                "matchedAmount": matched,
                "fee": "0.01",
            }
        ]


class _EmptyBalanceRestPolling(RestPollingPrivateStateAdapter):
    def __init__(self) -> None:
        super().__init__("nobitex", "BTC_IRT", api_token="t", poll_interval_ms=25)

    async def fetch_balances(self) -> dict:
        return {}

    async def fetch_open_orders(self) -> list[dict]:
        return []


class _FlakyRestPolling(RestPollingPrivateStateAdapter):
    def __init__(self) -> None:
        super().__init__("nobitex", "BTC_IRT", api_token="t", poll_interval_ms=25)
        self.balance_calls = 0

    async def fetch_balances(self) -> dict:
        self.balance_calls += 1
        if self.balance_calls == 1:
            return {"BTC": {"free": "0.25", "locked": "0.0"}}
        raise RuntimeError("wallet endpoint failed")

    async def fetch_open_orders(self) -> list[dict]:
        return []


@pytest.mark.asyncio
async def test_rest_polling_reconciler_uses_unified_snapshot_poll() -> None:
    adapter = _StubRestPolling()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    snapshots = bus.subscribe_queue("private:state:nobitex:snapshot")

    try:
        await reconciler.start()
        first = await asyncio.wait_for(snapshots.get(), timeout=1.0)
        assert first.balances["BTC"].free == pytest.approx(0.25)
        second = await asyncio.wait_for(snapshots.get(), timeout=1.0)
        assert second.balances["BTC"].free == pytest.approx(0.25)
        assert adapter.snapshot_calls >= 2
    finally:
        await reconciler.stop()


@pytest.mark.asyncio
async def test_rest_polling_reconciler_publishes_fill_deltas_from_open_order_snapshots() -> None:
    adapter = _FillingRestPolling()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler(
        "nobitex",
        adapter,
        bus,
        fill_event_source="open_order_delta",
    )
    fills = bus.subscribe_queue("private:state:nobitex:fills")

    try:
        await reconciler.start()
        fill = await asyncio.wait_for(fills.get(), timeout=1.0)

        assert fill.exchange_order_id == "o1"
        assert fill.exchange == "nobitex"
        assert fill.symbol == "BTC_IRT"
        assert fill.side == "buy"
        assert fill.qty == pytest.approx(0.1)
        assert fill.price == pytest.approx(1_000_000.0)
    finally:
        await reconciler.stop()


@pytest.mark.asyncio
async def test_rest_polling_reconciler_suppresses_open_order_delta_fills_by_default() -> None:
    adapter = _FillingRestPolling()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    snapshots = bus.subscribe_queue("private:state:nobitex:snapshot")
    fills = bus.subscribe_queue("private:state:nobitex:fills")

    try:
        await reconciler.start()
        await asyncio.wait_for(snapshots.get(), timeout=1.0)
        await asyncio.wait_for(snapshots.get(), timeout=1.0)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(fills.get(), timeout=0.1)
        assert adapter.open_order_calls >= 2
    finally:
        await reconciler.stop()


@pytest.mark.asyncio
async def test_rest_polling_reconciler_publishes_exchange_fill_history_by_default() -> None:
    adapter = _FillHistoryRestPolling()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    fills = bus.subscribe_queue("private:state:nobitex:fills")

    try:
        await reconciler.start()
        fill = await asyncio.wait_for(fills.get(), timeout=2.0)

        assert fill.exchange_order_id == "o1"
        assert fill.qty == pytest.approx(0.1)
        assert fill.fee == pytest.approx(0.01)
    finally:
        await reconciler.stop()


@pytest.mark.asyncio
async def test_rest_fill_stream_polls_done_order_history_after_seed() -> None:
    adapter = _FillHistoryRestPolling()
    await adapter.connect()
    stream = adapter.fill_stream()
    try:
        fill = await asyncio.wait_for(stream.__anext__(), timeout=1.5)

        assert fill.exchange_order_id == "o1"
        assert fill.exchange == "nobitex"
        assert fill.symbol == "BTC_IRT"
        assert fill.side == "buy"
        assert fill.qty == pytest.approx(0.1)
        assert fill.price == pytest.approx(1_000_000.0)
        assert fill.fee == pytest.approx(0.01)
    finally:
        await adapter.disconnect()
        await stream.aclose()


def test_private_state_store_keeps_economy_assets_explicit_zero() -> None:
    store = PrivateStateStore("nobitex", trading_symbol="BTC_IRT")
    store.apply_balance(Balance("nobitex", "BTC", 1.0, 0.0, 1.0, 1))
    snap = store.snapshot()
    assert snap.balances["BTC"].free == pytest.approx(1.0)
    assert snap.balances["IRT"].free == pytest.approx(0.0)
    store.apply_balance(Balance("nobitex", "IRT", 0.0, 0.0, 0.0, 2))
    snap2 = store.snapshot()
    assert "IRT" in snap2.balances
    assert snap2.balances["IRT"].free == pytest.approx(0.0)


def test_rest_private_state_normalize_injects_missing_economy_balances() -> None:
    adapter = RestPollingPrivateStateAdapter(
        "nobitex",
        "BTC_IRT",
        api_token="token",
    )
    out = adapter.normalize_balances({"BTC": {"free": 0.01, "locked": 0.0}}, timestamp_ms=1)
    assert set(out) == {"BTC", "IRT"}
    assert out["IRT"].free == pytest.approx(0.0)
    assert out["IRT"].locked == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_rest_private_state_snapshot_rejects_empty_balance_fetch() -> None:
    adapter = _EmptyBalanceRestPolling()
    await adapter.connect()
    try:
        with pytest.raises(RuntimeError, match="private balance fetch returned no balances"):
            await adapter.snapshot()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_rest_polling_reconciler_keeps_last_snapshot_on_fetch_failure() -> None:
    adapter = _FlakyRestPolling()
    bus = InMemoryBus()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    snapshots = bus.subscribe_queue("private:state:nobitex:snapshot")
    errors = bus.subscribe_queue("private:state:nobitex:errors")

    try:
        await reconciler.start()
        first = await asyncio.wait_for(snapshots.get(), timeout=1.0)
        assert first.balances["BTC"].free == pytest.approx(0.25)

        error = await asyncio.wait_for(errors.get(), timeout=1.0)
        assert error["source"] == "unified_rest_poll"
        assert error["error"] == "RuntimeError"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(snapshots.get(), timeout=0.05)
        assert reconciler.snapshot().balances["BTC"].free == pytest.approx(0.25)
    finally:
        await reconciler.stop()
