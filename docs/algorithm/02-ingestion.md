# 02 — Ingestion: from raw trades & orderbooks to per-interval `MarketEvent`s

This section walks the path from a WebSocket/REST feed into the `MarketEvent` consumed
by the strategy. The pipeline is per-`(exchange, symbol)` and is owned by
`SymbolEngine`.

## 2.1 Notational conventions

- `Δt` — aggregation interval, e.g. 1000 ms.
- `MarketEvent` — a closed-window snapshot over `[t_start_ms, t_end_ms]`.
- `RollingTradeBuffer` — sliding window of recent trades (capped by `trades_limit`).
- `OrderbookManager` — keeps the latest top-N levels for both sides.

## 2.2 `SymbolEngine`

The engine is created once per `(exchange, symbol, interval_ms)` triple. It launches:

1. a **feed task** that calls into `AbstractMarketDataAdapter.stream_trades()` and
   `stream_orderbook()`, and
2. a **publish task** that runs every `Δt` ms, builds a `MarketEvent`, and emits it to
   the bus.

```10:39:mm/ingestion/symbol_engine.py
class SymbolEngine:
    def __init__(self, *, adapter, exchange, symbol, interval_ms,
                 bus, output_topic, raw_trade_topic, raw_orderbook_topic,
                 depth_limit, trades_limit, ...):
        self.adapter = adapter
        self.bus = bus
        ...
        self.trade_buffer = RollingTradeBuffer(...)
        self.orderbook_mgr = OrderbookManager(...)
```

### 2.2.1 Feed task

The feed task receives trades and orderbook diffs from the adapter, **publishes the
raw payload** to a side topic (so other consumers can attach), and pushes the parsed
object into the in-memory buffers:

```71:99:mm/ingestion/symbol_engine.py
async def _feed_trades(self):
    async for trade in self.adapter.stream_trades():
        self.trade_buffer.add(trade)
        await self.bus.publish(self.raw_trade_topic(), trade)

async def _feed_orderbook(self):
    async for snapshot in self.adapter.stream_orderbook():
        self.orderbook_mgr.apply_snapshot(snapshot)
        await self.bus.publish(self.raw_orderbook_topic(), snapshot)
```

The raw topics are written *before* the in-memory buffer is touched, so any subscriber
that listens to `market:raw:*` will see a consistent view with the engine itself.
Adapter errors are funnelled through `_feed_adapter_errors` and emitted to
`market:errors:<ex>:<sym>`.

### 2.2.2 Publish task

Every `Δt` ms the engine drains the trade buffer for the *closed* window `[t_start,
t_end]` and combines it with the current orderbook snapshot to produce a
`MarketEvent`. The interval scheduler calls `_publish_window`:

```110:140:mm/ingestion/symbol_engine.py
def _publish_window(self, t_start_ms: int, t_end_ms: int) -> MarketEvent:
    trades = self.trade_buffer.window(t_start_ms, t_end_ms)
    orderbook = self.orderbook_mgr.snapshot()
    return build_market_event(
        exchange=..., symbol=..., interval_ms=..., t_start_ms=..., t_end_ms=...,
        orderbook=orderbook, trades=trades,
    )
```

The actual builder is `mm/aggregation/builder.py:build_market_event`, which fills the
`sell_volume`, `buy_volume`, `vwap`, `imbalance`, `microprice`, and the
`book_buy_vwap` / `book_sell_vwap` / `book_vwap_depth` aggregates (see
[09-target-book.md](09-target-book.md) for the math).

## 2.3 Aggregation: from trades to a `MarketEvent`

The builder's job is to map a closed window of trades plus a top-of-book snapshot to
the canonical event. The MPC's stochastic models consume the *aggregated* fields
(`buy_volume`, `sell_volume`, `vwap`, `imbalance`), not the raw trades; the orderbook
itself is kept as a `TopOfBook` (or L2 list) for `order_constraints` and for the
ladder builder.

```1:60:mm/aggregation/builder.py
def build_market_event(*, exchange, symbol, interval_ms, t_start_ms, t_end_ms,
                       orderbook, trades) -> MarketEvent:
    buy = sum(t.qty for t in trades if t.side == "buy")
    sell = sum(t.qty for t in trades if t.side == "sell")
    vwap = _vwap(trades) if trades else mid_from_orderbook(orderbook)
    imbalance = (buy - sell) / max(buy + sell, eps)
    microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
    book_buy_vwap, book_sell_vwap, book_vwap_depth = book_vwaps(orderbook)
    return MarketEvent(...)
```

**Definitions used here:**

- `vwap = Σ price·qty / Σ qty` (volume-weighted average price over the window).
- `imbalance = (buy − sell) / (buy + sell)` (sign indicates directional pressure).
- `microprice = (bid·ask_qty + ask·bid_qty) / (bid_qty + ask_qty)` (a depth-weighted
  inside fair price).
- `book_buy_vwap` and `book_sell_vwap` are cumulative VWAPs of the bid and ask
  sides, with `book_vwap_depth` being the notional traded through those levels.

These aggregates become the **flow features** consumed by `mm/mpc/market_order_flow_torch.py`
and the **price features** consumed by `mm/mpc/price_torch.py`. See
[06-mpc-models.md](06-mpc-models.md).

## 2.4 What happens on adapter error

If the adapter raises while streaming, the engine converts the exception into an
`AdapterError` payload on the `market:errors:<ex>:<sym>` topic. The strategy runner
ignores market data whose `MarketEvent` cannot be parsed and the control plane logs the
error; the next successful window resumes the pipeline.

## 2.5 What happens on slow consumers

The bus backpressure policy is configured globally:

- `block` (default) — `publish` awaits until the slowest subscriber drains.
- `drop_oldest` — the oldest queued message for a slow subscriber is discarded.
- `drop_newest` — the incoming message is discarded for a slow subscriber.

The runtime uses `block` by default for `market:events:*` and `drop_oldest` for raw
topics; this is enforced in `MarketDataPlane.build` (see `mm/runtime/engine.py`).
