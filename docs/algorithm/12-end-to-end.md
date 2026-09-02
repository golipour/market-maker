# 12 — A sample event traced end-to-end

This section walks **one** `MarketEvent` from the wire all the way to a
published order. The example uses plausible numbers, not real market data;
the goal is to make the data flow concrete.

## 12.1 Setup

- Local venue: `nobitex`, symbol `BTC-USDT`, `Δt = 1000 ms`.
- Global venue: `binance`, symbol `BTC-USDT`.
- Actor: `mm-btc-001`, with `n_components = 3`, `horizon = 5`, `n_samples = 128`.

## 12.2 Step 1 — ingestion

At `t = 12:00:00.000` the `SymbolEngine` for `(nobitex, BTC-USDT)` receives a
REST snapshot of the orderbook and a WebSocket batch of trades. The feed
task pushes them into `OrderbookManager` and `RollingTradeBuffer`. The
`market:raw:nobitex:BTC-USDT:orderbook` and `:trades` topics receive the raw
payloads.

At `t = 12:00:01.000` the publish task fires, drains the trade buffer for
`[12:00:00.000, 12:00:01.000)`, and produces:

```
MarketEvent(
    exchange="nobitex", symbol="BTC-USDT", interval_ms=1000,
    t_start_ms=..., t_end_ms=...,
    orderbook=TopOfBook(bid=67500.1 @ 0.05, ask=67501.3 @ 0.04, ...),
    trades=[Trade(price=67500.5, qty=0.01, side="buy"), ...],
    buy_volume=0.42, sell_volume=0.31,
    vwap=67500.6, imbalance=0.15, microprice=67500.7,
    book_buy_vwap=67490.2, book_sell_vwap=67512.8, book_vwap_depth=2.4,
)
```

## 12.3 Step 2 — frame assembly

The `StrategyInputFrameAssembler` for `mm-btc-001` is waiting on its local
queue. It pops the event, drains the global queue (which has the matching
binance event already), and calls `build_frame`. The `JointFrameAssembler`
checks freshness and publishes a `JointStrategyFrame`.

## 12.4 Step 3 — strategy runner

The `StrategyRunner.run_step` coroutine pops the frame:

1. **Open orders** — the ledger reports 4 resting orders (2 bids, 2 asks)
   on nobitex, 0 on binance.
2. **Clean** — `subtract_own_orders` removes the bot's quotes from the
   `nobitex` book. The cleaned book now shows the *rest of the market*.
3. **Optimizer dispatch** — the cleaned frame and joint ledger are passed
   to `MpcStrategyEngine.decide`, which builds a `State` and a
   `MonteCarloMPCTensorBatched` optimizer. The future runs in the
   `ThreadPoolExecutor` with `optimizer_timeout_ms = 2000`.
4. **Inside the optimizer** (4-8 ms typical on CPU):
   - The L-param decoder turns the current raw vector into `(L, k, x0)`
     tensors (3 components per side → 18 raw parameters per side, 36 total).
   - Adam runs `E = 12` epochs; each epoch draws 128 paths of length 5.
   - At each step the soft fill is computed, the inventory/cash are
     updated, the terminal objective is summed.
   - The best epoch's `(L, k, x0)` are returned along with diagnostics.
5. **Target book** — the sigmoid action is laid on the nobitex price grid
   (tick = 0.1 USDT, lot = 0.0001 BTC) and constrained to `min_qty = 0.001`
   and `min_notional = 10 USDT`. The result is 4 bids and 4 asks inside
   the validity bounds.
6. **Risk check** — the `NetDeltaRiskEngine` validates the target book
   against the joint ledger. All 8 orders pass.
7. **Dispatch** — the approved book is routed to the local `OrderManager`,
   which places the orders on nobitex via REST. The decision is published
   to `decisions:mm-btc-001` and the diagnostics to
   `diagnostics:controller:mm-btc-001`.

## 12.5 What gets logged

The `ControllerDiagnostics` payload contains:

- `optimizer_status: "ok"`,
- `objective_history: [...12 floats...]`,
- `best_risk_adjusted_return: 0.0003`,
- `warm_start_alpha: 0.7` (warm-started from the previous decision),
- `reservation_r_offset: -0.0001` (slight short-skew because the bot is
  long),
- `book_quality_score: 0.92`,
- `availability_shortfall: 0.0`,
- `viability_theta: 0.01` (stable),
- `gradient_norm_final: 0.4`,
- `decision_hash: "9f3a..."`.
