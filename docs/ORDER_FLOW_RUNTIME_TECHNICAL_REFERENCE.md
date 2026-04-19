# Order Flow Runtime Technical Reference

This document explains the runtime behavior of the market maker end-to-end, including:

- call intervals and scheduler behavior,
- exchange adapters (with Nobitex details),
- all main unit conversions and formulas,
- state transitions and PnL accounting,
- controller/MPC objective, constraints, and action conversion,
- mode switches (`simulated` vs `real`, polling vs websocket),
- data contracts between modules.

It is intended as an implementation-level reference for developers.

---

## 1. High-Level Runtime Topology

The runtime chain is:

1. `order_flow_pipeline.main()`
2. `build_pipeline(...)` creates models, optimizer, controller, execution objects
3. `create_runner(...)` creates `OrderFlowStepper` + `OrderFlowRunner`
4. `run_animation(...)` starts Matplotlib loop
5. Every animation callback:
   - `runner.advance_scheduled(...)`
   - one or more `runner.advance(...)`
   - plotting refresh

The compute loop and UI loop are decoupled:

- **UI refresh cadence**: `animation_interval_ms`
- **strategy step cadence**: `T_ms` (inside `advance_scheduled`)

So animation may run faster than market-making steps, while steps remain aligned to fixed wall-clock cadence.

---

## 2. Main Configuration and Build Phase

`order_flow_pipeline.build_pipeline(...)` wires these components:

- **Market order flow model**: `TorchHurdleLogNormalModel`
- **Flow tester**: `GeneralOrderFlowLogLossTester`
- **Price model**: `TorchResidualPriceModel`
- **Execution model (simulated)**: `DiscretizedExecutionModel`
- **Execution model (real/observed)**: `ObservedStepExecutionModel`
- **MPC optimizer**: `MonteCarloMPCTensorBatched`
- **Controller**: `SigmoidController`
- **Execution rule**: `DefaultExecutionRule` (constructed but not used in runner transitions)

### 2.1 Execution mode switch

`pipeline["execution_mode"]` controls runner transition model:

- `"simulated"` -> runner uses `DiscretizedExecutionModel`
- `"real"` -> runner uses `ObservedStepExecutionModel`

In real mode, inventory/cash are sourced from stepper balances/private events rather than simulated OB-relative fills.

### 2.2 Min notional wiring

Controller conversion can enforce minimum order notional per ladder level.

- If `min_order_notional_tick_lot` is absent and `exchange/symbol` are known:
  - loaded via `get_min_order_amount_tick_lot(symbol, exchange)`
- conversion used:
  - `min_tick_lot = min_notional_quote / (tick_size * lot_size)`

This value is enforced during sigmoid->discrete ladder conversion.

---

## 3. Timing and Interval Semantics

There are multiple interval classes:

1. **Strategy step interval** (`T_ms`)
   - each step window corresponds to one model/decision transition.
2. **Animation interval** (`animation_interval_ms`)
   - redraw frequency only.
3. **Polling intervals**
   - trades: `trades_poll_interval_ms`
   - orderbook: `orderbook_poll_interval_ms`
4. **Websocket reconnect interval**
   - `websocket_reconnect_ms`

### 3.1 Step scheduling

`OrderFlowRunner.advance_scheduled(max_steps_per_call)` does:

- tracks `_next_due_perf` (from `time.perf_counter`)
- tracks `_next_step_end_ms` (wall-clock ms)
- while due and under `max_steps_per_call`:
  - calls `advance(t_end_ms=_next_step_end_ms)`
  - increments both due time and step-end time by one period

This preserves fixed-step progression even if UI callback timing jitters.

### 3.2 Step window modes

`OrderFlowStepper.step_window_mode`:

- `"lookback"`: use `[t_end - T_ms, t_end]`
- `"previous_end"`: use `[previous_t_end, current_t_end]`

`previous_end` avoids overlap/gaps caused by wall-clock jitter and preserves contiguous windows.

### 3.3 Balance update modes

`OrderFlowStepper.balance_update_mode`:

- `"events"` (default): update local balances from private trade/order events in each step window.
- `"rest"`: refresh balances from exchange REST API each step (uses adapter `balances()`; inventory/cash derive from **total** balance).

---

## 4. Adapter Layer and Market Data Normalization

The adapter system is in `trade_stats.py`.

Supported adapters include:

- `TabdealAdapter`
- `NobitexAdapter`
- `WallexAdapter`
- `BitPinAdapter`
- `KucoinAdapter`
- `BinanceAdapter`

All adapters normalize into canonical trade/orderbook schema consumed by stepper.

### 4.1 Canonical trade schema

Every normalized trade is:

```python
{
  "time": <ms>,
  "price": <price_ticks int>,
  "qty": <qty_lots int>,
  "isBuyerMaker": <bool>,
}
```

### 4.2 Canonical orderbook schema

Normalized orderbook (`Orderbook`) fields:

- `t_ms`: timestamp in ms
- `mid`: mid in **ticks** (float)
- `tick_size`: quote currency per tick
- `bids`/`asks`: list of `Level(price_ticks, qty_lots, spread_ticks)`

`spread_ticks` convention:

- bid: `spread = mid - price_ticks`
- ask: `spread = price_ticks - mid`

### 4.3 Core conversion formulas

Given adapter raw values:

- `price_ticks = round(price_currency / tick_size)`
- `qty_lots = round(qty_base / lot_size)`
- `mid_ticks = ((best_bid + best_ask) / 2) / tick_size`

For account balances -> model units:

- `inventory_lots = base_balance_native / lot_size`
- `cash_tick_lot = quote_balance_native / (tick_size * lot_size)`

### 4.4 Nobitex-specific details

Nobitex mapping appears in both REST and WS adapters:

- symbol format: `USDT_IRT` -> `USDTIRT`
- trade side mapping:
  - Nobitex `type == "sell"` means taker sold -> maker buy -> `isBuyerMaker=True`
- private channels:
  - `private:orders#...`
  - `private:trades#...`
- quote alias handling in balances:
  - `IRT` / `RLS` / `TMN` alias logic exists in stepper balance lookup.

For private trade deltas in stepper:

- sell event:
  - source asset decreases by amount
  - destination asset increases by `(total - fee)`
- buy event:
  - source asset increases by amount
  - destination asset decreases by `(total + fee)`

---

## 5. Stepper Internals and Data Providers

`OrderFlowStepper` can run with:

- REST on-demand providers,
- buffered polling providers,
- websocket buffered provider.

### 5.1 Provider behavior

- `BufferedPollingTradeDataProvider`
  - periodic REST fetch with overlap,
  - dedupe by key `(time, price, qty, isBuyerMaker)`,
  - rolling time-pruned deque.
- `BufferedPollingOrderbookDataProvider`
  - periodic orderbook snapshots in deque,
  - returns latest snapshot with `t_ms <= t_end_ms`.
- `WebSocketBufferedMarketDataProvider`
  - ingests trades + books (+ private events if available),
  - normalizes websocket trades into tick/lot units,
  - no REST fallback for trade stream in websocket mode.

### 5.2 `Step` contract

Each `advance(...)` returns:

```python
(
  t_end_ms,
  ob,
  buy_lots,
  sell_lots,
  inventory_lots,
  cash_tick_lot,
  private_f_b_total,
  private_f_a_total,
  private_fills_bid,  # tuple[(spread_ticks, qty_lots)]
  private_fills_ask,  # tuple[(spread_ticks, qty_lots)]
)
```

Units:

- flow volumes, inventory, private fills: lots
- prices/spreads/mid: ticks
- cash: tick*lot

### 5.3 Private event handling and balance state

Stepper keeps mutable balance state:

- `balances[asset] = {free, locked, total}`
- order lock tracking per order id
- applied trade-id set for idempotency

Balance updates occur in this order per step:

1. fetch private trades/orders in window,
2. convert private events to side fill buckets,
3. if `balance_update_mode == "events"`:
   - apply trade deltas to free+total,
   - apply order lock transitions between free/locked (total invariant),
4. if `balance_update_mode == "rest"`:
   - replace local balances with REST snapshot from adapter,
5. recompute model inventory/cash from updated balances (from `total`, not `free`).

This state machine is what feeds observed inventory/cash in real mode.

---

## 6. Runner Transition Pipeline

`OrderFlowRunner.advance(...)` is the core per-step transition.

For step `t`:

1. Fetch `curr_step` from stepper.
2. Apply **previous** action (`t-1`) against current step market result via execution model.
3. Update inventory/cash and realized PnL histories.
4. Compute diagnostics and expected metrics.
5. Update models/testers with `(prev_step, curr_step)`.
6. Build current state and compute new action (`t`).
7. Optionally send live cancel/replace limit orders.
8. Cache new action as pending for next step execution.

Important sequencing:

- action chosen now is applied on next step window.
- execution uses pending state/action from previous decision point.

---

## 7. State, Action, and Conversion Semantics

### 7.1 State

`State(mid, ob_rel, buy_vol, sell_vol, inventory, cash)` where:

- `mid`: ticks
- `ob_rel`: relative book (`spread_ticks`, qty)
- `buy_vol/sell_vol`: lots in current window
- `inventory`: lots
- `cash`: tick*lot

### 7.2 Action families

1. `SigmoidMixtureAction`
   - continuous paramization, optimized by MPC.
2. `DiscretizedOrderAction`
   - ladder of marginal lots per spread tick, used by execution model.

Depth function per side:

- `D(s) = Σ_i L_i * sigmoid(k_i * (s - x0_i))`

Marginal at spread:

- `dD/ds = Σ_i L_i * k_i * sig * (1 - sig)`

### 7.3 Sigmoid -> discrete ladder conversion

`sigmoid_action_to_discretized_action(...)`:

- build integer spread grid `[s_min, ..., s_max]`
- compute cumulative depth at each level
- enforce monotonic cumulative (`cum = max(cum, prev_cum)`)
- marginal qty = quantized difference (`lot_size` multiple)
- optional notional filter:
  - `qty_lots * price_ticks >= min_order_notional_tick_lot * safety_mult`

Resulting ladder quantities are **marginal** per level, not cumulative.

---

## 8. Execution Models and PnL Accounting

Two execution models implement `state_after_step(...)`.

### 8.1 Simulated: `DiscretizedExecutionModel`

Given `prev_state`, `prev_action`, `curr_step`:

1. Compute fills by ladder level based on depth ahead and market flow reach.
2. Aggregate side fills:
   - `f_b` total bid fills
   - `f_a` total ask fills
3. Inventory update:
   - `inv_new = inv_prev + f_b*(1-fee) - f_a`
4. Cash update:
   - `ask_notional = Σ(fill_ask_level * (mid + spread))`
   - `bid_notional = Σ(fill_bid_level * (mid - spread))`
   - `cash_delta = ask_notional*(1-fee) - bid_notional`
5. Wealth update:
   - `W = cash + inv*mid`
   - `wealth_pnl = W_after - W_before`

PnL decomposition returned:

- `spread_pnl`
- `inventory_pnl`
- `cross_pnl`
- `fee_pnl`
- `wealth_pnl`

### 8.2 Real/observed: `ObservedStepExecutionModel`

State transition ignores simulated fill mechanics and trusts step payload:

- `inv_new = curr_step.inventory`
- `cash_new = curr_step.cash`

PnL components are computed algebraically so their sum equals `wealth_pnl`.

---

## 9. Model Updates and Predictive Diagnostics

Per step, runner updates:

1. MOF tester with previous params vs current observation.
2. MOF model EWMA parameters.
3. Price tester (if enabled).
4. Price model EWMA residual parameters.
5. Optional LOF model.

### 9.1 Market flow model (`TorchHurdleLogNormalModel`)

Per side:

- zero mass update:
  - `p0 <- gamma*p0 + (1-gamma)*I[x==0]`
- positive branch stats on `log(x)` for `x>0`:
  - EWMA mean and EWMA variance
  - derive `mu`, `sigma`

Sampling:

- draw uniform; with prob `p0` output zero
- else sample `LogNormal(mu, sigma)`

### 9.2 Price residual model (`TorchResidualPriceModel`)

It models representative buy/sell residual around MOF-only computed VWAPs:

- compute post-trade synthetic book from previous book and observed market flow,
- compute VWAP to shared target depth on both synthetic and realized current book (instantaneous; no EWMA on those levels),
- residual per side:
  - `e_buy = buy_real - buy_cmp`
  - `e_sell = sell_real - sell_cmp`
- EWMA update of `mu_buy`, `mu_sell`, and variance per side only.

Returned params include:

- `mu_mid`, `sigma_mid`
- side-specific `mu_buy/mu_sell/sigma_buy/sigma_sell`

---

## 10. MPC: Objective, Sampling, and Constraints

`MonteCarloMPCTensorBatched.optimize(state)`:

1. Build batched initial book tensors from `state.ob_rel`.
2. Optimize horizon-wide sigmoid parameters (`H * 6 * n_components`).
3. For each epoch:
   - clamp params to bounds,
   - optional project total `L` sums into `[total_q_min, total_q_max]`,
   - draw or reuse MC trajectories (`n_samples x H`),
   - rollout dynamics over horizon.

### 10.1 Rollout logic (batched)

At each horizon step:

- sample MOF (`buy_vol`, `sell_vol`),
- compute soft fills:
  - sigmoid(reach / soft_temp) gating by market reach beyond book queue,
- compute `cash_delta`, `inv_delta`,
- evolve synthetic book by **MOF consumption only**,
- compute representative buy/sell VWAP on the post-flow book and add sampled residual noise,
- step reward: change in mark-to-market wealth `W = cash + inventory * sell_ref` (refs in ticks).

Total trajectory reward is terminalized via `terminal_objective(...)`.

### 10.2 Reward modes

Given trajectory terminal `delta_W` and terminal inventory:

- `step_pnl` / `terminal_pnl`: `J = E[delta_W]`
- `terminal_pnl_inventory`: `J = E[delta_W] - gamma * E[q_T^2]`
- `terminal_mean_variance`: `J = E[delta_W] - lambda * Var(delta_W)`
- `terminal_mean_std`: `J = E[delta_W] - lambda * Std(delta_W)`
- `terminal_cara`: `J = -(1/eta) * log(E[exp(-eta*delta_W)])`

The optimizer minimizes `loss = -J`.

---

## 11. Controller Behavior and Safety Gates

`SigmoidController.act(state)`:

1. calls `optimizer.optimize(state)` -> `SigmoidMixtureAction`
2. optional suppression:
   - if `zero_action_if_negative_J_mean=True` and optimizer `_last_E_dW < 0`:
     - return `ZERO_DISCRETIZED_ACTION`
3. convert sigmoid action to discretized ladder with bounds and notional checks.

Separately, runner-level `use_zero_action=True` forces no-quote behavior independent of optimizer.

---

## 12. Live Order Execution Path

When `live_order_execution=True` (typically `execution_mode == "real"`), runner does:

1. `cancel_all_orders(...)`
2. fetch balances
3. convert each ladder level:
   - bid price ticks: `px = round(mid_ticks - spread)`
   - ask price ticks: `px = round(mid_ticks + spread)`
   - `amount_base = qty_lots * lot_size`
   - `price_quote = price_ticks * tick_size`
   - `notional_quote = amount_base * price_quote`
4. pre-check funds:
   - buy requires free quote >= notional
   - sell requires free base >= amount
5. place order via `place_limit_order_lots(...)`

The placement API is model-unit aware and internally converts tick/lot to exchange-native amount/price strings.

---

## 13. Data Histories and Plot Synchronization

Runner stores many histories used by animation:

- actions (`delta_b`, `delta_a`, `q_b`, `q_a`)
- realized and expected flow
- expected mid and decomposition terms
- realized `dW` and expected `E[dW]`
- cumulative PnL components (spread/inventory/cross/fee)
- inventory/cash/wealth
- optional residual diagnostics

Animation calls `advance_scheduled(...)` first, then redraws all windows from these histories so panels remain step-consistent.

---

## 14. End-to-End State Update Summary

For one cycle:

1. stepper returns normalized market + account snapshot in model units,
2. execution applies previous decision to get realized transition,
3. runner updates wealth/inventory/cash and pnl decomposition,
4. estimators/testers update from transition,
5. controller computes next ladder from current state,
6. optional live cancel/replace executes ladder externally,
7. next cycle consumes that action.

This is a delayed-control loop: decision at step `t` is evaluated at step `t+1`.

---

## 15. Known Nuances and Caveats

1. `execution_rule` is instantiated but runner transitions go through `ExecutionModel`.
2. In websocket mode, trade stream has no REST fallback.
3. Real mode still optimizes actions with model-based MPC while state transition is observed balances.
4. Model units are consistent (`ticks`, `lots`, `tick*lot`) but some hints use float/int mixes; keep canonical units explicit in new code.
5. Example `main()` currently contains a literal private token; treat as local sample only, not production-safe practice.

---

## 16. Quick Glossary

- **tick**: discrete price unit (`price_currency / tick_size`)
- **lot**: discrete size unit (`qty_base / lot_size`)
- **tick*lot**: cash unit used by state and PnL
- **MOF**: market order flow (buy/sell taker volume)
- **LOF**: limit order flow (book replenishment/change curves)
- **`V_plus`**: market buy volume (hits ask)
- **`V_minus`**: market sell volume (hits bid)
- **Reach**: market flow remaining after consuming queue ahead
- **`delta_W`**: wealth change over a step or horizon

