# Code-Derived Market Maker System Documentation

This document describes the current low-frequency market-making system from code analysis only. Existing repository documentation was not used as source of truth.

## 1. High-Level Architecture

The system is a Python, asyncio-driven, config-wired market-maker runtime. A single runtime config currently starts one bot with:

- one local quoting venue,
- an optional global/reference or liquidity-source venue,
- public market data adapters,
- optional private state adapters,
- optional order command adapters,
- a ledger-backed strategy frame assembler,
- an MPC controller,
- a target-book risk and execution pipeline,
- optional persistence to Postgres and Parquet,
- runtime observability through Prometheus metrics, JSON logs, optional OTLP traces, and Grafana dashboards.

The main runtime builder is `mm/runtime/engine.py`. The command-line entrypoint is `python -m mm.runtime`, which delegates to `mm.tools.print_events`. With `--config`, that tool loads a JSON runtime config, builds runtime components, prints the resolved runtime contract, starts the runtime, and optionally prints live bus events.

The runtime is built around an `AbstractEventBus` interface:

- `InMemoryBus` provides in-process fan-out through bounded tracked subscriber queues with configurable backpressure (`block`, `drop_oldest`, `drop_newest`).
- `RedisStreamsBus` writes compact event envelopes to Redis Streams, fans out original typed objects to local subscribers, and forwards new stream messages from other bus instances into normal subscribers.

Most runtime components communicate through topic names rather than direct calls. The one major exception is the strategy pipeline, which directly owns the `MpcStrategyEngine`, risk engine, and order-manager map.

## 2. Major Components

### Runtime Wiring

`build_runtime_components()` in `mm/runtime/engine.py` is the composition root. It:

1. Loads `.env` through `mm/env_loader.py`.
2. Configures metrics/logging/tracing with `configure_observability()`.
3. Builds an event bus.
4. Builds an `IntervalScheduler`.
5. Selects public market data adapters per source.
6. Builds one `SymbolEngine` per configured source.
7. Builds private state adapters and reconcilers if credentials are configured.
8. Builds order command adapters and `OrderManager` instances if execution is enabled.
9. Builds the local/global `StrategyInputFrameAssembler`.
10. Builds a `PortfolioLedgerService` when private state is available.
11. Builds a `JointStrategyFrameAssembler`.
12. Builds a `RuntimeControlPlane`.
13. Builds the `StrategyRunner` if the controller is enabled.
14. Builds persistence writers and a runtime observability subscriber.

`RuntimeComponents.start()` starts components in this order: persistence, observability, frame assembler, strategy runner, ledger, control plane, symbol engines, execution reconcilers, private reconcilers, then the interval scheduler. `stop()` cancels and closes them in roughly reverse order.

### Configuration

`mm/runtime/config.py` defines the full runtime config model:

- `MarketDataSourceConfig`: exchange, symbol, transport mode, polling limits, and public topic helpers.
- `PrivateRuntimeConfig`: private adapter mode, token or API-key credentials, polling interval, and fill source.
- `BusRuntimeConfig`: memory or Redis Streams settings.
- `PersistenceRuntimeConfig`: Postgres and Parquet sinks, batch size, flush interval, and topics.
- `ObservabilityRuntimeConfig`: metrics, JSON logs, log path, run id, and OTLP settings.
- `LiquiditySourcingRuntimeConfig`: single venue vs liquidity sourcing mode.
- `PricingRuntimeConfig`: quote-anchor selection and inventory-mark venue selection.
- `ExecutionRuntimeConfig`: per-venue planner, dry-run, max order count, max size, and max quote distance.
- `ControllerRuntimeConfig`: MPC model configs, optimizer config, orderbook grid, decision and placement timeouts, risk, reconciliation, dry-run, warmup, and topics.
- `MarketMakerRuntimeConfig`: the resolved bot runtime.

`runtime_config_from_mapping()` expects one bot per config file. It resolves:

- `bots.<id>` into a quoting exchange, liquidity exchange, symbol, actor id, and strategy preset,
- `strategy_presets.<preset>` into interval timing, controller, pricing, liquidity sourcing, execution, and risk settings,
- local/global sources,
- private exchange blocks,
- optional exchange instrument precision overrides merged with venue trade-stat instrument specs,
- controller payload merged with exchange maker fees and derived pricing reference behavior,
- derived liquidity-sourcing behavior,
- execution and persistence settings.

The config schema is intentionally strict around recent runtime migrations. It rejects older fields instead of silently translating them:

- `controller.reference_price_source` was removed from user config; it is derived from `pricing.inventory_mark.venue`.
- `controller.vwap_depth` was removed; VWAP depth is derived from inventory every frame.
- old controller model keys `price_model` and `market_flow_model` were renamed to `local_price_model` and `local_market_flow_model`.
- global model config belongs directly under `controller`, not under `controller.optimizer`.
- venue optimizer keys belong under `controller.optimizer.local` and `controller.optimizer.global`; old aliases such as `inv_coef`, `target_inventory`, `soft_fill_queue_penalty`, and `spread_structure` are rejected with pointers to replacements.
- flat `book_quality` soft keys (`spread_target`, `depth_target`, `measure`, sub-weights) must live under `book_quality.soft_target`; hard first-touch geometry lives under `book_quality.hard_structure`.
- removed flat quote keys `quote_spread_max`, `quote_spread_min_depth`, `quote_spread_k_epsilon`, `quote_capacity_ratio`, and `quote_imbalance_skew_strength` must not appear under venue optimizer blocks.
- optional `controller.optimizer.objective_normalize` scales the CE return term by an EWMA of `|CE|` so it can share numeric range with `w_book` and `w_inv`.
- `private.enable_execution` was removed; live execution is controlled by the strategy preset's `execution` block and controller gates.
- `execution.liquidity_source` was removed; liquidity-venue execution policy is configured under `execution.liquidity`, while its live-enabled flag is derived from `liquidity_sourcing.mode`.

The example config in `mm/configs/market_maker.runtime.example.json` uses Redis Streams, Postgres or Parquet persistence, Prometheus metrics on port 8000, and an MPC controller.

Pricing is configured in each strategy preset:

- `pricing.quote_anchor.mode` controls where limit-order ladders are centered:
  - `quoting_mid`: best bid/ask mid on the quoting venue.
  - `liquidity_mid`: best bid/ask mid on the liquidity/reference venue, or the local venue when no global source exists.
  - `liquidity_weighted_mid`: computes bid-side and ask-side VWAP windows within `anchor_vwap_bps` on both quoting and liquidity books, weights each side by visible base quantity inside the window, then averages the weighted bid and ask prices.
- `pricing.inventory_mark.venue` controls whether wealth/reference marking uses the quoting venue or liquidity venue. Internally this becomes `controller.reference_price_source = local` or `global`.

The runtime is still one bot per process/config. To scale horizontally, run multiple configs/processes with different bot ids, symbols, or venues.

### Exchange Adapters

`mm/exchanges/registry.py` holds one registration per supported exchange (`nobitex`, `wallex`, `tabdeal`, `bitpin`). Each package exposes adapter builders in its own `registration.py`. `mm/exchanges/factory.py` resolves capabilities, looks up the registration, and constructs:

- public market data: REST or websocket, resolved through exchange capabilities,
- private state: REST for Nobitex and Wallex; REST or websocket for Tabdeal and Bitpin,
- order commands: REST only,
- venue limits: pulled from trade-stats exchange info where registration enables it.

The public market-data adapter contract is `AbstractMarketDataAdapter`:

- `connect()`,
- `disconnect()`,
- `trade_stream()`,
- `orderbook_stream()`.

Shared market-data adapter base classes live in `mm/exchanges/shared/provider_adapter.py`. REST polling adapters fetch trades and orderbooks periodically. Websocket adapters reconnect with exponential backoff and push normalized trade/book events to internal queues.

All strategy-facing prices are quote-per-base floats and all quantities are base-asset floats. Venue API unit conversions are handled by exchange-specific wire adapters and command adapters.

### Ingestion And Aggregation

`SymbolEngine` in `mm/ingestion/symbol_engine.py` owns one exchange/symbol public feed. It:

- connects an adapter,
- starts trade and orderbook feed tasks,
- adds deduplicated trades to `RollingTradeBuffer`,
- applies snapshots or deltas to `OrderbookManager`,
- publishes raw trade/orderbook events,
- publishes adapter and normalization errors,
- registers interval callbacks that build `MarketEvent` objects.

`RollingTradeBuffer` keeps sorted trades in a time-bounded deque and deduplicates by trade id or `(exchange, symbol, timestamp, price, qty, side)`.

`OrderbookManager` materializes snapshots from REST snapshots or websocket deltas. It keeps a bounded history of full snapshots and can return the latest snapshot at a requested timestamp.

`MarketEventBuilder` creates a low-frequency `MarketEvent` at each scheduler interval:

- latest orderbook snapshot at `t_end_ms`,
- trades in `[t_start_ms, t_end_ms)`,
- buy volume,
- sell volume,
- VWAP,
- top-level imbalance,
- microprice.

`IntervalScheduler` aligns callbacks to wall-clock interval boundaries. Due callbacks for one or more intervals run with bounded concurrency (`scheduler.max_concurrency`, default 8). Callback latency and missed-deadline counters are exported when a callback exceeds its interval budget.

### Frame Assembly

`StrategyInputFrameAssembler` consumes local `MarketEvent` objects and optionally global `MarketEvent` objects. For each local event, it:

- drains latest global/private inputs,
- optionally waits up to `global_grace_ms` for a matching global `t_end_ms`,
- selects the latest global event with `t_end_ms <= local.t_end_ms`,
- marks global status as `disabled`, `missing`, `stale`, or `ready`,
- attaches latest private snapshots if queues are configured,
- returns a `StrategyInputFrame`.

`JointStrategyFrameAssembler` wraps this with ledger state. It:

- computes a reference price from local or global microprice,
- asks `PortfolioLedgerService` for a joint ledger snapshot when available,
- computes market/private freshness,
- builds and publishes a `JointStrategyFrame` on `strategy:joint:frames:<exchange>:<symbol>:<interval>`.

### Private State And Ledger

`PrivateStateReconciler` owns authenticated private state ingestion. It:

- obtains an initial private snapshot,
- for REST polling adapters, runs a unified snapshot loop and publishes one authoritative snapshot per poll plus optional fill events,
- for incremental adapters, listens to balance, open-order, order-update, and fill streams,
- publishes all private events to topics under `private:state:<exchange>:...`.

`PrivateStateStore` keeps the normalized private state. It:

- stores balances, open orders, and fills,
- removes terminal or zero-remaining orders,
- preserves zero balances for the trading pair's base and quote assets,
- normalizes locked balances so live open-order collateral is reflected.

`PortfolioLedgerService` subscribes to private topics and maintains `VenueLedgerStateMachine` instances for the local and optional global venue. After each update, it publishes:

- `ledger:snapshot:<exchange>` with a venue payload,
- `ledger:snapshot` with a joint payload,
- optional ledger vs order-manager consistency diagnostics.

When ledger live orders and `OrderManager.live_orders` disagree with the same mismatch for three consecutive diagnostics, the service publishes `LedgerOrderManagerMismatch` on `ledger:diagnostics:mismatch:<exchange>`. This event includes mismatch reason, order id, ledger value, manager value, first-seen time, age, and consecutive count.

The strategy pipeline requires a `JointStrategyFrame` with a ledger. If no ledger exists, `StrategyRunner.on_joint_frame()` raises.

### Strategy Runner

`StrategyRunner` in `mm/application/strategy_runner.py` is the main decision loop. For each `JointStrategyFrame`, it:

1. Ensures an input frame and ledger are present.
2. Increments `frame_step`.
3. Syncs the runtime control-plane kill switch into the risk engine.
4. Extracts live orders from the ledger.
5. Produces a clean frame by subtracting own live orders from local and global orderbooks using tolerance-aware price matching.
6. Builds `TradingState` from the local ledger.
7. Creates a `DecisionContext` for diagnostics. The context is the final assembler for `ControllerDiagnostics`, with timing, risk, execution-gate, and action-pipeline facts held in narrower diagnostic state records. The runner records lifecycle facts through those state records rather than broad compatibility aliases on the context.
8. Runs `MpcStrategyEngine.decide()` in a worker thread under `decision_timeout_ms`.
9. Prepares the execution target book according to per-venue execution settings.
10. Computes venue execution gates.
11. Runs target-book risk validation.
12. Applies risk output to diagnostics.
13. Publishes `StrategyDecision`, target-book, and approved `Action` events.
14. Optionally syncs live orders through order managers with a timeout.
15. Publishes a compact journal event.
16. Always publishes `ControllerDiagnostics` in `finally`.

When decision compute exceeds `decision_timeout_ms`, the runner leaves the optimizer task running in the worker thread, publishes a `ControllerFallback` event with `fallback_policy="hold_current_orders"`, and does not publish a new target book or approved action for that frame. If the next frame arrives before the previous timed-out optimizer call completes, the runner publishes the same hold-current-orders fallback for the new frame. Existing exchange orders are left unchanged in both cases.

For each processed frame, the runner assigns deterministic lineage IDs: `frame_id`, `decision_id`, `target_book_id`, and per-venue `execution_plan_id`. `StrategyDecision`, `ControllerDiagnostics`, `JointTargetBook`, venue target-book payloads, execution sync events, and router request/accepted/failed events carry these IDs. Execution lifecycle, snapshot, strategy-decision, controller-diagnostics, private-snapshot, ledger-snapshot, and market-event bus payloads are built only through versioned helpers in `mm/core/payload_schema.py` (`mm.execution_lifecycle` v1, `mm.execution_snapshot` v1, `mm.strategy_decision` v1, `mm.controller_diagnostics` v1, `mm.private_state_snapshot` v1, `mm.venue_ledger_snapshot` v1, `mm.joint_ledger_snapshot` v1, and `mm.market_event` v1). `SymbolEngine` publishes versioned market-event payloads; `PrivateStateReconciler` publishes private snapshots as versioned dict payloads; `PortfolioLedgerService` publishes venue and joint ledger snapshots the same way. `StrategyInputFrameAssembler` parses versioned market-event payloads back into domain structs at the queue boundary. This lets replay and incident analysis follow frame -> decision -> target book -> execution without inferring relationships from timestamp/topic proximity.

`MpcStrategyEngine` depends on a strategy-level optimizer backend protocol rather than the concrete tensor optimizer directly. The default backend wraps the existing `MonteCarloMPCTensorBatched` controller, while tests or future worker-process implementations can inject another backend that provides `optimize(state)`, model handles, optimizer diagnostics, and optional global sigmoid action.

Live execution is blocked by:

- controller dry-run,
- control-plane dry-run,
- warmup steps,
- paused actors,
- paused venue roles,
- missing order manager,
- global live-orders disabled,
- stale global market data,
- no orders unless a replace policy allows empty-book sync,
- risk rejection,
- execution timeout or sync error.

### Control Plane

`RuntimeControlPlane` consumes `admin:commands` and maintains runtime safety state:

- pause/resume actor,
- kill switch,
- dry-run mode,
- pause/resume local or global venue roles,
- request cancel actor orders,
- request cancel all.

It publishes state to `admin:events` and updates Prometheus gauges for kill switch and dry-run state.

`mm/api/admin.py` exposes a FastAPI app with health, status, config, metrics, and command endpoints. The admin app creates its own bus from environment unless injected.

## 3. Core Market-Making Engine

### State Construction

`MpcStrategyEngine.decide()` converts a `StrategyInputFrame` and ledger into an MPC `State`:

- a quote anchor from the configured pricing mode,
- local orderbook depth relative to that anchor, clipped by `spread_max`,
- local buy/sell volume,
- local inventory and cash,
- local buy/sell VWAP for a dynamic depth,
- optional global mid/orderbook/volume/VWAP,
- optional global inventory and cash,
- an inventory mark/reference price selected from local or global inventory-depth VWAPs.

The quote anchor is computed by `mm/pricing/anchor.py`:

- `quoting_mid` requires a two-sided local book and uses local best bid/ask mid.
- `liquidity_mid` requires a two-sided liquidity/reference book and uses that best bid/ask mid.
- `liquidity_weighted_mid` builds bid and ask VWAP windows on both books within `anchor_vwap_bps`, weights each side by visible quantity, and uses the average of the weighted bid and weighted ask.

When a distinct global/liquidity venue is not configured, the local market event is also used as the liquidity event.

VWAP depth is derived from inventory:

- single venue: `abs(local_inventory)`,
- liquidity sourcing: `abs(local_inventory + global_inventory)`.

The mark price is side-aware: long inventory is marked at sell-side VWAP, short inventory is marked at buy-side VWAP, and flat inventory is marked at the quote anchor. The user-facing `pricing.inventory_mark.venue` chooses whether those VWAPs come from the quoting venue or the liquidity venue when a distinct liquidity venue exists.

If no ledger snapshot is fresh, state construction falls back to private snapshots attached to the frame. In the normal runtime path, the strategy runner requires a ledger-backed frame.

The state object passed to the optimizer is `State` in `mm/mpc/mpc_state.py`. Relevant fields are:

- `mid`: local clean-book mid around the selected quote anchor.
- `ob_rel`: local orderbook as `(spread, qty)` levels, where bid spread is `mid - bid_price` and ask spread is `ask_price - mid`.
- `buy_vol`, `sell_vol`: local interval market buy/sell flow.
- `inventory`, `cash`: local venue account state.
- `buy_vwap`, `sell_vwap`, `vwap_depth`: local liquidation/reference VWAPs at the inventory-derived depth.
- `global_mid`, `global_ob_rel`, `global_buy_vol`, `global_sell_vol`, `global_inventory`, `global_cash`, `global_buy_vwap`, `global_sell_vwap`: corresponding liquidity/global venue fields when present.
- `reference_price`: selected inventory mark at state construction time.
- `quote_anchor_price`, `inventory_mark_price`, `global_active`: diagnostics and rollout flags.

### Online Models

The controller owns separate local and global flow/price models:

- Local market flow: gamma, Weibull, hurdle gamma, hurdle Weibull, or hurdle lognormal.
- Global market flow: same model family.
- Local price: residual VWAP model or conditional mechanical mixture model.
- Global price: EWMA Gaussian VWAP delta model, static Gaussian deltas, residual model, conditional mechanical mixture model, or disabled.

Model scoring happens before updates. `score_frame_models()` computes likelihood or NLL diagnostics against pre-update parameters. After scoring:

- global VWAP price model updates from observed global buy/sell VWAPs,
- local/global market-flow models update from interval volumes,
- local residual price model updates from the previous/current local frame,
- global residual price model updates from previous/current global frame when configured.

### Market Order-Flow Models

Market order flow is sampled independently per side and per horizon step through the configured model's `.sample((n_samples, horizon))` method. The optimizer treats sampled `buy_vol` as market buys that consume ask-side depth and sampled `sell_vol` as market sells that consume bid-side depth.

Supported model families in `mm/mpc/market_order_flow_torch.py` are:

- `gamma`: EWMA estimates positive-volume Gamma parameters by side. For side `s in {buy, sell}`, positive volume is sampled as `V_s ~ Gamma(shape_s, scale_s)`, clamped to a small positive value in the tensor path. EWMA stores `S_s`, `S2_s`, and `W_s`; `mean_s = S_s / W_s`, `var_s = S2_s / W_s - mean_s^2`, `shape_s = mean_s^2 / var_s`, `scale_s = var_s / mean_s`.
- `weibull`: EWMA estimates side-specific mean/variance, converts coefficient of variation to Weibull shape `k_s`, then sets `lambda_s = mean_s / Gamma(1 + 1/k_s)`. Tensor sampling uses `V_s ~ Weibull(k_s, lambda_s)`.
- `hurdle_gamma`: EWMA estimates `p0_s = P(V_s = 0)` from zero observations and Gamma parameters from positive observations. Tensor sampling draws a Bernoulli zero event, otherwise samples Gamma.
- `hurdle_weibull`: same hurdle structure, with positive volumes sampled from Weibull.
- `hurdle_lognormal`: EWMA estimates zero mass and log-positive-volume moments; tensor sampling draws zero with probability `p0_s`, otherwise samples `exp(N(mu_s, sigma_s^2))`.

### Price And Reference Models

The optimizer forecasts side-specific VWAP/reference prices used to mark inventory during simulated rollouts.

The `residual` price model (`TorchResidualPriceModel`) learns Gaussian residuals around a market-order-flow-only VWAP transition:

```text
e_buy,t  = observed_buy_vwap_t  - mechanical_buy_vwap_t
e_sell,t = observed_sell_vwap_t - mechanical_sell_vwap_t
mu_side,t     = gamma * mu_side,t-1 + (1 - gamma) * e_side,t
sigma2_side,t = gamma * sigma2_side,t-1 + (1 - gamma) * (e_side,t - mu_side,t)^2
```

During rollout it samples independent residuals:

```text
epsilon_buy  ~ N(mu_buy,  sigma_buy^2)
epsilon_sell ~ N(mu_sell, sigma_sell^2)
buy_vwap_rollout  = mechanical_buy_vwap  + epsilon_buy
sell_vwap_rollout = mechanical_sell_vwap + epsilon_sell
```

The `conditional_mechanical_mixture` model (`TorchConditionalMechanicalMixturePriceModel`) learns fixed-depth book-state transitions. For a reference depth `V_ref`, each observed transition is represented by four components:

```text
mid_delta         = mid_obs - mid_prev
log_spread_delta = log(spread_obs) - log(spread_prev)
log_buy_shape    = log(buy_shape_obs) - log(buy_shape_prev)
log_sell_shape   = log(sell_shape_obs) - log(sell_shape_prev)

buy_shape  = buy_vwap  - mid - 0.5 * spread
sell_shape = mid - sell_vwap - 0.5 * spread
```

For each component, the model maintains a market-mechanical regression branch and a refresh branch:

```text
q_market = sigmoid(beta_gate[0] + beta_gate[1] * min(log1p(total_flow), 12))
market_mean_component  = beta_component[0] + beta_component[1] * mechanical_component
refresh_mean_component = EWMA refresh mean
```

The market branch uses Gaussian innovations; the refresh branch uses Student-t innovations. Posterior responsibility from the observed transition updates the logistic gate, weighted RLS market regressions, refresh means, and branch variances. During rollout, the model samples either branch, reconstructs predicted mid/spread/shape, and converts fixed-depth shape to dynamic-depth VWAPs:

```text
buy_vwap  = pred_mid + 0.5 * pred_spread + pred_buy_shape  * ratio(buy_depth, V_ref)
sell_vwap = pred_mid - 0.5 * pred_spread - pred_sell_shape * ratio(sell_depth, V_ref)
```

The depth-shape ratio is a shifted power curve controlled by `shape_alpha`; it is zero at zero depth, one near `reference_vwap_vol`, and has finite derivative at zero for autograd stability.

Global price models may be `ewma_gaussian`, static `gaussian`, `residual`, `conditional_mechanical_mixture`, or disabled. Current joint-rollout code consumes global models that expose `sample_innovations()`, `sample_residual()`, or `sample_dynamic_vwap()`. The EWMA/static Gaussian global models expose `sample_deltas()` for diagnostics/modeling, but `joint_optimizer.py` does not currently inject those deltas into rollout VWAPs. When `pricing.inventory_mark.venue` maps to global, the rollout marks consolidated inventory using global side-aware VWAPs. Otherwise it uses local VWAPs.

### Action Parameterization

The optimizer does not optimize discrete exchange orders directly. It optimizes continuous cumulative quote-depth curves. Later, the first-step continuous action is discretized into target orders.

Each venue has one bid curve and one ask curve. For side `s in {bid, ask}`, cumulative raw depth at spread `x >= 0` is a mixture of sigmoids:

```text
D_s_raw(x) = sum_{i=1..n} L_{s,i} * sigmoid(k_{s,i} * (x - x0_{s,i}))
sigmoid(z) = 1 / (1 + exp(-z))
D_s(x) = max(0, D_s_raw(x) - D_s_raw(0))
```

The raw optimizer vector length per venue is `6 * n_components + 3`: bid `(L_raw, k_raw, x0_raw)` components, ask components, one total-depth logit per side, and one reservation-offset logit `u_r`. The joint optimizer concatenates local and global venue vectors for every horizon step:

```text
x = [a_local,0, a_global,0, a_local,1, a_global,1, ..., a_local,H-1, a_global,H-1]
```

`LParamConfig` converts raw parameters into effective constrained parameters:

- `k_eff = softplus(k_raw) + epsilon`, except the first spread-structured leg has an additional minimum slope.
- Without spread-structured constraints, centers are `x0_eff = spread_max * sigmoid(x0_raw)`.
- Total per-side depth is selected by `Q_s_total = Q_s_min + sigmoid(u_s_depth) * (Q_s_max - Q_s_min)`.
- Component weights for `L_eff` use a softmax over `L_raw`, normalized so `D_s(spread_max) = Q_s_total`.
- When `reservation_offset.r_max > 0`, the decoder exposes `r_offset = r_max * tanh(u_r)` and evaluates curves on the market spread grid with `mid_eff = mid * (1 + r_offset)`. Bid depth uses `D_bid(x + mid * r_offset)` and ask depth uses `D_ask(x - mid * r_offset)`; target-book discretization still uses the true quote anchor mid.

When `book_quality.hard_structure.action_max_spread` is configured (mapped internally to `quote_spread_max`), a spread-structured parameterization is used:

```text
scale = sigmoid(u_scale)
[x0_bid,1, x0_ask,1] = scale * action_max_spread * softmax([u_bid, u_ask])
x0_bid,1 + x0_ask,1 <= action_max_spread
```

The first component slope is forced high enough that `sigmoid(-k_1 * x0_1) <= action_k_epsilon`; the first leg carries at least `action_min_depth` at its center when configured; tail components fill the remaining total depth up to `Q_s_total` at `spread_max`. Spread structure requires `n_components >= 2`. If `hard_structure` is omitted, the venue decodes in unstructured mode with per-side centers in `[0, spread_max]`.

Example venue optimizer blocks in `mm/configs/market_maker.runtime.example.json`:

```json
"book_quality": {
  "hard_structure": {
    "action_max_spread": 100.0,
    "action_min_depth": 0.3,
    "action_k_epsilon": 0.01
  },
  "soft_target": {
    "spread_target": 100.0,
    "depth_target": 1.0,
    "measure": "quoted"
  }
},
"reservation_offset": { "r_max": 0.002 },
"w_book": 0.002,
"w_inv": 0.01,
"inventory_zone": {
  "comfort_low": -0.2,
  "comfort_high": 0.2,
  "low_ramp_width": 0.05,
  "high_ramp_width": 0.05,
  "low_penalty_cap": 0.7,
  "high_penalty_cap": 1.0
}
```

Local and global venues can use different `book_quality`, `w_book`, `w_inv`, `inventory_zone`, and `reservation_offset` settings. The example config keeps hard structure on global only and soft targets on both venues.

### Quote Capacity And Structural Constraints

At each rollout step and for each venue, dynamic side capacity is computed from simulated account state and optional per-venue caps:

```text
bid_max = min(quote_size_max_per_side, cash / mid)   when quote_size_max_per_side is set
bid_max = cash / mid                                   otherwise
ask_max = min(quote_size_max_per_side, inventory)    when quote_size_max_per_side is set
ask_max = inventory                                    otherwise
bid_min = ask_min = 0
```

Side mins stay at zero; two-sided presence is encouraged through the optional book-quality term rather than hard quote floors. A venue with no quote cash has `bid_max = 0`, and a venue with no base inventory has `ask_max = 0`. Raw optimizer variables are box-constrained before each Adam closure:

```text
logit-like params: [-30, 30]
k-like params:     [1e-6, 100]
x0-like params:    [-spread_max, spread_max]
```

The last three raw slots per venue (bid depth logit, ask depth logit, reservation logit) use logit bounds.

### Fill Simulation

`soft_fill_batched()` simulates the combined market book plus the strategy's quote depth. It converts current cumulative market depth into marginal levels, converts quoted cumulative sigmoid depth into marginal quote levels (using the same reservation-offset curve shift as decoding when configured), combines public and own quote levels, applies market order flow through a differentiable fill fraction, splits consumed volume between public book and own quotes, and returns the next market-only book plus cash and inventory deltas from own fills.

For each side and spread-grid level:

```text
combined_level = public_level + own_quote_level
combined_cum   = cumsum(combined_level)
reach_raw      = market_order_volume - combined_cum
reach          = normalized(reach_raw)
fill_fraction  = sigmoid(reach / soft_fill_temp)
```

If `soft_fill_activation = hardsigmoid`, the activation is `hardsigmoid(reach / soft_fill_temp)`, which can produce exactly zero fills for far unreached levels.

The queue split is:

```text
own_share = own_quote_level / (combined_level + ahead_public_depth)
own_fill_level = consumed_level * own_share
```

For `pro_rata`, `ahead_public_depth = 0`. For `back_of_queue`, `ahead_public_depth` is the next worse public level. Cash/inventory deltas from own fills are:

```text
cash_bid = sum(fill_bid_level * (mid - spread))
cash_ask = sum(fill_ask_level * (mid + spread))
cash_delta = cash_ask * (1 - maker_fee) - cash_bid
inventory_delta = fill_bid * (1 - maker_fee) - fill_ask
```

### MPC Optimization

The optimizer is `MonteCarloMPCTensorBatched` in `mm/mpc/mpc_tensor_batched.py`. It is a differentiable Monte Carlo MPC optimizer using PyTorch and Adam. `MpcStrategyEngine` builds it from `controller.optimizer` through `_build_controller()` in `mm/strategies/mpc_engine.py`.

Venue-scoped optimizer parameters are parsed into `VenueOptimizerParams` (`mm/mpc/mpc_tensor_batched.py`) for `controller.optimizer.local` and `controller.optimizer.global`:

| Key | Role |
|-----|------|
| `n_components` | Sigmoid mixture count per side |
| `return_risk_aversion_beta` | Entropic CE risk aversion `beta` |
| `objective_weight` | Parsed and stored per venue; not currently multiplied into the terminal CE in rollout code |
| `quote_size_min_per_side` | Lot-size floor used in spread-structured depth decoding |
| `quote_size_max_per_side` | Optional hard cap on cumulative quote depth per side |
| `book_quality.hard_structure` | First-touch spread cap / min depth / slope epsilon |
| `book_quality.soft_target` | Book-quality score targets and sub-weights |
| `reservation_offset.r_max` | Bounds relative reservation shift `r_offset` |
| `w_book` | Weight on h=0 book-quality score |
| `w_inv` | Weight on trajectory inventory-zone penalty |
| `inventory_zone` | Sigmoid comfort band on wallet imbalance |
| `soft_fill_temp`, `soft_fill_activation`, `soft_fill_queue_model` | Differentiable fill model |

Optimizer-wide settings live directly under `controller.optimizer`:

| Key | Role |
|-----|------|
| `kind` | `monte_carlo_mpc_tensor_batched` |
| `n_samples`, `horizon`, `adam_epochs`, `adam_lr`, `device` | Monte Carlo and Adam loop |
| `warm_start`, `fixed_mc_samples` | Action warm start and fixed sample reuse |
| `objective_normalize.enabled` | Scale CE by EWMA(`|CE|`) during optimization |
| `objective_normalize.ce_ewma_decay` | EWMA decay in `[0, 1)`; higher = slower adaptation |

For each optimize call, the controller converts orderbooks into spread-grid tensors, computes dynamic capacity bounds, draws Monte Carlo market-flow and price innovations, initializes or warm-starts the horizon action vector, clamps raw parameters, rolls out every path over the horizon, computes normalized returns, maximizes the terminal objective for `adam_epochs`, keeps the best finite epoch, updates the CE EWMA scale from the best epoch's raw CE, and returns the best first-step local action. If joint/global mode is active, it also stores the best global action for target-book construction and diagnostics.

Default `_build_controller()` optimizer settings are `n_samples = 8`, `horizon = 2`, `adam_epochs = 10`, CPU device, and spread grid from `ControllerRuntimeConfig.orderbook_grid`.

### Single-Venue Rollout And Objective

When liquidity sourcing is disabled and global reference pricing is inactive, `MonteCarloMPCTensorBatched.optimize_with_diagnostics()` uses the single-venue rollout.

For Monte Carlo path `i` and horizon step `h`, `soft_fill_batched()` produces:

```text
Q_{i,h+1} = Q_{i,h} + inventory_delta_{i,h}
X_{i,h+1} = X_{i,h} + cash_delta_{i,h}
book_{i,h+1} = market-only public book after market-flow consumption
```

Single-venue wealth uses side-aware liquidation marking:

```text
mark(Q) =
  buy_vwap  if Q < 0
  sell_vwap if Q > 0
  mid       if Q = 0

W_h = X_h + Q_h * mark(Q_h)
R_i = (sum_{h=0..H-1} Delta W_{i,h}) / W_0
```

The terminal objective in `mm/mpc/reward_tensor.py` starts from the entropic certainty equivalent:

```text
CE_beta(R) = min(R) - (1 / beta) * log(mean_i(exp(-beta * (R_i - min(R)))))
```

The `min(R)` shift is numerical stabilization and is equivalent to `-(1 / beta) * log(mean_i(exp(-beta * R_i)))` for positive `beta`.

Optional soft terms from `mm/mpc/objective_terms.py` are added after CE computation:

```text
J = CE_beta(R) + w_book * B - w_inv * Z
```

- `B` is the h=0 book-quality score in `[0, 1]` from `book_quality_score()`.
- `Z` is the mean trajectory inventory-zone penalty in `[0, 1]` from `inventory_zone_path_mean()`.

When `controller.optimizer.objective_normalize.enabled = true`, only the CE term is rescaled before the soft terms are added:

```text
CE_norm = CE / scale
scale   = EWMA(|CE|) from previous frames, or |CE| on the first frame
J       = CE_norm + w_book * B - w_inv * Z
```

During a single optimize call, `scale` is fixed from the previous frame. After the best epoch is chosen, the controller updates the EWMA from that epoch's raw CE (`risk_adjusted_return_raw` in diagnostics). Book and inventory terms are not EWMA-normalized. Within one optimize call, dividing CE by a positive constant preserves the ranking of actions by return; the main effect of normalization is to keep CE on a comparable numeric scale to `w_book` and `w_inv` across regimes.

Book-quality scoring (`book_quality_score()`) evaluates the decoded h=0 action:

- `measure = quoted`: score only the strategy's decoded quote depth.
- `measure = combined`: add resting public depth at the touch and full cumulative depth when scoring presence/uniformity.

Sub-scores are presence, tightness (from first-component spread sum vs `spread_target`), balance (bid/ask notional symmetry), and uniformity (marginal depth evenness). Sub-weights default to `0.25` each under `book_quality.soft_target`.

Inventory-zone scoring uses wallet imbalance:

```text
imb = (Q * mid - cash) / (|Q * mid| + |cash| + eps)
Z   = (a_lo * sigmoid((comfort_low - imb) / d_lo)
       + a_hi * sigmoid((imb - comfort_high) / d_hi)) / (a_lo + a_hi)
```

Diagnostics expose `risk_adjusted_return`, optional `risk_adjusted_return_raw`, `objective_ce_ewma_scale`, `book_quality`, and `inventory_zone_penalty` when the corresponding weights are non-zero. Adam minimizes `loss = -J`.

### Joint Local/Global Rollout And Objective

If liquidity sourcing is enabled, or if global reference pricing is active with available global VWAPs, the optimizer delegates to `mm/mpc/joint_optimizer.py`.

The joint optimizer rolls out four curves: local bid, local ask, global bid, and global ask. It maintains separate local and global account states:

```text
Q^L_h, X^L_h = local inventory and cash
Q^G_h, X^G_h = global inventory and cash
```

The dynamic VWAP depth in joint mode is:

```text
vwap_depth_h = abs(Q^L_h + Q^G_h)  when liquidity_sourcing_enabled
vwap_depth_h = abs(Q^L_h)          otherwise
```

The reference price marks consolidated net inventory:

```text
Q_net = Q^L + Q^G

ref(Q_net) =
  buy_vwap_source  if Q_net < 0
  sell_vwap_source if Q_net > 0
  flat_mid_source  if Q_net = 0
```

`source` is global when `reference_price_source = global` and valid global VWAPs exist; otherwise it is local. Consolidated wealth is:

```text
W_h = X^L_h + X^G_h + (Q^L_h + Q^G_h) * ref_h
```

The objective uses venue-scoped wealth contributions, normalized by initial consolidated wealth:

```text
Delta W^L_h = X^L_{h+1} + Q^L_{h+1} * ref_{h+1}
              - (X^L_h + Q^L_h * ref_h)

Delta W^G_h = X^G_{h+1} + Q^G_{h+1} * ref_{h+1}
              - (X^G_h + Q^G_h * ref_h)

R^L_i = (sum_h Delta W^L_{i,h}) / W_0
R^G_i = (sum_h Delta W^G_{i,h}) / W_0
```

The joint objective in `mm/mpc/joint_mpc.py` optimizes consolidated portfolio return:

```text
R_i = (sum_h Delta W_{i,h}) / W_0
J   = CE_{beta_L}(R)
```

where `Delta W` is the sum of local and global wealth increments under the shared reference mark. `joint_terminal_objective()` also reports `local_risk_adjusted_return` and `global_risk_adjusted_return` as diagnostic CE components on venue-scoped return paths, but those venue CE values are not mixed into `J`.

After CE computation, joint mode applies the same optional normalization and soft terms as single-venue mode, separately per venue where configured:

- local `w_book` / `w_inv` / `inventory_zone` on local h=0 action and local inventory/cash paths,
- global `w_book` / `w_inv` / `inventory_zone` on global h=0 action and global paths when the global venue is active.

Diagnostics report local/global risk-adjusted return components, expected total reward/return, expected next-step reward, fill means, wealth contribution means, book-quality and inventory-zone components, reservation offset when enabled, and raw local/global action vectors.

Inventory risk is represented through side-aware liquidation marking, wallet-based quote capacity caps, optional inventory-zone penalty, reservation-price shifting, and downstream risk validation. Current code does not include a hard terminal-inventory constraint, cash-threshold penalty, or drawdown constraint inside the terminal objective.

### Quote Discretization

`MpcStrategyEngine.decide()` converts the best continuous action to a target book through `build_joint_target_book_from_sigmoid()`:

1. Build a diagnostic continuous depth series.
2. Evaluate cumulative depth at each configured spread ladder point.
3. Convert cumulative depth to marginal order sizes.
4. Build local bid orders at `quote_anchor - spread` and ask orders at `quote_anchor + spread`.
5. Drop zero marginal quantities and invalid prices.
6. Cap to `max_orders_per_side` by keeping the largest-quantity orders per side.
7. Build `VenueTargetBook` and `JointTargetBook`.

When liquidity sourcing is active, the joint optimizer may also expose a global sigmoid action. If present, the target-book builder discretizes the global action on the global venue's own market event and appends a global `VenueTargetBook` with its configured policy. Later, `prepare_execution_joint_book()` removes global orders unless `liquidity_sourcing.global_live_orders` is true, applies per-venue execution caps, and sets the final planner policy used by live routing.

The strategy-side order pipeline intentionally does not perform final venue rounding in the current path. The execution boundary performs venue limit normalization and validation through `ExchangeRulesGateway`.

## 4. Risk Management

The main runtime builds `NetDeltaRiskEngine` when risk is enabled. It validates the approved target-book transition against ledger state.

Frame-level gates reject all orders when:

- kill switch is enabled,
- local market data is stale,
- local private state is missing or stale,
- global market data is missing or stale when global live execution requires it,
- global private state is missing or stale when required.

Venue transition checks evaluate:

- actor live-order collateral,
- target locked base and quote,
- open notional,
- available base and quote,
- wallet totals,
- per-order max notional,
- max open notional,
- sufficient base for asks,
- sufficient quote for bids,
- min cash,
- min inventory,
- max inventory.

For replace policies, the engine assumes existing actor or unattributed scoped orders will be released. For reconcile policies, it uses available balances plus the actor's already locked collateral.

Risk may approve a partial target book by dropping only the rejected side when the rejection reason maps cleanly to buy-side or sell-side constraints.

## 5. Execution Workflow

Execution is target-state reconciliation, not fire-and-forget order placement.

`OrderManager` is a facade over:

- an execution planner,
- an exchange rules gateway,
- a stateless order router,
- private-confirmed live-order state,
- pending place/cancel changes accepted by the router but not yet confirmed by private state.

For each venue target book:

1. Select planner by policy:
   - `reconcile`: keep live orders that match side, symbol, price tolerance, and quantity tolerance; cancel unmatched live orders; place unmatched targets.
   - `replace`: cancel all scoped live orders, including unattributed orders, then place the full target book.
2. Normalize and validate place commands through venue limits.
3. Execute cancels before places.
4. Publish requested/accepted/failed lifecycle events.
5. Record successful router responses as pending places/cancels without mutating confirmed live orders.
6. Hold additional syncs for the same venue/symbol/actor while pending confirmations exist, avoiding duplicate replacement while private state catches up.
7. Publish execution snapshot with confirmed live-order count, pending place/cancel count, and oldest pending age.

The authoritative private-state path then reconciles exchange-confirmed state back into the order manager through `ExecutionStateReconciler` and into the ledger through `PortfolioLedgerService`. A private open-order event confirms a pending place and moves it into live orders; terminal private updates or snapshots remove pending cancels once the order disappears or becomes terminal.

## 6. Event Model And Data Lifecycle

### Event Envelope

Durable events use `EventEnvelope` from `mm/core/events.py`:

- schema and version,
- event id,
- run id,
- source,
- topic,
- event type,
- exchange,
- symbol,
- actor id,
- timestamp,
- JSON-compatible payload.

`envelope_event()` extracts top-level metadata from payload fields when possible.

### Core Topics

Representative runtime topics:

- `market:raw:<exchange>:<symbol>:trades`
- `market:raw:<exchange>:<symbol>:orderbook`
- `market:errors:<exchange>:<symbol>`
- `market:events:<exchange>:<symbol>:<interval_ms>`
- `private:state:<exchange>:snapshot`
- `private:state:<exchange>:balances` for incremental adapters
- `private:state:<exchange>:open_orders` for incremental adapters
- `private:state:<exchange>:order_updates`
- `private:state:<exchange>:fills`
- `private:state:<exchange>:errors`
- `ledger:snapshot:<exchange>`
- `ledger:snapshot`
- `strategy:joint:frames:<exchange>:<symbol>:<interval_ms>`
- `strategy:decisions`
- `strategy:actions`
- `strategy:diagnostics:<actor_id>`
- `strategy:target_books:<actor_id>`
- `risk:events`
- `execution:orders:<exchange>`
- `execution:state:<exchange>`
- `execution:errors`
- `admin:commands`
- `admin:events`

### Default Persistence Topics

`MarketMakerRuntimeConfig.default_persistence_topics` persists:

- local and global market events,
- joint frames,
- ledger snapshots,
- strategy actions,
- execution errors,
- execution orders and state per exchange,
- private snapshots/fills/errors per exchange,
- persistent ledger/order-manager mismatch diagnostics per exchange,
- decisions, diagnostics, and target books when the controller is configured.

Raw market-data topics are not in the default persistence topic list, but the observability subscriber watches raw topics for metrics.

### Persistence

`PostgresEventWriter` subscribes to configured topics and writes compacted event envelopes to:

- `mm_events`: append-only event table,
- `mm_market_facts`: typed market interval facts for fast dashboards and analytics,
- `mm_action_stage_facts`: typed controller action-pipeline stage facts,
- `mm_execution_facts`: typed order execution lifecycle facts,
- `mm_target_order_facts`: typed target-book intent orders,
- `mm_model_score_facts`: typed model score and update-skip facts,
- `mm_optimizer_facts`: typed optimizer timing and objective facts,
- `mm_optimizer_component_facts`: typed optimizer objective component facts,
- `mm_latest_state`: latest state table for private and execution state.

It compacts persisted market events by dropping raw trade lists after interval summaries are present, compacts persisted joint frames by replacing the embedded ledger with a summary plus `ledger:snapshot` reference, and compacts persisted controller-diagnostics action-pipeline order/rejection arrays into counts, side counts, digests, and small samples. Private snapshots remain full in Postgres so `mm_latest_state` can preserve current-state truth. It also materializes every `MarketEvent` into `mm_market_facts` with typed columns for top-of-book prices and quantities, mid, spread, microprice, VWAP, interval volumes, imbalance, trade count, level counts, and event timestamps. Every `ControllerDiagnostics.action_pipeline` stage is materialized into `mm_action_stage_facts` with typed columns for stage counts, side counts, quantities, notionals, rejection counts, compacted counts, and order/rejection digests. Execution sync, place, cancel, validation rejection, fill, and execution snapshot events are materialized into `mm_execution_facts` with typed columns for lifecycle, success, confirmed/pending state kind, order ids, side, price, quantity, remaining/filled quantity, fee, status, error, latency, count, and lineage IDs. Every `JointTargetBook` order is materialized into `mm_target_order_facts` with typed columns for actor, mode, venue role, exchange, symbol, policy, side, price, quantity, notional, order type, tags, and lineage IDs. Controller model scores are materialized into `mm_model_score_facts` with model, kind, update status, skip reason, likelihood/NLL/residual, and small shape counts for observation and parameter dictionaries, and lineage IDs. Optimizer diagnostics are materialized into `mm_optimizer_facts` and `mm_optimizer_component_facts` with timing, objective values, improvement, best expected reward/return fields, gradient/rollout counts, and per-snapshot objective components, and lineage IDs. It creates indexes on run/topic/timestamp, exchange/symbol/timestamp, topic/type/timestamp, actor/timestamp, market-fact exchange/symbol/timestamp, action-stage actor/exchange/symbol/stage/timestamp, execution action/order/timestamp, target-order actor/venue/exchange/symbol/timestamp, model actor/exchange/symbol/timestamp, and optimizer actor/exchange/symbol/timestamp. Batches are written in a worker thread through a reusable psycopg2 connection. Failed writes drop the connection, retry with bounded backoff, and spill exhausted batches to local JSONL files under `persistence_spool/postgres`.

Failed Postgres spool files can be reingested with `python -m mm.tools.reingest_postgres_spool --database-url "$MM_DATABASE_URL"`. The tool scans failed JSONL spool records, decodes their original envelopes, writes them through the same Postgres writer path, and archives successfully reingested files. Inserts remain idempotent because the base event table uses `event_id` conflict handling and the typed fact tables use matching conflict keys.

Retention and tiering live in `mm/persistence/retention.py` and are configured under `persistence.retention` in the runtime JSON. Default tiers are: `mm_events` 14 days, typed fact tables 90 days, `mm_latest_state` retained until explicitly configured, `admin_audit_log` 365 days, Parquet `event_date=*` partitions 90 days, reingested spool archives 7 days, and failed spool JSONL files 30 days. Apply them with `python -m mm.tools.apply_retention --config mm/configs/market_maker.runtime.example.json --dry-run` or by passing `--database-url`, `--parquet-root`, and `--spool-dir` directly. Postgres cleanup deletes rows older than each tier cutoff by `ts` or `updated_at`. Parquet cleanup removes whole date partitions. Spool cleanup deletes aged `.jsonl` files by file mtime.

`ParquetEventWriter` writes partitioned Parquet rows under:

```
event_date=<yyyy-mm-dd>/exchange=<exchange>/symbol=<symbol>/event_type=<event_type>/hour=<hh>/part-<timestamp>-<uuid>.parquet
```

Each flush writes immutable part files; live writes do not read or rewrite existing Parquet files. Parquet rows retain one compact `envelope_json` for replay plus typed nullable columns for high-value analytics fields. The writer does not store a separate `payload_json` copy. For archival storage, it compacts `MarketEvent.trades` after interval features are already present, compacts joint-frame embedded ledgers to summaries/references, compacts controller-diagnostics action-pipeline order/rejection arrays, and removes zero private balances while preserving counts and compaction flags.

Typed Parquet columns cover market top-of-book/depth/flow facts, controller state/model-count/optimizer-objective facts, target-book order counts and notionals, and execution lifecycle/order fields. The replay path still reads `envelope_json`, while offline scans can filter and aggregate common values without parsing JSON. This makes Parquet suitable for compact replay and offline analysis of summarized facts, not exact raw trade-list recovery.

If pyarrow is unavailable, it falls back to JSONL files in the same partition layout.

`mm/persistence/replay.py` can read Parquet or JSONL rows back into event envelopes and group replayed events by explicit lineage IDs. `read_lineage_chains()` returns one chain per `frame_id`, separating decisions, diagnostics, target books, execution events, and other linked events. Each `ReplayLineageChain` can be folded with `reconstruct_state()` into latest decision, diagnostics, target book, per-venue execution plans, target orders, place/cancel outcomes, validation rejections, latest execution snapshots, pending optimistic changes, and confirmed live orders. `verify_lineage_fidelity()` and `verify_persisted_lineage_fidelity()` check reconstructed chains for lifecycle coherence: lineage id consistency, target-book to execution coverage, place requests matching target minus validation rejections, lifecycle event ordering, and execution snapshot count consistency. `mm/application/frame_simulator.py` provides offline `FrameSimulator.simulate()` for strategy -> risk -> execution planning and `simulate_from_target_book()` for deterministic risk/execution replay from a persisted target book. `mm/persistence/frame_replay.py` loads persisted `MarketEvent` + `JointLedgerSnapshot` inputs, replays frames offline, and compares simulated target books and execution plans to persisted lineage outputs via `compare_frame_simulation()` and `replay_and_compare_persisted_frame()`. Replay contract tests verify that persisted rows can reconstruct decision -> target book -> execution call chains, preserve failed/rejected execution calls, distinguish pending optimistic execution changes from confirmed live orders, pass fidelity checks for coherent frame lifecycles, and match offline execution replay for persisted target books.

`RedisStreamsBus` writes compact envelopes to Redis streams named from topics. It compacts zero balances and private snapshots before stream writing. Local subscribers receive the original in-process object, while remote stream messages are decoded to envelope payloads and forwarded through `subscribe()`/`subscribe_queue()`. Each bus instance tags its own stream writes and skips them in its stream forwarder to avoid duplicate same-process delivery.

## 7. Observability

### Metrics

`RuntimeObservabilitySubscriber` subscribes to selected topics and converts events into metrics:

- raw market event count, timestamps, ingest lag, and gaps,
- joint frame readiness and local/global latency skew,
- controller decision latency, order count, inventory, cash,
- controller diagnostics stage latency,
- model NLL/log-likelihood/residuals,
- action pipeline stage counts, quantities, notionals, and rejections,
- risk approval, risk rejections, and exposure fields,
- execution gate blocks,
- order sync counts and latency,
- placement/cancel accepted counts,
- live order counts and quantities by side,
- placement failures, fill counts and quantities,
- bus subscriber queue depth, lag, blocked publish latency, and dropped events,
- persistence queue depth and write latency,
- scheduler callback latency and missed-deadline totals,
- event payload and envelope bytes by topic, event type, and pipeline stage,
- runtime errors.

Metrics are stored by a custom in-process Prometheus text registry in `mm/observability/registry.py`. `configure_observability()` starts an HTTP server on the configured metrics port when metrics are enabled.

### Logs And Tracing

`log_event()` writes structured JSON to:

- stdout when `MM_JSON_LOG` is set,
- `MM_LOG_PATH` when configured.

The Docker Compose stack runs Grafana Alloy, which tails `./logs/*.jsonl` into Loki. `mm/observability/traces.py` provides optional OpenTelemetry spans; `configure_observability()` can configure an OTLP HTTP trace exporter.

### Grafana

Docker Compose provisions:

- Postgres,
- Prometheus,
- Loki,
- Tempo,
- Alloy,
- Grafana,
- Redis,
- RedisInsight.

Grafana datasources are provisioned with fixed UIDs:

- `postgres-mm`,
- `prometheus-mm`,
- `loki-mm`,
- `tempo-mm`.

There are two dashboard folders:

- `Market Maker`: older dashboards under `grafana/dashboards`.
- `Market Maker - Observability v2`: newer dashboards under `grafana/observability-v2`.

The v2 dashboards cover runtime overview, controller health, market data, optimizer diagnostics, model quality, action pipeline, risk/execution, target-order stems, orderbook depth, liquidity, portfolio/PnL, fills, execution errors, incidents, and runtime logs.

Most common operational panels use typed Postgres fact tables or Prometheus metrics. Deep debug panels and replay still use `mm_events` JSONB. Prometheus alert rules are provisioned locally; notification routing is not.

## 8. Testing Surface

The repository contains 73 unit test files under `tests/unit`. The tests cover:

- exchange adapter registry and payload-schema contracts,
- runtime config and migration contracts,
- Redis Streams bus,
- orderbook grids and reference VWAPs,
- market data pipeline,
- frame assembly,
- MPC objective and optimizer diagnostics,
- strategy target books and runner gates,
- risk limits and net-delta risk,
- execution planning, reconciliation, validation, and order managers,
- private state reconciler,
- observability dashboards and subscriber behavior,
- Postgres latest-state deduplication,
- bus backpressure, scheduler concurrency, replay, lineage fidelity checks, offline frame simulation, retention/tiering, and parquet writing.

The test suite is broad for unit-level behavior. The codebase does not show a dedicated end-to-end exchange simulator or load/stress test harness in the inspected files.
