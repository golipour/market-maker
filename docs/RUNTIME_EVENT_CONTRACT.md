# Runtime Event Contract

This document describes the bus-published events in the async `mm/` runtime. The legacy `mm.layers` order-flow runtime has separate internal diagnostics and is not the source of these topics.

## Envelope

Durable sinks wrap each event in `mm.core.events.EventEnvelope` before writing:

- `schema`: fixed `mm.event`
- `version`: fixed `1`
- `event_id`: unique id
- `run_id`: observability run id
- `source`: writer/source name
- `topic`: original bus topic
- `event_type`: payload `type` field when present, otherwise Python class name
- `exchange`, `symbol`, `actor_id`: best-effort extracted metadata
- `timestamp_ms`: event timestamp, `t_end_ms`, `created_at_ms`, or writer time
- `payload`: JSON-safe event body

Postgres stores envelopes in `mm_events` and snapshot-like events in `mm_latest_state`.

## Public Market Data

`market:raw:{exchange}:{symbol}:trades`

- Payload: `RawTrade`
- Fields: `trade_id`, `timestamp_ms`, `price`, `qty`, `side`, `exchange`, `symbol`
- Meaning: normalized trade tick.
- Producer: `mm.ingestion.symbol_engine.SymbolEngine`
- Persisted by default: no

`market:raw:{exchange}:{symbol}:orderbook`

- Payload: `RawOrderbook`
- Fields: `exchange`, `symbol`, `timestamp_ms`, `bids`, `asks`, `is_snapshot`
- Nested `PriceLevel`: `price`, `qty`
- Meaning: normalized materialized orderbook snapshot or delta update.
- Producer: `SymbolEngine`
- Persisted by default: no

`market:events:{exchange}:{symbol}:{interval_ms}`

- Payload: `MarketEvent`
- Fields: `exchange`, `symbol`, `interval_ms`, `t_start_ms`, `t_end_ms`, `orderbook`, `trades`, `buy_volume`, `sell_volume`, `vwap`, `imbalance`, `microprice`
- Meaning: interval aggregate used by strategy and controller.
- Producer: `SymbolEngine.publish_on_intervals`
- Persisted by default: yes for primary and hedge

`market:errors:{exchange}:{symbol}`

- Payload: dict
- Typical fields: `source`, `error`, `message`, optional `timestamp_ms`, `is_snapshot`
- Meaning: adapter, normalization, or feed errors.
- Producer: `SymbolEngine`
- Persisted by default: no

## Strategy Pipeline

`strategy:frames:{primary_exchange}:{symbol}:{interval_ms}`

- Payload: `StrategyInputFrame`
- Fields: `primary`, `hedge`, `primary_private`, `hedge_private`, `hedge_status`, `hedge_lag_ms`, `max_hedge_lag_ms`, `created_at_ms`
- `hedge_status`: `disabled`, `ready`, `missing`, or `stale`
- Meaning: aligned primary market event plus optional hedge and private snapshots.
- Producer: `mm.actors.frame_assembler.StrategyInputFrameAssembler`
- Persisted by default: yes

`strategy:controller`

- Payload: dict with `type=ControllerDecision` or `type=ControllerError`
- Decision fields: `actor_id`, `exchange`, `symbol`, `t_end_ms`, `observe_only`, `frame_step`, `action_start_after_steps`, `hedge_status`, `has_hedge`, `mid`, `inventory`, `cash`, volumes, `orders`, model summaries, `optimizer`, `orderbook_grid`, `latency_ms`, `unit_policy`, `decision_hash`
- Error fields: `type`, `actor_id`, `exchange`, `symbol`, `t_end_ms`, `error`, `message`
- Meaning: MPC decision diagnostics or controller failure.
- Producer: `mm.strategies.mpc_controller.MpcControllerRunner`
- Persisted by default: yes when controller is configured

`strategy:actions`

- Payload: `Action`
- Fields: `actor_id`, `orders_to_create`, `orders_to_cancel`
- Nested `Order`: `exchange`, `symbol`, `side`, `price`, `qty`, `order_type`, `client_order_id`
- Meaning: intended order batch. In observe-only mode this is emitted without live order side effects.
- Producer: `MpcControllerRunner`
- Persisted by default: yes

## Private Exchange State

`private:state:{exchange}:snapshot`

- Payload: `PrivateStateSnapshot`
- Fields: `exchange`, `timestamp_ms`, `balances`, `open_orders`
- Meaning: full private state view for balances and open orders.
- Producer: `mm.private_state.reconciler.PrivateStateReconciler`
- Persisted by default: yes

`private:state:{exchange}:balances`

- Payload: `Balance`
- Fields: `exchange`, `asset`, `free`, `locked`, `timestamp_ms`
- Meaning: single balance update.
- Persisted by default: no

`private:state:{exchange}:open_orders`

- Payload: `OpenOrder`
- Fields: `exchange_order_id`, `actor_id`, `order`, `remaining_qty`, `filled_qty`, `status`, `timestamp_ms`
- Meaning: single open-order update.
- Persisted by default: no

`private:state:{exchange}:order_updates`

- Payload: `OrderUpdate`
- Fields: `exchange`, `symbol`, `exchange_order_id`, `actor_id`, `status`, `remaining_qty`, `filled_qty`, `timestamp_ms`
- Meaning: order status or fill-progress update.
- Persisted by default: no

`private:state:{exchange}:fills`

- Payload: `Fill`
- Fields: `exchange_order_id`, `actor_id`, `exchange`, `symbol`, `side`, `price`, `qty`, `fee`, `timestamp_ms`
- Meaning: exchange fill.
- Persisted by default: no

`private:state:{exchange}:errors`

- Payload: adapter-dependent dict
- Meaning: private-state adapter failures.
- Persisted by default: no

## Execution

`execution:state:{exchange}`

- Payload: `ExecutionSnapshot`
- Fields: `exchange`, `timestamp_ms`, `live_order_count`, `live_orders`
- Meaning: local live-order mirror after private reconciliation or order-manager sync.
- Producers: `ExecutionStateReconciler`, `OrderManager`
- Persisted by default: yes

`execution:orders:{exchange}`

- Payload: dict with `type`
- Event types: `ExecutionSyncRequested`, `CancelOrdersRequested`, `CancelOrdersAccepted`, `PlaceOrdersRequested`, `PlaceOrdersAccepted`
- Typical fields: `exchange`, `actor_id`, order ids or orders, counts, `timestamp_ms`
- Meaning: order-manager command lifecycle.
- Producer: `OrderManager`
- Persisted by default: yes

`execution:cancel_all:{exchange}` and `execution:cancel_all:all`

- Payload: dict with `type=CancelAllRequested`
- Fields: `command_id`, `exchange`, `actor_id`, `reason`, `timestamp_ms`
- Meaning: admin-triggered cancel-all command.
- Producer: `RuntimeControlPlane`
- Persisted by default: no

## Admin

`admin:commands`

- Payload: dict with `type=AdminCommand`
- Fields: `command_id`, `user_id`, `command`, `actor_id`, `exchange`, `enabled`, `reason`, `payload`, `timestamp_ms`
- Commands: `pause`, `resume`, `kill_switch`, `cancel_all`, `observe_only`, `reload_risk`
- Meaning: accepted admin request for runtime control.
- Producer: `mm.api.admin`
- Persisted by default: no

`admin:events`

- Payload: dict with `type=AdminCommandAccepted` or `type=RuntimeControlState`
- Runtime state fields: `paused_actors`, `observe_only`, `kill_switch`, `timestamp_ms`, optional `command_id`
- Meaning: admin acknowledgement or resulting control-plane state.
- Producers: `mm.api.admin`, `RuntimeControlPlane`
- Persisted by default: no

## Optional Actor/Risk Events

`risk:events`

- Payload: dict with `type=RiskActionsFiltered`
- Meaning: risk layer filtered proposed actions.
- Producer: `mm.actors.actor.Actor`
- Runtime status: defined but not wired by default `build_runtime_components`.

`actor:journal:{actor_id}`

- Payload: dict with frame/action counts and latency metadata.
- Meaning: actor loop journal.
- Producer: `Actor`
- Runtime status: defined but not wired by default `build_runtime_components`.

## Diagnostics Side Channel

Controller decisions are also mapped by `mm.diagnostics_bridge` into a `DiagnosticsFrame`-like dict and emitted through `mm.diagnostics_emitter` when `MM_DIAGNOSTICS` is set. This is not an `AbstractEventBus` topic and is not stored by `PostgresEventWriter` unless also published on a persisted bus topic.

## Persistence Defaults

When `persistence.enabled` is true, the default topics are:

- primary `market:events:{exchange}:{symbol}:{interval_ms}`
- hedge `market:events:{exchange}:{symbol}:{interval_ms}` when configured
- `strategy:frames:{primary_exchange}:{symbol}:{interval_ms}`
- `strategy:actions`
- configured controller diagnostics topic
- `execution:orders:{exchange}` for primary and hedge exchanges
- `execution:state:{exchange}` for primary and hedge exchanges
- `private:state:{exchange}:snapshot` for primary and hedge exchanges

Raw market data, granular private updates, admin events, cancel-all commands, risk events, and actor journals must be explicitly added to `persistence.topics` if they need durable storage.
