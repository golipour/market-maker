# Bitpin API Used By This Codebase

This document is derived from the repository code and tests. It is intended for
someone who needs to reuse the Bitpin API surface this project depends on
without access to Bitpin's web documentation.

It is not a complete Bitpin API manual. It documents only endpoints, websocket
channels, request bodies, headers, and response fields that our code uses or
accepts.

## Hosts

| Purpose | Base URL |
| --- | --- |
| Main trading, private state, public market data | `https://api.bitpin.ir` |
| Public websocket and private user websocket | `wss://centrifugo.bitpin.ir/connection/websocket` |
| Historical trade-stat helper only | `https://api.bitpin.org` |

## Symbol Format

Most Bitpin calls in this code use uppercase underscore symbols:

```text
BASE_QUOTE
USDT_IRT
BTC_IRT
PAXG_IRT
```

The adapter normalizes symbols with `api_symbol(symbol)`, which splits the
internal symbol into base and quote and returns `BASE_QUOTE`.

Some main trading endpoints require a numeric `market_id`. The code resolves it
from the public markets endpoint and caches it.

## Authentication

The code supports two auth forms:

| Auth form | Inputs | Use |
| --- | --- | --- |
| API key auth | `api_key`, `secret_key` | Login, token refresh, private REST, private websocket info |
| Bearer token auth | Access token string | Private REST calls directly |

Environment/config mapping in this repo:

| Config/env | Meaning |
| --- | --- |
| `BITPIN_API_TOKEN` | Bitpin API key in this repo's runtime config |
| `BITPIN_API_PRIVATE_KEY` | Bitpin secret key |

Authenticated REST requests use:

```http
Accept: application/json
Content-Type: application/json
Authorization: Bearer <access_token>
```

If an authenticated request returns a `401` whose body indicates
`token_not_valid` or "token is invalid or expired", the API-key auth path tries
to refresh the access token once and retries the request.

The code applies local cooldowns for these authentication failures:

| Response | Local behavior |
| --- | --- |
| `429` with text like `Expected available in 19 seconds` | Cool down for that duration, or 30 seconds if no duration is parsed |
| `406` containing `wrong ip` | Cool down for 300 seconds |

## Public REST: Market Discovery

### List Markets

Used to resolve `USDT_IRT` style symbols into numeric market ids.

```http
GET https://api.bitpin.ir/v1/mkt/markets/?page=<page>
```

Query parameters:

| Name | Required | Notes |
| --- | --- | --- |
| `page` | Yes in our code | Starts at `1`; code follows `next` until null/false |

Response shape accepted by the code:

```json
{
  "count": 1,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 5,
      "code": "USDT_IRT",
      "currency1": {"code": "USDT"},
      "currency2": {"code": "IRT"}
    }
  ]
}
```

Market symbol matching accepts either `code` or `symbol`; if neither exists, it
falls back to `currency1.code` and `currency2.code`. Hyphens and underscores are
treated as equivalent for matching.

## Public REST: Market Data

### Active Orders / Orderbook Side

Used by the main REST market-data adapter. The adapter calls this endpoint once
for buys and once for sells.

```http
GET https://api.bitpin.ir/v2/mth/actives/{market_id}/?type=buy
GET https://api.bitpin.ir/v2/mth/actives/{market_id}/?type=sell
```

Query parameters:

| Name | Values | Notes |
| --- | --- | --- |
| `type` | `buy`, `sell` | `buy` becomes bids; `sell` becomes asks |

Response rows accepted by the code:

```json
{
  "results": [
    {"id": 1, "price": "100000", "remain_amount": "1.5"},
    {"id": 2, "price": "99900", "remain_amount": "2.0"}
  ]
}
```

The parser accepts rows as either `[price, qty]` arrays or dictionaries.

Accepted price fields:

```text
price, price_limit, active_limit
```

Accepted quantity fields:

```text
remain_amount, remain, amount1, amount, quantity, value
```

Accepted payload timestamp fields, if present:

```text
timestamp_ms, timestamp, time, created_at, updated_at
```

If no timestamp is present, the shared timestamp parser supplies the current
time/default behavior.

### Recent Matches / Trades By Market Id

Used by the main REST market-data adapter.

```http
GET https://api.bitpin.ir/v1/mth/matches/{market_id}/
```

Response rows accepted by the code:

```json
{
  "results": [
    {
      "id": 10,
      "type": "sell",
      "price": "100000",
      "amount1": "0.5",
      "created_at": "2024-03-10T12:00:01+00:00"
    }
  ]
}
```

Accepted row containers:

```text
list payload, results, orders, matches, data
```

Accepted trade id fields:

```text
id, match_id, trade_id
```

Accepted timestamp fields:

```text
created_at, time, timestamp
```

Accepted price fields:

```text
price, price_limit, average_price
```

Accepted quantity fields:

```text
match_amount, amount1, amount, quantity, exchanged1
```

Accepted side fields:

```text
type, side
```

The adapter maps `buy` to buy; every other value becomes sell.

## Private REST: Auth Endpoints

### Login / Authenticate

```http
POST https://api.bitpin.ir/api/v1/usr/authenticate/
Content-Type: application/json
```

Request body:

```json
{
  "api_key": "<api key>",
  "secret_key": "<secret key>"
}
```

Response fields accepted:

```json
{
  "access": "<access token>",
  "refresh": "<refresh token>"
}
```

The code also accepts `access_token` instead of `access`, and `refresh_token`
instead of `refresh`.

### Refresh Access Token

```http
POST https://api.bitpin.ir/api/v1/usr/refresh_token/
Content-Type: application/json
```

Request body:

```json
{
  "refresh": "<refresh token>"
}
```

Response fields accepted:

```json
{
  "access": "<new access token>"
}
```

The code also accepts `access_token`.

### Websocket Info

Used only for private websocket state.

```http
POST https://api.bitpin.ir/api/v1/usr/ws-info/
Content-Type: application/json
```

Request body:

```json
{
  "api_key": "<api key>",
  "secret_key": "<secret key>"
}
```

Response body expected:

```json
{
  "user_identifier": "V33EQ5LV",
  "ws_token": "jwt-token"
}
```

If either value is missing, the private websocket adapter falls back to REST
polling only.

## Private REST: Account State

All endpoints in this section use the bearer headers shown in the authentication
section.

### Wallets / Balances

```http
GET https://api.bitpin.ir/api/v1/wlt/wallets/
Authorization: Bearer <access_token>
```

Response rows may appear in `results`, `wallets`, `data`, or as a list payload.

Accepted asset fields:

```text
currency, asset, coin
```

Each asset may be either a string or an object containing `code` or `symbol`.

Accepted balance fields:

| Normalized value | Accepted fields |
| --- | --- |
| free | `balance`, `available`, `free` |
| locked | `frozen`, `blocked`, `locked`, `freeze` |
| total | `total`, or `free + locked` if absent |

Example:

```json
{
  "results": [
    {
      "currency": {"code": "USDT"},
      "balance": "1.5",
      "frozen": "0.2",
      "total": "1.7"
    }
  ]
}
```

### Open Orders

The code first resolves `market_id` via `GET /v1/mkt/markets/`.

```http
GET https://api.bitpin.ir/api/v1/odr/orders/?market_id=<market_id>&state=active
Authorization: Bearer <access_token>
```

Response rows may appear in `results`, `orders`, `data`, or as a list payload.

Accepted order fields for normalization:

| Normalized value | Accepted fields |
| --- | --- |
| exchange order id | `orderId`, `order_id`, `id` |
| side | `side`, or `type` |
| quantity | `quantity`, `base_amount`, `amount1`, `amount`, `resource` |
| executed quantity | `executedQty`, `base_amount`, `exchanged1`, `matchedAmount` |
| remaining quantity | `remaining`, `remain_amount`, `remain` |
| price | `price`, `price_limit`, `active_limit`, `average_price` |
| timestamp | `timestamp`, `created_at`, `activated_at`, `closed_at` |
| fee | `fee`, `commission`, `user_commission` |
| client/order actor id | `actor_id`, `identifier` |

Example open order row:

```json
{
  "id": 10,
  "type": "sell",
  "price_limit": "100100",
  "amount1": "3",
  "exchanged1": "1",
  "remain_amount": "2",
  "state": "active",
  "identifier": "cid"
}
```

### Filled Orders / Fills

Primary endpoint:

```http
GET https://api.bitpin.ir/api/v1/odr/fills/?symbol=<BASE_QUOTE>&limit=100
Authorization: Bearer <access_token>
```

Example:

```http
GET https://api.bitpin.ir/api/v1/odr/fills/?symbol=USDT_IRT&limit=100
```

Response may be a list payload or rows under `results`, `orders`, `matches`,
`wallets`, or `data`.

Example fill row:

```json
{
  "id": 65375307,
  "symbol": "USDT_IRT",
  "base_amount": "63.0",
  "quote_amount": "123417",
  "price": "1959",
  "created_at": "2024-07-29T14:07:34.534824+03:30",
  "commission": "0",
  "side": "sell",
  "commission_currency": "IRT",
  "order_id": 535179385,
  "identifier": "cid"
}
```

Fallback endpoint if the primary fills endpoint returns no rows:

```http
GET https://api.bitpin.ir/api/v1/odr/orders/?market_id=<market_id>&state=closed
Authorization: Bearer <access_token>
```

The fallback keeps only rows where `exchanged1` or `fulfilled` is greater than
zero.

## Private REST: Order Commands

All order command endpoints use bearer authentication.

### Place One Order

Used by helper functions and by market-order support. The runtime order adapter
prefers the bulk endpoint below.

```http
POST https://api.bitpin.ir/api/v1/odr/orders/
Authorization: Bearer <access_token>
Content-Type: application/json
```

Limit order request body:

```json
{
  "market": "5",
  "amount1": "2",
  "price": "100000",
  "mode": "limit",
  "type": "buy",
  "identifier": "optional-client-id"
}
```

Market order request body:

```json
{
  "market": "5",
  "amount1": "2",
  "price": "0",
  "mode": "market",
  "type": "buy",
  "identifier": "optional-client-id"
}
```

Notes:

| Field | Meaning |
| --- | --- |
| `market` | Numeric market id as a string |
| `amount1` | Base asset amount |
| `price` | Limit price, or `0` for market mode in this code |
| `mode` | `limit` or `market` |
| `type` | `buy` or `sell` |
| `identifier` | Optional client id, truncated to 64 characters |

The code treats any returned dictionary as the response. Higher-level order id
parsing accepts `orderId`, `order_id`, `id`, `clientOrderId`, or
`client_order_id`, including nested `order`, `data`, or `result` dictionaries.

### Place Bulk Orders

Used by the runtime order adapter. Orders are chunked into batches of at most
10, and each batch must contain only one market symbol.

```http
POST https://api.bitpin.ir/api/v1/odr/orders/bulk
Authorization: Bearer <access_token>
Content-Type: application/json
```

Request body is a JSON array:

```json
[
  {
    "symbol": "USDT_IRT",
    "base_amount": "2",
    "price": "100000",
    "side": "buy",
    "type": "limit",
    "identifier": "cid"
  }
]
```

Input field mapping used by the code:

| API field | Source fields accepted by helper |
| --- | --- |
| `symbol` | `symbol`, normalized to `BASE_QUOTE` |
| `base_amount` | `base_amount`, `amount`, `amount1`, `qty` |
| `price` | `price` |
| `side` | `side`, `type` |
| `type` | `order_type`, `mode`, default `limit` |
| `identifier` | `identifier`, `client_order_id`, truncated to 64 chars |

Expected response shape:

```json
{
  "orders": [
    {"id": 42, "state": "active"}
  ]
}
```

The order adapter requires:

| Requirement | Behavior if not met |
| --- | --- |
| Response has `orders` as a list | Raises `place bulk orders failed on bitpin` |
| Number of returned orders equals number submitted | Raises malformed/expected count error |
| Each returned order has an id and state is not `failed`, `error`, or `rejected` | Raises `place order failed on bitpin` |

### Cancel One Order

```http
DELETE https://api.bitpin.ir/api/v1/odr/orders/{order_id}/
Authorization: Bearer <access_token>
```

The code sends no JSON body for this helper.

### Cancel Bulk Orders

```http
DELETE https://api.bitpin.ir/api/v1/odr/orders/bulk/
Authorization: Bearer <access_token>
Content-Type: application/json
```

Cancel by exchange order ids:

```json
{
  "ids": ["1", "2"]
}
```

Cancel by client identifiers, used only if no ids are supplied:

```json
{
  "identifiers": ["cid-1", "cid-2"]
}
```

Expected response shape:

```json
{
  "canceled_orders": ["1", "2"],
  "not_canceled_orders": []
}
```

The runtime order adapter treats the cancel as failed if:

```text
state/status is failed, error, or rejected
not_canceled_orders is a non-empty list
```

## Public Websocket Market Data

Endpoint:

```text
wss://centrifugo.bitpin.ir/connection/websocket
```

The code speaks a Centrifuge-style websocket protocol. Frames are JSON objects,
and multiple JSON frames may arrive in one websocket message separated by
newlines.

### Public Connect Frame

```json
{"id": 1, "connect": {}}
```

### Public Subscriptions

Orderbook channel:

```json
{
  "id": 12345,
  "subscribe": {"channel": "orderbook:USDT_IRT"}
}
```

Trades channel:

```json
{
  "id": 12346,
  "subscribe": {"channel": "matches:USDT_IRT"}
}
```

The code computes subscription ids from the channel hash. Any unique integer id
should be fine for a standalone client.

### Public Orderbook Event

Expected event name:

```text
market_data
```

Example payload:

```json
{
  "push": {
    "channel": "orderbook:USDT_IRT",
    "pub": {
      "data": {
        "asks": [["100100", "0.4"], ["100200", "2.5"]],
        "bids": [["100000", "1.5"], ["99900", "2.0"]],
        "volume_ask": "2.9",
        "volume_bid": "3.5",
        "price": "100000",
        "symbol": "USDT_IRT",
        "event_time": "2024-08-14T14:37:54.487062Z",
        "event": "market_data"
      }
    }
  }
}
```

The parser also accepts `push.pub.data` as a JSON string rather than an object.

### Public Trade Event

Expected event name:

```text
matches_update
```

Example payload:

```json
{
  "push": {
    "channel": "matches:USDT_IRT",
    "pub": {
      "data": {
        "matches": [
          {
            "id": "73802972_73807087",
            "price": "100000",
            "base_amount": "0.5",
            "quote_amount": "50000",
            "side": "sell",
            "time": 1760873797.149016
          }
        ],
        "symbol": "USDT_IRT",
        "event_time": "2025-10-19T11:38:27.929164Z",
        "event": "matches_update"
      }
    }
  }
}
```

Trade fields consumed:

| Normalized value | Field |
| --- | --- |
| trade id | `id` |
| timestamp | `time`, fallback to parent `event_time` |
| price | `price` |
| quantity | `base_amount` |
| side | `side`; `buy` stays buy, anything else becomes sell |

## Private Websocket User State

Endpoint:

```text
wss://centrifugo.bitpin.ir/connection/websocket
```

Before connecting, call `POST /api/v1/usr/ws-info/` to get:

```json
{
  "user_identifier": "<identifier>",
  "ws_token": "<jwt token>"
}
```

### Private Connect Frame

```json
{
  "id": 1,
  "connect": {"token": "<ws_token>"}
}
```

### Private Subscription

```json
{
  "id": 2,
  "subscribe": {"channel": "user:order_info#<user_identifier>"}
}
```

### User Order Update Event

Expected event name:

```text
user_order_update
```

Example payload:

```json
{
  "push": {
    "pub": {
      "data": {
        "id": 123,
        "symbol": "PAXG_IRT",
        "type": "market",
        "side": "sell",
        "price": "464414599",
        "base_amount": "0.0012",
        "dealed_base_amount": "0.0004",
        "identifier": "cid",
        "state": "active",
        "event": "user_order_update",
        "event_time": "2025-10-19T12:03:47.620634Z"
      }
    }
  }
}
```

Fields consumed:

| Normalized value | Field |
| --- | --- |
| exchange order id | `id` |
| actor/client id | `identifier` |
| status | `state` |
| original quantity | `base_amount` |
| filled quantity | `dealed_base_amount` |
| remaining quantity | `base_amount - dealed_base_amount` |
| timestamp | `event_time`, fallback to `created_at`, fallback to `closed_at` |

### User Match / Fill Event

Expected event name:

```text
user_match_update
```

Example payload:

```json
{
  "push": {
    "pub": {
      "data": {
        "id": 1234,
        "symbol": "PAXG_IRT",
        "base_amount": "0.0004",
        "price": "464314617",
        "commission": "0",
        "side": "sell",
        "order_id": 123,
        "identifier": "cid",
        "event": "user_match_update",
        "event_time": "2025-10-19T12:03:47.651296Z"
      }
    }
  }
}
```

Fields consumed:

| Normalized value | Field |
| --- | --- |
| exchange order id | `order_id` |
| actor/client id | `identifier` |
| side | `side`; `buy` stays buy, anything else becomes sell |
| price | `price` |
| quantity | `base_amount`; zero or negative fills are ignored |
| fee | `commission` |
| timestamp | `event_time`, fallback to `created_at` |

## Public Trade-Stats Helper Only

The file `mm/integrations/trade_stats.py` has a separate `BitPinAdapter` used
for market statistics and public historical sampling. It uses `api.bitpin.org`,
not the main `api.bitpin.ir` host.

### Orderbook By Symbol

```http
GET https://api.bitpin.org/api/v1/mth/orderbook/{BASE_QUOTE}/
```

Example:

```http
GET https://api.bitpin.org/api/v1/mth/orderbook/BTC_IRT/
```

Accepted bid/ask containers:

```text
bids, asks, buy, sell
```

Accepted row shapes:

```json
["100000", "1.5"]
```

or:

```json
{"price": "100000", "amount": "1.5"}
```

Dictionary quantity fields accepted:

```text
amount, base_amount, quantity, size
```

### Trades By Symbol

```http
GET https://api.bitpin.org/api/v1/mth/matches/{BASE_QUOTE}/
```

Accepted response containers:

```text
list payload, results, matches, trades
```

Accepted trade fields:

| Normalized value | Accepted fields |
| --- | --- |
| timestamp | `time`, `created_at`, `timestamp`, `date` |
| price | `price`, `price_str` |
| quantity | `amount`, `base_amount`, `quantity`, `volume` |
| side | `side`, `type`, or `is_sell` |

The adapter converts side to Binance-style `isBuyerMaker`:

```text
SELL or SELLER => true
BUY => false
empty side with is_sell truthy => true
```

### Exchange Info / Tick Size

```http
GET https://api.bitpin.org/api/v1/mkt/markets/
```

The adapter searches rows in `results` or `markets` for a row whose `symbol` or
`market` equals the requested `BASE_QUOTE` symbol.

If `price_precision` exists, tick size is computed as:

```text
10 ^ (-price_precision)
```

If no matching market or precision is found, the adapter returns tick size `1`.

## Error Handling Expectations

The private REST helper raises `BitpinAuthError` for non-OK responses from
private/auth endpoints. The exception includes:

```text
status_code
path
detail
```

Response detail is parsed as JSON if possible, otherwise the first 4000
characters of the response text are used.

Public REST market-data calls rely on the shared `RestClient`; non-success
handling comes from that shared client rather than Bitpin-specific code.

## Code References

Primary implementation files:

```text
mm/exchanges/bitpin/rest.py
mm/exchanges/bitpin/ws.py
mm/exchanges/bitpin/private_rest.py
mm/exchanges/bitpin/private_state.py
mm/exchanges/bitpin/order_adapter.py
mm/integrations/trade_stats.py
```

Contract tests and example payloads:

```text
tests/unit/test_bitpin_adapter.py
tests/unit/test_exchange_payload_normalization.py
tests/fixtures/exchange_payloads/bitpin_markets.json
tests/fixtures/exchange_payloads/bitpin_rest_buy_orderbook.json
tests/fixtures/exchange_payloads/bitpin_rest_sell_orderbook.json
tests/fixtures/exchange_payloads/bitpin_rest_trades.json
tests/fixtures/exchange_payloads/bitpin_ws_orderbook.json
tests/fixtures/exchange_payloads/bitpin_ws_trade.json
```
