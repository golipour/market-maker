# Jaguar Bot — How It Quotes and How It Trades

This document explains the two things the bot does on every market:

1. **Quoting**: it keeps a ladder of buy and sell limit orders on the book.
2. **Trade generation (candles)**: at random times it creates trades, so candles and volume keep moving even when no users are trading.

The two run independently of each other.

---

## Part 1 — Quoting (order ladder)

### In one sentence
The bot's quotes come **only from the external reference price**. It never looks at users' orders when it decides prices or sizes.

### The loop
Every **500 ms** (`tick.interval`), for every tradable market:

```
 external reference price ──┐
   (bestBid / bestAsk)      ├─► build desired ladder ─► compare with bot's own ─► cancel / place
 bot's own open orders ─────┘   (layers + jitter)       open orders
```

1. **Get the reference price**, a `bestBid` / `bestAsk` pair from external exchanges (see [Reference price](#reference-price)).
2. **Get the bot's own open orders.** The bot sees only its own account's orders, not the full order book.
3. **Build the desired ladder** from the market's layer settings (see [Layers](#layers)).
4. **Compare** the desired ladder with the live bot orders, and decide what to keep, cancel or place (see [Keeping orders vs. replacing them](#keeping-orders-vs-replacing-them)).
5. **Send** the actions to the exchange (see [Sending orders](#sending-orders)).

A new market is picked up within about 5 seconds of becoming tradable. Market-making settings are re-read every 5 seconds, so changes to layers apply without a restart.

### Reference price
- **Venues:** KuCoin, Nobitex, Wallex, Tabdeal and Binance are available as price sources. The final price is a **weighted average** of the venues that respond within 4 s.
- **USDT_IRT:** the venue weights come from the index settings in admin.
- **Other USDT-quoted markets** (e.g. `BTC_USDT`): the weights are fixed, KuCoin = 1000 and Nobitex = 1, so in practice this is the KuCoin price.
- **IRT-quoted markets** (e.g. `BTC_IRT`): the price is derived through USDT, `BTC_IRT = BTC_USDT × USDT_IRT`.
- If no venue answers in time, the tick fails and the ladder is left unchanged until the next tick.

### Layers
Each market has one or more **layers** per side. A layer is a band of orders between a minimum and a maximum distance from the reference price.

Settings per market (JSON, set in admin):

| Field | Meaning |
|---|---|
| `totalBidQuantity` | Total size of the **bid** side, in **quote** currency (USDT / IRT) |
| `totalAskQuantity` | Total size of the **ask** side, in **base** currency (BTC, ETH, …) |
| `layers[].type` | `bid` or `ask` |
| `layers[].minDepthPCT` / `maxDepthPCT` | The layer's band, in % from the reference price |
| `layers[].orderCount` | Number of orders in the band |
| `layers[].quantityAllocation` | % of the side total given to this layer (`0` = 100 %) |
| `layers[].sizeVariation` | Random size variation per order, in % (capped at 90 %) |

**Prices**
- Bids are placed from `refBestBid × (1 − minDepth)` down to `refBestBid × (1 − maxDepth)`.
- Asks are placed from `refBestAsk × (1 + minDepth)` up to `refBestAsk × (1 + maxDepth)`.
- Orders are **evenly spaced** across the band and rounded to the market's price precision.
- If the band is too narrow to fit `orderCount` distinct price ticks, fewer orders are placed.
- Orders that land on the same price are merged into one.

**Sizes**
- The layer's quantity is split **equally** across its orders. Bid quantities in quote currency are converted to base using the reference mid price.
- Each order is then varied randomly by up to `sizeVariation` %, so the book doesn't look machine-made.

**Default when a market has no settings:** 20 bids and 20 asks, each between 0.05 % and 2.5 % from the reference price, with a 36 % size variation. Each side totals 250,000 USDT on USDT markets, or 50,000,000,000 IRT on IRT markets.

#### Example
`BTC_USDT`, reference bid 60,000 / ask 60,010, one bid layer with `minDepthPCT=0.1`, `maxDepthPCT=1`, `orderCount=10`, `totalBidQuantity=10000`:

- Prices: 59,940 → 59,400, one order every 60 USDT
- Total size: 10,000 / 60,005 ≈ 0.1667 BTC, so about 0.0167 BTC per order, ± `sizeVariation`

### Keeping orders vs. replacing them
The bot matches each desired order to one of its live orders, from the inside of the book outward. Each live order is then handled by the first rule that applies:

| Situation | Action |
|---|---|
| Live order is **inside the protected spread** (tighter than the innermost layer's `minDepth` after the reference price moved) | Cancel immediately (`SPREAD_GUARD`) |
| Live order has no matching desired order (e.g. the order count was reduced) | Cancel (`UNMATCHED`) |
| Desired order has no live order (e.g. it was filled or is new) | Place |
| Price is off by more than the tolerance | Replace (`PRICE_DRIFT`) |
| Remaining size is off by more than the tolerance (partly filled) | Replace (`REPLENISH_QUANTITY`) |
| Otherwise | Keep as is |

**Tolerances** get wider the further an order is from the reference price. Deep orders move less often.

- Price tolerance = `1 bps + 0.2 × depth(bps)`. Example: an order 100 bps away may drift 21 bps before it is moved.
- Size tolerance = `4 % + 10 × depth(bps) bps`, capped at 50 %. Example: an order 100 bps away is refilled once it is about 14 % smaller than its target.

**Aging refresh:** in addition to the rules above, the **10 oldest orders per side per minute** are replaced (`bot.reconciler.refresh-per-minute`), so the book keeps changing even when the price is flat.

### Sending orders
- New orders and spread-guard cancels are sent **at the same time**.
- Other cancels (drift, refill, aging) are sent **after** the new orders are placed. The replacement is on the book before the old order is removed, so there is no gap in the ladder.
- An optional per-tick placement limit exists (`bot.placement.strategy=token-bucket`, inner levels first). It is **off by default**, so every placement goes out on the same tick.

### What this means in practice
- If the external price moves, the whole ladder moves with it within one or two ticks.
- If a user places a large order, the ladder does **not** move.
- A user order priced **beyond** the reference price (e.g. a bid above `refAsk × (1 + minDepth)`) will trade against the bot's resting orders. The spread guard only checks against the reference price, not against the live book.

---

## Part 2 — Trade generation (candles)

The ladder by itself never trades, so without user activity no candles would form. A separate job creates trades.

### When it runs
- A cron runs at `16 */4 * ? * *`, i.e. second 16 of every 4th minute (`job.tradeBot.cron`).
- On each run, each market is processed with a **~30 % chance** (`job.tradeBot.runPercent`). Trades therefore come at irregular times: on average about one trade every 13 minutes per market.

### What it does
It reads the **top buy** and **top sell** of the real order book and checks who owns each one. It stops if either side is empty.

| Top sell | Top buy | Action |
|---|---|---|
| bot | bot | Picks one side at random and fills it with an opposite order |
| user | bot | Fills the bot's bid |
| bot | user | Fills the bot's ask |
| user | user | Picks a random price **between** the best bid and best ask, then sends a matching buy and sell at that price, so the bot trades with itself |

**User orders are never filled by this job.** When the top of the book belongs to users, the bot trades inside the spread instead.

### Trade size
- The value of the trade is a random number between `minimumOrder` and `5 × minimumOrder`, skewed toward the small end. Most trades are close to the minimum.
- When the job fills a bot order, the trade can't be larger than that order.
- If a market has no `minimumOrder`, the minimum is **5 USDT** on USDT markets and **500,000 IRT** on others.

---

## Known issues

1. **The "user / user" case currently places zero-size orders.** When the bot loads market configs, it sets `minimumOrder` to 0 for every market. This is a side effect of how the default layers are built. The self-trade size is calculated from `minimumOrder`, so it comes out as 0. As a result, when both top orders belong to users, the job produces **no trade**. The other cases work, and they use the 5 USDT / 500,000 IRT fallback because `minimumOrder` is 0.
2. **Self-trade size doesn't depend on price.** In the "user / user" case the size is calculated as *value ÷ minimumOrder* rather than *value ÷ price*. Once issue 1 is fixed, every self-trade will be 1–5 units of the base asset. For example, 1–5 BTC per trade.
3. **Self-trades aren't guaranteed.** The matching buy and sell go out in one request, but another order can fill one of them first. In that case the bot ends up with a one-sided position or a resting order inside the spread.
4. **Candle prices don't follow the external market.** Trade prices come from the exchange's own top of book, not the reference price. On a quiet market, candles only move as far as the bot's own ladder moves.

---

## Settings summary

| Key | Default | Meaning |
|---|---|---|
| `tick.interval` | 500 ms | How often the ladder is recalculated |
| `bot.reconciler.refresh-per-minute` | 10 | Old orders replaced per side per minute |
| `bot.jitter.refresh-per-minute` | 0 | How often size variation is re-rolled (0 = fixed per order) |
| `bot.placement.strategy` | `unbounded` | `token-bucket` limits how many orders are placed per tick |
| `job.tradeBot.cron` | `16 */4 * ? * *` | Trade job schedule |
| `job.tradeBot.runPercent` | 30 | Chance that a market trades on each run |
| Admin: MM layer settings | per market | Layers, depths, counts, sizes (see [Layers](#layers)) |
| Admin: index settings | USDT_IRT | Venue weights for the USDT_IRT reference price |
