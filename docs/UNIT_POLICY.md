## Canonical Units Policy (Target Design)

### Goal
Make controller/MPC/execution logic human-readable and exchange-agnostic by using **real float units** internally, and restricting tick/lot rounding to exchange adapters only.

### Canonical units (inside `mm.layers.core/`)

- **price**: quote per base (float)  
  Example: BTC/USDT → `price = 65000.25` means 65,000.25 USDT per 1 BTC.

- **qty**: base quantity (float)  
  Example: `qty = 0.0123` BTC.

- **notional**: quote amount (float)  
  Example: `notional = price * qty` in USDT.

- **fees**: explicit `(amount, currency)` and optionally a converted quote value.

### Where tick/lot/min-size belongs

All exchange microstructure constraints live in `mm.layers.adapters/`:

- tick size (price step)
- quantity step / lot size
- min qty
- min notional
- price/qty precision

The adapter converts:

- inbound venue payloads → canonical floats
- outbound `OrderIntent` floats → venue-compliant orders (rounded/quantized)

### Valuation policy

The current chosen convention is **Option B**:

- mark inventory using **sell-side representative price** (`sell_ref`, i.e. bid VWAP to target depth).

This choice must be applied consistently in execution PnL, wealth tracking, and MPC reward.

