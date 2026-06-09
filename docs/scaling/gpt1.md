After reading both documents, my recommendation is that **the biggest architectural change is not Kubernetes, Kafka, or optimizer acceleration. It is introducing a Portfolio/Capital Allocation layer and separating exchange truth from strategy truth.**

Your current runtime is actually quite sophisticated already. It already has:

* event-driven architecture
* ledger abstraction
* replayability
* lineage tracking
* execution reconciliation
* MPC isolation
* observability
* persistence

The main scaling limitation is not technical performance. It is that the current ownership model is:

```text
Strategy
   ↓
Ledger
   ↓
Exchange Account
```

This model becomes increasingly painful once you have:

* many symbols
* many strategies
* multiple accounts
* portfolio-level risk
* capital allocation
* shared balances

---

# First: Account Model Recommendation

I would strongly recommend:

```text
Platform
    ↓
Exchange
    ↓
Account / Subaccount
    ↓
Portfolio
    ↓
Capital Allocation
    ↓
Strategy Instance
```

not:

```text
Strategy
    ↓
Exchange Account
```

and not:

```text
Strategy
    ↓
API Key
```

---

# Why Not One Account Per Strategy?

Initially it sounds attractive.

```text
BTC MM -> account A
ETH MM -> account B
DOGE MM -> account C
```

Advantages:

* simple PnL
* simple inventory
* simple risk

But operationally:

* many exchanges do not support enough subaccounts
* API key management explodes
* transfers become painful
* capital becomes fragmented
* idle balances accumulate

Imagine:

```text
BTC MM needs 50k
ETH MM needs 50k

BTC uses only 10k
ETH needs 90k
```

Capital becomes stranded.

This becomes a treasury-management nightmare.

---

# Why Multiple API Keys Does Not Solve It

Model C is mostly illusionary isolation.

```text
Account
   ├── API Key A
   ├── API Key B
   └── API Key C
```

Balances remain shared.

Inventory remains shared.

Risk remains shared.

PnL remains shared.

Therefore API keys are an access-control concept, not a portfolio concept.

I would never model ownership through API keys.

---

# Recommended Ownership Model

Separate:

## Exchange Truth

```text
Exchange Account Ledger
```

Tracks:

* balances
* positions
* orders
* fills

This is the source of truth.

---

## Portfolio Ledger

Tracks:

```text
Portfolio A
Portfolio B
Portfolio C
```

Example:

```text
Wallex Main Account
    ↓
Trading Portfolio
```

Portfolio tracks:

* allocations
* exposures
* risk budgets

---

## Strategy Ledger

Tracks:

```text
BTC MM
ETH MM
DOGE MM
```

Each strategy gets:

* attributed inventory
* attributed PnL
* attributed capital
* attributed fills

This is an accounting view.

Not exchange truth.

---

# Three-Layer Ledger Architecture

I would redesign current PortfolioLedgerService into:

```text
Exchange Ledger
        ↓
Portfolio Ledger
        ↓
Strategy Ledger
```

---

## Layer 1: Exchange Ledger

Source of truth.

```text
balances
orders
fills
positions
```

Generated from exchange reconciliation.

Current PrivateStateReconciler already resembles this layer. 

---

## Layer 2: Portfolio Ledger

Tracks:

```text
allocated cash
allocated inventory
allocated risk
```

Example:

```text
Account = 1,000,000 USDT

Portfolio:
    MM Portfolio = 700k
    Arb Portfolio = 300k
```

---

## Layer 3: Strategy Ledger

Tracks:

```text
BTC MM = 100k allocation
ETH MM = 50k allocation
DOGE MM = 30k allocation
```

The strategy never owns exchange balances.

It owns allocation.

---

# Capital Allocation Engine

Create a dedicated service:

```text
Capital Allocation Service
```

Responsibilities:

```text
allocate capital
reclaim capital
rebalance capital
enforce limits
```

Example:

```text
BTC MM
allocated = 100k

actual account = 1M
```

Strategy sees:

```text
available = 100k
```

not

```text
available = 1M
```

---

# Risk Architecture

Current system:

```text
Strategy
    ↓
Risk
    ↓
Execution
```

For scale:

```text
Strategy Risk
        ↓
Portfolio Risk
        ↓
Account Risk
        ↓
Execution
```

---

## Strategy Risk

Per-strategy:

```text
inventory limits
cash limits
spread limits
```

---

## Portfolio Risk

Across strategies:

```text
sector exposure
inventory aggregation
capital allocation
```

---

## Account Risk

Final hard stop.

Uses actual exchange balances.

---

# Service Decomposition

Today almost everything lives inside one runtime. 

For large scale I would split into:

```text
Market Data Service
Private State Service
Execution Service
Portfolio Service
Strategy Workers
Control Plane
Persistence Service
Observability Stack
```

---

# Market Data Service

Independent service.

Responsibilities:

```text
exchange websocket
book building
trade normalization
publishing
```

One orderbook build per symbol.

Not per strategy.

---

Instead of:

```text
100 BTC strategies
↓
100 websocket connections
```

do:

```text
1 websocket
↓
normalized stream
↓
100 subscribers
```

---

# Private State Service

Independent service.

One per exchange account.

Maintains:

```text
balances
orders
fills
positions
```

Publishes:

```text
account state events
```

---

# Execution Service

I would absolutely separate execution.

Current execution is embedded in StrategyRunner. 

Target:

```text
Strategy
    ↓
Target Book
    ↓
Execution Service
```

Execution owns:

```text
order managers
exchange adapters
reconciliation
```

---

# Strategy Workers

These become stateless compute workers.

Input:

```text
market state
portfolio state
allocation state
```

Output:

```text
target book
```

Nothing else.

---

# Kubernetes Model

I would not run:

```text
1 strategy = 1 pod
```

unless strategies are extremely heavy.

That becomes:

```text
1000 strategies
=
1000 pods
```

Operational nightmare.

---

Recommended:

```text
Strategy Worker Pod
     ↓
50-200 strategies
```

per pod.

Workers load strategy definitions dynamically.

---

# Process Model

For Python:

## IO

Use:

```text
asyncio
```

for:

* websockets
* REST
* event transport

Current architecture already does this well. 

---

## CPU

Use:

```text
multiprocessing
```

or

```text
distributed workers
```

for MPC.

Never threads.

The GIL will hurt.

---

# MPC Scaling

This is probably the largest compute cost.

Potential gain ranking:

### Highest ROI

Batch optimization across strategies.

Today:

```text
strategy A optimize
strategy B optimize
strategy C optimize
```

Instead:

```text
batch optimize
```

especially on GPU.

Huge gain.

---

### Second

GPU PyTorch.

Likely:

```text
5-20x
```

improvement.

---

### Third

Reduce Monte Carlo sample count adaptively.

Most market states do not require full optimization.

---

### Fourth

Move rollout engine into Rust/C++.

Only if profiling proves optimizer dominates.

---

# Messaging

Current Redis Streams is fine up to moderate scale. 

My recommendation:

### Phase 1

Keep Redis Streams.

---

### Phase 2

Move to:

Apache Kafka

for:

```text
market data
execution events
replay
```

Kafka is a better fit than:

* RabbitMQ
* NATS

for trading-event history.

---

# Event Partitioning

Partition by:

```text
exchange + symbol
```

Example:

```text
wallex.btcusdt
wallex.ethusdt
```

This preserves ordering where it matters.

---

# Persistence

I would evolve into:

### Postgres

Use for:

```text
control plane
config
users
allocations
orders
```

---

### Object Storage

Use for:

```text
event archive
replay
market history
```

Example:

Amazon S3-style storage.

---

### Prometheus

Keep for metrics.

---

### Grafana

Keep for dashboards.

Current observability foundation is already good. 

---

# Control Plane

This becomes the heart of the system.

Responsibilities:

```text
Exchange registry
Account registry
Portfolio registry
Strategy registry
Deployments
Lifecycle
Capital allocation
Risk controls
```

---

New hierarchy:

```text
Exchange
    ↓
Account
    ↓
Portfolio
    ↓
Strategy
```

not:

```text
Exchange
    ↓
Strategy
```

---

# Migration Roadmap

## Phase 1

Keep current runtime.

Extract:

```text
Market Data Service
Private State Service
```

---

## Phase 2

Introduce:

```text
Exchange Ledger
Portfolio Ledger
Strategy Ledger
```

This is the most important phase.

---

## Phase 3

Extract:

```text
Execution Service
```

---

## Phase 4

Introduce:

```text
Control Plane
Strategy Registry
Capital Allocation Service
```

---

## Phase 5

Move strategy execution to:

```text
Strategy Worker Pool
```

---

## Phase 6

Introduce:

```text
Kafka
GPU MPC
Kubernetes autoscaling
```

only when needed.

---

If I were the architect of this platform, the single most important design decision would be:

```text
Exchange Account
        ↓
Exchange Ledger
        ↓
Portfolio
        ↓
Capital Allocation
        ↓
Strategy Ledger
        ↓
Strategy
```

and I would explicitly prevent strategies from directly owning exchange balances. That decision will make hundreds of strategies, shared accounts, centralized risk, portfolio optimization, and future multi-strategy capital management all much easier to support.
