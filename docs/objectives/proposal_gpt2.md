Based on the code-derived documentation, the current system is already surprisingly close to a good architecture for balancing profitability, inventory safety, and book quality. The biggest issue is that several existing inventory-control mechanisms are compensating for a simulation deficiency: **adverse selection is not structurally present in rollouts** because flow and price innovations are largely sampled independently.

As a result, the optimizer is rewarded for providing liquidity in situations where, in reality, fills would tend to occur before adverse price moves.

---

# 1. First: classify existing constraints/mechanisms

## A. Imbalance skew floors

Current:

```text
bid_floor = quote_size_min_per_side * (...)
ask_floor = quote_size_min_per_side * (...)
```

Location:

* dynamic capacity computation inside MPC rollout
* likely `mm/mpc/mpc_tensor_batched.py`

### Classification

**(c) Replace with different architecture**

Reason:

The current mechanism manipulates **size floors** to force inventory rebalancing.

Inventory management should primarily occur through:

```text
price skew
quote offset
reservation price shift
```

not through:

```text
artificial minimum sizes
```

Size-based inventory control produces ugly books:

* one side becomes oversized
* other side disappears
* depth profile becomes distorted

Instead:

```text
inventory → reservation price shift
```

and

```text
inventory → quote asymmetry
```

should be the primary channel.

Keep only a tiny structural minimum-size floor needed for execution viability.

---

## B. Capacity ratio

Current:

```text
bid_max=min(max_side_size,
            capacity_ratio * cash / mid)

ask_max=min(max_side_size,
            capacity_ratio * inventory)
```

### Classification

**(a) Hard structural constraint**

Reason:

This is not a preference.

It represents actual deployable capital.

A solution violating it is infeasible.

Ideally this should move into action decoding itself.

Decoder should never produce:

```text
ask size > inventory
bid size > cash/mid
```

by construction.

---

## C. Quote size max per side

### Classification

**(a) Hard structural constraint**

Reason:

Operational limit.

Represents concentration risk and execution risk.

Should remain hard.

---

## D. Quote size min per side

### Classification

Split:

### Execution minimum

```text
exchange minimum viable quoting size
```

→ **(a) Hard constraint**

### Inventory-skewed minimum

Current imbalance-floor logic

→ **(c) Replace**

---

## E. Spread cap

Current:

```text
quote_spread_max
```

### Classification

Mostly **(a) hard structural**

Reason:

Parameterization assumes bounded support.

Without it optimizer can:

```text
hide inventory risk
```

by placing orders infinitely far away.

Need finite action space.

However:

Current value must be re-evaluated (see later).

---

## F. Fractional inventory sizing

Any logic like:

```text
inventory high
→ reduce ask size
```

### Classification

**(c) Replace**

Inventory control should happen through:

```text
price
```

not:

```text
shrinking liquidity
```

unless actual inventory capacity is reached.

---

# 2. Highest-priority fix: adverse selection coupling

Current architecture:

Flow model:

```text
sample buy_vol
sample sell_vol
```

Price model:

```text
sample residual
sample mixture innovation
```

independently.

This is the single biggest realism issue.

---

# Proposed architecture

## Shared latent factor

For each path:

```text
z_h ~ N(0,1)
```

Generate both:

```text
flow
price innovation
```

from z.

---

Example

Let

```text
f_h = net_flow_h
```

Then:

```text
f_h = flow_model(z_h, eps_f)
```

and

```text
price_h = price_model(
    mechanical_h,
    z_h,
    eps_p
)
```

where

```text
corr(z_h, price_h) > 0
corr(z_h, flow_h) > 0
```

---

Meaning

Large:

```text
buy flow
```

path tends to coincide with:

```text
upward price innovation
```

Large:

```text
sell flow
```

path tends to coincide with:

```text
downward price innovation
```

creating adverse selection naturally.

---

# Exact files

Primary:

### `mm/mpc/mpc_tensor_batched.py`

Current rollout sampling.

Introduce:

```python
latent_z = torch.randn(...)
```

per:

```text
sample
horizon
```

---

### `mm/mpc/market_order_flow_torch.py`

Add:

```python
sample_conditional(z)
```

or

```python
sample_with_latent(z)
```

---

### `TorchConditionalMechanicalMixturePriceModel`

Add:

```python
sample_dynamic_vwap(
    z=z
)
```

---

# Mathematical structure

Simplest version

```text
z ~ N(0,1)

flow = F(theta,z,eps1)

price = G(theta,
          mechanical,
          rho*z + sqrt(1-rho²)*eps2)
```

with

```text
rho learned online
```

---

# Effort

Phase 1

fixed rho:

~80–120 LOC

---

Phase 2

online rho estimation:

~150 LOC

---

Phase 3

copula model:

~200–400 LOC

---

I would stop at Phase 2.

---

# 3. Inventory risk redesign

After adverse selection is fixed:

Inventory risk naturally enters wealth.

Because:

```text
long inventory
+
negative innovation
```

creates losses.

Thus:

```text
CVaR / CE
```

already penalizes inventory.

---

## Separate operational inventory term

Keep explicit inventory objective.

Purpose:

NOT risk.

Purpose:

capacity management.

---

Desired shape

```text
zone of indifference
```

Example:

```text
inventory_ratio=q/qmax
```

Penalty:

```text
0
```

inside:

```text
[-0.2,+0.2]
```

Then smooth growth.

Example:

```text
P(q)=0
      if |q|<band

P(q)=A*(|q|-band)^2
      otherwise
```

bounded near limits.

---

Location

### `mm/mpc/reward_tensor.py`

Add:

```text
inventory_capacity_term
```

---

Classification

Soft objective:

**(b)**

---

# Interim stopgap if coupling not implemented

Need explicit inventory risk term.

Because current CE underestimates risk.

Use:

```text
inventory² * realized_vol
```

temporary penalty.

Remove once coupling is implemented.

---

# 4. Price shading vs size capping

This is the most important decoder change after adverse selection.

---

Current

Inventory pressure affects:

```text
sizes
capacity
minimums
```

too much.

---

Desired

Inventory pressure affects:

```text
reservation price
```

primarily.

---

# Decoder change

Introduce inventory offset:

```text
r = anchor
    - λinv * inventory_ratio
```

Then:

```text
bid_prices around r
ask_prices around r
```

instead of anchor.

---

Files

### `mm/mpc/mpc_tensor_batched.py`

rollout quote construction

### `build_joint_target_book_from_sigmoid`

final discretization

---

# Hard inventory cap in decoder

Current:

```text
ask_max=min(...inventory...)
```

during rollout.

Good.

But make action parameterization itself inventory-aware.

Example:

```text
ask_total_depth =
    inventory_available *
    sigmoid(u_ask)

bid_total_depth =
    cash_available/mid *
    sigmoid(u_bid)
```

Now impossible to violate.

---

Classification

Hard feasibility.

---

# Interaction with spread cap

Current question:

Is quote_spread_max large enough?

---

Need diagnostics.

Compute:

```text
inventory skew offset
/
spread_max
```

If

```text
skew offset ≈ spread_max
```

then optimizer saturates.

Bad.

---

Rule of thumb

Need:

```text
max inventory skew
<
30%-50% spread_max
```

so optimizer still has room.

---

Would add diagnostic:

```text
reservation_shift /
spread_max
```

Location:

optimizer diagnostics.

---

# 5. Book-quality objective terms

These should all become soft objectives.

---

## Spread tightness

```text
spread_score
```

Reward:

```text
-smoothed_spread
```

---

## Two-sided presence

Reward:

```text
bid_depth>0
and
ask_depth>0
```

smooth approximation.

---

## Depth

Reward:

```text
total displayed depth
```

---

## Balance

Reward:

```text
1-|bid_depth-ask_depth|
        /(sum depth)
```

---

## Constant-slippage profile

Most valuable book-quality term.

Reward:

```text
variance(
marginal depth profile
)
```

low variance.

Equivalent to smooth liquidity curve.

---

# Dimensional scaling

Do NOT combine raw quantities.

Normalize every component.

Each becomes dimensionless:

```text
quality_j
```

scaled to:

```text
[0,1]
```

or

```text
[-1,1]
```

---

Then objective:

```text
J =
risk_adjusted_return

+ λ_book * Q_book

- λ_inv  * P_inv
```

where

```text
Q_book
=
w1 depth
+w2 balance
+w3 spread
+w4 profile
```

and

```text
Σ wi =1
```

---

# Smooth handoff between inventory and book quality

Desired behavior:

Near target inventory:

```text
inventory penalty = 0
```

Book-quality dominates.

Optimizer posts:

```text
tight
balanced
deep
```

book.

---

As inventory approaches limits:

```text
inventory penalty rises
```

Reservation price shifts.

Book becomes intentionally skewed.

---

Because penalty is smooth:

```text
C1 continuous
```

there is no discontinuity.

No regime switch.

Only gradual transition.

---

# Recommended implementation order (impact/hour)

| Rank | Proposal                                   | Impact          | Effort   |
| ---- | ------------------------------------------ | --------------- | -------- |
| 1    | Flow-price coupling (shared latent factor) | Very High       | Medium   |
| 2    | Reservation-price inventory skew           | Very High       | Low      |
| 3    | Decoder hard-cap by inventory/cash         | High            | Low      |
| 4    | Zone-of-indifference inventory objective   | High            | Low      |
| 5    | Remove imbalance-size floors               | Medium          | Very Low |
| 6    | Book-quality soft objective bundle         | Medium          | Medium   |
| 7    | Learn latent correlation online            | Medium          | Medium   |
| 8    | Copula-based sampler                       | Low incremental | High     |

If I were implementing this codebase, I would do:

1. **Flow-price coupling**
2. **Reservation-price inventory skew**
3. **Inventory-aware decoder**
4. **Zone inventory term**
5. **Book-quality objective**

Only after those five are complete would I consider more sophisticated copulas, queue models, or microstructure enhancements. They address the dominant structural deficiencies while preserving the existing MPC architecture.
