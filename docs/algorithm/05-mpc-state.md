# 05 — MPC state and action: the canonical data structures

This section defines the two objects that everything in the optimizer reads and
writes: the `State` (input) and the `SigmoidMixtureAction` (output). Both have
two implementations:

- a **NumPy-friendly** form in `mm/mpc/mpc_state.py` (used by the strategy layer
  for diagnostics and warm-start),
- a **PyTorch tensor** form in `mm/mpc/state_tensor.py` (used inside the optimizer
  so that everything is differentiable through `torch.autograd`).

The NumPy and tensor versions are kept in sync by `state_from_*` and
`action_from_*` helpers; the optimizer never crosses the boundary inside a hot
path.

## 5.1 Notation used in this section

- `mid` — current mid price.
- `ob_rel` — the orderbook in *relative* coordinates (distance from mid, signed).
- `inventory` — net base position, positive = long.
- `cash` — quote balance.
- `V₊, V₋` — buy and sell market volumes in the last window.
- `vwap` — volume-weighted average price of the trades.
- `book_buy_vwap, book_sell_vwap, book_vwap_depth` — depth aggregates of the
  resting book.
- `L, k, x0` — sigmoid parameters (see §5.4).
- `n_components` — number of sigmoids per side (the "mixture size").
- `reservation_r_offset` — small spread shift used to incentivize inventory
  rebalancing.

## 5.2 `OrderbookRelative`

A relative orderbook is a 2-tuple `(s, q)` of signed `np.ndarray` representing
*spread* (price − mid) and *quantity* per level. Both sides are stacked: levels
0..N-1 are bids (negative spread), levels N..2N-1 are asks (positive spread).

```1:60:mm/mpc/mpc_state.py
@dataclass(slots=True)
class OrderbookRelative:
    spread: np.ndarray       # shape (2N,), signed
    quantity: np.ndarray     # shape (2N,), non-negative
    @property
    def bid_mask(self) -> np.ndarray: ...
    @property
    def ask_mask(self) -> np.ndarray: ...
```

Conversion to relative is done by `ob_rel_from_market_event` in
`mm/mpc/joint_mpc.py:ob_rel_from_market_event`, which subtracts the local mid
from every level price.

## 5.3 `State`

The MPC state is everything the optimizer needs to evaluate the objective on a
single decision. It is the *output* of `MpcStrategyEngine._build_state` and the
*input* of `MonteCarloMPCTensorBatched.optimize`.

```90:160:mm/mpc/mpc_state.py
@dataclass(slots=True)
class State:
    mid: float
    ob_rel: OrderbookRelative
    buy_vol: float
    sell_vol: float
    inventory: float
    cash: float
    vwap: float
    book_buy_vwap: float
    book_sell_vwap: float
    book_vwap_depth: float
    inventory_position: float              # notional
    reference_price: float                 # usually the global mid
    reservation_r_offset: float = 0.0
    global_ob_rel: OrderbookRelative | None = None
    global_mid: float | None = None
    global_status: str = "disabled"        # "ready" / "missing" / "stale"
```

The state also carries the *reference price* used for inventory marking and PnL
attribution (typically the global mid if available, else the local mid). The
`reservation_r_offset` is a small spread shift driven by inventory and the
venue-specific `reservation_skew` config (see
`mm/mpc/objective_terms.py:VenueOptimizerParams.reservation_skew`).

## 5.4 `SigmoidMixtureAction` — the heart of the parameterization

The action is *not* a list of orders. It is a **sum of `n_components` sigmoids
per side**, where each sigmoid is a smooth step from 0 to `L` as the spread
increases, with steepness `k` and center `x0`:

```1:60:mm/mpc/mpc_state.py
@dataclass(slots=True)
class SigmoidMixtureAction:
    L_bid: np.ndarray        # shape (n_components,), cumulative depth caps
    k_bid: np.ndarray        # shape (n_components,), steepness
    x0_bid: np.ndarray       # shape (n_components,), center
    L_ask: np.ndarray
    k_ask: np.ndarray
    x0_ask: np.ndarray
    reservation_r_offset: float = 0.0
```

Cumulative depth on the bid side at a given spread `s ≥ 0` is:

```
D_bid(s) = Σ_{i=1..K} L_bid[i] · σ(k_bid[i] · (s − x0_bid[i]))
```

with `σ(z) = 1 / (1 + exp(−z))`. The ask side is the mirror with `s ≤ 0`
(matched by negating `s`). The `reservation_r_offset` shifts the center
locations so that the curve is evaluated at `s − r` (or `s + r` for asks) — this
is how inventory pressure is implemented.

### 5.4.1 Why a mixture of sigmoids?

A single sigmoid cannot express the empirically observed shape of a real order
book (flat near the inside, steep far away). A sum of `K` sigmoids with
monotonically increasing centers gives a *convex, monotone* depth curve that
matches the empirical depth well with only a handful of parameters per side
(typically K = 3..6).

### 5.4.2 Effective action

`EffectiveSigmoidMixtureAction` is a thin view of `SigmoidMixtureAction` that
*also* returns the executable depth `D(s) − D(0)`. The MPC optimizer works in
this "effective" space so that the inside level starts at zero additional
quantity (the optimizer never posts inside the best bid/ask) unless
`reservation_r_offset` is non-zero.

### 5.4.3 `Action` and `JointAction`

`Action` is a *single-level* representation used for diagnostics (one price,
one quantity, one side). The optimizer produces a `SigmoidMixtureAction`, and
`Action` is reconstructed by the target book builder when it lays the ladder
out level-by-level.

## 5.5 L-parameter decoding: turning raw parameters into `(L, k, x0)`

The optimizer never touches `(L, k, x0)` directly. It works in a *raw* parameter
space of length `3 * n_components` per side, and the structured decoder in
`mm/mpc/l_param.py` maps the raw vector to the effective `(L, k, x0)`:

```110:160:mm/mpc/l_param.py
@dataclass(frozen=True, slots=True)
class LParamConfig:
    n_components: int
    quantity_floor: float = 0.0
    quantity_ceiling: float | None = None
    steepness_min: float = 4.0
    steepness_max: float = 60.0
    center_floor_ticks: float = 1.0
    center_ceiling_ticks: float | None = None
    tick_size: float = 0.01
    ...
```

`materialize_side_torch` does the decoding inside the optimizer:

- **L** is `sigmoid(raw_L) · (ceiling − floor) + floor`.
- **k** is `sigmoid(raw_k) · (steepness_max − steepness_min) + steepness_min`.
- **x0** is enforced monotone via cumulative softmax: each center is sorted in
  increasing order so the resulting depth curve is convex.

A few extra rules:

- `center_floor_ticks` is the minimum number of ticks from the mid for the
  *innermost* component — keeps the optimizer from crossing the inside.
- `cumulative_qty_floor_at_spread_max` and `cumulative_qty_cap_at_spread_max` are
  hard bounds on the depth at the maximum allowed spread.
- `min_spread_ticks` and `max_spread_ticks` bound the curve's range in tick
  units.

The output is a differentiable `ActionTensor` that the rest of the optimizer
uses directly.

## 5.6 Joint-venue extension: `JointSigmoidMixtureAction`

When the strategy runs in *joint* mode (one local venue + one global venue), the
action is the concatenation of two `SigmoidMixtureAction`s, one per venue:

```60:90:mm/mpc/joint_mpc.py
@dataclass(slots=True)
class JointSigmoidMixtureAction:
    local: SigmoidMixtureAction
    global_: SigmoidMixtureAction
```

The total parameter length is `joint_action_param_len(n_local, n_global)` which
is `2 * 3 * (n_local + n_global)`. The optimizer slices the flat vector into
local and global halves and optimizes them jointly because the consolidated
wealth objective depends on both.

## 5.7 State extension for joint venues

`enrich_state_for_joint` in `mm/mpc/joint_mpc.py` adds joint-venue fields to the
state:

- `global_ob_rel` and `global_mid` — global venue market data.
- `global_status` — `"ready"`, `"missing"`, `"stale"`, or `"disabled"`.
- `reference_price` — chosen by `reference_price_from_state` (usually the global
  mid if `ready`, else the local mid).

These fields are used by the consolidated wealth function and by the joint
optimization backend.

## 5.8 What gets serialized and what doesn't

For warm-starting, the runner stores the **raw flat parameter vector** of the
last decision (not the decoded `(L, k, x0)`). The vector is the same shape
across decisions as long as `n_components` does not change, which it does not
in production. Warm-start is detailed in §11.4.
