# 09 — From sigmoid action to a venue-ready `JointTargetBook`

The MPC produces a *continuous* `SigmoidMixtureAction`. The exchange does not
accept curves — it accepts a list of `(price, quantity)` orders. The conversion
happens in two stages:

1. `build_orders_from_sigmoid` lays the depth curve on a discrete price grid
   and produces a ladder of `Order`s.
2. `build_joint_target_book_from_sigmoid` combines the local and global ladders
   (when in joint mode) into a `JointTargetBook` and a pipeline report.

The discretization must respect:

- the venue's `price_step` (tick size) and `qty_step` (lot size);
- the venue's `min_qty`, `min_notional`, and `max_orders_per_side`;
- the optimizer's structural limits (`max_spread_ticks`, `cumulative_qty_floor_at_spread_max`,
  `cumulative_qty_cap_at_spread_max`).

## 9.1 Notation

- `Δ_spread` — venue's price step (tick).
- `Δ_qty` — venue's quantity step (lot).
- `Q_min` — minimum order quantity.
- `N_min` — minimum order notional.
- `K` — maximum number of orders per side.
- `s_max` — maximum allowed spread (ticks from mid).
- `D(s)` — cumulative depth at spread `s` from the sigmoid action.

## 9.2 Laying the ladder

`ladder_orders_from_sigmoid_with_report` enumerates the price grid from
`s = price_step` (one tick from the mid) up to `s_max`, computes
`q(s) = D(s) − D(s − Δ_spread)` for each level, and rounds to the lot size.
Levels whose cumulative quantity is below `Q_min` or notional below `N_min`
are dropped.

```90:140:mm/core/order_constraints.py
def ladder_orders_from_sigmoid_with_report(action, mid, side, constraints):
    grid = _spread_grid(constraints)
    cumulative = _cumulative_depth(action, side, grid)
    per_level = np.diff(cumulative, prepend=0.0)
    orders = []
    for s, q in zip(grid, per_level):
        if q < constraints.qty_step:
            continue
        q_rounded = _quantize_qty(q, constraints.qty_step)
        if q_rounded < constraints.min_qty:
            continue
        price = mid + s if side == "ask" else mid - s
        if abs(price * q_rounded) < constraints.min_notional:
            continue
        orders.append(Order(price=price, quantity=q_rounded, side=side))
    return orders, LadderReport(...)
```

The function returns a `LadderReport` with diagnostics: the per-tick
cumulative depth (so the operator can plot the curve against the realized
depth) and a list of dropped levels with reasons.

## 9.3 Structural constraints

The decoder (`LParamConfig` in `mm/mpc/l_param.py`) enforces:

- `min_spread_ticks` — every level is at least this far from the mid;
- `max_spread_ticks` — no level is further than this;
- `cumulative_qty_floor_at_spread_max` — at the max spread, total depth
  must be at least this value (a "presence" floor);
- `cumulative_qty_cap_at_spread_max` — at the max spread, total depth
  must be at most this value (a "max exposure" cap).

These are *soft* in the optimizer (the L-param decoder allows them to be
violated slightly) but are *hard* in the target book builder: a level that
exceeds the cap is truncated, and a level that doesn't reach the floor is
filled to it.

## 9.4 `OrderConstraints`

```1:60:mm/core/order_constraints.py
@dataclass(frozen=True, slots=True)
class OrderConstraints:
    price_step: float
    qty_step: float
    min_qty: float
    min_notional: float
    max_orders_per_side: int
    ...
```

These are sourced from the `InstrumentSpec` (built from
`InstrumentPrecisionConfig` in the runtime config). The same constraints are
applied to every order produced by the strategy and to every order received
from the venue, so the runner and the order manager share a single source of
truth.

## 9.5 `apply_venue_constraints_with_report`

After the ladder is built, the report's pipeline applies:

- `apply_venue_constraints_with_report` — rounds price and quantity, drops
  orders that fail the venue rules, and produces a per-order reason list.
- `cap_orders_per_side_with_report` — keeps the `K` deepest orders per side
  and reports the rest as `capped:order_count`.

The final `ActionPipelineReport` carries:

- `n_orders_in` — orders before constraints;
- `n_orders_kept` — orders after constraints;
- `n_orders_capped` — orders dropped by the per-side cap;
- `n_orders_dropped_min_qty`, `n_orders_dropped_min_notional`,
  `n_orders_dropped_price_step` — orders dropped by the venue rules;
- `cumulative_qty_at_spread_max_actual` — actual depth at the cap, useful
  for verifying the floor/cap constraints.

## 9.6 `JointTargetBook`

`build_joint_target_book_from_sigmoid` runs the pipeline for the local venue
and (optionally) the global venue, and assembles the result into a
`JointTargetBook`:

```40:80:mm/strategies/target_book_builder.py
def build_joint_target_book_from_sigmoid(local_action, global_action, ...):
    local_book, local_report = build_venue_target_book(local_action, local_spec, ...)
    if global_action is not None:
        global_book, global_report = build_venue_target_book(global_action, global_spec, ...)
    else:
        global_book, global_report = None, None
    return JointTargetBook(
        local=local_book,
        global_=global_book,
        pipeline_reports=(local_report, global_report),
    ), ActionPipelineReport(combined=...)
```

The `JointTargetBook` is the input to the risk engine. After risk approval it
becomes the input to the order manager (see [04-strategy-runner.md](04-strategy-runner.md)).
