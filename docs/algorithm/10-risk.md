# 10 — Risk: `NetDeltaRiskEngine`, validity, viability

The risk layer is the **last gate** before orders go to the venue. It has two
parts:

1. **`NetDeltaRiskEngine`** — applies the *venue* and *actor* risk limits
   (max notional, allocation caps, kill switch).
2. **Validity and viability** — the *soft* constraints already in the MPC
   objective. The risk engine is a *hard* backstop: it discards any order
   that would push the venue outside the configured limits, even if the
   optimizer's soft penalties missed it.

## 10.1 Notation

- `P_base, P_quote` — base and quote prices.
- `B, Q` — base and quote balances (free + locked).
- `X` — net base inventory (signed).
- `T_base_locked, T_quote_locked` — base and quote amounts that the target
  orders would lock.
- `R_max_order, R_max_open` — per-order and per-venue notional limits.
- `alloc_local`, `alloc_global` — actor's allocation shares for each venue.

## 10.2 Frame-level gates: `_frame_gate_reasons`

Before the per-venue checks, the engine inspects the `JointStrategyFrame` for
reasons to *skip the entire decision*:

```80:110:mm/risk/net_delta.py
def _frame_gate_reasons(frame, ledger, *, max_frame_age_ms, max_ledger_age_ms):
    reasons = []
    if frame.input.local.t_end_ms + max_frame_age_ms < now_ms():
        reasons.append("stale_market_data")
    if ledger.last_update_ms + max_ledger_age_ms < now_ms():
        reasons.append("stale_ledger")
    if frame.input.global_status == "stale":
        reasons.append("stale_global")
    if frame.input.global_status == "missing":
        reasons.append("missing_global")
    ...
    return reasons
```

If any reason is present, the engine returns an empty `TargetBookRiskResult`
with `approved_target_book=None` and the reason list. The runner then
publishes a "skipped" decision and the venue's open orders are *not* modified.

## 10.3 Per-venue checks: `evaluate_venue_target_transition`

For each venue the engine checks:

- **max order notional** — `price * qty ≤ R_max_order`.
- **max open notional** — `Σ (price * qty) over the venue's open orders
  ≤ R_max_open`.
- **insufficient base / quote** — would the proposed order push the venue
  balance below zero?
- **min cash / inventory** — is the venue allowed to operate with this
  much cash or inventory? (typically a safety floor).
- **max inventory** — is the venue allowed to take this much inventory?
  (typically a safety ceiling).
- **allocation caps** — would the proposed exposure push the actor's
  `alloc_local` (or `alloc_global`) above the configured cap?

```50:90:mm/risk/net_delta.py
def evaluate_venue_target_transition(target_orders, ledger, *, limits):
    base_locked = sum(o.qty for o in target_orders if o.side == "buy")
    quote_locked = sum(o.price * o.qty for o in target_orders if o.side == "buy")
    ...
    if base_locked > ledger.free_base + eps:
        return VenueRiskResult(approved=False, reason="insufficient_base")
    if quote_locked > ledger.free_quote + eps:
        return VenueRiskResult(approved=False, reason="insufficient_quote")
    ...
```

The function returns a `VenueRiskResult` with a list of `accepted` and
`rejected` orders. The engine is *partial-approval capable*: if a venue has
five bids and one of them would over-spend the cap, the other four are
approved and the fifth is rejected with `reason="exceeds_max_notional"`.

## 10.4 Kill switch and allocation breaker

The engine consults the `ControlPlane` for:

- `actor_kill_switch[actor].armed` — if true, every order is rejected with
  `reason="actor_kill_switch"`.
- `actor_allocation_breaker[actor].tripped` — if true, every order is
  rejected with `reason="actor_allocation_breaker"`. The breaker is tripped
  by a *separate* monitor that watches the actor's inventory against a
  configured `max_inventory` (this is in `mm/control/allocation_breaker.py`).

The kill switch is the *emergency stop*; the allocation breaker is the
*graduated* stop. The kill switch is a hard stop with no auto-recovery; the
allocation breaker resets when the actor's exposure returns inside the
configured band.

## 10.5 Validity backstop

The optimizer already penalizes a low `book_quality_score`, but it may still
return a structurally bad book if the soft penalty is small relative to the
PnL gradient. The `NetDeltaRiskEngine` does *not* re-evaluate validity, but
the strategy runner checks `decision.target_book.book_quality_score` against
`min_book_quality_score` (configured per actor). If the score is below the
threshold, the runner drops the book and logs `reason="low_book_quality"`.

## 10.6 Viability backstop

The viability dual variable `λ` is updated *after* every decision by
`update_viability_lambda`. The function EWMA-filters the realized
quadratic deficit and steps a bounded linear multiplier:

```text
S̄ ← α S̄ + (1−α) S
λ ← clip(λ + η (S̄ − ε), 0, λ_max)
```

The updated `λ` is kept on the MPC instance and used in the next decision.

## 10.7 `validate_target_book_with_result`

The top-level entry point composes everything:

```130:170:mm/risk/net_delta.py
def validate_target_book_with_result(target_book, frame, ledger, *, limits, control_plane):
    frame_reasons = _frame_gate_reasons(frame, ledger, ...)
    if frame_reasons:
        return TargetBookRiskResult(approved=False, reasons=frame_reasons)
    local_result = evaluate_venue_target_transition(target_book.local, ledger.local, ...)
    global_result = evaluate_venue_target_transition(target_book.global_, ledger.global_, ...)
    return TargetBookRiskResult(
        approved_target_book=JointTargetBook(
            local=local_result.accepted, global_=global_result.accepted
        ),
        rejected_orders=local_result.rejected + global_result.rejected,
        reasons=local_result.reasons + global_result.reasons,
    )
```

The runner then publishes the decision with both the approved book and the
list of rejections, so the operator can see exactly which orders were
filtered.
