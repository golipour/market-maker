# 04 — Strategy runner: the per-decision loop

`mm/application/strategy_runner.py` is the **per-actor decision loop**. It subscribes
to the per-actor `JointStrategyFrame` topic, runs the MPC optimizer in a worker
thread, validates the result through the `NetDeltaRiskEngine`, and routes the
approved target book to the order manager.

## 4.1 High-level control flow

For each `JointStrategyFrame`:

1. Read the current open orders from the ledger (`ledger.snapshot_open_orders()`).
2. **Clean** the local and global books by subtracting the bot's own quotes
   (so the optimizer does not see itself in the depth).
3. Call `MpcStrategyEngine.decide(clean_frame, ledger)` on a worker thread with a
   timeout (`optimizer_timeout_ms`).
4. Run `NetDeltaRiskEngine.validate_frame(decision, ledger)` to filter the
   `JointTargetBook` to the *approved* orders.
5. Publish:
   - the `StrategyDecision` to `decisions:<actor>` (always);
   - the `JointTargetBook` to the bus;
   - the `ControllerDiagnostics` payload to `diagnostics:controller:<actor>`.
6. If `execution_gate` is open, route the approved book to the order manager;
   otherwise route to the preview bus.

## 4.2 Subtracting own orders: `clean_strategy_frame`

The MPC needs the *market* depth, not the depth it has just posted. Otherwise the
optimizer would "compete with itself". The cleanup is performed by
`mm/application/frame_preparation.py:clean_strategy_frame`:

```15:42:mm/application/frame_preparation.py
def clean_strategy_frame(frame, open_orders):
    clean_local = _replace_event_orderbook(
        frame.local,
        subtract_own_orders(frame.local.orderbook, open_orders),
    )
    clean_global = None
    if frame.global_ is not None:
        clean_global = _replace_event_orderbook(
            frame.global_.orderbook,
            subtract_own_orders(frame.global_.orderbook, open_orders),
        )
    return StrategyInputFrame(...)
```

`subtract_own_orders` walks each level and subtracts the bot's resting qty from the
opposite side; the book VWAP aggregates are recomputed on the cleaned book
(`book_vwaps` in `mm/aggregation/features.py`).

## 4.3 The decision coroutine

```73:96:mm/application/strategy_runner.py
async def run_step(self, joint_frame: JointStrategyFrame) -> None:
    open_orders = self.ledger.snapshot_open_orders(actor=self.actor)
    clean_frame = clean_strategy_frame(joint_frame.input, open_orders)
    decision = await self._run_optimizer(clean_frame, joint_frame.joint_ledger)
    if decision is None:
        return
    target_book = decision.target_book
    risk_result = self.risk_engine.validate_frame(target_book, joint_frame.joint_ledger)
    approved = risk_result.approved_target_book
    self._publish_decision(decision, risk_result)
    await self._dispatch(approved, joint_frame)
```

The four side effects of every step are:

- a `StrategyDecision` published to the decisions topic (always);
- a `ControllerDiagnostics` payload published with optimizer stats and validity/viability;
- a `JointTargetBook` published to the execution plane (only when the execution gate
  is open);
- ledger updates from the eventual fills, which the runner does not perform itself
  but which the reconciler does on the next private snapshot.

## 4.4 Optimizer timeouts & cooperative cancellation

The optimizer is CPU-bound and runs in a dedicated `ThreadPoolExecutor` so the
asyncio event loop stays responsive. The runner uses a `threading.Event` for
cooperative cancellation:

```98:135:mm/application/strategy_runner.py
def __init__(self, ..., optimizer_timeout_ms=2_000, max_optimizer_workers=1):
    self._executor = ThreadPoolExecutor(max_workers=max_optimizer_workers)
    self._cancel_event = threading.Event()

async def _run_optimizer(self, frame, ledger):
    loop = asyncio.get_running_loop()
    self._cancel_event.clear()
    future = loop.run_in_executor(
        self._executor, self._decide_blocking, frame, ledger, self._cancel_event,
    )
    try:
        return await asyncio.wait_for(future, timeout=self.optimizer_timeout_ms / 1000)
    except asyncio.TimeoutError:
        self._cancel_event.set()     # ask the optimizer to stop at the next guard
        return None                  # runner skips this decision cycle
```

The optimizer checks the event at the end of every Adam epoch (see
[07-mpc-optimizer.md](07-mpc-optimizer.md)). A timed-out decision is recorded in
diagnostics as `"optimizer_status": "cancelled"` and the runner falls back to
"keep current orders" — it does **not** cancel live orders, because the previous
target book is still on the venue.

## 4.5 Execution gates and warmup

`compute_venue_execution_gates` decides whether the local and global venues are
allowed to send live orders. The result is a `VenueExecutionGate` per venue:

```10:30:mm/application/venue_execution.py
@dataclass(frozen=True)
class VenueExecutionGate:
    can_sync_live: bool
    reasons: tuple[str, ...] = ()
    dry_run: bool = False
```

Reasons include:

- `dry_run` — runtime configured to never send live orders.
- `warmup` — process has been alive for less than `actor.warmup_seconds`.
- `actor_paused` — operator hit the kill switch.
- `actor_allocation_breaker` — `NetDeltaRiskEngine` flagged an allocation breach.
- `venue_paused` — per-venue pause flag.
- `no_decision` — the previous step returned no decision (timeout, error).
- `target_book_empty` — the optimizer returned an empty book on purpose (e.g.
  negative-objective policy chose `zero_action`).

If the gate is closed, the runner publishes the target book to the *preview* bus
and emits a `ControllerDiagnostics` entry with the reason. No orders are sent.

## 4.6 What happens if no frame arrives

The runner does not time itself out; it only acts when a frame arrives. The
`MarketDataPlane` is responsible for emitting frames; if the symbol engine stalls
(adapter down), the next frame will be older than `max_frame_age_ms` and the
`JointFrameAssembler` will drop it, which causes the runner to keep its last
target book on the venue until fresh data arrives.
