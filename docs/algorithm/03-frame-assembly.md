# 03 — Frame assembly: from `MarketEvent`s to a `JointStrategyFrame`

The strategy does not subscribe to the bus directly. There is a dedicated *frame
assembler* per actor that:

1. Pulls local and (optionally) global `MarketEvent`s from the bus.
2. Pulls the latest `PrivateStateSnapshot`s (balances, open orders, fills) for each
   venue.
3. Aligns them in time (so the local and global windows close together, with a
   configurable grace).
4. Combines them with the actor's portfolio state (cash, inventory, ledger
   snapshot) to produce a `JointStrategyFrame`.

This section is the *seam* between ingestion and the strategy.

## 3.1 `StrategyInputFrameAssembler`

`mm/application/frame_assembler.py` holds the simpler per-actor assembler. It owns
three queues:

- `local_queue: asyncio.Queue[MarketEvent]`
- `global_queue: asyncio.Queue[MarketEvent] | None` (when a global venue exists)
- `local_private_queue: asyncio.Queue[PrivateStateSnapshot] | None`
- `global_private_queue: asyncio.Queue[PrivateStateSnapshot] | None`

The "local" role is the venue the actor is quoting; the "global" role is an
informational reference venue used for cross-venue features only.

```19:48:mm/application/frame_assembler.py
class StrategyInputFrameAssembler:
    def __init__(self, *, local_queue, global_queue=None,
                 local_private_queue=None, global_private_queue=None,
                 bus=None, output_topic=None,
                 max_global_lag_ms=2_000, global_grace_ms=250,
                 global_history_size=128, clock_ms=now_ms):
        ...
        self._global_events: deque[MarketEvent] = deque(maxlen=...)
        self._local_private: PrivateStateSnapshot | None = None
        self._global_private: PrivateStateSnapshot | None = None
```

The loop is started with `await assembler.start()`. The publish loop is `_run_local`:

```92:103:mm/application/frame_assembler.py
async def _run_local(self) -> None:
    while not self._stopped.is_set():
        frame = await self.next_frame()
        if self.bus is not None and self.output_topic:
            await self.bus.publish(self.output_topic, frame)
```

So the bus sees the *assembled* `StrategyInputFrame` on the `output_topic` configured
in runtime settings (e.g. `frames:default:local`).

### 3.1.1 Global alignment

`build_frame` waits up to `global_grace_ms` for a global event whose `t_end_ms` matches
the local `t_end_ms`:

```75:99:mm/application/frame_assembler.py
async def build_frame(self, local: MarketEvent) -> StrategyInputFrame:
    self._drain_latest_inputs()
    if self.global_queue is not None:
        await self._wait_for_matching_global(local.t_end_ms)
    self._drain_latest_inputs()
    global_event, global_status, global_lag_ms = self._select_global(local)
    return StrategyInputFrame(
        local=local, global_=global_event,
        local_private=self._local_private,
        global_private=self._global_private,
        global_status=global_status,
        global_lag_ms=global_lag_ms,
        max_global_lag_ms=self.max_global_lag_ms,
        created_at_ms=self.clock_ms(),
    )
```

`_select_global` returns a tuple `(event, status, lag_ms)` with one of four
`GlobalMarketStatus` values:

| Status | Meaning | Strategy reaction |
| --- | --- | --- |
| `disabled` | No global venue configured | Local-only MPC; `global_=None`. |
| `missing` | No global event in the history ring | Treat as `disabled` for MPC features, mark frame. |
| `stale` | Latest global is older than `max_global_lag_ms` | Mark frame, set `global_lag_ms`. |
| `ready` | Global event available within lag budget | Use as `global_`. |

The lag budget is enforced even when a global event is "available": any candidate
older than `local.t_end_ms − max_global_lag_ms` is downgraded to `stale`.

### 3.1.2 Why this matters

The MPC optimizer cares about the *spread* and *imbalance* of the global venue, and
about the *reference price* (often the global mid). If the global feed is stale, the
optimizer is told so via `frame.global_status` and `frame.global_lag_ms` so it can
choose a wider spread, switch to the local mid as a reference, or fall back to a
defensive policy.

## 3.2 `JointFrameAssembler`

`mm/application/joint_frame_assembler.py` is the per-actor object that subscribes to
the *assembled* `StrategyInputFrame` topic and combines it with the *joint* ledger
snapshot. The joint ledger snapshot is the actor's combined view across both venues
(cash, inventory, weighted reserved balances, allocation breaches). The frame becomes a
`JointStrategyFrame`.

```1:80:mm/application/joint_frame_assembler.py
class JointFrameAssembler:
    def __init__(self, *, bus, input_topic, joint_ledger,
                 output_topic, max_frame_age_ms=2000,
                 max_ledger_age_ms=2000, clock_ms=now_ms):
        self.bus = bus
        self.input_topic = input_topic
        self.joint_ledger = joint_ledger
        self.output_topic = output_topic
        ...
```

The freshness checks are *guard rails* for the strategy. A frame is dropped (and
counted) if:

- the market event is older than `max_frame_age_ms`; or
- the joint ledger snapshot is older than `max_ledger_age_ms`; or
- the ledger is "ahead" of the market by a noticeable amount (i.e. it contains
  fills whose origin time is in the future of `frame.local.t_end_ms`), which usually
  indicates a clock skew issue between the private feed and the public feed.

```70:99:mm/application/joint_frame_assembler.py
def _ledger_ahead_of_market_ms(self, frame, ledger):
    return (ledger.latest_fill_ts_ms or 0) - frame.local.t_end_ms
```

If everything is fresh, `build_joint_frame` is called:

```100:140:mm/application/joint_frame_assembler.py
def build_joint_frame(self, input_frame, joint_ledger):
    local_open = self._open_orders(input_frame.local_private)
    global_open = self._open_orders(input_frame.global_private)
    return JointStrategyFrame(
        input=input_frame,
        joint_ledger=joint_ledger,
        local_open_orders=local_open,
        global_open_orders=global_open,
        joint_open_orders={**local_open, **global_open},
    )
```

The `joint_open_orders` map is what the strategy runner uses to subtract its own
quotes from the local and global books (see [04-strategy-runner.md](04-strategy-runner.md)).

## 3.3 What "joint" really means

`JointStrategyFrame` is a struct that bundles the *input frame* (public market data)
with the *joint ledger* (combined venue state) and the *open orders* (currently
resting). Everything that the MPC needs to make a decision is reachable from this one
object. The frame is serialized and published to the strategy runner on a per-actor
topic such as `frames:default:joint`.

The freshness guard is the only place where the runtime can decide to **not** act on
new information. If the freshness budget is exhausted, the strategy runs *one more*
decision with the stale frame, then the next frame is dropped until both the public
and private feeds recover. This avoids a "thundering herd" effect after a network
blip where the runner would otherwise queue up many partial decisions.
