# Lesson 08 - `mm/bus/memory.py`

Original file: [`mm/bus/memory.py`](../../mm/bus/memory.py)

Lesson 7 showed the bus interface. This file shows a concrete implementation:

```python
class InMemoryBus(AbstractEventBus):
```

It is a development/local event bus. It keeps subscribers in memory and delivers
published events to subscriber queues.

The key lesson:

```text
an implementation satisfies an interface by filling in the real behavior
```

## Imports

```python
import itertools
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any
```

`itertools.count` gives an endless counter for subscriber IDs.

`defaultdict(list)` creates missing topic entries automatically.

`AsyncIterator` describes the async stream returned by `subscribe`.

`Any` means the bus can transport any event object.

## Local imports

```python
from mm.bus.base import AbstractEventBus
from mm.bus.delivery import (
    BusBackpressurePolicy,
    TrackedSubscriberQueue,
    create_subscriber_queue,
    deliver_to_subscriber,
    record_subscriber_queue_metrics,
)
from mm.observability.event_payloads import observe_runtime_event
```

This file composes smaller pieces:

- the interface comes from `base`
- queue/backpressure helpers come from `delivery`
- observability hook comes from `observability`

Common real-app habit:

Keep the main class readable by moving specialized helper behavior into focused
modules.

## Class docstring

```python
"""Development event bus: one tracked asyncio.Queue per subscriber."""
```

This tells us the intended use:

- development/local runtime
- one queue per subscriber

That is enough to understand the design.

## Constructor

```python
def __init__(
    self,
    max_queue_size: int = 1024,
    *,
    backpressure_policy: BusBackpressurePolicy = "block",
):
```

The constructor accepts queue behavior settings.

`max_queue_size` defaults to `1024`.

`backpressure_policy` is keyword-only because it comes after `*`.

This makes calls clearer:

```python
InMemoryBus(max_queue_size=100, backpressure_policy="drop_oldest")
```

## Instance state

```python
self._max_queue_size = int(max_queue_size)
self._backpressure_policy = backpressure_policy
self._subscribers: dict[str, list[TrackedSubscriberQueue]] = defaultdict(list)
self._subscriber_ids = itertools.count(1)
self._closed = False
```

These private attributes store the bus state.

Leading underscores mean "internal to this class."

`_subscribers` maps:

```text
topic -> list of subscriber queues
```

For example:

```text
"strategy:actions" -> [queue1, queue2]
```

`itertools.count(1)` creates IDs:

```text
1, 2, 3, 4, ...
```

`_closed` is a lifecycle flag.

## Publishing

```python
async def publish(self, topic: str, event: Any) -> None:
    if self._closed:
        return
    observe_runtime_event(topic, event, stage="bus_publish")
    subscribers = list(self._subscribers.get(topic, ()))
    for queue in subscribers:
        await deliver_to_subscriber(
            queue,
            event,
            max_queue_size=self._max_queue_size,
            policy=self._backpressure_policy,
        )
    if subscribers:
        record_subscriber_queue_metrics(subscribers)
```

This is the heart of the implementation.

Step by step:

1. if the bus is closed, ignore the publish
2. record an observability event
3. copy the current subscribers for the topic
4. deliver the event to each subscriber queue
5. record queue metrics

## Why copy the subscriber list?

```python
subscribers = list(self._subscribers.get(topic, ()))
```

The code copies the subscriber list before iterating.

Why?

Subscribers might be added or removed while publishing is happening. Iterating
over a copy avoids surprises from mutating the original list during iteration.

This is a small but common defensive programming habit.

## Fan-out

```python
for queue in subscribers:
    await deliver_to_subscriber(...)
```

One published event is delivered to every subscriber of that topic.

This is called fan-out.

For example:

```text
market event -> frame assembler
market event -> persistence writer
market event -> observability subscriber
```

The producer does not know all these consumers. It only publishes to a topic.

## Backpressure is delegated

```python
await deliver_to_subscriber(
    queue,
    event,
    max_queue_size=self._max_queue_size,
    policy=self._backpressure_policy,
)
```

The bus does not inline queue-full behavior. It delegates to
`deliver_to_subscriber`.

That helper handles policies such as:

- block
- drop oldest
- drop newest

Common design habit:

When a detail has its own policy and tests, move it into a helper module.

## `collect` helper

```python
async def collect(self, topic: str, limit: int) -> list[Any]:
    out = []
    async for event in self.subscribe(topic):
        out.append(event)
        if len(out) >= limit:
            break
    return out
```

This helper subscribes and collects a fixed number of events.

It is probably useful in tests or small tools.

Pattern:

```text
consume async stream until condition, then return list
```

Notice that `out = []` has no explicit type. Python can infer enough for human
readers here, and the return annotation says `list[Any]`.

## Subscribing with a queue

```python
def subscribe_queue(self, topic: str) -> TrackedSubscriberQueue:
    """Register a subscriber queue explicitly for tests and in-process actors."""
    queue = create_subscriber_queue(
        topic,
        f"subscriber-{next(self._subscriber_ids)}",
        max_queue_size=self._max_queue_size,
    )
    self._subscribers[topic].append(queue)
    return queue
```

This method creates a tracked queue, stores it under the topic, and returns it.

The subscriber ID uses an f-string:

```python
f"subscriber-{next(self._subscriber_ids)}"
```

Each call gets a new ID:

```text
subscriber-1
subscriber-2
subscriber-3
```

These IDs are useful for metrics and debugging.

## Overriding the base helper

`AbstractEventBus` already had a `subscribe_queue` method, but `InMemoryBus`
overrides it.

Why?

The base version creates a plain `asyncio.Queue` and starts a background
forwarding task.

The in-memory implementation can do better: it can register a tracked queue
directly in its subscriber table.

This is a common object-oriented pattern:

```text
base class provides generic behavior; subclass overrides with a more direct implementation
```

## Unsubscribe

```python
async def _unsubscribe(self, topic: str, queue: TrackedSubscriberQueue) -> None:
    subscribers = self._subscribers.get(topic)
    if subscribers and queue in subscribers:
        subscribers.remove(queue)
```

This private helper removes a queue from the subscriber list.

It is async even though it currently does no `await`.

Why might that be?

It keeps the shape compatible with implementations where unsubscribe might need
async cleanup. It also lets `subscribe` use `await self._unsubscribe(...)` in a
consistent way.

## Async generator subscription

```python
async def subscribe(self, topic: str) -> AsyncIterator[Any]:
    queue = self.subscribe_queue(topic)
    try:
        while not self._closed:
            yield await queue.get()
    finally:
        await self._unsubscribe(topic, queue)
```

This is an async generator.

It combines `async def` with `yield`.

Callers can consume it with:

```python
async for event in bus.subscribe(topic):
    ...
```

Inside the loop:

```python
yield await queue.get()
```

means:

1. wait for the next queue item
2. yield it to the subscriber
3. come back and wait for the next item

## `try/finally`

```python
try:
    while not self._closed:
        yield await queue.get()
finally:
    await self._unsubscribe(topic, queue)
```

The `finally` block runs when the subscriber stops consuming, errors, or is
cancelled.

This prevents abandoned queues from staying registered forever.

Common real-app habit:

Use `try/finally` for cleanup around long-running streams.

## Closing

```python
async def close(self) -> None:
    self._closed = True
    self._subscribers.clear()
```

Closing marks the bus as closed and clears all subscribers.

After this, `publish` returns immediately.

This is lifecycle management. Real services need a way to release resources and
stop background work cleanly.

## What this implementation does not do

This bus is in-memory, so:

- it does not persist events
- it does not share events across processes
- events disappear if the process exits
- subscribers only exist inside this Python process

That is fine for local development and tests. For durability or cross-process
communication, the architecture can use Redis Streams instead.

This is why the abstract interface matters.

## Big lesson

`InMemoryBus` teaches these real-code patterns:

- implement an abstract interface with a concrete class
- store subscribers by topic
- use queues for async communication
- fan out one event to many subscribers
- delegate backpressure policy to helper functions
- record observability during infrastructure operations
- use `try/finally` to clean up stream subscriptions
- provide a `close` lifecycle method

## Tiny exercise

Answer these:

1. What does `_subscribers` store?
2. Why does `publish` copy the subscriber list before iterating?
3. What does `yield await queue.get()` do?
4. Why does `subscribe` use `try/finally`?
5. Why is an in-memory bus useful even if production might use Redis Streams?

