# Lesson 07 - `mm/bus/base.py`

Original file: [`mm/bus/base.py`](../../mm/bus/base.py)

This file defines the interface for the event bus.

The key lesson:

```text
real applications often define what a component must do before defining how one implementation does it
```

`mm/bus/base.py` says: "an event bus must support publish, subscribe, queue
subscription, and close."

`mm/bus/memory.py` then provides one concrete implementation.

## Imports

```python
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
```

`asyncio` is Python's async runtime library.

`ABC` and `abstractmethod` are used to define abstract base classes.

`AsyncIterator` describes something you can loop over with `async for`.

`Any` means events can be arbitrary payload objects.

## Abstract base class

```python
class AbstractEventBus(ABC):
```

An abstract base class defines a contract. It is not meant to be used directly.

Concrete subclasses must implement the abstract methods.

Why use this pattern?

- application code can depend on `AbstractEventBus`
- tests can use a fake or in-memory bus
- production can use a Redis-backed bus
- implementations can change without rewriting callers

This is the classic "program to an interface" idea.

## Abstract async method

```python
@abstractmethod
async def publish(self, topic: str, event: Any) -> None:
    ...
```

This says every event bus must provide an async `publish` method.

`async def` means callers use:

```python
await bus.publish(topic, event)
```

Why async?

Some bus implementations may do network I/O, such as Redis Streams. Even the
in-memory bus keeps the same async interface so callers do not care which
implementation is active.

Common real-app rule:

If one implementation needs async, the interface is often async for all
implementations.

## Topic and event

```python
topic: str
event: Any
```

The topic is the routing key, such as:

```text
strategy:actions
market:events:nobitex:BTC_IRT:1000
```

The event is the payload. It may be a dataclass, msgspec struct, dictionary, or
other object.

The bus interface does not care what the event is. It just transports it.

That is separation of concerns.

## Abstract subscription method

```python
@abstractmethod
def subscribe(self, topic: str) -> AsyncIterator[Any]:
    ...
```

This says a bus must let callers subscribe to a topic.

The return type is `AsyncIterator[Any]`, so callers can write:

```python
async for event in bus.subscribe("strategy:actions"):
    ...
```

This is a natural pattern for streams of events.

Beginner mental model:

- normal list: `for item in items`
- async stream: `async for item in event_stream`

## Why `subscribe` is not marked `async`

This can feel surprising:

```python
def subscribe(self, topic: str) -> AsyncIterator[Any]:
```

It is a normal `def`, not `async def`, because calling it returns an async
iterator immediately. The waiting happens when the caller iterates:

```python
async for event in self.subscribe(topic):
    ...
```

This is a common shape for async stream APIs.

## Queue adapter method

```python
def subscribe_queue(self, topic: str) -> asyncio.Queue[Any]:
```

This method adapts the async iterator interface into an `asyncio.Queue`.

Some in-process actors prefer a queue:

```python
event = await queue.get()
```

instead of:

```python
async for event in bus.subscribe(topic):
    ...
```

Providing both styles makes the bus easier to use in different runtime
components.

## Creating a queue

```python
queue: asyncio.Queue[Any] = asyncio.Queue()
```

`asyncio.Queue` is an async-safe queue for tasks in the same event loop.

One task can put events into it:

```python
await queue.put(event)
```

Another task can wait for events:

```python
event = await queue.get()
```

Queues are extremely common in async Python applications.

## Nested async function

```python
async def _forward() -> None:
    async for event in self.subscribe(topic):
        await queue.put(event)
```

This function is defined inside `subscribe_queue`.

Nested functions are common when a small helper is only useful inside one
surrounding function.

Here `_forward` bridges:

```text
bus async iterator -> local queue
```

## Background task

```python
asyncio.create_task(_forward())
```

This schedules `_forward` to run in the background.

The caller immediately receives the queue, while the background task keeps
feeding that queue with subscribed events.

This is a real async pattern:

```text
start background forwarding task, return handle/queue to caller
```

One thing to be aware of: background tasks need lifecycle management. In this
base implementation, `close()` exists so bus implementations can release
resources.

## Default `close`

```python
async def close(self) -> None:
    """Release bus resources. In-memory implementations may no-op."""

    return None
```

`close` is not abstract. It has a default no-op implementation.

Why?

Some buses may need cleanup, such as network connections. Others may not.

The interface still gives runtime code one safe thing to call:

```python
await bus.close()
```

Even if the concrete implementation has nothing to release, the call is valid.

## Design pattern

This file uses the Template/Interface style:

- abstract methods define required behavior
- a concrete helper method is implemented in terms of abstract behavior
- subclasses can inherit the helper or override it

`subscribe_queue` is the concrete helper. It works for any subclass that
correctly implements `subscribe`.

## What to copy into your own code

From this file, copy these habits:

- define small interfaces for important infrastructure
- use abstract methods for behavior every implementation must provide
- keep event payloads generic if the transport should not know their domain
- provide adapter helpers when callers naturally need a different access style
- include cleanup methods in components that might own resources

## Tiny exercise

Answer these:

1. Why does the app depend on `AbstractEventBus` instead of directly depending
   on `InMemoryBus`?
2. Why is `publish` async even for an in-memory implementation?
3. What is the difference between `async for event in bus.subscribe(...)` and
   `await queue.get()`?
4. Why does `close()` have a default implementation?

