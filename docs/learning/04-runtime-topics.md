# Lesson 04 - `mm/core/topics.py`

Original file: [`mm/core/topics.py`](../../mm/core/topics.py)

This is our first real code lesson.

The file is small, but it teaches a very common professional pattern:

```text
centralize important string contracts in one module
```

In an event-bus application, topic names are not casual strings. They are
contracts between producers and consumers. If one part publishes to
`strategy:actions` and another part listens to `strategy:action`, the system
silently stops connecting. So programmers put these names in one place.

## The whole file shape

The file has four main parts:

1. imports
2. module-level constants
3. a `Protocol` that describes the required behavior of a market data source
4. a frozen dataclass that builds all runtime topic names

That shape is common:

```text
imports -> constants -> small interfaces/contracts -> main class
```

## Future annotations

The file starts with:

```python
from __future__ import annotations
```

This line changes how Python stores type annotations. In practical terms, it
makes annotations more flexible and can avoid problems when one type refers to
another type that is defined later.

You will see this in many modern Python projects.

Beginner rule:

If you see this line at the top of a modern typed Python file, do not worry
about it. It is usually there to make type hints behave more smoothly.

## Imports

```python
from dataclasses import dataclass
from typing import Protocol
```

These imports come from the Python standard library.

`dataclass` helps you define classes that mostly store data.

`Protocol` helps you define an interface by behavior.

Common practice:

- standard-library imports first
- third-party imports next
- local project imports last

This file only needs standard-library imports, which is a sign that it is a
small core contract module.

## Constants

```python
STRATEGY_ACTIONS_TOPIC = "strategy:actions"
DEFAULT_STRATEGY_DECISIONS_TOPIC = "strategy:decisions"
DEFAULT_STRATEGY_DIAGNOSTICS_TOPIC_PREFIX = "strategy:diagnostics"
```

These are module-level constants.

Python does not enforce constants, but uppercase names are the convention for
"do not casually change this value."

Why constants are better than repeated strings:

```python
publish("strategy:actions", action)
subscribe("strategy:actions", handler)
```

If you type that string in ten places, one typo can create a bug.

Instead:

```python
publish(STRATEGY_ACTIONS_TOPIC, action)
subscribe(STRATEGY_ACTIONS_TOPIC, handler)
```

Now the important string has one home.

Common use cases for constants:

- topic names
- environment variable names
- default config values
- status strings
- metric names
- API path fragments

## Protocol

```python
class MarketDataSourceLike(Protocol):
    def market_event_topic(self, interval_ms: int) -> str:
        ...
```

This is one of the most important patterns in the file.

A `Protocol` says: "I do not care what class you are. I care that you provide
this behavior."

Here, `RuntimeTopics` needs a `local` object that can build a market event topic.
It does not require a specific class. It only requires a method:

```python
market_event_topic(interval_ms: int) -> str
```

This is called structural typing.

Beginner version:

If it has the right method, it fits.

Real code example:

[`mm/runtime/config.py`](../../mm/runtime/config.py) defines
`MarketDataSourceConfig`, and that class has:

```python
def market_event_topic(self, interval_ms: int) -> str:
    return f"market:events:{self.normalized_exchange()}:{self.symbol}:{interval_ms}"
```

So `MarketDataSourceConfig` satisfies `MarketDataSourceLike` even though it does
not inherit from it.

Why programmers like this:

- less coupling between modules
- easier tests with fake objects
- clearer statement of what behavior is needed
- no need for a heavy inheritance hierarchy

## The ellipsis body

```python
...
```

The three dots are the Python ellipsis literal.

In Protocols, it commonly means: "this method is only described here; the real
implementation is somewhere else."

You may also see `pass` used in simple placeholder bodies, but `...` is very
common in type stubs, Protocols, and abstract-looking definitions.

## Dataclass

```python
@dataclass(frozen=True, slots=True)
class RuntimeTopics:
    """Central topic contract for one market-maker runtime."""
```

`@dataclass(...)` is a decorator. A decorator modifies the class or function
defined immediately below it.

For dataclasses, Python automatically creates useful methods such as `__init__`.

Without a dataclass, you might write:

```python
class RuntimeTopics:
    def __init__(self, local, interval_ms, global_=None, frame_topic=None):
        self.local = local
        self.interval_ms = interval_ms
        self.global_ = global_
        self.frame_topic = frame_topic
```

With a dataclass, the fields become the constructor:

```python
topics = RuntimeTopics(local=source, interval_ms=1000)
```

Common use:

Use dataclasses for simple data containers with some helper methods.

## `frozen=True`

`frozen=True` means the object is intended to be immutable after creation.

So this should not be normal code:

```python
topics.interval_ms = 500
```

Why immutability is useful:

- config/contract objects are safer when they do not change halfway through a run
- fewer surprising side effects
- easier reasoning in async systems

In real applications, immutable configuration and contract objects are very
common.

## `slots=True`

`slots=True` tells Python not to give each instance a normal dynamic attribute
dictionary.

Practical effects:

- can use less memory
- can be a bit faster
- prevents accidental new attributes

For example, without slots, this kind of typo might create a new attribute:

```python
topics.intervla_ms = 1000
```

With slots, Python rejects attributes not declared in the dataclass fields.

This is useful in long-running applications where quiet mistakes are expensive.

## Type-annotated fields

```python
local: MarketDataSourceLike
interval_ms: int
global_: MarketDataSourceLike | None = None
frame_topic: str | None = None
action_topic: str = STRATEGY_ACTIONS_TOPIC
```

Each field has:

```text
name: type = default
```

Fields without defaults must be passed when creating the object:

```python
RuntimeTopics(local=source, interval_ms=1000)
```

Fields with defaults can be omitted.

`MarketDataSourceLike | None` means either a `MarketDataSourceLike` object or
`None`.

This `|` union syntax is modern Python. Older code often used:

```python
Optional[MarketDataSourceLike]
```

## Why `global_` has an underscore

```python
global_: MarketDataSourceLike | None = None
```

`global` is a Python keyword, so it cannot be used as a variable or attribute
name.

Programmers often add a trailing underscore when the natural name conflicts with
a keyword:

```python
class_
from_
global_
```

This is a normal Python convention.

## Docstring

```python
"""Central topic contract for one market-maker runtime."""
```

This is a class docstring. It explains the class's purpose.

Good docstrings answer: "why does this thing exist?"

This one tells us that `RuntimeTopics` is not just a string helper. It is the
central contract for topic names in one runtime.

## Properties

```python
@property
def strategy_actions(self) -> str:
    return self.action_topic
```

`@property` lets a method be accessed like an attribute:

```python
topics.strategy_actions
```

instead of:

```python
topics.strategy_actions()
```

Use a property when the caller should think of the result as a value, not an
operation.

In this file, topic names feel like values, so properties make the calling code
nice to read.

## Computed property with fallback

```python
@property
def strategy_frame(self) -> str:
    if self.frame_topic:
        return self.frame_topic
    return self.local.market_event_topic(self.interval_ms).replace("market:events", "strategy:frames", 1)
```

This property has a common pattern:

```text
if explicit override exists, use it
otherwise compute a default
```

That pattern appears constantly in configuration code.

Here:

- if `frame_topic` was provided, use it
- otherwise derive a frame topic from the local market event topic

Example:

```text
market:events:nobitex:BTC_IRT:1000
```

becomes:

```text
strategy:frames:nobitex:BTC_IRT:1000
```

The string method:

```python
.replace("market:events", "strategy:frames", 1)
```

means replace only the first occurrence.

The `1` matters. It prevents accidental replacement later in the string if the
same text appears again.

## Conditional expression

```python
return self.global_.market_event_topic(self.interval_ms) if self.global_ is not None else None
```

This is Python's inline conditional expression.

General form:

```python
value_if_true if condition else value_if_false
```

Here it means:

- if there is a global data source, return its market event topic
- otherwise return `None`

This is common for short, simple conditionals.

For longer logic, programmers usually prefer a normal `if` block.

## Static methods

```python
@staticmethod
def execution_orders(exchange: str) -> str:
    return f"execution:orders:{exchange.lower().strip()}"
```

A static method is grouped inside the class but does not use `self`.

Why put it in the class at all?

Because it belongs conceptually to runtime topics.

Calling code can say:

```python
RuntimeTopics.execution_orders("Wallex")
```

and get:

```text
execution:orders:wallex
```

This keeps related topic builders grouped together.

## F-strings

```python
f"private:state:{exchange.lower().strip()}:snapshot"
```

An f-string lets you put Python expressions inside `{...}`.

This expression:

```python
exchange.lower().strip()
```

normalizes the exchange name.

`.lower()` makes it lowercase.

`.strip()` removes leading and trailing whitespace.

So:

```python
RuntimeTopics.private_snapshot(" Wallex ")
```

returns:

```text
private:state:wallex:snapshot
```

Normalization is a common defensive habit when building identifiers.

## Why these methods are small

Every method in this file is tiny.

That is intentional. This module has one job:

```text
make topic names consistently
```

It does not publish events. It does not subscribe to events. It does not talk to
Redis. It just defines names.

This is good design. Small modules with clear jobs are easier to test, reuse,
and trust.

## Architectural pattern

This file is part of the `core` package:

```text
mm/core/topics.py
```

`core` usually means shared domain contracts and value objects.

Many other parts of the app can import from `core`, but `core` should be careful
about importing from high-level application modules. That helps avoid circular
dependencies.

Common dependency direction:

```text
runtime/application/execution -> core
core should not depend heavily on runtime/application/execution
```

This is one way real projects keep code organized.

## What to copy into your own programming

From this file, copy these habits:

- Give important strings one central home.
- Use uppercase names for constants.
- Use dataclasses for small structured objects.
- Use `Protocol` when you only need behavior, not a specific class.
- Use properties for computed values that read like attributes.
- Normalize strings before putting them into identifiers.
- Keep core contract modules small and boring.

Small and boring is a compliment in infrastructure code.

## Tiny exercise

Answer these before moving on:

1. Why is `MarketDataSourceLike` a `Protocol` instead of a concrete parent class?
2. What is the advantage of `RuntimeTopics.execution_orders(" Wallex ")`
   returning lowercase/stripped output?
3. Why does `strategy_frame` allow `frame_topic` to override the computed
   default?
4. What kind of bug could happen if topic strings were typed manually all over
   the codebase?

