# Lesson 05 - `mm/runtime/config.py`, Part 1

Original file: [`mm/runtime/config.py`](../../mm/runtime/config.py)

This is our first larger real-code file.

Configuration files are where many real applications start to feel different
from beginner scripts. A beginner script often has values written directly in
the code:

```python
exchange = "nobitex"
interval_ms = 1000
```

A real application usually reads settings from JSON, environment variables, and
command-line options, then turns them into validated Python objects.

That is what `mm/runtime/config.py` does.

This lesson focuses on the top part of the file: imports, type aliases,
normalization helpers, and the first runtime config dataclasses.

## Why config code matters

Configuration lets the same code run in different modes:

- local development vs production-like runtime
- in-memory bus vs Redis Streams
- observe-only vs live execution
- one exchange vs two exchanges
- Postgres persistence enabled vs disabled

Common rule:

```text
code defines possible behavior; config chooses behavior for this run
```

This is one reason real apps have more structure than small scripts.

## Imports

The file begins:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
```

These imports tell us what kind of file this is.

`json` means the file probably reads JSON config.

`os` means it probably reads environment variables.

`dataclass` and `field` mean it defines structured config objects.

`Path` means it works with filesystem paths.

`Any` means some raw input may be untyped or loosely typed.

`Literal` means some fields only allow specific string values.

This is common in config parsing: raw input starts messy, then the code turns it
into stricter objects.

## Local project imports

```python
from mm.core.models.instrument import (
    InstrumentPrecisionConfig,
    SymbolBinding,
    build_instrument_spec,
    instrument_precision_from_mapping,
    merge_instrument_spec,
)
from mm.core.risk_limits import GlobalVenueRiskLimits, VenueRiskLimits
from mm.persistence.retention import RetentionPolicy, retention_policy_from_mapping
```

These are local project imports. They show that runtime config is not only
simple strings and numbers. It also creates or references:

- instrument precision settings
- symbol bindings
- risk limits
- persistence retention policy

Common pattern:

Config modules often sit near the top of the dependency graph. Many runtime
components need config, so config code imports shared models from `core` and
small helper types from other packages.

## Literal type aliases

```python
SourceRole = Literal["local", "global"]
BusKind = Literal["memory", "redis_streams"]
FillEventSource = Literal["exchange_fill_history", "open_order_delta"]
```

`Literal` means a value should be one of exact allowed values.

For example:

```python
BusKind = Literal["memory", "redis_streams"]
```

means `BusKind` should be either:

```text
memory
redis_streams
```

not:

```text
redis
stream
redis-streams
```

Why this is useful:

- documents allowed values
- helps type checkers catch mistakes
- makes invalid config easier to reject
- gives readers a quick vocabulary list

Common pattern:

Use `Literal[...]` for small sets of string modes.

## Why type aliases exist

The code could write this everywhere:

```python
Literal["local", "global"]
```

But instead it creates:

```python
SourceRole = Literal["local", "global"]
```

Now function signatures and dataclass fields are easier to read:

```python
role: SourceRole
```

This is common: give a repeated type concept a name.

## Normalization helpers

The file defines helpers like:

```python
def _normalize_reference_price_source(value: Any) -> ReferenceVenueRole:
    source = str(value or "global").strip().lower()
    if source not in ("global", "local"):
        raise ValueError("controller.reference_price_source must be 'global' or 'local'")
    return source  # type: ignore[return-value]
```

This is a classic config-parsing function.

It does four things:

1. accepts messy input
2. applies a default
3. normalizes spelling
4. rejects invalid values

Let's unpack this line:

```python
source = str(value or "global").strip().lower()
```

`value or "global"` means:

- use `value` if it is truthy
- otherwise use `"global"`

So if `value` is `None`, empty string, or another falsey value, the default is
`"global"`.

Then:

```python
str(...)
```

turns it into a string.

Then:

```python
.strip().lower()
```

removes outside whitespace and makes it lowercase.

So these all become `"global"`:

```text
None
" Global "
"GLOBAL"
"global"
```

That is normalization.

## Validation with `ValueError`

```python
if source not in ("global", "local"):
    raise ValueError("controller.reference_price_source must be 'global' or 'local'")
```

This is validation.

If the config value is not allowed, the function raises `ValueError`.

Real applications should fail clearly at startup when config is invalid. That is
much better than running for ten minutes and then failing deep inside the
strategy.

Common practice:

Make invalid configuration fail early.

## Why helper names start with `_`

```python
def _normalize_quote_venue(...)
```

A leading underscore means "private to this module" by convention.

Python does not fully prevent other modules from importing it, but the name
tells readers:

```text
this helper is an implementation detail
```

Common convention:

- public function: `load_runtime_config`
- private helper: `_normalize_quote_venue`

## Keyword-only parameters

```python
def _normalize_quote_venue(value: Any, *, field_name: str) -> QuoteVenueRole:
```

The `*` means every parameter after it must be passed by keyword.

So this is allowed:

```python
_normalize_quote_venue(value, field_name="pricing.inventory_mark.venue")
```

This is not allowed:

```python
_normalize_quote_venue(value, "pricing.inventory_mark.venue")
```

Why programmers do this:

- makes calls more readable
- prevents accidentally swapping arguments
- useful when a parameter is a label, flag, or context string

## The first config dataclass

```python
@dataclass(frozen=True, slots=True)
class MarketDataSourceConfig:
    """Public market data source for one strategy role."""

    role: SourceRole
    exchange: str
    symbol: str
    mode: str = "rest"
    depth_limit: int = 10
    trades_limit: int = 100
    poll_interval_ms: int = 1000
    instrument: InstrumentPrecisionConfig | None = None
```

This class represents public market data config for one role.

Required fields:

- `role`
- `exchange`
- `symbol`

Optional fields with defaults:

- `mode`
- `depth_limit`
- `trades_limit`
- `poll_interval_ms`
- `instrument`

Common pattern:

Put required fields first, default fields after. Python dataclasses require this
order.

## Defaults

```python
mode: str = "rest"
depth_limit: int = 10
trades_limit: int = 100
poll_interval_ms: int = 1000
```

Defaults are practical choices. They let callers create a minimal config:

```python
MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT")
```

without specifying every tiny detail.

Common real-app habit:

Good defaults make simple cases easy, while still allowing advanced overrides.

## Methods on config objects

```python
def normalized_exchange(self) -> str:
    return self.exchange.lower().strip()
```

This method keeps normalization logic close to the data it normalizes.

Then other methods reuse it:

```python
def market_event_topic(self, interval_ms: int) -> str:
    return f"market:events:{self.normalized_exchange()}:{self.symbol}:{interval_ms}"
```

This is important: `MarketDataSourceConfig` is not just a bag of fields. It also
knows how to derive related values from those fields.

Common pattern:

Config objects often have small helper methods for derived names, paths, URLs,
or contracts.

## How this connects to lesson 4

In lesson 4, `RuntimeTopics` required this behavior:

```python
class MarketDataSourceLike(Protocol):
    def market_event_topic(self, interval_ms: int) -> str:
        ...
```

`MarketDataSourceConfig` has exactly that method:

```python
def market_event_topic(self, interval_ms: int) -> str:
    ...
```

So it can be used as a `MarketDataSourceLike`.

This is interface-based design. The topic code does not need to know the full
config class. It only needs one method.

## Private runtime config

```python
@dataclass(frozen=True, slots=True)
class PrivateRuntimeConfig:
    """Private state and execution observation config for one exchange."""
```

This config is for private exchange access: account state, open orders, fills,
and maybe execution observation.

It contains direct credential fields:

```python
api_token: str | None = None
api_key: str | None = None
api_private_key: str | None = None
```

and environment-variable-name fields:

```python
api_token_env: str | None = None
api_key_env: str | None = None
api_private_key_env: str | None = None
```

This is a common secret-handling pattern.

The config can either contain the secret directly, or contain the name of the
environment variable where the secret lives.

In real projects, environment variables are preferred for secrets.

## `__post_init__`

```python
def __post_init__(self) -> None:
    if self.fill_event_source not in ("exchange_fill_history", "open_order_delta"):
        raise ValueError(
            "private fill_event_source must be 'exchange_fill_history' or 'open_order_delta'"
        )
```

For dataclasses, `__post_init__` runs after the generated `__init__`.

It is commonly used for:

- validation
- normalization
- derived fields

Here it validates that `fill_event_source` is one of the allowed strings.

## `enabled` property

```python
@property
def enabled(self) -> bool:
    return bool(
        self.resolved_api_token()
        or (
            self.resolved_api_key()
            and self.resolved_api_private_key()
        )
    )
```

This property computes whether private access is enabled.

It returns true if either:

- an API token is available
- both API key and private key are available

The `bool(...)` converts the result into a clean `True` or `False`.

This is a nice example of a computed property: no one has to separately set
`enabled`. The object can infer it from credentials.

## Environment variable resolution

```python
def resolved_api_token(self, exchange: str | None = None) -> str | None:
    if self.api_token:
        return self.api_token
    if self.api_token_env:
        return os.environ.get(self.api_token_env)
    if exchange:
        return os.environ.get(f"{exchange.upper()}_API_TOKEN")
    return None
```

This function implements priority order:

1. use direct token if provided
2. use configured environment variable name if provided
3. use default exchange-based environment variable if exchange is provided
4. otherwise return `None`

This kind of fallback chain is everywhere in config code.

Common pattern:

```text
explicit value -> explicit env var -> conventional env var -> default/None
```

## Lazy imports

Inside `resolved_api_credentials`, the code imports exchange auth classes inside
the function:

```python
if exchange_key == "nobitex" and api_key and api_private_key:
    from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth

    return NobitexApiKeyAuth(api_key, api_private_key)
```

This is called a local import or lazy import.

Why programmers sometimes do this:

- avoid importing exchange-specific code unless needed
- reduce circular import risk
- keep startup lighter
- avoid requiring dependencies for unused paths

It is not always better than top-level imports, but it is common in runtime
selection code.

## Scheduler config

```python
@dataclass(frozen=True, slots=True)
class SchedulerRuntimeConfig:
    """Interval scheduler concurrency and timing behavior."""

    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("scheduler.max_concurrency must be >= 1")
```

This is a tiny, clean config object.

It shows an important habit:

Even simple config deserves validation when invalid values would break the
runtime.

Concurrency cannot be less than 1, so the class rejects it immediately.

## Bus config

```python
@dataclass(frozen=True, slots=True)
class BusRuntimeConfig:
    kind: BusKind = "memory"
    redis_url: str = "redis://localhost:6379/0"
    stream_prefix: str = "mm"
    consumer_group: str = "mm-runtime"
    consumer_name: str = "runtime"
    max_stream_length: int | None = 100_000
    read_block_ms: int = 1_000
    max_queue_size: int = 1024
    backpressure_policy: BusBackpressurePolicy = "block"
```

This config chooses event bus behavior.

Notice the defaults:

- `kind` defaults to `"memory"`
- Redis settings exist, but they only matter when using Redis Streams
- queue and backpressure settings have conservative defaults

This is common in real apps: one config class may contain settings for multiple
modes, and only some settings apply in a given mode.

The validation confirms allowed modes:

```python
if self.kind not in ("memory", "redis_streams"):
    raise ValueError("bus.kind must be 'memory' or 'redis_streams'")
```

## Numeric separators

```python
max_stream_length: int | None = 100_000
read_block_ms: int = 1_000
```

Python allows underscores in numeric literals.

These are the same as:

```python
100000
1000
```

but easier to read.

Common habit:

Use underscores for large numbers or time-like constants.

## Persistence config and `field(default_factory=...)`

```python
@dataclass(frozen=True, slots=True)
class PersistenceRuntimeConfig:
    enabled: bool = False
    database_url: str | None = None
    database_url_env: str = "MM_DATABASE_URL"
    parquet_root: str | None = None
    parquet_root_env: str = "MM_PARQUET_ROOT"
    topics: tuple[str, ...] = ()
    batch_size: int = 100
    flush_interval_ms: int = 1000
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
```

`field(default_factory=RetentionPolicy)` means:

```text
create a fresh RetentionPolicy object for each PersistenceRuntimeConfig
```

This matters for mutable or object defaults.

Bad pattern:

```python
retention: RetentionPolicy = RetentionPolicy()
```

That can accidentally share one object between instances.

Good dataclass pattern:

```python
retention: RetentionPolicy = field(default_factory=RetentionPolicy)
```

Use `default_factory` when the default should be created fresh.

## Tuple type with ellipsis

```python
topics: tuple[str, ...] = ()
```

This means a tuple containing zero or more strings.

The `...` here means "any length", not the Protocol placeholder from lesson 4.

So valid values include:

```python
()
("strategy:actions",)
("strategy:actions", "strategy:decisions")
```

Notice the comma in a one-item tuple:

```python
("strategy:actions",)
```

Without the comma, Python treats it as just a parenthesized string.

## Resolved persistence paths

```python
def resolved_database_url(self) -> str | None:
    if self.database_url:
        return self.database_url
    return os.environ.get(self.database_url_env, "").strip() or None
```

Again we see fallback order:

1. direct config value
2. environment variable
3. `None`

This pattern repeats because it is practical and readable.

## Big lesson from this file section

The top of `mm/runtime/config.py` teaches that real config code should:

- name allowed modes with `Literal`
- normalize messy input
- validate early with clear errors
- represent config as typed dataclasses
- use properties for derived state
- resolve secrets from environment variables
- use fallback chains deliberately
- keep external-service choices configurable
- use `field(default_factory=...)` for fresh object defaults

This is real application code: not fancy, but full of careful small decisions.

## Tiny exercise

Answer these before moving on:

1. Why does config code normalize strings with `.strip().lower()`?
2. Why is `Literal["memory", "redis_streams"]` better than just `str` for
   `BusKind`?
3. Why should invalid config fail at startup?
4. Why does `PrivateRuntimeConfig.enabled` compute from credentials instead of
   being a manually stored boolean?
5. Why is `field(default_factory=RetentionPolicy)` safer than
   `RetentionPolicy()` as a default?

