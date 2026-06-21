# Lesson 06 - `mm/runtime/config.py`, Part 2

Original file: [`mm/runtime/config.py`](../../mm/runtime/config.py)

Lesson 5 covered the top of `config.py`: typed config classes, defaults,
validation, environment variable lookup, and small normalization helpers.

This lesson covers the next real-app pattern:

```text
raw JSON dictionary -> validated typed runtime config object
```

This is where configuration stops being loose input and becomes application
structure.

## The public loading function

```python
def load_runtime_config(path: str | Path) -> MarketMakerRuntimeConfig:
    """Load a full runtime config from JSON.

    JSON is used intentionally here because it has no additional runtime
    dependency and keeps deployment/configuration separate from CLI defaults.
    """

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("runtime config file must contain a JSON object")
    return runtime_config_from_mapping(payload)
```

This function is a public entry point. It accepts a file path, reads JSON, checks
that the top-level JSON value is an object, then delegates to
`runtime_config_from_mapping`.

Common pattern:

```text
I/O wrapper -> pure-ish parser/builder
```

`load_runtime_config` does file I/O. `runtime_config_from_mapping` handles the
dictionary. This separation is useful because tests can call
`runtime_config_from_mapping` directly without creating files.

## `str | Path`

```python
path: str | Path
```

The caller may pass either a string path:

```python
"mm/configs/market_maker.runtime.example.json"
```

or a `Path` object:

```python
Path("mm/configs/market_maker.runtime.example.json")
```

This is a convenience pattern. Many Python projects accept both.

## Turning paths into `Path`

```python
Path(path).read_text()
```

Even if `path` is already a string, `Path(path)` wraps it in a `Path` object.
Then `read_text()` reads the file into a string.

In production code you will often see `Path` preferred over older `open(...)`
style for small file reads because it is compact and expressive.

## Runtime config from a mapping

```python
def runtime_config_from_mapping(payload: dict[str, Any]) -> MarketMakerRuntimeConfig:
    bot, preset, local_payload, global_payload = _resolve_bot_runtime(payload)
    ...
    return MarketMakerRuntimeConfig(...)
```

This function is the main builder.

It takes one raw dictionary and returns one `MarketMakerRuntimeConfig`.

Notice the high-level sequence:

1. resolve the bot and preset
2. resolve local and global exchange payloads
3. derive liquidity sourcing and pricing
4. build controller payload
5. construct `MarketMakerRuntimeConfig`

This is orchestration code. It does not perform every tiny conversion inline.
Instead, it calls smaller helpers with focused names.

Common real-app habit:

Keep the main builder readable by extracting named helpers.

## Multiple assignment

```python
bot, preset, local_payload, global_payload = _resolve_bot_runtime(payload)
```

This is tuple unpacking.

`_resolve_bot_runtime` returns four things, and Python assigns them to four
variables.

This is common when a helper naturally resolves a small bundle of related
values.

Do not overuse it for unrelated values. Here it is reasonable because those four
values belong to the same config-resolution step.

## Raw payload vs typed object

In this file, names ending with `_payload` usually mean raw or semi-raw
dictionaries:

```python
local_private_payload
global_private_payload
controller_payload
```

Names like `bot`, `pricing`, and `liquidity_sourcing` usually mean typed config
objects:

```python
bot = BotRuntimeConfig(...)
pricing = PricingRuntimeConfig(...)
liquidity_sourcing = LiquiditySourcingRuntimeConfig(...)
```

This naming distinction is useful.

Common pattern:

```text
payload = raw dictionary
config = typed object
```

## Resolving local and global roles

```python
local_private_payload = _exchange_private_payload(payload, bot.normalized_quoting_exchange)
global_private_payload = (
    _exchange_private_payload(payload, str(global_payload["exchange"]))
    if isinstance(global_payload, dict)
    else None
)
```

This code says:

- local private config comes from the quoting exchange
- global private config exists only if there is a global payload

The inline conditional:

```python
value_if_true if condition else value_if_false
```

is used because this is a short branch.

## Defensive type checks

```python
if isinstance(global_payload, dict)
```

The raw config can be missing or malformed, so the code checks types before
accessing fields.

This is common when working with data from JSON because JSON parsing gives you
general Python values:

- `dict`
- `list`
- `str`
- `int`
- `float`
- `bool`
- `None`

The code must decide what shapes are valid.

## Building the final dataclass

```python
return MarketMakerRuntimeConfig(
    local=_source_from_mapping(local_payload, role="local"),
    global_=_source_from_mapping(global_payload, role="global") if isinstance(global_payload, dict) else None,
    interval_ms=int(preset.get("interval_ms", payload.get("interval_ms", 5000))),
    ...
)
```

This is the moment raw data becomes the main typed runtime config.

Notice several patterns:

- helper functions convert nested sections
- missing global config becomes `None`
- numeric values are converted with `int(...)`
- preset values override top-level defaults
- fallback defaults are supplied

This line:

```python
preset.get("interval_ms", payload.get("interval_ms", 5000))
```

means:

1. use `preset["interval_ms"]` if present
2. otherwise use top-level `payload["interval_ms"]` if present
3. otherwise use `5000`

That is a priority chain.

## One runtime, one bot

```python
def _single_bot_payload(payload: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, dict):
        if not payload:
            raise ValueError("runtime config requires at least one bot")
        if len(payload) != 1:
            raise ValueError("this runtime starts one bot; use one bots entry per config file")
        bot_id, bot_payload = next(iter(payload.items()))
        return str(bot_id), bot_payload
    raise ValueError("runtime config requires bots object")
```

This helper encodes a product decision:

```text
one runtime config file starts one bot
```

It rejects zero bots and more than one bot.

That is not just syntax. It is architecture expressed as validation.

Common lesson:

Validation often reveals the system's intended operating model.

## `next(iter(...))`

```python
bot_id, bot_payload = next(iter(payload.items()))
```

`payload.items()` returns key/value pairs.

`iter(...)` creates an iterator.

`next(...)` gets the first item.

Because the function already checked `len(payload) == 1`, the "first" item is
the only item.

This is a compact Python idiom for extracting the only key/value pair from a
dictionary.

## Required fields

```python
preset_name = str(bot_payload.get("preset", "")).strip()
if not preset_name:
    raise ValueError(f"bots.{bot_id}.preset is required")
```

The pattern is:

1. read optional raw value
2. convert to string
3. strip whitespace
4. reject empty result

This repeats for fields like `symbol`, `quoting_exchange`, and
`liquidity_exchange`.

Good config errors include the path:

```text
bots.my_bot.preset is required
```

That helps the user fix the JSON quickly.

## Copying dictionaries

```python
return bot, dict(preset), local_payload, global_payload
```

`dict(preset)` creates a shallow copy.

Why copy?

So later code can mutate or merge values without accidentally changing the
original dictionary from the raw payload.

This is a common defensive practice with dictionaries.

## Exchange config extraction

```python
def _exchange_config(payload: dict[str, Any], exchange: str) -> dict[str, Any]:
    exchanges = payload.get("exchanges")
    if not isinstance(exchanges, dict):
        raise ValueError("runtime config requires exchanges object")
    config = exchanges.get(exchange)
    if not isinstance(config, dict):
        raise ValueError(f"exchanges.{exchange} is required")
    return dict(config)
```

This helper centralizes a repeated lookup:

```text
payload["exchanges"][exchange]
```

It also validates the shape and returns a copy.

Common pattern:

If a nested lookup appears more than once, wrap it in a helper with validation.

## Merging exchange market data

```python
out = dict(market_data)
out["exchange"] = exchange
out["symbol"] = symbol
```

This creates a new dictionary based on exchange market-data settings, then adds
standard fields needed by `MarketDataSourceConfig`.

This is a transformation step:

```text
exchange config section + bot symbol -> source payload
```

Real config code often combines data from multiple sections. Here, the bot owns
the symbol and role assignment; the exchange section owns exchange-specific
market-data options.

## Unknown key validation

In `_pricing_from_preset`, the code does:

```python
allowed = {"quote_anchor", "inventory_mark"}
unknown = sorted(key for key in payload if not str(key).startswith("_") and key not in allowed)
if unknown:
    raise ValueError(f"unknown strategy preset pricing keys: {', '.join(unknown)}")
```

This rejects unexpected config keys.

Why this is useful:

- catches typos
- prevents users from thinking a setting is active when it is ignored
- keeps config schemas tight

The `not str(key).startswith("_")` part allows private/comment-like metadata
keys. That is a practical compromise.

## Removed config keys

```python
if "reference_price_source" in controller:
    raise ValueError(
        "controller.reference_price_source was removed; use pricing.inventory_mark.venue"
    )
```

This is compatibility guidance.

Instead of silently ignoring an old key, the parser raises a useful error and
tells the user what to use now.

Common real-app habit:

When you remove or rename config, fail with a migration message.

## Source conversion

```python
def _source_from_mapping(payload: dict[str, Any], *, role: SourceRole) -> MarketDataSourceConfig:
    return MarketDataSourceConfig(
        role=role,
        exchange=str(payload["exchange"]),
        symbol=str(payload["symbol"]),
        market_data_source=_normalize_market_data_source(payload.get("market_data_source", "local")),
        mode=str(payload.get("mode", "rest")),
        depth_limit=int(payload.get("depth_limit", 10)),
        trades_limit=int(payload.get("trades_limit", 100)),
        poll_interval_ms=int(payload.get("poll_interval_ms", 1000)),
        instrument=instrument_precision_from_mapping(payload.get("instrument")),
    )
```

This is a direct mapping function.

It converts raw dictionary values into a typed dataclass.

Notice:

- required values use `payload["exchange"]`
- optional values use `payload.get(...)`
- raw strings and numbers are converted
- nested instrument config is delegated to another parser

This is one of the most common shapes in config-heavy Python applications.

## Big lesson

This part of `config.py` teaches several real application habits:

- separate file loading from dictionary parsing
- validate raw JSON shapes early
- use small helpers for nested sections
- express product rules as validation
- copy dictionaries before merging or modifying them
- reject unknown or removed config keys
- use clear config-path error messages
- convert raw payload dictionaries into typed dataclasses at boundaries

## Tiny exercise

Answer these:

1. Why does `load_runtime_config` delegate to `runtime_config_from_mapping`?
2. Why does `_single_bot_payload` reject more than one bot?
3. What is the advantage of error messages like `bots.my_bot.symbol is required`?
4. Why is unknown-key validation helpful in configuration files?
5. What is the difference between `payload["exchange"]` and
   `payload.get("mode", "rest")` in `_source_from_mapping`?

