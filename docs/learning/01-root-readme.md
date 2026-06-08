# Lesson 01 - Root `README.md`

Original file: [`README.md`](../../README.md)

This first lesson starts with the root `README.md`, not a Python file. That may
feel surprising, but it is a very common real-world pattern: before you read
implementation code, you learn what the project thinks it is.

In a serious application, the root README is usually the front door. It answers:

- What is this project?
- Where is the deeper documentation?
- How do I run it?
- What should I be careful about?

This repository's root README is short, which is also common. Large projects
often avoid putting every detail in the root README because that file becomes
hard to maintain. Instead, it points to focused documents under `docs/`.

## The full file

````md
# Market maker

Event-bus market-making runtime: multi-venue ingestion, interval market events, MPC strategy frames, optional live limit-order execution, and Postgres/Grafana observability.

## Documentation

Full architecture and operations guides live in **[docs/](docs/README.md)**:

- [Architecture overview](docs/architecture.md)
- [Data flow](docs/data-flow.md)
- [Runtime mechanisms](docs/mechanisms.md) (detailed event flows & internals)
- [Controller & execution](docs/controller-and-execution.md)
- [Observability & operations](docs/observability.md)
- [Configuration](docs/configuration.md)

## Quick start

```bash
cp .env.example .env          # add API tokens, MM_DATABASE_URL, etc.
docker compose up -d          # optional: Postgres, Redis, Grafana stack
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

Use `controller.observe_only: true` in JSON until you intend to place live orders.
````

## Line 1: the project name

```md
# Market maker
```

The `#` is Markdown syntax for a level-1 heading. In most project READMEs,
there should be exactly one level-1 heading. It names the project.

Common programmer habit:

- Use `#` for the page title.
- Use `##` for major sections.
- Use `###` for smaller subsections.

This is not Python syntax; it is Markdown. Markdown is a lightweight text format
used in README files, GitHub docs, issue templates, many static sites, and
technical notes.

Why this matters:

Good programmers do not only write code. They write context. A project with no
front-door explanation is hard for teammates, future maintainers, and even the
original author after a few months.

## Line 3: the one-sentence architecture summary

```md
Event-bus market-making runtime: multi-venue ingestion, interval market events, MPC strategy frames, optional live limit-order execution, and Postgres/Grafana observability.
```

This single sentence carries a lot. It is doing the job of a miniature system
overview.

Let's unpack it.

### "Event-bus"

An event bus is a communication pattern. Instead of one object directly calling
every other object, parts of the system publish events to named topics, and
other parts subscribe to those topics.

In simple beginner code, you might write:

```python
market_data = get_market_data()
decision = strategy.decide(market_data)
order_manager.place(decision)
```

That is direct, linear code. It is often fine for small scripts.

In a larger real application, you may instead have:

```text
market data producer -> event bus -> strategy frame assembler
strategy runner      -> event bus -> persistence and observability
order manager        -> event bus -> execution state dashboards
```

Why programmers do this:

- Multiple consumers can react to the same event.
- Producers do not need to know every consumer.
- Logging, persistence, dashboards, and control commands can be added without
  rewriting the core trading logic.
- The system becomes easier to replay and debug.

Tradeoff:

Event-bus systems can be harder to follow at first because the flow is not just
one function calling the next function. You have to learn the topics and event
types.

In this project, the event bus lives under `mm/bus/`, and topic names are
defined around `mm/core/topics.py`.

### "Market-making runtime"

Runtime means the live program that runs continuously. This is different from a
library.

A library gives functions or classes to other code.

A runtime starts, reads configuration, connects to services, starts loops, and
keeps working until stopped.

This project is a runtime because it:

- connects to exchanges
- listens to market data
- builds strategy inputs
- runs a decision engine
- may place or cancel orders
- writes telemetry and persistence data

Common pattern:

Real applications often have both:

- reusable modules with business logic
- a runtime entry point that wires those modules together

For Python projects, runtime entry points often appear as:

```bash
python -m package_name
python -m package_name.submodule
```

This README later uses:

```bash
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

That tells Python to run a module as a program.

### "Multi-venue ingestion"

Venue means exchange or trading venue. This project can ingest data from
multiple exchanges such as Nobitex, Wallex, Bitpin, and Tabdeal.

In real systems, programmers avoid hard-coding everything for one provider.
They create adapters.

An adapter translates one external system's format into the application's
internal format.

For example:

```text
Nobitex JSON  -> Nobitex adapter -> internal OrderBook model
Wallex JSON   -> Wallex adapter  -> internal OrderBook model
Bitpin JSON   -> Bitpin adapter  -> internal OrderBook model
```

Why this is common:

- Each exchange has different API fields.
- The strategy should not care about every exchange's wire format.
- Tests can verify that all adapters produce the same internal model shape.

In this repository, exchange-specific code lives under `mm/exchanges/`.

### "Interval market events"

Trading data arrives continuously, but many strategies make decisions on a
schedule: for example every 500 ms, 1 second, or 5 seconds.

An interval market event is a snapshot or summary for a time window.

Common pattern:

```text
raw websocket updates -> local state/buffers -> interval event -> strategy
```

Why programmers do this:

- It gives the strategy stable inputs.
- It reduces noise from every tiny raw update.
- It creates a consistent timeline for replay, metrics, and tests.

This is an example of separating raw input handling from decision-making.

### "MPC strategy frames"

MPC means Model Predictive Control. In simple terms, the strategy simulates or
optimizes over possible future actions, then chooses what to do now.

A strategy frame is the input package given to the strategy for one decision.
It likely contains things like:

- primary market event
- hedge market event
- order book state
- recent trades
- private account or order state
- timing metadata

Common application pattern:

Do not pass ten unrelated arguments everywhere. Put related decision input into
a named object, often a dataclass or structured model.

Instead of:

```python
decide(orderbook, trades, wallet, hedge_book, timestamp, config, limits)
```

real projects often prefer something closer to:

```python
decide(frame)
```

The frame object then becomes a contract.

Why this helps:

- Easier to test.
- Easier to log.
- Easier to version.
- Easier to pass through queues or event buses.

### "Optional live limit-order execution"

This phrase is important. The system can place live orders, but live execution
is optional.

A limit order says: buy or sell at this price or better. Market makers commonly
place limit orders on both sides of the book.

Why "optional" matters:

Trading software can lose real money if it behaves incorrectly. A serious
system usually has a mode where it observes, computes, logs, and validates
without actually placing live orders.

This README later warns:

```md
Use `controller.observe_only: true` in JSON until you intend to place live orders.
```

This is a safety pattern. Many production systems have similar switches:

- dry run
- observe only
- paper trading
- simulation mode
- read-only mode
- kill switch

### "Postgres/Grafana observability"

Postgres is a database. Grafana is a dashboarding/visualization tool.

Observability means the system is designed to be inspected while running. It is
more than printing a few logs.

Common observability pieces:

- logs: what happened
- metrics: numeric measurements over time
- traces: timing and relationships between operations
- persisted events: durable history for replay and debugging
- dashboards: visual views for humans

Real applications care about observability because bugs often happen in
production-like conditions, not only inside simple unit tests.

## Lines 5-14: documentation section

```md
## Documentation

Full architecture and operations guides live in **[docs/](docs/README.md)**:

- [Architecture overview](docs/architecture.md)
- [Data flow](docs/data-flow.md)
- [Runtime mechanisms](docs/mechanisms.md) (detailed event flows & internals)
- [Controller & execution](docs/controller-and-execution.md)
- [Observability & operations](docs/observability.md)
- [Configuration](docs/configuration.md)
```

The `##` creates a level-2 heading.

The `**...**` syntax makes text bold.

The `[label](target)` syntax creates a link.

The `-` lines create a bullet list.

This section shows a common documentation pattern:

```text
root README -> docs index -> focused guide documents
```

Why programmers do this:

- The root README stays readable.
- Each document has one job.
- People can choose the guide they need.
- It avoids a giant document that is impossible to navigate.

Notice the guide names:

- Architecture overview
- Data flow
- Runtime mechanisms
- Controller and execution
- Observability and operations
- Configuration

Those are common categories in real applications.

### Architecture overview

Architecture explains the large pieces and how they relate.

Typical architecture questions:

- What are the main modules?
- Where does data enter?
- Where does data leave?
- Which module owns which responsibility?
- What are the main design principles?

As you grow as a programmer, you learn that code is not just lines. Code is
organized responsibility.

### Data flow

Data flow explains the path information takes through the system.

For this project:

```text
exchange feeds -> ingestion -> market events -> strategy frames -> decisions -> risk -> execution
```

This is one of the most valuable things to understand in any real app.

For example, in a web app the data flow might be:

```text
browser -> HTTP route -> service -> database -> response -> browser
```

In this market maker, the shape is more event-driven and continuous.

### Runtime mechanisms

Mechanisms are the detailed moving parts.

Architecture says "there is an event bus."

Mechanisms say:

- which tasks run
- which queues are drained
- when events are published
- what happens if hedge data is late
- where own orders are subtracted

This split is common:

- architecture: big map
- mechanisms: how the machine actually moves

### Controller and execution

This is probably about the decision pipeline and the order-management pipeline.

Common real-world separation:

- strategy decides what it wants
- risk checks whether it is allowed
- execution turns approved actions into exchange orders

This separation prevents one giant function from doing everything.

### Observability and operations

Operations means running the application in practice.

A program is not only "correct code"; it also needs:

- configuration
- logs
- dashboards
- database setup
- restart behavior
- debugging tools

Real application programming includes these concerns.

### Configuration

Configuration is how behavior changes without editing code.

For example:

- which exchange to use
- which symbol to trade
- whether live execution is enabled
- which database URL to use
- which event bus implementation to use

Common pattern:

Code defines possible behavior. Configuration chooses behavior for a run.

## Lines 16-22: quick start commands

````md
## Quick start

```bash
cp .env.example .env          # add API tokens, MM_DATABASE_URL, etc.
docker compose up -d          # optional: Postgres, Redis, Grafana stack
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```
````

The triple backticks start and end a fenced code block.

The word `bash` after the opening backticks tells Markdown renderers that this
is shell code. Many renderers use that for syntax highlighting.

This is a common README pattern: put runnable commands in code fences so users
can copy and run them.

### Command 1

```bash
cp .env.example .env
```

`cp` copies a file.

`.env.example` is usually a template file containing environment variable names
without real secrets.

`.env` is usually the local private configuration file.

Why this pattern is common:

- The project can show which variables are needed.
- Real secrets do not need to be committed to Git.
- Each developer or server can have different settings.

Typical `.env` values:

```text
API_KEY=...
DATABASE_URL=...
LOG_LEVEL=...
```

Important practice:

Real `.env` files are commonly ignored by Git. Example files such as
`.env.example` are committed.

### Inline shell comments

```bash
cp .env.example .env          # add API tokens, MM_DATABASE_URL, etc.
```

The `#` begins a comment in shell syntax. The shell ignores the comment.

In README commands, comments explain what a command is for.

### Command 2

```bash
docker compose up -d
```

This starts supporting services from `docker-compose.yml`.

In this project, those services include things like:

- Redis
- Postgres
- Prometheus
- Loki
- Tempo
- Grafana

Why real apps use Docker Compose:

- The Python app may need databases or infrastructure.
- Developers can start a local stack consistently.
- The app environment becomes easier to reproduce.

The `-d` option means detached mode. The containers run in the background.

### Command 3

```bash
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

This is the command that starts the Python runtime.

Let's break it down:

```text
python
```

Run the Python interpreter.

```text
-m mm.runtime
```

Run the module named `mm.runtime` as a program.

This is a common Python application pattern. Python looks for runnable module
entry behavior. Usually that means a `__main__.py` file inside the package or a
module-level entry point.

```text
--config mm/configs/market_maker.runtime.example.json
```

Pass a command-line option named `--config`.

The value is the path to a JSON configuration file.

Common pattern:

```bash
python -m some_app --config path/to/config.json
```

The command line chooses the config, and the code reads the config during
startup.

## Line 24: safety warning

```md
Use `controller.observe_only: true` in JSON until you intend to place live orders.
```

The backticks around `controller.observe_only: true` create inline code
formatting.

This is used for:

- commands
- file paths
- config keys
- function names
- class names
- exact values

The meaning is: this is a literal thing, not just normal prose.

The warning itself is an example of responsible engineering. In trading systems,
the default should protect the user from accidental live execution.

Common pattern:

```text
dangerous behavior requires explicit configuration
```

Here, that behavior is live order placement.

## Big-picture lessons from this small file

This README demonstrates several real-application practices:

- Start with a short system summary.
- Put deeper docs in a `docs/` folder.
- Separate architecture, data flow, mechanisms, operations, and configuration.
- Provide runnable setup commands.
- Use environment variables for local secrets and machine-specific settings.
- Use Docker Compose for local infrastructure.
- Run Python applications with `python -m package`.
- Keep dangerous live behavior behind an explicit safety switch.

## Syntax patterns to remember

Markdown:

````md
# Page title
## Section title
**bold text**
[link label](path/to/file.md)
- bullet item
`inline code`
```bash
command here
```
````

Shell:

```bash
cp source target
docker compose up -d
python -m package.module --option value
# comment
```

Configuration style:

```text
section.key: value
```

In this README, `controller.observe_only: true` means the `observe_only` setting
inside the `controller` part of the JSON config.

## How a senior programmer reads this README

A senior programmer would probably think:

1. This is an event-driven trading runtime.
2. It has external dependencies, so I should check `docker-compose.yml`.
3. It has safety-sensitive behavior, so I should inspect config defaults before
   running live execution.
4. It has a docs folder, so I should read the data-flow and mechanisms docs
   before editing strategy or execution code.
5. The real entry point is probably under `mm/runtime/`.
6. The exchange-specific code is probably separated into adapters.
7. Tests likely exist for data normalization, risk, execution, and strategy
   behavior.

That is the habit you want to build: do not only read words. Use the README to
form a map of where the real code probably lives.

## Tiny exercise

Before moving to the next lesson, answer these in your own words:

1. What is the difference between raw market data and an interval market event?
2. Why might a project use an event bus instead of direct function calls?
3. Why is `observe_only` a good default for a trading runtime?
4. What does `python -m mm.runtime --config ...` tell you about the project
   structure?
