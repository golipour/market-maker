# Lesson 02 - `docs/README.md`

Original file: [`docs/README.md`](../README.md)

The root README pointed us to `docs/`. Now we read the docs index. In real
projects, this kind of file is a map. It does not explain every detail. It tells
you which detailed explanation to open next.

That is a common engineering habit: create one place where readers can orient
themselves, then split details into focused documents.

## The role of this file

The first paragraph says:

```md
This folder describes the **event-bus runtime** in `mm/`: how market data becomes strategy frames, how the MPC controller decides quotes, how orders reach the exchange, and how telemetry and persistence fit in.
```

Notice what it does:

- names the main subject: the event-bus runtime
- names the implementation folder: `mm/`
- summarizes the life of data: market data to frames to decisions to orders
- includes side concerns: telemetry and persistence

This is how experienced programmers compress a system into one sentence. They
do not list every class. They name the flow and the responsibilities.

## Markdown syntax: bold and inline code

```md
**event-bus runtime**
`mm/`
```

`**...**` means bold. It tells the reader "this phrase matters."

Backticks mean inline code or literal text. Use them for exact names:

- folders: `mm/`
- files: `docker-compose.yml`
- classes: `StrategyRunner`
- config keys: `controller.dry_run`
- commands: `python -m mm.runtime`

Common practice:

Use backticks when the exact spelling matters. Do not use them for ordinary
ideas.

## The guide table

The file then uses a Markdown table:

```md
| Document | Contents |
|----------|----------|
| [Architecture overview](architecture.md) | Layers, packages, event bus, envelopes, design principles |
```

Table syntax has three parts:

- header row
- separator row
- data rows

The separator row:

```md
|----------|----------|
```

tells Markdown that the previous row was the header.

Why use a table here?

- It is easy to scan.
- It pairs each document with its purpose.
- It prevents the docs index from becoming a long paragraph.

Common real-world pattern:

```text
Document index = link + one-sentence purpose
```

That one-sentence purpose is valuable. A link alone is often not enough.

## What the guide names teach us

The table lists documents such as:

- Architecture overview
- Data flow
- Runtime mechanisms
- Controller and execution
- Observability and operations
- Configuration
- Price and orderflow models
- Joint local/global MPC
- Batched tensor MPC optimizer
- Hedging in MPC

This teaches a big application lesson: real systems are usually understood from
several angles, not one.

For example:

- architecture explains structure
- data flow explains movement
- mechanisms explain details
- configuration explains runtime choices
- operations explains how to run and debug it
- model docs explain domain logic

If you try to understand a serious codebase from only one angle, it feels
messy. If you learn which angle you are using, the code becomes more readable.

## Why `Runtime mechanisms` is emphasized

This row is bold:

```md
| **[Runtime mechanisms](mechanisms.md)** | **Detailed internals: tasks, queues, hedge drain/grace, raw vs cleaned books, execution** |
```

The docs author is guiding your attention. They are saying: "when you need to
know why the code behaves this way, this document matters most."

This is common in mature projects. Not every doc has equal importance. Good docs
help the reader choose.

The phrase "tasks, queues, hedge drain/grace" hints that this is an async
system. Async programs often have multiple tasks running concurrently, and data
arrives through queues.

## Recommended reading order

The file says:

```md
Start with [Data flow](data-flow.md) for the big picture, then read **[Runtime mechanisms](mechanisms.md)** when you need to know *why* the code does something...
```

The `*why*` syntax italicizes text.

This sentence is teaching a reading strategy:

1. First learn the big path.
2. Then learn detailed behavior.
3. Then inspect implementation code.

This is how senior engineers often approach unfamiliar systems. They resist
opening random files immediately because that creates many disconnected facts.

## Quick start repeated

The same quick start appears again:

```bash
cp .env.example .env
docker compose up -d
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
```

Repeating setup commands in a docs index is normal. People may land directly in
`docs/README.md` from a link, so this page should still help them run the
system.

Common docs principle:

Important operational information can be repeated in more than one place if it
reduces reader friction.

## Repository layout tree

The last section shows:

```text
market_maker/
+-- mm/
|   +-- runtime/           # build_runtime_components, CLI entry
|   +-- application/       # Frame assembler, StrategyRunner, frame prep
|   +-- strategies/        # MpcStrategyEngine
|   +-- ingestion/         # SymbolEngine
|   +-- bus/               # memory | redis_streams
|   +-- execution/         # OrderManager
|   +-- risk/              # RuntimeRiskEngine
|   +-- configs/           # Example runtime JSON
+-- grafana/
+-- docker-compose.yml
+-- docs/
```

This is a directory tree. It is common in READMEs because it gives readers a
map before they browse files.

The comments after `#` are not Python comments here. They are plain text inside
a code block, but they borrow the visual style of shell/Python comments.

## Package names as architecture

Look at these folders:

- `runtime`
- `application`
- `strategies`
- `ingestion`
- `bus`
- `execution`
- `risk`
- `configs`

These are not random names. They are responsibility boundaries.

Common practice:

Name packages by what they own.

For example:

- `ingestion` owns getting external data into the system
- `execution` owns turning approved actions into orders
- `risk` owns safety checks and limits
- `bus` owns message passing
- `runtime` owns startup and wiring

This is one of the main differences between beginner scripts and real
applications. In a beginner script, code is often organized by the order you
wrote it. In a real application, code is organized by responsibility.

## The `mm/` folder

The main Python package is `mm/`.

A folder becomes a Python package when it has an `__init__.py` file. This repo
has `mm/__init__.py`, so code can import modules like:

```python
from mm.runtime import something
from mm.bus import something
```

Later, when we read implementation files, pay attention to imports. Imports show
which package depends on which other package.

## Tiny exercise

After reading this lesson, answer these:

1. Why is a docs index useful in a large project?
2. What is the difference between architecture, data flow, and mechanisms?
3. Which package would you inspect first for exchange websocket data?
4. Which package would you inspect first for live order placement?
