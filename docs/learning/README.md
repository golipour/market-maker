# Learning path through this project

This folder is a guided tour of the market maker codebase for someone who knows
basic Python and wants to learn how real applications are written.

The goal is not only to say what each file does. The goal is to explain:

- how the file fits into the whole application
- why programmers commonly structure code this way
- what syntax patterns are being used
- what design patterns or architecture ideas appear
- what to notice when reading similar code in other projects

## How to use these notes

Read one lesson at a time. After each lesson, pause and inspect the real file it
discusses. The best rhythm is:

1. Read the lesson.
2. Open the original file.
3. Try to explain the file back in your own words.
4. Continue to the next lesson.

## Lessons

| Lesson | File | Why this comes first |
|--------|------|----------------------|
| [01 - Root README](01-root-readme.md) | `README.md` | The top-level README is the front door of many real projects. It tells you what the system is, where the real docs live, and how the app starts. |
| [02 - Docs README](02-docs-readme.md) | `docs/README.md` | A documentation index teaches you how the project wants to be navigated. |
| [03 - Architecture Overview](03-architecture-overview.md) | `docs/architecture.md` | The architecture doc teaches the main layers, responsibilities, and contracts before we enter implementation code. |
| [04 - Runtime Topics](04-runtime-topics.md) | `mm/core/topics.py` | First real code lesson: constants, Protocols, dataclasses, properties, static methods, f-strings, and centralized contracts. |
| [05 - Runtime Config Part 1](05-runtime-config-part-1.md) | `mm/runtime/config.py` | How real apps define typed config objects, normalize inputs, validate values, read environment variables, and build defaults. |
| [06 - Runtime Config Part 2](06-runtime-config-part-2.md) | `mm/runtime/config.py` | How raw JSON dictionaries are resolved into one typed runtime config object. |
| [07 - Event Bus Interface](07-event-bus-interface.md) | `mm/bus/base.py` | Abstract base classes, async method contracts, async iterators, queues, and adapter methods. |
| [08 - In-Memory Event Bus](08-in-memory-event-bus.md) | `mm/bus/memory.py` | A concrete async event-bus implementation with subscribers, queues, backpressure helpers, and cleanup. |

## Suggested next files

After lesson 8, a good order is:

1. `mm/bus/delivery.py` - learn queue backpressure and observability helpers.
2. `mm/runtime/engine.py` - learn how components are wired together.
3. `mm/application/strategy_runner.py` - learn the main decision orchestration pipeline.
4. `mm/application/frame_assembler.py` - learn how async inputs become strategy frames.
