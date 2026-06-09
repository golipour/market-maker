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

## Suggested next files

After lesson 5, a good order is:

1. More of `mm/runtime/config.py` - learn mapping JSON into config objects.
2. `mm/bus/base.py` and `mm/bus/memory.py` - learn interfaces and async implementations.
3. `mm/runtime/engine.py` - learn how components are wired together.
4. `mm/application/strategy_runner.py` - learn the main decision orchestration pipeline.
