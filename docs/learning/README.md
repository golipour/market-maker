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

## Suggested next files

After lesson 1, a good order is:

1. `docs/README.md` - learn how project documentation is organized.
2. `docs/architecture.md` - learn the big-picture system shape.
3. `docs/data-flow.md` - learn how data moves through the application.
4. `requirements.txt` - learn how dependencies are declared.
5. `docker-compose.yml` - learn how real apps run supporting services.
6. `mm/runtime/__main__.py` and `mm/runtime/config.py` - learn how the Python program starts and reads configuration.

