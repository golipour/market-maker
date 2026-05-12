"""Backward-compatible CLI shim; implementation lives in ``mm.tools.print_events``."""

from mm.tools.print_events import (
    PrintStats,
    build_print_queues,
    main,
    parse_args,
    print_events,
    resolve_api_token,
    run,
    run_print_loop,
    summarize_contract,
    summarize_event,
)

__all__ = [
    "PrintStats",
    "build_print_queues",
    "main",
    "parse_args",
    "print_events",
    "resolve_api_token",
    "run",
    "run_print_loop",
    "summarize_contract",
    "summarize_event",
]


if __name__ == "__main__":
    main()
