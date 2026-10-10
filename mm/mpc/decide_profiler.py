"""Optional ``torch.profiler`` hook for the MPC decide path.

The decide step costs ~1.5-2.5 s of wall time and the per-stage counters
recorded by :class:`OptimizerStageTimer` attribute it only coarsely
(``objective`` / ``adam`` / ``diagnostics``). That is not enough to tell an
FLOP-bound path (more cores would help) from a dispatch- or memory-bound one
(more cores would not), which is exactly the question the current 2-core CPU
limit raises.

This wraps one optimizer entry point with ``torch.profiler`` and writes a
Chrome/Perfetto trace. It is opt-in and costs nothing when disabled:

* ``MM_PROFILE_DECIDE=<path>`` -- profile and write the trace to ``path``.
* ``MM_PROFILE_DECIDE=1`` -- profile and write to
  ``$MM_PROFILE_DECIDE_PATH`` or ``/var/cache/torchinductor/decide_trace.json``.

Set ``MM_PROFILE_DECIDE_LABEL`` to tag the trace, and
``MM_PROFILE_DECIDE_MAX_DURATION_S`` to cap how long profiling may run before
the trace is flushed and profiling disables itself, so a forgotten flag cannot
add overhead to every subsequent decide.

Usage in the optimize path::

    from mm.mpc.decide_profiler import maybe_profile_decide

    with maybe_profile_decide("bitwana"):
        loss = optimizer.step(...)

Then read the trace::

    # Chrome: chrome://tracing -> Load profile
    # or:  perfetto ui trace.json
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Iterator

DEFAULT_TRACE_PATH = "/var/cache/torchinductor/decide_trace.json"
_ENABLED_ENV = "MM_PROFILE_DECIDE"
_LABEL_ENV = "MM_PROFILE_DECIDE_LABEL"
_PATH_ENV = "MM_PROFILE_DECIDE_PATH"
_MAX_S_ENV = "MM_PROFILE_DECIDE_MAX_DURATION_S"

_state_lock = threading.Lock()
# Set once the trace has been flushed so a persistent flag profiles exactly
# one decide rather than every decide (profiling adds real overhead).
_profiled_once = False


def profiling_enabled() -> bool:
    """Whether ``MM_PROFILE_DECIDE`` asks for profiling."""
    return bool(os.environ.get(_ENABLED_ENV, "").strip())


def _resolve_path() -> str:
    raw = os.environ.get(_ENABLED_ENV, "").strip()
    # ``MM_PROFILE_DECIDE=1`` means "enabled, use the default/PATH location";
    # any other value is treated as the path itself.
    if raw and raw not in {"1", "true", "yes", "on"}:
        return raw
    return os.environ.get(_PATH_ENV, "").strip() or DEFAULT_TRACE_PATH


def _max_duration_s() -> float | None:
    raw = os.environ.get(_MAX_S_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _disable_flag(reason: str) -> None:
    """Clear the enable var so profiling does not repeat."""
    os.environ.pop(_ENABLED_ENV, None)
    print(
        f"[decide-profiler] disabled: {reason}",
        flush=True,
    )


def _write_summary(trace_path: str, label: str, duration_ms: float) -> None:
    """Write a small text summary next to the trace for quick triage.

    The trace itself is the source of truth, but a one-screen summary of the
    top operators saves opening the Perfetto UI for the common case.
    """
    summary_path = trace_path.rsplit(".", 1)[0] + "_summary.txt"
    try:
        with open(trace_path) as fh:
            trace = json.load(fh)
    except (OSError, ValueError):
        # Some torch versions emit a JSON list of events rather than an object.
        try:
            with open(trace_path) as fh:
                trace = json.load(fh)
        except Exception:  # noqa: BLE001
            print(
                f"[decide-profiler] could not summarize {trace_path}",
                flush=True,
            )
            return

    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    by_name: dict[str, float] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") not in {"X"}:
            continue
        name = str(event.get("name", ""))
        dur = float(event.get("dur", 0) or 0)
        # Torch reports CPU-side kernel time in microseconds.
        if dur <= 0:
            continue
        by_name[name] = by_name.get(name, 0.0) + dur

    top = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:20]
    try:
        with open(summary_path, "w") as fh:
            fh.write(f"label: {label}\n")
            fh.write(f"wall: {duration_ms:.1f} ms\n")
            fh.write(f"trace: {trace_path}\n\n")
            fh.write(f"{'cum us':>14}  {'share':>6}  name\n")
            total = sum(v for _, v in top) or 1.0
            for name, dur in top:
                fh.write(f"{dur:14.0f}  {100 * dur / total:5.1f}%  {name}\n")
    except OSError:
        return
    print(f"[decide-profiler] summary -> {summary_path}", flush=True)


@contextlib.contextmanager
def maybe_profile_decide(label: str = "") -> Iterator[None]:
    """Profile the enclosed decide block if ``MM_PROFILE_DECIDE`` is set.

    A no-op context manager when disabled, so callers can wrap the hot path
    unconditionally without branching.
    """
    global _profiled_once

    if not profiling_enabled():
        yield
        return

    with _state_lock:
        if _profiled_once:
            yield
            return
        _profiled_once = True

    import torch

    trace_path = _resolve_path()
    label = label or os.environ.get(_LABEL_ENV, "").strip() or "decide"
    max_s = _max_duration_s()
    started = time.perf_counter()

    activities = [torch.profiler.ProfilerActivity.CPU]
    print(
        f"[decide-profiler] profiling '{label}' -> {trace_path}",
        flush=True,
    )

    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            yield
    except Exception as exc:  # noqa: BLE001
        _disable_flag(f"profiler error: {type(exc).__name__}: {exc}")
        raise

    elapsed = time.perf_counter() - started
    try:
        prof.export_chrome_trace(trace_path)
    except Exception as exc:  # noqa: BLE001
        _disable_flag(f"trace export failed: {type(exc).__name__}: {exc}")
        return

    _write_summary(trace_path, label, elapsed * 1000.0)
    if max_s is not None and elapsed > max_s:
        _disable_flag(f"exceeded MM_PROFILE_DECIDE_MAX_DURATION_S={max_s}")
    else:
        # Profile exactly one decide; a persistent flag would otherwise add
        # profiler overhead to every subsequent cycle.
        _disable_flag("captured one decide")


def reset_for_tests() -> None:
    """Allow a second profile in the same process (used by tests)."""
    global _profiled_once
    with _state_lock:
        _profiled_once = False