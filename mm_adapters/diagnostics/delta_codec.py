"""
Compact diagnostics deltas for Pub/Sub (Redis SET still stores the merged full frame).

Envelope:
  {"schema":"mm.diag.v2","kind":"full","step_counter":N,"frame":{...}}
  {"schema":"mm.diag.v2","kind":"delta","step_counter":N,"patch":...}

Patch nodes:
  - dict: shallow-merge recursively; omitted keys unchanged.
  - list growth with identical prefix: {"$append": [new elements]}.
  - otherwise: {"$replace": <whole new value>}.
"""

from __future__ import annotations

import copy
from typing import Any

SCHEMA = "mm.diag.v2"
KIND_FULL = "full"
KIND_DELTA = "delta"

REPLACE = "$replace"
APPEND = "$append"


def compute_patch(prev: Any, curr: Any) -> Any | None:
    if prev == curr:
        return None
    if isinstance(curr, dict) and isinstance(prev, dict):
        for k in prev:
            if k not in curr:
                return {REPLACE: copy.deepcopy(curr)}
        out: dict[str, Any] = {}
        for k, v in curr.items():
            if k not in prev:
                out[k] = copy.deepcopy(v)
            else:
                sub = compute_patch(prev[k], v)
                if sub is not None:
                    out[k] = sub
        return out if out else None
    if isinstance(curr, list) and isinstance(prev, list):
        if len(curr) >= len(prev) and curr[: len(prev)] == prev:
            if len(curr) == len(prev):
                return None
            return {APPEND: copy.deepcopy(curr[len(prev) :])}
        return {REPLACE: copy.deepcopy(curr)}
    return {REPLACE: copy.deepcopy(curr)}


def apply_patch(base: Any, patch: Any) -> Any:
    """Apply *patch* onto *base*; immutability-safe for callers (deepcopy internally)."""
    if patch is None:
        return copy.deepcopy(base)
    if isinstance(patch, dict):
        if REPLACE in patch:
            return copy.deepcopy(patch[REPLACE])
        if APPEND in patch:
            b = base if isinstance(base, list) else []
            return copy.deepcopy(b) + copy.deepcopy(patch[APPEND])
        if isinstance(base, dict):
            out = copy.deepcopy(base)
            for k, v in patch.items():
                if k not in out:
                    out[k] = copy.deepcopy(v)
                else:
                    out[k] = apply_patch(out[k], v)
            return out
    return copy.deepcopy(patch)


def envelope_full(step_counter: Any, frame: dict[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "kind": KIND_FULL, "step_counter": step_counter, "frame": frame}


def envelope_delta(step_counter: Any, patch: Any) -> dict[str, Any]:
    return {"schema": SCHEMA, "kind": KIND_DELTA, "step_counter": step_counter, "patch": patch}


def is_v2_envelope(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("schema") == SCHEMA
