"""
Choose a torch.device for sampling / small ops.

Recent PyTorch wheels often ship **CUDA kernels only for sm_70+**. Older GPUs (e.g. GTX 960M,
sm_50) still report ``torch.cuda.is_available()==True`` but any CUDA tensor op fails at runtime.
This module defaults to **CPU** in that situation unless ``MM_TORCH_DEVICE`` forces otherwise.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_torch_device(explicit: str | None = None) -> "torch.device":
    """
    Resolve device for lightweight Torch ops.

    - ``explicit``: ``\"cpu\"``, ``\"cuda\"``, or ``None`` for automatic selection.
    - Env ``MM_TORCH_DEVICE``: ``cpu`` | ``cuda`` | unset (auto).
    - ``CUDA_VISIBLE_DEVICES=`` (empty) forces CPU.
    - Auto: CUDA only if PyTorch can actually run a tiny kernel on device 0 and capability looks supported.
    """
    import torch as torch_mod

    if explicit is not None and str(explicit).strip():
        d = torch_mod.device(explicit)
        if d.type == "cuda":
            return d if _cuda_runtime_ok(torch_mod) else torch_mod.device("cpu")
        return d

    env = os.environ.get("MM_TORCH_DEVICE", "").strip().lower()
    if env == "cpu":
        return torch_mod.device("cpu")
    if env == "cuda":
        return torch_mod.device("cuda") if _cuda_runtime_ok(torch_mod) else torch_mod.device("cpu")

    # Hide all GPUs → CPU-only
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return torch_mod.device("cpu")

    if not torch_mod.cuda.is_available():
        return torch_mod.device("cpu")

    # Skip GPUs too old for typical pip wheels (Maxwell etc.)
    try:
        major, _minor = torch_mod.cuda.get_device_capability(0)
        if major < 7:
            return torch_mod.device("cpu")
    except Exception:
        return torch_mod.device("cpu")

    return torch_mod.device("cuda") if _cuda_runtime_ok(torch_mod) else torch_mod.device("cpu")


def _cuda_runtime_ok(torch_mod) -> bool:
    try:
        x = torch_mod.zeros(2, dtype=torch_mod.float64, device="cuda")
        return bool(torch_mod.isfinite(x + 1.0).all().item())
    except Exception:
        return False


def resolve_torch_threads(explicit: int | str | None = None) -> int | None:
    """
    Resolve the intra-op thread count for the MPC rollout.

    The rollout tensors are small (a few thousand elements), so the default
    OpenMP fan-out costs more in synchronization than it recovers in parallel
    work. Measured on the Bitwana optimizer at ``N=64, H=4, E=40`` on CPU:

    ============  ============
    threads       wall time
    ============  ============
    1             178.7 ms
    2             186.7 ms
    4 (default)   195.1 ms
    8             215.9 ms
    ============  ============

    Extra threads made it monotonically slower, so the default is pinned to a
    single thread unless ``MM_TORCH_THREADS`` overrides it.

    Returns ``None`` when no override applies, meaning "leave torch as-is".
    """
    raw: str | int | None = explicit
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = os.environ.get("MM_TORCH_THREADS", "").strip()
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < 1:
        return None
    return value


def apply_torch_threads(explicit: int | str | None = None) -> int | None:
    """Apply ``resolve_torch_threads`` to the global torch thread pool."""
    import torch as torch_mod

    threads = resolve_torch_threads(explicit)
    if threads is None:
        return None
    try:
        torch_mod.set_num_threads(threads)
    except Exception:
        return None
    return threads
