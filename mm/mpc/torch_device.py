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


def _cgroup_cpu_quota() -> int | None:
    """Return the cgroup v2 ``cpu.max`` quota in whole CPUs, if declared.

    Returns ``None`` when the quota is unset, unlimited, or the file is not
    readable (no cgroup v2, or a plain local run).
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota_raw, _period = fh.read().split()[:2]
    except (OSError, ValueError):
        return None
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
    except ValueError:
        return None
    if quota <= 0:
        return None
    return max(1, quota // 100_000)


def _cgroup_v1_cpu_quota() -> int | None:
    """Return the cgroup v1 CPU quota in whole CPUs, if declared."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    if quota <= 0:
        return None
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    if period <= 0:
        return None
    return max(1, quota // period)


def resolve_cpu_quota() -> int | None:
    """Whole-CPU quota this container is limited to, or ``None`` if unlimited.

    ``torch.get_num_threads()`` defaults to the *node's* core count, which
    inside a container routinely exceeds the CPU quota the scheduler actually
    grants. The oversubscribed threads then contend with each other and the
    rollout gets slower, not faster.
    """
    return _cgroup_cpu_quota() or _cgroup_v1_cpu_quota()


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

    The oversubscription penalty is far worse for the ``torch.compile``d
    kernel, whose generated C++ spawns parallel regions that then contend for
    the quota. Measured in the production worker (2-CPU quota on an 8-CPU
    node) on one ``soft_fill_batched`` call, forward+backward:

    ============  ==========  ============
    threads       eager        compiled
    ============  ==========  ============
    1             5.54 ms      4.21 ms
    2            13.77 ms      2.79 ms
    4             6.68 ms      3.42 ms
    8            35.33 ms    622.83 ms
    ============  ==========  ============

    At 8 threads the compiled kernel ran 17x slower than eager: 0.057x, a
    regression rather than a speedup, purely from thread contention.

    So the count is pinned to the container's CPU quota when no explicit
    override is given, capped by the visible core count. An explicit argument
    or ``MM_TORCH_THREADS`` still wins.
    """
    raw: str | int | None = explicit
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = os.environ.get("MM_TORCH_THREADS", "").strip()
    if raw:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = 0
        if value >= 1:
            return value

    # No override: pin to the quota so the compiled kernel is not starved.
    quota = resolve_cpu_quota()
    if quota is None:
        return None
    try:
        import torch as torch_mod

        visible = torch_mod.get_num_threads()
    except Exception:
        visible = None
    if visible is None or visible < 1:
        return quota
    return min(quota, visible)


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
