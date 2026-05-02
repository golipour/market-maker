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
