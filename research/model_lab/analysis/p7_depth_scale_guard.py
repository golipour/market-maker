"""Does ``depth_scale`` force a separate inductor specialization?

The joint path calls ``soft_fill_batched`` twice per horizon step:

    local  -> depth_scale=1.0        (hardcoded)
    global -> depth_scale=global_mask (0.0 or 1.0, per decide)

``depth_scale`` is a Python ``float`` consumed as ``marg * scale``. Dynamo
guards on Python scalars, so 0.0 and 1.0 may compile to two different kernels.
The existing warm-up never passes ``depth_scale`` (it defaults to 1.0), so if
the value *is* specialized, the global chain would hit an unwarmed
specialization and pay a lazy compile inside a decide -- the exact failure
``_SOFT_FILL_WARMED_SIGNATURES`` was added to prevent (measured 162 s, which
blew ``decision_timeout_ms`` and wedged every controller).

This measures the guard behaviour directly so the wiring is designed against
fact rather than assumption:

  1. compile once with depth_scale=1.0, count frames
  2. call with depth_scale=0.0, count frames again
  3. if frames grew, the scalar is specialized and the signature must include it

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p7_depth_scale_guard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mm.mpc.l_param import LParamConfig  # noqa: E402
from mm.mpc.mpc_tensor_batched import soft_fill_batched  # noqa: E402

N_COMPONENTS = 2
N_SAMPLES = 4
GRID = 2


def _frames() -> int:
    return int(torch._dynamo.utils.counters["frames"].get("total", 0))


def main() -> None:
    torch._dynamo.reset()
    compiled = torch.compile(soft_fill_batched, fullgraph=False)

    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=float(GRID),
        n_components=N_COMPONENTS,
    )
    dt = torch.float64
    params = torch.zeros(3 * 2 * N_COMPONENTS + 3, dtype=dt, requires_grad=True)
    mid = torch.ones((N_SAMPLES, 1), dtype=dt)
    book = torch.zeros((N_SAMPLES, GRID), dtype=dt)
    vols = torch.ones((N_SAMPLES, 1), dtype=dt)
    grid = torch.arange(1, GRID + 1, dtype=dt)
    size = torch.full((N_SAMPLES, 1), 1.0, dtype=dt)

    def call(scale: float) -> float:
        t0 = time.perf_counter()
        ob_b, ob_a, cash, inv = compiled(
            params, N_COMPONENTS, mid, book, book, vols, vols, grid, 0.0,
            soft_temp=1.0,
            soft_fill_activation="sigmoid",
            soft_fill_queue_model="back_of_queue",
            bid_quote_size_min=size, bid_quote_size_max=size,
            ask_quote_size_min=size, ask_quote_size_max=size,
            use_softplus=True, l_param=l_param,
            depth_scale=scale,
        )
        (ob_b.sum() + ob_a.sum() + cash.sum() + inv.sum()).backward()
        params.grad = None
        return (time.perf_counter() - t0) * 1000.0

    print("--- depth_scale=1.0 (local chain) ---")
    for i in range(3):
        ms = call(1.0)
        print(f"  pass {i}: {ms:9.2f} ms   frames_total={_frames()}")
    frames_after_one = _frames()

    print("\n--- depth_scale=0.0 (global chain, masked) ---")
    for i in range(3):
        ms = call(0.0)
        print(f"  pass {i}: {ms:9.2f} ms   frames_total={_frames()}")
    frames_after_zero = _frames()

    print("\n--- back to depth_scale=1.0 ---")
    for i in range(2):
        ms = call(1.0)
        print(f"  pass {i}: {ms:9.2f} ms   frames_total={_frames()}")
    frames_final = _frames()

    print(f"\nframes after 1.0 warm : {frames_after_one}")
    print(f"frames after 0.0 call : {frames_after_zero}")
    print(f"frames after 1.0 again: {frames_final}")

    grew = frames_after_zero > frames_after_one
    if grew:
        print("\nVERDICT: depth_scale IS specialized.")
        print("  -> the signature MUST include depth_scale, and warm-up must")
        print("     cover 0.0 as well as 1.0, or the global chain triggers a")
        print("     lazy compile on the decide path.")
    else:
        print("\nVERDICT: depth_scale is not specialized; one signature covers both.")


if __name__ == "__main__":
    main()