"""Does the production venue matrix fit under the default recompile limit (8)?

Reproduces the real production shape -- two venues (nobitex local, bitwana
global) x two sides, each with its own ``LParamConfig`` -- at dynamo's
*default* ``cache_size_limit`` of 8, and asserts dynamo does not trip.

Pre-fix, every distinct bound value also inserted a ``_SCALAR_TENSOR_CACHE``
key, and each insertion installed a ``len(G[...]) == N`` guard, so the frame
count multiplied and tripped the limit within a couple of decides. Post-fix the
cache is not touched while tracing, leaving one specialization per config.

Run: OMP_NUM_THREADS=2 python3 scripts/verify_production_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._dynamo as dynamo

from mm.mpc.l_param import LParamConfig, materialize_side_torch

# venue, n_components, quote_size_max
PRODUCTION = [
    ("nobitex-local  bid", 4, None),
    ("nobitex-local  ask", 4, None),
    ("bitwana-global bid", 2, None),
    ("bitwana-global ask", 2, None),
]


def main() -> None:
    dynamo.reset()
    dynamo.config.recompile_limit = 8  # dynamo default cache_size_limit
    dynamo.config.accumulated_recompile_limit = 256
    print(f"recompile_limit = {dynamo.config.recompile_limit}")

    compiled = torch.compile(materialize_side_torch, fullgraph=False)

    for i, (name, n_comp, cap) in enumerate(PRODUCTION, start=1):
        l_param = LParamConfig(
            quote_size_min=1.0,
            quote_size_max=cap,
            spread_max=200.0,
            n_components=n_comp,
        )
        # Same bounds every decide -- this is the key production invariant.
        params = torch.zeros(
            3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True
        )
        size_min = torch.zeros((1, 1), dtype=torch.float64)
        size_max = torch.full((1, 1), 50.0, dtype=torch.float64)
        for _ in range(5):  # 5 decides, same config
            out = compiled(
                params,
                side_off=0,
                n_components=n_comp,
                l_param=l_param,
                quote_size_min=size_min,
                quote_size_max=size_max,
            )
            sum(t.sum() for t in out).backward()
            params.grad = None
        frames = dynamo.utils.counters["frames"].get("total", 0)
        print(f"  [{i}] {name:20s} frames_total={frames}")

    frames = dynamo.utils.counters["frames"].get("total", 0)
    print(f"\nframes_total = {frames} / limit {dynamo.config.recompile_limit}")

    from mm.mpc.l_param import _SCALAR_TENSOR_CACHE

    print(f"_SCALAR_TENSOR_CACHE keys touched while tracing = {len(_SCALAR_TENSOR_CACHE)}")

    if frames > dynamo.config.recompile_limit:
        print("FAILED: exceeds the default recompile limit; would fall back to eager")
        raise SystemExit(1)
    print("OK: production matrix fits under the default limit")


if __name__ == "__main__":
    main()