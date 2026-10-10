"""Does recompilation STOP once warm, or continue forever?

``verify_recompile_triggers`` showed 3 frames for 12 calls with identical
shapes and only values changing. That is ambiguous: 3 could be 1 compile plus
2 warmup variations, or it could be churn that would never stop.

This prints the frame count after EVERY call, so the shape of the curve is
visible. A curve that flattens means "compiled once, then stable"; a line
that keeps climbing means "recompiling forever".

Run: OMP_NUM_THREADS=2 python3 scripts/verify_recompile_steady.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._dynamo as dynamo

from mm.mpc.l_param import LParamConfig, materialize_side_torch

N_CALLS = 40
N_COMP = 4


def main() -> None:
    dynamo.reset()
    dynamo.config.recompile_limit = 64
    dynamo.config.accumulated_recompile_limit = 4096
    compiled = torch.compile(materialize_side_torch, fullgraph=False)

    lp = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=200.0,
        n_components=N_COMP,
        reservation_r_max=0.002,
    )
    smin = torch.zeros((1, 1), dtype=torch.float64)
    smax = torch.full((1, 1), 50.0, dtype=torch.float64)

    print(f"=== frame count per call, identical shapes+config, values change ===")
    prev = dynamo.utils.counters["frames"].get("total", 0)
    timeline = []
    for i in range(N_CALLS):
        params = torch.full(
            (3 * 2 * N_COMP + 3,), float(i) * 0.01, dtype=torch.float64,
            requires_grad=True,
        )
        compiled(
            params,
            side_off=0,
            n_components=N_COMP,
            l_param=lp,
            quote_size_min=smin,
            quote_size_max=smax,
        )
        now = dynamo.utils.counters["frames"].get("total", 0)
        timeline.append(now)
        marker = "  <-- RECOMPILED" if now != prev else ""
        print(f"  call {i:3d}: frames={now:2d}{marker}")
        prev = now

    total = timeline[-1]
    first_five = timeline[:5]
    last_five = timeline[-5:]
    print()
    print(f"  frames after call 1  : {first_five[0]}")
    print(f"  frames after call 5  : {first_five[-1]}")
    print(f"  frames after call {N_CALLS}: {total}")
    print(f"  frames added in last 35 calls: {total - first_five[-1]}")
    print()
    if total - first_five[-1] == 0:
        print("RESULT: compilation STABILIZES. It compiles a few times during")
        print("        warmup, then never again. Values do not cause churn.")
    else:
        print(f"RESULT: STILL RECOMPILING ({(total - first_five[-1]) / 35:.1f}/call).")
        print("        Something other than values is varying.")
    _ = last_five


if __name__ == "__main__":
    main()