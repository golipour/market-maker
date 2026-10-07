"""Count dynamo specializations of ``materialize_side_torch`` before/after.

Production hit ``torch._dynamo hit config.recompile_limit (8)`` with
``last reason: l_param.quote_size_max is None``, after which dynamo falls back
to eager permanently for the function -- silently disabling the compiled
kernel. This counts how many distinct specializations the production venue
matrix produces.

Run: python3 scripts/verify_dynamo_specializations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._dynamo as dynamo

from mm.mpc.l_param import LParamConfig

# Production venue matrix: local (4 components) + global (2 components), each
# with and without a quote-size cap, matching the two configured venues.
VENUES = [
    ("local-uncapped", 4, None),
    ("local-capped", 4, 100.0),
    ("global-uncapped", 2, None),
    ("global-capped", 2, 100.0),
]


def main() -> None:
    dynamo.reset()
    dynamo.config.recompile_limit = 64  # observe, don't trip
    dynamo.config.accumulated_recompile_limit = 1024

    from mm.mpc.l_param import materialize_side_torch

    # dynamo only traces inside a compiled region; in production this is
    # reached through the compiled ``soft_fill_batched``, so compile it here
    # too or the counters stay at zero.
    compiled = torch.compile(materialize_side_torch, fullgraph=False)

    for name, n_comp, cap in VENUES:
        l_param = LParamConfig(
            quote_size_min=1.0,
            quote_size_max=cap,
            spread_max=200.0,
            n_components=n_comp,
        )
        params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
        size_min = torch.zeros((1, 1), dtype=torch.float64)
        size_max = torch.full((1, 1), 50.0, dtype=torch.float64)
        try:
            out = compiled(
                params,
                side_off=0,
                n_components=n_comp,
                l_param=l_param,
                quote_size_min=size_min,
                quote_size_max=size_max,
            )
            finite = all(bool(torch.isfinite(t).all()) for t in out)
            frames_so_far = dynamo.utils.counters["frames"].get("total", 0)
            print(f"  {name:18s} ok finite={finite} frames={frames_so_far}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:18s} FAILED {type(exc).__name__}: {exc}")

    frames = dynamo.utils.counters["frames"]
    recompiles = dict(dynamo.utils.counters.get("recompiles", {}))
    print()
    print(f"frames_total = {frames.get('total', 0)}")
    print(f"recompiles   = {recompiles}")
    total = frames.get("total", 0)
    limit = dynamo.config.cache_size_limit
    print(f"cache_size_limit = {limit}")
    print()
    if total > limit:
        print(f"WOULD TRIP THE LIMIT ({total} > {limit})")
        raise SystemExit(1)
    print(f"OK: {total} specializations, under the default limit of {limit}")


if __name__ == "__main__":
    main()