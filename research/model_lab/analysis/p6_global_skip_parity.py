"""Can the global rollout chain be skipped when ``global_active`` is False?

p5 established that with ``global_active=False`` the joint path costs the same
as with it True (43.2 vs 43.1 ms) and returns a bit-identical *local* action.
But bit-identical local action is necessary, not sufficient: the rollout also
feeds

  * ``net_inv_next = inv_next + global_inv_next`` into
    ``inventory_mark_price_batched``, which when ``use_global_reference`` prices
    the *local* inventory off the **global** buy/sell VWAP, and
  * ``step_reward = local_step_reward + global_step_reward`` -- note this sum is
    *not* multiplied by ``global_mask``.

So the global chain can influence the local decision even when masked. This
script implements the skip the way ``joint_optimizer.py`` would have to (drop
the global chain, feed a masked constant into the global terms) and checks
whether the local action survives bit-exactly.

If it does not, the skip is only safe behind an explicit reference-price
configuration, and the honest conclusion is that the win is conditional.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p6_global_skip_parity.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_state import OrderbookRelative  # noqa: E402
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402
import mm.mpc.joint_optimizer as joint_mod  # noqa: E402

PROD = dict(
    n_components=4,
    n_samples=64,
    horizon=2,
    spread_max=200.0,
    spread_step=20.0,
    adam_lr=1.0,
    adam_epochs=20,
)
GLOBAL_N_COMPONENTS = 2


class _FixedFlow:
    def sample(self, shape, device=None, dtype=None):
        dt = dtype or torch.float64
        return (
            torch.ones(shape, dtype=dt, device=device),
            torch.ones(shape, dtype=dt, device=device),
        )


class _FixedPrice:
    def sample_residual(self, shape, device=None, dtype=None):
        dt = dtype or torch.float64
        z = torch.zeros(shape, dtype=dt, device=device)
        return z, z


class _Exec:
    maker_fee = 0.0
    local_maker_fee = 0.0
    global_maker_fee = 0.0


def _with_global(base, *, active: bool):
    fields = dict(
        global_ob_rel=OrderbookRelative(bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))),
        global_active=active,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=base.buy_vwap * 0.999,
        global_sell_vwap=base.sell_vwap * 1.001,
        reference_price=base.sell_vwap,
    )
    for key, value in fields.items():
        object.__setattr__(base, key, value)
    return base


def _opt(**over):
    kw = {**PROD, **over}
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source="global",
        global_optimizer={"n_components": GLOBAL_N_COMPONENTS},
        local_optimizer={"n_components": PROD["n_components"]},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P6_THREADS", "1")),
        device="cpu",
        **kw,
    )


def _act_vec(a) -> np.ndarray:
    return np.concatenate([
        np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
        for s in ("bid", "ask") for p in ("L", "k", "x0")
    ])


def timeit(opt, states, warm=2, reps=5):
    for _ in range(warm):
        for st in states:
            opt.optimize_with_diagnostics(st)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for st in states:
            opt.optimize_with_diagnostics(st)
        out.append((time.perf_counter() - t0) * 1000.0 / len(states))
    return out


def main() -> None:
    n_states = int(os.environ.get("P6_STATES", "6"))
    raw = build_states(n_states=n_states)

    # --- baseline: unmodified joint path, global inactive ---------------------
    base_states = [_with_global(st, active=False) for st, _ in raw]
    base_opt = _opt()
    base_ms = statistics.median(timeit(base_opt, base_states))
    base_actions = [_act_vec(base_opt.optimize_with_diagnostics(st)[0]) for st in base_states]

    # --- patched: global chain short-circuited when global_mask == 0 -----------
    real_soft_fill = joint_mod.soft_fill_batched
    n_samples = PROD["n_samples"]

    def _skipped_soft_fill(h, n, *args, **kwargs):
        """Return the masked no-op the skip would produce."""
        if kwargs.get("depth_scale", 1.0) == 0.0:
            zeros = torch.zeros(n_samples, args[2].shape[-1], dtype=args[2].dtype)
            return (zeros, zeros.clone(), zeros[:, :1].clone(), zeros[:, :1].clone())
        return real_soft_fill(h, n, *args, **kwargs)

    joint_mod.soft_fill_batched = _skipped_soft_fill
    try:
        skip_opt = _opt()
        skip_ms = statistics.median(timeit(skip_opt, base_states))
        skip_actions = [_act_vec(skip_opt.optimize_with_diagnostics(st)[0]) for st in base_states]
    finally:
        joint_mod.soft_fill_batched = real_soft_fill

    print(f"threads={base_opt.torch_threads}  corpus={len(base_states)} states")
    print(f"\nbaseline joint, global inactive : {base_ms:8.2f} ms/decide")
    print(f"global chain skipped            : {skip_ms:8.2f} ms/decide")
    print(f"speedup                         : {base_ms / skip_ms:.3f}x")

    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(base_actions, skip_actions)]
    worst = max(diffs)
    print(f"\nmax |local action diff| across corpus = {worst:.3e}")
    exact = worst == 0.0
    print("VERDICT: " + (
        "bit-exact -- skip is safe to ship for global_active=False"
        if exact else
        "NOT exact -- the global chain is load-bearing for the local decision; skip is unsafe"
    ))


if __name__ == "__main__":
    main()