"""Is the optimizer's action actually sensitive to the market state?

Two red flags in the early-stopping trace:
  * ``argmax_epoch`` was identical (28) for 5 of 6 states
  * ``dAct`` at a given epoch was bit-identical across those states

Either would be explained if the action were largely state-independent. This
script measures the action's sensitivity directly, and checks whether the
book/flow inputs actually reach the objective.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_state import OrderbookRelative, State  # noqa: E402
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402


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


def _vec(a) -> np.ndarray:
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(a.bid_L, dtype=float)),
            np.atleast_1d(np.asarray(a.bid_k, dtype=float)),
            np.atleast_1d(np.asarray(a.bid_x0, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_L, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_k, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_x0, dtype=float)),
        ]
    )


def _opt(epochs=20):
    return MonteCarloMPCTensorBatched(
        local_price_model=_FixedPrice(),
        local_market_flow_model=_FixedFlow(),
        execution_model=_Exec(),
        n_components=4,
        n_samples=64,
        horizon=2,
        spread_max=200.0,
        spread_step=20.0,
        adam_lr=1.0,
        adam_epochs=epochs,
        reference_price_source="local",
        fixed_mc_samples=True,
        torch_threads=1,
    )


def main() -> None:
    corpus = build_states(n_states=6)
    ref = corpus[0][0]

    print("=== A. does the action move when the book changes? ===")
    base = _opt()
    for _ in range(4):
        base.optimize_with_diagnostics(ref)
    a_base, d_base = base.optimize_with_diagnostics(ref)
    v_base = _vec(a_base)
    j_base = float(d_base.best.objective_value)
    print(f"  base J={j_base: .6e}")

    for name, mult in [("depth x0.1", 0.1), ("depth x10", 10.0)]:
        st = State(
            mid=ref.mid,
            ob_rel=OrderbookRelative(
                bids=tuple((p, q * mult) for p, q in ref.ob_rel.bids),
                asks=tuple((p, q * mult) for p, q in ref.ob_rel.asks),
            ),
            buy_vol=ref.buy_vol,
            sell_vol=ref.sell_vol,
            inventory=0.0,
            cash=10_000.0,
            buy_vwap=ref.buy_vwap,
            sell_vwap=ref.sell_vwap,
        )
        o = _opt()
        for _ in range(4):
            o.optimize_with_diagnostics(st)
        a, d = o.optimize_with_diagnostics(st)
        j = float(d.best.objective_value)
        print(
            f"  {name:12s} J={j: .6e} dJ={j - j_base: .3e} "
            f"|dAct|={np.max(np.abs(_vec(a) - v_base)):.6f}"
        )

    print("\n=== B. does mid price move the action? ===")
    for mult in [0.99, 1.01]:
        st = State(
            mid=ref.mid * mult,
            ob_rel=OrderbookRelative(
                bids=tuple((p * mult, q) for p, q in ref.ob_rel.bids),
                asks=tuple((p * mult, q) for p, q in ref.ob_rel.asks),
            ),
            buy_vol=ref.buy_vol,
            sell_vol=ref.sell_vol,
            inventory=0.0,
            cash=10_000.0,
            buy_vwap=ref.buy_vwap * mult,
            sell_vwap=ref.sell_vwap * mult,
        )
        o = _opt()
        for _ in range(4):
            o.optimize_with_diagnostics(st)
        a, d = o.optimize_with_diagnostics(st)
        j = float(d.best.objective_value)
        print(
            f"  mid x{mult}: J={j: .6e} dJ={j - j_base: .3e} "
            f"|dAct|={np.max(np.abs(_vec(a) - v_base)):.6f}"
        )

    print("\n=== C. are the replay states actually different? ===")
    for st, meta in corpus:
        bb, ba = st.ob_rel.bids[0][0], st.ob_rel.asks[0][0]
        print(
            f"  idx={meta['index']:6d} mid={st.mid:11.2f} bid={bb:11.2f} ask={ba:11.2f} "
            f"spread={ba - bb:7.1f} topQ_b={st.ob_rel.bids[0][1]:10.3f} "
            f"topQ_a={st.ob_rel.asks[0][1]:10.3f} bv={st.buy_vol:8.2f} sv={st.sell_vol:8.2f}"
        )

    print("\n=== D. per-state optimizer, no warm carry ===")
    vecs, js = [], []
    for st, meta in corpus:
        o = _opt()
        a, d = o.optimize_with_diagnostics(st)
        vecs.append(_vec(a))
        js.append(float(d.best.objective_value))
        print(f"  idx={meta['index']:6d} J={d.best.objective_value: .8e} bid_x0={np.asarray(a.bid_x0)}")
    A = np.array(vecs)
    J = np.array(js)
    print("\n  pairwise max |action_i - action_j| x1000:")
    G = np.array([[np.max(np.abs(A[i] - A[j])) * 1000 for j in range(len(A))] for i in range(len(A))])
    print(np.array2string(G, precision=3, suppress_small=True))
    print("\n  pairwise |J_i - J_j| x1e6:")
    G2 = np.array([[abs(J[i] - J[j]) * 1e6 for j in range(len(J))] for i in range(len(J))])
    print(np.array2string(G2, precision=3, suppress_small=True))
    print(
        f"\n  J spread across states: {J.max() - J.min():.6e} "
        f"(relative {abs(J.max() - J.min()) / max(abs(J.mean()), 1e-30):.3e})"
    )

    print("\n=== E. what do the action units mean? ===")
    print("  spread_max=200, spread_step=20 -> quote grid ticks 20,40,...,200")
    print(f"  bid_x0 (ticks) = {np.asarray(a_base.bid_x0)}")
    print(f"  bid_L (softmax logits) = {np.asarray(a_base.bid_L)}")
    print(f"  bid_k = {np.asarray(a_base.bid_k)}")


if __name__ == "__main__":
    main()
