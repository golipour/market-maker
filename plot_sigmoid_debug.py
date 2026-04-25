"""
Standalone debug plot for bid/ask sigmoid-mixture depth curves.

Usage:
  python3 plot_sigmoid_debug.py
  python3 plot_sigmoid_debug.py --spread-max 100 --num-points 1201
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _softplus_positive(v: float) -> float:
    """Map raw optimizer param to positive effective value."""
    x = float(v)
    if x >= 20.0:
        return x + 1e-8
    if x <= -20.0:
        return 1e-8
    return math.log(1.0 + math.exp(x)) + 1e-8


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable enough for our parameter ranges.
    return 1.0 / (1.0 + np.exp(-z))


def mixture_depth(
    x: np.ndarray,
    L: Sequence[float],
    k: Sequence[float],
    x0: Sequence[float],
) -> np.ndarray:
    if not (len(L) == len(k) == len(x0)):
        raise ValueError("L, k, x0 must have equal lengths.")
    y = np.zeros_like(x, dtype=np.float64)
    for Li, ki, x0i in zip(L, k, x0):
        y += float(Li) * _sigmoid(float(ki) * (x - float(x0i)))
    return y


def mixture_marginal(
    x: np.ndarray,
    L: Sequence[float],
    k: Sequence[float],
    x0: Sequence[float],
) -> np.ndarray:
    if not (len(L) == len(k) == len(x0)):
        raise ValueError("L, k, x0 must have equal lengths.")
    y = np.zeros_like(x, dtype=np.float64)
    for Li, ki, x0i in zip(L, k, x0):
        s = _sigmoid(float(ki) * (x - float(x0i)))
        y += float(Li) * float(ki) * s * (1.0 - s)
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot bid/ask sigmoid-mixture curves.")
    p.add_argument("--spread-max", type=float, default=100.0, help="Plot x in [-spread_max, +spread_max].")
    p.add_argument("--num-points", type=int, default=1001, help="Number of points in x grid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spread_max = max(1.0, float(args.spread_max))
    num_points = max(201, int(args.num_points))

    # Defaults from your debug example (raw optimizer-space params).
    ask_L_raw = (23,)
    ask_k_raw = (1e-6,)
    ask_x0 = (50.0,)
    bid_L_raw = (23,)
    bid_k_raw = (1e-6,)
    bid_x0 = (50.0,)

    # Plot using effective model-space params (L,k > 0).
    ask_L = tuple(_softplus_positive(v) for v in ask_L_raw)
    ask_k = tuple(_softplus_positive(v) for v in ask_k_raw)
    bid_L = tuple(_softplus_positive(v) for v in bid_L_raw)
    bid_k = tuple(_softplus_positive(v) for v in bid_k_raw)
            

    x = np.linspace(-spread_max, spread_max, num_points, dtype=np.float64)
    bid_depth = mixture_depth(x, bid_L, bid_k, bid_x0)
    ask_depth = mixture_depth(x, ask_L, ask_k, ask_x0)
    bid_marg = mixture_marginal(x, bid_L, bid_k, bid_x0)
    ask_marg = mixture_marginal(x, ask_L, ask_k, ask_x0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(x, bid_depth, label="Bid cumulative depth", color="tab:blue", linewidth=1.8)
    ax0.plot(x, ask_depth, label="Ask cumulative depth", color="tab:red", linewidth=1.8)
    ax0.set_ylabel("Cumulative depth")
    ax0.set_title("Sigmoid-mixture cumulative depth")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best")

    ax1 = axes[1]
    ax1.plot(x, bid_marg, label="Bid marginal depth dD/dx", color="tab:blue", linestyle="--", linewidth=1.6)
    ax1.plot(x, ask_marg, label="Ask marginal depth dD/dx", color="tab:red", linestyle="--", linewidth=1.6)
    ax1.set_xlabel("Spread ticks (x)")
    ax1.set_ylabel("Marginal depth")
    ax1.set_title("Sigmoid-mixture marginal depth (derivative)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
