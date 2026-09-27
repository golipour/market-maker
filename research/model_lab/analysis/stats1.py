"""Shared, alignment-safe statistics for the analysis scripts.

Everything assumes a single uniform time axis (see harness.derived).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def corr(a, b):
    """Pearson corr on the finite overlap; returns (r, n)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 3:
        return np.nan, n
    x, y = a[m], b[m]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan, n
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)), n


def acf(a, k, mask=None):
    """corr(a[t], a[t+k]) on the true lag, restricted to mask[t] and mask[t+k]."""
    a = np.asarray(a, dtype=float)
    if k <= 0:
        return np.nan, len(a)
    mask = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    m = mask[:-k] & mask[k:] & np.isfinite(a[:-k]) & np.isfinite(a[k:])
    return corr(a[:-k][m], a[k:][m])


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, int(m.sum())
    r, p = stats.spearmanr(a[m], b[m])
    return float(r), int(m.sum())


def block_length(n, frac=3.0):
    return max(2, int(round(n ** (1.0 / frac))))


def moving_block_ci(diffs, block=None, B=2000, alpha=0.05, seed=0):
    """Percentile CI for the mean of a moving-block-resampled loss difference.

    diffs is the per-step paired loss difference (candidate - baseline);
    negative mean means the candidate is better.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return (np.nan, np.nan, np.nan, n)
    if block is None:
        block = block_length(n)
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    means = np.empty(B)
    for b in range(B):
        idx = []
        pos = 0
        while pos < n:
            start = int(rng.integers(0, n))
            L = int(np.floor(np.log(rng.random()) / np.log1p(-p))) + 1
            idx.append(np.arange(start, start + L) % n)
            pos += L
        means[b] = d[np.concatenate(idx)[:n]].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi), n


def dm_test(diffs, block=None):
    """HAC (Newey-West) t-test that the mean loss difference is zero."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dm = d - d.mean()
    if block is None:
        block = block_length(n)
    lag = block - 1
    s = float((dm ** 2).sum() / n)
    for k in range(1, lag + 1):
        c = float((dm[k:] * dm[:-k]).sum() / n)
        s += 2.0 * (1.0 - k / (lag + 1.0)) * c
    s = max(s, 1e-300)
    t = float(dm.sum() / np.sqrt(s * n))
    p = float(2.0 * stats.t.sf(abs(t), df=max(1, n - 1)))
    return t, p


def entropy_bits(counts, base=2.0):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(base))


def hist_entropy(x, bins, base=2.0):
    """Discrete entropy of a binned variable; returns (bits, n)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 0
    c, _ = np.histogram(x, bins=bins)
    return entropy_bits(c, base), len(x)


def knn_cond_mi(x, y, k=32, block=4000, seed=0):
    """Kozachenko-Leonenko conditional mutual information estimate, in bits.

    I(X;Y) where Y is the target column and X the conditioning features,
    estimated on a block subsample by the K-nearest-neighbour radius
    comparison between the joint and the product metric.
    """
    from scipy.special import digamma

    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    m = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[m], Y[m]
    n = len(X)
    if n < 400:
        return np.nan, 0
    if n > block:
        idx = np.sort(np.random.default_rng(seed).choice(n, block, replace=False))
        X, Y = X[idx], Y[idx]
        n = block

    def nn_radius(A, skip):
        diff = A[:, None, :] - A[None, :, :]
        D = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))
        if skip:
            D[np.arange(len(A)), np.arange(len(A)) + skip] = np.inf
        np.sort(D, axis=1, out=D)
        return np.maximum(D[:, k - 1], 1e-12)

    Dxy = nn_radius(np.hstack([X, Y]), 1)
    Dx = nn_radius(X, 1)
    Dy = nn_radius(Y, 1)
    mi = (digamma(k) - digamma(n)
          + np.mean(digamma(Dx)) + np.mean(digamma(Dy)) - np.mean(digamma(Dxy)))
    return float(mi / np.log(2.0)), n
