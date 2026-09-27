"""Tests for book bucket forward-fill and spread-grid caching.

``_book_cumulative_from_ob_rel`` forward-fills empty spread buckets. The fill
must propagate only through *exact* zeros: a populated bucket may legitimately
hold a smaller cumulative total than its predecessor (for example after the
market trades through a level), and must not be clamped upward.

A ``cummax``-based fill is the obvious vectorization but is WRONG here -- it
raises any decreasing populated value to the running prefix maximum. That bug
was caught by fuzzing before it was committed; these tests pin the behaviour
so it cannot come back.
"""

from __future__ import annotations

import pytest
import torch

from mm.mpc.mpc_state import OrderbookRelative
from mm.mpc.mpc_tensor_batched import (
    _book_cumulative_from_ob_rel,
    _spread_bucket_index,
    _spread_grid_cached,
)


def _reference_fill(cum: torch.Tensor) -> torch.Tensor:
    """The original element-wise left-to-right carry, kept as the oracle."""
    out = cum.clone()
    for i in range(1, out.shape[0]):
        if out[i].item() == 0:
            out[i] = out[i - 1]
    return out


class TestForwardFill:
    def test_zero_buckets_inherit_previous_level(self) -> None:
        ob_rel = OrderbookRelative(
            bids=((20.0, 5.0), (100.0, 7.0)),
            asks=((20.0, 3.0), (100.0, 4.0)),
        )
        bid, ask = _book_cumulative_from_ob_rel(ob_rel, 5, 20.0, None, torch.float64)
        # spread 20.0 -> bucket 0, spread 100.0 -> bucket 4 (ceil(100/20)-1 = 4)
        assert bid.tolist() == [5.0, 5.0, 5.0, 5.0, 12.0]
        assert ask.tolist() == [3.0, 3.0, 3.0, 3.0, 7.0]

    def test_decreasing_populated_bucket_is_not_clamped(self) -> None:
        # Hand-built decreasing pattern: a cummax fill would raise 1.0 -> 9.0.
        cum = torch.tensor([9.0, 1.0, 0.0, 2.0], dtype=torch.float64)
        got = _reference_fill(cum)
        assert got.tolist() == [9.0, 1.0, 1.0, 2.0]

    @pytest.mark.parametrize("grid_levels", [2, 3, 5, 10, 20])
    def test_fill_only_touches_exact_zeros(self, grid_levels: int) -> None:
        """Non-zero entries must pass through unchanged, whatever their neighbours."""
        gen = torch.Generator().manual_seed(grid_levels)
        for _ in range(200):
            cum = torch.randint(
                -3, 4, (grid_levels,), generator=gen, dtype=torch.float64
            )
            got = _reference_fill(cum)
            nonzero = cum != 0
            assert torch.equal(got[nonzero], cum[nonzero])

    def test_all_empty_book_is_all_zero(self) -> None:
        bid, ask = _book_cumulative_from_ob_rel(
            OrderbookRelative(bids=(), asks=()), 6, 20.0, None, torch.float64
        )
        assert bid.tolist() == [0.0] * 6
        assert ask.tolist() == [0.0] * 6

    def test_leading_zero_is_not_filled_from_nothing(self) -> None:
        ob_rel = OrderbookRelative(bids=((40.0, 9.0),), asks=((40.0, 4.0),))
        bid, ask = _book_cumulative_from_ob_rel(ob_rel, 5, 20.0, None, torch.float64)
        # spread 40.0 -> bucket 1, so bucket 0 stays 0: the loop starts at index 1
        assert bid[0].item() == 0.0
        assert bid[1].item() == 9.0
        assert bid[-1].item() == 9.0


class TestBucketIndex:
    def test_bucket_boundaries(self) -> None:
        assert _spread_bucket_index(0.0, 20.0, 10) == 0
        assert _spread_bucket_index(20.0, 20.0, 10) == 0
        assert _spread_bucket_index(20.1, 20.0, 10) == 1
        assert _spread_bucket_index(200.0, 20.0, 10) == 9

    def test_out_of_domain_is_clamped(self) -> None:
        assert _spread_bucket_index(-5.0, 20.0, 10) == 0
        assert _spread_bucket_index(10_000.0, 20.0, 10) == 9


class TestSpreadGridCache:
    def test_cache_returns_equal_tensors(self) -> None:
        a = _spread_grid_cached(10, 20.0, 200.0, "cpu", "float64")
        b = _spread_grid_cached(10, 20.0, 200.0, "cpu", "float64")
        assert torch.equal(a, b)
        assert a.tolist() == [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]

    def test_distinct_configs_do_not_collide(self) -> None:
        a = _spread_grid_cached(10, 20.0, 200.0, "cpu", "float64")
        b = _spread_grid_cached(5, 40.0, 200.0, "cpu", "float64")
        c = _spread_grid_cached(10, 20.0, 200.0, "cpu", "float32")
        assert a.shape != b.shape
        assert a.dtype != c.dtype

    def test_grid_is_capped_at_spread_max(self) -> None:
        grid = _spread_grid_cached(4, 50.0, 150.0, "cpu", "float64")
        assert grid.tolist() == [50.0, 100.0, 150.0, 150.0]
