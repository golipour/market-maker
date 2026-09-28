"""Regression tests for the stacked two-sided soft-fill chain.

``soft_fill_batched`` evaluates bid and ask with identical formulas and
now runs them as one ``(2, n_samples, grid_levels)`` chain instead of two
separate ``(n_samples, grid_levels)`` chains. That halves the aten call
count, which matters because the fill chain is dispatch-bound rather than
FLOP-bound at production sizes.

These tests pin the stacked chain to the original two-sided semantics
across both queue models, every soft-fill activation, and degenerate books
(empty, single-level, fully-occupied) so the refactor cannot silently
diverge.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from mm.mpc.mpc_tensor_batched import soft_fill_batched, _soft_fill_fraction
from mm.mpc.l_param import LParamConfig
from mm.mpc.state_tensor import effective_depth_ask_tensor, effective_depth_bid_tensor

N_SAMPLES = 8
GRID = 4
SPREAD_MAX = 80.0
SPREAD_STEP = 20.0


def _reference_two_sided(book_cum, vol, our, activation, queue_model, eps=1e-12):
    """The pre-refactor implementation, one independent (N, G) chain."""
    book_prev = F.pad(book_cum[:, :-1], (1, 0))
    book_level = torch.relu(book_cum - book_prev)
    # ``our`` is (G,) when the decode produced a grid-shaped depth and (N, G)
    # when per-sample bounds are in play; match the broadcast the production
    # path performs.
    if our.ndim == 1:
        our = our.unsqueeze(0).expand_as(book_cum)
    depth_prev = F.pad(our[:, :-1], (1, 0))
    marg = torch.relu(our - depth_prev)
    comb_level = book_level + marg
    comb_cum = torch.cumsum(comb_level, dim=1)
    reach_raw = vol - comb_cum
    pos = torch.clamp(reach_raw.max(dim=1, keepdim=True).values, min=eps)
    neg = torch.clamp((-reach_raw).max(dim=1, keepdim=True).values, min=eps)
    reach = torch.where(reach_raw >= 0.0, reach_raw / pos, reach_raw / neg)
    frac = _soft_fill_fraction(reach, 1.0, activation)
    consumed = comb_level * frac
    nxt = (
        F.pad(book_level[:, 1:], (0, 1))
        if queue_model == "back_of_queue"
        else torch.zeros_like(book_level)
    )
    den = comb_level + nxt
    share = torch.where(
        den > eps, marg / torch.clamp(den, min=eps), torch.zeros_like(comb_level)
    )
    fill_level = consumed * share
    consumed_market = torch.minimum(torch.relu(consumed - fill_level), book_level)
    return torch.relu(book_level - consumed_market), fill_level


def _build_case(kind, gen):
    l_param = LParamConfig(
        quote_size_min=1.0, quote_size_max=500.0, spread_max=SPREAD_MAX, n_components=2
    )
    params = torch.randn(2 * (6 * 2 + 3), dtype=torch.float64, generator=gen)
    spread_grid = torch.arange(1, GRID + 1, dtype=torch.float64) * SPREAD_STEP
    mid_price = torch.full((N_SAMPLES, 1), 100.0, dtype=torch.float64)

    if kind == "empty":
        book = torch.zeros(N_SAMPLES, GRID, dtype=torch.float64)
    elif kind == "single_level":
        book = torch.zeros(N_SAMPLES, GRID, dtype=torch.float64)
        book[:, 0] = 25.0
    elif kind == "full":
        book = torch.full((N_SAMPLES, GRID), 40.0, dtype=torch.float64)
    else:
        book = torch.rand(N_SAMPLES, GRID, generator=gen, dtype=torch.float64) * 60.0
    book_cum = torch.cumsum(book, dim=1)
    return dict(
        params=params,
        n_components=2,
        mid_price=mid_price,
        book_cum_bid=book_cum.clone(),
        book_cum_ask=book_cum.clone(),
        sell_vol=torch.rand(N_SAMPLES, 1, generator=gen, dtype=torch.float64) * 120.0,
        buy_vol=torch.rand(N_SAMPLES, 1, generator=gen, dtype=torch.float64) * 120.0,
        spread_grid=spread_grid,
        maker_fee=0.001,
        soft_temp=1.0,
        soft_fill_activation="sigmoid",
        soft_fill_queue_model="back_of_queue",
        bid_quote_size_min=None,
        bid_quote_size_max=None,
        ask_quote_size_min=None,
        ask_quote_size_max=None,
        use_softplus=True,
        l_param=l_param,
    )


def _depths(kwargs):
    depth_kw = dict(
        use_softplus=kwargs["use_softplus"],
        l_param=kwargs["l_param"],
        mid_price=kwargs["mid_price"],
        reservation_r_offset=None,
    )
    bid = effective_depth_bid_tensor(
        kwargs["params"], kwargs["spread_grid"], kwargs["n_components"], **depth_kw
    )
    ask = effective_depth_ask_tensor(
        kwargs["params"], kwargs["spread_grid"], kwargs["n_components"], **depth_kw
    )
    return bid, ask


@pytest.mark.parametrize("kind", ["random", "empty", "single_level", "full"])
@pytest.mark.parametrize("queue_model", ["back_of_queue", "pro_rata"])
@pytest.mark.parametrize("activation", ["sigmoid", "hardsigmoid"])
def test_stacked_matches_two_sided(kind, queue_model, activation):
    gen = torch.Generator().manual_seed(1234)
    kwargs = _build_case(kind, gen)
    kwargs["soft_fill_queue_model"] = queue_model
    kwargs["soft_fill_activation"] = activation

    got = soft_fill_batched(**kwargs)
    depth_bid, depth_ask = _depths(kwargs)

    ob_bid, fill_bid_level = _reference_two_sided(
        kwargs["book_cum_bid"], kwargs["sell_vol"], depth_bid, activation, queue_model
    )
    ob_ask, fill_ask_level = _reference_two_sided(
        kwargs["book_cum_ask"], kwargs["buy_vol"], depth_ask, activation, queue_model
    )

    assert torch.equal(got[0], ob_bid), f"bid book mismatch ({kind})"
    assert torch.equal(got[1], ob_ask), f"ask book mismatch ({kind})"

    grid = kwargs["spread_grid"]
    mid_exp = kwargs["mid_price"].expand(-1, grid.shape[0])
    cash_bid = (fill_bid_level * (mid_exp - grid.unsqueeze(0))).sum(dim=1)
    cash_ask = (fill_ask_level * (mid_exp + grid.unsqueeze(0))).sum(dim=1)
    f = 1.0 - kwargs["maker_fee"]
    assert torch.equal(got[2], cash_ask * f - cash_bid), f"cash mismatch ({kind})"
    assert torch.equal(
        got[3], fill_bid_level.sum(dim=1) * f - fill_ask_level.sum(dim=1)
    ), f"inventory mismatch ({kind})"


def test_stacked_sides_are_independent():
    """Perturbing one side must not move the other.

    A stacked *leading* axis preserves this; stacking on the grid axis
    would leak across the seam.
    """
    gen = torch.Generator().manual_seed(99)
    kwargs = _build_case("random", gen)
    base = soft_fill_batched(**kwargs)

    bumped = dict(kwargs)
    bumped["book_cum_ask"] = kwargs["book_cum_ask"] * 1.5
    got = soft_fill_batched(**bumped)
    assert torch.equal(base[0], got[0]), "bid book changed with ask book"
    assert not torch.equal(base[1], got[1]), "ask book ignored the perturbation"

    bumped = dict(kwargs)
    bumped["book_cum_bid"] = kwargs["book_cum_bid"] * 1.5
    got = soft_fill_batched(**bumped)
    assert torch.equal(base[1], got[1]), "ask book changed with bid book"
    assert not torch.equal(base[0], got[0]), "bid book ignored the perturbation"


def test_stacked_is_differentiable():
    gen = torch.Generator().manual_seed(5)
    kwargs = _build_case("random", gen)
    kwargs["params"] = kwargs["params"].clone().requires_grad_(True)
    out = soft_fill_batched(**kwargs)
    loss = out[0].sum() + out[1].sum() + out[2].sum() + out[3].sum()
    loss.backward()
    assert kwargs["params"].grad is not None
    assert torch.isfinite(kwargs["params"].grad).all()
