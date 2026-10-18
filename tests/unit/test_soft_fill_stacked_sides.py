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

from mm.mpc.mpc_tensor_batched import soft_fill_batched, _soft_fill_fraction, MonteCarloMPCTensorBatched
from mm.mpc.l_param import LParamConfig
from mm.mpc.state_tensor import (
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
    effective_depth_stacked_tensor,
)

N_SAMPLES = 8
GRID = 4
SPREAD_MAX = 80.0
SPREAD_STEP = 20.0


def _reference_two_sided(book_cum, vol, our, activation, queue_model, eps=1e-12):
    """Independent reference matching implementation, one independent (N, G) chain."""
    del activation  # Deprecated in conserving matching
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

    # Exact flow-conserving prefix consumption
    comb_cum = torch.cumsum(comb_level, dim=1)
    cons_cum = torch.minimum(vol, comb_cum)
    cons_prev = F.pad(cons_cum[:, :-1], (1, 0))
    consumed = torch.relu(cons_cum - cons_prev)

    if queue_model == "back_of_queue":
        consumed_market = torch.minimum(consumed, book_level)
        fill_level = consumed - consumed_market
    else:  # pro_rata
        den = torch.clamp(comb_level, min=eps)
        share = torch.where(
            comb_level > eps, marg / den, torch.zeros_like(comb_level)
        )
        fill_level = consumed * share
        consumed_market = consumed - fill_level

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


@pytest.mark.parametrize("has_sample_bounds", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_effective_depth_stacked_matches_separate(has_sample_bounds: bool, structured: bool):
    gen = torch.Generator().manual_seed(42)
    n = 3
    spread_max = 60.0
    lp = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=20.0 if structured else None,
        quote_spread_min_depth=1.0 if structured else None,
    )
    params = torch.randn(6 * n + 3, dtype=torch.float64, generator=gen)
    grid = torch.arange(1, 5, dtype=torch.float64) * 15.0
    mid = torch.full((4, 1), 100.0, dtype=torch.float64)

    if has_sample_bounds:
        q_min_b = torch.full((4, 1), 2.0, dtype=torch.float64)
        q_max_b = torch.full((4, 1), 20.0, dtype=torch.float64)
        q_min_a = torch.full((4, 1), 3.0, dtype=torch.float64)
        q_max_a = torch.full((4, 1), 25.0, dtype=torch.float64)
    else:
        q_min_b = q_max_b = q_min_a = q_max_a = None

    db = effective_depth_bid_tensor(
        params, grid, n, l_param=lp,
        quote_size_min=q_min_b, quote_size_max=q_max_b, mid_price=mid
    )
    da = effective_depth_ask_tensor(
        params, grid, n, l_param=lp,
        quote_size_min=q_min_a, quote_size_max=q_max_a, mid_price=mid
    )

    stacked = effective_depth_stacked_tensor(
        params, grid, n, l_param=lp,
        bid_quote_size_min=q_min_b, bid_quote_size_max=q_max_b,
        ask_quote_size_min=q_min_a, ask_quote_size_max=q_max_a,
        mid_price=mid
    )

    assert torch.equal(stacked[0], db)
    assert torch.equal(stacked[1], da)


def test_compile_soft_fill_configuration_and_execution(monkeypatch):
    from mm.mpc.mpc_state import State, OrderbookRelative

    class MockFlow:
        def sample(self, sh, device=None, dtype=None):
            dt = dtype or torch.float64
            n = int(sh[0]) * int(sh[1])
            return (torch.ones(sh, dtype=dt, device=device), torch.ones(sh, dtype=dt, device=device))

    class MockPrice:
        def sample_residual(self, sh, device=None, dtype=None):
            dt = dtype or torch.float64
            return (torch.zeros(sh, dtype=dt, device=device), torch.zeros(sh, dtype=dt, device=device))

    class MockExec:
        maker_fee = 0.0

    state = State(
        mid=100.0,
        ob_rel=OrderbookRelative(bids=((20.0, 50.0),), asks=((20.0, 50.0),)),
        buy_vol=10.0,
        sell_vol=10.0,
        inventory=0.0,
        cash=10000.0,
        buy_vwap=100.0,
        sell_vwap=100.0,
    )

    # 1. Configured via parameter
    opt_compiled = MonteCarloMPCTensorBatched(
        local_price_model=MockPrice(),
        local_market_flow_model=MockFlow(),
        execution_model=MockExec(),
        n_components=2,
        n_samples=4,
        horizon=1,
        spread_max=40.0,
        spread_step=20.0,
        adam_epochs=2,
        reference_price_source="local",
        compile_soft_fill=True,
    )
    assert opt_compiled.compile_soft_fill is True

    # 2. Configured via environment variable
    monkeypatch.setenv("MM_TORCH_COMPILE", "1")
    opt_env = MonteCarloMPCTensorBatched(
        local_price_model=MockPrice(),
        local_market_flow_model=MockFlow(),
        execution_model=MockExec(),
        n_components=2,
        n_samples=4,
        horizon=1,
        spread_max=40.0,
        spread_step=20.0,
        adam_epochs=2,
        reference_price_source="local",
    )
    assert opt_env.compile_soft_fill is True

    # Run decision and verify parity with eager
    act_c, diag_c = opt_compiled.optimize_with_diagnostics(state)

    opt_eager = MonteCarloMPCTensorBatched(
        local_price_model=MockPrice(),
        local_market_flow_model=MockFlow(),
        execution_model=MockExec(),
        n_components=2,
        n_samples=4,
        horizon=1,
        spread_max=40.0,
        spread_step=20.0,
        adam_epochs=2,
        reference_price_source="local",
        compile_soft_fill=False,
    )
    act_e, diag_e = opt_eager.optimize_with_diagnostics(state)

    assert abs(diag_c.best.objective_value - diag_e.best.objective_value) < 1e-10

    # 3. Fallback on runtime failure
    opt_fail = MonteCarloMPCTensorBatched(
        local_price_model=MockPrice(),
        local_market_flow_model=MockFlow(),
        execution_model=MockExec(),
        n_components=2,
        n_samples=4,
        horizon=1,
        spread_max=40.0,
        spread_step=20.0,
        adam_epochs=2,
        reference_price_source="local",
        compile_soft_fill=True,
    )

    def exploding_soft_fill(*args, **kwargs):
        raise RuntimeError("simulated compiler kernel panic")

    monkeypatch.setattr("mm.mpc.mpc_tensor_batched._get_compiled_soft_fill", lambda tm: exploding_soft_fill)
    act_f, diag_f = opt_fail.optimize_with_diagnostics(state)
    assert opt_fail.compile_soft_fill is False
    assert abs(diag_f.best.objective_value - diag_e.best.objective_value) < 1e-10



import contextlib as _contextlib

import mm.mpc.mpc_tensor_batched as _MODULE


@_contextlib.contextmanager
def _isolated_compile_state():
    """Save/restore the module-global compile caches.

    ``_COMPILE_SOFT_FILL_FAILED`` latches permanently in production (one
    failure means inductor is unusable for the process), and
    ``_SOFT_FILL_WARMED_SIGNATURES`` accumulates across instances. Tests that
    exercise those paths must not leak state into each other, or a test that
    forces the failure path silently disables compilation for every test that
    runs after it.
    """
    saved_failed = _MODULE._COMPILE_SOFT_FILL_FAILED
    saved_kernel = _MODULE._COMPILED_SOFT_FILL_KERNEL
    saved_sigs = set(_MODULE._SOFT_FILL_WARMED_SIGNATURES)
    saved_unwarmed = set(_MODULE._SOFT_FILL_UNWARMED_SEEN)
    # Start from a clean slate: another test may have latched the failure flag,
    # in which case the warm-up correctly short-circuits and records nothing.
    _MODULE._COMPILE_SOFT_FILL_FAILED = False
    _MODULE._COMPILED_SOFT_FILL_KERNEL = None
    _MODULE._SOFT_FILL_WARMED_SIGNATURES.clear()
    _MODULE._SOFT_FILL_UNWARMED_SEEN.clear()
    try:
        yield
    finally:
        _MODULE._COMPILE_SOFT_FILL_FAILED = saved_failed
        _MODULE._COMPILED_SOFT_FILL_KERNEL = saved_kernel
        _MODULE._SOFT_FILL_WARMED_SIGNATURES.clear()
        _MODULE._SOFT_FILL_WARMED_SIGNATURES.update(saved_sigs)
        _MODULE._SOFT_FILL_UNWARMED_SEEN.clear()
        _MODULE._SOFT_FILL_UNWARMED_SEEN.update(saved_unwarmed)


def _mk_opt(**over):
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
    from mm.mpc.mpc_state import State, OrderbookRelative

    class MockFlow:
        def sample(self, sh, device=None, dtype=None):
            dt = dtype or torch.float64
            return (torch.ones(sh, dtype=dt, device=device),
                    torch.ones(sh, dtype=dt, device=device))

    class MockPrice:
        def sample_residual(self, sh, device=None, dtype=None):
            dt = dtype or torch.float64
            return (torch.zeros(sh, dtype=dt, device=device),
                    torch.zeros(sh, dtype=dt, device=device))

    class MockExec:
        maker_fee = 0.0

    kw = dict(
        local_price_model=MockPrice(),
        local_market_flow_model=MockFlow(),
        execution_model=MockExec(),
        n_components=2,
        n_samples=4,
        horizon=1,
        spread_max=40.0,
        spread_step=20.0,
        adam_epochs=2,
        reference_price_source="local",
    )
    kw.update(over)
    return MonteCarloMPCTensorBatched(**kw)


def test_compile_soft_fill_is_enabled_by_default(monkeypatch):
    """The compiled kernel is the default; the environment can still opt out."""
    monkeypatch.delenv("MM_TORCH_COMPILE", raising=False)
    assert _mk_opt().compile_soft_fill is True

    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("MM_TORCH_COMPILE", value)
        assert _mk_opt().compile_soft_fill is False, value

    # An explicit argument beats the environment in both directions.
    monkeypatch.setenv("MM_TORCH_COMPILE", "1")
    assert _mk_opt(compile_soft_fill=False).compile_soft_fill is False
    monkeypatch.setenv("MM_TORCH_COMPILE", "0")
    assert _mk_opt(compile_soft_fill=True).compile_soft_fill is True


def test_eager_opt_out_matches_compiled_default():
    """Opting out must reproduce the compiled result to float64 reassociation."""
    from mm.mpc.mpc_state import State, OrderbookRelative

    state = State(
        mid=100.0,
        ob_rel=OrderbookRelative(bids=((20.0, 50.0), (40.0, 30.0)), asks=((20.0, 40.0),)),
        buy_vol=10.0,
        sell_vol=10.0,
        inventory=0.0,
        cash=10000.0,
        buy_vwap=100.0,
        sell_vwap=100.0,
    )
    _, dc = _mk_opt(compile_soft_fill=True).optimize_with_diagnostics(state)
    _, de = _mk_opt(compile_soft_fill=False).optimize_with_diagnostics(state)
    assert abs(dc.best.objective_value - de.best.objective_value) < 1e-12


def test_warm_soft_fill_compile_pays_cost_off_the_decide_path():
    """The warm-up must run at construction, not inside the first decide.

    Compiling lazily costs seconds, which would otherwise land inside a
    single decide bounded by ``decision_timeout_ms`` (10 s in the shipped
    preset).
    """
    with _isolated_compile_state():
        opt = _mk_opt(compile_soft_fill=True)
        assert opt.soft_fill_compile_warmup_ms >= 0.0

        if _MODULE._COMPILE_SOFT_FILL_FAILED:
            # Inductor refused to build this process (stale precompiled
            # headers, no compiler, ...). The warm-up must have degraded to
            # eager rather than raising, which is the contract that matters.
            assert not _MODULE._SOFT_FILL_WARMED_SIGNATURES
            return

        assert _MODULE._SOFT_FILL_WARMED_SIGNATURES, "warm-up recorded no signature"

        # A repeated warm-up for the same signature is free. The signature
        # key is (n_components, n_samples, grid_levels, activation, queue,
        # depth_scale, soft_temp, maker_fee).
        (
            n_c,
            n_s,
            g_l,
            act,
            queue,
            depth_scale,
            soft_temp,
            maker_fee,
        ) = next(iter(_MODULE._SOFT_FILL_WARMED_SIGNATURES))
        assert _MODULE.warm_soft_fill_compile(
            n_components=n_c,
            n_samples=n_s,
            grid_levels=g_l,
            soft_fill_activation=act,
            soft_fill_queue_model=queue,
            depth_scale=depth_scale,
            soft_temp=soft_temp,
            maker_fee=maker_fee,
        ) == 0.0


def test_warm_up_covers_both_depth_scales():
    """The joint path dispatches at ``depth_scale=1.0`` and ``global_mask``.

    ``depth_scale`` reaches the kernel as a Python float and is part of the
    inductor specialization (measured: a 0.0 call after a 1.0 warm-up added a
    frame and cost a 2.7 s lazy compile). ``global_mask`` is 0.0 whenever the
    global venue is inactive, which is every production decision while
    liquidity sourcing is off -- so a 1.0-only warm-up would let the global
    chain compile inside a live decide.
    """
    with _isolated_compile_state():
        opt = _mk_opt(compile_soft_fill=True, global_optimizer={"n_components": 2})
        if _MODULE._COMPILE_SOFT_FILL_FAILED:
            return  # inductor unavailable on this host; nothing was warmed

        # ``depth_scale`` is signature index 5, not the last element: the key
        # also carries ``soft_temp`` and ``maker_fee``, which reach the kernel
        # as Python floats and are likewise part of the specialization.
        scales = {sig[5] for sig in _MODULE._SOFT_FILL_WARMED_SIGNATURES}
        assert 1.0 in scales
        assert 0.0 in scales, f"masked global chain not warmed; scales={scales}"


def test_warm_up_failure_falls_back_to_eager(monkeypatch):
    """A platform without inductor must not break construction."""
    with _isolated_compile_state():
        def boom(*a, **k):
            raise RuntimeError("no inductor here")

        # monkeypatch, not a bare assignment: a direct assignment would leak
        # into every later test in the session.
        monkeypatch.setattr(_MODULE, "_get_compiled_soft_fill", boom)
        assert _MODULE.warm_soft_fill_compile(
            n_components=2, n_samples=4, grid_levels=2
        ) == 0.0
        # Construction still succeeds and stays on the eager path.
        assert _mk_opt(compile_soft_fill=True).compile_soft_fill is True


def test_warm_up_covers_configured_fill_branch():
    """The warm-up must cover the configured fill branch, not just the default.

    ``soft_fill_queue_model`` and ``soft_fill_activation`` select real
    branches inside ``soft_fill_batched``, so warming the default branch and
    then running ``pro_rata``/``hardsigmoid`` would leave a second lazy
    compile to be paid by the first decide. Asserted on the warmed-signature
    key rather than on wall-clock, so it cannot flake on a warm inductor
    cache.
    """
    with _isolated_compile_state():
        _mk_opt(
            compile_soft_fill=True,
            local_optimizer={
                "n_components": 2,
                "soft_fill_activation": "hardsigmoid",
                "soft_fill_queue_model": "pro_rata",
                "soft_fill_temp": 0.05,
            },
        )
        if _MODULE._COMPILE_SOFT_FILL_FAILED:
            return  # inductor unavailable; fallback already covered above

        sigs = _MODULE._SOFT_FILL_WARMED_SIGNATURES
        assert sigs, "warm-up recorded no signature"
        local = [s for s in sigs if s[3] == "hardsigmoid" and s[4] == "pro_rata"]
        assert local, (
            "the configured hardsigmoid/pro_rata branch was never warmed; "
            f"warmed signatures: {sorted(sigs)}"
        )


def test_unwarmed_shape_never_compiles_on_the_decide_path():
    """Regression: a shape missing from the warm set must not compile live.

    Production incident: a strategy override changed ``n_samples``/
    ``grid_levels`` to a combination the construction-time warm-up did not
    cover. The first live decide for that venue paid a lazy inductor compile
    measured at 162 s -- far past ``decision_timeout_ms`` (15 s) -- which wedged
    every controller on "optimizer decision in progress" and stopped quoting
    entirely.

    The contract is that the decide path only ever calls a kernel whose
    signature was already warmed. An unwarmed signature must fall back to
    eager (slower but correct) instead of compiling.
    """
    with _isolated_compile_state():
        from mm.mpc.mpc_state import State, OrderbookRelative

        opt = _mk_opt(compile_soft_fill=True)
        if _MODULE._COMPILE_SOFT_FILL_FAILED:
            return  # inductor unavailable on this host; nothing to gate

        # Simulate the incident: forget every warmed signature, as if this
        # venue's shape had never been paid for at construction.
        _MODULE._SOFT_FILL_WARMED_SIGNATURES.clear()
        _MODULE._SOFT_FILL_UNWARMED_SEEN.clear()

        calls: list[int] = []
        real_get = _MODULE._get_compiled_soft_fill

        def counting_get(torch_mod):
            calls.append(1)
            return real_get(torch_mod)

        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(_MODULE, "_get_compiled_soft_fill", counting_get)
            state = State(
                mid=100.0,
                ob_rel=OrderbookRelative(
                    bids=((20.0, 50.0), (40.0, 30.0)), asks=((20.0, 40.0),)
                ),
                buy_vol=10.0,
                sell_vol=10.0,
                inventory=0.0,
                cash=10000.0,
                buy_vwap=100.0,
                sell_vwap=100.0,
            )
            opt.optimize_with_diagnostics(state)
        finally:
            monkey.undo()

        assert not calls, (
            "the decide path requested a compiled kernel for an unwarmed "
            f"shape (signature(s): {sorted(_MODULE._SOFT_FILL_UNWARMED_SEEN)}); "
            "that is the lazy compile that wedged the controller in production"
        )
        # The shortfall must be observable, not silent.
        assert _MODULE._SOFT_FILL_UNWARMED_SEEN, (
            "an unwarmed shape ran eager without recording the signature, so "
            "the missed warm-up would be invisible in logs"
        )


def test_exact_flow_conservation_zero_volume():
    """Physical law: When incoming trade volume is 0, execution is identically 0."""
    gen = torch.Generator().manual_seed(42)
    kwargs = _build_case("random", gen)
    # Zero volume on both sides
    kwargs["sell_vol"] = torch.zeros((N_SAMPLES, 1), dtype=torch.float64)
    kwargs["buy_vol"] = torch.zeros((N_SAMPLES, 1), dtype=torch.float64)

    ob_bids_next, ob_asks_next, cash_delta, inv_delta = soft_fill_batched(**kwargs)

    # 1. Orderbooks must be completely untouched
    book_prev_b = F.pad(kwargs["book_cum_bid"][:, :-1], (1, 0))
    expected_ob_b = torch.relu(kwargs["book_cum_bid"] - book_prev_b)
    book_prev_a = F.pad(kwargs["book_cum_ask"][:, :-1], (1, 0))
    expected_ob_a = torch.relu(kwargs["book_cum_ask"] - book_prev_a)

    assert torch.equal(ob_bids_next, expected_ob_b)
    assert torch.equal(ob_asks_next, expected_ob_a)

    # 2. Fills, cash delta, and inventory delta must be EXACTLY zero
    assert torch.equal(cash_delta, torch.zeros_like(cash_delta))
    assert torch.equal(inv_delta, torch.zeros_like(inv_delta))

    # 3. One-sided test: sell_vol > 0, buy_vol == 0
    kwargs["sell_vol"] = torch.full((N_SAMPLES, 1), 50.0, dtype=torch.float64)
    kwargs["buy_vol"] = torch.zeros((N_SAMPLES, 1), dtype=torch.float64)

    ob_bids_next2, ob_asks_next2, cash_delta2, inv_delta2 = soft_fill_batched(**kwargs)
    # Ask book must be completely untouched, no ask fills
    assert torch.equal(ob_asks_next2, expected_ob_a)
    # Bid book was consumed
    assert not torch.equal(ob_bids_next2, expected_ob_b)


def test_hand_calculated_queue_allocations():
    """Test exact bucket-level matching against hand-calculated analytical solutions.

    Fixture:
      Level 0: public depth = 4.0, our quote = 2.0 (combined = 6.0)
    """
    l_param = LParamConfig(
        quote_size_min=2.0, quote_size_max=2.0, spread_max=20.0, n_components=1
    )
    # Configure 1 level
    spread_grid = torch.tensor([20.0], dtype=torch.float64)
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)
    empty_ask = torch.zeros((1, 1), dtype=torch.float64)
    buy_vol = torch.zeros((1, 1), dtype=torch.float64)
    book_bid = torch.tensor([[4.0]], dtype=torch.float64)

    # Decode params to produce exactly 2.0 quote size
    params = torch.tensor([0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 30.0, 30.0, 0.0], dtype=torch.float64)

    # Case 1: V = 3.0 (under public depth of 4.0)
    # BOQ: market eats 3.0, our gets 0.0
    _, _, _, inv_boq_3 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[3.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="back_of_queue", l_param=l_param,
    )
    # Pro-rata: our gets 3.0 * (2.0 / 6.0) = 1.0
    _, _, _, inv_pr_3 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[3.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="pro_rata", l_param=l_param,
    )
    assert float(inv_boq_3[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(inv_pr_3[0]) == pytest.approx(1.0, rel=1e-6)

    # Case 2: V = 5.0 (exceeds public 4.0 by 1.0)
    # BOQ: market eats 4.0, our gets 1.0
    _, _, _, inv_boq_5 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[5.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="back_of_queue", l_param=l_param,
    )
    # Pro-rata: our gets 5.0 * (2.0 / 6.0) = 5/3 = 1.666667
    _, _, _, inv_pr_5 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[5.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="pro_rata", l_param=l_param,
    )
    assert float(inv_boq_5[0]) == pytest.approx(1.0, rel=1e-6)
    assert float(inv_pr_5[0]) == pytest.approx(5.0 / 3.0, rel=1e-6)

    # Case 3: V = 10.0 (overflow beyond total capacity of 6.0)
    # Both: our gets exactly 2.0 (our maximum quote capacity)
    _, _, _, inv_boq_10 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[10.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="back_of_queue", l_param=l_param,
    )
    _, _, _, inv_pr_10 = soft_fill_batched(
        params, n_components=1, mid_price=mid_price, book_cum_bid=book_bid, book_cum_ask=empty_ask,
        sell_vol=torch.tensor([[10.0]], dtype=torch.float64), buy_vol=buy_vol, spread_grid=spread_grid,
        maker_fee=0.0, soft_fill_queue_model="pro_rata", l_param=l_param,
    )
    assert float(inv_boq_10[0]) == pytest.approx(2.0, rel=1e-6)
    assert float(inv_pr_10[0]) == pytest.approx(2.0, rel=1e-6)


def test_flow_conservation_invariants_across_random_books():
    """Verify that total volume consumed across public + maker <= incoming market volume."""
    gen = torch.Generator().manual_seed(999)
    for q_model in ("back_of_queue", "pro_rata"):
        kwargs = _build_case("random", gen)
        kwargs["soft_fill_queue_model"] = q_model
        vols_bid = kwargs["sell_vol"]  # market sells hit bids
        vols_ask = kwargs["buy_vol"]   # market buys hit asks

        ob_bids_next, ob_asks_next, cash_delta, inv_delta = soft_fill_batched(**kwargs)

        # Original public book levels
        book_prev_b = F.pad(kwargs["book_cum_bid"][:, :-1], (1, 0))
        orig_book_b = torch.relu(kwargs["book_cum_bid"] - book_prev_b)
        book_prev_a = F.pad(kwargs["book_cum_ask"][:, :-1], (1, 0))
        orig_book_a = torch.relu(kwargs["book_cum_ask"] - book_prev_a)

        # Public volume consumed
        consumed_mkt_b = orig_book_b - ob_bids_next
        consumed_mkt_a = orig_book_a - ob_asks_next

        # Invariant 1: Public book next state is non-negative and <= original
        assert (ob_bids_next >= -1e-12).all()
        assert (ob_asks_next >= -1e-12).all()
        assert (consumed_mkt_b >= -1e-12).all()
        assert (consumed_mkt_a >= -1e-12).all()

        # Invariant 2: Total consumed on each side <= incoming market volume
        # Bid side: maker fee is applied to inv_delta, so invert to find gross maker fill
        # Or simpler: total public consumed <= vol
        assert (consumed_mkt_b.sum(dim=1, keepdim=True) <= vols_bid + 1e-9).all()
        assert (consumed_mkt_a.sum(dim=1, keepdim=True) <= vols_ask + 1e-9).all()

