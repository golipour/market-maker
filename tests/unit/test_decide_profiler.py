"""Tests for the opt-in decide profiler and the thread-count export.

The profiler must be a no-op when disabled, must capture exactly one decide
when enabled, and must disable itself so a persistent flag cannot tax every
subsequent cycle.

``apply_torch_threads`` must also export the count to the OpenMP-style env
vars: inductor's generated C++ reads ``OMP_NUM_THREADS`` at torch-import time,
which is before the optimizer (and therefore ``apply_torch_threads``) exists.
Without the export, ``set_num_threads`` constrains only the eager path and the
compiled kernel still fans out to the node's full core count.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _clean_profiler_env(monkeypatch: pytest.MonkeyPatch):
    from mm.mpc import decide_profiler

    for var in (
        decide_profiler._ENABLED_ENV,
        decide_profiler._LABEL_ENV,
        decide_profiler._PATH_ENV,
        decide_profiler._MAX_S_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    decide_profiler.reset_for_tests()
    yield
    decide_profiler.reset_for_tests()


def test_disabled_profiler_is_a_noop_that_runs_the_block(monkeypatch):
    from mm.mpc.decide_profiler import maybe_profile_decide, profiling_enabled

    assert profiling_enabled() is False
    calls = []
    with maybe_profile_decide("x"):
        calls.append(1)
    assert calls == [1]


def test_enabled_profiler_writes_trace_and_disables_itself(monkeypatch, tmp_path):
    import torch

    from mm.mpc import decide_profiler

    trace = tmp_path / "trace.json"
    monkeypatch.setenv(decide_profiler._ENABLED_ENV, str(trace))

    with decide_profiler.maybe_profile_decide("unit"):
        torch.mm(torch.ones(64, 64), torch.ones(64, 64))

    assert trace.exists(), "trace file should be written"
    # A persistent flag must not profile every subsequent decide.
    assert decide_profiler.profiling_enabled() is False
    assert decide_profiler._ENABLED_ENV not in __import__("os").environ


def test_enabled_profiler_writes_readable_summary(monkeypatch, tmp_path):
    import torch

    from mm.mpc import decide_profiler

    trace = tmp_path / "trace.json"
    monkeypatch.setenv(decide_profiler._ENABLED_ENV, str(trace))

    with decide_profiler.maybe_profile_decide("unit"):
        torch.mm(torch.ones(32, 32), torch.ones(32, 32))

    summary = tmp_path / "trace_summary.txt"
    assert summary.exists()
    text = summary.read_text()
    assert "unit" in text
    assert "wall:" in text


def test_numeric_flag_uses_default_path(monkeypatch, tmp_path):
    import torch

    from mm.mpc import decide_profiler

    target = tmp_path / "from_path.json"
    monkeypatch.setenv(decide_profiler._ENABLED_ENV, "1")
    monkeypatch.setenv(decide_profiler._PATH_ENV, str(target))

    with decide_profiler.maybe_profile_decide("unit"):
        torch.mm(torch.ones(8, 8), torch.ones(8, 8))

    assert target.exists()


def test_profiling_runs_the_block_exactly_once(monkeypatch, tmp_path):
    import torch

    from mm.mpc import decide_profiler

    monkeypatch.setenv(decide_profiler._ENABLED_ENV, str(tmp_path / "t.json"))
    calls = []
    for _ in range(3):
        with decide_profiler.maybe_profile_decide():
            calls.append(1)
            torch.mm(torch.ones(4, 4), torch.ones(4, 4))
    # The block must always execute; only the profiling is one-shot.
    assert calls == [1, 1, 1]


def test_trace_is_valid_json_array_or_object(monkeypatch, tmp_path):
    import torch

    from mm.mpc import decide_profiler

    trace = tmp_path / "t.json"
    monkeypatch.setenv(decide_profiler._ENABLED_ENV, str(trace))
    with decide_profiler.maybe_profile_decide():
        torch.mm(torch.ones(4, 4), torch.ones(4, 4))

    parsed = json.loads(trace.read_text())
    assert isinstance(parsed, (dict, list))


def test_apply_torch_threads_exports_omp_env(monkeypatch):
    """The count must reach the OpenMP/inductor runtimes, not just torch."""
    import os

    from mm.mpc.torch_device import apply_torch_threads

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)

    applied = apply_torch_threads(3)

    assert applied == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["TORCHINDUCTOR_NUM_THREADS"] == "3"
    import torch

    assert torch.get_num_threads() == 3


def test_apply_torch_threads_respects_preexisting_env(monkeypatch):
    """An operator-provided value must not be overwritten."""
    import os

    from mm.mpc.torch_device import apply_torch_threads

    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_NUM_THREADS", raising=False)

    apply_torch_threads(2)

    assert os.environ["OMP_NUM_THREADS"] == "7"
    # Vars that were unset are still filled in from the resolved count.
    assert os.environ["TORCHINDUCTOR_NUM_THREADS"] == "2"


def test_apply_torch_threads_survives_already_initialized_pool(monkeypatch):
    """A pre-warmed pool must not make thread setup raise."""
    import torch

    from mm.mpc.torch_device import apply_torch_threads

    torch.set_num_threads(2)  # force the interop pool to exist
    assert apply_torch_threads(2) == 2