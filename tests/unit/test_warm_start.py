from __future__ import annotations

import math

import numpy as np
import pytest

from mm.mpc.warm_start import (
    WarmStartParams,
    blend_action_flat,
    prepare_warm_start_initial_flat,
    resolve_warm_start_one_action,
)


def test_warm_start_from_bool_true_matches_legacy() -> None:
    params = WarmStartParams.from_mapping(True)
    assert params.enabled is True
    assert params.preserve_optimizer_state is True
    assert params.alpha == pytest.approx(1.0)


def test_warm_start_from_bool_false_matches_cold_start() -> None:
    params = WarmStartParams.from_mapping(False)
    assert params.enabled is False
    assert params.alpha == pytest.approx(0.0)


def test_blend_alpha_extremes() -> None:
    warm = np.array([1.0, 2.0], dtype=np.float64)
    default = np.array([3.0, 4.0], dtype=np.float64)
    assert np.allclose(blend_action_flat(warm, default, 0.0), default)
    assert np.allclose(blend_action_flat(warm, default, 1.0), warm)
    assert np.allclose(blend_action_flat(warm, default, 0.5), np.array([2.0, 3.0]))


def test_fixed_alpha_blend() -> None:
    params = WarmStartParams(enabled=True, alpha=0.8)
    warm = np.ones(3, dtype=np.float64)
    default = np.zeros(3, dtype=np.float64)
    one, meta = resolve_warm_start_one_action(
        params=params,
        u_default=default,
        u_warm=warm,
    )
    assert np.allclose(one, np.full(3, 0.8))
    assert meta["warm_start_alpha"] == pytest.approx(0.8)


def test_adaptive_alpha_prefers_default_when_warm_gradient_is_flat() -> None:
    params = WarmStartParams(enabled=True, alpha_mode="adaptive", alpha_max=0.95)
    alpha = params.resolved_alpha(grad_norm_warm=0.01, grad_norm_default=1.0)
    assert alpha == pytest.approx(0.01 / 1.01, rel=1e-6)
    alpha_high = params.resolved_alpha(grad_norm_warm=10.0, grad_norm_default=0.01)
    assert alpha_high == pytest.approx(0.95)


def test_prepare_warm_start_tiles_horizon() -> None:
    params = WarmStartParams(enabled=True, alpha=1.0)
    u_default = np.array([0.0, 1.0], dtype=np.float64)
    u_warm = np.array([2.0, 3.0], dtype=np.float64)
    flat, meta = prepare_warm_start_initial_flat(
        params=params,
        u_default=u_default,
        u_warm=u_warm,
        horizon=3,
    )
    assert flat.shape == (6,)
    assert np.allclose(flat[:2], u_warm)
    assert meta["warm_start_alpha"] == pytest.approx(1.0)


def test_adaptive_probe_callback_used() -> None:
    params = WarmStartParams(enabled=True, alpha_mode="adaptive")
    calls: list[str] = []

    def probe(arr: np.ndarray) -> float:
        calls.append("warm" if arr[0] > 0 else "default")
        return 1.0 if arr[0] > 0 else 0.1

    prepare_warm_start_initial_flat(
        params=params,
        u_default=np.array([0.0], dtype=np.float64),
        u_warm=np.array([1.0], dtype=np.float64),
        horizon=1,
        grad_norm_probe=probe,
    )
    assert calls == ["warm", "default"]
