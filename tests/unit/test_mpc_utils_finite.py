"""Tests for the centralised finite-arithmetic primitives in ``mm.mpc_utils``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm.mpc_utils import (
    exp_clamped,
    log_clamped,
    log1p_clamped,
    logaddexp,
    logsumexp_pair,
    logsumexp,
    expit,
    sigmoid,
    sigmoid_diff,
    softplus,
    softmax,
    log_softmax,
    log_poisson_pmf,
    log_normal_pdf,
    log_student_t_pdf,
    safe_divide,
    safe_sqrt,
    finite_or,
    is_finite,
)


# ---------------------------------------------------------------------------
# exp_clamped
# ---------------------------------------------------------------------------


class TestExpClamped:
    def test_matches_math_exp_for_ordinary_inputs(self) -> None:
        for v in [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0]:
            assert exp_clamped(v) == pytest.approx(math.exp(v), rel=1e-9)

    def test_returns_inf_for_huge_input(self) -> None:
        assert exp_clamped(1e6) == math.inf

    def test_returns_zero_for_very_negative_input(self) -> None:
        assert exp_clamped(-1e6) == 0.0

    def test_finite_at_boundary(self) -> None:
        # 700 is just inside the clamp; exp(700) is finite.
        v = exp_clamped(700.0)
        assert math.isfinite(v) and v > 0.0

    def test_finite_just_below_boundary(self) -> None:
        # At 700.0 the value is finite; just past 700 both the scalar
        # and numpy branches saturate to +inf.
        assert math.isfinite(exp_clamped(700.0)) and exp_clamped(700.0) > 0.0
        assert exp_clamped(701.0) == math.inf
        assert exp_clamped(np.array([701.0]))[0] == math.inf

    def test_numpy_path_does_not_overflow(self) -> None:
        arr = np.array([-1e6, -700.0, 0.0, 700.0, 1e6])
        out = exp_clamped(arr)
        # Below clamp -> 0.0, above clamp -> +inf, within -> finite
        assert out[0] == 0.0
        assert np.all(np.isfinite(out[1:4]))
        assert out[4] == math.inf

    def test_numpy_path_does_not_raise(self) -> None:
        # Without the errstate context, np.exp(1000) raises in some numpy
        # builds; with the clamp it should just return a sentinel.
        arr = np.array([1000.0, -1000.0])
        with np.errstate(all="ignore"):
            out = exp_clamped(arr)
        assert out[0] == math.inf
        assert out[1] == 0.0


# ---------------------------------------------------------------------------
# log_clamped
# ---------------------------------------------------------------------------


class TestLogClamped:
    def test_matches_math_log_for_positive_input(self) -> None:
        for v in [0.5, 1.0, 2.0, 10.0, 100.0]:
            assert log_clamped(v) == pytest.approx(math.log(v), rel=1e-9)

    def test_returns_neg_inf_for_zero(self) -> None:
        assert log_clamped(0.0) == -math.inf

    def test_returns_neg_inf_for_negative(self) -> None:
        assert log_clamped(-3.0) == -math.inf

    def test_returns_neg_inf_for_none(self) -> None:
        assert log_clamped(None) == -math.inf

    def test_finite_for_tiny_positive(self) -> None:
        # 1e-300 is the documented floor; log should not underflow.
        v = log_clamped(1e-300)
        assert math.isfinite(v)

    def test_numpy_path_handles_zero_and_negative(self) -> None:
        arr = np.array([-3.0, 0.0, 1.0, 100.0])
        out = log_clamped(arr)
        assert out[0] == -math.inf
        assert out[1] == -math.inf
        assert math.isfinite(out[2]) and math.isfinite(out[3])


# ---------------------------------------------------------------------------
# log1p_clamped
# ---------------------------------------------------------------------------


class TestLog1pClamped:
    def test_matches_math_log1p_for_safe_inputs(self) -> None:
        for v in [-0.5, 0.0, 0.5, 1.0, 5.0, 100.0]:
            assert log1p_clamped(v) == pytest.approx(math.log1p(v), rel=1e-9)

    def test_returns_neg_inf_at_minus_one(self) -> None:
        assert log1p_clamped(-1.0) == -math.inf

    def test_returns_neg_inf_below_minus_one(self) -> None:
        assert log1p_clamped(-5.0) == -math.inf

    def test_numpy_path(self) -> None:
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 1000.0])
        out = log1p_clamped(arr)
        assert out[0] == -math.inf
        assert out[1] == -math.inf
        assert math.isfinite(out[2])
        assert math.isfinite(out[-1])


# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------


class TestSoftplus:
    def test_matches_log1p_exp_for_small_inputs(self) -> None:
        for v in [-5.0, -1.0, 0.0, 0.5, 1.0, 5.0]:
            assert softplus(v) == pytest.approx(math.log1p(math.exp(v)), rel=1e-9)

    def test_large_positive_returns_itself(self) -> None:
        # 50 is well past the crossover; softplus(x) == x
        assert softplus(50.0) == pytest.approx(50.0, abs=1e-9)

    def test_large_negative_returns_exp(self) -> None:
        # -50: log(1 + exp(-50)) ~ exp(-50)
        assert softplus(-50.0) == pytest.approx(math.exp(-50.0), rel=1e-9)

    def test_never_raises_on_extremes(self) -> None:
        # The whole point of this module: these calls must not raise.
        softplus(1e6)
        softplus(-1e6)
        softplus(700.0)
        softplus(-700.0)

    def test_numpy_path_never_raises(self) -> None:
        arr = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
        out = softplus(arr)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# expit / sigmoid
# ---------------------------------------------------------------------------


class TestExpit:
    def test_matches_math_for_ordinary_inputs(self) -> None:
        for v in [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0]:
            assert expit(v) == pytest.approx(1.0 / (1.0 + math.exp(-v)), rel=1e-9)

    def test_saturates_in_both_tails(self) -> None:
        assert expit(1000.0) == 1.0
        assert expit(-1000.0) == 0.0

    def test_never_raises(self) -> None:
        expit(1e9)
        expit(-1e9)

    def test_sigmoid_alias(self) -> None:
        assert sigmoid(0.0) == expit(0.0)
        assert sigmoid(0.5) == expit(0.5)

    def test_sigmoid_diff_at_zero(self) -> None:
        # sigmoid_diff(0, center=0) is sigmoid(0) - sigmoid(0) == 0
        assert sigmoid_diff(0.0, center=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_sigmoid_diff_antisymmetric(self) -> None:
        # sigmoid_diff(x, center) is the logistic derivative at the point
        # (x) with respect to the +/-center shift. The function is
        # anti-symmetric in (x, center) jointly:
        # sigmoid_diff(x, c) = -sigmoid_diff(-x, -c).
        assert sigmoid_diff(2.0, center=1.0) == pytest.approx(
            -sigmoid_diff(-2.0, center=-1.0), abs=1e-12
        )

    def test_numpy_path(self) -> None:
        arr = np.array([-1e6, 0.0, 1e6])
        out = expit(arr)
        assert np.allclose(out, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# logaddexp / logsumexp
# ---------------------------------------------------------------------------


class TestLogAddExp:
    def test_both_zero(self) -> None:
        assert logaddexp(0.0, 0.0) == pytest.approx(math.log(2.0), rel=1e-9)

    def test_with_neg_inf(self) -> None:
        assert logaddexp(-math.inf, 1.0) == 1.0
        assert logaddexp(1.0, -math.inf) == 1.0

    def test_both_neg_inf(self) -> None:
        assert logaddexp(-math.inf, -math.inf) == -math.inf

    def test_pair_matches_general(self) -> None:
        for a, b in [(0.0, 0.0), (1.0, 2.0), (5.0, 5.0), (10.0, -10.0)]:
            assert logsumexp_pair(a, b) == pytest.approx(logaddexp(a, b), rel=1e-9)


class TestLogSumExp:
    def test_empty_returns_neg_inf(self) -> None:
        assert logsumexp([]) == -math.inf

    def test_all_neg_inf(self) -> None:
        assert logsumexp([-math.inf, -math.inf]) == -math.inf

    def test_two_zeros(self) -> None:
        assert logsumexp([0.0, 0.0]) == pytest.approx(math.log(2.0), rel=1e-9)

    def test_above_max(self) -> None:
        # logsumexp(1, 2, 3) = 3 + log(e^-2 + e^-1 + 1) > 3
        assert logsumexp([1.0, 2.0, 3.0]) > 3.0

    def test_does_not_overflow(self) -> None:
        # Without stable impl, logsumexp(1000, 1000) raises; with it
        # it's just 1000 + log(2).
        v = logsumexp([1000.0, 1000.0])
        assert v == pytest.approx(1000.0 + math.log(2.0), rel=1e-9)


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_uniform(self) -> None:
        probs = softmax(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])

    def test_picks_argmax(self) -> None:
        probs = softmax(np.array([0.0, 0.0, 100.0]))
        assert probs[2] == pytest.approx(1.0, abs=1e-9)
        assert probs[0] < 1e-30 and probs[1] < 1e-30

    def test_does_not_overflow(self) -> None:
        # 1000 is the canonical overflow bait for naive softmax.
        probs = softmax(np.array([1000.0, 1000.0, 1000.0]))
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])


class TestLogSoftmax:
    def test_uniform(self) -> None:
        logp = log_softmax(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(logp, np.log(np.array([1 / 3, 1 / 3, 1 / 3])))

    def test_does_not_overflow(self) -> None:
        logp = log_softmax(np.array([1000.0, 1000.0, 1000.0]))
        assert np.allclose(logp, np.log(np.array([1 / 3, 1 / 3, 1 / 3])))


# ---------------------------------------------------------------------------
# probability helpers
# ---------------------------------------------------------------------------


class TestLogPoissonPmf:
    def test_zero(self) -> None:
        # log P(K=0; rate=1) = -1
        assert log_poisson_pmf(0, 1.0) == pytest.approx(-1.0, rel=1e-9)

    def test_matches_scipy(self) -> None:
        from scipy.stats import poisson
        for k, rate in [(0, 1.0), (2, 1.5), (5, 0.3), (10, 10.0)]:
            expected = float(poisson.logpmf(k, rate))
            assert log_poisson_pmf(k, rate) == pytest.approx(expected, rel=1e-9)

    def test_zero_rate(self) -> None:
        assert log_poisson_pmf(0, 0.0) == -math.inf

    def test_negative_k(self) -> None:
        assert log_poisson_pmf(-1, 1.0) == -math.inf


class TestLogNormalPdf:
    def test_unit_variance_zero_error(self) -> None:
        # log N(0; 0, 1) = -0.5*log(2π) ≈ -0.9189
        assert log_normal_pdf(0.0, 1.0) == pytest.approx(-0.5 * math.log(2 * math.pi), rel=1e-9)

    def test_does_not_raise_on_zero_variance(self) -> None:
        # Variance is floored to sigma_min**2; result is finite, just very
        # large in magnitude.
        v = log_normal_pdf(0.0, 0.0)
        assert math.isfinite(v)

    def test_does_not_raise_on_nan_variance(self) -> None:
        v = log_normal_pdf(0.0, float("nan"))
        assert math.isfinite(v) or v == -math.inf


class TestLogStudentTPdf:
    def test_finite_for_typical_inputs(self) -> None:
        v = log_student_t_pdf(0.5, 0.1, df=5.0)
        assert math.isfinite(v)

    def test_df_floor(self) -> None:
        # df=0 should return -inf rather than divide by zero.
        assert log_student_t_pdf(0.0, 1.0, df=0.0) == -math.inf


# ---------------------------------------------------------------------------
# safe_divide / safe_sqrt / finite_or / is_finite
# ---------------------------------------------------------------------------


class TestSafeDivide:
    def test_zero_denominator(self) -> None:
        assert safe_divide(1.0, 0.0) == 0.0

    def test_nan_numerator(self) -> None:
        assert safe_divide(float("nan"), 1.0) == 0.0

    def test_inf_denominator(self) -> None:
        assert safe_divide(1.0, float("inf")) == 0.0

    def test_normal_case(self) -> None:
        assert safe_divide(6.0, 2.0) == 3.0

    def test_custom_default(self) -> None:
        assert safe_divide(1.0, 0.0, default=42.0) == 42.0


class TestSafeSqrt:
    def test_negative(self) -> None:
        assert safe_sqrt(-1.0) == 0.0

    def test_nan(self) -> None:
        assert safe_sqrt(float("nan")) == 0.0

    def test_positive(self) -> None:
        assert safe_sqrt(4.0) == 2.0

    def test_none(self) -> None:
        assert safe_sqrt(None) == 0.0


class TestFiniteOr:
    def test_finite(self) -> None:
        assert finite_or(2.5) == 2.5

    def test_nan(self) -> None:
        assert finite_or(float("nan"), 1.0) == 1.0

    def test_inf(self) -> None:
        assert finite_or(float("inf"), 1.0) == 1.0

    def test_none(self) -> None:
        assert finite_or(None, 1.0) == 1.0


class TestIsFinite:
    def test_scalar(self) -> None:
        assert is_finite(1.0)
        assert not is_finite(float("inf"))
        assert not is_finite(float("nan"))

    def test_numpy(self) -> None:
        assert is_finite(np.array([1.0, 2.0, 3.0]))
        assert not is_finite(np.array([1.0, float("inf")]))
        assert not is_finite(np.array([1.0, float("nan")]))


# ---------------------------------------------------------------------------
# smoke test: every export works
# ---------------------------------------------------------------------------


def test_all_exports_callable() -> None:
    # If a function were renamed or removed from the public API, this test
    # breaks at import time, which is the early-warning we want.
    public = [
        "exp_clamped",
        "log_clamped",
        "log1p_clamped",
        "logaddexp",
        "logsumexp_pair",
        "logsumexp",
        "expit",
        "sigmoid",
        "sigmoid_diff",
        "softplus",
        "softmax",
        "log_softmax",
        "log_poisson_pmf",
        "log_normal_pdf",
        "log_student_t_pdf",
        "safe_divide",
        "safe_sqrt",
        "finite_or",
        "is_finite",
    ]
    for name in public:
        assert callable(globals()[name])
