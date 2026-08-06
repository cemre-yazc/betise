"""Statistical tests for the ARFIMA simulator.

Unlike the smoke tests in ``test_generators.py``, these assert that the
generated series actually have the theoretical second-order properties of an
ARFIMA(p, d, q) process:

- the ARFIMA(0, d, 0) autocovariance matches its closed form, including near
  the stationarity boundary where a truncated binomial expansion fails;
- integer d reduces exactly to the corresponding ARIMA process;
- ``alpha`` is an additive constant and ``numseas`` is a burn-in count;
- out-of-range arguments are rejected.

Tolerances are chosen to leave a wide margin over the Monte Carlo error at the
fixed seeds used here, so the tests are deterministic in practice.
"""
from math import lgamma

import numpy as np
import pytest

from betise.utils.arfima_simulator import (
    ARFIMA_sim,
    _davies_harte,
    _fractional_autocovariance,
)

SEED = 20240804


def _acf(x: np.ndarray, kmax: int) -> np.ndarray:
    """Sample autocorrelations for lags 1..kmax."""
    x = x - x.mean()
    n = x.size
    c0 = x @ x / n
    return np.array([x[: -k] @ x[k:] / n / c0 for k in range(1, kmax + 1)])


def _theoretical_acf(d: float, kmax: int) -> np.ndarray:
    """rho(k) = rho(k-1) * (k - 1 + d) / (k - d) for an ARFIMA(0, d, 0)."""
    rho = [1.0]
    for k in range(1, kmax + 1):
        rho.append(rho[-1] * (k - 1 + d) / (k - d))
    return np.array(rho[1:])


def _theoretical_variance(d: float, sigma: float = 1.0) -> float:
    """gamma(0) = sigma^2 * Gamma(1 - 2d) / Gamma(1 - d)^2."""
    return sigma**2 * np.exp(lgamma(1.0 - 2.0 * d) - 2.0 * lgamma(1.0 - d))


def _mean_acov(d: float, n: int, reps: int, sigma: float = 1.0, kmax: int = 6):
    """Monte Carlo autocovariances of the fractional sampler.

    The sample mean is deliberately *not* removed: for a long-memory process it
    is strongly correlated with the series and biases the estimator downwards,
    which would mask what is being tested here (whether the sampler reproduces
    the requested autocovariance).
    """
    target = _fractional_autocovariance(d, n, sigma)
    np.random.seed(SEED)
    var = 0.0
    acov = np.zeros(kmax)
    for _ in range(reps):
        w = _davies_harte(target)
        var += w @ w / n
        acov += np.array([w[: -k] @ w[k:] / n for k in range(1, kmax + 1)])
    return var / reps, acov / reps, target


# ── Shape / finiteness across the whole admissible d range ───────────────────
@pytest.mark.parametrize("d", [-0.99, -0.5, -0.3, 0.0, 0.25, 0.45, 0.49, 0.5, 0.7, 1.0])
def test_shape_and_finite(d):
    np.random.seed(SEED)
    x = ARFIMA_sim([0.4, -0.3], [0.5], d, 500, alpha=0.5, sigma=0.7)
    assert x.shape == (500,)
    assert x.ndim == 1
    assert np.isfinite(x).all()


# ── Exactness of the fractional sampler ──────────────────────────────────────
@pytest.mark.parametrize("d", [0.25, 0.3, 0.45, -0.3, -0.45])
def test_fractional_sampler_reproduces_closed_form_acov(d):
    """The core guarantee: no truncation bias in the fractional integration."""
    var, acov, target = _mean_acov(d, n=256, reps=1500)
    assert var == pytest.approx(target[0], rel=0.05)
    np.testing.assert_allclose(acov, target[1:7], rtol=0.12, atol=0.02)


def test_no_truncation_bias_near_stationarity_boundary():
    """Regression test for binomial-expansion truncation.

    A 1000-term truncated expansion understates the d = 0.49 variance by about
    80% (roughly 3.0 against a closed form of 16.4), so this fails loudly if the
    exact sampler is ever replaced by a truncated convolution.
    """
    var, acov, target = _mean_acov(0.49, n=256, reps=2000)
    assert target[0] == pytest.approx(_theoretical_variance(0.49), rel=1e-10)
    assert var == pytest.approx(target[0], rel=0.10)
    # Long memory must persist, not decay away.
    assert acov[5] / var > 0.85


def test_variance_scales_with_sigma_squared():
    for sigma in (0.5, 2.0):
        got = _fractional_autocovariance(0.3, 1, sigma)[0]
        assert got == pytest.approx(_theoretical_variance(0.3, sigma), rel=1e-12)


# ── Long memory / anti-persistence at the series level ───────────────────────
def test_long_memory_acf_matches_theory():
    d = 0.3
    np.random.seed(SEED)
    acc = np.zeros(4)
    reps = 120
    for _ in range(reps):
        acc += _acf(ARFIMA_sim([], [], d, 4000), 4)
    np.testing.assert_allclose(acc / reps, _theoretical_acf(d, 4), atol=0.04)


def test_negative_d_is_anti_persistent():
    d = -0.3
    np.random.seed(SEED)
    acc = np.zeros(3)
    reps = 120
    for _ in range(reps):
        acc += _acf(ARFIMA_sim([], [], d, 2000), 3)
    acc /= reps
    assert acc[0] < 0, "anti-persistent series must have negative lag-1 acf"
    np.testing.assert_allclose(acc, _theoretical_acf(d, 3), atol=0.02)


# ── Integer d must reduce to ARIMA ───────────────────────────────────────────
def test_d_one_is_a_random_walk():
    np.random.seed(SEED)
    stds, acc = [], np.zeros(3)
    reps = 60
    for _ in range(reps):
        dx = np.diff(ARFIMA_sim([], [], 1.0, 2000))
        stds.append(dx.std())
        acc += _acf(dx, 3)
    assert np.mean(stds) == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(acc / reps, np.zeros(3), atol=0.03)


def test_d_one_with_ar_is_arima():
    """ARFIMA(1, 1, 0) differenced once must be an AR(1)."""
    np.random.seed(SEED)
    acc = np.zeros(2)
    reps = 60
    for _ in range(reps):
        acc += _acf(np.diff(ARFIMA_sim([0.5], [], 1.0, 4000)), 2)
    np.testing.assert_allclose(acc / reps, [0.5, 0.25], atol=0.05)


def test_d_zero_reduces_to_arma():
    np.random.seed(SEED)
    ma_acc = np.zeros(1)
    ar_acc = np.zeros(3)
    reps = 150
    for _ in range(reps):
        ma_acc += _acf(ARFIMA_sim([], [0.6], 0.0, 2000), 1)
        ar_acc += _acf(ARFIMA_sim([0.7], [], 0.0, 2000), 3)
    # Summation (statsmodels) MA convention: rho(1) = theta / (1 + theta^2).
    assert ma_acc[0] / reps == pytest.approx(0.6 / 1.36, abs=0.03)
    np.testing.assert_allclose(ar_acc / reps, [0.7, 0.49, 0.343], atol=0.03)


@pytest.mark.parametrize("d", [0.5, 0.7, 0.8, 1.0])
def test_d_above_half_differences_to_the_right_stationary_process(d):
    """Differencing a d >= 0.5 series once must give an ARFIMA(0, d-1, 0).

    This is what makes the integer split exact: the non-stationary case is one
    cumulative sum of a stationary ARFIMA(p, d-1, q) core.
    """
    np.random.seed(SEED)
    var = np.mean([np.diff(ARFIMA_sim([], [], d, 2000)).var() for _ in range(60)])
    assert var == pytest.approx(_theoretical_variance(d - 1.0), rel=0.05)


def test_non_stationary_variance_grows_with_sample_size():
    """Var(x_n) ~ n^(2d-1) for d > 0.5, while a stationary series stays flat."""
    np.random.seed(SEED)
    reps = 80

    def mean_var(d, n):
        return np.mean([ARFIMA_sim([], [], d, n).var() for _ in range(reps)])

    non_stationary = mean_var(0.8, 8000) / mean_var(0.8, 1000)
    stationary = mean_var(0.3, 8000) / mean_var(0.3, 1000)
    # Theory gives 8 ** 0.6 = 3.48 for d = 0.8 and 1.0 for the stationary case.
    assert non_stationary > 2.5
    assert stationary < 1.3


# ── alpha and numseas semantics ──────────────────────────────────────────────
def test_alpha_is_an_additive_constant():
    for alpha in (3.0, -3.0):
        np.random.seed(SEED)
        base = ARFIMA_sim([0.5], [0.2], 0.3, 500)
        np.random.seed(SEED)
        shifted = ARFIMA_sim([0.5], [0.2], 0.3, 500, alpha=alpha)
        np.testing.assert_allclose(shifted - base, alpha, atol=1e-12)


def test_alpha_does_not_inject_a_seasonal_component():
    """A large alpha must shift the level without changing the variance."""
    np.random.seed(SEED)
    plain = ARFIMA_sim([], [], 0.0, 4000, alpha=0.0, numseas=50)
    np.random.seed(SEED)
    shifted = ARFIMA_sim([], [], 0.0, 4000, alpha=5.0, numseas=50)
    assert shifted.std() == pytest.approx(plain.std(), rel=1e-10)
    assert shifted.mean() - plain.mean() == pytest.approx(5.0, abs=1e-10)


def test_numseas_is_a_burn_in_count():
    np.random.seed(SEED)
    no_burn = ARFIMA_sim([0.5], [], 0.3, 200, numseas=0)
    np.random.seed(SEED)
    burned = ARFIMA_sim([0.5], [], 0.3, 200, numseas=500)
    assert no_burn.shape == burned.shape == (200,)
    assert not np.allclose(no_burn, burned), "numseas had no effect on the draw"


def test_reproducible_under_global_seed():
    np.random.seed(SEED)
    a = ARFIMA_sim([0.5], [0.2], 0.35, 300)
    np.random.seed(SEED)
    b = ARFIMA_sim([0.5], [0.2], 0.35, 300)
    np.testing.assert_array_equal(a, b)


# ── Argument validation ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d=1.5),
        dict(d=2.5),
        dict(d=-1.0),
        dict(d=-3.0),
        dict(slen=9),
        dict(slen=100_001),
        dict(numseas=-1),
        dict(numseas=10_001),
        dict(sigma=0.0),
        dict(sigma=-1.0),
        dict(sigma=np.nan),
        dict(alpha=np.inf),
    ],
)
def test_rejects_out_of_range_arguments(kwargs):
    call = dict(p_coeffs=[], q_coeffs=[], d=0.3, slen=200)
    call.update(kwargs)
    with pytest.raises(ValueError):
        ARFIMA_sim(**call)


@pytest.mark.parametrize(
    "coeffs",
    [
        np.full(11, 0.01),          # order above the documented cap
        np.zeros((2, 2)),           # 2d but not a single column
        np.zeros((2, 2, 2)),        # more than 2 dimensions
        [np.nan],                   # non-finite coefficient
    ],
)
def test_rejects_invalid_coefficients(coeffs):
    with pytest.raises(ValueError):
        ARFIMA_sim(coeffs, [], 0.3, 200)
    with pytest.raises(ValueError):
        ARFIMA_sim([], coeffs, 0.3, 200)


@pytest.mark.parametrize(
    "coeffs", [[], np.array([]), 0.5, [0.5], np.array([[0.5]]), [0.05] * 10]
)
def test_accepts_valid_coefficient_shapes(coeffs):
    np.random.seed(SEED)
    x = ARFIMA_sim(coeffs, coeffs, 0.3, 100)
    assert x.shape == (100,)
    assert np.isfinite(x).all()


def test_fractional_autocovariance_rejects_non_stationary_d():
    with pytest.raises(ValueError):
        _fractional_autocovariance(0.5, 10)
