"""ARFIMA (AutoRegressive Fractionally Integrated Moving Average) simulator.

The ARFIMA(p, d, q) process is defined by

    Phi(B) (1 - B)^d X_t = Theta(B) e_t,      e_t ~ N(0, sigma^2)

Writing W_t = (1 - B)^(-d) e_t for the fractionally integrated noise, the
process factorises as an ordinary ARMA(p, q) filter applied to W_t:

    X_t = Phi(B)^(-1) Theta(B) W_t

This module exploits that factorisation. The fractional part is simulated
*exactly* with the Davies-Harte circulant-embedding method, which draws a
Gaussian vector with the closed-form ARFIMA(0, d, 0) autocovariance instead of
truncating the binomial expansion of (1 - B)^(-d). Truncation attenuates long
memory badly near the stationarity boundary (at d = 0.49 a 1000-term expansion
understates the lag-10 autocorrelation by roughly 20% and the variance by 30%),
which is exactly the region a long-memory benchmark cares about.

Only d < 0.5 is stationary. For d >= 0.5 the parameter is split into a
stationary fractional part and one integer integration, so integer d reduces
exactly to the corresponding ARIMA process (d = 1 yields a true random walk).

Notes
-----
MA coefficients follow the *summation* (statsmodels) convention

    u_t = W_t + theta_1 W_{t-1} + ... + theta_q W_{t-q}

which matches ``statsmodels.tsa.arima_process.ArmaProcess`` and therefore the
rest of BeTiSe's ARMA/ARIMA generators. Reference implementations aimed at
SAS/farmasim and R/arfima use the opposite (Box-Jenkins difference) sign, so
theta estimates obtained from those packages carry the opposite sign.

References
----------
Davies, R.B. and Harte, D.S. (1987). Tests for Hurst effect. Biometrika,
74(1), 95-101.

Wood, A.T.A. and Chan, G. (1994). Simulation of stationary Gaussian processes
in [0, 1]^d. Journal of Computational and Graphical Statistics, 3(4), 409-432.

Hosking, J.R.M. (1981). Fractional differencing. Biometrika, 68(1), 165-176.
"""

from math import lgamma

import numpy as np
from numpy.fft import fft

# Reference ARFIMA simulators (SAS/farmasim, R/arfima) cap the model orders and
# the series/seasoning lengths; the same limits are kept here so that parameter
# validation stays comparable across tools.
_MAX_ORDER = 10
_MIN_SLEN = 10
_MAX_SLEN = 100_000
_MAX_NUMSEAS = 10_000


def _as_coef_vector(coeffs, name: str) -> np.ndarray:
    """Validate and flatten an AR/MA coefficient argument to a 1d float array."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim > 2 or (c.ndim == 2 and c.shape[1] != 1):
        raise ValueError(
            f"ARFIMA_sim: {name} must be a 1d array or a 2d array with a "
            f"single column, got shape {c.shape}"
        )
    c = np.reshape(c, -1)
    if c.size > _MAX_ORDER:
        raise ValueError(
            f"ARFIMA_sim: {name} order must be <= {_MAX_ORDER}, got {c.size}"
        )
    if not np.all(np.isfinite(c)):
        raise ValueError(f"ARFIMA_sim: {name} must contain only finite values")
    return c


def _fractional_autocovariance(d: float, n: int, sigma: float = 1.0) -> np.ndarray:
    """Autocovariances gamma(0), ..., gamma(n-1) of an ARFIMA(0, d, 0) process.

    Uses the closed forms

        gamma(0) = sigma^2 * Gamma(1 - 2d) / Gamma(1 - d)^2
        rho(k)   = rho(k-1) * (k - 1 + d) / (k - d)

    Parameters
    ----------
    d : float
        Fractional differencing parameter, must satisfy d < 0.5 (stationarity).
    n : int
        Number of autocovariances to return.
    sigma : float, optional
        Innovation standard deviation (default: 1.0).

    Returns
    -------
    np.ndarray
        Autocovariance sequence of length ``n``.

    Notes
    -----
    gamma(0) is evaluated through ``lgamma`` and the remaining lags through a
    ratio recursion, so nothing overflows even for d approaching 0.5 where the
    variance itself grows without bound.
    """
    if d >= 0.5:
        raise ValueError(
            f"_fractional_autocovariance: d must be < 0.5 for a stationary "
            f"process, got {d}"
        )
    gamma0 = sigma**2 * np.exp(lgamma(1.0 - 2.0 * d) - 2.0 * lgamma(1.0 - d))
    if n == 1:
        return np.array([gamma0])
    k = np.arange(1, n, dtype=float)
    ratios = (k - 1.0 + d) / (k - d)
    return gamma0 * np.concatenate(([1.0], np.cumprod(ratios)))


def _davies_harte(acov: np.ndarray) -> np.ndarray:
    """Draw an exact Gaussian sample with the given autocovariance sequence.

    Circulant-embedding (Davies-Harte / Wood-Chan) method: the autocovariance
    sequence is wrapped into a circulant matrix whose eigenvalues are obtained
    by an FFT, and the sample is synthesised in the spectral domain. The result
    has *exactly* the requested autocovariance structure, not a truncated
    approximation of it.

    Parameters
    ----------
    acov : np.ndarray
        Autocovariances gamma(0), ..., gamma(n-1) of a stationary process.

    Returns
    -------
    np.ndarray
        Sample of length ``len(acov)``.
    """
    n = acov.shape[0]
    if n < 2:
        return np.sqrt(acov[0]) * np.random.normal(size=n)

    m = 2 * (n - 1)
    # Symmetric wrap: [g0, g1, ..., g_{n-1}, g_{n-2}, ..., g1], length 2(n-1).
    g = np.concatenate([acov, acov[-2:0:-1]])
    lam = fft(g).real

    # Eigenvalues are non-negative for any valid ARFIMA spectral density; guard
    # against round-off producing tiny negatives rather than silently emitting
    # NaNs from the square root below.
    lam_max = float(lam.max())
    if lam.min() < -1e-6 * max(lam_max, 1.0):
        raise RuntimeError(
            "_davies_harte: circulant embedding produced negative eigenvalues "
            f"(min={lam.min():.3e}, max={lam_max:.3e}); the autocovariance "
            "sequence is not non-negative definite"
        )
    np.clip(lam, 0.0, None, out=lam)

    half = m // 2  # == n - 1
    z = np.random.normal(size=(2, half + 1))
    y = np.empty(m, dtype=complex)
    # The zero and Nyquist frequencies must be real; the rest come in
    # conjugate pairs so that the inverse transform is real-valued.
    y[0] = np.sqrt(lam[0]) * z[0, 0]
    y[half] = np.sqrt(lam[half]) * z[0, half]
    k = np.arange(1, half)
    y[k] = np.sqrt(lam[k] / 2.0) * (z[0, k] + 1j * z[1, k])
    y[m - k] = np.conj(y[k])

    return (fft(y).real / np.sqrt(m))[:n]


def ARFIMA_sim(
    p_coeffs,
    q_coeffs,
    d: float,
    slen: int,
    alpha: float = 0.0,
    sigma: float = 1.0,
    numseas: int = 100,
) -> np.ndarray:
    """Generate a random ARFIMA(p, d, q) series.

    Generalises to ARMA(p, q) when d = 0 and to ARIMA(p, 1, q) when d = 1.

    Parameters
    ----------
    p_coeffs : array_like
        AR(p) coefficients, ``len(p_coeffs) <= 10``. Empty for no AR part.
    q_coeffs : array_like
        MA(q) coefficients, ``len(q_coeffs) <= 10``. Empty for no MA part.
        Summation sign convention, see module docstring.
    d : float
        Fractional differencing parameter, ``-1 < d <= 1``.

        - ``d < 0.5``: stationary (and invertible for ``d > -0.5``)
        - ``0 < d < 0.5``: long memory (persistence)
        - ``-0.5 < d < 0``: anti-persistence
        - ``d >= 0.5``: non-stationary
    slen : int
        Number of samples in the returned series, ``10 <= slen <= 100000``.
    alpha : float, optional
        Additive series constant (default: 0.0). For a stationary series this
        is the process mean; for ``d >= 0.5`` it sets the starting level.
    sigma : float, optional
        Standard deviation of the Gaussian innovations (default: 1.0).
    numseas : int, optional
        Number of seasoning (burn-in) samples generated and discarded before
        the series is recorded, ``0 <= numseas <= 10000`` (default: 100). Only
        the ARMA recursion needs a burn-in here: the fractional part is drawn
        exactly and needs none.

    Returns
    -------
    np.ndarray
        1d ARFIMA(p, d, q) series of length ``slen``.

    Raises
    ------
    ValueError
        If any argument falls outside the documented range.

    Examples
    --------
    >>> np.random.seed(0)
    >>> x = ARFIMA_sim([0.5], [-0.2, 0.2], 0.3, 1000)
    >>> x.shape
    (1000,)
    """
    p = _as_coef_vector(p_coeffs, "p")
    q = _as_coef_vector(q_coeffs, "q")

    if not -1.0 < d <= 1.0:
        raise ValueError(
            f"ARFIMA_sim: differencing parameter must be in range (-1, 1], got {d}"
        )
    if not _MIN_SLEN <= slen <= _MAX_SLEN:
        raise ValueError(
            f"ARFIMA_sim: series length must be in range "
            f"[{_MIN_SLEN}, {_MAX_SLEN}], got {slen}"
        )
    if not 0 <= numseas <= _MAX_NUMSEAS:
        raise ValueError(
            f"ARFIMA_sim: seasoning length must be in range "
            f"[0, {_MAX_NUMSEAS}], got {numseas}"
        )
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"ARFIMA_sim: sigma must be positive and finite, got {sigma}")
    if not np.isfinite(alpha):
        raise ValueError(f"ARFIMA_sim: alpha must be finite, got {alpha}")

    slen = int(slen)
    numseas = int(numseas)

    # Split d so that the simulated core is stationary: d >= 0.5 becomes one
    # integer integration of an ARFIMA(p, d-1, q) process. This makes integer d
    # reduce exactly to the corresponding ARIMA process.
    n_integrate = 1 if d >= 0.5 else 0
    d_frac = d - n_integrate

    total = slen + numseas

    # 1. Fractionally integrated noise  W_t = (1 - B)^(-d_frac) e_t.
    if abs(d_frac) < 1e-12:
        w = np.random.normal(scale=sigma, size=total)
    else:
        w = _davies_harte(_fractional_autocovariance(d_frac, total, sigma))

    # 2. MA(q) filter:  u_t = W_t + theta_1 W_{t-1} + ... + theta_q W_{t-q}.
    u = np.convolve(w, np.r_[1.0, q], mode="full")[:total] if q.size else w

    # 3. AR(p) recursion:  x_t = phi_1 x_{t-1} + ... + phi_p x_{t-p} + u_t.
    if p.size:
        x = np.empty(total)
        for t in range(total):
            k = min(p.size, t)
            x[t] = u[t] + (p[:k] @ x[t - k : t][::-1] if k else 0.0)
    else:
        x = np.asarray(u, dtype=float)

    # 4. Drop the ARMA start-up transient.
    x = x[numseas:]

    # 5. Integer integration for the non-stationary case, applied after the
    #    burn-in so the series starts at the requested level rather than at an
    #    arbitrary random walk offset.
    for _ in range(n_integrate):
        x = np.cumsum(x)

    return alpha + x
