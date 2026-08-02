"""ARFIMA (AutoRegressive Fractionally Integrated Moving Average) simulator.

This module implements ARFIMA series generation using fractional differencing.
"""

import numpy as np
from numpy.fft import fft, ifft


def _fractional_diff_weights(d: float, length: int) -> np.ndarray:
    """Compute fractional differencing weights using binomial expansion.
    
    Parameters
    ----------
    d : float
        Fractional differencing parameter
    length : int
        Number of weights to compute
        
    Returns
    -------
    np.ndarray
        Array of fractional differencing weights
    """
    weights = np.zeros(length)
    weights[0] = 1.0
    
    for k in range(1, length):
        weights[k] = weights[k-1] * (k - 1 - d) / k
    
    return weights


def ARFIMA_sim(
    p_coeffs: np.ndarray,
    q_coeffs: np.ndarray,
    d: float,
    slen: int,
    alpha: float = 0.0,
    sigma: float = 1.0,
    numseas: int = 100
) -> np.ndarray:
    """Generate ARFIMA(p, d, q) time series.
    
    Parameters
    ----------
    p_coeffs : np.ndarray
        AR coefficients (phi_1, ..., phi_p)
    q_coeffs : np.ndarray
        MA coefficients (theta_1, ..., theta_q)
    d : float
        Fractional differencing parameter
        - d in (-0.5, 0.5): stationary
        - d in (0, 0.5): long memory (persistence)
        - d in (-0.5, 0): anti-persistence
    slen : int
        Length of series to generate
    alpha : float, optional
        Seasonal component strength (default: 0.0)
    sigma : float, optional
        Standard deviation of innovations (default: 1.0)
    numseas : int, optional
        Seasonal period for optional seasonal component (default: 100)
        
    Returns
    -------
    np.ndarray
        Generated ARFIMA series
        
    Notes
    -----
    The ARFIMA(p,d,q) model is defined as:
        Φ(B) (1-B)^d X_t = Θ(B) ε_t
    where:
        - Φ(B) is the AR polynomial
        - Θ(B) is the MA polynomial
        - (1-B)^d is the fractional differencing operator
        - ε_t ~ N(0, σ²)
    """
    p = len(p_coeffs)
    q = len(q_coeffs)
    
    # Generate extra samples for burn-in
    burnin = max(500, 2 * slen)
    total_length = slen + burnin
    
    # Generate white noise innovations
    epsilon = np.random.normal(0, sigma, total_length)
    
    # Apply MA component if present
    if q > 0:
        # MA polynomial: 1 + theta_1*B + ... + theta_q*B^q
        ma_poly = np.r_[1, q_coeffs]
        epsilon = np.convolve(epsilon, ma_poly, mode='full')[:total_length]
    
    # Apply fractional integration: (1-B)^(-d)
    # This is the inverse operation of fractional differencing
    if abs(d) > 1e-10:  # Only if d is significantly different from 0
        frac_weights = _fractional_diff_weights(-d, min(total_length, 1000))
        # Use convolution to apply fractional integration
        series = np.convolve(epsilon, frac_weights, mode='full')[:total_length]
    else:
        series = epsilon.copy()
    
    # Apply AR component if present
    if p > 0:
        # AR polynomial: 1 - phi_1*B - ... - phi_p*B^p
        # Solve: X_t = phi_1*X_{t-1} + ... + phi_p*X_{t-p} + innovation
        ar_series = series.copy()
        for t in range(p, total_length):
            ar_component = np.dot(p_coeffs, ar_series[t-p:t][::-1])
            ar_series[t] = series[t] + ar_component
        series = ar_series
    
    # Add optional seasonal component
    if alpha > 0 and numseas > 0:
        t = np.arange(total_length)
        seasonal = alpha * np.sin(2 * np.pi * t / numseas)
        series += seasonal
    
    # Remove burn-in period
    series = series[burnin:]
    
    return series[:slen]
