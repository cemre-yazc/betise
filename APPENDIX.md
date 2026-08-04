# BeTiSe — Paper Appendix

Supplementary material for *BeTiSe: A Comprehensive Benchmark Dataset for Univariate Time Series Stationarity and Structural Analysis* (ITISE 2026).

- Dataset (CSV via Zenodo): https://doi.org/10.5281/zenodo.18513505
- Python package: https://pypi.org/project/betise/

## Appendix A — Database and Metadata Schema

The BeTiSe library stores generated series in `dataset_generation.zip` using both CSV and Apache Parquet formats. The generation logic is contained within `Data_Generation.ipynb`. Each series is accompanied by a metadata record that enables granular analysis of model performance across specific time-series characteristics.

### A.1 Labeling and Column Definitions

Binary flags indicate the presence (1) or absence (0) of a specific characteristic.

| Column Name | Statistical or Structural Meaning |
|---|---|
| `data` | Time-ordered values (Z-normalized). |
| `stationary` | Set to 1 for raw, stationary baseline series. |
| `det_lin_up` | Deterministic linear upward trend. |
| `det_lin_down` | Deterministic linear downward trend. |
| `det_quad` | Quadratic (parabolic) trend components. |
| `det_cubic` | Cubic (S-shaped) growth dynamics. |
| `det_exp` | Nonlinear exponential growth or decay. |
| `det_damped` | Directional trend with exponential damping. |
| `stoc_trend` | Stochastic processes (RW, RWD, ARI, IMA, ARIMA). |
| `volatility` | Time-varying variance (ARCH, GARCH, EGARCH, APARCH). |
| `single_seas` | Periodic patterns with a single dominant frequency. |
| `multiple_seas` | Patterns containing multiple cyclic frequencies. |
| `seasonal_base` | Seasonality derived from SARMA or SARIMA models. |
| `point_anom_single` | Isolated, sudden point-wise deviation. |
| `point_anom_multi` | Multiple uncorrelated point-wise deviations. |
| `collect_anom` | Sustained deviations over a temporal segment. |
| `context_anom` | Pattern inversions within seasonal structures. |
| `mean_shift` | Permanent shift in the series mean level. |
| `var_shift` | Permanent change in the variance (scale). |
| `trend_shift` | Abrupt change in slope or trend direction. |

### A.2 Data Generation Parameters

The base series coefficients are sampled to ensure stationarity and invertibility. For example, AR coefficients are typically sampled from U(−0.9, 0.9) with the exclusion of the range (−0.3, 0.3) to ensure statistically significant patterns. Trends are scaled relative to the series length *L* to maintain visual and statistical consistency across short (*L* ∈ [50, 100]), medium (*L* ∈ [300, 500]), and long (*L* ∈ [1000, 10,000]) series.

## Appendix B — Data Generation Templates

Representative templates used to construct the BeTiSe dataset, ranging from isolated statistical properties to complex combinations of trends, breaks, and anomalies.

| ID | Trend Type | Structural Break | Seasonality | Volatility | Anomaly Type | Model Type | Notes |
|---|---|---|---|---|---|---|---|
| 1 | None | None | None | None | None | AR(1), MA(1) | Stationary base series |
| 2 | Linear ↑ | None | None | None | None | Linear + WN | Deterministic upward trend |
| 3 | Linear ↓ | None | None | None | None | Linear + WN | Deterministic downward trend |
| 4 | Quadratic | None | None | None | None | Quadratic + WN | Parabolic growth dynamics |
| 5 | RW w/ Drift | None | None | None | None | RW, ARIMA(1,1,1) | Stochastic trend component |
| 6 | None | Mean Shift (at 500) | None | None | None | ARMA(2,2) | Single structural level shift |
| 7 | None | Var. Shift (at 750) | None | None | None | AR(2) + GARCH | Sudden variance increase |
| 8 | None | Trend Shift (at 500) | None | None | None | AR + Linear Slope | Piecewise trend change |
| 9 | None | None | Single | None | None | SARMA(1,0,0) | Monthly seasonal pattern |
| 10 | None | None | Multi (Fourier) | None | None | Fourier + WN | Combined daily/weekly cycles |
| 11 | None | None | None | GARCH | None | ARMA + GARCH | Clustered volatility |
| 12 | None | None | None | EGARCH | None | MA + EGARCH | Asymmetric volatility dynamics |
| 13 | None | None | Single | None | Point (at 500) | SARMA + Anomaly | Sudden outlier drop |
| 14 | None | None | Single | None | Multi-Point | SARMA + Anomaly | Outliers at multiple indices |
| 15 | None | None | Multi | None | Collective | SARIMA + Anomaly | Gradual seasonal pattern shift |
| 16 | None | None | Multi | None | Contextual | Fourier + Anomaly | Outlier at seasonal peak |
| 17 | Cubic | Trend Shift (at 500) | Multi | GARCH | Point (at 500) | ARIMA + GARCH | 5-tuple non-stationary case |
| 18 | Linear ↑ | Mean Shift (at 500) | None | None | Point (at 750) | Linear + ARMA + Anom | Triple: Trend, Shift, and Outlier |

## Appendix C — LLM Prompts for Zero-Shot Classification

### C.1 Stationary vs. Non-Stationary Classification Prompt

**System Prompt:**
You are a helpful assistant.

**Feature Hierarchy:**
HIERARCHY OF TIME-SERIES FEATURES:

1. **STATIONARY:** Mean approximately constant over time; Variance approximately constant; No persistent trend or drift; Fluctuates around a stable level.
2. **NON-STATIONARY:** Presence of trend, drift, or evolving level; Mean or variance changes over time; Structural breaks or regime shifts; Long-term upward or downward movement.

**Allowed Labels:** Stationary, Non-Stationary

**Classification Prompt:**
You are an expert in time-series feature identification. Given an observation (a univariate series), decide which SINGLE feature from the allowed set best describes the series' MOST PROMINENT characteristic. Use the hierarchy for reasoning and keep the response concise.

**Decision Rule:**
- If there is any clear long-term trend, drift, or level change, choose Non-Stationary.
- Only choose Stationary if the series clearly fluctuates around a constant level with stable variance.
- When uncertain, prefer Stationary.

**Format:**
Action: Boxed {\<one label from the allowed set above\>}

### C.2 Five-Class Time-Series Feature Classification Prompt

**System Prompt:**
You are a helpful assistant.

**Feature Hierarchy:**
HIERARCHY OF TIME-SERIES FEATURES:

1. **Deterministic Trends:** Linear (constant rate), Quadratic (curved trajectory), Cubic (complex patterns/direction changes), Exponential (exponential growth/decline), Damped (starts strong but slows).
2. **Stochastic Trends:** Trends incorporating random fluctuations influenced by external shocks.
3. **Structural Breaks:** Abrupt changes: Mean Shift (average level change), Variance Shift (variability change), Trend Shift (direction/magnitude change).
4. **Volatility:** High degree of fluctuation or dispersion over time.
5. **Anomaly:** Point Anomaly (single different point), Collective Anomaly (group forming unusual pattern), Contextual Anomaly (abnormal only within specific context).

**Allowed Labels:** Deterministic Trend, Stochastic Trend, Structural Break, Volatility, Anomaly

**Classification Prompt:**
You are an expert in time-series feature identification. Given an Observation (a univariate series), decide which SINGLE feature from the allowed set best describes the series' MOST SALIENT characteristic. Use the hierarchy for reasoning and keep the response concise.

**Format:**
Action: Boxed {\<one label from the allowed set above\>}

## Appendix D — Illustrative Examples of Generated Time Series

### D.1 Series with Two Characteristics Combined

| | |
|---|---|
| ![Quadratic trend with variance shift](docs/appendix-figures/quad_var.png) | ![Damped trend with 2 mean shifts](docs/appendix-figures/damped_2_mean.png) |
| (a) quadratic trend with variance shift | (b) damped trend with 2 mean shifts |
| ![Cubic trend with a point anomaly](docs/appendix-figures/cubic_point_anom.png) | ![Linear trend with a collective anomaly](docs/appendix-figures/linear_up_var.png) |
| (c) cubic trend with a point anomaly | (d) linear trend with a collective anomaly |

### D.2 Series with Single Characteristics

| | | |
|---|---|---|
| ![Cubic trend](docs/appendix-figures/ar_cubic.png) | ![ARIMA series](docs/appendix-figures/arima_medium.png) | ![Single seasonality](docs/appendix-figures/single_seas.png) |
| (a) cubic trend | (b) ARIMA series | (c) single seasonality |
| ![Multiple seasonality](docs/appendix-figures/multi_seas.png) | ![Multiple point anomalies](docs/appendix-figures/multi_point_anom.png) | ![Multiple collective anomalies](docs/appendix-figures/multi_collect_anom.png) |
| (d) multiple seasonality | (e) multiple point anomalies | (f) multiple collective anomalies |
| ![Multiple mean shifts](docs/appendix-figures/multi_mean_shift.png) | ![Multiple variance shifts](docs/appendix-figures/multi_var_shift.png) | ![Trend shift](docs/appendix-figures/trend_shift.png) |
| (g) multiple mean shifts | (h) multiple variance shifts | (i) trend shift |

## Appendix E — Fractional Integration and ARFIMA Processes

### E.1 Overview of ARFIMA Models

**ARFIMA(p, d, q)** (AutoRegressive Fractionally Integrated Moving Average) extends classical ARIMA by allowing the differencing parameter *d* to take fractional values, enabling the modeling of long-memory processes. Unlike ARIMA models where *d* ∈ {0, 1, 2, ...}, ARFIMA admits *d* ∈ ℝ, typically constrained to (−0.5, 0.5) for stationarity *and* invertibility.

The process is defined by

$$\Phi(B)\,(1 - B)^d X_t = \Theta(B)\,\epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Key Properties:**

- **Stationarity Condition:** *d* < 0.5. This is the flag written to `is_stationary`; the range (−0.5, 0.5) additionally guarantees invertibility.
- **Long Memory:** When 0 < *d* < 0.5, the series exhibits positive long-range dependence (autocorrelations decay hyperbolically rather than exponentially).
- **Anti-Persistence:** When −0.5 < *d* < 0, the series exhibits negative dependence (mean-reverting behavior stronger than white noise).
- **Standard ARIMA:** When *d* ∈ {0, 1}, ARFIMA reduces exactly to the corresponding ARIMA process; *d* = 1 with no AR/MA terms is a random walk.

The fractional differencing operator is defined via the binomial series expansion:

$$(1 - B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-1)^k B^k$$

where the generalized binomial coefficient is:

$$\binom{d}{k} = \frac{d(d-1)(d-2)\cdots(d-k+1)}{k!} = \prod_{j=0}^{k-1} \frac{d-j}{j+1}$$

For a stationary ARFIMA(0, *d*, 0) process this expansion yields the closed-form second moments used throughout the implementation and its tests:

$$\gamma(0) = \sigma^2 \frac{\Gamma(1 - 2d)}{\Gamma(1 - d)^2}, \qquad \rho(k) = \rho(k-1)\,\frac{k - 1 + d}{k - d}, \qquad \rho(1) = \frac{d}{1 - d}$$

### E.2 Metadata Extensions for Fractional Processes

The BeTiSe metadata schema is extended to accommodate fractional base series:

| Column Name | Meaning |
|---|---|
| `fractional_integrated` | Set to 1 for ARFIMA series with fractional *d* parameter. |
| `long_memory` | Set to 1 when 0 < *d* < 0.5 (positive long-range dependence). |
| `d_parameter` | The fractional differencing parameter value (stored in context metadata). |
| `ar_order` | Number of autoregressive terms *p*. |
| `ma_order` | Number of moving average terms *q*. |

### E.3 Generation Parameters for ARFIMA

Default parameter ranges used in BeTiSe:

| Parameter | Range | Purpose |
|---|---|---|
| `d_range` | [0.25, 0.49] | Fractional parameter ensuring long memory and stationarity. |
| `p` | [1, 3] | AR order sampled uniformly. |
| `q` | [1, 3] | MA order sampled uniformly. |
| `ar_coeffs` | Stationary | Generated via `generate_arma_params()` to ensure stationarity. |
| `ma_coeffs` | Invertible | Generated to ensure invertibility. |
| `alpha` | 0 | Additive series constant. For a stationary series this is the process mean; for *d* ≥ 0.5 it sets the starting level. |
| `numseas` | 100 | Number of seasoning (burn-in) samples generated and discarded before the series is recorded. |

`alpha` and `numseas` follow the reference ARFIMA simulators (SAS/farmasim, R/arfima): a **series constant** and a **burn-in count**, not a seasonal amplitude and period. Seasonality is a separate feature category (Appendix B) and is never injected by the base generator.

**MA sign convention.** BeTiSe uses the summation convention

$$u_t = W_t + \theta_1 W_{t-1} + \cdots + \theta_q W_{t-q}$$

matching `statsmodels.tsa.arima_process.ArmaProcess` and the library's own ARMA/ARIMA generators. SAS/farmafit and R/arfima use the opposite (Box–Jenkins difference) convention, so θ estimates from those packages carry the opposite sign to the values configured here.

### E.4 Representative ARFIMA Templates

| ID | p | d | q | AR Coeffs | MA Coeffs | Notes |
|---|---|---|---|---|---|---|
| 19 | 1 | 0.30 | 1 | [0.6] | [0.4] | Moderate long memory, simple structure |
| 20 | 2 | 0.45 | 2 | [0.7, −0.3] | [0.5, 0.3] | Strong long memory, complex dynamics |
| 21 | 3 | 0.25 | 1 | [0.5, 0.2, 0.15] | [0.6] | Weak long memory, higher AR order |
| 22 | 1 | 0.40 | 3 | [0.55] | [0.4, 0.3, 0.2] | Moderate memory, higher MA order |

### E.5 Implementation Notes

BeTiSe simulates the fractional part **exactly** rather than by truncating the binomial expansion. Writing $W_t = (1 - B)^{-d}\epsilon_t$ for the fractionally integrated noise, the process factorises as an ordinary ARMA filter applied to $W_t$:

$$X_t = \Phi(B)^{-1}\,\Theta(B)\,W_t$$

so only $W_t$ needs fractional machinery.

1. **Split of *d*.** *d* is written as a stationary fractional part plus integer integrations: for *d* ≥ 0.5 the simulator draws an ARFIMA(*p*, *d*−1, *q*) core and integrates it once. This is what makes integer *d* reduce exactly to ARIMA.

2. **Exact fractional noise.** $W_t$ is drawn with the Davies–Harte circulant-embedding method from the closed-form autocovariance $\gamma(k)$ of E.1: the autocovariance sequence is wrapped into a circulant matrix, its eigenvalues are obtained by an FFT, and the sample is synthesised in the spectral domain. The result has exactly the target autocovariance, at $O(T \log T)$ cost.

3. **ARMA filtering.**
   - Apply the MA component: $u_t = W_t + \theta_1 W_{t-1} + \cdots + \theta_q W_{t-q}$
   - Apply the AR recursion: $y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + u_t$

4. **Burn-in.** `numseas` samples (default 100) are generated and discarded to remove the ARMA start-up transient. The fractional part needs no burn-in because it is drawn exactly and stationarily. Integer integration is applied *after* the burn-in so a non-stationary series starts at the requested level.

5. **Validation.** Arguments outside the documented ranges (`-1 < d ≤ 1`, `10 ≤ slen ≤ 100000`, `0 ≤ numseas ≤ 10000`, orders ≤ 10, `sigma > 0`) raise `ValueError` instead of silently producing a degenerate series.

**Why not truncation.** A truncated expansion of $(1-B)^{-d}$ attenuates exactly the long memory the model is meant to exhibit. With a 1000-term expansion the implied lag-10 autocorrelation and variance are understated by:

| *d* | lag-10 ACF | variance |
|---|---|---|
| 0.25 | −3.2% | −0.4% |
| 0.30 | −5.8% | −1.2% |
| 0.45 | −17.8% | −18.5% |
| 0.49 | −19.5% | −30.6% |

The damage is worst at the top of the default `d_range`, and truncation also breaks the integer-*d* reduction (a truncated $(1-B)^{-1}$ turns first differences into $\epsilon_t - \epsilon_{t-1000}$, giving a differenced variance of 2σ² instead of σ²). The exact sampler has neither problem.

See the implementation in `betise/utils/arfima_simulator.py`, the statistical tests in `tests/test_arfima.py`, and the usage example in `examples/09_arfima_example.ipynb`.
