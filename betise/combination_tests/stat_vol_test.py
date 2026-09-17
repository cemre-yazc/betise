"""
FINAL STATISTICAL VALIDATION
Stationary + Volatility

Tested combinations:
    AR / MA / ARMA
    x
    ARCH / GARCH / EGARCH / APARCH

Main idea
---------
Volatility is used as the innovation/error process of the stationary model:

    volatility innovations
            ↓
       AR / MA / ARMA
            ↓
        final series

We compare:

1. SOURCE detection:
   Is volatility detectable directly in the original
   ARCH/GARCH/EGARCH/APARCH innovations?

2. RECOVERED detection:
   After the innovations pass through AR/MA/ARMA,
   can we inverse-filter the final series and recover
   the same volatility structure?

If source and recovered detection rates are close,
the stationary process is NOT destroying the volatility information.

Gaussian controls are also generated.
They show the false-positive rate of the statistical tests.

Saved outputs
-------------
test_outputs/stationary_volatility/
    detection_rates.csv
    gaussian_control.csv
    plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import lfilter
from statsmodels.stats.diagnostic import (
    het_arch,
    acorr_ljungbox,
)

from betise.core.generator import TimeSeriesGenerator


# =========================================================
# SETTINGS
# =========================================================

STATIONARY_TYPES = [
    "ar",
    "ma",
    "arma",
]

VOLATILITY_TYPES = [
    "arch",
    "garch",
    "egarch",
    "aparch",
]

LENGTH = 400
N_TRIALS = 50

TEST_LAG = 10
ALPHA = 0.05

# The beginning of an inverse-filtered signal may contain
# initialization effects. We remove this short region.
WARMUP = 50

ROLLING_WINDOW = 25

OUTPUT_DIR = Path(
    "test_outputs/stationary_volatility"
)

PLOT_DIR = OUTPUT_DIR / "plots"


# =========================================================
# RECOVER INNOVATIONS
# =========================================================

def recover_innovations(
    series,
    stationary_kind,
    info,
):
    """
    Recover the underlying innovations from the final
    stationary AR / MA / ARMA series.

    Example:

        innovations
            ↓
          ARMA
            ↓
          series

    We apply the inverse filter:

        series
            ↓
      inverse ARMA
            ↓
       innovations
    """

    series = np.asarray(
        series,
        dtype=float
    )

    # -----------------------------------------------------
    # AR
    # -----------------------------------------------------

    if stationary_kind == "ar":

        ar_coefs = np.asarray(
            info["ar_coefs"],
            dtype=float
        )

        ar = np.r_[
            1.0,
            -ar_coefs
        ]

        ma = np.array([
            1.0
        ])

        return lfilter(
            ar,
            ma,
            series
        )

    # -----------------------------------------------------
    # MA
    # -----------------------------------------------------

    if stationary_kind == "ma":

        ma_coefs = np.asarray(
            info["ma_coefs"],
            dtype=float
        )

        ar = np.array([
            1.0
        ])

        ma = np.r_[
            1.0,
            ma_coefs
        ]

        return lfilter(
            ar,
            ma,
            series
        )

    # -----------------------------------------------------
    # ARMA
    # -----------------------------------------------------

    if stationary_kind == "arma":

        ar_coefs = np.asarray(
            info["ar_coefs"],
            dtype=float
        )

        ma_coefs = np.asarray(
            info["ma_coefs"],
            dtype=float
        )

        ar = np.r_[
            1.0,
            -ar_coefs
        ]

        ma = np.r_[
            1.0,
            ma_coefs
        ]

        return lfilter(
            ar,
            ma,
            series
        )

    raise ValueError(
        f"Unknown stationary type: "
        f"{stationary_kind}"
    )


# =========================================================
# VOLATILITY DETECTION
# =========================================================

def volatility_tests(
    residuals,
):
    """
    Detect conditional heteroskedasticity.

    ARCH-LM:
        Tests whether squared errors depend on previous
        squared errors.

    Squared Ljung-Box:
        Tests whether squared errors contain serial
        dependence.

    A small p-value (< 0.05) suggests volatility clustering.

    'any_detected' is True if at least one of the two
    tests detects volatility.
    """

    x = np.asarray(
        residuals,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    x = (
        x
        - np.mean(x)
    )

    # ARCH-LM
    arch_result = het_arch(
        x,
        nlags=TEST_LAG
    )

    arch_p = float(
        arch_result[1]
    )

    # Ljung-Box on squared residuals
    lb_result = acorr_ljungbox(
        x ** 2,
        lags=[TEST_LAG],
        return_df=True
    )

    lb_p = float(
        lb_result[
            "lb_pvalue"
        ].iloc[0]
    )

    arch_detected = (
        arch_p < ALPHA
    )

    lb_detected = (
        lb_p < ALPHA
    )

    return {
        "arch_detected":
            arch_detected,

        "lb_detected":
            lb_detected,

        "any_detected":
            (
                arch_detected
                or lb_detected
            ),
    }


# =========================================================
# VOLATILITY TRIAL
# =========================================================

def run_volatility_trial(
    stationary_kind,
    volatility_kind,
):

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    # Generate ARCH/GARCH/EGARCH/APARCH innovations
    innovations, _ = (
        ts.generate_volatility(
            kind=volatility_kind,
            as_innovations=True
        )
    )

    innovations = np.asarray(
        innovations,
        dtype=float
    )

    # Use these innovations as the error process
    # of AR / MA / ARMA.
    df, stationary_info = (
        ts.generate_stationary_base_series(
            distribution=stationary_kind,
            innovations=innovations
        )
    )

    series = df[
        "data"
    ].to_numpy(
        dtype=float
    )

    # Recover innovations from final stationary series.
    recovered = recover_innovations(
        series,
        stationary_kind,
        stationary_info
    )

    # Remove initialization transient.
    recovered = recovered[
        WARMUP:
    ]

    # Match source length to recovered length.
    source = innovations[
        -len(recovered):
    ]

    source_test = volatility_tests(
        source
    )

    recovered_test = volatility_tests(
        recovered
    )

    return {
        "source_ARCH":
            source_test[
                "arch_detected"
            ],

        "source_LB2":
            source_test[
                "lb_detected"
            ],

        "source_any":
            source_test[
                "any_detected"
            ],

        "recovered_ARCH":
            recovered_test[
                "arch_detected"
            ],

        "recovered_LB2":
            recovered_test[
                "lb_detected"
            ],

        "recovered_any":
            recovered_test[
                "any_detected"
            ],
    }


# =========================================================
# GAUSSIAN CONTROL
# =========================================================

def run_gaussian_control(
    stationary_kind,
):
    """
    Negative control.

    Ordinary Gaussian innovations should NOT systematically
    look heteroskedastic.

    Therefore this gives us the false-positive rate of our
    statistical validation procedure.
    """

    ts = TimeSeriesGenerator(
        length=LENGTH
    )

    innovations = np.random.normal(
        0,
        1,
        LENGTH
    )

    df, stationary_info = (
        ts.generate_stationary_base_series(
            distribution=stationary_kind,
            innovations=innovations
        )
    )

    series = df[
        "data"
    ].to_numpy(
        dtype=float
    )

    recovered = recover_innovations(
        series,
        stationary_kind,
        stationary_info
    )

    recovered = recovered[
        WARMUP:
    ]

    result = volatility_tests(
        recovered
    )

    return result


# =========================================================
# REPEATED VOLATILITY VALIDATION
# =========================================================

def run_validation():

    rows = []

    print(
        "\n========================================"
    )
    print(
        "STATIONARY + VOLATILITY"
    )
    print(
        "FINAL STATISTICAL VALIDATION"
    )
    print(
        "========================================\n"
    )

    for stationary_kind in STATIONARY_TYPES:

        for volatility_kind in VOLATILITY_TYPES:

            print(
                f"Running "
                f"{stationary_kind.upper()} + "
                f"{volatility_kind.upper()} ..."
            )

            source_arch = 0
            source_lb = 0
            source_any = 0

            recovered_arch = 0
            recovered_lb = 0
            recovered_any = 0

            for _ in range(
                N_TRIALS
            ):

                result = run_volatility_trial(
                    stationary_kind,
                    volatility_kind
                )

                source_arch += int(
                    result[
                        "source_ARCH"
                    ]
                )

                source_lb += int(
                    result[
                        "source_LB2"
                    ]
                )

                source_any += int(
                    result[
                        "source_any"
                    ]
                )

                recovered_arch += int(
                    result[
                        "recovered_ARCH"
                    ]
                )

                recovered_lb += int(
                    result[
                        "recovered_LB2"
                    ]
                )

                recovered_any += int(
                    result[
                        "recovered_any"
                    ]
                )

            rows.append({
                "stationary":
                    stationary_kind,

                "volatility":
                    volatility_kind,

                "source_ARCH":
                    source_arch / N_TRIALS,

                "source_LB2":
                    source_lb / N_TRIALS,

                "source_any":
                    source_any / N_TRIALS,

                "recovered_ARCH":
                    recovered_arch / N_TRIALS,

                "recovered_LB2":
                    recovered_lb / N_TRIALS,

                "recovered_any":
                    recovered_any / N_TRIALS,
            })

    return pd.DataFrame(
        rows
    )


# =========================================================
# GAUSSIAN CONTROL VALIDATION
# =========================================================

def run_gaussian_validation():

    rows = []

    print(
        "\n========================================"
    )
    print(
        "GAUSSIAN CONTROL"
    )
    print(
        "========================================\n"
    )

    for stationary_kind in STATIONARY_TYPES:

        arch_count = 0
        lb_count = 0
        any_count = 0

        print(
            f"Running Gaussian "
            f"{stationary_kind.upper()} ..."
        )

        for _ in range(
            N_TRIALS
        ):

            result = run_gaussian_control(
                stationary_kind
            )

            arch_count += int(
                result[
                    "arch_detected"
                ]
            )

            lb_count += int(
                result[
                    "lb_detected"
                ]
            )

            any_count += int(
                result[
                    "any_detected"
                ]
            )

        rows.append({
            "stationary":
                stationary_kind,

            "ARCH_false_positive":
                arch_count / N_TRIALS,

            "LB2_false_positive":
                lb_count / N_TRIALS,

            "ANY_false_positive":
                any_count / N_TRIALS,
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# REPRESENTATIVE PLOTS
# =========================================================

def save_representative_plots():
    """
    These plots are qualitative examples.

    Each plot shows recovered innovations together with a
    rolling standard deviation.

    Interpretation:
    ----------------
    If volatility clustering exists, the rolling standard
    deviation should show regions of relatively high and
    relatively low variance instead of remaining constant.

    These plots are NOT the statistical test themselves.
    They are visual examples supporting the numerical tests.
    """

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for stationary_kind in STATIONARY_TYPES:

        for volatility_kind in VOLATILITY_TYPES:

            ts = TimeSeriesGenerator(
                length=LENGTH
            )

            innovations, _ = (
                ts.generate_volatility(
                    kind=volatility_kind,
                    as_innovations=True
                )
            )

            df, info = (
                ts.generate_stationary_base_series(
                    distribution=stationary_kind,
                    innovations=innovations
                )
            )

            recovered = recover_innovations(
                df["data"].to_numpy(
                    dtype=float
                ),
                stationary_kind,
                info
            )

            recovered = recovered[
                WARMUP:
            ]

            rolling_std = (
                pd.Series(recovered)
                .rolling(
                    ROLLING_WINDOW
                )
                .std()
            )

            # ---------------------------------------------
            # Recovered innovations
            # ---------------------------------------------

            plt.figure(
                figsize=(10, 4)
            )

            plt.plot(
                recovered
            )

            plt.xlabel(
                "Time"
            )

            plt.ylabel(
                "Recovered innovation"
            )

            plt.title(
                f"{stationary_kind.upper()} + "
                f"{volatility_kind.upper()} "
                "— Recovered Innovations"
            )

            plt.tight_layout()

            plt.savefig(
                PLOT_DIR
                / (
                    f"{stationary_kind}_"
                    f"{volatility_kind}_innovations.png"
                ),
                dpi=300
            )

            plt.close()

            # ---------------------------------------------
            # Rolling standard deviation
            # ---------------------------------------------

            plt.figure(
                figsize=(10, 4)
            )

            plt.plot(
                rolling_std
            )

            plt.xlabel(
                "Time"
            )

            plt.ylabel(
                "Rolling standard deviation"
            )

            plt.title(
                f"{stationary_kind.upper()} + "
                f"{volatility_kind.upper()} "
                "— Rolling Volatility"
            )

            plt.tight_layout()

            plt.savefig(
                PLOT_DIR
                / (
                    f"{stationary_kind}_"
                    f"{volatility_kind}_rolling_std.png"
                ),
                dpi=300
            )

            plt.close()


# =========================================================
# SAVE SUMMARY PLOTS
# =========================================================

def save_summary_plots(
    detection_results,
    gaussian_results,
):
    """
    SOURCE vs RECOVERED plot
    ------------------------
    Source:
        Detectability of volatility BEFORE AR/MA/ARMA.

    Recovered:
        Detectability AFTER passing through the stationary
        process and inverse-filtering it again.

    Bars close to each other indicate that the stationary
    model preserved the volatility structure.

    Gaussian control plot
    ---------------------
    Shows how often ordinary Gaussian innovations are
    incorrectly identified as volatile.

    This value should remain much lower than the volatility
    detection rates.
    """

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for stationary_kind in STATIONARY_TYPES:

        subset = detection_results[
            detection_results[
                "stationary"
            ] == stationary_kind
        ].copy()

        x = np.arange(
            len(subset)
        )

        width = 0.35

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            x - width / 2,
            subset["source_any"],
            width,
            label="Source"
        )

        plt.bar(
            x + width / 2,
            subset["recovered_any"],
            width,
            label="Recovered"
        )

        plt.xticks(
            x,
            subset[
                "volatility"
            ].str.upper()
        )

        plt.ylim(
            0,
            1.05
        )

        plt.ylabel(
            "Detection rate"
        )

        plt.xlabel(
            "Volatility model"
        )

        plt.title(
            f"{stationary_kind.upper()} "
            "— Source vs Recovered Volatility"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR
            / (
                f"{stationary_kind}"
                "_source_vs_recovered.png"
            ),
            dpi=300
        )

        plt.close()

    # Gaussian false-positive plot
    x = np.arange(
        len(gaussian_results)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x,
        gaussian_results[
            "ANY_false_positive"
        ]
    )

    plt.xticks(
        x,
        gaussian_results[
            "stationary"
        ].str.upper()
    )

    plt.ylim(
        0,
        1.0
    )

    plt.xlabel(
        "Stationary model"
    )

    plt.ylabel(
        "False-positive rate"
    )

    plt.title(
        "Gaussian Control — "
        "False-Positive Volatility Detection"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "gaussian_false_positive.png",
        dpi=300
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detection_results = (
        run_validation()
    )

    gaussian_results = (
        run_gaussian_validation()
    )

    # Save numerical results
    detection_results.to_csv(
        OUTPUT_DIR
        / "detection_rates.csv",
        index=False
    )

    gaussian_results.to_csv(
        OUTPUT_DIR
        / "gaussian_control.csv",
        index=False
    )

    print(
        "\n========================================"
    )
    print(
        "VOLATILITY DETECTION RATES"
    )
    print(
        "========================================"
    )

    print(
        detection_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    print(
        "\n========================================"
    )
    print(
        "GAUSSIAN FALSE-POSITIVE RATES"
    )
    print(
        "========================================"
    )

    print(
        gaussian_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    save_summary_plots(
        detection_results,
        gaussian_results
    )

    save_representative_plots()

    print(
        "\n========================================"
    )
    print(
        "INTERPRETATION"
    )
    print(
        "========================================"
    )

    print(
        "source_any      : volatility detection "
        "before AR/MA/ARMA filtering"
    )

    print(
        "recovered_any   : volatility detection "
        "after inverse-filtering the final series"
    )

    print(
        "Similar source/recovered rates mean that "
        "the stationary model preserves volatility."
    )

    print(
        "Gaussian false-positive rates provide the "
        "non-volatility baseline."
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )