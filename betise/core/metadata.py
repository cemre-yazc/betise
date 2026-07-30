"""
Metadata management for seasonal time series datasets.

This module creates metadata records from the info dictionaries returned by:
- generate_single_seasonality
- generate_multiple_seasonality
- generate_sarima_series
- generate_deterministic_sarma_series
"""

import json
import numpy as np
import pandas as pd


def create_metadata_record(
    # === CORE ===
    series_id,
    length,
    label,
    is_stationary=0,

    # === SEASONAL INFO ===
    series_type=None,          # info["type"] -> "seasonal"
    subtype=None,              # info["subtype"] -> single/multiple/SARIMA/SARMA
    periods=None,
    period_meanings=None,

    # === FOURIER / SEASONALITY PARAMETERS ===
    amplitude=None,            # single, SARIMA, SARMA
    amplitudes=None,           # multiple
    noise_std=None,
    scale_factor=None,
    num_harmonics=None,
    coefficients=None,         # single, multiple, SARIMA
    fourier_coefficients=None, # SARMA

    # === SARIMA-SPECIFIC ===
    diff=None,
    seasonal_diff=None,
    unit_root=None,
    initial_std=None,

    # === SARMA-SPECIFIC ===
    ar_order=None,
    ma_order=None,
    seasonal_ar_order=None,
    seasonal_ma_order=None,
    ar_coefs=None,
    ma_coefs=None,
    seasonal_ar_coefs=None,
    seasonal_ma_coefs=None,
):
    """
    Create metadata record for seasonal time series.
    """

    record = {
        # === Core ===
        "series_id": series_id,
        "length": length,
        "label": label,
        "is_stationary": is_stationary,

        # === Seasonal identity ===
        "type": series_type,
        "subtype": subtype,
        "periods": periods,
        "period_meanings": period_meanings,

        # === Fourier / seasonality parameters ===
        "amplitude": amplitude,
        "amplitudes": amplitudes,
        "noise_std": noise_std,
        "scale_factor": scale_factor,
        "num_harmonics": num_harmonics,
        "coefficients": coefficients,
        "fourier_coefficients": fourier_coefficients,

        # === SARIMA-specific ===
        "diff": diff,
        "seasonal_diff": seasonal_diff,
        "unit_root": unit_root,
        "initial_std": initial_std,

        # === SARMA-specific ===
        "ar_order": ar_order,
        "ma_order": ma_order,
        "seasonal_ar_order": seasonal_ar_order,
        "seasonal_ma_order": seasonal_ma_order,
        "ar_coefs": ar_coefs,
        "ma_coefs": ma_coefs,
        "seasonal_ar_coefs": seasonal_ar_coefs,
        "seasonal_ma_coefs": seasonal_ma_coefs,
    }

    return record


def create_metadata_from_info(
    series_id,
    length,
    label,
    info,
    is_stationary=0
):
    """
    Create metadata directly from the info dictionary returned by the generator.
    """

    record = create_metadata_record(
        series_id=series_id,
        length=length,
        label=label,
        is_stationary=is_stationary,

        # Common seasonal fields
        series_type=info.get("type"),
        subtype=info.get("subtype"),
        periods=info.get("periods"),
        period_meanings=info.get("period_meanings"),

        # Fourier / seasonality parameters
        amplitude=info.get("amplitude"),
        amplitudes=info.get("amplitudes"),
        noise_std=info.get("noise_std"),
        scale_factor=info.get("scale_factor"),
        num_harmonics=info.get("num_harmonics"),
        coefficients=info.get("coefficients"),
        fourier_coefficients=info.get("fourier_coefficients"),

        # SARIMA-specific
        diff=info.get("diff"),
        seasonal_diff=info.get("seasonal_diff"),
        unit_root=info.get("unit_root"),
        initial_std=info.get("initial_std"),

        # SARMA-specific
        ar_order=info.get("ar_order"),
        ma_order=info.get("ma_order"),
        seasonal_ar_order=info.get("seasonal_ar_order"),
        seasonal_ma_order=info.get("seasonal_ma_order"),
        ar_coefs=info.get("ar_coefs"),
        ma_coefs=info.get("ma_coefs"),
        seasonal_ar_coefs=info.get("seasonal_ar_coefs"),
        seasonal_ma_coefs=info.get("seasonal_ma_coefs"),
    )

    return record


def make_json_serializable(obj):
    """
    Convert numpy objects to JSON-serializable Python objects.
    """

    if obj is None:
        return None

    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]

    if isinstance(obj, set):
        return [make_json_serializable(x) for x in sorted(obj)]

    if isinstance(obj, dict):
        return {
            str(make_json_serializable(k)): make_json_serializable(v)
            for k, v in obj.items()
        }

    return obj


def metadata_value_to_cell(value):
    """
    Convert metadata values into dataframe-cell-friendly values.

    Scalars stay as scalars.
    Lists/dicts become JSON strings.
    """

    value = make_json_serializable(value)

    if value is None:
        return None

    if isinstance(value, (int, float, str, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def get_metadata_columns_defaults():
    """
    Get metadata column names and default values.
    """

    dummy = create_metadata_record(
        series_id=0,
        length=0,
        label="",
        is_stationary=0
    )

    return list(dummy.keys()), dummy


def attach_metadata_columns_to_df(df, metadata_record):
    """
    Attach metadata columns to a generated time series dataframe.
    """

    df = df.copy()

    metadata_cols, default_record = get_metadata_columns_defaults()

    for col in metadata_cols:
        val = metadata_record.get(col, default_record[col])
        df[col] = metadata_value_to_cell(val)

    df["label"] = metadata_record["label"]

    core_cols = ["series_id", "time", "data"]

    optional_series_cols = [
        "seasonal_diff"
    ]

    meta_cols = [
        col for col in metadata_cols
        if col not in core_cols + ["label"] and col in df.columns
    ]

    final_cols_order = (
        core_cols
        + [col for col in optional_series_cols if col in df.columns]
        + meta_cols
        + ["label"]
    )

    final_cols_in_df = [
        col for col in final_cols_order
        if col in df.columns
    ]

    df = df[final_cols_in_df]

    return df