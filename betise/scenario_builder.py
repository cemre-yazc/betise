"""Programmatic BeTiSe scenario enumeration and categorical expansion.

Canonical coexistence rules live in betise.core.rules. This module enumerates
candidate scenarios, validates them, and materializes categorical feature
variants from categorical_params.json.

Combination size = len(base_components) + len(features).
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from betise.core.rules import (
    CANONICAL_BASE_SERIES,
    OVERLAY_FEATURES,
    validate_requested_combination,
)


def load_categorical_params(
    config_dir: str | Path | None = None,
) -> Dict[str, Any]:
    base_dir = (
        Path(config_dir)
        if config_dir is not None
        else Path(__file__).resolve().parent / "config"
    )
    path = base_dir / "categorical_params.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _axis_product(
    axes: Mapping[str, Sequence[Any]] | None,
) -> List[Dict[str, Any]]:
    axes = axes or {}
    if not axes:
        return [{}]

    keys = list(axes)
    values = [list(axes[key]) for key in keys]

    return [
        dict(zip(keys, choice))
        for choice in product(*values)
    ]


def _variant_id(
    feature_name: str,
    params: Mapping[str, Any],
) -> str:
    parts = [feature_name]
    for key in sorted(params):
        value = params[key]
        value_text = (
            str(value).lower()
            if isinstance(value, bool)
            else str(value)
        )
        parts.append(f"{key}-{value_text}")
    return "__".join(parts)


def build_feature_variants(
    categorical_cfg: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Expand compact categorical axes into concrete feature variants."""

    features_cfg = categorical_cfg.get("features", {})
    output: Dict[str, List[Dict[str, Any]]] = {}

    for feature_name, spec in features_cfg.items():
        variants: List[Dict[str, Any]] = []

        if "cases" in spec:
            cases = spec["cases"]
        else:
            cases = {
                "default": {
                    "fixed": spec.get("fixed", {}),
                    "axes": spec.get("axes", {}),
                }
            }

        for case_spec in cases.values():
            fixed = deepcopy(case_spec.get("fixed", {}))
            for axis_values in _axis_product(
                case_spec.get("axes", {})
            ):
                params = {
                    **fixed,
                    **axis_values,
                }
                variants.append({
                    "variant_id": _variant_id(
                        feature_name,
                        params,
                    ),
                    "params": params,
                })

        output[feature_name] = variants

    return output


def _valid_base_compositions(
    max_size: int,
) -> List[Tuple[str, ...]]:
    bases = sorted(CANONICAL_BASE_SERIES)
    valid: List[Tuple[str, ...]] = []

    for count in range(
        1,
        min(max_size, len(bases)) + 1,
    ):
        for candidate in combinations(
            bases,
            count,
        ):
            report = validate_requested_combination(
                base_components=candidate,
                feature_components=(),
            )
            if report.valid:
                valid.append(candidate)

    return valid


def enumerate_type_scenarios(
    *,
    min_size: int = 1,
    max_size: int = 5,
) -> List[Dict[str, Any]]:
    """Enumerate every canonical type-level scenario within the size bounds."""

    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size.")

    valid_bases = _valid_base_compositions(
        max_size=max_size
    )
    features = sorted(OVERLAY_FEATURES)

    raw: List[Dict[str, Any]] = []

    for base_components in valid_bases:
        min_feature_count = max(
            0,
            min_size - len(base_components),
        )
        max_feature_count = min(
            len(features),
            max_size - len(base_components),
        )

        for feature_count in range(
            min_feature_count,
            max_feature_count + 1,
        ):
            for feature_components in combinations(
                features,
                feature_count,
            ):
                report = validate_requested_combination(
                    base_components=base_components,
                    feature_components=feature_components,
                )
                if not report.valid:
                    continue

                canonical_features = tuple(
                    report.feature_components
                )
                size = (
                    len(report.base_components)
                    + len(canonical_features)
                )

                raw.append({
                    "base_components": list(
                        report.base_components
                    ),
                    "features": list(
                        canonical_features
                    ),
                    "combination_size": size,
                    "composition_steps": list(
                        report.composition_steps
                    ),
                })

    raw.sort(
        key=lambda row: (
            row["combination_size"],
            tuple(row["base_components"]),
            tuple(row["features"]),
        )
    )

    scenarios: List[Dict[str, Any]] = []
    for index, row in enumerate(
        raw,
        start=1,
    ):
        name_parts = (
            row["base_components"]
            + row["features"]
        )
        scenarios.append({
            "id": f"C{index:06d}",
            "name": "__".join(name_parts),
            "group": f"{row['combination_size']}-way",
            **row,
        })

    return scenarios


def _normalize_variant_params(
    feature_name: str,
    raw_params: Mapping[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Translate categorical schema fields to apply_feature arguments."""

    params = deepcopy(dict(raw_params))

    if (
        feature_name in {"mean_shift", "variance_shift"}
        and params.get("mode") == "multiple"
    ):
        params.pop("direction", None)
        params.pop("direction_strategy", None)
        params.pop("location", None)

    if feature_name == "point_anomaly":
        params.pop("count_strategy", None)
        if params.get("mode") == "multiple":
            params.pop("location", None)
            params.pop("num_anomalies", None)
            params.pop("is_spike", None)

    if feature_name == "collective_anomaly":
        strategy = params.pop(
            "shape_strategy",
            None,
        )
        count = int(
            params.get(
                "num_anomalies",
                1,
            )
        )

        if strategy == "mixed":
            shapes = [
                "rectangular",
                "gaussian",
                "triangular",
                "ramp",
                "decay",
            ]
            params["anomaly_shapes"] = [
                rng.choice(shapes)
                for _ in range(count)
            ]
        elif strategy is not None:
            params["anomaly_shapes"] = [
                strategy
            ]

        if params.get("mode") == "multiple":
            params.pop("location", None)

    if (
        feature_name in {
            "contextual_anomaly",
            "trend_shift",
        }
        and params.get("mode") == "multiple"
    ):
        params.pop("location", None)

    return params


def _materialize_variant_choice(
    type_scenario: Mapping[str, Any],
    choices: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
) -> Dict[str, Any]:
    concrete = deepcopy(dict(type_scenario))
    overrides: Dict[str, Dict[str, Any]] = {}
    variant_ids: Dict[str, str] = {}

    for feature_name, variant in zip(
        concrete.get("features", []),
        choices,
    ):
        overrides[feature_name] = (
            _normalize_variant_params(
                feature_name,
                variant.get("params", {}),
                rng,
            )
        )
        variant_ids[feature_name] = variant[
            "variant_id"
        ]

    concrete["feature_overrides"] = overrides
    concrete["categorical_variant_ids"] = variant_ids

    if variant_ids:
        suffix = "__".join(
            variant_ids[name]
            for name in concrete["features"]
        )
        concrete["scenario_id"] = (
            f"{concrete['id']}__{suffix}"
        )
    else:
        concrete["scenario_id"] = concrete["id"]

    return concrete


def iter_materialized_scenarios(
    type_scenarios: Iterable[Mapping[str, Any]],
    categorical_cfg: Mapping[str, Any],
    *,
    policy: str = "exhaustive",
    seed: int = 42,
    samples_per_type: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Yield variants without materializing a giant manifest in memory."""

    policy = policy.lower()
    if policy not in {
        "exhaustive",
        "balanced",
        "sampled",
    }:
        raise ValueError(
            "policy must be exhaustive, balanced, or sampled."
        )

    variants_by_feature = build_feature_variants(
        categorical_cfg
    )
    rng = random.Random(seed)

    for type_scenario in type_scenarios:
        active = list(
            type_scenario.get("features", [])
        )

        if not active:
            yield _materialize_variant_choice(
                type_scenario,
                (),
                rng=rng,
            )
            continue

        variant_lists = [
            variants_by_feature[feature]
            for feature in active
        ]

        if policy == "exhaustive":
            choices_iter = product(
                *variant_lists
            )

        elif policy == "balanced":
            count = max(
                len(items)
                for items in variant_lists
            )
            choices_iter = (
                tuple(
                    items[index % len(items)]
                    for items in variant_lists
                )
                for index in range(count)
            )

        else:
            if samples_per_type <= 0:
                raise ValueError(
                    "samples_per_type must be positive."
                )
            choices_iter = (
                tuple(
                    rng.choice(items)
                    for items in variant_lists
                )
                for _ in range(samples_per_type)
            )

        for choices in choices_iter:
            yield _materialize_variant_choice(
                type_scenario,
                choices,
                rng=rng,
            )


def count_type_scenarios(
    *,
    min_size: int = 1,
    max_size: int = 5,
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for scenario in enumerate_type_scenarios(
        min_size=min_size,
        max_size=max_size,
    ):
        size = int(
            scenario["combination_size"]
        )
        counts[size] = (
            counts.get(size, 0)
            + 1
        )
    return counts
