from __future__ import annotations

from betise.core.rules import validate_requested_combination
from betise.scenario_builder import (
    build_feature_variants,
    enumerate_type_scenarios,
    iter_materialized_scenarios,
    load_categorical_params,
)


EXPECTED_VARIANT_COUNTS = {
    "linear_trend": 2,
    "quadratic_trend": 6,
    "cubic_trend": 6,
    "exponential_trend": 2,
    "damped_trend": 2,
    "mean_shift": 9,
    "variance_shift": 9,
    "trend_shift": 21,
    "point_anomaly": 7,
    "collective_anomaly": 33,
    "contextual_anomaly": 6,
}


def test_compact_categorical_schema_preserves_variant_counts():
    cfg = load_categorical_params()
    variants = build_feature_variants(cfg)

    assert {
        name: len(items)
        for name, items in variants.items()
    } == EXPECTED_VARIANT_COUNTS


def test_type_scenarios_cover_1_to_5_way_and_are_unique():
    scenarios = enumerate_type_scenarios(
        min_size=1,
        max_size=5,
    )

    sizes = {
        scenario["combination_size"]
        for scenario in scenarios
    }

    assert sizes == {1, 2, 3, 4, 5}

    counts = {
        size: sum(
            scenario["combination_size"] == size
            for scenario in scenarios
        )
        for size in sizes
    }
    assert counts == {
        1: 18,
        2: 233,
        3: 1140,
        4: 2652,
        5: 2979,
    }
    assert len(scenarios) == 7022

    keys = {
        (
            tuple(scenario["base_components"]),
            tuple(scenario["features"]),
        )
        for scenario in scenarios
    }

    assert len(keys) == len(scenarios)


def test_every_enumerated_type_scenario_passes_canonical_rules():
    scenarios = enumerate_type_scenarios(
        min_size=1,
        max_size=5,
    )

    for scenario in scenarios:
        report = validate_requested_combination(
            base_components=scenario[
                "base_components"
            ],
            feature_components=scenario[
                "features"
            ],
        )
        assert report.valid, (
            scenario["id"],
            report.errors,
        )


def test_exhaustive_variant_materialization_for_simple_case():
    cfg = load_categorical_params()

    type_scenario = {
        "id": "TEST",
        "name": "ar__linear_trend",
        "group": "2-way",
        "base_components": ["ar"],
        "features": ["linear_trend"],
        "combination_size": 2,
        "composition_steps": ["standalone"],
    }

    materialized = list(
        iter_materialized_scenarios(
            [type_scenario],
            cfg,
            policy="exhaustive",
            seed=42,
        )
    )

    assert len(materialized) == 2
    assert {
        item["feature_overrides"][
            "linear_trend"
        ]["direction"]
        for item in materialized
    } == {"up", "down"}


def test_balanced_policy_does_not_cartesian_explode():
    cfg = load_categorical_params()

    type_scenario = {
        "id": "TEST",
        "name": (
            "single_seasonality__"
            "linear_trend__contextual_anomaly"
        ),
        "group": "3-way",
        "base_components": [
            "single_seasonality"
        ],
        "features": [
            "linear_trend",
            "contextual_anomaly",
        ],
        "combination_size": 3,
        "composition_steps": [
            "standalone"
        ],
    }

    materialized = list(
        iter_materialized_scenarios(
            [type_scenario],
            cfg,
            policy="balanced",
            seed=42,
        )
    )

    # max(2 linear variants, 6 contextual variants) = 6,
    # rather than 2 * 6 = 12.
    assert len(materialized) == 6


def test_sampled_policy_is_seed_reproducible():
    cfg = load_categorical_params()

    type_scenario = {
        "id": "TEST",
        "name": "ar__mean_shift",
        "group": "2-way",
        "base_components": ["ar"],
        "features": ["mean_shift"],
        "combination_size": 2,
        "composition_steps": ["standalone"],
    }

    first = list(
        iter_materialized_scenarios(
            [type_scenario],
            cfg,
            policy="sampled",
            seed=17,
            samples_per_type=5,
        )
    )
    second = list(
        iter_materialized_scenarios(
            [type_scenario],
            cfg,
            policy="sampled",
            seed=17,
            samples_per_type=5,
        )
    )

    assert [
        row["categorical_variant_ids"]
        for row in first
    ] == [
        row["categorical_variant_ids"]
        for row in second
    ]
