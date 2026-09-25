from collections import Counter

from betise.core.rules import (
    validate_requested_combination,
)

from betise.scenario_builder import (
    build_feature_variants,
    count_categorical_recipes,
    count_feature_variants,
    count_materialized_scenarios,
    count_scenario_categorical_variants,
    count_type_scenarios,
    enumerate_type_scenarios,
    enumerate_valid_base_compositions,
    iter_materialized_scenarios,
    to_generation_composition,
)


EXPECTED_BASE_COUNTS = {
    1: 18,
    2: 71,
    3: 68,
}

EXPECTED_TYPE_COUNTS = {
    1: 18,
    2: 233,
    3: 1140,
    4: 2652,
    5: 2979,
}

EXPECTED_FEATURE_VARIANT_COUNTS = {
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

EXPECTED_ALL_COUNTS = {
    1: 18,
    2: 1427,
    3: 36490,
    4: 359126,
    5: 862116,
}

EXPECTED_SAMPLED_3_COUNTS = {
    1: 18,
    2: 503,
    3: 3071,
    4: 7752,
    5: 8937,
}


def test_valid_base_composition_counts():
    compositions = (
        enumerate_valid_base_compositions()
    )

    counts = Counter(
        len(composition)
        for composition in compositions
    )

    assert dict(counts) == (
        EXPECTED_BASE_COUNTS
    )

    assert len(compositions) == 157


def test_every_base_composition_passes_rules():
    compositions = (
        enumerate_valid_base_compositions()
    )

    for base_components in compositions:
        report = (
            validate_requested_combination(
                base_components=base_components,
                feature_components=(),
            )
        )

        assert report.valid, (
            base_components,
            report.errors,
        )


def test_type_scenario_counts():
    counts = (
        count_type_scenarios(
            min_size=1,
            max_size=5,
        )
    )

    assert counts == (
        EXPECTED_TYPE_COUNTS
    )

    assert sum(
        counts.values()
    ) == 7022


def test_all_combination_sizes_exist():
    scenarios = (
        enumerate_type_scenarios(
            min_size=1,
            max_size=5,
        )
    )

    sizes = {
        scenario[
            "combination_size"
        ]
        for scenario in scenarios
    }

    assert sizes == {
        1,
        2,
        3,
        4,
        5,
    }


def test_type_scenario_ids_are_unique():
    scenarios = (
        enumerate_type_scenarios()
    )

    scenario_ids = [
        scenario[
            "scenario_id"
        ]
        for scenario in scenarios
    ]

    assert len(
        scenario_ids
    ) == len(
        set(
            scenario_ids
        )
    )


def test_every_type_scenario_passes_rules():
    scenarios = (
        enumerate_type_scenarios()
    )

    for scenario in scenarios:
        report = (
            validate_requested_combination(
                base_components=(
                    scenario[
                        "base_components"
                    ]
                ),
                feature_components=(
                    scenario[
                        "feature_components"
                    ]
                ),
            )
        )

        assert report.valid, (
            scenario[
                "scenario_id"
            ],
            report.errors,
        )


def test_combination_size_matches_components():
    scenarios = (
        enumerate_type_scenarios()
    )

    for scenario in scenarios:
        expected_size = (
            len(
                scenario[
                    "base_components"
                ]
            )
            +
            len(
                scenario[
                    "feature_components"
                ]
            )
        )

        assert (
            scenario[
                "combination_size"
            ]
            == expected_size
        )


def test_feature_variant_counts():
    counts = (
        count_feature_variants()
    )

    assert counts == (
        EXPECTED_FEATURE_VARIANT_COUNTS
    )


def test_quadratic_trend_expands_to_six_variants():
    variants = (
        build_feature_variants()[
            "quadratic_trend"
        ]
    )

    params = {
        (
            variant[
                "params"
            ][
                "direction"
            ],
            variant[
                "params"
            ][
                "location"
            ],
        )
        for variant in variants
    }

    assert params == {
        ("up", "left"),
        ("up", "center"),
        ("up", "right"),
        ("down", "left"),
        ("down", "center"),
        ("down", "right"),
    }


def test_ar_quadratic_has_six_categorical_recipes():
    scenarios = [
        scenario
        for scenario
        in enumerate_type_scenarios()
        if (
            scenario[
                "base_components"
            ] == ["ar"]
            and scenario[
                "feature_components"
            ] == [
                "quadratic_trend"
            ]
        )
    ]

    assert len(
        scenarios
    ) == 1

    assert (
        count_scenario_categorical_variants(
            scenarios[0]
        )
        == 6
    )


def test_linear_mean_shift_all_mode_is_18():
    scenarios = [
        scenario
        for scenario
        in enumerate_type_scenarios()
        if (
            scenario[
                "base_components"
            ] == ["ar"]
            and scenario[
                "feature_components"
            ] == [
                "linear_trend",
                "mean_shift",
            ]
        )
    ]

    assert len(
        scenarios
    ) == 1

    materialized = list(
        iter_materialized_scenarios(
            scenarios,
            categorical_mode="all",
        )
    )

    assert len(
        materialized
    ) == 18


def test_all_mode_counts():
    counts = (
        count_categorical_recipes(
            categorical_mode="all"
        )
    )

    assert counts == (
        EXPECTED_ALL_COUNTS
    )

    assert sum(
        counts.values()
    ) == 1259177


def test_sampled_mode_counts():
    counts = (
        count_categorical_recipes(
            categorical_mode="sampled",
            variants_per_type=3,
        )
    )

    assert counts == (
        EXPECTED_SAMPLED_3_COUNTS
    )

    assert sum(
        counts.values()
    ) == 20281


def test_sampled_does_not_duplicate_when_space_is_small():
    scenarios = [
        scenario
        for scenario
        in enumerate_type_scenarios()
        if (
            scenario[
                "base_components"
            ] == ["ar"]
            and scenario[
                "feature_components"
            ] == [
                "linear_trend"
            ]
        )
    ]

    materialized = list(
        iter_materialized_scenarios(
            scenarios,
            categorical_mode="sampled",
            variants_per_type=3,
            seed=42,
        )
    )

    # linear_trend only has two categorical variants.
    assert len(
        materialized
    ) == 2

    ids = [
        item[
            "materialized_scenario_id"
        ]
        for item in materialized
    ]

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )


def test_materialized_quadratic_contains_override():
    scenarios = [
        scenario
        for scenario
        in enumerate_type_scenarios()
        if (
            scenario[
                "base_components"
            ] == ["ar"]
            and scenario[
                "feature_components"
            ] == [
                "quadratic_trend"
            ]
        )
    ]

    materialized = next(
        iter_materialized_scenarios(
            scenarios,
            categorical_mode="all",
        )
    )

    assert (
        materialized[
            "feature_overrides"
        ][
            "quadratic_trend"
        ]
        == {
            "direction": "up",
            "location": "left",
        }
    )


def test_generation_adapter_schema():
    scenarios = [
        scenario
        for scenario
        in enumerate_type_scenarios()
        if (
            scenario[
                "base_components"
            ] == ["ar"]
            and scenario[
                "feature_components"
            ] == [
                "quadratic_trend"
            ]
        )
    ]

    materialized = next(
        iter_materialized_scenarios(
            scenarios,
            categorical_mode="all",
        )
    )

    composition = (
        to_generation_composition(
            materialized
        )
    )

    assert composition[
        "base_components"
    ] == ["ar"]

    assert composition[
        "features"
    ] == [
        "quadratic_trend"
    ]

    assert composition[
        "feature_overrides"
    ][
        "quadratic_trend"
    ] == {
        "direction": "up",
        "location": "left",
    }

    assert composition[
        "combination_size"
    ] == 2