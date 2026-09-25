import numpy as np

from betise.scenario_generation import (
    iter_generated_series,
)


# ============================================================================
# BASIC GENERATION
# ============================================================================

def test_generated_series_count_respects_max_recipes():
    """
    5 categorical recipes × 1 realization
    must produce exactly 5 series.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 320),
            seed=42,
            max_recipes=5,
        )
    )

    assert len(
        generated
    ) == 5


def test_series_per_recipe_creates_multiple_realizations():
    """
    3 recipes × 2 numerical realizations
    must produce 6 actual time series.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=3,
        )
    )

    assert len(
        generated
    ) == 6


# ============================================================================
# DATAFRAME CONTRACT
# ============================================================================

def test_generated_dataframe_length_and_values():
    generated = list(
        iter_generated_series(
            min_size=2,
            max_size=2,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 320),
            seed=42,
            max_recipes=3,
        )
    )

    for dataframe, context in generated:

        assert (
            300
            <= len(dataframe)
            <= 320
        )

        assert len(
            dataframe
        ) == context[
            "length"
        ]

        assert np.all(
            np.isfinite(
                dataframe[
                    "data"
                ].to_numpy()
            )
        )

        assert (
            "series_id"
            in dataframe.columns
        )

        assert (
            "time"
            in dataframe.columns
        )

        assert (
            "data"
            in dataframe.columns
        )


# ============================================================================
# CONTEXT CONTRACT
# ============================================================================

def test_generation_context_matches_combination_size():
    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 300),
            seed=42,
            max_recipes=5,
        )
    )

    for _, context in generated:

        expected_size = (
            len(
                context[
                    "base_components"
                ]
            )
            +
            len(
                context[
                    "feature_components"
                ]
            )
        )

        assert expected_size == 3

        assert (
            context[
                "combination_size"
            ]
            == expected_size
        )


# ============================================================================
# RECIPE VS NUMERICAL REALIZATION
# ============================================================================

def test_same_recipe_produces_different_realizations():
    """
    series_per_recipe=2 means:

        same categorical recipe
        +
        two independent numerical realizations.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=1,
        )
    )

    assert len(
        generated
    ) == 2

    df_a, context_a = (
        generated[0]
    )

    df_b, context_b = (
        generated[1]
    )

    # Same categorical recipe.
    assert (
        context_a[
            "materialized_scenario_id"
        ]
        ==
        context_b[
            "materialized_scenario_id"
        ]
    )

    assert (
        context_a[
            "categorical_variant_ids"
        ]
        ==
        context_b[
            "categorical_variant_ids"
        ]
    )

    # But different actual series.
    assert (
        context_a[
            "series_id"
        ]
        !=
        context_b[
            "series_id"
        ]
    )

    assert not np.array_equal(
        df_a[
            "data"
        ].to_numpy(),
        df_b[
            "data"
        ].to_numpy(),
    )


# ============================================================================
# SERIES IDS
# ============================================================================

def test_series_ids_are_sequential():
    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=3,
        )
    )

    series_ids = [
        context[
            "series_id"
        ]
        for _, context
        in generated
    ]

    assert series_ids == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def test_same_seed_is_reproducible():
    """
    The same generation request with the same seed
    should select the same recipes and generate the same data.
    """

    kwargs = {
        "min_size": 3,
        "max_size": 3,
        "categorical_mode": "sampled",
        "variants_per_type": 1,
        "series_per_recipe": 1,
        "length_range": (300, 300),
        "seed": 42,
        "max_recipes": 3,
    }

    first = list(
        iter_generated_series(
            **kwargs
        )
    )

    second = list(
        iter_generated_series(
            **kwargs
        )
    )

    assert len(
        first
    ) == len(
        second
    )

    for (
        df_a,
        context_a,
    ), (
        df_b,
        context_b,
    ) in zip(
        first,
        second,
    ):

        assert (
            context_a[
                "materialized_scenario_id"
            ]
            ==
            context_b[
                "materialized_scenario_id"
            ]
        )

        assert (
            context_a[
                "categorical_variant_ids"
            ]
            ==
            context_b[
                "categorical_variant_ids"
            ]
        )

        assert np.array_equal(
            df_a[
                "data"
            ].to_numpy(),
            df_b[
                "data"
            ].to_numpy(),
        )