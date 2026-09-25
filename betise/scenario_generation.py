"""
Scenario-driven BeTiSe time-series generation.

This module connects:

    rules.py
        ↓
    scenario_builder.py
        ↓
    categorical variants
        ↓
    params.json
        ↓
    full_dataset_generation.generate_full_series()

It does not define combination validity.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd

from betise.config import load_config
from betise.full_dataset_generation import (
    generate_full_series,
)

from betise.scenario_builder import (
    enumerate_type_scenarios,
    iter_materialized_scenarios,
    to_generation_composition,
)


# ============================================================================
# LENGTH SAMPLING
# ============================================================================

def _sample_length(
    length_range,
) -> int:
    """
    Sample one series length from [low, high].
    """

    if len(length_range) != 2:
        raise ValueError(
            "length_range must contain "
            "[low, high]."
        )

    low = int(
        length_range[0]
    )

    high = int(
        length_range[1]
    )

    if low <= 0:
        raise ValueError(
            "Minimum length must be positive."
        )

    if high < low:
        raise ValueError(
            "Maximum length must be >= minimum length."
        )

    return int(
        np.random.randint(
            low,
            high + 1,
        )
    )


# ============================================================================
# SERIES ITERATOR
# ============================================================================

def iter_generated_series(
    *,
    min_size: int = 1,
    max_size: int = 5,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
    series_per_recipe: int = 1,
    length_range=(300, 500),
    seed: int = 42,
    max_recipes: Optional[int] = None,
) -> Iterator[
    tuple[
        pd.DataFrame,
        Dict[str, Any],
    ]
]:
    """
    Generate actual BeTiSe time series.

    Parameters
    ----------
    min_size, max_size:
        Allowed total combination size.

    categorical_mode:
        Categorical variant selection mode:

            "all"
                Use every categorical variant combination
                available for each type-level scenario.

            "sampled"
                Select a limited number of categorical recipes
                for each type-level scenario.

    variants_per_type:
        Number of categorical recipes selected per type-level
        scenario when categorical_mode="sampled".

        If a type scenario has fewer possible categorical recipes
        than this value, all available recipes are used.

    series_per_recipe:
        Number of independent numerical realizations generated
        from each categorical recipe.

    length_range:
        [minimum_length, maximum_length]

    seed:
        Reproducibility seed.

    max_recipes:
        Optional safety/debug limit.

        Example:
            max_recipes=10

        means:
            process only the first 10 categorical recipes.

        This is especially useful for smoke tests.

    Yields
    ------
    (dataframe, generation_context)
    """

    if series_per_recipe <= 0:
        raise ValueError(
            "series_per_recipe must be positive."
        )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    # ------------------------------------------------------------
    # Numerical parameter configuration
    # ------------------------------------------------------------

    cfg = load_config()

    params_cfg = cfg[
        "params"
    ]

    # full_dataset_generation only needs
    # feature_defaults from full_cfg here.
    full_cfg = {
        "feature_defaults": {}
    }

    # ------------------------------------------------------------
    # Type-level scenario space
    # ------------------------------------------------------------

    type_scenarios = (
        enumerate_type_scenarios(
            min_size=min_size,
            max_size=max_size,
        )
    )

    # ------------------------------------------------------------
    # Categorical materialization
    # ------------------------------------------------------------

    materialized_iterator = (
        iter_materialized_scenarios(
        type_scenarios=type_scenarios,
        categorical_mode=categorical_mode,
        seed=seed,
        )
    )

    series_id = 1
    recipe_index = 0

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    for materialized in (
        materialized_iterator
    ):

        if (
            max_recipes is not None
            and recipe_index
            >= max_recipes
        ):
            break

        recipe_index += 1

        composition = (
            to_generation_composition(
                materialized
            )
        )

        for realization_index in range(
            series_per_recipe
        ):

            length = _sample_length(
                length_range
            )

            dataframe = (
                generate_full_series(
                    composition=composition,
                    full_cfg=full_cfg,
                    params_cfg=params_cfg,
                    series_id=series_id,
                    length=length,
                )
            )

            context = {
                "series_id": (
                    series_id
                ),

                "recipe_index": (
                    recipe_index
                ),

                "realization_index": (
                    realization_index
                ),

                "type_scenario_id": (
                    materialized[
                        "scenario_id"
                    ]
                ),

                "materialized_scenario_id": (
                    materialized[
                        "materialized_scenario_id"
                    ]
                ),

                "combination_size": (
                    materialized[
                        "combination_size"
                    ]
                ),

                "base_components": list(
                    materialized[
                        "base_components"
                    ]
                ),

                "feature_components": list(
                    materialized[
                        "feature_components"
                    ]
                ),

                "categorical_variant_ids": dict(
                    materialized.get(
                        "categorical_variant_ids",
                        {},
                    )
                ),

                "feature_overrides": dict(
                    materialized.get(
                        "feature_overrides",
                        {},
                    )
                ),

                "length": (
                    length
                ),

                "seed": (
                    seed
                ),
            }

            yield (
                dataframe,
                context,
            )

            series_id += 1