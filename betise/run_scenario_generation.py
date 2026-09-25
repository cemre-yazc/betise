"""
User-facing runner for scenario-driven BeTiSe dataset generation.

Usage
-----
python -m betise.run_scenario_generation

Optional:
python -m betise.run_scenario_generation path/to/generation_config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from betise.scenario_builder import (
    count_categorical_recipes,
)

from betise.scenario_output import (
    generate_dataset_to_parquet,
)


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_generation_config(
    config_path=None,
) -> Dict[str, Any]:
    """
    Load generation_config.json.
    """

    if config_path is None:

        config_path = (
            Path(__file__).resolve().parent
            / "config"
            / "generation_config.json"
        )

    else:

        config_path = Path(
            config_path
        )

    if not config_path.exists():

        raise FileNotFoundError(
            "Generation config was not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    return config


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_generation_config(
    config: Dict[str, Any],
) -> None:
    """
    Validate user-facing generation settings.
    """

    required = {
        "output_dir",
        "min_size",
        "max_size",
        "categorical_mode",
        "variants_per_type",
        "series_per_recipe",
        "length_range",
        "seed",
        "shard_size",
    }

    missing = (
        required
        - set(config)
    )

    if missing:

        raise ValueError(
            "generation_config.json is missing: "
            f"{sorted(missing)}"
        )

    min_size = int(
        config["min_size"]
    )

    max_size = int(
        config["max_size"]
    )

    if min_size < 1:

        raise ValueError(
            "min_size must be >= 1."
        )

    if max_size < min_size:

        raise ValueError(
            "max_size must be >= min_size."
        )

    if max_size > 5:

        raise ValueError(
            "Current canonical generation "
            "supports combination sizes up to 5."
        )

    categorical_mode = str(
        config[
            "categorical_mode"
        ]
    ).lower()

    if categorical_mode not in {
        "all",
        "sampled",
    }:
        raise ValueError(
            "categorical_mode must be "
            "'all' or 'sampled'."
        )

    if int(
        config["variants_per_type"]
    ) <= 0:
        raise ValueError(
            "variants_per_type must be positive."
        )

    if int(
        config["series_per_recipe"]
    ) <= 0:

        raise ValueError(
            "series_per_recipe must be positive."
        )

    if int(
        config["shard_size"]
    ) <= 0:

        raise ValueError(
            "shard_size must be positive."
        )

    length_range = config[
        "length_range"
    ]

    if (
        not isinstance(
            length_range,
            list,
        )
        or len(length_range) != 2
    ):

        raise ValueError(
            "length_range must be "
            "[minimum, maximum]."
        )

    low = int(
        length_range[0]
    )

    high = int(
        length_range[1]
    )

    if low <= 0 or high < low:

        raise ValueError(
            "Invalid length_range: "
            f"{length_range}"
        )

    max_recipes = config.get(
        "max_recipes"
    )

    if (
        max_recipes is not None
        and int(max_recipes) <= 0
    ):

        raise ValueError(
            "max_recipes must be null "
            "or a positive integer."
        )


# ============================================================================
# GENERATION ESTIMATE
# ============================================================================

def build_generation_estimate(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estimate number of recipes and actual series before generation.
    """

    recipe_counts = (
        count_categorical_recipes(
            min_size=int(
                config["min_size"]
            ),
            max_size=int(
                config["max_size"]
            ),
            categorical_mode=str(
                config[
                    "categorical_mode"
                ]
            ),
            variants_per_type=int(
                config[
                    "variants_per_type"
                ]
            ),
        )
    )

    total_recipes = sum(
        recipe_counts.values()
    )

    max_recipes = config.get(
        "max_recipes"
    )

    if max_recipes is not None:

        effective_recipes = min(
            total_recipes,
            int(max_recipes),
        )

    else:

        effective_recipes = (
            total_recipes
        )

    total_series = (
        effective_recipes
        * int(
            config[
                "series_per_recipe"
            ]
        )
    )

    return {
        "recipe_counts": (
            recipe_counts
        ),

        "total_available_recipes": (
            total_recipes
        ),

        "effective_recipes": (
            effective_recipes
        ),

        "estimated_series": (
            total_series
        ),
    }


# ============================================================================
# RUNNER
# ============================================================================

def run_generation(
    config_path=None,
):
    """
    Load config, validate it, show generation estimate,
    and start dataset generation.
    """

    config = (
        load_generation_config(
            config_path
        )
    )

    validate_generation_config(
        config
    )

    estimate = (
        build_generation_estimate(
            config
        )
    )

    print("=" * 72)
    print("BeTiSe SCENARIO GENERATION")
    print("=" * 72)

    print(
        "Categorical mode    :",
        config[
            "categorical_mode"
        ],
    )

    print(
        "Variants per type   :",
        config[
            "variants_per_type"
        ],
    )

    print(
        "Combination sizes   :",
        f"{config['min_size']}–{config['max_size']}",
    )

    print(
        "Available recipes   :",
        estimate[
            "total_available_recipes"
        ],
    )

    print(
        "Recipes to generate :",
        estimate[
            "effective_recipes"
        ],
    )

    print(
        "Series per recipe   :",
        config[
            "series_per_recipe"
        ],
    )

    print(
        "Estimated series    :",
        estimate[
            "estimated_series"
        ],
    )

    print(
        "Output directory    :",
        config[
            "output_dir"
        ],
    )

    print("=" * 72)

    # ------------------------------------------------------------
    # Safety guard
    # ------------------------------------------------------------

    if (
        config[
            "categorical_mode"
        ]
        == "all"
        and estimate[
            "estimated_series"
        ] > 100000
        and not config.get(
            "allow_large_run",
            False,
        )
    ):

        raise RuntimeError(
            "Large exhaustive generation blocked. "
            f"This configuration would generate approximately "
            f"{estimate['estimated_series']} series. "
            "Set allow_large_run=true in generation_config.json "
            "if this is intentional."
        )

    # ------------------------------------------------------------
    # Actual generation
    # ------------------------------------------------------------

    summary = (
        generate_dataset_to_parquet(
            output_dir=(
                config[
                    "output_dir"
                ]
            ),

            min_size=int(
                config[
                    "min_size"
                ]
            ),

            max_size=int(
                config[
                    "max_size"
                ]
            ),

            categorical_mode=str(
                config[
                    "categorical_mode"
                ]
            ),

            variants_per_type=int(
                config[
                    "variants_per_type"
                ]
            ),

            series_per_recipe=int(
                config[
                    "series_per_recipe"
                ]
            ),

            length_range=tuple(
                config[
                    "length_range"
                ]
            ),

            seed=int(
                config[
                    "seed"
                ]
            ),

            shard_size=int(
                config[
                    "shard_size"
                ]
            ),

            max_recipes=(
                config.get(
                    "max_recipes"
                )
            ),
        )
    )

    print()
    print("=" * 72)
    print("GENERATION COMPLETE")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    custom_config = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    run_generation(
        custom_config
    )