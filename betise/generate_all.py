"""CLI for rule-driven BeTiSe dataset generation.

Example
-------
python -m betise.generate_all \
    --min-size 1 \
    --max-size 5 \
    --variant-policy balanced \
    --series-per-variant 1

The command reads only:
- betise/config/params.json
- betise/config/categorical_params.json

Canonical coexistence rules are imported from betise.core.rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from betise.dataset_generation import (
    _patch_pyarrow_unregister_extension_type,
)
from betise.full_dataset_generation import (
    _normalize_object_columns,
    generate_full_series,
)
from betise.scenario_builder import (
    enumerate_type_scenarios,
    iter_materialized_scenarios,
    load_categorical_params,
)
from betise.utils.helpers import add_indices_column


def _load_params(
    config_dir: Path,
) -> Dict[str, Any]:
    path = config_dir / "params.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_type_catalog(
    type_scenarios: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    path = output_dir / "type_scenarios.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for scenario in type_scenarios:
            handle.write(
                json.dumps(
                    scenario,
                    sort_keys=True,
                )
                + "\n"
            )


def _flush_shard(
    *,
    frames: List[pd.DataFrame],
    output_dir: Path,
    combination_size: int,
    shard_index: int,
) -> None:
    if not frames:
        return

    size_dir = (
        output_dir
        / f"{combination_size}-way"
    )
    size_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )
    dataframe = _normalize_object_columns(
        dataframe
    )

    output_path = (
        size_dir
        / f"part-{shard_index:05d}.parquet"
    )
    dataframe.to_parquet(
        output_path,
        index=False,
    )


def generate_all(
    *,
    config_dir: str | Path | None = None,
    output_dir: str | Path = "generated-dataset/scenarios",
    min_size: int = 1,
    max_size: int = 5,
    variant_policy: str = "balanced",
    samples_per_type: int = 1,
    series_per_variant: int = 1,
    length_range: tuple[int, int] = (300, 500),
    seed: int = 42,
    shard_size: int = 250,
    include_indices: bool = True,
    write_catalog: bool = True,
) -> Dict[str, Any]:
    if series_per_variant <= 0:
        raise ValueError(
            "series_per_variant must be positive."
        )
    if shard_size <= 0:
        raise ValueError(
            "shard_size must be positive."
        )
    if len(length_range) != 2:
        raise ValueError(
            "length_range must contain exactly two integers."
        )

    low, high = map(
        int,
        length_range,
    )
    if low <= 0 or high < low:
        raise ValueError(
            f"Invalid length_range: {length_range}"
        )

    _patch_pyarrow_unregister_extension_type()

    random.seed(seed)
    np.random.seed(seed)

    config_path = (
        Path(config_dir)
        if config_dir is not None
        else Path(__file__).resolve().parent / "config"
    )
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_cfg = _load_params(
        config_path
    )
    categorical_cfg = load_categorical_params(
        config_path
    )

    type_scenarios = enumerate_type_scenarios(
        min_size=min_size,
        max_size=max_size,
    )

    if write_catalog:
        _write_type_catalog(
            type_scenarios,
            output_path,
        )

    type_counts: Dict[int, int] = {}
    for scenario in type_scenarios:
        size = int(
            scenario["combination_size"]
        )
        type_counts[size] = (
            type_counts.get(size, 0)
            + 1
        )

    buffers: Dict[int, List[pd.DataFrame]] = {
        size: []
        for size in range(
            min_size,
            max_size + 1,
        )
    }
    buffered_series: Dict[int, int] = {
        size: 0
        for size in buffers
    }
    shard_indices: Dict[int, int] = {
        size: 0
        for size in buffers
    }
    generated_counts: Dict[int, int] = {
        size: 0
        for size in buffers
    }

    series_id = 1

    materialized = iter_materialized_scenarios(
        type_scenarios,
        categorical_cfg,
        policy=variant_policy,
        seed=seed,
        samples_per_type=samples_per_type,
    )

    generation_cfg = {
        "feature_defaults": {},
    }

    for scenario in materialized:
        size = int(
            scenario["combination_size"]
        )

        for _ in range(
            series_per_variant
        ):
            length = int(
                np.random.randint(
                    low,
                    high + 1,
                )
            )

            dataframe = generate_full_series(
                composition=scenario,
                full_cfg=generation_cfg,
                params_cfg=params_cfg,
                series_id=series_id,
                length=length,
            )

            if include_indices:
                dataframe = add_indices_column(
                    dataframe
                )

            buffers[size].append(
                dataframe
            )
            buffered_series[size] += 1
            generated_counts[size] += 1
            series_id += 1

            if (
                buffered_series[size]
                >= shard_size
            ):
                _flush_shard(
                    frames=buffers[size],
                    output_dir=output_path,
                    combination_size=size,
                    shard_index=shard_indices[size],
                )
                buffers[size] = []
                buffered_series[size] = 0
                shard_indices[size] += 1

    for size in sorted(buffers):
        if buffers[size]:
            _flush_shard(
                frames=buffers[size],
                output_dir=output_path,
                combination_size=size,
                shard_index=shard_indices[size],
            )
            shard_indices[size] += 1

    summary = {
        "schema_version": 1,
        "seed": seed,
        "min_size": min_size,
        "max_size": max_size,
        "variant_policy": variant_policy,
        "samples_per_type": samples_per_type,
        "series_per_variant": series_per_variant,
        "length_range": [low, high],
        "type_scenario_counts": {
            str(key): value
            for key, value in sorted(
                type_counts.items()
            )
        },
        "generated_series_counts": {
            str(key): value
            for key, value in sorted(
                generated_counts.items()
            )
        },
        "shard_counts": {
            str(key): value
            for key, value in sorted(
                shard_indices.items()
            )
        },
        "total_type_scenarios": len(
            type_scenarios
        ),
        "total_generated_series": sum(
            generated_counts.values()
        ),
    }

    with (
        output_path / "generation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical BeTiSe 1-to-N way "
            "scenarios from rules and two compact JSON configs."
        )
    )
    parser.add_argument(
        "--config-dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="generated-dataset/scenarios",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--variant-policy",
        choices=[
            "exhaustive",
            "balanced",
            "sampled",
        ],
        default="balanced",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--series-per-variant",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--length-min",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--length-max",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--no-indices",
        action="store_true",
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = generate_all(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        variant_policy=args.variant_policy,
        samples_per_type=args.samples_per_type,
        series_per_variant=args.series_per_variant,
        length_range=(
            args.length_min,
            args.length_max,
        ),
        seed=args.seed,
        shard_size=args.shard_size,
        include_indices=not args.no_indices,
        write_catalog=not args.no_catalog,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
