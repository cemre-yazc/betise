# BeTiSe Rule-Driven Combination Generation

This document describes the canonical generation architecture for 1-way through
5-way BeTiSe scenarios.

## Source of truth

Generation is intentionally split across four responsibilities:

1. `betise/core/rules.py`
   - Defines which base components and overlay features may coexist.
   - Enforces dependencies and exclusions.
   - Defines canonical composition order.

2. `betise/config/params.json`
   - Contains numerical values and ranges used while generating realizations.
   - Examples: slope ranges, anomaly scale factors, ARFIMA d range,
     seasonality amplitudes.

3. `betise/config/categorical_params.json`
   - Contains discrete dataset variants.
   - Examples: direction, location, single/multiple mode, anomaly shape,
     trend-shift change type.

4. `betise/scenario_builder.py`
   - Enumerates valid type-level scenarios from rules.
   - Defines combination size as:
     `len(base_components) + len(features)`.
   - Expands categorical variants at runtime.

Static multi-megabyte composition JSON files are not the canonical source of
truth for new generation.

## Combination sizes

Examples:

- 1-way: `ar`
- 2-way: `ar + garch`
- 3-way: `ar + linear_trend + point_anomaly`
- 4-way: `arima + garch + multiple_seasonality + linear_trend`
- 5-way:
  `arfima + garch + single_seasonality + linear_trend + collective_anomaly`

Every candidate is passed through `validate_requested_combination()` before it
is admitted to the scenario catalog.

## Variant policies

The runtime supports three categorical expansion policies.

### exhaustive

Produces the full Cartesian product of active categorical feature variants.

Use this only when complete coverage is explicitly required.

### balanced

Cycles through each active feature's variant list without building the full
Cartesian product.

This is the recommended default for broad dataset production and smoke testing.

### sampled

Draws a deterministic random set of categorical variants per type-level
scenario.

Use `--samples-per-type` to control the number of samples.

## Main command

From the repository root:

```bash
python -m betise.generate_all \
  --min-size 1 \
  --max-size 5 \
  --variant-policy balanced \
  --series-per-variant 1 \
  --length-min 300 \
  --length-max 500 \
  --seed 42
```

For full categorical expansion:

```bash
python -m betise.generate_all \
  --min-size 1 \
  --max-size 5 \
  --variant-policy exhaustive
```

For sampled generation:

```bash
python -m betise.generate_all \
  --min-size 1 \
  --max-size 5 \
  --variant-policy sampled \
  --samples-per-type 3
```

## Output layout

The generator writes Parquet shards by combination size:

```text
generated-dataset/scenarios/
  generation_summary.json
  type_scenarios.jsonl
  1-way/
    part-00000.parquet
  2-way/
    part-00000.parquet
  3-way/
    ...
  4-way/
    ...
  5-way/
    ...
```

`type_scenarios.jsonl` is a generated catalog for inspection and audit. It is
not a configuration input.

## Metadata

Generated series persist:

- base components
- base families
- overlay feature components
- overlay feature families
- composition steps
- composition id/name/group
- scenario id
- combination size
- categorical variant ids
- existing family-specific numerical metadata

This keeps each generated realization traceable to the exact scenario that
created it.

## Backward compatibility

The legacy `dataset.json` and `dataset_generation.py` path remains available
for simple single-root-base generation.

The new rule-driven path is:

```text
params.json
categorical_params.json
        |
        v
scenario_builder.py
        |
        v
rules.py validation
        |
        v
full_dataset_generation.py
        |
        v
generate_all.py
        |
        v
Parquet shards
```

## Tests

Run:

```bash
pytest tests/test_scenario_builder.py -v
pytest tests/ -v
```

The scenario-builder tests verify:

- compact categorical definitions preserve the previous variant counts,
- 1-way through 5-way scenarios are produced,
- duplicate type scenarios are not produced,
- every enumerated scenario passes canonical rules,
- balanced expansion avoids full Cartesian explosion,
- sampled expansion is seed reproducible.
