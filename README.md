# ai-models-fourcastnetv2

`ai-models-fourcastnetv2` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin for running the FourCastNet v2 small model.

This fork now keeps model-specific backprop logic in the plugin and delegates reusable sensitivity configuration/output/plotting workflows to the shared `ai-models` sensitivity module.

## What this fork adds

- Differentiable rollout and input-gradient computation for `fourcastnetv2-small`.
- FourCastNet v2 specific target-field resolution and objective assembly.
- Shared sensitivity UX (YAML, NetCDF, plotting) consumed from `ai-models`.

## Installation

```bash
pip install ai-models-fourcastnetv2
```

For local development:

```bash
pip install -e ./ai-models -e ./ai-models-fourcastnetv2
```

Optional plotting extras:

- `matplotlib` for PNG output.
- `cartopy` for coastline overlays. If missing, plots are still produced without coastlines.

## Quick start

Single target from CLI flags:

```bash
ai-models --input cds --date 20230110 --time 0000 --lead-time 24 \
  fourcastnetv2-small --sensitivity \
  --target-param r --target-level 850 --target-area 50,230,30,245 \
  --sensitivity-path ./sensitivity-r850-west-coast-24h.nc
```

YAML-driven multi-target run:

```bash
ai-models --input cds --date 20230110 --time 0000 fourcastnetv2-small \
  --sensitivity-config ./examples/sensitivity-config.example.yaml
```

Note: when `--sensitivity-config` is provided, sensitivity mode is enabled automatically.

## Outputs

Each sensitivity run writes:

- NetCDF (`.nc`) with dimensions:
  - `target`
  - `channel`
  - `latitude`
  - `longitude`
- Data variables:
  - `sensitivity[target, channel, latitude, longitude]`
  - `channel_score[target, channel]`
  - `total_sensitivity[target, latitude, longitude]`
  - `objective[target]`
  - `target_area[target, area_coord]`
- Coordinates:
  - `target` (target names)
  - `channel` (input state channels)
  - `latitude`, `longitude`
  - `target_field`, `target_metric`
- JSON summary (`.json`) with per-target objective values, peak locations, and top channels.
- PNG plots (if enabled):
  - `<prefix>-<target>-total.png`
  - `<prefix>-<target>-top-channels.png`

Output convention is intentionally close to model-input style: structured geospatial arrays on `latitude`/`longitude`, with explicit variable names and per-target metadata. The format is produced by the shared `ai-models` sensitivity module.

## Configuration interface

### Recommended workflow

Use a YAML config for production/reproducible runs and reserve CLI flags for quick one-off experiments.

### Config schema

Top-level keys:

- `run`
  - `lead_time` (hours)
- `output`
  - `path` (NetCDF sensitivity output path)
  - `summary_path` (JSON summary path)
- `plotting`
  - `enabled` (bool)
  - `prefix` (plot filename prefix)
  - `top_k` (number of channel maps in panel plot, must be non-negative)
  - `area` (`[north, west, south, east]`)
  - `coastlines` (bool, requires cartopy)
- `targets` (list)
  - `name` (identifier used in outputs)
  - one of:
    - `field` (e.g. `r850`, `t500`, `2t`)
    - `param` + `level` (e.g. `r` + `850`)
  - optional `area` (`[north, west, south, east]`)
  - optional `metric` (`mean` or `mean-square`)

### Example config

See: `examples/sensitivity-config.example.yaml`

## CLI reference

Sensitivity controls:

- `--sensitivity`
- `--sensitivity-config FILE`
- `--sensitivity-metric {mean,mean-square}`
- `--sensitivity-path FILE` (NetCDF output)
- `--summary-path FILE`

Single-target controls (used directly when no config is supplied):

- `--target-field FIELD`
- `--target-param PARAM`
- `--target-level LEVEL`
- `--target-area N,W,S,E`

Plot controls:

- `--plot-sensitivity` / `--no-plot-sensitivity`
- `--plot-top-k N`
- `--plot-prefix PREFIX`
- `--plot-area N,W,S,E`
- `--plot-coastlines` / `--no-plot-coastlines`

Performance controls:

- `--model-checkpointing` / `--no-model-checkpointing`
- `--rollout-checkpointing` / `--no-rollout-checkpointing`

## Notes and caveats

- This implementation currently targets native FourCastNet v2 output fields. Derived diagnostics (for example specific humidity) can be added in a follow-up.
- Forecast and sensitivity can still run without plotting dependencies.
- Coastline overlays are optional and degrade gracefully if cartopy is unavailable.
- If cartopy is not installed and coastlines are requested, the run logs a warning and continues.

## Development

Run tests:

```bash
pytest ai-models-fourcastnetv2/tests -q
```

## Repository strategy (recommended)

For now, keep this work private and structured as:

- a private fork of `ai-models-fourcastnetv2` (primary code here)
- a private fork of `ai-models` only if/when core framework changes are needed

This keeps the sensitivity feature isolated to the model plugin, minimizes merge burden, and makes future public release easier.

## Architecture note

For this refactor:

- `ai-models-fourcastnetv2` owns model-specific components:
  - differentiable model rollout
  - channel/field mapping for FourCastNet v2
  - model-specific scalar objective tensors
- `ai-models` owns reusable components:
  - CLI/YAML sensitivity interface
  - NetCDF and JSON output writing
  - plotting and coastline overlays

This separation is intended to make future model support easier without duplicating UI/output logic in each plugin.
