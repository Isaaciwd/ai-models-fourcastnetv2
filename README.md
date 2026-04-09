# ai-models-fourcastnetv2

`ai-models-fourcastnetv2` is an `ai-models` plugin that adds the `fourcastnetv2-small` model.

This fork also supports gradient-based sensitivity analysis (backpropagation from forecast outputs to input fields).

## Install

Install core + plugin:

```bash
pip install ai-models ai-models-fourcastnetv2
```

For local development from sibling repos:

```bash
pip install -e ./ai-models -e ./ai-models-fourcastnetv2
```

Verify model is available:

```bash
ai-models --models
```

You should see `fourcastnetv2-small`.

## Fastest Way to Run

Use one YAML file and run:

```bash
ai-models --yaml ./examples/sensitivity-config.example.yaml
```

## Minimal YAML (Forecast + Sensitivity)

```yaml
model: fourcastnetv2-small

run:
  input: cds
  output: none
  date: 20230110
  time: 0000
  lead_time: 24
  only_gpu: true

runtime:
  assets_dir: ./assets/fourcastnetv2-small

output:
  path: ./sensitivity-results.nc
  summary_path: ./sensitivity-results.json

plotting:
  enabled: true
  prefix: ./sensitivity-results
  top_k: 6

targets:
  - name: west-coast-r850
    param: r
    level: 850
    area: [50, 230, 30, 245]
    metric: mean-square
```

Full example file: `examples/sensitivity-config.example.yaml`.

## What Date Means

- `run.date` + `run.time` is the forecast start (initial condition)
- `run.lead_time` is hours forward to target
- sensitivity is backpropagated from the final forecast time (`start + lead_time`) to the input state

Example: `date=20230101`, `time=0000`, `lead_time=48` means forecast target at `2023-01-03 00:00`.

## Outputs

A sensitivity run writes:

- NetCDF file (`output.path`) with:
  - `sensitivity[target, channel, latitude, longitude]`
  - `channel_score[target, channel]`
  - `total_sensitivity[target, latitude, longitude]`
  - `objective[target]`
- JSON summary (`output.summary_path`) with per-target top channels and peak sensitivity location
- PNG plots (if enabled), for each target:
  - `<prefix>-<target>-total.png`
  - `<prefix>-<target>-top-channels.png`
  - optional signed versions when enabled

Plot titles include:

- lead time
- backpropagation target datetime (forecast valid time)

## Targets

Each `targets[]` entry defines one scalar objective and produces one sensitivity result set.

Field-specific target:

```yaml
- name: west-coast-t500
  field: t500
  area: [50, 230, 30, 245]
  metric: mean
```

Param+level target:

```yaml
- name: west-coast-r850
  param: r
  level: 850
  area: [50, 230, 30, 245]
  metric: mean-square
```

Full-state target (all channels):

```yaml
- name: full-state-baseline
  metric: mean-square
```

## YAML Keys Reference

Top-level keys used by this plugin:

- `model`: should be `fourcastnetv2-small`
- `run`:
  - `input`, `output`, `date`, `time`, `lead_time`, `only_gpu`
  - `model_checkpointing`, `rollout_checkpointing` (memory/perf controls)
- `runtime`:
  - `assets_dir`
  - `omp_num_threads`, `mkl_num_threads`, `mpl_backend`
  - `run_dir` (used by wrapper scripts)
- `output`:
  - `path`, `summary_path`
- `plotting`:
  - `enabled`, `prefix`, `top_k`, `area`, `coastlines`, `signed_gradients`
- `targets`: list of target definitions
- `cli` (optional): advanced mapping to extra ai-models options

Notes:

- `plotting.area` must be `null` or a 4-value list `[north, west, south, east]`
- `date` accepts `YYYYMMDD` or `YYYY-MM-DD`
- `time` accepts `HHMM` or `HH:MM`

## Optional Dependencies

- `matplotlib` for plot output
- `cartopy` for coastlines/borders overlay (plots still work without it)

## Minimal PBS Script Example

```bash
#!/bin/bash -l
#PBS -N fcnetv2-sens
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_type=a100
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -q casper
#PBS -A P93300042

set -euo pipefail
cd /path/to/workdir

/path/to/miniconda3/condabin/conda run -n ai-models-gfs \
  ai-models --yaml ./sensitivity-config.yaml
```

## Development

Run plugin tests:

```bash
pytest ai-models-fourcastnetv2/tests -q
```

## License

Apache License 2.0. See `LICENSE` and `LICENSE_FourCastNetv2`.
