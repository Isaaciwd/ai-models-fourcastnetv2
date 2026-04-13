# ai-models Plugin Migration Guide for Shared Sensitivity Framework

This guide is based on the `feat/sensitivity-yaml-nc-release` branch changes in `ai-models-fourcastnetv2`.

Use it as a checklist when updating any `ai-models-<model>` plugin to work with the new shared sensitivity framework in `ai-models`.

## What moved to `ai-models` core

Do **not** re-implement these in plugin repos:

- sensitivity CLI argument definitions (`add_sensitivity_parser_arguments`)
- sensitivity config loading and top-level YAML parsing (`SensitivityManager`)
- target metadata dataclass (`SensitivityTarget`)
- NetCDF/JSON summary writing
- optional plotting (including coastline overlays and signed gradients)

Plugin repos should focus on model-specific pieces only.

## Required plugin changes

## 1) Initialize sensitivity manager in model constructor

In your model `__init__`, call `self.init_sensitivity(model_name, default_sensitivity_path)` after core model attributes are set.

Required inputs:

- plugin model name string (for metadata)
- deterministic default output path (usually includes model name and lead time)

Why: this wires the plugin into shared config parsing, output writing, and plotting.

## 2) Extend `parse_model_args` with shared sensitivity options

In plugin `parse_model_args`, include:

- `add_sensitivity_parser_arguments(parser)`

Then add plugin-specific memory/performance controls if needed (for example model-internal checkpointing flags).

Why: plugins must accept the shared sensitivity flags and any model-specific rollout controls.

## 3) Implement target parsing for model-native fields

Implement a parser that maps config/CLI target selectors to model-native channel keys.

Typical logic:

- accept `field` directly when already in plugin ordering
- accept `param + level` and map to field key (for pressure-level variables)
- validate unsupported combinations and raise clear `ValueError`s

Why: the shared framework delegates field semantics to the model plugin.

## 4) Implement `default_target()` and `config_targets(config)`

Required behavior:

- `default_target()` returns one `SensitivityTarget` when YAML `targets` is absent
- `config_targets(config)` parses and validates each `targets[]` entry
- enforce valid metrics (`mean`, `mean-square`)
- parse optional area bounds using `parse_target_area`
- create stable names using `target_slug`

Why: the shared framework calls these hooks to build target definitions per model.

## 5) Implement model-specific objective function

Implement `sensitivity_objective(output, target)` with model semantics:

- full-state objective if `target.field is None`
- field-specific objective when `target.field` is set
- optional spatial weighting over area (lat/lon masks + weighting)

Also expose:

- `ordering` (channel names)
- `latitudes` and `longitudes` arrays in model grid coordinates

Why: the framework computes gradients from this scalar objective.

## 6) Separate forecast and sensitivity execution paths

Plugin `run()` should:

- load model and input state once
- write step-0 input fields
- branch to forecast path or sensitivity path based on `self.sensitivity`

Sensitivity path should:

- keep gradients enabled
- run full autoregressive rollout to final forecast step
- compute one scalar objective per target
- call `torch.autograd.grad` w.r.t. input state
- stack gradients and call `self.sensitivity_manager.save(...)`

Why: gradients need a full differentiable forward path from initial state to final objective.

## 7) Make model checkpointing gradient-safe

If plugin model uses `torch.utils.checkpoint.checkpoint`:

- gate checkpoint calls with `torch.is_grad_enabled()`
- use `use_reentrant=False` for modern PyTorch compatibility

Apply this in both:

- per-block network layers
- rollout-level model stepping wrappers

Why: this avoids backward/graph issues during sensitivity mode while still reducing memory use.

## 8) Validate lead-time stepping for model cadence

Add a helper like `forecast_step_count()` that enforces model cadence constraints (for example, lead time must be a multiple of 6 hours).

Why: invalid lead time causes silent objective mismatch if not checked.

## 9) Keep plugin dependencies minimal

Remove plugin-level dependencies that are now provided by core framework behavior (for example YAML parser dependencies no longer needed in plugin if config parsing is centralized in `ai-models`).

## 10) Update docs/examples and tests in plugin repo

Add or update:

- README: one-file YAML workflow, targets schema, output expectations
- `examples/sensitivity-config.example.yaml`
- tests for parser/target/objective/checkpointing/lead-time validation

Minimum test coverage to add:

- shared sensitivity args are accepted by plugin parser
- default target resolution (`param + level` and field path)
- multi-target config parsing and invalid metric rejection
- objective uses selected target channel
- backward pass works with checkpointing enabled
- forecast step count validation for valid/invalid lead times

## File-level mapping from this branch

These branch changes are the concrete reference implementation:

- `ai_models_fourcastnetv2/model.py`
  - added plugin integration hooks for shared sensitivity manager
  - added target parsing + objective + sensitivity rollout path
  - split forecast and sensitivity execution
- `ai_models_fourcastnetv2/fourcastnetv2/sfnonet.py`
  - enabled block checkpointing path guarded by grad mode
- `ai_models_fourcastnetv2/fourcastnetv2/layers.py`
  - updated checkpoint call to `use_reentrant=False` and grad guard
- `tests/test_sensitivity.py`
  - replaced placeholder coverage with sensitivity-focused tests
- `README.md`, `examples/sensitivity-config.example.yaml`
  - documented YAML-first sensitivity workflow
- `setup.py`
  - removed plugin-local YAML dependency now handled in shared stack

## Quick migration checklist for new plugins

- [ ] Call `init_sensitivity(...)` in model constructor
- [ ] Add shared sensitivity CLI args in `parse_model_args`
- [ ] Implement `default_target()`, `config_targets(config)`, `sensitivity_objective(...)`
- [ ] Expose channel ordering and lat/lon coordinates
- [ ] Implement sensitivity rollout and gradient save path
- [ ] Ensure checkpointing is grad-aware and `use_reentrant=False`
- [ ] Add cadence/lead-time validation helper
- [ ] Update README + example YAML
- [ ] Add sensitivity-focused tests
- [ ] Keep plugin dependencies limited to model-specific needs
