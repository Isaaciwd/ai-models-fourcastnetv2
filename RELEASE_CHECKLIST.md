# Release Checklist (Private)

Use this checklist before pushing to your private GitHub fork.

## Code and tests

- [ ] `pytest ai-models-fourcastnetv2/tests -q` passes
- [ ] YAML config example parses and runs
- [ ] NetCDF output opens with xarray
- [ ] Plot output is generated (with and without cartopy)

## Documentation

- [ ] `README.md` reflects current CLI and YAML schema
- [ ] `examples/sensitivity-config.example.yaml` matches parser behavior
- [ ] `docs/repository-strategy.md` aligns with your publish plan

## Branching and commits

- [ ] Create branch from `main` (for example `feat/sensitivity-yaml-nc`)
- [ ] Commit in logical chunks (core engine, docs/examples, tests)
- [ ] Avoid committing large local artifacts (`*.nc`, `*.png`, temporary configs)

## Private GitHub publication

- [ ] Add your private fork as `origin` or `private`
- [ ] Push branch
- [ ] Open private PR
- [ ] Confirm CI and reviewer approval before merge

## Suggested commit breakdown

1. `feat: add fourcastnetv2 targeted sensitivity workflow`
   - target-aware objective
   - checkpoint-aware multi-target gradients
   - summary JSON and plotting hooks
2. `feat: add YAML-config and NetCDF sensitivity outputs`
   - `--sensitivity-config`
   - NetCDF schema + metadata
3. `docs: add user guide, examples, and repository strategy`
   - README rewrite
   - config example
   - publication strategy note
4. `test: add sensitivity parser/objective/config coverage`
   - replace placeholder test file
