# Repository Strategy

## Recommended layout (private now, public-ready later)

For the current phase, keep this project as a private plugin-first repository.

- Primary implementation in a private fork of `ai-models-fourcastnetv2`.
- Keep a private fork of `ai-models` only if framework-level changes are required.

This minimizes maintenance burden and keeps model-specific sensitivity logic isolated.

## Why this is the best near-term choice

- The sensitivity feature is currently model-specific (`fourcastnetv2-small`).
- Most changes live naturally in the plugin layer.
- You avoid carrying unnecessary divergence in core `ai-models`.
- Future public release is cleaner and easier to explain.

## When to promote to a combined/toolbox repository

Create a dedicated top-level toolbox repository only when both are true:

1. You support more than one model with a common user-facing sensitivity API.
2. You need shared orchestration code that does not belong to a single model plugin.

At that point, keep model plugins as dependencies and put shared CLI/config workflows in the toolbox repo.

## Suggested private release workflow

1. Create private forks:
   - `ai-models` (optional until needed)
   - `ai-models-fourcastnetv2` (required)
2. Protect `main` and require PR reviews.
3. Use semantic tags for internal milestones (for example `v0.1.0-private`).
4. Maintain a concise changelog focused on user-facing behavior.
5. Add CI for tests and linting before public release.

## Public-readiness checklist

- remove environment-specific paths
- ensure docs include reproducible examples
- include licensing/attribution notes for dependencies and data
- verify no private credentials or URLs in configs/history
- add issue templates and contribution notes
