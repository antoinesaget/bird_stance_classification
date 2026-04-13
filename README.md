# BirdSys

BirdSys is currently a stripped-down monorepo for the bird annotation, dataset, training, and serving pipeline. After the recent cleanup, this repo is no longer a turnkey workspace. What remains is the minimal source tree, two deployment surfaces, and a small set of direct Python entrypoints.

## Current Layout

```text
projects/
  datasets/
  labelstudio/
  ml_backend/
  ml_experiments/
shared/
  birdsys_core/
OPERATIONS.md
README.md
```

## What Still Exists

- `shared/birdsys_core`: shared label taxonomies, shared model definitions, and Model B artifact loading helpers
- `projects/datasets`: Label Studio export normalization, crop generation, and versioned dataset building
- `projects/labelstudio`: Label Studio API workflows, task batch generation, prediction prefill, and TrueNAS deployment assets
- `projects/ml_backend`: FastAPI prediction service, model promotion helper, Dockerfile, and `iats` deployment assets
- `projects/ml_experiments`: Model B training, grouped CV, offline evaluation, and the remaining experiment config

## What No Longer Exists

- No root `pyproject.toml`
- No root `Makefile`
- No `ops/` workspace orchestration layer
- No repo-wide CLI wrapper
- No split `ENVIRONMENT.md` / `DEPLOYMENT.md` docs anymore
- No test tree in the current checkout
- No autoresearch sandbox templates in `projects/ml_experiments`

## Host Roles

- Local: source-of-truth engineering checkout
- `iats`: full checkout, training host, experiment host, live ML backend host
- TrueNAS: minimal Label Studio deployment checkout, Postgres host, public UI host, canonical bird and lines storage host

## Deployment Surfaces

- `projects/ml_backend/deploy/docker-compose.iats-ml.yml`
- `projects/labelstudio/deploy/docker-compose.truenas.yml`

## Canonical Ops Doc

- `OPERATIONS.md` replaces the old split between `ENVIRONMENT.md` and `DEPLOYMENT.md`.

## Current Runtime Status

- The surviving code is mostly plain Python modules and scripts under `projects/*/src`.
- The host uses `python3`; `python` is not available on `PATH`.
- `projects/labelstudio/src/birdsys/labelstudio/export_snapshot.py` still imports cleanly from the remaining tree.
- `shared/birdsys_core/src/birdsys/core/__init__.py` still imports deleted modules: `config.py`, `contracts.py`, `logging.py`, `paths.py`, and `reporting.py`.
- Any entrypoint that imports `birdsys.core` currently fails with `ModuleNotFoundError: No module named 'birdsys.core.config'`.

This README is meant to describe the repo exactly as it exists now, not the fully repaired state we still need to rebuild.
