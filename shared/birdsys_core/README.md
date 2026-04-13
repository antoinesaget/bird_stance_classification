# BirdSys Core

Internal shared package for the remaining BirdSys subprojects. Its job is to hold the smallest possible cross-project surface: shared label taxonomies, shared model definitions, and shared artifact/runtime helpers.

## Surviving Modules

- `src/birdsys/core/attributes.py`: canonical label maps, legacy label normalization, and head-mask rules
- `src/birdsys/core/contracts.py`: small shared payload types still used by active subprojects
- `src/birdsys/core/models.py`: shared neural network definitions for the image status model and the multi-head attribute model
- `src/birdsys/core/model_b_artifacts.py`: Model B artifact loading, label decoding, and prediction guard logic
- `src/birdsys/core/paths.py`: canonical data-root layout and version-directory helpers
- `src/birdsys/core/reporting.py`: previous-version lookup and numeric diff helpers
- `src/birdsys/core/__init__.py`: minimal export surface for the active extraction and dataset flow

## Current Status

- `pyproject.toml` still declares the package metadata and dependencies.
- `birdsys.core` is importable for the extraction and dataset flow without the ML stack installed.
- The shared surface is intentionally minimal and no longer re-exports the old config/logging helpers.

## Intent

- Keep this package small.
- Keep it host-agnostic.
- Put shared code here only when at least two surviving subprojects actually need it.
