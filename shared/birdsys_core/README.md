# BirdSys Core

Internal shared package for the remaining BirdSys subprojects. Its job is to hold the smallest possible cross-project surface: shared label taxonomies, shared model definitions, and shared artifact/runtime helpers.

## Surviving Modules

- `src/birdsys/core/attributes.py`: canonical label maps, legacy label normalization, and head-mask rules
- `src/birdsys/core/models.py`: shared neural network definitions for the image status model and the multi-head attribute model
- `src/birdsys/core/model_b_artifacts.py`: Model B artifact loading, label decoding, and prediction guard logic
- `src/birdsys/core/__init__.py`: package export surface, currently stale

## Current Status

- `pyproject.toml` still declares the package metadata and dependencies.
- The package is not currently importable through `import birdsys.core`.
- `src/birdsys/core/__init__.py` still references deleted modules: `config.py`, `contracts.py`, `logging.py`, `paths.py`, and `reporting.py`.
- Subprojects that import from `birdsys.core` are blocked until that shared surface is repaired.

## Intent

- Keep this package small.
- Keep it host-agnostic.
- Put shared code here only when at least two surviving subprojects actually need it.
