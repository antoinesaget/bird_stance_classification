# Datasets

This subproject owns the offline data-prep path from Label Studio export to model-ready datasets.

## Surviving Entry Points

- `src/birdsys/datasets/export_normalize.py`: normalize a Label Studio export JSON into deterministic parquet tables
- `src/birdsys/datasets/make_crops.py`: generate JPEG crops from normalized bird boxes
- `src/birdsys/datasets/build_dataset.py`: build a versioned `train` / `val` / `test` dataset directory and report files
- `sql/build_dataset_duckdb.sql`: remaining SQL reference asset

## Current Shape

- There is no `cli.py` wrapper anymore.
- There are no tests in the current checkout.
- The package metadata remains in `pyproject.toml`, but the practical surface is just the three scripts above.

## Data Contract

- The scripts expect a BirdSys data root at `BIRDS_DATA_ROOT` and operate against the shared project layout under that root.
- The exact directory layout is currently mediated by `birdsys.core.ensure_layout()`.

## Current Status

- All three surviving entrypoints import `birdsys.core`.
- Because the shared core package still references deleted modules, this subproject is not currently runnable.
- The code here still describes the intended normalization, crop, and dataset-build flow, but the shared-core path/helpers need to be restored before it works again.
