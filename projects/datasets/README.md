# Datasets

This subproject owns the offline data-prep path from Label Studio export to model-ready datasets.

## Surviving Entry Points

- `src/birdsys/datasets/export_normalize.py`: strict current-schema normalization, comparison to previous extract, and Markdown/plot report generation
- `src/birdsys/datasets/make_crops.py`: generate JPEG crops from normalized bird boxes
- `src/birdsys/datasets/build_dataset.py`: build a versioned `train` / `val` / `test` dataset directory and report files
- `sql/build_dataset_duckdb.sql`: remaining SQL reference asset

## Current Shape

- There is no `cli.py` wrapper anymore.
- Focused extraction regression tests now live under `tests/integration`.
- The package metadata remains in `pyproject.toml`, but the practical surface is just the three scripts above.

## Data Contract

- The scripts expect a BirdSys data root at `BIRDS_DATA_ROOT` and operate against the shared project layout under that root.
- The exact directory layout is currently mediated by `birdsys.core.ensure_layout()`.

## Current Status

- The strict annotation normalizer now writes versioned `ann_vNNN` outputs with `birds.parquet`, `images_labels.parquet`, `manifest.json`, JSON reports, Markdown reports, and embedded plots.
- Comparison to the previous extract is now first-class in the normalization output.
- The next dataset-building step can consume the normalized `ann_vNNN` output directly.
