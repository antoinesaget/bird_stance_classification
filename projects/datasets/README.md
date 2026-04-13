# Datasets

This subproject owns the offline data-prep path from Label Studio export to split artifacts, crop artifacts, and model-ready datasets.

## Surviving Entry Points

- `src/birdsys/datasets/export_normalize.py`: strict current-schema normalization, comparison to previous extract, and Markdown/plot report generation
- `src/birdsys/datasets/build_split.py`: build a stable grouped test/train-pool split artifact with stored fold assignments and plots
- `src/birdsys/datasets/make_crops.py`: generate named crop-spec artifacts from normalized bird boxes
- `src/birdsys/datasets/build_dataset.py`: assemble versioned `test` / `train_pool` / `all_data` datasets from split + crop artifacts
- `sql/build_dataset_duckdb.sql`: remaining SQL reference asset

## Current Shape

- There is no `cli.py` wrapper anymore.
- Focused extraction and dataset pipeline regression tests now live under `tests/integration`.
- The package metadata remains in `pyproject.toml`, but the practical surface is the four scripts above.

## Data Contract

- The scripts now expect a species-aware data home at `BIRD_DATA_HOME`, plus a selected `BIRD_SPECIES_SLUG`.
- The exact directory layout is mediated by `birdsys.core.ensure_layout(data_home, species_slug)`.

## Current Status

- The strict annotation normalizer writes versioned `ann_vNNN` outputs with `birds.parquet`, `images_labels.parquet`, manifests, reports, and plots.
- Dataset creation is now split into three explicit stages:
  - `split_vNNN` for stable test/train-pool membership and stored folds
  - `crop_spec_id` artifacts under an annotation version for tunable crop sweeps
  - `ds_vNNN` pooled datasets with `test.parquet`, `train_pool.parquet`, `all_data.parquet`, and `fold_assignments.parquet`
- Comparison to the previous extract, split, and dataset versions is first-class in the generated reports.
