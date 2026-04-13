# ML Experiments

This subproject contains the remaining offline Model B training and evaluation code.

## Surviving Entry Points

- `src/birdsys/ml_experiments/train_attributes.py`: train a final multi-head Model B artifact on a dataset split
- `src/birdsys/ml_experiments/train_attributes_cv.py`: grouped cross-validation on the stored train-pool fold assignments plus old-vs-new comparison
- `src/birdsys/ml_experiments/model_b_evaluation.py`: offline checkpoint evaluation helpers
- `src/birdsys/ml_experiments/metrics.py`: confusion matrix and macro-F1 helpers
- `config/train_attributes.yaml`: remaining training config

## Current Shape

- The old experiment `cli.py` wrapper is gone.
- The autoresearch sandbox templates are gone.
- What remains is the minimal training, CV, and evaluation code surface.

## Workflow Dependencies

- This code expects pooled dataset directories produced by `projects/datasets`.
- `train_attributes_cv.py` now consumes `train_pool.parquet` plus `fold_assignments.parquet` instead of recomputing folds.
- `train_attributes.py` now targets `train_pool`, `test`, or `all_data` splits instead of the removed `train` / `val` / `test` contract.
- It also depends on the shared BirdSys core package for label maps, path/layout helpers, and shared model definitions.

## Current Status

- The surviving training and evaluation entrypoints still encode the intended Model B workflow.
- The shared `birdsys.core` import surface is restored for the active training code.
