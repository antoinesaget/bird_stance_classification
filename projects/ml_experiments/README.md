# ML Experiments

This subproject contains the remaining offline Model B training and evaluation code.

## Surviving Entry Points

- `src/birdsys/ml_experiments/train_attributes.py`: train a final multi-head Model B artifact on a dataset split
- `src/birdsys/ml_experiments/train_attributes_cv.py`: grouped cross-validation and old-vs-new comparison workflow
- `src/birdsys/ml_experiments/model_b_evaluation.py`: offline checkpoint evaluation helpers
- `src/birdsys/ml_experiments/metrics.py`: confusion matrix and macro-F1 helpers
- `config/train_attributes.yaml`: remaining training config

## Current Shape

- The old experiment `cli.py` wrapper is gone.
- The autoresearch sandbox templates are gone.
- What remains is the minimal training, CV, and evaluation code surface.

## Workflow Dependencies

- This code expects dataset directories produced by `projects/datasets`.
- It also depends on the shared BirdSys core package for label maps, path/layout helpers, and shared model definitions.

## Current Status

- The surviving training and evaluation entrypoints still encode the intended Model B workflow.
- They are not currently runnable because they import `birdsys.core`, and that shared package still references deleted modules from before the cleanup.
