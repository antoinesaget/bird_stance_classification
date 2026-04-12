# Model B Autoresearch Sandbox

This is an isolated, nested-git sandbox for autonomous Model B research on the current project 7 stilts dataset.

## Goals

- fixed 5-fold CV evaluation on frozen project 7 data
- fixed 5-minute experiment budget
- one primary mutable file: `train.py`
- append-only `results.tsv`
- keep/discard decisions computed by the fixed harness
- clean path to export a servable `checkpoint.pt` candidate later

## Layout

- `prepare.py`: snapshot project 7 data and freeze folds/baselines
- `eval.py`: fixed prediction scoring, metric aggregation, guardrails, keep/discard logic
- `run_once.py`: executes one experiment run and scores saved predictions through the fixed evaluator
- `train.py`: agent-editable training code
- `inspect_results.py`: leaderboard and plots
- `export_candidate.py`: verify and export a selected run for the current backend contract
- `clear_unannotated_project7_predictions.py`: remove stale predictions only on untouched tasks
- `refresh_project7_predictions.py`: promote a selected exported candidate and regenerate untouched project 7 predictions

## Fixed Artifact Contract

Each fold run must write these files under `runs/<run_id>/folds/fold_<k>/candidate/`:

- `predictions.parquet`: one row per test crop with canonical label columns and `row_id`
- `artifact_meta.json`: supported labels, model family, and prediction file reference

The fixed evaluator reads those files directly and computes the score. `train.py` may report its own local metrics, but they are not authoritative.

## Setup

```bash
python3 prepare.py
python3 run_once.py --description baseline
python3 inspect_results.py
```

## Benchmarks

- current served Model B on these exact fixed folds:
  - `baselines/current_served_model_fixed_folds_cv.json`
- old full-pool project 7 evaluation:
  - `baselines/old_served_model_full_pool_eval.json`
- previous project 7 ConvNeXt CV report:
  - `baselines/project7_convnext_baseline_cv.json`
- baseline run-to-run variance on 3 fresh seeds of the current baseline recipe:
  - `baselines/baseline_seed_variance_summary.json`
- current official baseline anchor:
  - `20260326_150758_sampler-cap-2p5-v1`
  - `search_score = 0.5183739491703461`

## Acceptance policy

- objective: `0.50 * stance_f1 + 0.30 * behavior_f1 + 0.20 * substrate_f1`
- guardrails:
  - `readability_f1_mean >= 0.30`
  - `specie_f1_mean >= 0.30`
- keep threshold:
  - `delta_to_keep = 0.020`
  - this is the empirical 3-seed baseline `search_score` std, rounded up from `0.0196169`

## Observability

- fold runs now write `training_history.json` with:
  - train and validation total loss
  - per-head train and validation losses
  - validation search score
  - best-step selection metadata
- the agent may add extra logs, counters, timing, utilization metrics, or other observability inside `train.py` when useful for an idea, as long as it does not change folds, evaluation, budgets, or the fixed artifact contract
- use `plot_training_curves.py` to inspect:
  - train vs val loss
  - per-head losses
  - validation search score
  - phase transition diagnostics
- use `inspect_results.py` to regenerate:
  - `baseline_dynamics_report.json`
  - `baseline_dynamics_report.md`

Interpretation rule:

- `val_search_score` is the primary selection and stopping signal
- `val_total_loss` is a calibration and overfit diagnostic
- do not reject a run solely because validation loss rises if validation search score continues to improve or remains near its best value
- `dynamics_pass` now judges the selected checkpoint we would actually keep:
  - clean unfreeze transition
  - best checkpoint found after the first `20%` of training
  - a valid selected checkpoint exists
- the unused tail after the selected checkpoint is informational only and does not block `dynamics_pass`

## Git workflow

This directory is its own nested git repo. The intended autonomous loop is:

1. edit `train.py`
2. commit
3. run `python3 run_once.py --description ...`
4. inspect the fixed keep/discard result in `runs/<run_id>/summary.json`
5. if discarded, reset to the previous best commit

## Promotion flow

```bash
python3 export_candidate.py --run-id <run_id>
python3 refresh_project7_predictions.py --project-id 7 --run-id <run_id>
```

The refresh script uses the parent BirdSys repo env files under `../deploy/env/`.
