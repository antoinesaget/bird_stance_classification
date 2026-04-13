# Model B Autoresearch Program

This sandbox follows the spirit of Karpathy's `autoresearch`, adapted to BirdSys Model B.

## In-scope files

Read these files for full context before you start:

- `README.md`
- `program.md`
- `prepare.py`
- `eval.py`
- `run_once.py`
- `train.py`
- `results.tsv`
- `baselines/reference_thresholds.json`
- `baselines/old_served_model_full_pool_eval.json`
- `baselines/project7_convnext_baseline_cv.json`
- `baselines/current_served_model_fixed_folds_cv.json`
- `baseline_dynamics_report.md` if it exists
- `runs/<current best>/training_curves_report.json` if it exists

## What you may edit

- `train.py`
- `results.tsv`
- files under the current run directory in `runs/`

## What you must not edit

- `prepare.py`
- `eval.py`
- `run_once.py`
- `inspect_results.py`
- `export_candidate.py`
- `clear_unannotated_project7_predictions.py`
- `refresh_project7_predictions.py`
- any file under `data/`
- any file under `baselines/`
- dependency files
- fold manifests

## Hard rules

- Do not install packages.
- Do not change folds.
- Do not change the metric.
- Do not change the 5-minute run budget.
- Do not cheat evaluation.
- Do not write predictions from the ground truth.
- The evaluation is fixed cross-validation only.
- Behavior, substrate, and stance are the search objective.
- Readability and specie are guardrails and must not regress too far.
- The authoritative score comes from the fixed evaluator reading `candidate/predictions.parquet`, not from metrics returned by `train.py`.
- Validation search score is the primary training-time objective signal.
- Validation total loss is a calibration and overfit diagnostic, not the primary stop criterion.
- It is acceptable to add extra logs, metrics, timers, GPU-utilization tracking, or similar observability inside `train.py` if an idea needs them, as long as all other sandbox rules still hold.

## Objective

The scalar objective is fixed in `eval.py`:

`search_score = 0.50 * stance_f1_mean + 0.30 * behavior_f1_mean + 0.20 * substrate_f1_mean`

Guardrails:

- `readability_f1_mean >= 0.30`
- `specie_f1_mean >= 0.30`

Minimum meaningful improvement:

- `delta_to_keep = 0.020`
- this is tied to the measured 3-seed baseline `search_score` std for the current sampler-cap baseline recipe in `baselines/baseline_seed_variance_summary.json`

Current official baseline anchor:

- `run_id = 20260326_150758_sampler-cap-2p5-v1`
- `search_score = 0.5183739491703461`

Deployment comparison target for human review:

- current served Model B fixed-fold score:
  - `search_score = 0.15045880464389927`
- fixed-fold means:
  - `readability_f1 = 0.5924177484924534`
  - `specie_f1 = 0.3332340852521676`
  - `behavior_f1 = 0.17097640614161297`
  - `substrate_f1 = 0.3634210572567528`
  - `stance_f1 = 0.052963342700129666`

Do not confuse “beats search score” with “good promotion candidate”. The search score is stance-heavy by design.

## Run loop

If `results.tsv` is empty, establish the baseline first:

```bash
python3 run_once.py --description baseline
```

Then loop forever until manually stopped:

1. inspect current best results
2. make one experimental change in `train.py`
3. commit
4. run `python3 run_once.py --description "..."`
5. inspect `runs/<run_id>/summary.json`
6. keep the change only if the fixed harness says `keep`
7. if the harness says `discard` or `crash`, reset back to the previous best commit

## Research guidance

Everything inside `train.py` is fair game:

- model architecture
- timm backbone selection
- image size
- optimizer
- scheduler
- augmentations
- weighted sampling
- losses
- regularization
- EMA
- mixup/cutmix if you keep the head contract coherent

Your `run_fold` implementation must still write canonical label predictions with this schema:

- `row_id`
- `readability`
- `specie`
- `behavior`
- `substrate`
- `stance`

Prefer simpler changes when gains are tiny.

## Observability guidance

Before judging a run, inspect:

- validation search score plots
- train vs validation total loss
- per-head losses
- `baseline_dynamics_report.md`
- any extra observability the current idea added inside the run directory

Decision rule:

- prefer runs where validation search score improves cleanly and produces a good selected checkpoint
- judge dynamics by the selected checkpoint, not by the unused tail after the selected checkpoint
- treat rising validation loss as a warning only when it damages the selected checkpoint or indicates a bad transition

## Serving compatibility

Research can be broader than the current backend contract, but zero-code export is only guaranteed when the run remains compatible with the current BirdSys `checkpoint.pt` contract.

The fixed harness distinguishes:

- research-best run
- servable-now run

If a run is not exportable yet, the harness should mark it clearly instead of pretending it is ready.
