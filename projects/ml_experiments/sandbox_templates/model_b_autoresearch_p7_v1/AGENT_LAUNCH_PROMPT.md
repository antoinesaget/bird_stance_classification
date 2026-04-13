# Model B Sandbox Agent Launch Prompt

Use this as the initial prompt for the autonomous experiment agent you launch inside the sandbox.

## Prompt

You are operating inside an isolated nested git repo rooted at the current sandbox directory.

Read these files first:

- `README.md`
- `program.md`
- `prepare.py`
- `eval.py`
- `run_once.py`
- `train.py`
- `results.tsv`
- `baselines/reference_thresholds.json`
- `baselines/current_served_model_fixed_folds_cv.json`
- `baselines/old_served_model_full_pool_eval.json`
- `baselines/project7_convnext_baseline_cv.json`
- `baseline_dynamics_report.md` if present

Hard constraints:

- edit only `train.py`, `results.tsv`, and files under the current run directory in `runs/`
- do not change folds, metrics, budgets, dependencies, or fixed harness files
- do not install packages
- do not touch anything outside this sandbox repo
- the run budget is fixed at 5 minutes per experiment
- the evaluator is authoritative and reads `candidate/predictions.parquet`
- validation search score is the primary training-time objective signal
- validation total loss is a secondary calibration / overfit diagnostic
- it is acceptable to add extra logs, metrics, timers, GPU-utilization tracking, or similar observability in `train.py` when useful for an idea, as long as the fixed harness contract and the other sandbox rules still hold

Objective:

- maximize the fixed `search_score = 0.50 * stance_f1_mean + 0.30 * behavior_f1_mean + 0.20 * substrate_f1_mean`
- keep `readability_f1_mean >= 0.30`
- keep `specie_f1_mean >= 0.30`
- only treat an improvement as meaningful if it beats the current kept best by at least `0.020` search-score points

Important human benchmark:

- current served model on these exact fixed folds has:
  - `search_score = 0.15045880464389927`
  - `readability_f1 = 0.5924177484924534`
  - `specie_f1 = 0.3332340852521676`
  - `behavior_f1 = 0.17097640614161297`
  - `substrate_f1 = 0.3634210572567528`
  - `stance_f1 = 0.052963342700129666`

The current sandbox baseline already beats served on stance and scalar score, but is much worse on readability, specie, behavior, and substrate. Treat that as a warning, not a success.

Current official baseline anchor:

- `run_id = 20260326_150758_sampler-cap-2p5-v1`
- `search_score = 0.5183739491703461`

Loop:

1. inspect current results
2. make one focused change in `train.py`
3. commit
4. run `python3 run_once.py --description "<short idea>"`
5. inspect `runs/<run_id>/summary.json`
6. keep only meaningful improvements
7. if discarded or crashed, reset to the previous best commit

Research priorities:

- prefer changes likely to improve behavior/substrate without collapsing readability/specie
- inspect the observability artifacts before deciding whether a run is truly better:
  - validation search score plots
  - train vs val loss plots
  - per-head loss plots
  - `baseline_dynamics_report.md`
- treat the selected checkpoint as the real output of the run; do not reject a run just because the unused tail after the selected checkpoint looks worse
- stay within the servable `multihead_timm_backbone` family unless there is a clear upside worth losing zero-code export
- prefer simple changes when gains are small

Stop and summarize the best runs found, including whether each is servable now.
